using System.Collections.Concurrent;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace SharpInfer.Core.Engine;

/// <summary>
/// Distributed inference: split model layers across multiple GPUs or network nodes.
///
/// Supports two parallelism strategies:
///
/// 1. **Tensor Parallelism (TP)**: Splits individual weight matrices across devices.
///    Each device holds a shard of every layer. All-reduce after attention/FFN.
///    Best for: single machine with multiple GPUs. Low latency, high bandwidth needed.
///
/// 2. **Pipeline Parallelism (PP)**: Assigns entire layers to different devices.
///    Data flows sequentially through pipeline stages. Micro-batching for efficiency.
///    Best for: multi-node clusters. Tolerates higher latency between nodes.
///
/// Architecture:
///   Coordinator (rank 0) ↔ Workers (rank 1..N) via TCP or shared memory
///   - TCP transport for multi-node (network)
///   - Shared memory transport for multi-GPU on same machine
///
/// Usage:
///   var cluster = new InferenceCluster(config);
///   await cluster.InitializeAsync();
///   var result = await cluster.GenerateAsync(prompt, genConfig);
/// </summary>
public class InferenceCluster : IAsyncDisposable
{
    private readonly DistributedConfig _config;
    private readonly List<WorkerConnection> _workers = new();
    private readonly ConcurrentDictionary<string, TaskCompletionSource<WorkerResponse>> _pending = new();
    private ClusterTopology? _topology;
    private bool _initialized;
    private int _requestCounter;

    public ClusterTopology? Topology => _topology;
    public int WorkerCount => _workers.Count;
    public bool IsInitialized => _initialized;
    public Action<string>? OnLog { get; set; }

    public InferenceCluster(DistributedConfig config)
    {
        _config = config;
    }

    /// <summary>
    /// Initialize the cluster: connect to all workers, negotiate topology,
    /// distribute model shards.
    /// </summary>
    public async Task InitializeAsync()
    {
        OnLog?.Invoke($"Initializing {_config.Strategy} cluster with {_config.Workers.Count} worker(s)...");

        // Connect to workers
        foreach (var workerConfig in _config.Workers)
        {
            var worker = new WorkerConnection(workerConfig);
            await worker.ConnectAsync();
            _workers.Add(worker);
            OnLog?.Invoke($"Connected to worker {workerConfig.Name} ({workerConfig.Host}:{workerConfig.Port})");
        }

        // Build topology
        _topology = BuildTopology();
        OnLog?.Invoke($"Topology: {_topology}");

        // Distribute model shards
        await DistributeModelAsync();

        _initialized = true;
        OnLog?.Invoke("Cluster initialized successfully.");
    }

    /// <summary>
    /// Run inference across the cluster.
    /// </summary>
    public async Task<string> GenerateAsync(string prompt, int maxTokens = 512)
    {
        if (!_initialized)
            throw new InvalidOperationException("Cluster not initialized. Call InitializeAsync first.");

        var requestId = $"req_{Interlocked.Increment(ref _requestCounter)}";

        return _config.Strategy switch
        {
            ParallelismStrategy.TensorParallel => await GenerateTensorParallelAsync(requestId, prompt, maxTokens),
            ParallelismStrategy.PipelineParallel => await GeneratePipelineParallelAsync(requestId, prompt, maxTokens),
            _ => throw new NotSupportedException($"Strategy {_config.Strategy} not supported.")
        };
    }

    /// <summary>
    /// Stream tokens from distributed inference.
    /// </summary>
    public async IAsyncEnumerable<string> GenerateStreamingAsync(string prompt, int maxTokens = 512)
    {
        if (!_initialized)
            throw new InvalidOperationException("Cluster not initialized.");

        var requestId = $"req_{Interlocked.Increment(ref _requestCounter)}";

        // Send request to all workers
        var request = new WorkerRequest
        {
            RequestId = requestId,
            Type = RequestType.GenerateStreaming,
            Prompt = prompt,
            MaxTokens = maxTokens,
        };

        // For TP: coordinator collects from rank 0 after all-reduce
        // For PP: last stage streams back tokens
        var streamWorker = _config.Strategy == ParallelismStrategy.PipelineParallel
            ? _workers[^1]  // Last pipeline stage
            : _workers[0];  // Rank 0 in TP

        await BroadcastRequestAsync(request);

        await foreach (var token in streamWorker.StreamResponseAsync(requestId))
        {
            yield return token;
        }
    }

    // ─── Tensor Parallelism ────────────────────────────────────

    /// <summary>
    /// Tensor parallel inference: each worker holds shards of all layers.
    ///
    /// For each forward pass step:
    ///   1. Coordinator broadcasts input tokens to all workers
    ///   2. Each worker computes attention/FFN on their shard
    ///   3. All-reduce combines partial results
    ///   4. Coordinator receives the final logits from rank 0
    ///   5. Coordinator samples next token
    /// </summary>
    private async Task<string> GenerateTensorParallelAsync(string requestId, string prompt, int maxTokens)
    {
        var request = new WorkerRequest
        {
            RequestId = requestId,
            Type = RequestType.Generate,
            Prompt = prompt,
            MaxTokens = maxTokens,
            Topology = _topology,
        };

        // All workers participate in tensor-parallel forward passes
        await BroadcastRequestAsync(request);

        // Rank 0 collects and returns the final result
        var response = await _workers[0].WaitForResponseAsync(requestId, _config.TimeoutMs);
        return response.GeneratedText ?? "";
    }

    // ─── Pipeline Parallelism ──────────────────────────────────

    /// <summary>
    /// Pipeline parallel inference: layers are distributed across stages.
    ///
    /// For each micro-batch:
    ///   1. Stage 0 processes its layers, sends activations to stage 1
    ///   2. Stage 1 processes, sends to stage 2, etc.
    ///   3. Last stage produces logits, samples token
    ///   4. Token is sent back to coordinator
    ///
    /// Micro-batching: while stage N processes batch K,
    /// stage N-1 can process batch K+1 (pipeline fill).
    /// </summary>
    private async Task<string> GeneratePipelineParallelAsync(string requestId, string prompt, int maxTokens)
    {
        var request = new WorkerRequest
        {
            RequestId = requestId,
            Type = RequestType.Generate,
            Prompt = prompt,
            MaxTokens = maxTokens,
            Topology = _topology,
        };

        // Send request to all pipeline stages
        await BroadcastRequestAsync(request);

        // Stage 0 kicks off the pipeline, last stage returns result
        var lastStage = _workers[^1];
        var response = await lastStage.WaitForResponseAsync(requestId, _config.TimeoutMs);
        return response.GeneratedText ?? "";
    }

    // ─── Model Distribution ────────────────────────────────────

    private async Task DistributeModelAsync()
    {
        if (_topology == null) return;

        foreach (var assignment in _topology.Assignments)
        {
            var worker = _workers.FirstOrDefault(w => w.Config.Name == assignment.WorkerName);
            if (worker == null) continue;

            var loadRequest = new WorkerRequest
            {
                RequestId = $"load_{assignment.WorkerName}",
                Type = RequestType.LoadModel,
                ModelPath = _config.ModelPath,
                LayerRange = assignment.LayerRange,
                ShardIndex = assignment.ShardIndex,
                TotalShards = assignment.TotalShards,
            };

            await worker.SendRequestAsync(loadRequest);
            var response = await worker.WaitForResponseAsync(loadRequest.RequestId, _config.ModelLoadTimeoutMs);

            if (!response.Success)
                throw new Exception($"Worker {assignment.WorkerName} failed to load: {response.Error}");

            OnLog?.Invoke($"Worker {assignment.WorkerName}: loaded {assignment}");
        }
    }

    private ClusterTopology BuildTopology()
    {
        var topology = new ClusterTopology
        {
            Strategy = _config.Strategy,
            TotalLayers = _config.TotalLayers,
        };

        switch (_config.Strategy)
        {
            case ParallelismStrategy.TensorParallel:
                // Each worker gets a shard of ALL layers
                for (int i = 0; i < _workers.Count; i++)
                {
                    topology.Assignments.Add(new LayerAssignment
                    {
                        WorkerName = _workers[i].Config.Name,
                        LayerRange = (0, _config.TotalLayers - 1),
                        ShardIndex = i,
                        TotalShards = _workers.Count,
                        Rank = i,
                    });
                }
                break;

            case ParallelismStrategy.PipelineParallel:
                // Distribute layers evenly across workers
                int layersPerWorker = _config.TotalLayers / _workers.Count;
                int remainder = _config.TotalLayers % _workers.Count;
                int currentLayer = 0;

                for (int i = 0; i < _workers.Count; i++)
                {
                    int numLayers = layersPerWorker + (i < remainder ? 1 : 0);
                    topology.Assignments.Add(new LayerAssignment
                    {
                        WorkerName = _workers[i].Config.Name,
                        LayerRange = (currentLayer, currentLayer + numLayers - 1),
                        ShardIndex = 0,
                        TotalShards = 1,
                        Rank = i,
                    });
                    currentLayer += numLayers;
                }
                break;
        }

        return topology;
    }

    private async Task BroadcastRequestAsync(WorkerRequest request)
    {
        var tasks = _workers.Select(w => w.SendRequestAsync(request));
        await Task.WhenAll(tasks);
    }

    public async ValueTask DisposeAsync()
    {
        foreach (var worker in _workers)
        {
            try
            {
                await worker.SendRequestAsync(new WorkerRequest { Type = RequestType.Shutdown });
                worker.Dispose();
            }
            catch { /* best-effort cleanup */ }
        }
        _workers.Clear();
    }
}

/// <summary>
/// Connection to a single worker node.
/// </summary>
public class WorkerConnection : IDisposable
{
    private TcpClient? _client;
    private NetworkStream? _stream;
    private StreamReader? _reader;
    private StreamWriter? _writer;
    private readonly SemaphoreSlim _writeLock = new(1, 1);
    private readonly ConcurrentDictionary<string, TaskCompletionSource<WorkerResponse>> _pending = new();
    private Task? _readLoop;
    private CancellationTokenSource _cts = new();

    public WorkerConfig Config { get; }
    public bool IsConnected => _client?.Connected ?? false;

    public WorkerConnection(WorkerConfig config)
    {
        Config = config;
    }

    public async Task ConnectAsync()
    {
        _client = new TcpClient();
        await _client.ConnectAsync(Config.Host, Config.Port);
        _stream = _client.GetStream();
        _reader = new StreamReader(_stream, Encoding.UTF8);
        _writer = new StreamWriter(_stream, Encoding.UTF8) { AutoFlush = true };

        // Start reading responses
        _readLoop = ReadLoopAsync(_cts.Token);
    }

    public async Task SendRequestAsync(WorkerRequest request)
    {
        if (_writer == null) throw new InvalidOperationException("Not connected.");

        var json = JsonSerializer.Serialize(request);
        await _writeLock.WaitAsync();
        try
        {
            await _writer.WriteLineAsync(json);
        }
        finally
        {
            _writeLock.Release();
        }
    }

    public async Task<WorkerResponse> WaitForResponseAsync(string requestId, int timeoutMs = 30000)
    {
        var tcs = new TaskCompletionSource<WorkerResponse>();
        _pending[requestId] = tcs;

        using var cts = new CancellationTokenSource(timeoutMs);
        cts.Token.Register(() => tcs.TrySetException(
            new TimeoutException($"Worker {Config.Name} did not respond within {timeoutMs}ms")));

        return await tcs.Task;
    }

    public async IAsyncEnumerable<string> StreamResponseAsync(string requestId)
    {
        var channel = System.Threading.Channels.Channel.CreateUnbounded<string>();

        // Register a streaming handler
        _ = Task.Run(async () =>
        {
            var tcs = new TaskCompletionSource<WorkerResponse>();
            _pending[$"{requestId}_stream"] = tcs;
            try
            {
                while (true)
                {
                    var response = await tcs.Task;
                    if (response.IsStreamEnd) break;
                    if (response.Token != null)
                        await channel.Writer.WriteAsync(response.Token);

                    // Reset for next token
                    tcs = new TaskCompletionSource<WorkerResponse>();
                    _pending[$"{requestId}_stream"] = tcs;
                }
            }
            finally
            {
                channel.Writer.Complete();
            }
        });

        await foreach (var token in channel.Reader.ReadAllAsync())
        {
            yield return token;
        }
    }

    private async Task ReadLoopAsync(CancellationToken ct)
    {
        try
        {
            while (!ct.IsCancellationRequested && _reader != null)
            {
                var line = await _reader.ReadLineAsync(ct);
                if (line == null) break;

                try
                {
                    var response = JsonSerializer.Deserialize<WorkerResponse>(line);
                    if (response?.RequestId != null)
                    {
                        // Check for streaming response first
                        if (response.IsStreamToken &&
                            _pending.TryGetValue($"{response.RequestId}_stream", out var streamTcs))
                        {
                            streamTcs.TrySetResult(response);
                        }
                        else if (_pending.TryRemove(response.RequestId, out var tcs))
                        {
                            tcs.TrySetResult(response);
                        }
                    }
                }
                catch (JsonException)
                {
                    // Malformed response, skip
                }
            }
        }
        catch (OperationCanceledException) { }
        catch (IOException) { }
    }

    public void Dispose()
    {
        _cts.Cancel();
        _reader?.Dispose();
        _writer?.Dispose();
        _stream?.Dispose();
        _client?.Dispose();
    }
}

/// <summary>
/// Worker process: loads model shards and executes forward passes.
/// Runs as a standalone process on each node.
///
/// Usage:
///   var worker = new InferenceWorker(port: 9001);
///   await worker.RunAsync(cancellationToken);
/// </summary>
public class InferenceWorker
{
    private readonly int _port;
    private readonly string _bindAddress;
    private TcpListener? _listener;
    private LayerAssignment? _assignment;
    private float[]? _activations; // Current activation buffer

    public Action<string>? OnLog { get; set; }

    public InferenceWorker(int port, string bindAddress = "0.0.0.0")
    {
        _port = port;
        _bindAddress = bindAddress;
    }

    /// <summary>Run the worker and listen for coordinator commands.</summary>
    public async Task RunAsync(CancellationToken ct)
    {
        _listener = new TcpListener(IPAddress.Parse(_bindAddress), _port);
        _listener.Start();
        OnLog?.Invoke($"Worker listening on {_bindAddress}:{_port}");

        while (!ct.IsCancellationRequested)
        {
            var client = await _listener.AcceptTcpClientAsync(ct);
            _ = HandleClientAsync(client, ct);
        }
    }

    private async Task HandleClientAsync(TcpClient client, CancellationToken ct)
    {
        using var stream = client.GetStream();
        using var reader = new StreamReader(stream, Encoding.UTF8);
        using var writer = new StreamWriter(stream, Encoding.UTF8) { AutoFlush = true };

        OnLog?.Invoke($"Coordinator connected from {client.Client.RemoteEndPoint}");

        while (!ct.IsCancellationRequested)
        {
            var line = await reader.ReadLineAsync(ct);
            if (line == null) break;

            try
            {
                var request = JsonSerializer.Deserialize<WorkerRequest>(line);
                if (request == null) continue;

                var response = await ProcessRequestAsync(request);
                var json = JsonSerializer.Serialize(response);
                await writer.WriteLineAsync(json);
            }
            catch (Exception ex)
            {
                var error = new WorkerResponse
                {
                    Success = false,
                    Error = ex.Message,
                };
                await writer.WriteLineAsync(JsonSerializer.Serialize(error));
            }
        }
    }

    private async Task<WorkerResponse> ProcessRequestAsync(WorkerRequest request)
    {
        switch (request.Type)
        {
            case RequestType.LoadModel:
                return await LoadModelShardAsync(request);

            case RequestType.Generate:
                return await RunForwardPassAsync(request);

            case RequestType.ForwardActivations:
                return await ReceiveActivationsAsync(request);

            case RequestType.AllReduce:
                return await AllReduceAsync(request);

            case RequestType.Ping:
                return new WorkerResponse
                {
                    RequestId = request.RequestId,
                    Success = true,
                    WorkerName = Environment.MachineName,
                };

            case RequestType.Shutdown:
                OnLog?.Invoke("Shutdown requested.");
                return new WorkerResponse { RequestId = request.RequestId, Success = true };

            default:
                return new WorkerResponse
                {
                    RequestId = request.RequestId,
                    Success = false,
                    Error = $"Unknown request type: {request.Type}",
                };
        }
    }

    private Task<WorkerResponse> LoadModelShardAsync(WorkerRequest request)
    {
        _assignment = new LayerAssignment
        {
            LayerRange = request.LayerRange,
            ShardIndex = request.ShardIndex,
            TotalShards = request.TotalShards,
        };

        OnLog?.Invoke($"Loading shard {request.ShardIndex}/{request.TotalShards} " +
                      $"layers {request.LayerRange.Start}-{request.LayerRange.End} " +
                      $"from {request.ModelPath}");

        // In production: load specific weight shards from model file
        // For TP: load 1/N columns of QKV matrices, 1/N rows of output projection
        // For PP: load complete layers in the assigned range

        return Task.FromResult(new WorkerResponse
        {
            RequestId = request.RequestId,
            Success = true,
            Info = $"Loaded layers {request.LayerRange.Start}-{request.LayerRange.End}, " +
                   $"shard {request.ShardIndex}/{request.TotalShards}",
        });
    }

    private Task<WorkerResponse> RunForwardPassAsync(WorkerRequest request)
    {
        // In production: run transformer forward pass on assigned layers/shards
        // This is where the actual computation happens

        OnLog?.Invoke($"Forward pass for request {request.RequestId}");

        return Task.FromResult(new WorkerResponse
        {
            RequestId = request.RequestId,
            Success = true,
            GeneratedText = "", // Would contain actual generated text
        });
    }

    private Task<WorkerResponse> ReceiveActivationsAsync(WorkerRequest request)
    {
        // Pipeline parallelism: receive activations from previous stage
        _activations = request.Activations;
        OnLog?.Invoke($"Received activations: {_activations?.Length ?? 0} floats");

        return Task.FromResult(new WorkerResponse
        {
            RequestId = request.RequestId,
            Success = true,
        });
    }

    private Task<WorkerResponse> AllReduceAsync(WorkerRequest request)
    {
        // Tensor parallelism: all-reduce partial sums
        // In production: use NCCL for GPU-to-GPU, or ring all-reduce over TCP
        OnLog?.Invoke("All-reduce step");

        return Task.FromResult(new WorkerResponse
        {
            RequestId = request.RequestId,
            Success = true,
            Activations = _activations, // Return reduced activations
        });
    }
}

/// <summary>
/// All-reduce implementations for combining partial results across workers.
/// </summary>
public static class AllReduce
{
    /// <summary>
    /// Ring all-reduce: O(N) bandwidth-optimal algorithm.
    ///
    /// Phase 1 (Reduce-Scatter): each worker sends 1/N of its data clockwise,
    ///   receiving and adding partial sums. After N-1 steps, each worker holds
    ///   the complete sum for 1/N of the data.
    ///
    /// Phase 2 (All-Gather): each worker sends its complete 1/N clockwise.
    ///   After N-1 steps, every worker has the full reduced result.
    ///
    /// Total data transferred per worker: 2 * (N-1)/N * DataSize
    /// (optimal; matches the theoretical lower bound)
    /// </summary>
    public static async Task RingAllReduceAsync(
        IReadOnlyList<WorkerConnection> workers,
        float[][] partialResults)
    {
        int n = workers.Count;
        int dataSize = partialResults[0].Length;
        int chunkSize = dataSize / n;

        // Phase 1: Reduce-Scatter
        for (int step = 0; step < n - 1; step++)
        {
            var tasks = new List<Task>();
            for (int rank = 0; rank < n; rank++)
            {
                int sendChunk = (rank - step + n) % n;
                int recvChunk = (rank - step - 1 + n) % n;
                int sendTo = (rank + 1) % n;
                int recvFrom = (rank - 1 + n) % n;

                // In production: async send/recv between workers
                // Each worker adds received chunk to its local partial result
                int chunkStart = recvChunk * chunkSize;
                int chunkEnd = Math.Min(chunkStart + chunkSize, dataSize);

                for (int j = chunkStart; j < chunkEnd; j++)
                    partialResults[rank][j] += partialResults[recvFrom][j];
            }
        }

        // Phase 2: All-Gather
        for (int step = 0; step < n - 1; step++)
        {
            for (int rank = 0; rank < n; rank++)
            {
                int sendChunk = (rank - step + 1 + n) % n;
                int recvChunk = (rank - step + n) % n;
                int sendTo = (rank + 1) % n;

                int chunkStart = recvChunk * chunkSize;
                int chunkEnd = Math.Min(chunkStart + chunkSize, dataSize);

                // Copy complete chunk from sender to receiver
                Array.Copy(partialResults[sendTo], chunkStart,
                          partialResults[rank], chunkStart, chunkEnd - chunkStart);
            }
        }

        await Task.CompletedTask; // In production: actual async network transfers
    }

    /// <summary>
    /// Simple all-reduce for small data: gather at rank 0, broadcast back.
    /// </summary>
    public static void NaiveAllReduce(float[][] partialResults)
    {
        int n = partialResults.Length;
        int len = partialResults[0].Length;

        // Sum into rank 0
        for (int i = 1; i < n; i++)
            for (int j = 0; j < len; j++)
                partialResults[0][j] += partialResults[i][j];

        // Broadcast back
        for (int i = 1; i < n; i++)
            Array.Copy(partialResults[0], partialResults[i], len);
    }
}

// ─── Protocol Types ────────────────────────────────────────

public enum ParallelismStrategy
{
    TensorParallel,
    PipelineParallel,
}

public enum RequestType
{
    Ping,
    LoadModel,
    Generate,
    GenerateStreaming,
    ForwardActivations,
    AllReduce,
    Shutdown,
}

public class WorkerRequest
{
    [JsonPropertyName("request_id")] public string? RequestId { get; set; }
    [JsonPropertyName("type")] public RequestType Type { get; set; }
    [JsonPropertyName("prompt")] public string? Prompt { get; set; }
    [JsonPropertyName("max_tokens")] public int MaxTokens { get; set; }
    [JsonPropertyName("model_path")] public string? ModelPath { get; set; }
    [JsonPropertyName("layer_range")] public LayerRange LayerRange { get; set; }
    [JsonPropertyName("shard_index")] public int ShardIndex { get; set; }
    [JsonPropertyName("total_shards")] public int TotalShards { get; set; }
    [JsonPropertyName("activations")] public float[]? Activations { get; set; }
    [JsonPropertyName("topology")] public ClusterTopology? Topology { get; set; }
}

public class WorkerResponse
{
    [JsonPropertyName("request_id")] public string? RequestId { get; set; }
    [JsonPropertyName("success")] public bool Success { get; set; }
    [JsonPropertyName("error")] public string? Error { get; set; }
    [JsonPropertyName("generated_text")] public string? GeneratedText { get; set; }
    [JsonPropertyName("token")] public string? Token { get; set; }
    [JsonPropertyName("is_stream_token")] public bool IsStreamToken { get; set; }
    [JsonPropertyName("is_stream_end")] public bool IsStreamEnd { get; set; }
    [JsonPropertyName("activations")] public float[]? Activations { get; set; }
    [JsonPropertyName("worker_name")] public string? WorkerName { get; set; }
    [JsonPropertyName("info")] public string? Info { get; set; }
}

public struct LayerRange
{
    [JsonPropertyName("start")] public int Start { get; set; }
    [JsonPropertyName("end")] public int End { get; set; }

    public LayerRange(int start, int end) { Start = start; End = end; }

    public static implicit operator LayerRange((int start, int end) tuple) =>
        new(tuple.start, tuple.end);

    public static implicit operator (int Start, int End)(LayerRange lr) =>
        (lr.Start, lr.End);

    public int Count => End - Start + 1;
    public override string ToString() => $"[{Start}..{End}]";
}

public class LayerAssignment
{
    [JsonPropertyName("worker_name")] public string WorkerName { get; set; } = "";
    [JsonPropertyName("layer_range")] public LayerRange LayerRange { get; set; }
    [JsonPropertyName("shard_index")] public int ShardIndex { get; set; }
    [JsonPropertyName("total_shards")] public int TotalShards { get; set; }
    [JsonPropertyName("rank")] public int Rank { get; set; }

    public override string ToString() =>
        $"{WorkerName}: layers {LayerRange}, shard {ShardIndex}/{TotalShards}, rank {Rank}";
}

public class ClusterTopology
{
    [JsonPropertyName("strategy")] public ParallelismStrategy Strategy { get; set; }
    [JsonPropertyName("total_layers")] public int TotalLayers { get; set; }
    [JsonPropertyName("assignments")] public List<LayerAssignment> Assignments { get; set; } = new();

    public override string ToString() =>
        $"{Strategy}, {TotalLayers} layers, {Assignments.Count} workers";
}

// ─── Config ────────────────────────────────────────────────

public class DistributedConfig
{
    [JsonPropertyName("enabled")] public bool Enabled { get; set; } = false;
    [JsonPropertyName("strategy")] public ParallelismStrategy Strategy { get; set; } = ParallelismStrategy.TensorParallel;
    [JsonPropertyName("model_path")] public string ModelPath { get; set; } = "";
    [JsonPropertyName("total_layers")] public int TotalLayers { get; set; } = 32;
    [JsonPropertyName("workers")] public List<WorkerConfig> Workers { get; set; } = new();
    [JsonPropertyName("timeout_ms")] public int TimeoutMs { get; set; } = 30000;
    [JsonPropertyName("model_load_timeout_ms")] public int ModelLoadTimeoutMs { get; set; } = 300000;
}

public class WorkerConfig
{
    [JsonPropertyName("name")] public string Name { get; set; } = "";
    [JsonPropertyName("host")] public string Host { get; set; } = "localhost";
    [JsonPropertyName("port")] public int Port { get; set; } = 9001;
    [JsonPropertyName("gpu_ids")] public List<int> GpuIds { get; set; } = new();
    [JsonPropertyName("memory_gb")] public float MemoryGb { get; set; } = 0;
}
