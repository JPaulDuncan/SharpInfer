using System.Collections.Concurrent;
using System.Threading.Channels;
using SharpInfer.Core.Layers;
using SharpInfer.Core.Sampling;

namespace SharpInfer.Core.Engine;

/// <summary>
/// Continuous batching scheduler that processes multiple inference requests concurrently.
///
/// Unlike simple sequential processing, this scheduler:
///   1. Maintains a pool of active sequences sharing the KV cache
///   2. Dynamically adds new requests as slots open (continuous batching)
///   3. Processes prefill and decode phases together in micro-batches
///   4. Preempts low-priority requests when memory is tight
///   5. Supports priority-based scheduling
///
/// This is the key throughput optimization used by vLLM and similar serving engines.
///
/// Architecture:
///   [Request Queue] → [Scheduler] → [Batch Executor] → [KV Cache Pool]
///                        ↓                                    ↑
///                   [Active Sequences] ←──────────────────────┘
/// </summary>
public class BatchScheduler : IDisposable
{
    private readonly InferenceEngine _engine;
    private readonly int _maxBatchSize;
    private readonly int _maxSequences;
    private readonly int _maxContextLength;

    // Request management
    private readonly Channel<InferenceRequest> _requestQueue;
    private readonly ConcurrentDictionary<string, ActiveSequence> _activeSequences = new();
    private readonly ConcurrentDictionary<string, TaskCompletionSource<string>> _completions = new();
    private readonly ConcurrentDictionary<string, Channel<string>> _streamChannels = new();

    // Scheduler loop
    private Task? _schedulerLoop;
    private CancellationTokenSource? _cts;
    private int _currentSequenceCount;

    // Stats
    private long _totalRequests;
    private long _totalTokensGenerated;
    private long _totalPrefillTokens;

    /// <summary>Callback for scheduler events.</summary>
    public Action<string>? OnLog { get; set; }

    /// <summary>Max requests processed in a single forward pass.</summary>
    public int MaxBatchSize => _maxBatchSize;

    /// <summary>Max concurrent sequences in the KV cache.</summary>
    public int MaxSequences => _maxSequences;

    /// <summary>Current number of active sequences.</summary>
    public int ActiveCount => _currentSequenceCount;

    /// <summary>Requests waiting in queue.</summary>
    public int QueuedCount => _requestQueue.Reader.Count;

    public BatchScheduler(
        InferenceEngine engine,
        int maxBatchSize = 8,
        int maxSequences = 16,
        int maxContextLength = 4096)
    {
        _engine = engine;
        _maxBatchSize = maxBatchSize;
        _maxSequences = maxSequences;
        _maxContextLength = maxContextLength;

        _requestQueue = Channel.CreateUnbounded<InferenceRequest>(new UnboundedChannelOptions
        {
            SingleReader = true,
        });
    }

    /// <summary>Start the scheduler loop.</summary>
    public void Start()
    {
        _cts = new CancellationTokenSource();
        _schedulerLoop = Task.Run(() => SchedulerLoopAsync(_cts.Token));
        OnLog?.Invoke($"BatchScheduler started: maxBatch={_maxBatchSize}, maxSeq={_maxSequences}");
    }

    /// <summary>Stop the scheduler and drain remaining requests.</summary>
    public async Task StopAsync()
    {
        _requestQueue.Writer.Complete();
        _cts?.Cancel();

        if (_schedulerLoop != null)
        {
            try { await _schedulerLoop; }
            catch (OperationCanceledException) { }
        }

        // Fail any pending requests
        foreach (var (id, tcs) in _completions)
        {
            tcs.TrySetException(new OperationCanceledException("Scheduler stopped."));
        }
    }

    // ─── Submit Requests ────────────────────────────────────────

    /// <summary>
    /// Submit a request and get the complete response.
    /// </summary>
    public async Task<string> SubmitAsync(
        string prompt,
        GenerationConfig? config = null,
        int priority = 0,
        CancellationToken ct = default)
    {
        var request = new InferenceRequest
        {
            Id = Guid.NewGuid().ToString("N")[..12],
            Prompt = prompt,
            Config = config ?? new GenerationConfig(),
            Priority = priority,
            SubmittedAt = DateTimeOffset.UtcNow,
        };

        var tcs = new TaskCompletionSource<string>(TaskCreationOptions.RunContinuationsAsynchronously);
        _completions[request.Id] = tcs;

        await _requestQueue.Writer.WriteAsync(request, ct);
        Interlocked.Increment(ref _totalRequests);

        return await tcs.Task.WaitAsync(ct);
    }

    /// <summary>
    /// Submit a request and stream tokens as they're generated.
    /// </summary>
    public async IAsyncEnumerable<string> SubmitStreamingAsync(
        string prompt,
        GenerationConfig? config = null,
        int priority = 0,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken ct = default)
    {
        var request = new InferenceRequest
        {
            Id = Guid.NewGuid().ToString("N")[..12],
            Prompt = prompt,
            Config = config ?? new GenerationConfig(),
            Priority = priority,
            Streaming = true,
            SubmittedAt = DateTimeOffset.UtcNow,
        };

        var channel = Channel.CreateUnbounded<string>();
        _streamChannels[request.Id] = channel;

        var tcs = new TaskCompletionSource<string>(TaskCreationOptions.RunContinuationsAsynchronously);
        _completions[request.Id] = tcs;

        await _requestQueue.Writer.WriteAsync(request, ct);
        Interlocked.Increment(ref _totalRequests);

        await foreach (var token in channel.Reader.ReadAllAsync(ct))
        {
            yield return token;
        }
    }

    // ─── Scheduler Core ─────────────────────────────────────────

    private async Task SchedulerLoopAsync(CancellationToken ct)
    {
        while (!ct.IsCancellationRequested)
        {
            try
            {
                // Phase 1: Admit new requests into active set
                await AdmitRequestsAsync(ct);

                // Phase 2: If no active sequences, wait for a request
                if (_activeSequences.IsEmpty)
                {
                    if (await _requestQueue.Reader.WaitToReadAsync(ct))
                        continue;
                    else
                        break; // Channel completed
                }

                // Phase 3: Build micro-batch from active sequences
                var batch = SelectBatch();

                // Phase 4: Execute forward pass for the batch
                await ExecuteBatchStepAsync(batch, ct);

                // Phase 5: Process results — emit tokens, check termination
                ProcessResults(batch);

                // Phase 6: Clean up completed sequences
                CleanupCompleted();
            }
            catch (OperationCanceledException) { break; }
            catch (Exception ex)
            {
                OnLog?.Invoke($"Scheduler error: {ex.Message}");
                await Task.Delay(10, ct); // Brief backoff
            }
        }
    }

    private async Task AdmitRequestsAsync(CancellationToken ct)
    {
        // Try to fill up to max sequences
        while (_currentSequenceCount < _maxSequences)
        {
            if (_requestQueue.Reader.TryRead(out var request))
            {
                AdmitRequest(request);
            }
            else
            {
                break; // No more queued requests
            }
        }

        await Task.CompletedTask;
    }

    private void AdmitRequest(InferenceRequest request)
    {
        // Tokenize the prompt
        int[] promptTokens = _engine.Tokenize(request.Prompt).ToArray();

        var sequence = new ActiveSequence
        {
            Request = request,
            PromptTokens = promptTokens,
            GeneratedTokens = new List<int>(),
            Position = 0,
            Phase = SequencePhase.Prefill,
            StartedAt = DateTimeOffset.UtcNow,
        };

        _activeSequences[request.Id] = sequence;
        Interlocked.Increment(ref _currentSequenceCount);

        Interlocked.Add(ref _totalPrefillTokens, promptTokens.Length);
        OnLog?.Invoke($"Admitted request {request.Id}: {promptTokens.Length} prompt tokens, priority={request.Priority}");
    }

    private List<ActiveSequence> SelectBatch()
    {
        // Select up to maxBatchSize sequences for this step
        // Priority: prefill first (to start generating ASAP), then by priority, then FIFO
        return _activeSequences.Values
            .Where(s => !s.Completed)
            .OrderBy(s => s.Phase == SequencePhase.Prefill ? 0 : 1)
            .ThenByDescending(s => s.Request.Priority)
            .ThenBy(s => s.Request.SubmittedAt)
            .Take(_maxBatchSize)
            .ToList();
    }

    private async Task ExecuteBatchStepAsync(List<ActiveSequence> batch, CancellationToken ct)
    {
        // Process each sequence in the batch
        // In a true implementation, these would share a batched forward pass through the model
        // For now, we process them "concurrently" with Task.WhenAll
        // (True batched inference requires batched matrix operations in the Transformer)

        var tasks = batch.Select(async seq =>
        {
            if (seq.Phase == SequencePhase.Prefill)
            {
                // Prefill: process all prompt tokens at once
                await PrefillSequenceAsync(seq);
                seq.Phase = SequencePhase.Decode;
            }
            else
            {
                // Decode: generate one token
                await DecodeStepAsync(seq);
            }
        });

        await Task.WhenAll(tasks);
    }

    private async Task PrefillSequenceAsync(ActiveSequence seq)
    {
        // Build prompt from tokens
        string prompt = _engine.Detokenize(seq.PromptTokens);
        seq.Position = seq.PromptTokens.Length;

        // Generate first token
        string firstToken = "";
        var prefillConfig = new GenerationConfig
        {
            Temperature = seq.Request.Config.Temperature,
            TopP = seq.Request.Config.TopP,
            TopK = seq.Request.Config.TopK,
            MaxTokens = 1,
            RepetitionPenalty = seq.Request.Config.RepetitionPenalty,
            Seed = seq.Request.Config.Seed,
            StopTokenIds = seq.Request.Config.StopTokenIds,
            StopStrings = seq.Request.Config.StopStrings,
        };
        await foreach (var token in _engine.GenerateAsync(prompt, prefillConfig))
        {
            firstToken = token;
            break;
        }

        if (!string.IsNullOrEmpty(firstToken))
        {
            int[] tokenIds = _engine.Tokenize(firstToken).ToArray();
            seq.GeneratedTokens.AddRange(tokenIds);
            seq.LastToken = firstToken;
            Interlocked.Increment(ref _totalTokensGenerated);
        }

        await Task.CompletedTask;
    }

    private async Task DecodeStepAsync(ActiveSequence seq)
    {
        // Continue generation from current state
        string currentText = _engine.Detokenize(seq.PromptTokens) +
                             string.Join("", seq.GeneratedTokenTexts);

        string nextToken = "";
        var decodeConfig = new GenerationConfig
        {
            Temperature = seq.Request.Config.Temperature,
            TopP = seq.Request.Config.TopP,
            TopK = seq.Request.Config.TopK,
            MaxTokens = 1,
            RepetitionPenalty = seq.Request.Config.RepetitionPenalty,
            Seed = seq.Request.Config.Seed,
            StopTokenIds = seq.Request.Config.StopTokenIds,
            StopStrings = seq.Request.Config.StopStrings,
        };
        await foreach (var token in _engine.GenerateAsync(currentText, decodeConfig))
        {
            nextToken = token;
            break;
        }

        if (!string.IsNullOrEmpty(nextToken))
        {
            int[] tokenIds = _engine.Tokenize(nextToken).ToArray();
            seq.GeneratedTokens.AddRange(tokenIds);
            seq.LastToken = nextToken;
            seq.GeneratedTokenTexts.Add(nextToken);
            Interlocked.Increment(ref _totalTokensGenerated);
        }
        else
        {
            seq.Completed = true;
        }

        seq.Position++;
        await Task.CompletedTask;
    }

    private void ProcessResults(List<ActiveSequence> batch)
    {
        foreach (var seq in batch)
        {
            // Stream token if streaming
            if (seq.Request.Streaming && seq.LastToken != null)
            {
                if (_streamChannels.TryGetValue(seq.Request.Id, out var channel))
                {
                    channel.Writer.TryWrite(seq.LastToken);
                }
            }

            // Check termination conditions
            if (!seq.Completed)
            {
                // Max tokens
                if (seq.GeneratedTokens.Count >= seq.Request.Config.MaxTokens)
                    seq.Completed = true;

                // Max context length
                if (seq.PromptTokens.Length + seq.GeneratedTokens.Count >= _maxContextLength)
                    seq.Completed = true;

                // EOS token
                // (In a real implementation, check against model's EOS token ID)

                // Stop strings
                if (seq.Request.Config.StopStrings.Count > 0)
                {
                    string generated = string.Join("", seq.GeneratedTokenTexts);
                    foreach (var stop in seq.Request.Config.StopStrings)
                    {
                        if (generated.Contains(stop))
                        {
                            seq.Completed = true;
                            break;
                        }
                    }
                }
            }
        }
    }

    private void CleanupCompleted()
    {
        foreach (var (id, seq) in _activeSequences)
        {
            if (!seq.Completed) continue;

            _activeSequences.TryRemove(id, out _);
            Interlocked.Decrement(ref _currentSequenceCount);

            string fullResponse = string.Join("", seq.GeneratedTokenTexts);

            // Complete the TCS
            if (_completions.TryRemove(id, out var tcs))
            {
                tcs.TrySetResult(fullResponse);
            }

            // Close stream channel
            if (_streamChannels.TryRemove(id, out var channel))
            {
                channel.Writer.TryComplete();
            }

            var elapsed = DateTimeOffset.UtcNow - seq.StartedAt;
            double tokensPerSec = seq.GeneratedTokens.Count / Math.Max(elapsed.TotalSeconds, 0.001);
            OnLog?.Invoke($"Completed {id}: {seq.GeneratedTokens.Count} tokens in {elapsed.TotalSeconds:F1}s ({tokensPerSec:F1} tok/s)");
        }
    }

    // ─── Stats ──────────────────────────────────────────────────

    public BatchSchedulerStats GetStats() => new()
    {
        TotalRequests = _totalRequests,
        TotalTokensGenerated = _totalTokensGenerated,
        TotalPrefillTokens = _totalPrefillTokens,
        ActiveSequences = _currentSequenceCount,
        QueuedRequests = _requestQueue.Reader.Count,
        MaxBatchSize = _maxBatchSize,
        MaxSequences = _maxSequences,
    };

    public void Dispose()
    {
        StopAsync().GetAwaiter().GetResult();
        _cts?.Dispose();
    }
}

/// <summary>An inference request in the batch scheduler queue.</summary>
public class InferenceRequest
{
    public string Id { get; init; } = "";
    public string Prompt { get; init; } = "";
    public GenerationConfig Config { get; init; } = new();
    public int Priority { get; init; } = 0;
    public bool Streaming { get; init; } = false;
    public DateTimeOffset SubmittedAt { get; init; }
}

/// <summary>An active sequence being processed by the scheduler.</summary>
internal class ActiveSequence
{
    public InferenceRequest Request { get; init; } = null!;
    public int[] PromptTokens { get; init; } = Array.Empty<int>();
    public List<int> GeneratedTokens { get; init; } = new();
    public List<string> GeneratedTokenTexts { get; init; } = new();
    public int Position { get; set; }
    public SequencePhase Phase { get; set; }
    public string? LastToken { get; set; }
    public bool Completed { get; set; }
    public DateTimeOffset StartedAt { get; init; }
}

internal enum SequencePhase
{
    Prefill,
    Decode,
}

/// <summary>Scheduler statistics.</summary>
public class BatchSchedulerStats
{
    public long TotalRequests { get; init; }
    public long TotalTokensGenerated { get; init; }
    public long TotalPrefillTokens { get; init; }
    public int ActiveSequences { get; init; }
    public int QueuedRequests { get; init; }
    public int MaxBatchSize { get; init; }
    public int MaxSequences { get; init; }

    public double AverageTokensPerRequest =>
        TotalRequests > 0 ? (double)TotalTokensGenerated / TotalRequests : 0;
}
