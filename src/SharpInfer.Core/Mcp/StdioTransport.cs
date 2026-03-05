using System.Collections.Concurrent;
using System.Diagnostics;
using System.Text;
using System.Text.Json;

namespace SharpInfer.Core.Mcp;

/// <summary>
/// MCP transport over stdio — spawns a child process and communicates
/// via JSON-RPC messages over stdin/stdout.
///
/// Each message is a single JSON line (newline-delimited JSON).
/// Server stderr is captured for diagnostics.
/// </summary>
public class StdioTransport : IMcpTransport
{
    private readonly string _command;
    private readonly string[] _args;
    private readonly Dictionary<string, string>? _env;

    private Process? _process;
    private Task? _readLoop;
    private CancellationTokenSource? _cts;

    // Pending requests awaiting responses, keyed by request ID
    private readonly ConcurrentDictionary<int, TaskCompletionSource<JsonRpcResponse>> _pending = new();
    private int _nextId = 1;

    // Captured server notifications and stderr
    private readonly ConcurrentQueue<JsonRpcNotification> _notifications = new();
    private readonly ConcurrentQueue<string> _stderrLines = new();

    /// <summary>Callback for server-initiated notifications.</summary>
    public Action<JsonRpcNotification>? OnNotification { get; set; }

    /// <summary>Callback for stderr output from the server process.</summary>
    public Action<string>? OnStderr { get; set; }

    public bool IsConnected => _process is { HasExited: false };

    public StdioTransport(string command, string[]? args = null, Dictionary<string, string>? env = null)
    {
        _command = command;
        _args = args ?? Array.Empty<string>();
        _env = env;
    }

    public Task ConnectAsync(CancellationToken ct = default)
    {
        var psi = new ProcessStartInfo
        {
            FileName = _command,
            UseShellExecute = false,
            RedirectStandardInput = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true,
            StandardOutputEncoding = Encoding.UTF8,
            StandardErrorEncoding = Encoding.UTF8,
        };

        foreach (var arg in _args)
            psi.ArgumentList.Add(arg);

        // Set custom environment variables
        if (_env != null)
        {
            foreach (var (key, value) in _env)
                psi.Environment[key] = value;
        }

        _process = Process.Start(psi)
            ?? throw new InvalidOperationException($"Failed to start MCP server process: {_command}");

        _cts = new CancellationTokenSource();

        // Start background read loops
        _readLoop = Task.WhenAll(
            ReadStdoutLoopAsync(_cts.Token),
            ReadStderrLoopAsync(_cts.Token)
        );

        return Task.CompletedTask;
    }

    public async Task<JsonRpcResponse> SendRequestAsync(JsonRpcRequest request, CancellationToken ct = default)
    {
        if (!IsConnected)
            throw new InvalidOperationException("MCP transport is not connected.");

        request.Id = Interlocked.Increment(ref _nextId);
        var tcs = new TaskCompletionSource<JsonRpcResponse>(TaskCreationOptions.RunContinuationsAsynchronously);

        _pending[request.Id] = tcs;

        // Register cancellation
        await using var reg = ct.Register(() =>
        {
            if (_pending.TryRemove(request.Id, out var removed))
                removed.TrySetCanceled(ct);
        });

        try
        {
            string json = JsonSerializer.Serialize(request, JsonOpts.Default);
            await WriteLineAsync(json);
            return await tcs.Task;
        }
        catch
        {
            _pending.TryRemove(request.Id, out _);
            throw;
        }
    }

    public async Task SendNotificationAsync(JsonRpcNotification notification, CancellationToken ct = default)
    {
        if (!IsConnected)
            throw new InvalidOperationException("MCP transport is not connected.");

        string json = JsonSerializer.Serialize(notification, JsonOpts.Default);
        await WriteLineAsync(json);
    }

    public async Task DisconnectAsync()
    {
        if (_process == null) return;

        try
        {
            // Close stdin to signal the server to exit
            _process.StandardInput.Close();

            // Wait briefly for graceful shutdown
            if (!_process.WaitForExit(5000))
            {
                _process.Kill();
                _process.WaitForExit(2000);
            }
        }
        catch { }

        _cts?.Cancel();

        try
        {
            if (_readLoop != null)
                await _readLoop;
        }
        catch (OperationCanceledException) { }

        // Fail any pending requests
        foreach (var (id, tcs) in _pending)
        {
            tcs.TrySetException(new InvalidOperationException("MCP transport disconnected."));
            _pending.TryRemove(id, out _);
        }

        _process?.Dispose();
        _process = null;
    }

    public void Dispose()
    {
        DisconnectAsync().GetAwaiter().GetResult();
        _cts?.Dispose();
    }

    #region Internal I/O

    private readonly SemaphoreSlim _writeLock = new(1, 1);

    private async Task WriteLineAsync(string json)
    {
        await _writeLock.WaitAsync();
        try
        {
            await _process!.StandardInput.WriteLineAsync(json);
            await _process.StandardInput.FlushAsync();
        }
        finally
        {
            _writeLock.Release();
        }
    }

    private async Task ReadStdoutLoopAsync(CancellationToken ct)
    {
        var reader = _process!.StandardOutput;

        while (!ct.IsCancellationRequested)
        {
            string? line;
            try
            {
                line = await reader.ReadLineAsync();
            }
            catch (OperationCanceledException) { break; }
            catch { break; }

            if (line == null) break; // EOF — process exited
            if (string.IsNullOrWhiteSpace(line)) continue;

            ProcessMessage(line);
        }
    }

    private async Task ReadStderrLoopAsync(CancellationToken ct)
    {
        var reader = _process!.StandardError;

        while (!ct.IsCancellationRequested)
        {
            string? line;
            try
            {
                line = await reader.ReadLineAsync();
            }
            catch { break; }

            if (line == null) break;

            _stderrLines.Enqueue(line);
            OnStderr?.Invoke(line);
        }
    }

    private void ProcessMessage(string json)
    {
        try
        {
            using var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;

            // Check if it's a response (has "id" and either "result" or "error")
            if (root.TryGetProperty("id", out var idProp) && !idProp.ValueKind.Equals(JsonValueKind.Null))
            {
                var response = JsonSerializer.Deserialize<JsonRpcResponse>(json, JsonOpts.Default);
                if (response?.Id != null && _pending.TryRemove(response.Id.Value, out var tcs))
                {
                    tcs.TrySetResult(response);
                }
            }
            // Otherwise it's a notification (no id)
            else if (root.TryGetProperty("method", out _))
            {
                var notification = JsonSerializer.Deserialize<JsonRpcNotification>(json, JsonOpts.Default);
                if (notification != null)
                {
                    _notifications.Enqueue(notification);
                    OnNotification?.Invoke(notification);
                }
            }
        }
        catch (JsonException)
        {
            // Non-JSON output from the server — ignore
        }
    }

    #endregion

    /// <summary>Get captured stderr lines (for diagnostics).</summary>
    public IReadOnlyCollection<string> GetStderrOutput()
    {
        return _stderrLines.ToArray();
    }
}

/// <summary>Shared JSON serialization options.</summary>
internal static class JsonOpts
{
    public static readonly JsonSerializerOptions Default = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull,
        WriteIndented = false,
    };
}
