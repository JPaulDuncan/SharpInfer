using System.Collections.Concurrent;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;

namespace SharpInfer.Core.Mcp;

/// <summary>
/// MCP transport over HTTP with Server-Sent Events (SSE).
///
/// Protocol:
///   1. Client connects to the SSE endpoint (GET) to receive messages
///   2. Server sends an "endpoint" event with the POST URL for sending messages
///   3. Client sends JSON-RPC requests via POST to that endpoint
///   4. Server streams responses back via the SSE connection
///
/// This follows the MCP SSE transport specification.
/// </summary>
public class SseTransport : IMcpTransport
{
    private readonly string _sseUrl;
    private readonly Dictionary<string, string>? _headers;
    private readonly HttpClient _httpClient;

    private string? _postEndpoint;
    private Task? _sseLoop;
    private CancellationTokenSource? _cts;

    private readonly ConcurrentDictionary<int, TaskCompletionSource<JsonRpcResponse>> _pending = new();
    private int _nextId = 1;

    /// <summary>Callback for server-initiated notifications.</summary>
    public Action<JsonRpcNotification>? OnNotification { get; set; }

    public bool IsConnected => _postEndpoint != null && !(_cts?.IsCancellationRequested ?? true);

    public SseTransport(string sseUrl, Dictionary<string, string>? headers = null)
    {
        _sseUrl = sseUrl;
        _headers = headers;
        _httpClient = new HttpClient { Timeout = TimeSpan.FromMinutes(30) }; // Long timeout for SSE
    }

    public async Task ConnectAsync(CancellationToken ct = default)
    {
        _cts = new CancellationTokenSource();

        // Apply custom headers
        if (_headers != null)
        {
            foreach (var (key, value) in _headers)
                _httpClient.DefaultRequestHeaders.TryAddWithoutValidation(key, value);
        }

        // Start SSE connection — we need to receive the endpoint event first
        var endpointReady = new TaskCompletionSource<string>(TaskCreationOptions.RunContinuationsAsynchronously);

        _sseLoop = Task.Run(async () =>
        {
            try
            {
                await SseReadLoopAsync(endpointReady, _cts.Token);
            }
            catch (OperationCanceledException) { }
            catch (Exception ex)
            {
                endpointReady.TrySetException(ex);
            }
        }, _cts.Token);

        // Wait for the server to send us the POST endpoint
        using var timeoutCts = new CancellationTokenSource(TimeSpan.FromSeconds(30));
        using var linked = CancellationTokenSource.CreateLinkedTokenSource(ct, timeoutCts.Token);

        try
        {
            _postEndpoint = await endpointReady.Task.WaitAsync(linked.Token);
        }
        catch (OperationCanceledException)
        {
            throw new TimeoutException(
                $"Timed out waiting for MCP SSE endpoint event from {_sseUrl}");
        }
    }

    public async Task<JsonRpcResponse> SendRequestAsync(JsonRpcRequest request, CancellationToken ct = default)
    {
        if (!IsConnected || _postEndpoint == null)
            throw new InvalidOperationException("MCP SSE transport is not connected.");

        request.Id = Interlocked.Increment(ref _nextId);
        var tcs = new TaskCompletionSource<JsonRpcResponse>(TaskCreationOptions.RunContinuationsAsynchronously);
        _pending[request.Id] = tcs;

        await using var reg = ct.Register(() =>
        {
            if (_pending.TryRemove(request.Id, out var removed))
                removed.TrySetCanceled(ct);
        });

        try
        {
            string json = JsonSerializer.Serialize(request, JsonOpts.Default);
            var content = new StringContent(json, Encoding.UTF8, "application/json");
            var response = await _httpClient.PostAsync(_postEndpoint, content, ct);
            response.EnsureSuccessStatusCode();

            // The actual JSON-RPC response comes back via the SSE stream, not the HTTP response
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
        if (!IsConnected || _postEndpoint == null)
            throw new InvalidOperationException("MCP SSE transport is not connected.");

        string json = JsonSerializer.Serialize(notification, JsonOpts.Default);
        var content = new StringContent(json, Encoding.UTF8, "application/json");
        var response = await _httpClient.PostAsync(_postEndpoint, content, ct);
        response.EnsureSuccessStatusCode();
    }

    public async Task DisconnectAsync()
    {
        _cts?.Cancel();

        if (_sseLoop != null)
        {
            try { await _sseLoop; }
            catch (OperationCanceledException) { }
        }

        foreach (var (id, tcs) in _pending)
        {
            tcs.TrySetException(new InvalidOperationException("MCP SSE transport disconnected."));
            _pending.TryRemove(id, out _);
        }

        _postEndpoint = null;
    }

    public void Dispose()
    {
        DisconnectAsync().GetAwaiter().GetResult();
        _httpClient.Dispose();
        _cts?.Dispose();
    }

    #region SSE Parsing

    private async Task SseReadLoopAsync(TaskCompletionSource<string> endpointReady, CancellationToken ct)
    {
        var request = new HttpRequestMessage(HttpMethod.Get, _sseUrl);
        request.Headers.Accept.Add(new MediaTypeWithQualityHeaderValue("text/event-stream"));

        using var response = await _httpClient.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, ct);
        response.EnsureSuccessStatusCode();

        using var stream = await response.Content.ReadAsStreamAsync(ct);
        using var reader = new StreamReader(stream, Encoding.UTF8);

        string? currentEvent = null;
        var dataLines = new StringBuilder();

        while (!ct.IsCancellationRequested)
        {
            var line = await reader.ReadLineAsync();
            if (line == null) break; // Stream closed

            if (line == "")
            {
                // Empty line = end of event
                if (currentEvent != null || dataLines.Length > 0)
                {
                    ProcessSseEvent(currentEvent ?? "message", dataLines.ToString().TrimEnd('\n'), endpointReady);
                    currentEvent = null;
                    dataLines.Clear();
                }
                continue;
            }

            if (line.StartsWith("event:"))
            {
                currentEvent = line[6..].Trim();
            }
            else if (line.StartsWith("data:"))
            {
                if (dataLines.Length > 0) dataLines.Append('\n');
                dataLines.Append(line[5..].TrimStart());
            }
            // Lines starting with ":" are comments — ignore
            // "id:" and "retry:" fields — ignored for now
        }
    }

    private void ProcessSseEvent(string eventType, string data, TaskCompletionSource<string> endpointReady)
    {
        switch (eventType)
        {
            case "endpoint":
                // Server tells us where to POST requests
                string postUrl = data.Trim();

                // Handle relative URLs
                if (postUrl.StartsWith("/"))
                {
                    var baseUri = new Uri(_sseUrl);
                    postUrl = $"{baseUri.Scheme}://{baseUri.Authority}{postUrl}";
                }

                endpointReady.TrySetResult(postUrl);
                break;

            case "message":
                // JSON-RPC message from the server
                try
                {
                    using var doc = JsonDocument.Parse(data);
                    var root = doc.RootElement;

                    if (root.TryGetProperty("id", out var idProp) && idProp.ValueKind != JsonValueKind.Null)
                    {
                        // Response
                        var response = JsonSerializer.Deserialize<JsonRpcResponse>(data, JsonOpts.Default);
                        if (response?.Id != null && _pending.TryRemove(response.Id.Value, out var tcs))
                        {
                            tcs.TrySetResult(response);
                        }
                    }
                    else if (root.TryGetProperty("method", out _))
                    {
                        // Notification
                        var notification = JsonSerializer.Deserialize<JsonRpcNotification>(data, JsonOpts.Default);
                        if (notification != null)
                            OnNotification?.Invoke(notification);
                    }
                }
                catch (JsonException)
                {
                    // Malformed — ignore
                }
                break;
        }
    }

    #endregion
}
