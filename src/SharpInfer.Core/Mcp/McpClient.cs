using System.Text.Json;

namespace SharpInfer.Core.Mcp;

/// <summary>
/// MCP (Model Context Protocol) client.
///
/// Manages the full lifecycle of an MCP server connection:
///   1. Transport setup (stdio or SSE)
///   2. Initialize handshake (capabilities negotiation)
///   3. Tool discovery (tools/list)
///   4. Tool execution (tools/call)
///   5. Resource access (resources/list, resources/read)
///   6. Prompt templates (prompts/list, prompts/get)
///   7. Graceful shutdown
///
/// Usage:
///   var client = new McpClient(config);
///   await client.ConnectAsync();
///   var tools = await client.ListToolsAsync();
///   var result = await client.CallToolAsync("tool_name", args);
///   await client.DisconnectAsync();
/// </summary>
public class McpClient : IDisposable
{
    private readonly McpServerConfig _config;
    private IMcpTransport? _transport;
    private InitializeResult? _serverInfo;
    private List<McpToolDefinition>? _cachedTools;

    /// <summary>Client protocol version we advertise.</summary>
    public const string ProtocolVersion = "2024-11-05";

    /// <summary>Client info we send during init.</summary>
    public string ClientName { get; set; } = "SharpInfer";
    public string ClientVersion { get; set; } = "1.0.0";

    /// <summary>Server info received during init.</summary>
    public InitializeResult? ServerInfo => _serverInfo;

    /// <summary>Whether connected and initialized.</summary>
    public bool IsConnected => _transport?.IsConnected ?? false;

    /// <summary>The server config this client was created from.</summary>
    public McpServerConfig Config => _config;

    /// <summary>Callback for log messages.</summary>
    public Action<string>? OnLog { get; set; }

    /// <summary>Callback for server notifications.</summary>
    public Action<JsonRpcNotification>? OnNotification { get; set; }

    public McpClient(McpServerConfig config)
    {
        _config = config;
    }

    /// <summary>
    /// Connect to the MCP server, perform handshake, and discover capabilities.
    /// </summary>
    public async Task ConnectAsync(CancellationToken ct = default)
    {
        // Create transport
        _transport = CreateTransport();

        // Wire up notification handler
        if (_transport is StdioTransport stdio)
        {
            stdio.OnNotification = n => OnNotification?.Invoke(n);
            stdio.OnStderr = line => OnLog?.Invoke($"[{_config.Name} stderr] {line}");
        }
        else if (_transport is SseTransport sse)
        {
            sse.OnNotification = n => OnNotification?.Invoke(n);
        }

        // Start transport
        OnLog?.Invoke($"Connecting to MCP server '{_config.Name}' via {_config.Transport}...");
        await _transport.ConnectAsync(ct);

        // Initialize handshake
        _serverInfo = await InitializeAsync(ct);
        OnLog?.Invoke($"Connected to {_serverInfo.ServerInfo.Name} v{_serverInfo.ServerInfo.Version} " +
            $"(protocol {_serverInfo.ProtocolVersion})");

        // Send initialized notification
        await _transport.SendNotificationAsync(new JsonRpcNotification
        {
            Method = "notifications/initialized"
        }, ct);

        // Pre-fetch tools if the server supports them
        if (_serverInfo.Capabilities.Tools != null)
        {
            _cachedTools = (await ListToolsAsync(ct)).ToList();
            OnLog?.Invoke($"Discovered {_cachedTools.Count} tools: " +
                string.Join(", ", _cachedTools.Select(t => t.Name)));
        }
    }

    /// <summary>
    /// Perform the initialize handshake.
    /// </summary>
    private async Task<InitializeResult> InitializeAsync(CancellationToken ct)
    {
        var initParams = new
        {
            protocolVersion = ProtocolVersion,
            capabilities = new { },
            clientInfo = new
            {
                name = ClientName,
                version = ClientVersion,
            }
        };

        var response = await SendAsync("initialize",
            JsonSerializer.SerializeToElement(initParams, JsonOpts.Default), ct);

        return JsonSerializer.Deserialize<InitializeResult>(response.GetRawText(), JsonOpts.Default)
            ?? throw new InvalidOperationException("Failed to deserialize initialize result.");
    }

    // ─── Tools ──────────────────────────────────────────────────

    /// <summary>List available tools from the server.</summary>
    public async Task<IReadOnlyList<McpToolDefinition>> ListToolsAsync(CancellationToken ct = default)
    {
        var allTools = new List<McpToolDefinition>();
        string? cursor = null;

        do
        {
            var requestParams = cursor != null
                ? JsonSerializer.SerializeToElement(new { cursor }, JsonOpts.Default)
                : (JsonElement?)null;

            var result = await SendAsync("tools/list", requestParams, ct);
            var page = JsonSerializer.Deserialize<ToolsListResult>(result.GetRawText(), JsonOpts.Default)
                ?? new ToolsListResult();

            allTools.AddRange(page.Tools);
            cursor = page.NextCursor;

        } while (cursor != null);

        _cachedTools = allTools;
        return allTools;
    }

    /// <summary>Get cached tools (from last ListToolsAsync or connect).</summary>
    public IReadOnlyList<McpToolDefinition> GetCachedTools() =>
        _cachedTools ?? (IReadOnlyList<McpToolDefinition>)Array.Empty<McpToolDefinition>();

    /// <summary>
    /// Call a tool on the MCP server.
    /// </summary>
    /// <param name="toolName">Name of the tool to call.</param>
    /// <param name="arguments">Tool arguments as a JSON object.</param>
    /// <returns>Tool call result with content blocks.</returns>
    public async Task<ToolCallResult> CallToolAsync(
        string toolName,
        Dictionary<string, object>? arguments = null,
        CancellationToken ct = default)
    {
        var callParams = new
        {
            name = toolName,
            arguments = arguments ?? new Dictionary<string, object>(),
        };

        using var timeoutCts = new CancellationTokenSource(TimeSpan.FromSeconds(_config.Timeout));
        using var linked = CancellationTokenSource.CreateLinkedTokenSource(ct, timeoutCts.Token);

        var result = await SendAsync("tools/call",
            JsonSerializer.SerializeToElement(callParams, JsonOpts.Default),
            linked.Token);

        return JsonSerializer.Deserialize<ToolCallResult>(result.GetRawText(), JsonOpts.Default)
            ?? new ToolCallResult { Content = { new McpContent { Text = result.GetRawText() } } };
    }

    /// <summary>
    /// Call a tool with raw JSON arguments string.
    /// </summary>
    public async Task<ToolCallResult> CallToolRawAsync(
        string toolName,
        string argumentsJson,
        CancellationToken ct = default)
    {
        var argsElement = string.IsNullOrEmpty(argumentsJson) || argumentsJson == "{}"
            ? JsonSerializer.SerializeToElement(new { })
            : JsonDocument.Parse(argumentsJson).RootElement;

        var callParams = JsonSerializer.SerializeToElement(new
        {
            name = toolName,
            arguments = argsElement,
        }, JsonOpts.Default);

        using var timeoutCts = new CancellationTokenSource(TimeSpan.FromSeconds(_config.Timeout));
        using var linked = CancellationTokenSource.CreateLinkedTokenSource(ct, timeoutCts.Token);

        var result = await SendAsync("tools/call", callParams, linked.Token);

        return JsonSerializer.Deserialize<ToolCallResult>(result.GetRawText(), JsonOpts.Default)
            ?? new ToolCallResult { Content = { new McpContent { Text = result.GetRawText() } } };
    }

    // ─── Resources ──────────────────────────────────────────────

    /// <summary>List available resources.</summary>
    public async Task<IReadOnlyList<McpResource>> ListResourcesAsync(CancellationToken ct = default)
    {
        var allResources = new List<McpResource>();
        string? cursor = null;

        do
        {
            var requestParams = cursor != null
                ? JsonSerializer.SerializeToElement(new { cursor }, JsonOpts.Default)
                : (JsonElement?)null;

            var result = await SendAsync("resources/list", requestParams, ct);
            var page = JsonSerializer.Deserialize<ResourcesListResult>(result.GetRawText(), JsonOpts.Default)
                ?? new ResourcesListResult();

            allResources.AddRange(page.Resources);
            cursor = page.NextCursor;

        } while (cursor != null);

        return allResources;
    }

    /// <summary>Read a resource by URI.</summary>
    public async Task<ResourceReadResult> ReadResourceAsync(string uri, CancellationToken ct = default)
    {
        var result = await SendAsync("resources/read",
            JsonSerializer.SerializeToElement(new { uri }, JsonOpts.Default), ct);

        return JsonSerializer.Deserialize<ResourceReadResult>(result.GetRawText(), JsonOpts.Default)
            ?? new ResourceReadResult();
    }

    // ─── Prompts ────────────────────────────────────────────────

    /// <summary>List available prompt templates.</summary>
    public async Task<IReadOnlyList<McpPrompt>> ListPromptsAsync(CancellationToken ct = default)
    {
        var allPrompts = new List<McpPrompt>();
        string? cursor = null;

        do
        {
            var requestParams = cursor != null
                ? JsonSerializer.SerializeToElement(new { cursor }, JsonOpts.Default)
                : (JsonElement?)null;

            var result = await SendAsync("prompts/list", requestParams, ct);
            var page = JsonSerializer.Deserialize<PromptsListResult>(result.GetRawText(), JsonOpts.Default)
                ?? new PromptsListResult();

            allPrompts.AddRange(page.Prompts);
            cursor = page.NextCursor;

        } while (cursor != null);

        return allPrompts;
    }

    /// <summary>Get a prompt template with arguments filled in.</summary>
    public async Task<PromptGetResult> GetPromptAsync(
        string name,
        Dictionary<string, string>? arguments = null,
        CancellationToken ct = default)
    {
        var result = await SendAsync("prompts/get",
            JsonSerializer.SerializeToElement(new { name, arguments }, JsonOpts.Default), ct);

        return JsonSerializer.Deserialize<PromptGetResult>(result.GetRawText(), JsonOpts.Default)
            ?? new PromptGetResult();
    }

    // ─── Lifecycle ──────────────────────────────────────────────

    /// <summary>Ping the server to check connectivity.</summary>
    public async Task<bool> PingAsync(CancellationToken ct = default)
    {
        try
        {
            await SendAsync("ping", null, ct);
            return true;
        }
        catch
        {
            return false;
        }
    }

    /// <summary>Gracefully disconnect from the server.</summary>
    public async Task DisconnectAsync()
    {
        if (_transport == null) return;

        OnLog?.Invoke($"Disconnecting from MCP server '{_config.Name}'...");
        await _transport.DisconnectAsync();
        _transport = null;
        _serverInfo = null;
        _cachedTools = null;
    }

    public void Dispose()
    {
        _transport?.Dispose();
        _transport = null;
    }

    // ─── Internals ──────────────────────────────────────────────

    private async Task<JsonElement> SendAsync(string method, JsonElement? @params, CancellationToken ct)
    {
        if (_transport == null)
            throw new InvalidOperationException("Not connected to MCP server.");

        var request = new JsonRpcRequest
        {
            Method = method,
            Params = @params,
        };

        var response = await _transport.SendRequestAsync(request, ct);

        if (response.Error != null)
        {
            throw new McpException(
                $"MCP server error ({response.Error.Code}): {response.Error.Message}",
                response.Error.Code,
                response.Error.Data);
        }

        return response.Result ?? JsonSerializer.SerializeToElement(new { });
    }

    private IMcpTransport CreateTransport()
    {
        return _config.Transport.ToLowerInvariant() switch
        {
            "stdio" => new StdioTransport(
                _config.Command ?? throw new ArgumentException("MCP stdio transport requires 'command'."),
                _config.Args?.ToArray(),
                _config.Env),

            "sse" or "http" => new SseTransport(
                _config.Url ?? throw new ArgumentException("MCP SSE transport requires 'url'."),
                _config.Headers),

            _ => throw new ArgumentException($"Unknown MCP transport type: {_config.Transport}")
        };
    }
}

/// <summary>Exception thrown when an MCP server returns a JSON-RPC error.</summary>
public class McpException : Exception
{
    public int ErrorCode { get; }
    public new JsonElement? Data { get; }

    public McpException(string message, int code, JsonElement? data = null) : base(message)
    {
        ErrorCode = code;
        Data = data;
    }
}
