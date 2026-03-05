using System.Text.Json;
using SharpInfer.Core.Tools;

namespace SharpInfer.Core.Mcp;

/// <summary>
/// Bridges an MCP tool into the SharpInfer ITool interface.
///
/// Each MCP tool exposed by a server becomes an ITool that can be registered
/// with the ToolRegistry and used by the inference engine's tool-calling pipeline.
///
/// The tool name is prefixed with the server name to avoid collisions:
///   "serverName__toolName" (or just "toolName" if no prefix configured)
/// </summary>
public class McpToolBridge : ITool
{
    private readonly McpClient _client;
    private readonly McpToolDefinition _definition;
    private readonly string _prefix;

    public string Name { get; }
    public string Description => _definition.Description;

    public string ParameterSchema =>
        _definition.InputSchema?.GetRawText() ?? "{}";

    public McpToolBridge(McpClient client, McpToolDefinition definition, string serverPrefix = "")
    {
        _client = client;
        _definition = definition;
        _prefix = serverPrefix;

        // Prefix tool name with server name if multiple servers might have same tool names
        Name = string.IsNullOrEmpty(serverPrefix)
            ? definition.Name
            : $"{serverPrefix}__{definition.Name}";
    }

    /// <summary>The original MCP tool name (without server prefix).</summary>
    public string McpToolName => _definition.Name;

    /// <summary>The server this tool belongs to.</summary>
    public string ServerName => _client.Config.Name;

    public async Task<string> ExecuteAsync(string arguments)
    {
        try
        {
            var result = await _client.CallToolRawAsync(_definition.Name, arguments);

            if (result.IsError)
            {
                var errorText = string.Join("\n", result.Content
                    .Where(c => c.Text != null)
                    .Select(c => c.Text));
                return $"Error: {errorText}";
            }

            // Combine all text content blocks
            var parts = new List<string>();
            foreach (var content in result.Content)
            {
                switch (content.Type)
                {
                    case "text":
                        if (content.Text != null)
                            parts.Add(content.Text);
                        break;

                    case "image":
                        parts.Add($"[Image: {content.MimeType ?? "image/png"}, {(content.Data?.Length ?? 0)} bytes base64]");
                        break;

                    case "resource":
                        parts.Add($"[Resource embedded]");
                        if (content.Text != null) parts.Add(content.Text);
                        break;

                    default:
                        if (content.Text != null) parts.Add(content.Text);
                        break;
                }
            }

            return string.Join("\n", parts);
        }
        catch (McpException ex)
        {
            return $"MCP Error ({ex.ErrorCode}): {ex.Message}";
        }
        catch (Exception ex)
        {
            return $"Error calling MCP tool '{_definition.Name}': {ex.Message}";
        }
    }
}

/// <summary>
/// Manages multiple MCP server connections and exposes all their tools
/// through a unified interface.
///
/// Usage:
///   var manager = new McpManager();
///   await manager.ConnectAsync(config1);
///   await manager.ConnectAsync(config2);
///   manager.RegisterAllTools(toolRegistry); // all tools from all servers
/// </summary>
public class McpManager : IDisposable
{
    private readonly Dictionary<string, McpClient> _clients = new();
    private readonly List<McpToolBridge> _bridges = new();

    /// <summary>Callback for log messages.</summary>
    public Action<string>? OnLog { get; set; }

    /// <summary>All connected servers.</summary>
    public IReadOnlyDictionary<string, McpClient> Servers => _clients;

    /// <summary>All tool bridges across all servers.</summary>
    public IReadOnlyList<McpToolBridge> AllTools => _bridges;

    /// <summary>
    /// Connect to an MCP server and discover its tools.
    /// </summary>
    public async Task<McpClient> ConnectAsync(McpServerConfig config, CancellationToken ct = default)
    {
        if (_clients.ContainsKey(config.Name))
            throw new InvalidOperationException($"MCP server '{config.Name}' is already connected.");

        var client = new McpClient(config) { OnLog = OnLog };
        await client.ConnectAsync(ct);

        _clients[config.Name] = client;

        // Create tool bridges — use server name prefix when multiple servers connected
        bool usePrefix = _clients.Count > 1;
        string prefix = usePrefix ? config.Name : "";

        foreach (var tool in client.GetCachedTools())
        {
            _bridges.Add(new McpToolBridge(client, tool, prefix));
        }

        // If we now have multiple servers, re-prefix all existing bridges
        if (_clients.Count == 2)
        {
            RebuildBridgesWithPrefixes();
        }

        return client;
    }

    /// <summary>
    /// Connect to multiple MCP servers from a list of configs.
    /// Skips disabled servers. Logs errors but doesn't fail on individual server failures.
    /// </summary>
    public async Task ConnectAllAsync(IEnumerable<McpServerConfig> configs, CancellationToken ct = default)
    {
        foreach (var config in configs)
        {
            if (!config.Enabled)
            {
                OnLog?.Invoke($"MCP server '{config.Name}' is disabled, skipping.");
                continue;
            }

            try
            {
                await ConnectAsync(config, ct);
            }
            catch (Exception ex)
            {
                OnLog?.Invoke($"Failed to connect to MCP server '{config.Name}': {ex.Message}");
            }
        }
    }

    /// <summary>
    /// Register all MCP tools with a ToolRegistry.
    /// </summary>
    public void RegisterAllTools(ToolRegistry registry)
    {
        foreach (var bridge in _bridges)
        {
            registry.Register(bridge);
            OnLog?.Invoke($"Registered MCP tool: {bridge.Name} (from {bridge.ServerName})");
        }
    }

    /// <summary>
    /// Disconnect from a specific server and unregister its tools.
    /// </summary>
    public async Task DisconnectAsync(string serverName, ToolRegistry? registry = null)
    {
        if (!_clients.TryGetValue(serverName, out var client))
            return;

        // Remove bridges for this server
        var toRemove = _bridges.Where(b => b.ServerName == serverName).ToList();
        foreach (var bridge in toRemove)
        {
            _bridges.Remove(bridge);
            registry?.Unregister(bridge.Name);
        }

        await client.DisconnectAsync();
        _clients.Remove(serverName);
        client.Dispose();

        // If back to single server, remove prefixes
        if (_clients.Count == 1)
            RebuildBridgesWithPrefixes();
    }

    /// <summary>
    /// Disconnect from all servers.
    /// </summary>
    public async Task DisconnectAllAsync()
    {
        foreach (var (name, client) in _clients)
        {
            try { await client.DisconnectAsync(); }
            catch { }
            client.Dispose();
        }
        _clients.Clear();
        _bridges.Clear();
    }

    /// <summary>
    /// Get a tool bridge by name (supports both prefixed and unprefixed lookup).
    /// </summary>
    public McpToolBridge? FindTool(string name)
    {
        return _bridges.FirstOrDefault(b => b.Name == name)
            ?? _bridges.FirstOrDefault(b => b.McpToolName == name);
    }

    /// <summary>
    /// List all tools grouped by server.
    /// </summary>
    public Dictionary<string, List<McpToolBridge>> GetToolsByServer()
    {
        return _bridges
            .GroupBy(b => b.ServerName)
            .ToDictionary(g => g.Key, g => g.ToList());
    }

    /// <summary>
    /// Ping all connected servers to verify connectivity.
    /// </summary>
    public async Task<Dictionary<string, bool>> PingAllAsync(CancellationToken ct = default)
    {
        var results = new Dictionary<string, bool>();
        foreach (var (name, client) in _clients)
        {
            results[name] = await client.PingAsync(ct);
        }
        return results;
    }

    private void RebuildBridgesWithPrefixes()
    {
        _bridges.Clear();
        bool usePrefix = _clients.Count > 1;

        foreach (var (name, client) in _clients)
        {
            string prefix = usePrefix ? name : "";
            foreach (var tool in client.GetCachedTools())
            {
                _bridges.Add(new McpToolBridge(client, tool, prefix));
            }
        }
    }

    public void Dispose()
    {
        DisconnectAllAsync().GetAwaiter().GetResult();
    }
}
