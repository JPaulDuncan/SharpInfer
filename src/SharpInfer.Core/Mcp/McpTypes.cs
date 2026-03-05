using System.Text.Json;
using System.Text.Json.Serialization;

namespace SharpInfer.Core.Mcp;

// ─── JSON-RPC 2.0 ──────────────────────────────────────────────

/// <summary>JSON-RPC 2.0 request envelope.</summary>
public class JsonRpcRequest
{
    [JsonPropertyName("jsonrpc")] public string Jsonrpc { get; set; } = "2.0";
    [JsonPropertyName("id")] public int Id { get; set; }
    [JsonPropertyName("method")] public string Method { get; set; } = "";
    [JsonPropertyName("params")] public JsonElement? Params { get; set; }
}

/// <summary>JSON-RPC 2.0 response envelope.</summary>
public class JsonRpcResponse
{
    [JsonPropertyName("jsonrpc")] public string Jsonrpc { get; set; } = "2.0";
    [JsonPropertyName("id")] public int? Id { get; set; }
    [JsonPropertyName("result")] public JsonElement? Result { get; set; }
    [JsonPropertyName("error")] public JsonRpcError? Error { get; set; }
}

/// <summary>JSON-RPC 2.0 notification (no id).</summary>
public class JsonRpcNotification
{
    [JsonPropertyName("jsonrpc")] public string Jsonrpc { get; set; } = "2.0";
    [JsonPropertyName("method")] public string Method { get; set; } = "";
    [JsonPropertyName("params")] public JsonElement? Params { get; set; }
}

public class JsonRpcError
{
    [JsonPropertyName("code")] public int Code { get; set; }
    [JsonPropertyName("message")] public string Message { get; set; } = "";
    [JsonPropertyName("data")] public JsonElement? Data { get; set; }
}

// ─── MCP Protocol Types ─────────────────────────────────────────

/// <summary>MCP server capability flags returned during initialization.</summary>
public class ServerCapabilities
{
    [JsonPropertyName("tools")] public ToolsCapability? Tools { get; set; }
    [JsonPropertyName("resources")] public ResourcesCapability? Resources { get; set; }
    [JsonPropertyName("prompts")] public PromptsCapability? Prompts { get; set; }
    [JsonPropertyName("logging")] public JsonElement? Logging { get; set; }
}

public class ToolsCapability
{
    [JsonPropertyName("listChanged")] public bool ListChanged { get; set; }
}

public class ResourcesCapability
{
    [JsonPropertyName("subscribe")] public bool Subscribe { get; set; }
    [JsonPropertyName("listChanged")] public bool ListChanged { get; set; }
}

public class PromptsCapability
{
    [JsonPropertyName("listChanged")] public bool ListChanged { get; set; }
}

/// <summary>MCP server info returned at init.</summary>
public class ServerInfo
{
    [JsonPropertyName("name")] public string Name { get; set; } = "";
    [JsonPropertyName("version")] public string Version { get; set; } = "";
}

/// <summary>Result of the initialize handshake.</summary>
public class InitializeResult
{
    [JsonPropertyName("protocolVersion")] public string ProtocolVersion { get; set; } = "";
    [JsonPropertyName("capabilities")] public ServerCapabilities Capabilities { get; set; } = new();
    [JsonPropertyName("serverInfo")] public ServerInfo ServerInfo { get; set; } = new();
}

/// <summary>An MCP tool definition exposed by a server.</summary>
public class McpToolDefinition
{
    [JsonPropertyName("name")] public string Name { get; set; } = "";
    [JsonPropertyName("description")] public string Description { get; set; } = "";
    [JsonPropertyName("inputSchema")] public JsonElement? InputSchema { get; set; }
}

/// <summary>Result of tools/list.</summary>
public class ToolsListResult
{
    [JsonPropertyName("tools")] public List<McpToolDefinition> Tools { get; set; } = new();
    [JsonPropertyName("nextCursor")] public string? NextCursor { get; set; }
}

/// <summary>Content block in a tool call result.</summary>
public class McpContent
{
    [JsonPropertyName("type")] public string Type { get; set; } = "text";
    [JsonPropertyName("text")] public string? Text { get; set; }
    [JsonPropertyName("mimeType")] public string? MimeType { get; set; }
    [JsonPropertyName("data")] public string? Data { get; set; } // base64 for image/blob
}

/// <summary>Result of tools/call.</summary>
public class ToolCallResult
{
    [JsonPropertyName("content")] public List<McpContent> Content { get; set; } = new();
    [JsonPropertyName("isError")] public bool IsError { get; set; }
}

/// <summary>An MCP resource definition.</summary>
public class McpResource
{
    [JsonPropertyName("uri")] public string Uri { get; set; } = "";
    [JsonPropertyName("name")] public string Name { get; set; } = "";
    [JsonPropertyName("description")] public string? Description { get; set; }
    [JsonPropertyName("mimeType")] public string? MimeType { get; set; }
}

/// <summary>Result of resources/list.</summary>
public class ResourcesListResult
{
    [JsonPropertyName("resources")] public List<McpResource> Resources { get; set; } = new();
    [JsonPropertyName("nextCursor")] public string? NextCursor { get; set; }
}

/// <summary>Resource content returned from resources/read.</summary>
public class ResourceContent
{
    [JsonPropertyName("uri")] public string Uri { get; set; } = "";
    [JsonPropertyName("mimeType")] public string? MimeType { get; set; }
    [JsonPropertyName("text")] public string? Text { get; set; }
    [JsonPropertyName("blob")] public string? Blob { get; set; } // base64
}

/// <summary>Result of resources/read.</summary>
public class ResourceReadResult
{
    [JsonPropertyName("contents")] public List<ResourceContent> Contents { get; set; } = new();
}

/// <summary>An MCP prompt definition.</summary>
public class McpPrompt
{
    [JsonPropertyName("name")] public string Name { get; set; } = "";
    [JsonPropertyName("description")] public string? Description { get; set; }
    [JsonPropertyName("arguments")] public List<McpPromptArgument>? Arguments { get; set; }
}

public class McpPromptArgument
{
    [JsonPropertyName("name")] public string Name { get; set; } = "";
    [JsonPropertyName("description")] public string? Description { get; set; }
    [JsonPropertyName("required")] public bool Required { get; set; }
}

/// <summary>Result of prompts/list.</summary>
public class PromptsListResult
{
    [JsonPropertyName("prompts")] public List<McpPrompt> Prompts { get; set; } = new();
    [JsonPropertyName("nextCursor")] public string? NextCursor { get; set; }
}

/// <summary>A message in a prompt result.</summary>
public class PromptMessage
{
    [JsonPropertyName("role")] public string Role { get; set; } = "user";
    [JsonPropertyName("content")] public McpContent Content { get; set; } = new();
}

/// <summary>Result of prompts/get.</summary>
public class PromptGetResult
{
    [JsonPropertyName("description")] public string? Description { get; set; }
    [JsonPropertyName("messages")] public List<PromptMessage> Messages { get; set; } = new();
}

// ─── Transport Abstraction ──────────────────────────────────────

/// <summary>
/// Transport layer for MCP communication.
/// Implementations handle the wire protocol (stdio, HTTP/SSE, etc).
/// </summary>
public interface IMcpTransport : IDisposable
{
    /// <summary>Whether the transport connection is active.</summary>
    bool IsConnected { get; }

    /// <summary>Send a JSON-RPC request and wait for the response.</summary>
    Task<JsonRpcResponse> SendRequestAsync(JsonRpcRequest request, CancellationToken ct = default);

    /// <summary>Send a JSON-RPC notification (no response expected).</summary>
    Task SendNotificationAsync(JsonRpcNotification notification, CancellationToken ct = default);

    /// <summary>Start the transport (connect/spawn process).</summary>
    Task ConnectAsync(CancellationToken ct = default);

    /// <summary>Gracefully shut down the transport.</summary>
    Task DisconnectAsync();
}

// ─── Configuration ──────────────────────────────────────────────

/// <summary>Configuration for a single MCP server connection.</summary>
public class McpServerConfig
{
    /// <summary>Display name for this server.</summary>
    [JsonPropertyName("name")] public string Name { get; set; } = "";

    /// <summary>Transport type: "stdio" or "sse".</summary>
    [JsonPropertyName("transport")] public string Transport { get; set; } = "stdio";

    /// <summary>For stdio: command to execute (e.g., "npx", "python").</summary>
    [JsonPropertyName("command")] public string? Command { get; set; }

    /// <summary>For stdio: arguments to the command (e.g., ["-y", "@modelcontextprotocol/server-filesystem", "/path"]).</summary>
    [JsonPropertyName("args")] public List<string>? Args { get; set; }

    /// <summary>Environment variables to set for the server process.</summary>
    [JsonPropertyName("env")] public Dictionary<string, string>? Env { get; set; }

    /// <summary>For sse: URL of the SSE endpoint.</summary>
    [JsonPropertyName("url")] public string? Url { get; set; }

    /// <summary>For sse: optional headers (e.g., Authorization).</summary>
    [JsonPropertyName("headers")] public Dictionary<string, string>? Headers { get; set; }

    /// <summary>Whether to auto-connect at startup. Default true.</summary>
    [JsonPropertyName("enabled")] public bool Enabled { get; set; } = true;

    /// <summary>Timeout in seconds for tool calls. Default 30.</summary>
    [JsonPropertyName("timeout")] public int Timeout { get; set; } = 30;
}
