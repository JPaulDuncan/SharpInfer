using System.Text.Json;
using System.Text.RegularExpressions;

namespace SharpInfer.Core.Tools;

/// <summary>
/// Interface for tools that the model can invoke during generation.
/// Implement this to add new capabilities (web search, code execution, file I/O, etc.)
/// </summary>
public interface ITool
{
    /// <summary>Unique name the model uses to invoke this tool.</summary>
    string Name { get; }

    /// <summary>Description shown to the model so it knows when to use this tool.</summary>
    string Description { get; }

    /// <summary>JSON schema describing the tool's parameters.</summary>
    string ParameterSchema { get; }

    /// <summary>Execute the tool with the given arguments and return the result as text.</summary>
    Task<string> ExecuteAsync(string arguments);
}

/// <summary>Parsed tool call from model output.</summary>
public class ToolCall
{
    public required string ToolName { get; init; }
    public required string Arguments { get; init; }
}

/// <summary>
/// Registry of available tools. Handles tool registration, discovery,
/// parsing tool calls from model output, and execution.
///
/// Tool call format (customizable):
///   <tool_call>{"name": "tool_name", "arguments": {...}}</tool_call>
/// </summary>
public class ToolRegistry
{
    private readonly Dictionary<string, ITool> _tools = new();

    // Regex pattern for extracting tool calls from model output
    // Supports both XML-style and JSON-style tool call formats
    private static readonly Regex ToolCallPattern = new(
        @"<tool_call>\s*(\{.*?\})\s*</tool_call>",
        RegexOptions.Singleline | RegexOptions.Compiled);

    /// <summary>Register a tool.</summary>
    public void Register(ITool tool) => _tools[tool.Name] = tool;

    /// <summary>Remove a tool by name.</summary>
    public bool Unregister(string name) => _tools.Remove(name);

    /// <summary>Get all registered tools.</summary>
    public IReadOnlyCollection<ITool> GetAll() => _tools.Values;

    /// <summary>Check if any tools are registered.</summary>
    public bool HasTools => _tools.Count > 0;

    /// <summary>
    /// Build the system prompt section describing available tools.
    /// Inject this into the model's system prompt to enable tool use.
    /// </summary>
    public string BuildToolPrompt()
    {
        if (!HasTools) return "";

        var sb = new System.Text.StringBuilder();
        sb.AppendLine("You have access to the following tools:");
        sb.AppendLine();

        foreach (var tool in _tools.Values)
        {
            sb.AppendLine($"### {tool.Name}");
            sb.AppendLine(tool.Description);
            sb.AppendLine($"Parameters: {tool.ParameterSchema}");
            sb.AppendLine();
        }

        sb.AppendLine("To use a tool, output:");
        sb.AppendLine("<tool_call>{\"name\": \"tool_name\", \"arguments\": {...}}</tool_call>");

        return sb.ToString();
    }

    /// <summary>
    /// Parse a tool call from model output. Returns null if no tool call found.
    /// </summary>
    public ToolCall? ParseToolCall(string modelOutput)
    {
        var match = ToolCallPattern.Match(modelOutput);
        if (!match.Success) return null;

        try
        {
            var json = JsonDocument.Parse(match.Groups[1].Value).RootElement;
            return new ToolCall
            {
                ToolName = json.GetProperty("name").GetString()!,
                Arguments = json.TryGetProperty("arguments", out var args)
                    ? args.GetRawText()
                    : "{}"
            };
        }
        catch (JsonException)
        {
            return null; // Malformed JSON, not a real tool call
        }
    }

    /// <summary>Execute a parsed tool call.</summary>
    public async Task<string> ExecuteAsync(ToolCall call)
    {
        if (!_tools.TryGetValue(call.ToolName, out var tool))
            return $"Error: Unknown tool '{call.ToolName}'. Available tools: {string.Join(", ", _tools.Keys)}";

        try
        {
            return await tool.ExecuteAsync(call.Arguments);
        }
        catch (Exception ex)
        {
            return $"Error executing tool '{call.ToolName}': {ex.Message}";
        }
    }
}
