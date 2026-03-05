using System.Text.Json;
using System.Text.Json.Serialization;

namespace SharpInfer.Core.Agents;

/// <summary>
/// A message passed between agents in a multi-agent workflow.
/// Carries content, metadata, routing info, and optional structured data.
/// </summary>
public class AgentMessage
{
    /// <summary>Unique message ID.</summary>
    public string Id { get; init; } = Guid.NewGuid().ToString("N")[..12];

    /// <summary>Agent that produced this message.</summary>
    public string FromAgent { get; init; } = "";

    /// <summary>Target agent (empty = broadcast to flow).</summary>
    public string ToAgent { get; init; } = "";

    /// <summary>Message role: "user", "assistant", "system", "tool_result", "handoff".</summary>
    public string Role { get; init; } = "assistant";

    /// <summary>The text content of the message.</summary>
    public string Content { get; set; } = "";

    /// <summary>Optional structured data payload (JSON-serializable).</summary>
    public Dictionary<string, JsonElement>? Data { get; set; }

    /// <summary>When this message was created.</summary>
    public DateTimeOffset Timestamp { get; init; } = DateTimeOffset.UtcNow;

    /// <summary>Optional token/cost tracking.</summary>
    public UsageInfo? Usage { get; set; }

    /// <summary>Whether this message signals the end of a turn for its agent.</summary>
    public bool IsTerminal { get; set; }

    /// <summary>If Role is "handoff", which agent should take over.</summary>
    public string? HandoffTarget { get; set; }

    /// <summary>Tags for filtering/routing (e.g., "summary", "critique", "final").</summary>
    public HashSet<string> Tags { get; init; } = new();

    public override string ToString() =>
        $"[{FromAgent}->{(string.IsNullOrEmpty(ToAgent) ? "*" : ToAgent)}] ({Role}) {Truncate(Content, 80)}";

    private static string Truncate(string s, int max) =>
        s.Length <= max ? s : s[..max] + "...";
}

/// <summary>Token usage for a single generation.</summary>
public class UsageInfo
{
    public int PromptTokens { get; set; }
    public int CompletionTokens { get; set; }
    public int TotalTokens => PromptTokens + CompletionTokens;
}

/// <summary>
/// Shared state accessible to all agents in a flow.
/// Acts as a blackboard for inter-agent coordination.
/// </summary>
public class SharedState
{
    private readonly Dictionary<string, object> _store = new();
    private readonly object _lock = new();

    /// <summary>Full ordered message history across all agents.</summary>
    public List<AgentMessage> MessageHistory { get; } = new();

    /// <summary>Set a value in shared state.</summary>
    public void Set<T>(string key, T value) where T : notnull
    {
        lock (_lock) { _store[key] = value; }
    }

    /// <summary>Get a value from shared state.</summary>
    public T? Get<T>(string key)
    {
        lock (_lock)
        {
            return _store.TryGetValue(key, out var val) && val is T typed ? typed : default;
        }
    }

    /// <summary>Check if a key exists.</summary>
    public bool Has(string key)
    {
        lock (_lock) { return _store.ContainsKey(key); }
    }

    /// <summary>Remove a key.</summary>
    public bool Remove(string key)
    {
        lock (_lock) { return _store.Remove(key); }
    }

    /// <summary>Get all keys.</summary>
    public IReadOnlyCollection<string> Keys
    {
        get { lock (_lock) { return _store.Keys.ToList(); } }
    }

    /// <summary>Append a message to the shared history.</summary>
    public void AddMessage(AgentMessage message)
    {
        lock (_lock) { MessageHistory.Add(message); }
    }

    /// <summary>Get messages from a specific agent.</summary>
    public List<AgentMessage> GetMessagesFrom(string agentName)
    {
        lock (_lock)
        {
            return MessageHistory.Where(m => m.FromAgent == agentName).ToList();
        }
    }

    /// <summary>Get messages with a specific tag.</summary>
    public List<AgentMessage> GetMessagesWithTag(string tag)
    {
        lock (_lock)
        {
            return MessageHistory.Where(m => m.Tags.Contains(tag)).ToList();
        }
    }

    /// <summary>Get the last N messages.</summary>
    public List<AgentMessage> GetRecentMessages(int count)
    {
        lock (_lock)
        {
            return MessageHistory.Skip(Math.Max(0, MessageHistory.Count - count)).ToList();
        }
    }
}

/// <summary>
/// Result of an entire multi-agent flow execution.
/// </summary>
public class FlowResult
{
    /// <summary>The final output message(s) from the flow.</summary>
    public List<AgentMessage> FinalMessages { get; init; } = new();

    /// <summary>Combined final text output.</summary>
    public string FinalOutput => string.Join("\n\n", FinalMessages.Select(m => m.Content));

    /// <summary>Complete message history across all agents.</summary>
    public List<AgentMessage> FullHistory { get; init; } = new();

    /// <summary>Shared state at the end of the flow.</summary>
    public SharedState State { get; init; } = new();

    /// <summary>Total execution time.</summary>
    public TimeSpan Duration { get; init; }

    /// <summary>How many agent turns were used.</summary>
    public int TotalTurns { get; init; }

    /// <summary>Whether the flow completed successfully or was terminated early.</summary>
    public bool Success { get; init; } = true;

    /// <summary>Reason if terminated early.</summary>
    public string? TerminationReason { get; init; }
}

/// <summary>
/// Termination condition for a flow.
/// Return true to stop the flow.
/// </summary>
public delegate bool TerminationCheck(SharedState state, List<AgentMessage> recentMessages, int turnCount);

/// <summary>Common termination conditions.</summary>
public static class TerminationConditions
{
    /// <summary>Stop after N total turns across all agents.</summary>
    public static TerminationCheck MaxTurns(int max) =>
        (_, _, turns) => turns >= max;

    /// <summary>Stop when any message contains a specific keyword.</summary>
    public static TerminationCheck ContainsKeyword(string keyword) =>
        (_, msgs, _) => msgs.Any(m => m.Content.Contains(keyword, StringComparison.OrdinalIgnoreCase));

    /// <summary>Stop when a specific key is set in shared state.</summary>
    public static TerminationCheck StateKeySet(string key) =>
        (state, _, _) => state.Has(key);

    /// <summary>Stop when the last message has the "final" tag.</summary>
    public static TerminationCheck FinalTagged() =>
        (_, msgs, _) => msgs.LastOrDefault()?.Tags.Contains("final") ?? false;

    /// <summary>Stop when any agent marks its message as terminal.</summary>
    public static TerminationCheck AnyTerminal() =>
        (_, msgs, _) => msgs.Any(m => m.IsTerminal);

    /// <summary>Combine multiple conditions with OR logic.</summary>
    public static TerminationCheck Any(params TerminationCheck[] conditions) =>
        (state, msgs, turns) => conditions.Any(c => c(state, msgs, turns));

    /// <summary>Combine multiple conditions with AND logic.</summary>
    public static TerminationCheck All(params TerminationCheck[] conditions) =>
        (state, msgs, turns) => conditions.All(c => c(state, msgs, turns));
}
