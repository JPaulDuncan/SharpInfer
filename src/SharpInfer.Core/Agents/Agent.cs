using SharpInfer.Core.Engine;
using SharpInfer.Core.Sampling;
using SharpInfer.Core.Tools;

namespace SharpInfer.Core.Agents;

/// <summary>
/// An autonomous agent with its own identity, instructions, tools, and memory.
///
/// Each agent wraps an InferenceEngine but has:
///   - A unique name and role description
///   - Its own system prompt / persona
///   - Its own tool subset (not all tools available to every agent)
///   - Its own conversation memory (can be isolated or shared)
///   - Custom generation parameters
///   - Optional pre/post processing hooks
///
/// Agents can participate in multi-agent flows via the orchestration primitives.
/// </summary>
public class Agent
{
    /// <summary>Unique name identifying this agent in the flow.</summary>
    public string Name { get; init; } = "";

    /// <summary>Human-readable role description (e.g., "Research Analyst", "Code Reviewer").</summary>
    public string Role { get; init; } = "";

    /// <summary>System prompt defining this agent's persona and instructions.</summary>
    public string SystemPrompt { get; init; } = "You are a helpful assistant.";

    /// <summary>The inference engine this agent uses.</summary>
    public InferenceEngine Engine { get; init; } = null!;

    /// <summary>Agent-specific generation config (temperature, etc.).</summary>
    public GenerationConfig GenerationConfig { get; set; } = new();

    /// <summary>Tools available to THIS agent (subset of all registered tools).</summary>
    public ToolRegistry Tools { get; init; } = new();

    /// <summary>This agent's private conversation history.</summary>
    public List<AgentMessage> Memory { get; } = new();

    /// <summary>Max turns this agent can take in a single invocation before yielding.</summary>
    public int MaxTurnsPerInvocation { get; set; } = 5;

    /// <summary>Whether this agent can hand off to other agents.</summary>
    public bool CanHandoff { get; set; } = true;

    /// <summary>Names of agents this agent is allowed to hand off to. Empty = any.</summary>
    public HashSet<string> AllowedHandoffTargets { get; init; } = new();

    /// <summary>
    /// Optional hook: transform the prompt before sending to the engine.
    /// Receives (agent, fullPrompt) and returns modified prompt.
    /// </summary>
    public Func<Agent, string, string>? PreProcess { get; set; }

    /// <summary>
    /// Optional hook: transform the response after generation.
    /// Receives (agent, rawResponse) and returns modified response.
    /// </summary>
    public Func<Agent, string, string>? PostProcess { get; set; }

    /// <summary>
    /// Optional hook: decide whether to hand off to another agent.
    /// Receives (agent, response, sharedState) and returns target agent name or null.
    /// </summary>
    public Func<Agent, string, SharedState, string?>? HandoffDecider { get; set; }

    /// <summary>
    /// Run this agent on an input message, producing a response.
    /// Handles tool calls internally, respecting MaxTurnsPerInvocation.
    /// </summary>
    public async Task<AgentMessage> RunAsync(
        AgentMessage input,
        SharedState? sharedState = null,
        CancellationToken ct = default)
    {
        // Add input to memory
        Memory.Add(input);

        // Build the full prompt from system + memory
        string prompt = BuildPrompt();

        // Pre-process hook
        if (PreProcess != null)
            prompt = PreProcess(this, prompt);

        // Generate response
        string response;
        int toolTurns = 0;

        if (Tools.HasTools)
        {
            // Tool-calling loop
            var responseBuilder = new System.Text.StringBuilder();
            var genConfig = new GenerationConfig
            {
                Temperature = GenerationConfig.Temperature,
                TopP = GenerationConfig.TopP,
                TopK = GenerationConfig.TopK,
                MaxTokens = GenerationConfig.MaxTokens,
                RepetitionPenalty = GenerationConfig.RepetitionPenalty,
                Seed = GenerationConfig.Seed,
                StopTokenIds = GenerationConfig.StopTokenIds,
                StopStrings = new List<string> { "<|end|>", "<|user|>" }
            };

            await foreach (var evt in Engine.GenerateWithToolsAsync(prompt, genConfig))
            {
                switch (evt.Type)
                {
                    case EventType.Token:
                        responseBuilder.Append(evt.Text);
                        break;
                    case EventType.ToolCall:
                    case EventType.ToolResult:
                        toolTurns++;
                        if (toolTurns > MaxTurnsPerInvocation) break;
                        break;
                }

                if (ct.IsCancellationRequested) break;
            }

            response = responseBuilder.ToString();
        }
        else
        {
            response = await Engine.GenerateCompleteAsync(prompt, GenerationConfig);
        }

        // Post-process hook
        if (PostProcess != null)
            response = PostProcess(this, response);

        // Check for handoff
        string? handoffTarget = null;
        if (CanHandoff)
        {
            handoffTarget = DetectHandoff(response, sharedState);
        }

        // Build output message
        var output = new AgentMessage
        {
            FromAgent = Name,
            Role = handoffTarget != null ? "handoff" : "assistant",
            Content = response,
            HandoffTarget = handoffTarget,
            IsTerminal = handoffTarget == null && !Tools.HasTools, // terminal if no handoff and no tools
        };

        // Store in memory
        Memory.Add(output);

        // Add to shared state
        sharedState?.AddMessage(output);

        return output;
    }

    /// <summary>
    /// Run this agent in streaming mode, yielding tokens as they're generated.
    /// </summary>
    public async IAsyncEnumerable<string> RunStreamingAsync(
        AgentMessage input,
        SharedState? sharedState = null,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken ct = default)
    {
        Memory.Add(input);
        string prompt = BuildPrompt();

        if (PreProcess != null)
            prompt = PreProcess(this, prompt);

        var responseBuilder = new System.Text.StringBuilder();

        await foreach (var token in Engine.GenerateAsync(prompt, GenerationConfig))
        {
            if (ct.IsCancellationRequested) break;

            responseBuilder.Append(token);
            yield return token;
        }

        string response = responseBuilder.ToString();
        if (PostProcess != null)
            response = PostProcess(this, response);

        var output = new AgentMessage
        {
            FromAgent = Name,
            Role = "assistant",
            Content = response,
        };
        Memory.Add(output);
        sharedState?.AddMessage(output);
    }

    /// <summary>Clear this agent's conversation memory.</summary>
    public void ClearMemory() => Memory.Clear();

    /// <summary>Inject a message into this agent's memory without running inference.</summary>
    public void InjectMemory(AgentMessage message) => Memory.Add(message);

    /// <summary>Inject context from shared state into this agent's view.</summary>
    public void InjectContext(SharedState state, string tag)
    {
        foreach (var msg in state.GetMessagesWithTag(tag))
        {
            if (!Memory.Contains(msg))
                Memory.Add(msg);
        }
    }

    private string BuildPrompt()
    {
        var sb = new System.Text.StringBuilder();

        // System prompt with tools
        string sys = SystemPrompt;
        if (Tools.HasTools)
            sys += "\n\n" + Tools.BuildToolPrompt();

        sb.AppendLine($"<|system|>\n{sys}\n<|end|>");

        // Conversation history
        foreach (var msg in Memory)
        {
            string role = msg.Role switch
            {
                "user" or "handoff" => "user",
                "assistant" => "assistant",
                "tool_result" => "user",
                "system" => "system",
                _ => "user"
            };

            string content = msg.Content;

            // Prefix messages from other agents so this agent knows who said what
            if (!string.IsNullOrEmpty(msg.FromAgent) && msg.FromAgent != Name && role != "system")
            {
                content = $"[{msg.FromAgent}]: {content}";
            }

            sb.AppendLine($"<|{role}|>\n{content}\n<|end|>");
        }

        sb.Append("<|assistant|>\n");
        return sb.ToString();
    }

    private string? DetectHandoff(string response, SharedState? sharedState)
    {
        // Check custom decider first
        if (HandoffDecider != null && sharedState != null)
        {
            return HandoffDecider(this, response, sharedState);
        }

        // Check for explicit handoff markers in response
        // Format: <handoff agent="agent_name">optional context</handoff>
        var match = System.Text.RegularExpressions.Regex.Match(
            response,
            @"<handoff\s+agent=""(\w+)""",
            System.Text.RegularExpressions.RegexOptions.IgnoreCase);

        if (match.Success)
        {
            string target = match.Groups[1].Value;
            if (AllowedHandoffTargets.Count == 0 || AllowedHandoffTargets.Contains(target))
                return target;
        }

        return null;
    }
}

/// <summary>
/// Configuration for defining an agent in a config file.
/// </summary>
public class AgentConfig
{
    public string Name { get; set; } = "";
    public string Role { get; set; } = "";
    public string SystemPrompt { get; set; } = "You are a helpful assistant.";
    public float Temperature { get; set; } = 0.7f;
    public float TopP { get; set; } = 0.9f;
    public int TopK { get; set; } = 40;
    public int MaxTokens { get; set; } = 512;
    public int MaxTurnsPerInvocation { get; set; } = 5;
    public bool CanHandoff { get; set; } = true;
    public List<string> AllowedHandoffTargets { get; set; } = new();
    public List<string> Tools { get; set; } = new(); // tool names this agent can use
    public string? ModelPath { get; set; } // null = use the primary model
}
