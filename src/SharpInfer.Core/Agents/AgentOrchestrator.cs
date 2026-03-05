using SharpInfer.Core.Engine;
using SharpInfer.Core.Sampling;
using SharpInfer.Core.Tools;

namespace SharpInfer.Core.Agents;

/// <summary>
/// High-level orchestrator that manages agent creation, flow construction,
/// and workflow execution.
///
/// Provides a fluent API for building multi-agent systems:
///
///   var orch = new AgentOrchestrator(engine);
///
///   // Define agents
///   orch.DefineAgent("researcher", "Research Analyst",
///       "You are a research analyst. Find relevant information and cite sources.");
///   orch.DefineAgent("writer", "Technical Writer",
///       "You are a technical writer. Create clear, well-structured content.");
///   orch.DefineAgent("editor", "Editor",
///       "You are an editor. Improve clarity, fix errors, and polish the writing.");
///
///   // Build and run a chain
///   var result = await orch.Chain("researcher", "writer", "editor")
///       .RunAsync("Write a guide to transformer architectures");
///
///   // Or build a debate
///   var result = await orch.Debate("proponent", "critic")
///       .WithJudge(orch.GetAgent("judge"))
///       .WithRounds(3)
///       .RunAsync("Should we use microservices or a monolith?");
/// </summary>
public class AgentOrchestrator : IDisposable
{
    private readonly InferenceEngine _primaryEngine;
    private readonly Dictionary<string, Agent> _agents = new();
    private readonly Dictionary<string, InferenceEngine> _engines = new();
    private readonly ToolRegistry _globalTools;

    /// <summary>Callback for log messages.</summary>
    public Action<string>? OnLog { get; set; }

    /// <summary>Default generation config for new agents.</summary>
    public GenerationConfig DefaultGenerationConfig { get; set; } = new()
    {
        Temperature = 0.7f,
        TopP = 0.9f,
        TopK = 40,
        MaxTokens = 1024,
    };

    /// <summary>All defined agents.</summary>
    public IReadOnlyDictionary<string, Agent> Agents => _agents;

    public AgentOrchestrator(InferenceEngine primaryEngine, ToolRegistry? globalTools = null)
    {
        _primaryEngine = primaryEngine;
        _globalTools = globalTools ?? new ToolRegistry();
    }

    // ─── Agent Management ──────────────────────────────────────

    /// <summary>
    /// Define a new agent with a name, role, and system prompt.
    /// Uses the primary engine by default.
    /// </summary>
    public Agent DefineAgent(
        string name,
        string role,
        string systemPrompt,
        Action<Agent>? configure = null)
    {
        var agent = new Agent
        {
            Name = name,
            Role = role,
            SystemPrompt = systemPrompt,
            Engine = _primaryEngine,
            GenerationConfig = new GenerationConfig
            {
                Temperature = DefaultGenerationConfig.Temperature,
                TopP = DefaultGenerationConfig.TopP,
                TopK = DefaultGenerationConfig.TopK,
                MaxTokens = DefaultGenerationConfig.MaxTokens,
                RepetitionPenalty = DefaultGenerationConfig.RepetitionPenalty,
                Seed = DefaultGenerationConfig.Seed,
                StopTokenIds = DefaultGenerationConfig.StopTokenIds,
                StopStrings = DefaultGenerationConfig.StopStrings,
            },
        };

        configure?.Invoke(agent);
        _agents[name] = agent;

        OnLog?.Invoke($"Agent defined: {name} ({role})");
        return agent;
    }

    /// <summary>
    /// Define an agent from a config object.
    /// </summary>
    public Agent DefineAgent(AgentConfig config)
    {
        // Get or load engine
        InferenceEngine engine = _primaryEngine;
        if (!string.IsNullOrEmpty(config.ModelPath) && config.ModelPath != _primaryEngine.Config.ToString())
        {
            if (!_engines.TryGetValue(config.ModelPath, out var cached))
            {
                OnLog?.Invoke($"Loading model for agent '{config.Name}': {config.ModelPath}");
                cached = InferenceEngine.Load(config.ModelPath, new EngineOptions());
                _engines[config.ModelPath] = cached;
            }
            engine = cached;
        }

        var agent = new Agent
        {
            Name = config.Name,
            Role = config.Role,
            SystemPrompt = config.SystemPrompt,
            Engine = engine,
            GenerationConfig = new GenerationConfig
            {
                Temperature = config.Temperature,
                TopP = config.TopP,
                TopK = config.TopK,
                MaxTokens = config.MaxTokens,
            },
            MaxTurnsPerInvocation = config.MaxTurnsPerInvocation,
            CanHandoff = config.CanHandoff,
            AllowedHandoffTargets = new HashSet<string>(config.AllowedHandoffTargets),
        };

        // Assign tools from global registry
        if (config.Tools.Count > 0)
        {
            foreach (var toolName in config.Tools)
            {
                var tool = _globalTools.GetAll().FirstOrDefault(t => t.Name == toolName);
                if (tool != null)
                    agent.Tools.Register(tool);
                else
                    OnLog?.Invoke($"Warning: tool '{toolName}' not found for agent '{config.Name}'");
            }
        }

        _agents[config.Name] = agent;
        OnLog?.Invoke($"Agent defined: {config.Name} ({config.Role})");
        return agent;
    }

    /// <summary>Get an agent by name.</summary>
    public Agent GetAgent(string name)
    {
        if (_agents.TryGetValue(name, out var agent)) return agent;
        throw new KeyNotFoundException($"Agent '{name}' not defined. Available: {string.Join(", ", _agents.Keys)}");
    }

    /// <summary>Remove an agent.</summary>
    public bool RemoveAgent(string name) => _agents.Remove(name);

    /// <summary>
    /// Give an agent access to specific tools from the global registry.
    /// </summary>
    public AgentOrchestrator AssignTools(string agentName, params string[] toolNames)
    {
        var agent = GetAgent(agentName);
        foreach (var toolName in toolNames)
        {
            var tool = _globalTools.GetAll().FirstOrDefault(t => t.Name == toolName);
            if (tool != null)
                agent.Tools.Register(tool);
        }
        return this;
    }

    /// <summary>
    /// Give an agent access to ALL global tools.
    /// </summary>
    public AgentOrchestrator AssignAllTools(string agentName)
    {
        var agent = GetAgent(agentName);
        foreach (var tool in _globalTools.GetAll())
            agent.Tools.Register(tool);
        return this;
    }

    // ─── Flow Construction ─────────────────────────────────────

    /// <summary>
    /// Build a sequential chain flow from agent names.
    /// </summary>
    public ChainFlow Chain(params string[] agentNames)
    {
        var flow = new ChainFlow($"chain-{string.Join("-", agentNames)}")
        {
            OnLog = OnLog,
        };

        foreach (var name in agentNames)
            flow.Add(GetAgent(name));

        return flow;
    }

    /// <summary>
    /// Build a parallel flow from agent names, optionally with a synthesizer.
    /// </summary>
    public ParallelFlow Parallel(string? synthesizerName, params string[] agentNames)
    {
        var flow = new ParallelFlow($"parallel-{agentNames.Length}")
        {
            OnLog = OnLog,
        };

        foreach (var name in agentNames)
            flow.Add(GetAgent(name));

        if (synthesizerName != null)
            flow.WithSynthesizer(GetAgent(synthesizerName));

        return flow;
    }

    /// <summary>
    /// Build a debate flow from debater names.
    /// </summary>
    public DebateFlow Debate(params string[] debaterNames)
    {
        var flow = new DebateFlow($"debate-{debaterNames.Length}")
        {
            OnLog = OnLog,
        };

        foreach (var name in debaterNames)
            flow.AddDebater(GetAgent(name));

        return flow;
    }

    /// <summary>
    /// Build a router flow with a classifier agent and named routes.
    /// </summary>
    public RouterFlow Router(string routerAgentName, Dictionary<string, string> routes, string? fallbackAgent = null)
    {
        var flow = new RouterFlow($"router-{routerAgentName}")
        {
            OnLog = OnLog,
        };

        flow.WithRouter(GetAgent(routerAgentName));

        foreach (var (routeName, agentName) in routes)
            flow.AddRoute(routeName, GetAgent(agentName));

        if (fallbackAgent != null)
            flow.WithFallback(GetAgent(fallbackAgent));

        return flow;
    }

    /// <summary>
    /// Build a swarm-style handoff flow.
    /// </summary>
    public HandoffFlow Handoff(string entryAgentName, params string[] agentNames)
    {
        var flow = new HandoffFlow($"handoff-{entryAgentName}")
        {
            OnLog = OnLog,
        };

        // Add all named agents
        var allNames = new HashSet<string>(agentNames) { entryAgentName };
        foreach (var name in allNames)
            flow.AddAgent(GetAgent(name));

        flow.WithEntryAgent(entryAgentName);
        return flow;
    }

    /// <summary>
    /// Build a map-reduce flow.
    /// </summary>
    public MapReduceFlow MapReduce(string mapperName, string reducerName)
    {
        return new MapReduceFlow($"mapreduce-{mapperName}-{reducerName}")
        {
            OnLog = OnLog,
        }
        .WithMapper(GetAgent(mapperName))
        .WithReducer(GetAgent(reducerName));
    }

    // ─── Composite Patterns ────────────────────────────────────

    /// <summary>
    /// Build a "review loop" pattern: an author writes, a reviewer critiques,
    /// and the author revises. Repeats for N iterations.
    ///
    /// Under the hood this is a chain that alternates between author and reviewer.
    /// </summary>
    public ChainFlow ReviewLoop(string authorName, string reviewerName, int iterations = 2)
    {
        var flow = new ChainFlow($"review-loop-{iterations}")
        {
            OnLog = OnLog,
            PassFullHistory = true,
        };

        for (int i = 0; i < iterations; i++)
        {
            flow.Add(GetAgent(authorName));
            flow.Add(GetAgent(reviewerName));
        }

        // Final pass by author to incorporate last review
        flow.Add(GetAgent(authorName));

        return flow;
    }

    /// <summary>
    /// Build an "expert panel" pattern: multiple experts respond in parallel,
    /// then a moderator synthesizes.
    /// </summary>
    public ParallelFlow ExpertPanel(string moderatorName, params string[] expertNames)
    {
        return Parallel(moderatorName, expertNames);
    }

    /// <summary>
    /// Build a "red team / blue team" pattern: adversarial debate with a judge.
    /// Red team tries to find problems, blue team defends.
    /// </summary>
    public DebateFlow RedTeamBlueTeam(string redTeamName, string blueTeamName, string judgeName, int rounds = 3)
    {
        return Debate(redTeamName, blueTeamName)
            .WithJudge(GetAgent(judgeName))
            .WithRounds(rounds) as DebateFlow ?? throw new InvalidOperationException();
    }

    // ─── Lifecycle ─────────────────────────────────────────────

    /// <summary>Clear all agents' memories.</summary>
    public void ClearAllMemories()
    {
        foreach (var agent in _agents.Values)
            agent.ClearMemory();
    }

    /// <summary>List all defined agents with their roles.</summary>
    public Dictionary<string, string> ListAgents()
    {
        return _agents.ToDictionary(kv => kv.Key, kv => kv.Value.Role);
    }

    public void Dispose()
    {
        foreach (var engine in _engines.Values)
            engine.Dispose();
        _engines.Clear();
    }
}
