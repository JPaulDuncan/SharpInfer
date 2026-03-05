namespace SharpInfer.Core.Agents;

/// <summary>
/// Base class for agent flow patterns.
/// A flow defines how agents collaborate to produce a result.
/// </summary>
public abstract class AgentFlow
{
    /// <summary>Flow name for identification.</summary>
    public string Name { get; init; } = "";

    /// <summary>Shared state accessible to all agents in the flow.</summary>
    public SharedState State { get; } = new();

    /// <summary>Maximum total turns across all agents before termination.</summary>
    public int MaxTotalTurns { get; set; } = 50;

    /// <summary>Optional termination condition.</summary>
    public TerminationCheck? TerminationCondition { get; set; }

    /// <summary>Callback for observing flow progress.</summary>
    public Action<string>? OnLog { get; set; }

    /// <summary>Callback for each message produced during the flow.</summary>
    public Action<AgentMessage>? OnMessage { get; set; }

    /// <summary>Execute the flow with the given input.</summary>
    public abstract Task<FlowResult> RunAsync(string input, CancellationToken ct = default);

    protected void Log(string msg) => OnLog?.Invoke(msg);

    protected bool ShouldTerminate(List<AgentMessage> recent, int turns)
    {
        if (turns >= MaxTotalTurns) return true;
        return TerminationCondition?.Invoke(State, recent, turns) ?? false;
    }
}

// ═══════════════════════════════════════════════════════════════════
//  CHAIN FLOW — Sequential pipeline: A → B → C → ...
// ═══════════════════════════════════════════════════════════════════

/// <summary>
/// Agents execute sequentially. Each agent's output becomes the next agent's input.
/// Good for: research → draft → review → polish pipelines.
///
/// Example:
///   var chain = new ChainFlow("write-pipeline")
///       .Add(researchAgent)
///       .Add(writerAgent)
///       .Add(editorAgent);
///   var result = await chain.RunAsync("Write about quantum computing");
/// </summary>
public class ChainFlow : AgentFlow
{
    private readonly List<Agent> _agents = new();

    /// <summary>If true, each agent sees the full history. If false, only the previous agent's output.</summary>
    public bool PassFullHistory { get; set; } = false;

    public ChainFlow(string name = "chain") { Name = name; }

    public ChainFlow Add(Agent agent)
    {
        _agents.Add(agent);
        return this;
    }

    public override async Task<FlowResult> RunAsync(string input, CancellationToken ct = default)
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();
        int totalTurns = 0;

        var currentInput = new AgentMessage
        {
            FromAgent = "user",
            Role = "user",
            Content = input,
        };
        State.AddMessage(currentInput);

        AgentMessage? lastOutput = null;

        foreach (var agent in _agents)
        {
            if (ct.IsCancellationRequested) break;

            Log($"Chain: {agent.Name} processing...");

            // Optionally inject full history
            if (PassFullHistory)
            {
                foreach (var msg in State.MessageHistory)
                    agent.InjectMemory(msg);
            }

            lastOutput = await agent.RunAsync(currentInput, State, ct);
            OnMessage?.Invoke(lastOutput);
            totalTurns++;

            Log($"Chain: {agent.Name} → {Truncate(lastOutput.Content, 100)}");

            if (ShouldTerminate(State.GetRecentMessages(3), totalTurns))
            {
                Log("Chain: terminated early.");
                break;
            }

            // Next agent receives previous output as input
            currentInput = new AgentMessage
            {
                FromAgent = agent.Name,
                Role = "user",
                Content = lastOutput.Content,
            };
        }

        sw.Stop();
        return new FlowResult
        {
            FinalMessages = lastOutput != null ? new List<AgentMessage> { lastOutput } : new(),
            FullHistory = State.MessageHistory.ToList(),
            State = State,
            Duration = sw.Elapsed,
            TotalTurns = totalTurns,
            Success = true,
        };
    }

    private static string Truncate(string s, int max) =>
        s.Length <= max ? s : s[..max] + "...";
}

// ═══════════════════════════════════════════════════════════════════
//  PARALLEL FLOW — All agents process simultaneously, results merged
// ═══════════════════════════════════════════════════════════════════

/// <summary>
/// All agents process the same input in parallel. Results are collected and optionally
/// merged by a synthesizer agent.
///
/// Good for: getting multiple perspectives, parallel research, voting.
///
/// Example:
///   var parallel = new ParallelFlow("multi-perspective")
///       .Add(optimistAgent)
///       .Add(pessimistAgent)
///       .Add(realistAgent)
///       .WithSynthesizer(summaryAgent);
/// </summary>
public class ParallelFlow : AgentFlow
{
    private readonly List<Agent> _agents = new();
    private Agent? _synthesizer;

    /// <summary>Max degree of parallelism.</summary>
    public int MaxParallelism { get; set; } = 4;

    public ParallelFlow(string name = "parallel") { Name = name; }

    public ParallelFlow Add(Agent agent)
    {
        _agents.Add(agent);
        return this;
    }

    /// <summary>Optional agent that synthesizes all parallel results into one output.</summary>
    public ParallelFlow WithSynthesizer(Agent synthesizer)
    {
        _synthesizer = synthesizer;
        return this;
    }

    public override async Task<FlowResult> RunAsync(string input, CancellationToken ct = default)
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();

        var userMsg = new AgentMessage
        {
            FromAgent = "user",
            Role = "user",
            Content = input,
        };
        State.AddMessage(userMsg);

        Log($"Parallel: dispatching to {_agents.Count} agents...");

        // Run all agents in parallel
        var semaphore = new SemaphoreSlim(MaxParallelism);
        var tasks = _agents.Select(async agent =>
        {
            await semaphore.WaitAsync(ct);
            try
            {
                // Each agent gets its own copy of the input
                var agentInput = new AgentMessage
                {
                    FromAgent = "user",
                    Role = "user",
                    Content = input,
                };

                Log($"Parallel: {agent.Name} starting...");
                var result = await agent.RunAsync(agentInput, State, ct);
                OnMessage?.Invoke(result);
                Log($"Parallel: {agent.Name} done.");
                return result;
            }
            finally
            {
                semaphore.Release();
            }
        });

        var results = (await Task.WhenAll(tasks)).ToList();
        int totalTurns = results.Count;

        // If we have a synthesizer, feed it all results
        List<AgentMessage> finalMessages;
        if (_synthesizer != null)
        {
            Log("Parallel: synthesizing results...");

            // Build a synthesis prompt with all agent outputs
            var synthInput = new AgentMessage
            {
                FromAgent = "orchestrator",
                Role = "user",
                Content = BuildSynthesisPrompt(results),
            };

            var synthResult = await _synthesizer.RunAsync(synthInput, State, ct);
            OnMessage?.Invoke(synthResult);
            totalTurns++;

            finalMessages = new List<AgentMessage> { synthResult };
        }
        else
        {
            finalMessages = results;
        }

        sw.Stop();
        return new FlowResult
        {
            FinalMessages = finalMessages,
            FullHistory = State.MessageHistory.ToList(),
            State = State,
            Duration = sw.Elapsed,
            TotalTurns = totalTurns,
            Success = true,
        };
    }

    private string BuildSynthesisPrompt(List<AgentMessage> results)
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine("Multiple agents have provided their responses. Synthesize them into a single coherent answer.\n");

        foreach (var r in results)
        {
            sb.AppendLine($"--- {r.FromAgent} ---");
            sb.AppendLine(r.Content);
            sb.AppendLine();
        }

        sb.AppendLine("Please provide a unified synthesis of the above responses.");
        return sb.ToString();
    }
}

// ═══════════════════════════════════════════════════════════════════
//  ROUTER FLOW — Dynamically routes to the best agent per input
// ═══════════════════════════════════════════════════════════════════

/// <summary>
/// A routing agent examines the input and delegates to the most appropriate
/// specialist agent. Can also re-route based on intermediate results.
///
/// Good for: customer support triage, multi-domain Q&A, task classification.
///
/// Example:
///   var router = new RouterFlow("support-router")
///       .WithRouter(triageAgent)
///       .AddRoute("billing", billingAgent)
///       .AddRoute("technical", techAgent)
///       .AddRoute("general", generalAgent);
/// </summary>
public class RouterFlow : AgentFlow
{
    private Agent? _router;
    private readonly Dictionary<string, Agent> _routes = new();
    private Agent? _fallback;

    /// <summary>Custom routing function. If set, bypasses the router agent.</summary>
    public Func<string, SharedState, string>? RoutingFunction { get; set; }

    public RouterFlow(string name = "router") { Name = name; }

    /// <summary>Set the agent that decides routing.</summary>
    public RouterFlow WithRouter(Agent router)
    {
        _router = router;
        return this;
    }

    /// <summary>Add a named route to a specialist agent.</summary>
    public RouterFlow AddRoute(string routeName, Agent agent)
    {
        _routes[routeName] = agent;
        return this;
    }

    /// <summary>Fallback agent if no route matches.</summary>
    public RouterFlow WithFallback(Agent fallback)
    {
        _fallback = fallback;
        return this;
    }

    public override async Task<FlowResult> RunAsync(string input, CancellationToken ct = default)
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();
        int totalTurns = 0;

        var userMsg = new AgentMessage
        {
            FromAgent = "user",
            Role = "user",
            Content = input,
        };
        State.AddMessage(userMsg);

        // Determine route
        string routeName;

        if (RoutingFunction != null)
        {
            routeName = RoutingFunction(input, State);
            Log($"Router: custom function selected '{routeName}'");
        }
        else if (_router != null)
        {
            // Ask router agent to classify
            var routePrompt = new AgentMessage
            {
                FromAgent = "orchestrator",
                Role = "user",
                Content = BuildRoutingPrompt(input),
            };

            var routeResponse = await _router.RunAsync(routePrompt, State, ct);
            totalTurns++;

            routeName = ExtractRoute(routeResponse.Content);
            Log($"Router: agent selected '{routeName}'");
        }
        else
        {
            throw new InvalidOperationException("RouterFlow requires either a router agent or a routing function.");
        }

        // Execute the selected route
        Agent targetAgent;
        if (_routes.TryGetValue(routeName, out var routed))
        {
            targetAgent = routed;
        }
        else if (_fallback != null)
        {
            Log($"Router: route '{routeName}' not found, using fallback.");
            targetAgent = _fallback;
        }
        else
        {
            // Try fuzzy match
            var bestMatch = _routes.Keys
                .OrderByDescending(k => FuzzyScore(k, routeName))
                .FirstOrDefault();

            if (bestMatch != null && FuzzyScore(bestMatch, routeName) > 0.5)
            {
                Log($"Router: fuzzy matched '{routeName}' → '{bestMatch}'");
                targetAgent = _routes[bestMatch];
            }
            else
            {
                sw.Stop();
                return new FlowResult
                {
                    FinalMessages = new List<AgentMessage>
                    {
                        new() { FromAgent = "router", Content = $"No suitable agent found for route '{routeName}'." }
                    },
                    FullHistory = State.MessageHistory.ToList(),
                    State = State,
                    Duration = sw.Elapsed,
                    TotalTurns = totalTurns,
                    Success = false,
                    TerminationReason = $"No route matched: {routeName}"
                };
            }
        }

        Log($"Router: executing {targetAgent.Name}...");
        var result = await targetAgent.RunAsync(userMsg, State, ct);
        OnMessage?.Invoke(result);
        totalTurns++;

        sw.Stop();
        return new FlowResult
        {
            FinalMessages = new List<AgentMessage> { result },
            FullHistory = State.MessageHistory.ToList(),
            State = State,
            Duration = sw.Elapsed,
            TotalTurns = totalTurns,
            Success = true,
        };
    }

    private string BuildRoutingPrompt(string input)
    {
        var routes = string.Join(", ", _routes.Keys.Select(k =>
        {
            var agent = _routes[k];
            return $"\"{k}\" ({agent.Role})";
        }));

        return $"Classify the following user request into one of these categories: {routes}\n\n" +
               $"User request: {input}\n\n" +
               "Respond with ONLY the category name, nothing else.";
    }

    private string ExtractRoute(string response)
    {
        // Try to match against known routes
        string cleaned = response.Trim().Trim('"').ToLowerInvariant();

        foreach (var route in _routes.Keys)
        {
            if (cleaned.Contains(route.ToLowerInvariant()))
                return route;
        }

        return cleaned;
    }

    private static double FuzzyScore(string a, string b)
    {
        a = a.ToLowerInvariant();
        b = b.ToLowerInvariant();
        if (a == b) return 1.0;
        if (a.Contains(b) || b.Contains(a)) return 0.8;

        // Simple Jaccard on characters
        var setA = new HashSet<char>(a);
        var setB = new HashSet<char>(b);
        int intersection = setA.Intersect(setB).Count();
        int union = setA.Union(setB).Count();
        return union == 0 ? 0 : (double)intersection / union;
    }
}

// ═══════════════════════════════════════════════════════════════════
//  DEBATE FLOW — Agents argue, then a judge decides
// ═══════════════════════════════════════════════════════════════════

/// <summary>
/// Multiple agents debate a topic over several rounds, then a judge
/// agent evaluates and produces a final verdict.
///
/// Good for: complex reasoning, adversarial testing, red-team/blue-team.
///
/// Example:
///   var debate = new DebateFlow("code-review")
///       .AddDebater(proposerAgent)
///       .AddDebater(criticAgent)
///       .WithJudge(judgeAgent)
///       .WithRounds(3);
/// </summary>
public class DebateFlow : AgentFlow
{
    private readonly List<Agent> _debaters = new();
    private Agent? _judge;

    /// <summary>Number of debate rounds before judging.</summary>
    public int Rounds { get; set; } = 3;

    public DebateFlow(string name = "debate") { Name = name; }

    public DebateFlow AddDebater(Agent agent)
    {
        _debaters.Add(agent);
        return this;
    }

    public DebateFlow WithJudge(Agent judge)
    {
        _judge = judge;
        return this;
    }

    public DebateFlow WithRounds(int rounds)
    {
        Rounds = rounds;
        return this;
    }

    public override async Task<FlowResult> RunAsync(string input, CancellationToken ct = default)
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();
        int totalTurns = 0;

        var userMsg = new AgentMessage
        {
            FromAgent = "user",
            Role = "user",
            Content = input,
        };
        State.AddMessage(userMsg);

        // Debate rounds
        for (int round = 1; round <= Rounds; round++)
        {
            if (ct.IsCancellationRequested) break;
            Log($"Debate round {round}/{Rounds}:");

            foreach (var debater in _debaters)
            {
                // Build context: the original question + all previous debate messages
                var debateContext = new AgentMessage
                {
                    FromAgent = "moderator",
                    Role = "user",
                    Content = round == 1
                        ? $"Please provide your position on the following:\n\n{input}"
                        : BuildDebateRoundPrompt(debater.Name, round),
                };

                // Inject relevant history
                foreach (var msg in State.MessageHistory.Where(m => m.FromAgent != debater.Name))
                {
                    debater.InjectMemory(msg);
                }

                var response = await debater.RunAsync(debateContext, State, ct);
                response.Tags.Add($"round-{round}");
                OnMessage?.Invoke(response);
                totalTurns++;

                Log($"  {debater.Name}: {Truncate(response.Content, 100)}");
            }

            if (ShouldTerminate(State.GetRecentMessages(_debaters.Count), totalTurns))
            {
                Log("Debate: terminated early by condition.");
                break;
            }
        }

        // Judge phase
        AgentMessage finalMessage;
        if (_judge != null)
        {
            Log("Debate: judging...");

            var judgeInput = new AgentMessage
            {
                FromAgent = "moderator",
                Role = "user",
                Content = BuildJudgePrompt(input),
            };

            // Give judge the full debate history
            foreach (var msg in State.MessageHistory)
                _judge.InjectMemory(msg);

            finalMessage = await _judge.RunAsync(judgeInput, State, ct);
            finalMessage.Tags.Add("final");
            finalMessage.Tags.Add("verdict");
            OnMessage?.Invoke(finalMessage);
            totalTurns++;

            Log($"Judge verdict: {Truncate(finalMessage.Content, 100)}");
        }
        else
        {
            // No judge — return all final-round messages
            var lastRound = State.MessageHistory
                .Where(m => m.Tags.Contains($"round-{Rounds}"))
                .ToList();

            finalMessage = new AgentMessage
            {
                FromAgent = "debate",
                Role = "assistant",
                Content = string.Join("\n\n---\n\n",
                    lastRound.Select(m => $"**{m.FromAgent}**: {m.Content}")),
                Tags = { "final" },
            };
        }

        sw.Stop();
        return new FlowResult
        {
            FinalMessages = new List<AgentMessage> { finalMessage },
            FullHistory = State.MessageHistory.ToList(),
            State = State,
            Duration = sw.Elapsed,
            TotalTurns = totalTurns,
            Success = true,
        };
    }

    private string BuildDebateRoundPrompt(string currentAgent, int round)
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine($"This is round {round} of the debate. Here are the other participants' latest arguments:\n");

        var prevRound = State.MessageHistory
            .Where(m => m.Tags.Contains($"round-{round - 1}") && m.FromAgent != currentAgent);

        foreach (var msg in prevRound)
        {
            sb.AppendLine($"[{msg.FromAgent}]: {msg.Content}\n");
        }

        sb.AppendLine("Please respond to their arguments and strengthen your position.");
        return sb.ToString();
    }

    private string BuildJudgePrompt(string originalQuestion)
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine($"Original question: {originalQuestion}\n");
        sb.AppendLine("The following debate has taken place:\n");

        foreach (var msg in State.MessageHistory.Where(m => m.FromAgent != "user" && m.FromAgent != "moderator"))
        {
            var roundTag = msg.Tags.FirstOrDefault(t => t.StartsWith("round-"));
            string round = roundTag != null ? $" ({roundTag})" : "";
            sb.AppendLine($"[{msg.FromAgent}{round}]: {msg.Content}\n");
        }

        sb.AppendLine("Based on the debate above, provide your verdict. Consider the strength of arguments, " +
            "evidence provided, and logical consistency. Identify the strongest position and explain your reasoning.");
        return sb.ToString();
    }

    private static string Truncate(string s, int max) =>
        s.Length <= max ? s : s[..max] + "...";
}

// ═══════════════════════════════════════════════════════════════════
//  HANDOFF FLOW — Swarm-style dynamic agent handoffs
// ═══════════════════════════════════════════════════════════════════

/// <summary>
/// Agents dynamically hand off to each other based on context.
/// Similar to OpenAI Swarm — the active agent decides who should handle
/// the next turn based on the conversation state.
///
/// Good for: complex multi-step workflows, customer service escalation.
///
/// Example:
///   var swarm = new HandoffFlow("support-swarm")
///       .AddAgent(triageAgent)
///       .AddAgent(billingAgent)
///       .AddAgent(techAgent)
///       .WithEntryAgent("triage");
/// </summary>
public class HandoffFlow : AgentFlow
{
    private readonly Dictionary<string, Agent> _agents = new();
    private string _entryAgent = "";

    public HandoffFlow(string name = "handoff") { Name = name; }

    public HandoffFlow AddAgent(Agent agent)
    {
        _agents[agent.Name] = agent;
        return this;
    }

    public HandoffFlow WithEntryAgent(string agentName)
    {
        _entryAgent = agentName;
        return this;
    }

    public override async Task<FlowResult> RunAsync(string input, CancellationToken ct = default)
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();
        int totalTurns = 0;

        if (!_agents.ContainsKey(_entryAgent))
            throw new InvalidOperationException($"Entry agent '{_entryAgent}' not found.");

        var currentMessage = new AgentMessage
        {
            FromAgent = "user",
            Role = "user",
            Content = input,
        };
        State.AddMessage(currentMessage);

        string currentAgentName = _entryAgent;
        AgentMessage? lastOutput = null;

        while (!ct.IsCancellationRequested)
        {
            if (!_agents.TryGetValue(currentAgentName, out var agent))
            {
                Log($"Handoff: agent '{currentAgentName}' not found, ending flow.");
                break;
            }

            Log($"Handoff: {agent.Name} active...");

            lastOutput = await agent.RunAsync(currentMessage, State, ct);
            OnMessage?.Invoke(lastOutput);
            totalTurns++;

            Log($"Handoff: {agent.Name} → {Truncate(lastOutput.Content, 100)}");

            if (ShouldTerminate(State.GetRecentMessages(3), totalTurns))
            {
                Log("Handoff: terminated by condition.");
                break;
            }

            // Check for handoff
            if (lastOutput.HandoffTarget != null)
            {
                Log($"Handoff: {agent.Name} → {lastOutput.HandoffTarget}");
                currentAgentName = lastOutput.HandoffTarget;

                // Pass the response as context to the next agent
                currentMessage = new AgentMessage
                {
                    FromAgent = agent.Name,
                    Role = "handoff",
                    Content = lastOutput.Content,
                    HandoffTarget = lastOutput.HandoffTarget,
                };
            }
            else
            {
                // No handoff — agent is done
                break;
            }
        }

        sw.Stop();
        return new FlowResult
        {
            FinalMessages = lastOutput != null ? new List<AgentMessage> { lastOutput } : new(),
            FullHistory = State.MessageHistory.ToList(),
            State = State,
            Duration = sw.Elapsed,
            TotalTurns = totalTurns,
            Success = true,
        };
    }

    private static string Truncate(string s, int max) =>
        s.Length <= max ? s : s[..max] + "...";
}

// ═══════════════════════════════════════════════════════════════════
//  MAP-REDUCE FLOW — Parallel chunk processing + merge
// ═══════════════════════════════════════════════════════════════════

/// <summary>
/// Splits input into chunks, processes each with a mapper agent in parallel,
/// then reduces results with a reducer agent.
///
/// Good for: summarizing large documents, processing datasets, parallel analysis.
///
/// Example:
///   var mapReduce = new MapReduceFlow("summarize-doc")
///       .WithMapper(chunkSummarizerAgent)
///       .WithReducer(finalSummarizerAgent)
///       .WithChunkStrategy(text => ChunkByParagraphs(text, 1000));
/// </summary>
public class MapReduceFlow : AgentFlow
{
    private Agent? _mapper;
    private Agent? _reducer;

    /// <summary>Function that splits input text into chunks.</summary>
    public Func<string, List<string>>? ChunkStrategy { get; set; }

    /// <summary>Max parallel mapper invocations.</summary>
    public int MaxParallelism { get; set; } = 4;

    public MapReduceFlow(string name = "map-reduce") { Name = name; }

    public MapReduceFlow WithMapper(Agent mapper)
    {
        _mapper = mapper;
        return this;
    }

    public MapReduceFlow WithReducer(Agent reducer)
    {
        _reducer = reducer;
        return this;
    }

    public MapReduceFlow WithChunkStrategy(Func<string, List<string>> strategy)
    {
        ChunkStrategy = strategy;
        return this;
    }

    public override async Task<FlowResult> RunAsync(string input, CancellationToken ct = default)
    {
        if (_mapper == null) throw new InvalidOperationException("MapReduceFlow requires a mapper agent.");
        if (_reducer == null) throw new InvalidOperationException("MapReduceFlow requires a reducer agent.");

        var sw = System.Diagnostics.Stopwatch.StartNew();
        int totalTurns = 0;

        // Split into chunks
        var chunks = ChunkStrategy != null
            ? ChunkStrategy(input)
            : DefaultChunk(input, 2000);

        Log($"MapReduce: split into {chunks.Count} chunks, mapping...");
        State.Set("chunk_count", chunks.Count);

        // Map phase — parallel
        var semaphore = new SemaphoreSlim(MaxParallelism);
        var mapTasks = chunks.Select(async (chunk, index) =>
        {
            await semaphore.WaitAsync(ct);
            try
            {
                // Create a fresh mapper instance with same config
                var mapInput = new AgentMessage
                {
                    FromAgent = "orchestrator",
                    Role = "user",
                    Content = $"[Chunk {index + 1}/{chunks.Count}]\n\n{chunk}",
                    Tags = { "map", $"chunk-{index}" },
                };

                var result = await _mapper.RunAsync(mapInput, State, ct);
                result.Tags.Add("map-result");
                result.Tags.Add($"chunk-{index}");
                OnMessage?.Invoke(result);
                return result;
            }
            finally
            {
                semaphore.Release();
            }
        });

        var mapResults = await Task.WhenAll(mapTasks);
        totalTurns += mapResults.Length;
        Log($"MapReduce: mapped {mapResults.Length} chunks, reducing...");

        // Reduce phase
        var reduceInput = new AgentMessage
        {
            FromAgent = "orchestrator",
            Role = "user",
            Content = BuildReducePrompt(mapResults),
            Tags = { "reduce" },
        };

        var reduceResult = await _reducer.RunAsync(reduceInput, State, ct);
        reduceResult.Tags.Add("final");
        OnMessage?.Invoke(reduceResult);
        totalTurns++;

        sw.Stop();
        return new FlowResult
        {
            FinalMessages = new List<AgentMessage> { reduceResult },
            FullHistory = State.MessageHistory.ToList(),
            State = State,
            Duration = sw.Elapsed,
            TotalTurns = totalTurns,
            Success = true,
        };
    }

    private string BuildReducePrompt(AgentMessage[] mapResults)
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine("The following are results from processing individual chunks. " +
            "Synthesize them into a single coherent result:\n");

        for (int i = 0; i < mapResults.Length; i++)
        {
            sb.AppendLine($"--- Chunk {i + 1} result ---");
            sb.AppendLine(mapResults[i].Content);
            sb.AppendLine();
        }

        return sb.ToString();
    }

    private static List<string> DefaultChunk(string text, int chunkSize)
    {
        var chunks = new List<string>();
        var paragraphs = text.Split("\n\n", StringSplitOptions.RemoveEmptyEntries);

        var current = new System.Text.StringBuilder();
        foreach (var para in paragraphs)
        {
            if (current.Length + para.Length > chunkSize && current.Length > 0)
            {
                chunks.Add(current.ToString());
                current.Clear();
            }
            current.AppendLine(para);
            current.AppendLine();
        }

        if (current.Length > 0)
            chunks.Add(current.ToString());

        return chunks;
    }
}
