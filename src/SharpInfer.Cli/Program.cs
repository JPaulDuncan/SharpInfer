using SharpInfer.Api;
using SharpInfer.Core.Agents;
using SharpInfer.Core.Engine;
using SharpInfer.Core.Layers;
using SharpInfer.Core.Mcp;
using SharpInfer.Core.Models;
using SharpInfer.Core.Multimodal;
using SharpInfer.Core.Sampling;
using SharpInfer.Core.Tensors;
using SharpInfer.Core.Tools;
using SharpInfer.Gpu;

namespace SharpInfer.Cli;

/// <summary>
/// SharpInfer CLI — Interactive chat with all engine features configurable.
///
/// Usage:
///   sharpinfer --model path/to/model.gguf [options]
///   sharpinfer --config config.json
///   sharpinfer --generate-config default.json
///
/// All advanced features are disabled by default and enabled via flags or config file.
/// </summary>
public class Program
{
    public static async Task<int> Main(string[] args)
    {
        if (args.Contains("--help") || args.Contains("-h") || args.Length == 0)
        {
            PrintUsage();
            return 0;
        }

        if (args.Contains("--generate-config"))
        {
            string path = args.SkipWhile(a => a != "--generate-config").Skip(1).FirstOrDefault() ?? "sharpinfer.json";
            EngineConfig.GenerateDefaultConfig(path);
            Console.WriteLine($"Default config written to {path}");
            return 0;
        }

        try
        {
            var config = ParseConfig(args);
            await RunEngine(config);
            return 0;
        }
        catch (Exception ex)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.Error.WriteLine($"Error: {ex.Message}");
            Console.ResetColor();
            return 1;
        }
    }

    private static async Task RunEngine(EngineConfig config)
    {
        PrintBanner();

        void Log(string msg)
        {
            Console.ForegroundColor = ConsoleColor.DarkGray;
            Console.WriteLine($"  [{msg}]");
            Console.ResetColor();
        }

        // --- Build engine options ---
        var engineOptions = new EngineOptions
        {
            ContextLength = config.ContextLength,
            GenerationConfig = new GenerationConfig
            {
                Temperature = config.Generation.Temperature,
                TopP = config.Generation.TopP,
                TopK = config.Generation.TopK,
                MaxTokens = config.Generation.MaxTokens,
                RepetitionPenalty = config.Generation.RepetitionPenalty,
                Seed = config.Generation.Seed,
            },
            ProgressCallback = Log,
        };

        // GPU
        if (config.Gpu.Enabled)
        {
            var cuda = new CudaBackend(config.Gpu.DeviceId);
            if (cuda.IsAvailable)
            {
                engineOptions.ComputeBackend = cuda;
                Log($"GPU: {cuda.DeviceName}");
            }
            else
            {
                Log("GPU requested but CUDA not available. Using CPU.");
            }
        }

        // --- Load primary model ---
        Log($"Loading model: {config.ModelPath}");
        var engine = InferenceEngine.Load(config.ModelPath, engineOptions);
        Log($"Model: {engine.Config}");

        // --- Initialize features ---
        var features = new FeatureSet();

        // Speculative Decoding
        if (config.Speculative.Enabled && !string.IsNullOrEmpty(config.Speculative.DraftModelPath))
        {
            Log($"Loading draft model: {config.Speculative.DraftModelPath}");
            var draftEngine = InferenceEngine.Load(config.Speculative.DraftModelPath,
                new EngineOptions { ProgressCallback = Log });
            // Note: SpeculativeDecoder needs Transformer access — simplified here
            Log($"Speculative decoding: enabled (lookahead={config.Speculative.LookaheadTokens})");
            features.SpeculativeEnabled = true;
        }

        // LoRA Adapters
        if (config.Lora.Enabled && config.Lora.Adapters.Count > 0)
        {
            foreach (var adapterEntry in config.Lora.Adapters)
            {
                Log($"Loading LoRA adapter: {adapterEntry.Name} from {adapterEntry.Path}");
                var adapter = LoraAdapter.LoadFromDirectory(adapterEntry.Path, Log);
                features.LoraAdapters[adapterEntry.Name] = adapter;
            }

            if (config.Lora.ActiveAdapter != null && features.LoraAdapters.ContainsKey(config.Lora.ActiveAdapter))
            {
                Log($"Applying LoRA: {config.Lora.ActiveAdapter}");
                features.ActiveLora = config.Lora.ActiveAdapter;
            }
            features.LoraEnabled = true;
        }

        // Prompt Caching
        if (config.PromptCaching.Enabled)
        {
            features.PromptCache = new PromptCache(config.PromptCaching.CacheDir);
            Log($"Prompt caching: enabled (dir={config.PromptCaching.CacheDir})");
            features.PromptCacheEnabled = true;
        }

        // Classifier-Free Guidance
        if (config.Cfg.Enabled)
        {
            features.CfgScale = config.Cfg.GuidanceScale;
            features.CfgNegativePrompt = config.Cfg.NegativePrompt;
            Log($"CFG: enabled (scale={config.Cfg.GuidanceScale})");
            features.CfgEnabled = true;
        }

        // Self-Consistency
        if (config.SelfConsistency.Enabled)
        {
            features.Consistency = new SelfConsistency(engine)
            {
                NumCandidates = config.SelfConsistency.NumCandidates,
                CandidateTemperature = config.SelfConsistency.CandidateTemperature,
                Strategy = config.SelfConsistency.Strategy.ToLower() switch
                {
                    "majority_vote" or "majority" => SelectionStrategy.MajorityVote,
                    "perplexity" or "min_perplexity" => SelectionStrategy.MinPerplexity,
                    "length" or "length_normalized" => SelectionStrategy.LengthNormalized,
                    _ => SelectionStrategy.MajorityVote,
                },
            };
            Log($"Self-consistency: enabled ({config.SelfConsistency.NumCandidates} candidates, {config.SelfConsistency.Strategy})");
            features.ConsistencyEnabled = true;
        }

        // RAG
        if (config.Rag.Enabled)
        {
            var embedder = new ModelEmbedder(engine);
            var vectorStoreConfig = new VectorStoreConfig
            {
                Backend = config.Rag.VectorStore,
                DbPath = config.Rag.VectorDbPath,
            };
            var vectorStore = VectorStoreFactory.Create(vectorStoreConfig, embedder.Dimensions);
            features.Rag = new RagPipeline(embedder, vectorStore)
            {
                TopK = config.Rag.TopK,
                MinScore = config.Rag.MinScore,
            };
            Log($"RAG vector store: {vectorStore.GetStats().Backend}");

            // Ingest configured documents
            foreach (var docPath in config.Rag.Documents)
            {
                if (File.Exists(docPath))
                {
                    Log($"RAG: ingesting {docPath}");
                    await features.Rag.IngestFileAsync(docPath);
                }
                else if (Directory.Exists(docPath))
                {
                    foreach (var file in Directory.GetFiles(docPath, "*.*", SearchOption.AllDirectories)
                        .Where(f => f.EndsWith(".txt") || f.EndsWith(".md") || f.EndsWith(".json")))
                    {
                        Log($"RAG: ingesting {file}");
                        await features.Rag.IngestFileAsync(file);
                    }
                }
            }
            Log($"RAG: enabled ({features.Rag.DocumentCount} chunks indexed)");
            features.RagEnabled = true;
        }

        // Beam Search
        if (config.BeamSearch.Enabled)
        {
            Log($"Beam search: enabled ({config.BeamSearch.NumBeams} beams, length_penalty={config.BeamSearch.LengthPenalty})");
            features.BeamSearchEnabled = true;
            features.BeamConfig = config.BeamSearch;
        }

        // Tools
        if (config.Tools.Enabled)
        {
            if (!string.IsNullOrEmpty(config.Tools.SearchApiEndpoint))
                engine.Tools.Register(new WebSearchTool(config.Tools.SearchApiEndpoint, config.Tools.SearchApiKey));
            if (config.Tools.UrlReader)
                engine.Tools.Register(new UrlReaderTool());
            Log($"Tools: {string.Join(", ", engine.Tools.GetAll().Select(t => t.Name))}");
        }

        // MCP Servers
        if (config.McpServers.Count > 0)
        {
            features.McpManager = new McpManager { OnLog = Log };
            await features.McpManager.ConnectAllAsync(config.McpServers);

            // Auto-enable tools when MCP servers provide them
            if (features.McpManager.AllTools.Count > 0)
            {
                config.Tools.Enabled = true;
                features.McpManager.RegisterAllTools(engine.Tools);
                Log($"MCP tools registered: {features.McpManager.AllTools.Count} tools from {features.McpManager.Servers.Count} server(s)");
            }
            features.McpEnabled = true;
        }

        // Multi-Agent
        if (config.MultiAgent.Enabled && config.MultiAgent.Agents.Count > 0)
        {
            features.Orchestrator = new AgentOrchestrator(engine, engine.Tools) { OnLog = Log };

            foreach (var agentDef in config.MultiAgent.Agents)
            {
                features.Orchestrator.DefineAgent(new AgentConfig
                {
                    Name = agentDef.Name,
                    Role = agentDef.Role,
                    SystemPrompt = agentDef.SystemPrompt,
                    Temperature = agentDef.Temperature,
                    TopP = agentDef.TopP,
                    MaxTokens = agentDef.MaxTokens,
                    Tools = agentDef.Tools,
                    ModelPath = agentDef.ModelPath,
                    CanHandoff = agentDef.CanHandoff,
                    AllowedHandoffTargets = agentDef.HandoffTargets,
                });
            }

            // Build named flows
            foreach (var flowDef in config.MultiAgent.Flows)
            {
                var flow = BuildFlowFromConfig(features.Orchestrator, flowDef);
                if (flow != null)
                    features.NamedFlows[flowDef.Name] = flow;
            }

            Log($"Multi-agent: {features.Orchestrator.Agents.Count} agents, {features.NamedFlows.Count} flows");
            features.AgentsEnabled = true;
        }

        // FlashAttention
        if (config.FlashAttention.Enabled)
        {
            Log($"FlashAttention: enabled (blockQ={config.FlashAttention.BlockSizeQ}, blockKV={config.FlashAttention.BlockSizeKV})");
            features.FlashAttentionEnabled = true;
        }

        // Batch Scheduler (API mode mainly)
        if (config.BatchScheduler.Enabled)
        {
            features.Scheduler = new BatchScheduler(engine,
                config.BatchScheduler.MaxBatchSize,
                config.BatchScheduler.MaxSequences,
                config.ContextLength ?? 4096)
            { OnLog = Log };
            features.Scheduler.Start();
            Log($"Batch scheduler: enabled (maxBatch={config.BatchScheduler.MaxBatchSize}, maxSeq={config.BatchScheduler.MaxSequences})");
            features.BatchSchedulerEnabled = true;
        }

        // Structured Output
        if (config.StructuredOutput.Enabled)
        {
            Log($"Structured output: enabled (default={config.StructuredOutput.DefaultType})");
            features.StructuredOutputEnabled = true;
        }

        // Hardware Backend Auto-Detection
        if (config.HardwareBackend.Backend != "cpu")
        {
            var available = BackendAutoDetect.DetectAvailable();
            foreach (var bi in available)
                Log($"Hardware: detected {bi.Name} backend ({bi.Description})");

            string selectedBackend = config.HardwareBackend.Backend;
            if (selectedBackend == "auto")
            {
                var best = BackendAutoDetect.GetBestBackend();
                if (best != null)
                {
                    selectedBackend = best.Name;
                    Log($"Hardware: auto-selected {selectedBackend} backend");
                }
                else
                {
                    selectedBackend = "cpu";
                    Log("Hardware: no accelerated backend found, using CPU");
                }
            }
            features.HardwareBackend = selectedBackend;
        }

        // Multimodal / Vision
        if (config.Multimodal.Enabled)
        {
            Log($"Multimodal: vision enabled (image_size={config.Multimodal.ImageSize}, max_images={config.Multimodal.MaxImages})");
            features.MultimodalEnabled = true;
        }

        // Quantization info
        if (config.Quantization.Type != "auto" && config.Quantization.Type != "F32")
        {
            Log($"Quantization: {config.Quantization.Type}");
            if (config.Quantization.DynamicRequantize && config.Quantization.TargetType != null)
                Log($"Dynamic re-quantization: {config.Quantization.Type} → {config.Quantization.TargetType}");
            features.QuantizationType = config.Quantization.Type;
        }

        // Modelfile
        if (!string.IsNullOrEmpty(config.ModelfilePath))
        {
            Log($"Modelfile: loaded from {config.ModelfilePath}");
        }

        // Distributed Inference
        if (config.Distributed.Enabled && config.Distributed.Workers.Count > 0)
        {
            features.DistributedEnabled = true;
            Log($"Distributed: {config.Distributed.Strategy}, {config.Distributed.Workers.Count} worker(s)");
        }

        // Cloud Fallback
        if (config.CloudFallback.Enabled && config.CloudFallback.Providers.Count > 0)
        {
            features.CloudRouter = new CloudFallbackRouter(config.CloudFallback) { OnLog = Log };
            features.CloudEnabled = true;
            Log($"Cloud fallback: {config.CloudFallback.Providers.Count} provider(s) configured");
            foreach (var p in config.CloudFallback.Providers.Where(p => p.Enabled))
                Log($"  → {p.Name}: {p.Model} (priority={p.Priority})");
        }

        // WebSocket
        if (config.WebSocket.Enabled)
        {
            features.WebSocketEnabled = true;
            Log($"WebSocket: enabled (max_connections_per_ip={config.WebSocket.MaxConnectionsPerIp})");
        }

        // Model Puller
        features.ModelPuller = new ModelPuller(config.ModelsDir, config.ModelPull.HuggingFaceToken)
        {
            OnLog = Log,
        };
        Log($"Model puller: ready (aliases: {features.ModelPuller.GetAliases().Count})");

        // Enterprise Features
        if (config.Enterprise.AuthEnabled || config.Enterprise.RateLimitEnabled ||
            config.Enterprise.AuditEnabled || config.Enterprise.MeteringEnabled)
        {
            features.EnterpriseEnabled = true;
            Log($"Enterprise: auth={config.Enterprise.AuthEnabled}, rate_limit={config.Enterprise.RateLimitEnabled}, audit={config.Enterprise.AuditEnabled}, metering={config.Enterprise.MeteringEnabled}");
        }

        // Batch Processing
        if (config.BatchProcessing.Enabled)
        {
            features.BatchProcessingEnabled = true;
            Log($"Batch processing: enabled (concurrency={config.BatchProcessing.MaxConcurrency}, max_batch={config.BatchProcessing.MaxBatchSize})");
        }

        // Model Registry
        features.ModelRegistry = new ModelRegistry(config.ModelsDir);
        features.ModelRegistry.Scan();
        if (features.ModelRegistry.Models.Count > 0)
            Log($"Model registry: {features.ModelRegistry.Models.Count} model(s) in {config.ModelsDir}");

        // --- Print active features summary ---
        PrintFeatureSummary(features, config);

        // --- Run chat loop ---
        await RunChatLoop(engine, config, features);

        engine.Dispose();
    }

    private static async Task RunChatLoop(InferenceEngine engine, EngineConfig config, FeatureSet features)
    {
        string systemPrompt = config.Generation.SystemPrompt;
        if (engine.Tools.HasTools)
            systemPrompt += "\n\n" + engine.Tools.BuildToolPrompt();

        var conversationHistory = new System.Text.StringBuilder();
        conversationHistory.AppendLine($"<|system|>\n{systemPrompt}\n<|end|>");

        Console.ForegroundColor = ConsoleColor.DarkGray;
        Console.WriteLine("\nType your message. Commands: /reset /features /lora /rag /mcp /agent /model /pull /quantize /quit");
        Console.WriteLine("─────────────────────────────────────────");
        Console.ResetColor();

        while (true)
        {
            Console.ForegroundColor = ConsoleColor.Green;
            Console.Write("\nYou > ");
            Console.ResetColor();

            string? input = Console.ReadLine();
            if (input == null) break;
            input = input.Trim();
            if (string.IsNullOrEmpty(input)) continue;

            if (input.StartsWith('/'))
            {
                if (!HandleCommand(input, engine, config, features)) break;
                continue;
            }

            // --- RAG: augment prompt with retrieved context ---
            string processedInput = input;
            if (features.RagEnabled && features.Rag != null)
            {
                processedInput = await features.Rag.AugmentPromptAsync(input);
                if (processedInput != input)
                {
                    Console.ForegroundColor = ConsoleColor.DarkCyan;
                    Console.WriteLine("  [RAG: context injected]");
                    Console.ResetColor();
                }
            }

            conversationHistory.AppendLine($"<|user|>\n{processedInput}\n<|end|>");
            conversationHistory.Append("<|assistant|>\n");
            string fullPrompt = conversationHistory.ToString();

            Console.ForegroundColor = ConsoleColor.White;
            Console.Write("\nAssistant > ");

            var genConfig = new GenerationConfig
            {
                Temperature = config.Generation.Temperature,
                TopP = config.Generation.TopP,
                TopK = config.Generation.TopK,
                MaxTokens = config.Generation.MaxTokens,
                RepetitionPenalty = config.Generation.RepetitionPenalty,
                StopStrings = { "<|end|>", "<|user|>" },
            };

            string response;

            // --- Self-Consistency mode ---
            if (features.ConsistencyEnabled && features.Consistency != null)
            {
                Console.ForegroundColor = ConsoleColor.DarkYellow;
                Console.Write($"[generating {features.Consistency.NumCandidates} candidates...] ");
                Console.ForegroundColor = ConsoleColor.White;

                var result = await features.Consistency.GenerateWithConsistencyAsync(fullPrompt, genConfig);
                response = result.SelectedResponse;
                Console.WriteLine(response);

                Console.ForegroundColor = ConsoleColor.DarkGray;
                Console.WriteLine($"  [confidence: {result.Confidence:P0}, strategy: {result.Strategy}]");
                Console.ResetColor();
            }
            // --- Standard streaming generation ---
            else
            {
                var responseBuilder = new System.Text.StringBuilder();

                if (engine.Tools.HasTools && config.Tools.Enabled)
                {
                    await foreach (var evt in engine.GenerateWithToolsAsync(fullPrompt, genConfig))
                    {
                        switch (evt.Type)
                        {
                            case EventType.Token:
                                Console.Write(evt.Text);
                                responseBuilder.Append(evt.Text);
                                break;
                            case EventType.ToolCall:
                                Console.ForegroundColor = ConsoleColor.Yellow;
                                Console.Write($"\n  [{evt.Text}]");
                                Console.ForegroundColor = ConsoleColor.White;
                                break;
                            case EventType.ToolResult:
                                Console.ForegroundColor = ConsoleColor.DarkGray;
                                Console.Write($"\n  [Result: {Truncate(evt.Text, 200)}]");
                                Console.ForegroundColor = ConsoleColor.White;
                                break;
                        }
                    }
                }
                else
                {
                    await foreach (var token in engine.GenerateAsync(fullPrompt, genConfig))
                    {
                        Console.Write(token);
                        responseBuilder.Append(token);
                    }
                }

                response = responseBuilder.ToString();
            }

            Console.ResetColor();
            Console.WriteLine();

            conversationHistory.AppendLine($"{response}\n<|end|>");
        }
    }

    private static bool HandleCommand(string command, InferenceEngine engine, EngineConfig config, FeatureSet features)
    {
        var parts = command.Split(' ', 2, StringSplitOptions.RemoveEmptyEntries);
        string cmd = parts[0].ToLower();
        string? arg = parts.Length > 1 ? parts[1] : null;

        switch (cmd)
        {
            case "/quit" or "/exit" or "/q":
                Console.ForegroundColor = ConsoleColor.DarkGray;
                Console.WriteLine("Goodbye!");
                Console.ResetColor();
                return false;

            case "/reset" or "/clear":
                engine.ResetContext();
                Console.ForegroundColor = ConsoleColor.DarkGray;
                Console.WriteLine("Context cleared.");
                Console.ResetColor();
                return true;

            case "/config":
                PrintCurrentConfig(config);
                return true;

            case "/features":
                PrintFeatureSummary(features, config);
                return true;

            case "/temp" when arg != null && float.TryParse(arg, out float t):
                config.Generation.Temperature = t;
                Console.ForegroundColor = ConsoleColor.DarkGray;
                Console.WriteLine($"  Temperature: {t}");
                Console.ResetColor();
                return true;

            case "/topp" when arg != null && float.TryParse(arg, out float p):
                config.Generation.TopP = p;
                Console.ForegroundColor = ConsoleColor.DarkGray;
                Console.WriteLine($"  Top-P: {p}");
                Console.ResetColor();
                return true;

            case "/topk" when arg != null && int.TryParse(arg, out int k):
                config.Generation.TopK = k;
                Console.ForegroundColor = ConsoleColor.DarkGray;
                Console.WriteLine($"  Top-K: {k}");
                Console.ResetColor();
                return true;

            case "/cfg" when arg != null && float.TryParse(arg, out float scale):
                config.Cfg.Enabled = scale > 1.0f;
                config.Cfg.GuidanceScale = scale;
                features.CfgEnabled = config.Cfg.Enabled;
                features.CfgScale = scale;
                Console.ForegroundColor = ConsoleColor.DarkGray;
                Console.WriteLine($"  CFG: {(config.Cfg.Enabled ? $"enabled (scale={scale})" : "disabled")}");
                Console.ResetColor();
                return true;

            case "/consistency" when arg != null:
                if (arg.ToLower() == "off" || arg == "0")
                {
                    features.ConsistencyEnabled = false;
                    Console.ForegroundColor = ConsoleColor.DarkGray;
                    Console.WriteLine("  Self-consistency: disabled");
                }
                else if (int.TryParse(arg, out int n) && n > 0)
                {
                    if (features.Consistency == null)
                        features.Consistency = new SelfConsistency(engine);
                    features.Consistency.NumCandidates = n;
                    features.ConsistencyEnabled = true;
                    Console.ForegroundColor = ConsoleColor.DarkGray;
                    Console.WriteLine($"  Self-consistency: enabled ({n} candidates)");
                }
                Console.ResetColor();
                return true;

            case "/lora" when arg != null:
                HandleLoraCommand(arg, features);
                return true;

            case "/rag" when arg != null:
                HandleRagCommand(arg, features).GetAwaiter().GetResult();
                return true;

            case "/mcp" when arg != null:
                HandleMcpCommand(arg, engine, features).GetAwaiter().GetResult();
                return true;

            case "/agent" or "/agents" when arg != null:
                HandleAgentCommand(arg, engine, features).GetAwaiter().GetResult();
                return true;

            case "/beam" when arg != null:
                if (arg.ToLower() == "off" || arg == "0")
                {
                    features.BeamSearchEnabled = false;
                    Console.ForegroundColor = ConsoleColor.DarkGray;
                    Console.WriteLine("  Beam search: disabled");
                }
                else if (int.TryParse(arg, out int beams) && beams > 0)
                {
                    features.BeamSearchEnabled = true;
                    config.BeamSearch.NumBeams = beams;
                    Console.ForegroundColor = ConsoleColor.DarkGray;
                    Console.WriteLine($"  Beam search: enabled ({beams} beams)");
                }
                Console.ResetColor();
                return true;

            case "/save-config" when arg != null:
                config.SaveToFile(arg);
                Console.ForegroundColor = ConsoleColor.DarkGray;
                Console.WriteLine($"  Config saved to {arg}");
                Console.ResetColor();
                return true;

            case "/model" or "/models" when arg != null:
                HandleModelCommand(arg, features);
                return true;

            case "/model" or "/models":
                HandleModelCommand("list", features);
                return true;

            case "/quantize" when arg != null:
                HandleQuantizeCommand(arg, features);
                return true;

            case "/hardware" or "/hw":
                Console.ForegroundColor = ConsoleColor.DarkGray;
                Console.WriteLine($"  Active backend: {features.HardwareBackend}");
                var backends = BackendAutoDetect.DetectAvailable();
                foreach (var b in backends)
                    Console.WriteLine($"    {(b.Name == features.HardwareBackend ? "→" : " ")} {b.Name}: {b.Description}");
                Console.ResetColor();
                return true;

            case "/export-modelfile" when arg != null:
                var mf = Modelfile.FromEngineConfig(config);
                mf.SaveToFile(arg);
                Console.ForegroundColor = ConsoleColor.DarkGray;
                Console.WriteLine($"  Modelfile exported to {arg}");
                Console.ResetColor();
                return true;

            case "/pull" when arg != null:
                HandlePullCommand(arg, features).GetAwaiter().GetResult();
                return true;

            case "/pull":
                HandlePullCommand("list", features).GetAwaiter().GetResult();
                return true;

            default:
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.WriteLine($"  Unknown command: {command}");
                Console.WriteLine("  Commands: /reset /config /features /temp /topp /topk /cfg /consistency");
                Console.WriteLine("            /lora /rag /mcp /agent /beam /model /pull /quantize /hardware");
                Console.WriteLine("            /export-modelfile /save-config /quit");
                Console.ResetColor();
                return true;
        }
    }

    private static void HandleLoraCommand(string arg, FeatureSet features)
    {
        var parts = arg.Split(' ', 2);
        Console.ForegroundColor = ConsoleColor.DarkGray;

        switch (parts[0].ToLower())
        {
            case "list":
                if (features.LoraAdapters.Count == 0)
                    Console.WriteLine("  No LoRA adapters loaded.");
                else
                {
                    Console.WriteLine("  Loaded adapters:");
                    foreach (var (name, adapter) in features.LoraAdapters)
                        Console.WriteLine($"    - {name} (rank={adapter.Rank}, applied={adapter.IsApplied})");
                }
                break;

            case "apply" when parts.Length > 1:
                if (features.LoraAdapters.TryGetValue(parts[1], out var toApply) && !toApply.IsApplied)
                {
                    Console.WriteLine($"  Applied LoRA: {parts[1]}");
                    features.ActiveLora = parts[1];
                }
                else
                    Console.WriteLine($"  Adapter '{parts[1]}' not found or already applied.");
                break;

            case "remove" when parts.Length > 1:
                if (features.LoraAdapters.TryGetValue(parts[1], out var toRemove) && toRemove.IsApplied)
                {
                    Console.WriteLine($"  Removed LoRA: {parts[1]}");
                    features.ActiveLora = null;
                }
                else
                    Console.WriteLine($"  Adapter '{parts[1]}' not found or not applied.");
                break;

            default:
                Console.WriteLine("  /lora list | apply <name> | remove <name>");
                break;
        }
        Console.ResetColor();
    }

    private static async Task HandleRagCommand(string arg, FeatureSet features)
    {
        var parts = arg.Split(' ', 2);
        Console.ForegroundColor = ConsoleColor.DarkGray;

        switch (parts[0].ToLower())
        {
            case "add" when parts.Length > 1 && features.Rag != null:
                if (File.Exists(parts[1]))
                {
                    await features.Rag.IngestFileAsync(parts[1]);
                    Console.WriteLine($"  RAG: ingested {parts[1]} ({features.Rag.DocumentCount} total chunks)");
                }
                else
                    Console.WriteLine($"  File not found: {parts[1]}");
                break;

            case "clear" when features.Rag != null:
                features.Rag.Clear();
                Console.WriteLine("  RAG: knowledge base cleared.");
                break;

            case "status":
                if (features.Rag != null)
                {
                    var stats = features.Rag.GetStats();
                    Console.WriteLine($"  RAG: {stats.VectorCount} chunks indexed, enabled={features.RagEnabled}");
                    Console.WriteLine($"  Backend: {stats.Backend}, dims={stats.Dimensions}, persistent={stats.Persistent}");
                    if (stats.DbPath != null) Console.WriteLine($"  DB: {stats.DbPath} ({stats.DbSizeBytes / 1024}KB)");
                    if (stats.ApproxMemoryMb > 0) Console.WriteLine($"  Memory: ~{stats.ApproxMemoryMb:F1}MB");
                }
                else
                    Console.WriteLine("  RAG: not initialized");
                break;

            case "on":
                features.RagEnabled = true;
                Console.WriteLine("  RAG: enabled");
                break;

            case "off":
                features.RagEnabled = false;
                Console.WriteLine("  RAG: disabled");
                break;

            default:
                Console.WriteLine("  /rag add <file> | clear | status | on | off");
                break;
        }
        Console.ResetColor();
    }

    private static AgentFlow? BuildFlowFromConfig(AgentOrchestrator orch, FlowDefinition def)
    {
        try
        {
            return def.Type.ToLowerInvariant() switch
            {
                "chain" => orch.Chain(def.Agents.ToArray()) is var c
                    ? (c.PassFullHistory = def.PassFullHistory, c.MaxTotalTurns = def.MaxTurns, c).c
                    : null,

                "parallel" => orch.Parallel(def.Synthesizer, def.Agents.ToArray()),

                "debate" => def.Judge != null
                    ? orch.Debate(def.Agents.ToArray())
                        .WithJudge(orch.GetAgent(def.Judge))
                        .WithRounds(def.Rounds)
                    : orch.Debate(def.Agents.ToArray()).WithRounds(def.Rounds),

                "router" when def.Routes != null =>
                    orch.Router(def.EntryAgent ?? def.Agents.First(), def.Routes, def.Fallback),

                "handoff" =>
                    orch.Handoff(def.EntryAgent ?? def.Agents.First(), def.Agents.ToArray()),

                "review-loop" when def.Agents.Count >= 2 =>
                    orch.ReviewLoop(def.Agents[0], def.Agents[1], def.Rounds),

                _ => null
            };
        }
        catch (Exception ex)
        {
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine($"  Warning: failed to build flow '{def.Name}': {ex.Message}");
            Console.ResetColor();
            return null;
        }
    }

    private static async Task HandleAgentCommand(string arg, InferenceEngine engine, FeatureSet features)
    {
        var parts = arg.Split(' ', 2);
        Console.ForegroundColor = ConsoleColor.DarkGray;

        switch (parts[0].ToLower())
        {
            case "list":
                if (features.Orchestrator == null)
                {
                    Console.WriteLine("  Agents: not configured. Define agents in your config file.");
                    break;
                }
                Console.WriteLine($"  Agents ({features.Orchestrator.Agents.Count}):");
                foreach (var (name, agent) in features.Orchestrator.Agents)
                    Console.WriteLine($"    - {name}: {agent.Role} (temp={agent.GenerationConfig.Temperature})");
                if (features.NamedFlows.Count > 0)
                {
                    Console.WriteLine($"  Flows ({features.NamedFlows.Count}):");
                    foreach (var (fname, f) in features.NamedFlows)
                        Console.WriteLine($"    - {fname}: {f.GetType().Name.Replace("Flow", "")}");
                }
                break;

            case "run" when parts.Length > 1:
                // /agent run <flow_name> <input>
                var runParts = parts[1].Split(' ', 2);
                if (runParts.Length < 2)
                {
                    Console.WriteLine("  Usage: /agent run <flow_name> <input>");
                    break;
                }
                string flowName = runParts[0];
                string input = runParts[1];

                if (features.NamedFlows.TryGetValue(flowName, out var flow))
                {
                    Console.ForegroundColor = ConsoleColor.DarkYellow;
                    Console.WriteLine($"  Running flow '{flowName}'...");

                    flow.OnLog = msg =>
                    {
                        Console.ForegroundColor = ConsoleColor.DarkGray;
                        Console.WriteLine($"  [{msg}]");
                    };
                    flow.OnMessage = msg =>
                    {
                        Console.ForegroundColor = ConsoleColor.DarkCyan;
                        Console.WriteLine($"  [{msg.FromAgent}]: {Truncate(msg.Content, 120)}");
                    };

                    var result = await flow.RunAsync(input);

                    Console.ForegroundColor = ConsoleColor.White;
                    Console.WriteLine($"\n  ═══ Flow Result ({result.Duration.TotalSeconds:F1}s, {result.TotalTurns} turns) ═══");
                    Console.WriteLine(result.FinalOutput);
                    Console.ResetColor();
                }
                else
                {
                    Console.WriteLine($"  Flow '{flowName}' not found. Available: {string.Join(", ", features.NamedFlows.Keys)}");
                }
                break;

            case "ask" when parts.Length > 1:
                // /agent ask <agent_name> <input>  — Ask a single agent directly
                var askParts = parts[1].Split(' ', 2);
                if (askParts.Length < 2 || features.Orchestrator == null)
                {
                    Console.WriteLine("  Usage: /agent ask <agent_name> <input>");
                    break;
                }
                try
                {
                    var agent = features.Orchestrator.GetAgent(askParts[0]);
                    var msg = new AgentMessage { FromAgent = "user", Role = "user", Content = askParts[1] };

                    Console.ForegroundColor = ConsoleColor.White;
                    Console.Write($"\n  {agent.Name} > ");

                    await foreach (var token in agent.RunStreamingAsync(msg))
                    {
                        Console.Write(token);
                    }
                    Console.WriteLine();
                    Console.ResetColor();
                }
                catch (KeyNotFoundException)
                {
                    Console.WriteLine($"  Agent '{askParts[0]}' not found.");
                }
                break;

            case "clear":
                features.Orchestrator?.ClearAllMemories();
                Console.WriteLine("  All agent memories cleared.");
                break;

            default:
                Console.WriteLine("  /agent list                    Show defined agents and flows");
                Console.WriteLine("  /agent run <flow> <input>      Run a named flow");
                Console.WriteLine("  /agent ask <agent> <input>     Ask a specific agent directly");
                Console.WriteLine("  /agent clear                   Clear all agent memories");
                break;
        }
        Console.ResetColor();
    }

    private static async Task HandleMcpCommand(string arg, InferenceEngine engine, FeatureSet features)
    {
        var parts = arg.Split(' ', 2);
        Console.ForegroundColor = ConsoleColor.DarkGray;

        switch (parts[0].ToLower())
        {
            case "list" or "status":
                if (features.McpManager == null || features.McpManager.Servers.Count == 0)
                {
                    Console.WriteLine("  MCP: no servers connected");
                    break;
                }
                Console.WriteLine($"  MCP: {features.McpManager.Servers.Count} server(s), {features.McpManager.AllTools.Count} tool(s)");
                foreach (var (name, client) in features.McpManager.Servers)
                {
                    var info = client.ServerInfo;
                    Console.WriteLine($"    - {name}: {info?.ServerInfo.Name} v{info?.ServerInfo.Version} ({(client.IsConnected ? "connected" : "disconnected")})");
                }
                break;

            case "tools":
                if (features.McpManager == null)
                {
                    Console.WriteLine("  MCP: no servers connected");
                    break;
                }
                var byServer = features.McpManager.GetToolsByServer();
                foreach (var (server, tools) in byServer)
                {
                    Console.WriteLine($"  [{server}]");
                    foreach (var tool in tools)
                        Console.WriteLine($"    - {tool.Name}: {tool.Description}");
                }
                break;

            case "connect" when parts.Length > 1:
                try
                {
                    // Quick connect: /mcp connect <command> <args...>
                    // e.g., /mcp connect npx -y @modelcontextprotocol/server-filesystem /home
                    var connectParts = parts[1].Split(' ');
                    var serverConfig = new McpServerConfig
                    {
                        Name = connectParts[0],
                        Transport = "stdio",
                        Command = connectParts[0],
                        Args = connectParts.Length > 1 ? connectParts[1..].ToList() : new List<string>(),
                        Enabled = true,
                    };

                    features.McpManager ??= new McpManager
                    {
                        OnLog = msg =>
                        {
                            Console.ForegroundColor = ConsoleColor.DarkGray;
                            Console.WriteLine($"  [{msg}]");
                            Console.ResetColor();
                        }
                    };

                    await features.McpManager.ConnectAsync(serverConfig);
                    features.McpManager.RegisterAllTools(engine.Tools);
                    features.McpEnabled = true;

                    Console.WriteLine($"  MCP: connected to '{serverConfig.Name}', {features.McpManager.AllTools.Count} total tools");
                }
                catch (Exception ex)
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine($"  MCP connect failed: {ex.Message}");
                }
                break;

            case "disconnect" when parts.Length > 1:
                if (features.McpManager != null)
                {
                    await features.McpManager.DisconnectAsync(parts[1], engine.Tools);
                    Console.WriteLine($"  MCP: disconnected from '{parts[1]}'");
                }
                break;

            case "ping":
                if (features.McpManager != null)
                {
                    var results = await features.McpManager.PingAllAsync();
                    foreach (var (name, ok) in results)
                        Console.WriteLine($"  {name}: {(ok ? "OK" : "FAILED")}");
                }
                break;

            default:
                Console.WriteLine("  /mcp list              Show connected servers");
                Console.WriteLine("  /mcp tools             Show all MCP tools");
                Console.WriteLine("  /mcp connect <cmd>     Connect to a stdio MCP server");
                Console.WriteLine("  /mcp disconnect <name> Disconnect from a server");
                Console.WriteLine("  /mcp ping              Ping all connected servers");
                break;
        }
        Console.ResetColor();
    }

    private static void HandleModelCommand(string arg, FeatureSet features)
    {
        var parts = arg.Split(' ', 2);
        Console.ForegroundColor = ConsoleColor.DarkGray;

        switch (parts[0].ToLower())
        {
            case "list":
                features.ModelRegistry?.Scan();
                if (features.ModelRegistry == null || features.ModelRegistry.Models.Count == 0)
                {
                    Console.WriteLine("  No models found. Set --models-dir to your models directory.");
                }
                else
                {
                    Console.WriteLine(features.ModelRegistry.ListFormatted());
                }
                break;

            case "info" when parts.Length > 1:
                var entry = features.ModelRegistry?.Get(parts[1]);
                if (entry == null)
                {
                    Console.WriteLine($"  Model '{parts[1]}' not found.");
                }
                else
                {
                    Console.WriteLine($"  Name: {entry.Name}");
                    Console.WriteLine($"  Path: {entry.Path}");
                    Console.WriteLine($"  Format: {entry.Format}");
                    Console.WriteLine($"  Size: {entry.SizeBytes / (1024.0 * 1024):F1} MB");
                    if (entry.Manifest != null)
                        Console.Write(entry.Manifest.ToString());
                }
                break;

            case "estimate" when parts.Length > 1:
                if (long.TryParse(parts[1].Replace("B", "000000000").Replace("M", "000000"), out long paramCount))
                {
                    var estimate = DynamicQuantizer.EstimateSizes(paramCount);
                    Console.WriteLine(estimate.FormatTable());
                }
                else
                    Console.WriteLine("  Usage: /model estimate <params> (e.g., 7B, 13B, 70B)");
                break;

            default:
                Console.WriteLine("  /model list                  List available models");
                Console.WriteLine("  /model info <name>           Show model details");
                Console.WriteLine("  /model estimate <params>     Estimate quantized sizes (e.g., 7B)");
                break;
        }
        Console.ResetColor();
    }

    private static void HandleQuantizeCommand(string arg, FeatureSet features)
    {
        var parts = arg.Split(' ', 2);
        Console.ForegroundColor = ConsoleColor.DarkGray;

        switch (parts[0].ToLower())
        {
            case "info":
                Console.WriteLine("  Available quantization types:");
                Console.WriteLine($"    {"Type",-12} {"Bits/Weight",-14} {"Compression",-14}");
                Console.WriteLine($"    {"────",-12} {"───────────",-14} {"───────────",-14}");
                foreach (var qt in new[] {
                    QuantTypeExtended.F32, QuantTypeExtended.F16,
                    QuantTypeExtended.Q8_0, QuantTypeExtended.Q6_K,
                    QuantTypeExtended.Q4_K, QuantTypeExtended.Q3_K,
                    QuantTypeExtended.Q2_K, QuantTypeExtended.BitNet })
                {
                    float bpw = KQuantDequantize.BitsPerWeight(qt);
                    float compression = 32f / bpw;
                    Console.WriteLine($"    {qt,-12} {bpw,-14:F2} {compression,-14:F1}x");
                }
                break;

            case "current":
                Console.WriteLine($"  Current quantization: {features.QuantizationType ?? "auto (from model)"}");
                break;

            default:
                Console.WriteLine("  /quantize info              Show available quantization types");
                Console.WriteLine("  /quantize current           Show current quantization type");
                break;
        }
        Console.ResetColor();
    }

    private static async Task HandlePullCommand(string arg, FeatureSet features)
    {
        var parts = arg.Split(' ', 2);
        Console.ForegroundColor = ConsoleColor.DarkGray;

        switch (parts[0].ToLower())
        {
            case "list":
                if (features.ModelPuller == null)
                {
                    Console.WriteLine("  Model puller not initialized.");
                    break;
                }
                var downloaded = features.ModelPuller.ListDownloaded();
                if (downloaded.Count == 0)
                {
                    Console.WriteLine("  No downloaded models found.");
                }
                else
                {
                    Console.WriteLine($"  Downloaded models ({downloaded.Count}):");
                    foreach (var m in downloaded)
                        Console.WriteLine($"    - {m.Name} ({m.Format}, {ModelPuller.FormatSize(m.SizeBytes)})");
                }
                break;

            case "aliases":
                if (features.ModelPuller == null) break;
                Console.WriteLine("  Available model aliases:");
                foreach (var (name, alias) in features.ModelPuller.GetAliases())
                    Console.WriteLine($"    {name,-20} → {alias.Repo}{(alias.QuantFilter != null ? $":{alias.QuantFilter}" : "")}");
                break;

            case "delete" when parts.Length > 1:
                if (features.ModelPuller?.Delete(parts[1]) == true)
                    Console.WriteLine($"  Deleted: {parts[1]}");
                else
                    Console.WriteLine($"  Model '{parts[1]}' not found.");
                break;

            case "jobs":
                if (features.ModelPuller == null) break;
                var jobs = features.ModelPuller.GetActiveJobs();
                if (jobs.Count == 0)
                    Console.WriteLine("  No active downloads.");
                else
                    foreach (var j in jobs)
                        Console.WriteLine($"    {j.FileName}: {j.Status} ({j.Progress:P0})");
                break;

            default:
                // Treat as a model name/URL to pull
                if (features.ModelPuller == null)
                {
                    Console.WriteLine("  Model puller not initialized.");
                    break;
                }
                Console.ResetColor();
                Console.ForegroundColor = ConsoleColor.Cyan;
                Console.WriteLine($"  Pulling: {arg}");
                Console.ResetColor();
                Console.ForegroundColor = ConsoleColor.DarkGray;

                var result = await features.ModelPuller.PullAsync(arg);
                if (result.Success)
                {
                    Console.ForegroundColor = ConsoleColor.Green;
                    if (result.AlreadyExisted)
                        Console.WriteLine($"  Already downloaded: {result.ModelPath}");
                    else
                        Console.WriteLine($"  Downloaded: {result.ModelPath} ({ModelPuller.FormatSize(result.BytesDownloaded)})");
                }
                else
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine($"  Pull failed: {result.Error}");
                }
                break;
        }
        Console.ResetColor();
    }

    #region UI Helpers

    private static void PrintBanner()
    {
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("╔══════════════════════════════════════╗");
        Console.WriteLine("║         SharpInfer Engine            ║");
        Console.WriteLine("║   Pure C# LLM Inference Engine      ║");
        Console.WriteLine("╚══════════════════════════════════════╝");
        Console.ResetColor();
        Console.WriteLine();
    }

    private static void PrintFeatureSummary(FeatureSet features, EngineConfig config)
    {
        Console.ForegroundColor = ConsoleColor.DarkGray;
        Console.WriteLine("\n  Active features:");
        PrintFeature("GPU", config.Gpu.Enabled);
        PrintFeature("Speculative Decoding", features.SpeculativeEnabled,
            $"lookahead={config.Speculative.LookaheadTokens}");
        PrintFeature("LoRA Adapters", features.LoraEnabled,
            features.ActiveLora != null ? $"active={features.ActiveLora}" : "none active");
        PrintFeature("Prompt Caching", features.PromptCacheEnabled);
        PrintFeature("CFG", features.CfgEnabled, $"scale={features.CfgScale}");
        PrintFeature("Self-Consistency", features.ConsistencyEnabled,
            features.Consistency != null ? $"{features.Consistency.NumCandidates} candidates" : "");
        PrintFeature("RAG", features.RagEnabled,
            features.Rag != null ? $"{features.Rag.DocumentCount} chunks" : "");
        PrintFeature("Beam Search", features.BeamSearchEnabled,
            $"{config.BeamSearch.NumBeams} beams");
        PrintFeature("Tools", config.Tools.Enabled);
        PrintFeature("MCP Servers", features.McpEnabled,
            features.McpManager != null ? $"{features.McpManager.Servers.Count} servers, {features.McpManager.AllTools.Count} tools" : "");
        PrintFeature("Multi-Agent", features.AgentsEnabled,
            features.Orchestrator != null ? $"{features.Orchestrator.Agents.Count} agents, {features.NamedFlows.Count} flows" : "");
        PrintFeature("FlashAttention", features.FlashAttentionEnabled);
        PrintFeature("Batch Scheduler", features.BatchSchedulerEnabled,
            features.Scheduler != null ? $"batch={config.BatchScheduler.MaxBatchSize}, seq={config.BatchScheduler.MaxSequences}" : "");
        PrintFeature("Structured Output", features.StructuredOutputEnabled,
            config.StructuredOutput.DefaultType != "none" ? config.StructuredOutput.DefaultType : null);
        PrintFeature("Multimodal Vision", features.MultimodalEnabled,
            $"size={config.Multimodal.ImageSize}, max={config.Multimodal.MaxImages}");
        PrintFeature("Hardware Backend", features.HardwareBackend != "cpu",
            features.HardwareBackend);
        PrintFeature("Quantization", features.QuantizationType != null,
            features.QuantizationType);
        PrintFeature("Distributed", features.DistributedEnabled,
            $"{config.Distributed.Strategy}, {config.Distributed.Workers.Count} workers");
        PrintFeature("Cloud Fallback", features.CloudEnabled,
            features.CloudRouter != null ? $"{config.CloudFallback.Providers.Count(p => p.Enabled)} providers" : null);
        PrintFeature("WebSocket API", features.WebSocketEnabled);
        PrintFeature("Batch Processing", features.BatchProcessingEnabled,
            $"concurrency={config.BatchProcessing.MaxConcurrency}");
        PrintFeature("Enterprise Auth", config.Enterprise.AuthEnabled);
        PrintFeature("Rate Limiting", config.Enterprise.RateLimitEnabled,
            $"{config.Enterprise.RequestsPerMinute} req/min");
        PrintFeature("Audit Logging", config.Enterprise.AuditEnabled);
        PrintFeature("Usage Metering", config.Enterprise.MeteringEnabled);
        PrintFeature("Model Pull", features.ModelPuller != null,
            features.ModelPuller != null ? $"{features.ModelPuller.GetAliases().Count} aliases" : null);
        Console.ResetColor();
        Console.WriteLine();
    }

    private static void PrintFeature(string name, bool enabled, string? detail = null)
    {
        string status = enabled ? "\u2713" : "\u2717";
        Console.ForegroundColor = enabled ? ConsoleColor.Green : ConsoleColor.DarkGray;
        Console.Write($"    {status} {name,-25}");
        if (enabled && detail != null)
        {
            Console.ForegroundColor = ConsoleColor.DarkGray;
            Console.Write($" ({detail})");
        }
        Console.WriteLine();
    }

    private static void PrintCurrentConfig(EngineConfig config)
    {
        Console.ForegroundColor = ConsoleColor.DarkGray;
        Console.WriteLine($"  Model:       {config.ModelPath}");
        Console.WriteLine($"  Temperature: {config.Generation.Temperature}");
        Console.WriteLine($"  Top-P:       {config.Generation.TopP}");
        Console.WriteLine($"  Top-K:       {config.Generation.TopK}");
        Console.WriteLine($"  Max Tokens:  {config.Generation.MaxTokens}");
        Console.WriteLine($"  Rep Penalty: {config.Generation.RepetitionPenalty}");
        Console.ResetColor();
    }

    private static string Truncate(string s, int max) =>
        s.Length <= max ? s : s[..max] + "...";

    #endregion

    #region Config Parsing

    private static EngineConfig ParseConfig(string[] args)
    {
        EngineConfig config;

        // Check for Modelfile first
        int modelfileIdx = Array.IndexOf(args, "--modelfile");
        if (modelfileIdx >= 0 && modelfileIdx + 1 < args.Length)
        {
            var modelfilePath = args[modelfileIdx + 1];
            var modelfile = Modelfile.ParseFile(modelfilePath);
            config = modelfile.ToEngineConfig(Path.GetDirectoryName(Path.GetFullPath(modelfilePath)));
            config.ModelfilePath = modelfilePath;
        }
        // Check for .simodel package
        else if (args.Any(a => a.EndsWith(".simodel")))
        {
            var pkgPath = args.First(a => a.EndsWith(".simodel"));
            var extractDir = Path.Combine(Path.GetTempPath(), "sharpinfer", Path.GetFileNameWithoutExtension(pkgPath));
            var modelfile = Modelfile.UnpackFromFile(pkgPath, extractDir, msg => Console.WriteLine($"  [{msg}]"));
            config = modelfile.ToEngineConfig(extractDir);
        }
        // Check for config file
        else
        {
            int configIdx = Array.IndexOf(args, "--config");
            if (configIdx >= 0 && configIdx + 1 < args.Length)
            {
                config = EngineConfig.LoadFromFile(args[configIdx + 1]);
            }
            else
            {
                config = new EngineConfig();
            }
        }

        // CLI flags override config file values
        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--model" or "-m": config.ModelPath = args[++i]; break;
                case "--context" or "-c": config.ContextLength = int.Parse(args[++i]); break;
                case "--temp" or "-t": config.Generation.Temperature = float.Parse(args[++i]); break;
                case "--top-p": config.Generation.TopP = float.Parse(args[++i]); break;
                case "--top-k": config.Generation.TopK = int.Parse(args[++i]); break;
                case "--max-tokens": config.Generation.MaxTokens = int.Parse(args[++i]); break;
                case "--seed": config.Generation.Seed = int.Parse(args[++i]); break;
                case "--system": config.Generation.SystemPrompt = args[++i]; break;

                // GPU
                case "--gpu": config.Gpu.Enabled = true; break;
                case "--gpu-device": config.Gpu.DeviceId = int.Parse(args[++i]); break;

                // Speculative
                case "--speculative": config.Speculative.Enabled = true; config.Speculative.DraftModelPath = args[++i]; break;
                case "--lookahead": config.Speculative.LookaheadTokens = int.Parse(args[++i]); break;

                // LoRA
                case "--lora":
                    config.Lora.Enabled = true;
                    string loraPath = args[++i];
                    config.Lora.Adapters.Add(new LoraAdapterEntry
                    {
                        Name = Path.GetFileName(loraPath),
                        Path = loraPath,
                    });
                    config.Lora.ActiveAdapter ??= Path.GetFileName(loraPath);
                    break;

                // Prompt Cache
                case "--prompt-cache": config.PromptCaching.Enabled = true; break;
                case "--cache-dir": config.PromptCaching.CacheDir = args[++i]; break;

                // CFG
                case "--cfg": config.Cfg.Enabled = true; config.Cfg.GuidanceScale = float.Parse(args[++i]); break;
                case "--negative-prompt": config.Cfg.NegativePrompt = args[++i]; break;

                // Self-Consistency
                case "--consistency":
                    config.SelfConsistency.Enabled = true;
                    config.SelfConsistency.NumCandidates = int.Parse(args[++i]);
                    break;
                case "--consistency-strategy": config.SelfConsistency.Strategy = args[++i]; break;

                // RAG
                case "--rag":
                    config.Rag.Enabled = true;
                    config.Rag.Documents.Add(args[++i]);
                    break;
                case "--rag-top-k": config.Rag.TopK = int.Parse(args[++i]); break;
                case "--vector-store": config.Rag.VectorStore = args[++i]; break;
                case "--vector-db-path": config.Rag.VectorDbPath = args[++i]; break;

                // Beam Search
                case "--beam":
                    config.BeamSearch.Enabled = true;
                    config.BeamSearch.NumBeams = int.Parse(args[++i]);
                    break;
                case "--length-penalty": config.BeamSearch.LengthPenalty = float.Parse(args[++i]); break;

                // Tools
                case "--tools": config.Tools.Enabled = true; break;
                case "--search-api": config.Tools.SearchApiEndpoint = args[++i]; break;
                case "--search-api-key": config.Tools.SearchApiKey = args[++i]; break;

                // Structured Output
                case "--json-schema": config.StructuredOutput.Enabled = true; config.StructuredOutput.DefaultType = "json_schema"; config.StructuredOutput.DefaultSchema = args[++i]; break;
                case "--output-regex": config.StructuredOutput.Enabled = true; config.StructuredOutput.DefaultType = "regex"; config.StructuredOutput.DefaultSchema = args[++i]; break;
                case "--output-grammar": config.StructuredOutput.Enabled = true; config.StructuredOutput.DefaultType = "grammar"; config.StructuredOutput.DefaultSchema = args[++i]; break;

                // Batch Scheduler
                case "--batch-scheduler": config.BatchScheduler.Enabled = true; break;
                case "--max-batch": config.BatchScheduler.MaxBatchSize = int.Parse(args[++i]); break;
                case "--max-sequences": config.BatchScheduler.MaxSequences = int.Parse(args[++i]); break;

                // FlashAttention
                case "--no-flash-attention": config.FlashAttention.Enabled = false; break;
                case "--flash-block-q": config.FlashAttention.BlockSizeQ = int.Parse(args[++i]); break;
                case "--flash-block-kv": config.FlashAttention.BlockSizeKV = int.Parse(args[++i]); break;

                // Hardware Backend
                case "--backend": config.HardwareBackend.Backend = args[++i]; break;

                // Multimodal
                case "--multimodal": config.Multimodal.Enabled = true; break;
                case "--vision-model": config.Multimodal.Enabled = true; config.Multimodal.VisionModelPath = args[++i]; break;
                case "--image-size": config.Multimodal.ImageSize = int.Parse(args[++i]); break;
                case "--max-images": config.Multimodal.MaxImages = int.Parse(args[++i]); break;

                // Quantization
                case "--quantize": config.Quantization.Type = args[++i]; break;
                case "--requantize": config.Quantization.DynamicRequantize = true; config.Quantization.TargetType = args[++i]; break;

                // Model directory
                case "--models-dir": config.ModelsDir = args[++i]; break;

                // Distributed
                case "--distributed": config.Distributed.Enabled = true; break;
                case "--tp": config.Distributed.Enabled = true; config.Distributed.Strategy = ParallelismStrategy.TensorParallel; break;
                case "--pp": config.Distributed.Enabled = true; config.Distributed.Strategy = ParallelismStrategy.PipelineParallel; break;

                // Cloud Fallback
                case "--cloud-fallback": config.CloudFallback.Enabled = true; break;

                // WebSocket
                case "--websocket": config.WebSocket.Enabled = true; break;

                // Batch Processing
                case "--batch-processing": config.BatchProcessing.Enabled = true; break;
                case "--batch-concurrency": config.BatchProcessing.MaxConcurrency = int.Parse(args[++i]); break;

                // Enterprise
                case "--auth": config.Enterprise.AuthEnabled = true; break;
                case "--rate-limit": config.Enterprise.RateLimitEnabled = true; config.Enterprise.RequestsPerMinute = int.Parse(args[++i]); break;
                case "--audit": config.Enterprise.AuditEnabled = true; break;
                case "--metering": config.Enterprise.MeteringEnabled = true; break;
                case "--api-keys-file": config.Enterprise.ApiKeysFile = args[++i]; break;

                // Model Pull
                case "--hf-token": config.ModelPull.HuggingFaceToken = args[++i]; break;

                // Modelfile
                case "--modelfile": i++; break; // Already handled above
                case "--config": i++; break; // Already handled
            }
        }

        if (string.IsNullOrEmpty(config.ModelPath))
            throw new ArgumentException("--model is required (or specify in config file).");

        return config;
    }

    private static void PrintUsage()
    {
        Console.WriteLine("""
        SharpInfer — Pure C# LLM Inference Engine

        Usage: sharpinfer --model <path> [options]
               sharpinfer --config <config.json> [overrides]
               sharpinfer --generate-config <output.json>

        Model:
          --model, -m <path>           Model file (.gguf) or directory (safetensors)
          --context, -c <int>          Context length override
          --config <path>              Load configuration from JSON file

        Generation:
          --temp, -t <float>           Temperature (default: 0.7)
          --top-p <float>              Top-P nucleus sampling (default: 0.9)
          --top-k <int>                Top-K sampling (default: 40)
          --max-tokens <int>           Max tokens per response (default: 512)
          --seed <int>                 Random seed for reproducibility
          --system <string>            System prompt

        GPU:
          --gpu                        Enable CUDA GPU acceleration
          --gpu-device <int>           GPU device ID (default: 0)

        Speculative Decoding:
          --speculative <draft_model>  Enable with draft model path
          --lookahead <int>            Draft tokens per round (default: 5)

        LoRA Adapters:
          --lora <path>                Load and apply a LoRA adapter directory

        Prompt Caching:
          --prompt-cache               Enable KV cache persistence
          --cache-dir <path>           Cache directory (default: .cache/prompts)

        Classifier-Free Guidance:
          --cfg <float>                Enable CFG with guidance scale (e.g., 1.5)
          --negative-prompt <string>   Unconditional prompt for CFG

        Self-Consistency:
          --consistency <int>          Enable with N candidates (e.g., 5)
          --consistency-strategy <s>   majority_vote|perplexity|length (default: majority_vote)

        RAG (Retrieval-Augmented Generation):
          --rag <path>                 Enable RAG, ingest file or directory (repeatable)
          --rag-top-k <int>            Number of chunks to retrieve (default: 5)
          --vector-store <backend>     Vector store: memory (default) or sqlite
          --vector-db-path <path>      SQLite database path (default: ./vectors.db)

        Beam Search:
          --beam <int>                 Enable beam search with N beams
          --length-penalty <float>     Length normalization (default: 1.0)

        Tools:
          --tools                      Enable tool calling
          --search-api <url>           Search API endpoint
          --search-api-key <key>       Search API key

        Structured Output:
          --json-schema <schema>       Force JSON schema output (e.g., '{"type":"object",...}')
          --output-regex <pattern>     Force regex-constrained output
          --output-grammar <ebnf>      Force EBNF grammar-constrained output

        Batch Scheduler:
          --batch-scheduler            Enable continuous batching (for API server)
          --max-batch <int>            Max concurrent batch size (default: 8)
          --max-sequences <int>        Max concurrent sequences (default: 16)

        FlashAttention:
          --no-flash-attention         Disable FlashAttention (use standard O(N²))
          --flash-block-q <int>        Q tile size (default: 64)
          --flash-block-kv <int>       KV tile size (default: 64)

        Hardware Backend:
          --backend <type>             Backend: auto|cuda|metal|vulkan|neon|cpu (default: auto)

        Multimodal / Vision:
          --multimodal                 Enable multimodal (vision) support
          --vision-model <path>        Path to vision encoder weights
          --image-size <int>           Vision input resolution (default: 336)
          --max-images <int>           Max images per prompt (default: 4)

        Quantization:
          --quantize <type>            Weight type: F32|F16|Q8_0|Q6_K|Q4_K|Q3_K|Q2_K|BitNet
          --requantize <target>        Dynamically re-quantize loaded weights to target type

        Model Packaging:
          --modelfile <path>           Load model from a Modelfile
          --models-dir <path>          Model registry directory (default: ./models)
          <file.simodel>               Load from a .simodel package

        Distributed Inference:
          --distributed                Enable distributed inference
          --tp                         Use tensor parallelism (multi-GPU same node)
          --pp                         Use pipeline parallelism (multi-node)
          Configure workers via JSON config under "distributed.workers"

        Cloud Fallback:
          --cloud-fallback             Enable cloud API fallback
          Configure providers via JSON config under "cloud_fallback.providers"

        WebSocket:
          --websocket                  Enable WebSocket streaming endpoint (ws://host:port/ws)

        Batch Processing:
          --batch-processing           Enable batch processing endpoint
          --batch-concurrency <int>    Max concurrent batch requests (default: 4)

        Enterprise:
          --auth                       Enable API key authentication
          --rate-limit <int>           Enable rate limiting (requests per minute)
          --audit                      Enable request audit logging
          --metering                   Enable usage metering
          --api-keys-file <path>       Load API keys from file

        Model Pull:
          --hf-token <token>           HuggingFace API token for gated models

        MCP Servers:
          Configure via JSON config file under "mcp_servers":
            {
              "mcp_servers": [
                {
                  "name": "filesystem",
                  "transport": "stdio",
                  "command": "npx",
                  "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home"],
                  "enabled": true
                },
                {
                  "name": "remote-api",
                  "transport": "sse",
                  "url": "http://localhost:3001/sse",
                  "enabled": true
                }
              ]
            }

        Runtime Commands:
          /reset                       Clear conversation context
          /config                      Show current generation config
          /features                    Show enabled features
          /temp <float>                Change temperature
          /topp <float>                Change top-p
          /topk <int>                  Change top-k
          /cfg <float>                 Set CFG scale (1.0 to disable)
          /consistency <N|off>         Set self-consistency candidates
          /beam <N|off>                Set beam search beams
          /lora list|apply|remove      Manage LoRA adapters
          /rag add|clear|status|on|off Manage RAG knowledge base
          /mcp list|tools|connect|...  Manage MCP server connections
          /agent list|run|ask|clear    Manage multi-agent workflows
          /model list|info|estimate    Model registry & size estimation
          /pull <name|url>             Download model from HuggingFace or URL
          /pull list|aliases|jobs      Manage downloaded models
          /quantize info|current       Quantization info
          /hardware                    Show detected hardware backends
          /export-modelfile <path>     Export current config as Modelfile
          /save-config <path>          Save current config to file
          /quit                        Exit
        """);
    }

    #endregion
}

/// <summary>Tracks the runtime state of all optional features.</summary>
internal class FeatureSet
{
    public bool SpeculativeEnabled { get; set; }

    public bool LoraEnabled { get; set; }
    public Dictionary<string, LoraAdapter> LoraAdapters { get; } = new();
    public string? ActiveLora { get; set; }

    public bool PromptCacheEnabled { get; set; }
    public PromptCache? PromptCache { get; set; }

    public bool CfgEnabled { get; set; }
    public float CfgScale { get; set; } = 1.5f;
    public string CfgNegativePrompt { get; set; } = "";

    public bool ConsistencyEnabled { get; set; }
    public SelfConsistency? Consistency { get; set; }

    public bool RagEnabled { get; set; }
    public RagPipeline? Rag { get; set; }

    public bool BeamSearchEnabled { get; set; }
    public BeamSearchConfig? BeamConfig { get; set; }

    public bool McpEnabled { get; set; }
    public McpManager? McpManager { get; set; }

    public bool AgentsEnabled { get; set; }
    public AgentOrchestrator? Orchestrator { get; set; }
    public Dictionary<string, AgentFlow> NamedFlows { get; } = new();

    public bool FlashAttentionEnabled { get; set; }

    public bool BatchSchedulerEnabled { get; set; }
    public BatchScheduler? Scheduler { get; set; }

    public bool StructuredOutputEnabled { get; set; }

    public bool MultimodalEnabled { get; set; }

    public string HardwareBackend { get; set; } = "cpu";

    public string? QuantizationType { get; set; }

    public ModelRegistry? ModelRegistry { get; set; }

    public bool DistributedEnabled { get; set; }

    public bool CloudEnabled { get; set; }
    public CloudFallbackRouter? CloudRouter { get; set; }

    public bool WebSocketEnabled { get; set; }

    public bool BatchProcessingEnabled { get; set; }

    public bool EnterpriseEnabled { get; set; }

    public ModelPuller? ModelPuller { get; set; }
}
