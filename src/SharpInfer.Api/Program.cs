using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.OpenApi.Models;
using SharpInfer.Api.Models;
using SharpInfer.Core.Engine;
using SharpInfer.Core.Sampling;
using SharpInfer.Gpu;
using Microsoft.AspNetCore.Mvc;
using ChatMessage = SharpInfer.Api.Models.ChatMessage;

namespace SharpInfer.Api;

/// <summary>
/// SharpInfer REST API — OpenAI-compatible inference server with model management.
///
/// Compatible with:
/// - Continue.dev (VS Code AI coding assistant)
/// - Open WebUI
/// - Any tool that speaks the OpenAI API format
///
/// Usage:
///   dotnet run --project SharpInfer.Api -- [--model path/to/model.gguf] [--gpu] [--port 3512]
///
/// The API can start without a model. Use the model management endpoints to
/// pull and load models at runtime.
///
/// Chat endpoints:
///   POST /v1/chat/completions  — Chat completions (streaming + non-streaming)
///   GET  /v1/models            — List available models
///   GET  /health               — Health check
///
/// Model management endpoints:
///   GET    /api/tags           — List downloaded models
///   POST   /api/pull           — Pull/download a model (NDJSON streaming progress)
///   DELETE /api/delete         — Delete a downloaded model
///   POST   /api/show           — Show model info
///   POST   /api/load           — Load a model into the engine
/// </summary>
public class Program
{
    private static InferenceEngine? _engine;
    private static readonly object _engineLock = new();
    private static ModelPuller? _modelPuller;
    private static string _modelsDir = "./models";
    private static bool _useGpu = false;
    private static int? _contextLength;
    private static string? _activeModelPath;
    private static OrphanedModelFinder? _orphanFinder;

    private static readonly JsonSerializerOptions _ndjsonOptions = new()
    {
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
    };

    public static async Task Main(string[] args)
    {
        // Parse our custom args before passing to ASP.NET
        string? modelPath = null;
        int port = 3512;
        string? hfToken = null;

        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--model" or "-m": modelPath = args[++i]; break;
                case "--gpu": _useGpu = true; break;
                case "--port" or "-p": port = int.Parse(args[++i]); break;
                case "--context" or "-c": _contextLength = int.Parse(args[++i]); break;
                case "--models-dir": _modelsDir = args[++i]; break;
                case "--hf-token": hfToken = args[++i]; break;
            }
        }

        // Initialize model puller (always available for management)
        _modelPuller = new ModelPuller(_modelsDir, hfToken)
        {
            OnLog = msg => Console.WriteLine($"  [{msg}]"),
        };
        _orphanFinder = new OrphanedModelFinder(_modelsDir);
        Console.WriteLine($"Models directory: {Path.GetFullPath(_modelsDir)}");
        Console.WriteLine($"Model puller: ready ({_modelPuller.GetAliases().Count} aliases)");

        // Optionally pre-load a model (backward compatibility)
        if (!string.IsNullOrEmpty(modelPath))
        {
            Console.WriteLine($"Loading model: {modelPath}");
            LoadModel(modelPath);
            if (_engine != null)
                Console.WriteLine($"Model loaded: {_engine.Config}");
            else
                Console.WriteLine("Warning: Failed to load initial model. API starting without a model.");
        }
        else
        {
            Console.WriteLine("No model specified. API starting without a model.");
            Console.WriteLine("Use POST /api/pull to download models and POST /api/load to activate one.");
        }

        // Build web app
        var builder = WebApplication.CreateBuilder();
        builder.Services.AddCors();
        builder.Services.AddEndpointsApiExplorer();
        builder.Services.AddSwaggerGen(c =>
        {
            c.SwaggerDoc("v1", new OpenApiInfo
            {
                Title = "SharpInfer API",
                Version = "1.0",
                Description = "Pure C# LLM inference engine with OpenAI-compatible chat completions "
                            + "and built-in model management (pull, load, delete, list). "
                            + "Start the API with no model, then use the /api/* endpoints to download "
                            + "and load models at runtime.",
            });
        });
        var app = builder.Build();

        // --- Swagger ---
        app.UseSwagger();
        app.UseSwaggerUI(c =>
        {
            c.SwaggerEndpoint("/swagger/v1/swagger.json", "SharpInfer API v1");
            c.RoutePrefix = "swagger";
            c.DocumentTitle = "SharpInfer API Documentation";
        });

        // --- CORS for web UIs ---
        app.UseCors(policy => policy.AllowAnyOrigin().AllowAnyMethod().AllowAnyHeader());

        // --- OpenAI-compatible routes ---
        app.MapGet("/health", HandleHealth)
            .WithName("HealthCheck")
            .WithTags("Health")
            .WithSummary("Health check")
            .WithDescription("Returns API health status, loaded model information, and models directory path. "
                           + "Status is \"ok\" when a model is loaded, \"no_model\" when the API is running but no model is active.")
            .Produces<object>(StatusCodes.Status200OK);

        app.MapGet("/v1/models", HandleListModels)
            .WithName("ListModels")
            .WithTags("Chat")
            .WithSummary("List available models")
            .WithDescription("Returns an OpenAI-compatible list of models. Includes the currently loaded model "
                           + "(if any) and all downloaded models in the models directory.")
            .Produces<ModelListResponse>(StatusCodes.Status200OK);

        app.MapPost("/v1/chat/completions",
            async (ChatCompletionRequest request, HttpContext http) => await HandleChatCompletion(request, http))
            .WithName("ChatCompletions")
            .WithTags("Chat")
            .WithSummary("Chat completions")
            .WithDescription("Generate chat completions from a list of messages. Supports both streaming (SSE) "
                           + "and non-streaming responses. Compatible with the OpenAI Chat Completions API format. "
                           + "Returns 503 if no model is loaded.")
            .Produces<ChatCompletionResponse>(StatusCodes.Status200OK)
            .Produces(StatusCodes.Status400BadRequest)
            .Produces(StatusCodes.Status503ServiceUnavailable);

        // --- Model management routes ---
        app.MapGet("/api/tags", HandleTags)
            .WithName("ListDownloadedModels")
            .WithTags("Model Management")
            .WithSummary("List downloaded models")
            .WithDescription("Returns all model files found in the models directory with their name, size, "
                           + "format, download date, and SHA256 digest (if available).")
            .Produces<ModelTagsResponse>(StatusCodes.Status200OK);

        app.MapPost("/api/pull",
            async (ModelPullRequest request, HttpContext http) => await HandlePull(request, http))
            .WithName("PullModel")
            .WithTags("Model Management")
            .WithSummary("Pull/download a model")
            .WithDescription("Download a model from HuggingFace. Use the format \"hf.co/<user>/<repo>:<quant>\" "
                           + "(e.g., \"hf.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M\"). "
                           + "When streaming is enabled (default), returns NDJSON progress lines. "
                           + "Built-in aliases like \"llama3\" and \"mistral\" are also supported.")
            .Produces<ModelPullProgress>(StatusCodes.Status200OK)
            .Produces(StatusCodes.Status400BadRequest);

        app.MapDelete("/api/delete",
            async ([FromBody] ModelDeleteRequest request) => await HandleDelete(request))
            .WithName("DeleteModel")
            .WithTags("Model Management")
            .WithSummary("Delete a downloaded model")
            .WithDescription("Delete a model file from the models directory. If the model is currently loaded, "
                           + "it will be unloaded first. The name is matched against filenames (partial match supported).")
            .Produces(StatusCodes.Status200OK)
            .Produces(StatusCodes.Status404NotFound);

        app.MapPost("/api/show",
            async (ModelShowRequest request) => await HandleShow(request))
            .WithName("ShowModelInfo")
            .WithTags("Model Management")
            .WithSummary("Show model details")
            .WithDescription("Returns metadata about a downloaded model including its path, format, "
                           + "and engine parameters (context length, vocab size) if the model is currently loaded.")
            .Produces<ModelShowResponse>(StatusCodes.Status200OK)
            .Produces(StatusCodes.Status404NotFound);

        app.MapPost("/api/load",
            async (ModelLoadRequest request) => await HandleLoad(request))
            .WithName("LoadModel")
            .WithTags("Model Management")
            .WithSummary("Load a model into the engine")
            .WithDescription("Load a downloaded model into memory for inference. If another model is currently loaded, "
                           + "it is unloaded first. The name is resolved by searching the models directory "
                           + "(exact match, with .gguf extension, or partial filename match).")
            .Produces(StatusCodes.Status200OK)
            .Produces(StatusCodes.Status404NotFound)
            .Produces(StatusCodes.Status500InternalServerError);

        // --- Orphan management routes ---
        app.MapGet("/api/orphans", HandleListOrphans)
            .WithName("ListOrphanedModels")
            .WithTags("Model Management")
            .WithSummary("List orphaned models")
            .WithDescription("Find model files in the models directory that are not referenced by any "
                           + "Modelfile's FROM or ADAPTER directives. These models can be safely deleted to free disk space.")
            .Produces<OrphanedModelsResponse>(StatusCodes.Status200OK);

        app.MapDelete("/api/orphans",
            async ([FromBody] OrphanCleanupRequest request) => await HandleCleanupOrphans(request))
            .WithName("CleanupOrphanedModels")
            .WithTags("Model Management")
            .WithSummary("Clean up orphaned models")
            .WithDescription("Delete model files not referenced by any Modelfile. Defaults to dry_run=true "
                           + "(simulation only). Set dry_run=false to actually delete files. "
                           + "Optionally specify a list of model names to selectively delete.")
            .Produces<OrphanCleanupResponse>(StatusCodes.Status200OK);

        Console.WriteLine($"\nSharpInfer API listening on http://localhost:{port}");
        Console.WriteLine($"Swagger UI: http://localhost:{port}/swagger");
        Console.WriteLine("OpenAI-compatible endpoints:");
        Console.WriteLine($"  POST http://localhost:{port}/v1/chat/completions");
        Console.WriteLine($"  GET  http://localhost:{port}/v1/models");
        Console.WriteLine($"  GET  http://localhost:{port}/health");
        Console.WriteLine("Model management endpoints:");
        Console.WriteLine($"  GET    http://localhost:{port}/api/tags");
        Console.WriteLine($"  POST   http://localhost:{port}/api/pull");
        Console.WriteLine($"  DELETE http://localhost:{port}/api/delete");
        Console.WriteLine($"  POST   http://localhost:{port}/api/show");
        Console.WriteLine($"  POST   http://localhost:{port}/api/load");
        Console.WriteLine($"  GET    http://localhost:{port}/api/orphans");
        Console.WriteLine($"  DELETE http://localhost:{port}/api/orphans");

        await app.RunAsync($"http://0.0.0.0:{port}");
    }

    // ─── Model Lifecycle ──────────────────────────────────────────

    /// <summary>Load a model into the engine, disposing the previous one.</summary>
    private static bool LoadModel(string modelPath)
    {
        lock (_engineLock)
        {
            try
            {
                var engineOptions = new EngineOptions
                {
                    ContextLength = _contextLength,
                    ProgressCallback = msg => Console.WriteLine($"  [{msg}]"),
                };

                if (_useGpu)
                {
                    var cuda = new CudaBackend();
                    if (cuda.IsAvailable) engineOptions.ComputeBackend = cuda;
                }

                _engine?.Dispose();
                _engine = InferenceEngine.Load(modelPath, engineOptions);
                _activeModelPath = modelPath;
                return true;
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"Failed to load model: {ex.Message}");
                _engine = null;
                _activeModelPath = null;
                return false;
            }
        }
    }

    /// <summary>Get the current engine, or null if no model is loaded.</summary>
    private static InferenceEngine? GetEngine()
    {
        lock (_engineLock) { return _engine; }
    }

    /// <summary>Resolve a model name to a file path in the models directory.</summary>
    private static string? ResolveModelPath(string name)
    {
        // Direct path
        if (File.Exists(name)) return name;

        // In models dir by exact name
        string inDir = Path.Combine(_modelsDir, name);
        if (File.Exists(inDir)) return inDir;

        // With .gguf extension
        if (File.Exists(inDir + ".gguf")) return inDir + ".gguf";

        // Search models dir for partial match
        if (Directory.Exists(_modelsDir))
        {
            foreach (var file in Directory.GetFiles(_modelsDir, "*.gguf"))
            {
                if (Path.GetFileNameWithoutExtension(file)
                    .Contains(name, StringComparison.OrdinalIgnoreCase))
                    return file;
            }
        }

        return null;
    }

    // ─── Health & Models ──────────────────────────────────────────

    private static IResult HandleHealth()
    {
        var engine = GetEngine();
        return Results.Ok(new
        {
            status = engine != null ? "ok" : "no_model",
            model = engine?.Config?.ToString() ?? "none",
            models_dir = Path.GetFullPath(_modelsDir),
        });
    }

    private static IResult HandleListModels()
    {
        var engine = GetEngine();
        var models = new List<ModelInfo>();

        if (engine != null)
        {
            models.Add(new ModelInfo
            {
                Id = Path.GetFileNameWithoutExtension(_activeModelPath ?? "sharpinfer"),
                OwnedBy = "local"
            });
        }

        // Also list downloaded models not currently loaded
        if (_modelPuller != null)
        {
            foreach (var dm in _modelPuller.ListDownloaded())
            {
                string id = Path.GetFileNameWithoutExtension(dm.Name);
                if (models.All(m => m.Id != id))
                {
                    models.Add(new ModelInfo { Id = id, OwnedBy = "local" });
                }
            }
        }

        return Results.Ok(new ModelListResponse { Data = models });
    }

    // ─── Model Management: Tags ───────────────────────────────────

    private static IResult HandleTags()
    {
        if (_modelPuller == null)
            return Results.Ok(new ModelTagsResponse());

        var downloaded = _modelPuller.ListDownloaded();
        var tags = new ModelTagsResponse
        {
            Models = downloaded.Select(dm => new DownloadedModelEntry
            {
                Name = dm.Name,
                Size = dm.SizeBytes,
                ModifiedAt = dm.DownloadedAt.ToString("o"),
                Digest = "", // SHA256 read from companion file if available
                Details = new DownloadedModelDetails
                {
                    Format = dm.Format,
                },
            }).ToList()
        };

        // Try to read SHA256 digests from companion files
        foreach (var tag in tags.Models)
        {
            var downloaded_model = downloaded.FirstOrDefault(d => d.Name == tag.Name);
            if (downloaded_model != null)
            {
                string sha256Path = downloaded_model.Path + ".sha256";
                if (File.Exists(sha256Path))
                {
                    try { tag.Digest = $"sha256:{File.ReadAllText(sha256Path).Trim()}"; }
                    catch { /* ignore */ }
                }
            }
        }

        return Results.Ok(tags);
    }

    // ─── Model Management: Pull ─────────────────────────────────────

    private static async Task HandlePull(ModelPullRequest request, HttpContext http)
    {
        if (_modelPuller == null)
        {
            http.Response.StatusCode = 500;
            await http.Response.WriteAsJsonAsync(new { error = "Model puller not initialized" });
            return;
        }

        if (string.IsNullOrWhiteSpace(request.Name))
        {
            http.Response.StatusCode = 400;
            await http.Response.WriteAsJsonAsync(new { error = "Missing 'name' field" });
            return;
        }

        bool stream = request.Stream ?? true;
        string modelName = request.Name;

        // Convert hf.co/ format to HuggingFace repo format
        // e.g., "hf.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:latest"
        //     → "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
        if (modelName.StartsWith("hf.co/", StringComparison.OrdinalIgnoreCase))
        {
            modelName = modelName["hf.co/".Length..];
            // Strip :tag suffix if present (e.g., ":latest", ":Q4_K_M")
            int colonIdx = modelName.LastIndexOf(':');
            string? quantFilter = null;
            if (colonIdx > 0)
            {
                string tag = modelName[(colonIdx + 1)..];
                modelName = modelName[..colonIdx];
                // "latest" means no filter; anything else is a quant filter
                if (!tag.Equals("latest", StringComparison.OrdinalIgnoreCase))
                    quantFilter = tag;
            }
            // Reassemble with quant filter if specified
            if (quantFilter != null)
                modelName = $"{modelName}:{quantFilter}";
        }

        if (stream)
        {
            http.Response.ContentType = "application/x-ndjson";

            // Write initial status
            await WriteNdjson(http, new ModelPullProgress { Status = "pulling manifest" });

            // Set up progress callback
            _modelPuller.OnProgress = progress =>
            {
                try
                {
                    var line = new ModelPullProgress
                    {
                        Status = $"downloading {progress.FileName}",
                        Digest = $"sha256:{progress.FileName}",
                        Total = progress.TotalBytes,
                        Completed = progress.BytesDownloaded,
                    };
                    var json = JsonSerializer.Serialize(line, _ndjsonOptions);
                    http.Response.WriteAsync($"{json}\n").GetAwaiter().GetResult();
                    http.Response.Body.FlushAsync().GetAwaiter().GetResult();
                }
                catch { /* client may have disconnected */ }
            };

            try
            {
                var result = await _modelPuller.PullAsync(modelName);

                if (result.Success)
                {
                    await WriteNdjson(http, new ModelPullProgress { Status = "verifying sha256 digest" });
                    await WriteNdjson(http, new ModelPullProgress { Status = "writing manifest" });
                    await WriteNdjson(http, new ModelPullProgress { Status = "success" });
                }
                else
                {
                    await WriteNdjson(http, new ModelPullProgress
                    {
                        Status = $"error: {result.Error ?? "unknown error"}"
                    });
                }
            }
            catch (Exception ex)
            {
                await WriteNdjson(http, new ModelPullProgress
                {
                    Status = $"error: {ex.Message}"
                });
            }
            finally
            {
                _modelPuller.OnProgress = null;
            }
        }
        else
        {
            // Non-streaming: just return the final result
            try
            {
                var result = await _modelPuller.PullAsync(modelName);
                if (result.Success)
                    await http.Response.WriteAsJsonAsync(new { status = "success", model = result.ModelPath });
                else
                {
                    http.Response.StatusCode = 500;
                    await http.Response.WriteAsJsonAsync(new { error = result.Error });
                }
            }
            catch (Exception ex)
            {
                http.Response.StatusCode = 500;
                await http.Response.WriteAsJsonAsync(new { error = ex.Message });
            }
        }
    }

    // ─── Model Management: Delete ───────────────────────────────────

    private static Task<IResult> HandleDelete(ModelDeleteRequest request)
    {
        if (_modelPuller == null)
            return Task.FromResult(Results.Json(new { error = "Model puller not initialized" }, statusCode: 500));

        if (string.IsNullOrWhiteSpace(request.Name))
            return Task.FromResult(Results.BadRequest(new { error = "Missing 'name' field" }));

        // If this model is currently loaded, unload it
        lock (_engineLock)
        {
            if (_activeModelPath != null &&
                (Path.GetFileName(_activeModelPath).Contains(request.Name, StringComparison.OrdinalIgnoreCase) ||
                 Path.GetFileNameWithoutExtension(_activeModelPath).Contains(request.Name, StringComparison.OrdinalIgnoreCase)))
            {
                _engine?.Dispose();
                _engine = null;
                _activeModelPath = null;
                Console.WriteLine($"Unloaded active model (matched delete target: {request.Name})");
            }
        }

        bool deleted = _modelPuller.Delete(request.Name);
        if (deleted)
            return Task.FromResult(Results.Ok(new { status = "success" }));
        else
            return Task.FromResult(Results.NotFound(new { error = $"Model '{request.Name}' not found" }));
    }

    // ─── Model Management: Show ─────────────────────────────────────

    private static Task<IResult> HandleShow(ModelShowRequest request)
    {
        if (string.IsNullOrWhiteSpace(request.Name))
            return Task.FromResult(Results.BadRequest(new { error = "Missing 'name' field" }));

        // Find the model in downloaded list
        var model = _modelPuller?.ListDownloaded()
            .FirstOrDefault(m => m.Name.Contains(request.Name, StringComparison.OrdinalIgnoreCase));

        if (model == null)
            return Task.FromResult(Results.NotFound(new { error = $"Model '{request.Name}' not found" }));

        // Check if it's the currently loaded model — provide engine info
        var engine = GetEngine();
        string parameters = "";
        if (engine != null && _activeModelPath == model.Path)
        {
            parameters = $"context_length {engine.Config.MaxSequenceLength}\n"
                       + $"vocab_size {engine.Config.VocabSize}";
        }

        return Task.FromResult(Results.Ok<ModelShowResponse>(new ModelShowResponse
        {
            Modelfile = $"FROM {model.Path}",
            Parameters = parameters,
            Template = "<|system|>\n{{{{ .System }}}}\n<|end|>\n<|user|>\n{{{{ .Prompt }}}}\n<|end|>\n<|assistant|>",
            Details = new DownloadedModelDetails
            {
                Format = model.Format,
            },
        }));
    }

    // ─── Model Management: Load ─────────────────────────────────────

    private static Task<IResult> HandleLoad(ModelLoadRequest request)
    {
        if (string.IsNullOrWhiteSpace(request.Name))
            return Task.FromResult(Results.BadRequest(new { error = "Missing 'name' field" }));

        string? modelPath = ResolveModelPath(request.Name);
        if (modelPath == null)
        {
            return Task.FromResult(Results.NotFound(new
            {
                error = $"Model '{request.Name}' not found in {Path.GetFullPath(_modelsDir)}. "
                      + "Use POST /api/pull to download it first."
            }));
        }

        Console.WriteLine($"Loading model: {modelPath}");
        bool success = LoadModel(modelPath);
        if (success)
        {
            var engine = GetEngine();
            return Task.FromResult(Results.Ok(new
            {
                status = "success",
                model = engine?.Config?.ToString() ?? modelPath,
                model_path = modelPath,
            }));
        }
        else
        {
            return Task.FromResult(Results.Json(
                new { error = $"Failed to load model from {modelPath}" },
                statusCode: 500));
        }
    }

    // ─── Model Management: Orphans ──────────────────────────────────

    private static IResult HandleListOrphans()
    {
        if (_orphanFinder == null)
            return Results.Ok(new OrphanedModelsResponse());

        var orphaned = _orphanFinder.FindOrphaned();
        return Results.Ok(new OrphanedModelsResponse
        {
            OrphanedCount = orphaned.Count,
            TotalSizeBytes = orphaned.Sum(m => m.SizeBytes),
            Models = orphaned.Select(m => new OrphanedModelInfo
            {
                Name = m.Name,
                Path = m.Path,
                Format = m.Format,
                SizeBytes = m.SizeBytes,
                CreatedAt = m.CreatedAt.ToString("o"),
            }).ToList(),
        });
    }

    private static Task<IResult> HandleCleanupOrphans(OrphanCleanupRequest request)
    {
        if (_orphanFinder == null)
            return Task.FromResult(Results.Json(new { error = "Orphan finder not initialized" }, statusCode: 500));

        var result = _orphanFinder.Cleanup(request.DryRun, request.Models);

        return Task.FromResult(Results.Ok(new OrphanCleanupResponse
        {
            DeletedCount = result.DeletedCount,
            BytesFreed = result.BytesFreed,
            DeletedModels = result.DeletedPaths,
            Errors = result.Errors,
            DryRun = result.DryRun,
        }));
    }

    // ─── OpenAI: Chat Completions ───────────────────────────────────

    private static async Task HandleChatCompletion(ChatCompletionRequest request, HttpContext http)
    {
        var engine = GetEngine();
        if (engine == null)
        {
            http.Response.StatusCode = 503;
            await http.Response.WriteAsJsonAsync(new
            {
                error = new
                {
                    message = "No model loaded. Use POST /api/pull to download a model and POST /api/load to activate it.",
                    type = "server_error",
                    code = "model_not_loaded"
                }
            });
            return;
        }

        // Build prompt from messages
        string prompt = BuildPrompt(request.Messages);

        var genConfig = new GenerationConfig
        {
            Temperature = request.Temperature,
            TopP = request.TopP,
            MaxTokens = request.MaxTokens ?? 512,
            RepetitionPenalty = request.RepetitionPenalty,
            StopStrings = request.Stop ?? new(),
        };

        if (request.Stream)
        {
            await HandleStreaming(http, engine, prompt, genConfig, request.Model);
        }
        else
        {
            await HandleNonStreaming(http, engine, prompt, genConfig, request.Model);
        }

        // Reset context for next request (stateless API)
        engine.ResetContext();
    }

    private static async Task HandleNonStreaming(HttpContext http, InferenceEngine engine,
        string prompt, GenerationConfig config, string model)
    {
        string response = await engine.GenerateCompleteAsync(prompt, config);

        var result = new ChatCompletionResponse
        {
            Model = model,
            Choices = new List<ChatChoice>
            {
                new()
                {
                    Index = 0,
                    Message = new ChatMessage { Role = "assistant", Content = response },
                    FinishReason = "stop"
                }
            },
            Usage = new UsageInfo
            {
                PromptTokens = engine.Tokenize(prompt).Count,
                CompletionTokens = engine.Tokenize(response).Count,
                TotalTokens = engine.Tokenize(prompt).Count + engine.Tokenize(response).Count,
            }
        };

        await http.Response.WriteAsJsonAsync(result);
    }

    private static async Task HandleStreaming(HttpContext http, InferenceEngine engine,
        string prompt, GenerationConfig config, string model)
    {
        http.Response.ContentType = "text/event-stream";
        http.Response.Headers.CacheControl = "no-cache";
        http.Response.Headers.Connection = "keep-alive";

        string completionId = $"chatcmpl-{Guid.NewGuid():N}";

        // Send initial chunk with role
        var initialChunk = new ChatCompletionChunk
        {
            Id = completionId,
            Model = model,
            Choices = new List<StreamChoice>
            {
                new() { Delta = new ChatMessageDelta { Role = "assistant" } }
            }
        };
        await WriteSSE(http, initialChunk);

        // Stream tokens
        await foreach (var token in engine.GenerateAsync(prompt, config))
        {
            var chunk = new ChatCompletionChunk
            {
                Id = completionId,
                Model = model,
                Choices = new List<StreamChoice>
                {
                    new() { Delta = new ChatMessageDelta { Content = token } }
                }
            };
            await WriteSSE(http, chunk);
        }

        // Send final chunk
        var finalChunk = new ChatCompletionChunk
        {
            Id = completionId,
            Model = model,
            Choices = new List<StreamChoice>
            {
                new() { Delta = new ChatMessageDelta(), FinishReason = "stop" }
            }
        };
        await WriteSSE(http, finalChunk);

        await http.Response.WriteAsync("data: [DONE]\n\n");
        await http.Response.Body.FlushAsync();
    }

    // ─── Helpers ─────────────────────────────────────────────────────

    private static async Task WriteSSE<T>(HttpContext http, T data)
    {
        var json = JsonSerializer.Serialize(data);
        await http.Response.WriteAsync($"data: {json}\n\n");
        await http.Response.Body.FlushAsync();
    }

    private static async Task WriteNdjson<T>(HttpContext http, T data)
    {
        var json = JsonSerializer.Serialize(data, _ndjsonOptions);
        await http.Response.WriteAsync($"{json}\n");
        await http.Response.Body.FlushAsync();
    }

    /// <summary>
    /// Build a prompt string from chat messages.
    /// Uses a generic chat template — extend for model-specific templates.
    /// </summary>
    private static string BuildPrompt(List<ChatMessage> messages)
    {
        var sb = new System.Text.StringBuilder();
        foreach (var msg in messages)
        {
            sb.AppendLine($"<|{msg.Role}|>");
            sb.AppendLine(msg.Content);
            sb.AppendLine("<|end|>");
        }
        sb.Append("<|assistant|>");
        return sb.ToString();
    }
}
