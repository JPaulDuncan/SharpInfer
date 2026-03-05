using SharpInfer.Core.Layers;
using SharpInfer.Core.Models;
using SharpInfer.Core.Sampling;
using SharpInfer.Core.Tensors;
using SharpInfer.Core.Tokenizer;
using SharpInfer.Core.Tools;

namespace SharpInfer.Core.Engine;

/// <summary>
/// High-level inference engine that orchestrates model loading, tokenization,
/// generation, and tool calling. This is the main entry point for all interfaces
/// (CLI, API, VS Code extension).
/// </summary>
public class InferenceEngine : IDisposable
{
    private readonly Transformer _transformer;
    private readonly BpeTokenizer _tokenizer;
    private readonly SamplingPipeline _sampler;
    private readonly ToolRegistry _tools;
    private KVCache _kvCache;

    public ModelConfig Config => _transformer.Config;
    public ToolRegistry Tools => _tools;

    private InferenceEngine(Transformer transformer, BpeTokenizer tokenizer,
        SamplingPipeline sampler, ToolRegistry tools, KVCache kvCache)
    {
        _transformer = transformer;
        _tokenizer = tokenizer;
        _sampler = sampler;
        _tools = tools;
        _kvCache = kvCache;
    }

    /// <summary>
    /// Load a model from disk and create an inference engine.
    /// Auto-detects format (GGUF, safetensors).
    /// </summary>
    public static InferenceEngine Load(string modelPath, EngineOptions? options = null)
    {
        options ??= new EngineOptions();

        var factory = new ModelLoaderFactory();
        var (config, weights, tokenizerData) = factory.Load(modelPath, options.ProgressCallback);

        if (options.ContextLength.HasValue)
            config.MaxSequenceLength = options.ContextLength.Value;

        if (tokenizerData == null)
            throw new InvalidOperationException("Model does not contain tokenizer data. Provide a separate tokenizer.");

        IComputeBackend backend = options.ComputeBackend ?? new CpuBackend();
        options.ProgressCallback?.Invoke($"Compute backend: {backend.Name}");

        var transformer = new Transformer(config, weights, backend);
        var tokenizer = new BpeTokenizer(tokenizerData, config.BosTokenId, config.EosTokenId);
        var sampler = new SamplingPipeline(options.GenerationConfig.Seed);
        var tools = new ToolRegistry();
        var kvCache = new KVCache(config);

        return new InferenceEngine(transformer, tokenizer, sampler, tools, kvCache);
    }

    /// <summary>
    /// Generate text from a prompt, yielding tokens as they are produced.
    /// This is the core generation loop used by all interfaces.
    /// </summary>
    public async IAsyncEnumerable<string> GenerateAsync(string prompt, GenerationConfig? config = null)
    {
        config ??= new GenerationConfig();

        var tokenIds = _tokenizer.Encode(prompt, addBos: true);
        var generatedIds = new List<int>(tokenIds);

        // Prefill: process all prompt tokens to populate KV cache
        int startPos = _kvCache.CurrentLength;
        for (int i = 0; i < tokenIds.Count; i++)
        {
            _transformer.Forward(tokenIds[i], startPos + i, _kvCache);
        }

        int position = startPos + tokenIds.Count;

        // Build stop token set
        var stopTokens = new HashSet<int>(config.StopTokenIds) { _tokenizer.EosId };

        // Autoregressive generation
        int lastTokenId = tokenIds[^1];
        var generatedText = new System.Text.StringBuilder();

        for (int step = 0; step < config.MaxTokens; step++)
        {
            float[] logits = _transformer.Forward(lastTokenId, position, _kvCache).ToArray();
            int nextTokenId = _sampler.Sample(logits, config, generatedIds);

            if (stopTokens.Contains(nextTokenId))
                break;

            generatedIds.Add(nextTokenId);
            string tokenText = _tokenizer.Decode(nextTokenId);
            generatedText.Append(tokenText);

            // Check for stop strings
            bool shouldStop = false;
            foreach (var stopStr in config.StopStrings)
            {
                if (generatedText.ToString().EndsWith(stopStr))
                {
                    shouldStop = true;
                    break;
                }
            }

            yield return tokenText;

            if (shouldStop) break;

            lastTokenId = nextTokenId;
            position++;

            // Yield control to allow cancellation / streaming
            await Task.Yield();
        }
    }

    /// <summary>
    /// Generate a complete response (non-streaming).
    /// </summary>
    public async Task<string> GenerateCompleteAsync(string prompt, GenerationConfig? config = null)
    {
        var sb = new System.Text.StringBuilder();
        await foreach (var token in GenerateAsync(prompt, config))
        {
            sb.Append(token);
        }
        return sb.ToString();
    }

    /// <summary>
    /// Generate with tool calling support.
    /// If the model outputs a tool call, executes it and feeds the result back.
    /// </summary>
    public async IAsyncEnumerable<GenerationEvent> GenerateWithToolsAsync(
        string prompt, GenerationConfig? config = null, int maxToolRounds = 5)
    {
        config ??= new GenerationConfig();
        string currentPrompt = prompt;

        for (int round = 0; round < maxToolRounds; round++)
        {
            var fullResponse = new System.Text.StringBuilder();

            await foreach (var token in GenerateAsync(currentPrompt, config))
            {
                fullResponse.Append(token);
                yield return new GenerationEvent { Type = EventType.Token, Text = token };
            }

            string response = fullResponse.ToString();

            // Check if the response contains a tool call
            var toolCall = _tools.ParseToolCall(response);
            if (toolCall == null)
            {
                yield return new GenerationEvent { Type = EventType.Done, Text = response };
                yield break;
            }

            yield return new GenerationEvent
            {
                Type = EventType.ToolCall,
                Text = $"Calling tool: {toolCall.ToolName}({toolCall.Arguments})"
            };

            // Execute the tool
            var result = await _tools.ExecuteAsync(toolCall);

            yield return new GenerationEvent
            {
                Type = EventType.ToolResult,
                Text = result
            };

            // Feed tool result back into the conversation
            currentPrompt = $"{currentPrompt}\n{response}\n[Tool Result: {result}]\n";

            // Reset KV cache for the new prompt
            ResetContext();
        }
    }

    /// <summary>Reset the KV cache (start a new conversation).</summary>
    public void ResetContext()
    {
        _kvCache.Clear();
    }

    /// <summary>Tokenize text to token IDs (useful for debugging).</summary>
    public List<int> Tokenize(string text) => _tokenizer.Encode(text);

    /// <summary>Decode token IDs back to text.</summary>
    public string Detokenize(IEnumerable<int> ids) => _tokenizer.Decode(ids);

    public void Dispose()
    {
        _kvCache.Dispose();
        GC.SuppressFinalize(this);
    }
}

/// <summary>Options for creating an InferenceEngine.</summary>
public class EngineOptions
{
    public IComputeBackend? ComputeBackend { get; set; }
    public int? ContextLength { get; set; }
    public GenerationConfig GenerationConfig { get; set; } = new();
    public Action<string>? ProgressCallback { get; set; }
}

/// <summary>Events emitted during generation (for streaming + tool use).</summary>
public class GenerationEvent
{
    public EventType Type { get; set; }
    public string Text { get; set; } = "";
}

public enum EventType
{
    Token,
    ToolCall,
    ToolResult,
    Done
}
