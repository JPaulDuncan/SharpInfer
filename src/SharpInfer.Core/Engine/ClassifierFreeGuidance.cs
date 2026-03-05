using SharpInfer.Core.Layers;
using SharpInfer.Core.Sampling;

namespace SharpInfer.Core.Engine;

/// <summary>
/// Classifier-Free Guidance (CFG) for language models.
///
/// Runs the model twice per token:
/// 1. Conditional pass: with the full prompt (system + user instructions)
/// 2. Unconditional pass: with a minimal/empty prompt
///
/// The final logits are computed as:
///   logits_guided = logits_uncond + guidance_scale * (logits_cond - logits_uncond)
///
/// This amplifies the model's adherence to the prompt. Higher guidance scales
/// produce more instruction-following output at the cost of diversity.
///
/// Typical values:
///   1.0 = no guidance (standard generation)
///   1.5 = mild guidance (recommended default)
///   2.0-3.0 = strong guidance (very instruction-focused)
///   >3.0 = may cause repetition or degraded quality
///
/// Cost: 2x compute per token (two forward passes).
///
/// Reference: https://arxiv.org/abs/2306.17806
/// </summary>
public class ClassifierFreeGuidance
{
    private readonly Transformer _model;
    private readonly SamplingPipeline _sampler;

    // Pre-allocated per-token scratch buffers (avoids 3 × float[vocabSize] per token)
    private readonly float[] _condLogits;
    private readonly float[] _uncondLogits;
    private readonly float[] _guidedLogits;

    /// <summary>Guidance scale. 1.0 = disabled, >1.0 = stronger instruction following.</summary>
    public float GuidanceScale { get; set; } = 1.5f;

    /// <summary>The unconditional prompt (typically empty or a minimal "assistant" prefix).</summary>
    public string NegativePrompt { get; set; } = "";

    public ClassifierFreeGuidance(Transformer model, float guidanceScale = 1.5f, int seed = -1)
    {
        _model = model;
        GuidanceScale = guidanceScale;
        _sampler = new SamplingPipeline(seed, model.Config.VocabSize);

        int vocab = model.Config.VocabSize;
        _condLogits   = new float[vocab];
        _uncondLogits = new float[vocab];
        _guidedLogits = new float[vocab];
    }

    /// <summary>
    /// Generate tokens with classifier-free guidance.
    /// Requires two KV caches: one for the conditional pass, one for the unconditional.
    /// </summary>
    public async IAsyncEnumerable<string> GenerateAsync(
        List<int> condTokens, List<int> uncondTokens,
        KVCache condCache, KVCache uncondCache,
        GenerationConfig config,
        Func<int, string> detokenize)
    {
        var generated = new List<int>(condTokens);
        var stopTokens = new HashSet<int>(config.StopTokenIds) { _model.Config.EosTokenId };

        // Prefill both caches
        for (int i = 0; i < condTokens.Count; i++)
            _model.Forward(condTokens[i], i, condCache);

        for (int i = 0; i < uncondTokens.Count; i++)
            _model.Forward(uncondTokens[i], i, uncondCache);

        int condPos = condTokens.Count;
        int uncondPos = uncondTokens.Count;
        int lastCondToken = condTokens[^1];
        int lastUncondToken = uncondTokens[^1];

        for (int step = 0; step < config.MaxTokens; step++)
        {
            // Conditional forward pass (full prompt) — copy into pre-allocated buffer
            _model.Forward(lastCondToken, condPos, condCache).CopyTo(_condLogits);

            // Unconditional forward pass (empty/negative prompt)
            _model.Forward(lastUncondToken, uncondPos, uncondCache).CopyTo(_uncondLogits);

            // Apply guidance: guided = uncond + scale * (cond - uncond)
            for (int i = 0; i < _guidedLogits.Length; i++)
                _guidedLogits[i] = _uncondLogits[i] + GuidanceScale * (_condLogits[i] - _uncondLogits[i]);

            // Sample from guided logits
            int nextToken = _sampler.Sample(_guidedLogits, config, generated);

            if (stopTokens.Contains(nextToken))
                yield break;

            generated.Add(nextToken);
            yield return detokenize(nextToken);

            // Both caches advance with the same generated token
            lastCondToken = nextToken;
            lastUncondToken = nextToken;
            condPos++;
            uncondPos++;

            await Task.Yield();
        }
    }
}
