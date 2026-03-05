using SharpInfer.Core.Layers;
using SharpInfer.Core.Sampling;
using SharpInfer.Core.Tensors;

namespace SharpInfer.Core.Engine;

/// <summary>
/// Speculative decoding: use a fast draft model to generate candidate tokens,
/// then verify them in a single pass through the target model.
///
/// The key insight is that the target model can verify N draft tokens in roughly
/// the same time as generating 1 token, because verification is essentially a
/// prefill operation (all tokens processed in parallel).
///
/// Algorithm:
/// 1. Draft model generates K candidate tokens autoregressively
/// 2. Target model runs a single forward pass over all K candidates
/// 3. Compare draft vs target distributions at each position
/// 4. Accept tokens where draft and target agree (rejection sampling)
/// 5. On first rejection, sample from the adjusted distribution
/// 6. Repeat from the accepted prefix
///
/// Expected speedup: 2-3x for well-matched draft/target pairs.
///
/// Reference: https://arxiv.org/abs/2302.01318 (Leviathan et al.)
/// </summary>
public class SpeculativeDecoder
{
    private readonly Transformer _target;
    private readonly Transformer _draft;
    private readonly SamplingPipeline _sampler;
    private readonly Random _rng;

    /// <summary>Number of tokens the draft model generates per speculation round.</summary>
    public int LookaheadTokens { get; set; } = 5;

    /// <summary>Running statistics for acceptance rate monitoring.</summary>
    public int TotalDrafted { get; private set; }
    public int TotalAccepted { get; private set; }
    public float AcceptanceRate => TotalDrafted > 0 ? (float)TotalAccepted / TotalDrafted : 0f;

    public SpeculativeDecoder(Transformer target, Transformer draft, int seed = -1)
    {
        _target = target;
        _draft = draft;
        _sampler = new SamplingPipeline(seed);
        _rng = seed >= 0 ? new Random(seed) : new Random();

        if (draft.Config.VocabSize != target.Config.VocabSize)
            throw new ArgumentException("Draft and target models must have the same vocabulary size.");
    }

    /// <summary>
    /// Generate tokens using speculative decoding.
    /// Yields accepted tokens as they are verified.
    /// </summary>
    public async IAsyncEnumerable<int> GenerateAsync(
        List<int> promptTokens, GenerationConfig config, KVCache targetCache, KVCache draftCache)
    {
        var generated = new List<int>(promptTokens);
        int position = 0;

        // Prefill both models with the prompt
        for (int i = 0; i < promptTokens.Count; i++)
        {
            _target.Forward(promptTokens[i], position + i, targetCache);
            _draft.Forward(promptTokens[i], position + i, draftCache);
        }
        position = promptTokens.Count;

        int lastToken = promptTokens[^1];
        var stopTokens = new HashSet<int>(config.StopTokenIds) { _target.Config.EosTokenId };

        for (int step = 0; step < config.MaxTokens;)
        {
            // --- Phase 1: Draft K tokens ---
            var draftTokens = new List<int>();
            var draftLogits = new List<float[]>();
            int draftToken = lastToken;

            for (int k = 0; k < LookaheadTokens && step + k < config.MaxTokens; k++)
            {
                float[] logits = _draft.Forward(draftToken, position + k, draftCache).ToArray();
                draftLogits.Add(logits);

                draftToken = _sampler.Sample(logits, config, generated);
                draftTokens.Add(draftToken);

                if (stopTokens.Contains(draftToken)) break;
            }

            TotalDrafted += draftTokens.Count;

            // --- Phase 2: Verify with target model ---
            // Run target model on each draft token position
            var targetLogits = new List<float[]>();
            for (int k = 0; k < draftTokens.Count; k++)
            {
                int tokenToVerify = k == 0 ? lastToken : draftTokens[k - 1];
                float[] logits = _target.Forward(tokenToVerify, position + k, targetCache).ToArray();
                targetLogits.Add(logits);
            }

            // Also run target on the position after the last draft token
            // (to get the next token if all drafts are accepted)
            float[] bonusLogits = _target.Forward(draftTokens[^1], position + draftTokens.Count, targetCache).ToArray();

            // --- Phase 3: Accept/reject via rejection sampling ---
            int accepted = 0;
            for (int k = 0; k < draftTokens.Count; k++)
            {
                int candidateToken = draftTokens[k];

                // Compute acceptance probability
                float pTarget = GetTokenProbability(targetLogits[k], candidateToken, config);
                float pDraft = GetTokenProbability(draftLogits[k], candidateToken, config);

                float acceptProb = Math.Min(1f, pTarget / Math.Max(pDraft, 1e-10f));

                if ((float)_rng.NextDouble() < acceptProb)
                {
                    // Accept this draft token
                    generated.Add(candidateToken);
                    yield return candidateToken;
                    accepted++;
                    step++;

                    if (stopTokens.Contains(candidateToken))
                        yield break;
                }
                else
                {
                    // Reject: sample from adjusted distribution (target - draft)
                    var adjusted = ComputeAdjustedDistribution(targetLogits[k], draftLogits[k], config);
                    int correctedToken = SampleFromDistribution(adjusted);

                    generated.Add(correctedToken);
                    yield return correctedToken;
                    step++;

                    if (stopTokens.Contains(correctedToken))
                        yield break;

                    // Rollback draft cache to this position
                    // (target cache is fine since we verified up to here)
                    break;
                }
            }

            TotalAccepted += accepted;

            if (accepted == draftTokens.Count)
            {
                // All drafts accepted — bonus: sample one more from target
                int bonusToken = _sampler.Sample(bonusLogits, config, generated);
                generated.Add(bonusToken);
                yield return bonusToken;
                step++;

                if (stopTokens.Contains(bonusToken))
                    yield break;

                lastToken = bonusToken;
                position += accepted + 1;
            }
            else
            {
                // Partial acceptance — resume from the corrected position
                lastToken = generated[^1];
                position += accepted + 1;

                // Reset draft cache to match target position
                // (simplified: in production, you'd surgically truncate the cache)
                RebuildDraftCache(draftCache, generated, position);
            }

            await Task.Yield();
        }
    }

    /// <summary>Get the probability of a specific token after temperature scaling + softmax.</summary>
    private float GetTokenProbability(float[] logits, int tokenId, GenerationConfig config)
    {
        if (config.Temperature <= 0f)
            return tokenId == ArgMax(logits) ? 1f : 0f;

        float maxLogit = logits.Max();
        float sum = 0f;
        float tokenExp = 0f;

        for (int i = 0; i < logits.Length; i++)
        {
            float exp = MathF.Exp((logits[i] - maxLogit) / config.Temperature);
            sum += exp;
            if (i == tokenId) tokenExp = exp;
        }

        return tokenExp / sum;
    }

    /// <summary>
    /// Compute max(0, p_target - p_draft) normalized to a valid distribution.
    /// This is the "residual" distribution used when a draft token is rejected.
    /// </summary>
    private float[] ComputeAdjustedDistribution(float[] targetLogits, float[] draftLogits, GenerationConfig config)
    {
        int vocabSize = targetLogits.Length;
        var pTarget = SoftmaxWithTemp(targetLogits, config.Temperature);
        var pDraft = SoftmaxWithTemp(draftLogits, config.Temperature);

        var adjusted = new float[vocabSize];
        float sum = 0f;

        for (int i = 0; i < vocabSize; i++)
        {
            adjusted[i] = MathF.Max(0f, pTarget[i] - pDraft[i]);
            sum += adjusted[i];
        }

        if (sum > 0f)
        {
            float invSum = 1f / sum;
            for (int i = 0; i < vocabSize; i++)
                adjusted[i] *= invSum;
        }
        else
        {
            // Fallback to target distribution
            pTarget.CopyTo(adjusted, 0);
        }

        return adjusted;
    }

    private float[] SoftmaxWithTemp(float[] logits, float temperature)
    {
        var result = new float[logits.Length];
        float maxVal = logits.Max();
        float sum = 0f;
        float invT = temperature > 0 ? 1f / temperature : 1f;

        for (int i = 0; i < logits.Length; i++)
        {
            result[i] = MathF.Exp((logits[i] - maxVal) * invT);
            sum += result[i];
        }

        float inv = 1f / sum;
        for (int i = 0; i < logits.Length; i++)
            result[i] *= inv;

        return result;
    }

    private int SampleFromDistribution(float[] probs)
    {
        float r = (float)_rng.NextDouble();
        float cum = 0f;
        for (int i = 0; i < probs.Length; i++)
        {
            cum += probs[i];
            if (r <= cum) return i;
        }
        return probs.Length - 1;
    }

    private static int ArgMax(float[] arr)
    {
        int best = 0;
        for (int i = 1; i < arr.Length; i++)
            if (arr[i] > arr[best]) best = i;
        return best;
    }

    private void RebuildDraftCache(KVCache draftCache, List<int> tokens, int upToPosition)
    {
        draftCache.Clear();
        for (int i = 0; i < upToPosition && i < tokens.Count; i++)
        {
            _draft.Forward(tokens[i], i, draftCache);
        }
    }
}
