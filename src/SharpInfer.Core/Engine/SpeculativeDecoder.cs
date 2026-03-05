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

    /// <summary>
    /// Number of tokens the draft model generates per speculation round.
    /// Fixed at construction time — changing it post-construction would exceed the
    /// pre-allocated logit buffers (_draftLogitBuf / _targetLogitBuf).
    /// </summary>
    public int LookaheadTokens { get; }

    /// <summary>Running statistics for acceptance rate monitoring.</summary>
    public int TotalDrafted { get; private set; }
    public int TotalAccepted { get; private set; }
    public float AcceptanceRate => TotalDrafted > 0 ? (float)TotalAccepted / TotalDrafted : 0f;

    // Pre-allocated logit storage: [LookaheadTokens + 1][vocabSize]
    // Avoids ToArray() + new float[] on every draft/verify token.
    private readonly float[][] _draftLogitBuf;
    private readonly float[][] _targetLogitBuf;
    // Pre-allocated scratch for ComputeAdjustedDistribution
    private readonly float[] _pTarget;
    private readonly float[] _pDraft;
    private readonly float[] _adjusted;

    public SpeculativeDecoder(Transformer target, Transformer draft, int lookaheadTokens = 5, int seed = -1)
    {
        LookaheadTokens = lookaheadTokens;
        _target = target;
        _draft = draft;
        _sampler = new SamplingPipeline(seed, target.Config.VocabSize);
        _rng = seed >= 0 ? new Random(seed) : new Random();

        int vocab = target.Config.VocabSize;
        int maxLookahead = LookaheadTokens + 1; // +1 for bonus token
        _draftLogitBuf  = new float[maxLookahead][];
        _targetLogitBuf = new float[maxLookahead][];
        for (int i = 0; i < maxLookahead; i++)
        {
            _draftLogitBuf[i]  = new float[vocab];
            _targetLogitBuf[i] = new float[vocab];
        }
        _pTarget  = new float[vocab];
        _pDraft   = new float[vocab];
        _adjusted = new float[vocab];

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

        // Hoist these outside the loop — reuse across speculation rounds via Clear()
        var draftTokens = new List<int>(LookaheadTokens);
        int numDraft = 0;   // how many entries in _draftLogitBuf are valid this round
        int numTarget = 0;  // how many entries in _targetLogitBuf are valid this round

        for (int step = 0; step < config.MaxTokens;)
        {
            // --- Phase 1: Draft K tokens ---
            draftTokens.Clear();
            numDraft = 0;
            int draftToken = lastToken;

            for (int k = 0; k < LookaheadTokens && step + k < config.MaxTokens; k++)
            {
                // Copy logits directly into the pre-allocated slot (no ToArray())
                _draft.Forward(draftToken, position + k, draftCache)
                      .CopyTo(_draftLogitBuf[k]);
                numDraft = k + 1;

                draftToken = _sampler.Sample(_draftLogitBuf[k], config, generated);
                draftTokens.Add(draftToken);

                if (stopTokens.Contains(draftToken)) break;
            }

            TotalDrafted += draftTokens.Count;

            // --- Phase 2: Verify with target model ---
            numTarget = 0;
            for (int k = 0; k < draftTokens.Count; k++)
            {
                int tokenToVerify = k == 0 ? lastToken : draftTokens[k - 1];
                _target.Forward(tokenToVerify, position + k, targetCache)
                       .CopyTo(_targetLogitBuf[k]);
                numTarget = k + 1;
            }

            // Also run target on the position after the last draft token
            // (to get the next token if all drafts are accepted)
            _target.Forward(draftTokens[^1], position + draftTokens.Count, targetCache)
                   .CopyTo(_targetLogitBuf[numTarget]); // slot numTarget = bonusLogits

            // --- Phase 3: Accept/reject via rejection sampling ---
            int accepted = 0;
            for (int k = 0; k < draftTokens.Count; k++)
            {
                int candidateToken = draftTokens[k];

                // Compute acceptance probability
                float pTarget = GetTokenProbability(_targetLogitBuf[k], candidateToken, config);
                float pDraft  = GetTokenProbability(_draftLogitBuf[k],  candidateToken, config);

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
                    var adjusted = ComputeAdjustedDistribution(_targetLogitBuf[k], _draftLogitBuf[k], config);
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
                int bonusToken = _sampler.Sample(_targetLogitBuf[numTarget], config, generated);
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
    /// Writes into the pre-allocated _adjusted buffer; returns it (no allocation).
    /// </summary>
    private float[] ComputeAdjustedDistribution(float[] targetLogits, float[] draftLogits, GenerationConfig config)
    {
        int vocabSize = targetLogits.Length;
        SoftmaxWithTemp(targetLogits, config.Temperature, _pTarget);
        SoftmaxWithTemp(draftLogits,  config.Temperature, _pDraft);

        float sum = 0f;
        for (int i = 0; i < vocabSize; i++)
        {
            _adjusted[i] = MathF.Max(0f, _pTarget[i] - _pDraft[i]);
            sum += _adjusted[i];
        }

        if (sum > 0f)
        {
            float invSum = 1f / sum;
            for (int i = 0; i < vocabSize; i++)
                _adjusted[i] *= invSum;
        }
        else
        {
            // Fallback to target distribution
            _pTarget.CopyTo(_adjusted, 0);
        }

        return _adjusted;
    }

    /// <summary>
    /// Softmax with temperature, writing into a pre-allocated output buffer.
    /// No heap allocation.
    /// </summary>
    private static void SoftmaxWithTemp(float[] logits, float temperature, float[] output)
    {
        float maxVal = logits[0];
        for (int i = 1; i < logits.Length; i++) if (logits[i] > maxVal) maxVal = logits[i];

        float sum = 0f;
        float invT = temperature > 0 ? 1f / temperature : 1f;
        for (int i = 0; i < logits.Length; i++)
        {
            output[i] = MathF.Exp((logits[i] - maxVal) * invT);
            sum += output[i];
        }

        float inv = 1f / sum;
        for (int i = 0; i < logits.Length; i++)
            output[i] *= inv;
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
