namespace SharpInfer.Core.Sampling;

/// <summary>
/// Configuration for text generation.
/// </summary>
public class GenerationConfig
{
    public int MaxTokens { get; set; } = 512;
    public float Temperature { get; set; } = 0.7f;
    public float TopP { get; set; } = 0.9f;
    public int TopK { get; set; } = 40;
    public float RepetitionPenalty { get; set; } = 1.1f;
    public int Seed { get; set; } = -1; // -1 = random
    public List<int> StopTokenIds { get; set; } = new();
    public List<string> StopStrings { get; set; } = new();
}

/// <summary>
/// Composable sampling pipeline. Applies a chain of transforms to logits
/// then samples a token ID. Transforms run in order:
/// 1. Repetition penalty
/// 2. Temperature scaling
/// 3. Top-K filtering
/// 4. Top-P (nucleus) filtering
/// 5. Categorical sampling from the resulting distribution
///
/// All intermediate buffers are pre-allocated at construction time to avoid
/// per-token heap pressure during generation.
/// </summary>
public class SamplingPipeline
{
    private readonly Random _rng;

    // Pre-allocated scratch buffers — sized to vocabSize at construction.
    // Avoids allocation on every Sample() call (called once per output token).
    private readonly float[] _scores;   // Working copy of logits
    private readonly float[] _probs;    // Softmax probabilities (TopP scratch)
    private readonly int[] _indices;    // Sort-order scratch (TopK / TopP)
    private readonly bool[] _allowed;   // Nucleus membership mask (TopP)
    private readonly HashSet<int> _seen = new(); // Repetition-penalty dedup (reused via Clear())

    public SamplingPipeline(int seed, int vocabSize)
    {
        _rng = seed >= 0 ? new Random(seed) : new Random();
        _scores  = new float[vocabSize];
        _probs   = new float[vocabSize];
        _indices = new int[vocabSize];
        _allowed = new bool[vocabSize];
    }

    /// <summary>
    /// Sample the next token ID from logits using the configured strategy.
    /// </summary>
    public int Sample(ReadOnlySpan<float> logits, GenerationConfig config, List<int>? previousTokens = null)
    {
        // Copy logits into the pre-allocated working buffer
        logits.CopyTo(_scores);

        // 1. Repetition penalty
        if (previousTokens != null && config.RepetitionPenalty != 1.0f)
            ApplyRepetitionPenalty(_scores, previousTokens, config.RepetitionPenalty);

        // 2. Temperature
        if (config.Temperature <= 0f)
            return ArgMax(_scores);

        ApplyTemperature(_scores, config.Temperature);

        // 3. Top-K
        if (config.TopK > 0)
            ApplyTopK(_scores, config.TopK);

        // 4. Top-P
        if (config.TopP < 1.0f)
            ApplyTopP(_scores, config.TopP);

        // 5. Convert to probabilities and sample
        Softmax(_scores);
        return CategoricalSample(_scores);
    }

    /// <summary>Greedy selection: pick the highest-scoring token.</summary>
    private static int ArgMax(float[] scores)
    {
        int best = 0;
        for (int i = 1; i < scores.Length; i++)
            if (scores[i] > scores[best]) best = i;
        return best;
    }

    /// <summary>Divide all logits by temperature to sharpen or flatten the distribution.</summary>
    private static void ApplyTemperature(float[] scores, float temperature)
    {
        float invT = 1f / temperature;
        for (int i = 0; i < scores.Length; i++)
            scores[i] *= invT;
    }

    /// <summary>
    /// Penalize tokens that have already appeared in the sequence.
    /// Scores above 0 are divided by the penalty, scores below 0 are multiplied.
    /// Uses a reused HashSet to deduplicate repeated token IDs without allocating.
    /// </summary>
    private void ApplyRepetitionPenalty(float[] scores, List<int> previous, float penalty)
    {
        _seen.Clear();
        foreach (int id in previous)
        {
            if (id < 0 || id >= scores.Length) continue;
            if (_seen.Add(id)) // only penalise first occurrence
                scores[id] = scores[id] > 0 ? scores[id] / penalty : scores[id] * penalty;
        }
    }

    /// <summary>
    /// Keep only the top-K highest scoring tokens, set all others to -inf.
    /// Uses the pre-allocated _indices buffer for sorting.
    /// </summary>
    private void ApplyTopK(float[] scores, int k)
    {
        if (k >= scores.Length) return;

        // Fill index buffer: 0, 1, 2, …, n-1
        for (int i = 0; i < scores.Length; i++) _indices[i] = i;

        // Sort by score descending via a copy into _probs so we don't lose the
        // original order in scores (Array.Sort sorts both arrays in tandem).
        scores.CopyTo(_probs, 0);
        Array.Sort(_probs, _indices, 0, scores.Length);
        // _probs is now ascending; _indices[n-1] is the highest-scoring token.

        // Zero-out (set -inf) any token not in the top-k.
        // _indices[scores.Length - k .. scores.Length - 1] are the top-k indices.
        int cutoff = scores.Length - k;
        for (int i = 0; i < cutoff; i++)
            scores[_indices[i]] = float.NegativeInfinity;
    }

    /// <summary>
    /// Nucleus sampling: keep the smallest set of tokens whose cumulative
    /// probability exceeds topP, set all others to -inf.
    /// Uses pre-allocated _probs, _indices, _allowed buffers.
    /// </summary>
    private void ApplyTopP(float[] scores, float topP)
    {
        int n = scores.Length;

        // Compute softmax probabilities into _probs (without modifying scores)
        scores.CopyTo(_probs, 0);
        Softmax(_probs);

        // Build sorted index order by probability descending
        for (int i = 0; i < n; i++) _indices[i] = i;
        // Sort _probs ascending alongside _indices, then treat from the end
        Array.Sort(_probs, _indices, 0, n);
        // _probs[n-1] is highest prob; _indices[n-1] is its token ID

        // Find cutoff: walk from highest prob down until cumsum >= topP
        float cumSum = 0f;
        int cutoff = n; // how many from the top are "allowed"
        for (int i = n - 1; i >= 0; i--)
        {
            cumSum += _probs[i];
            if (cumSum >= topP)
            {
                cutoff = n - i; // i..n-1 are the allowed tokens
                break;
            }
        }

        // Mark allowed tokens using the bool[] array (no HashSet allocation)
        Array.Clear(_allowed, 0, n);
        for (int i = n - cutoff; i < n; i++)
            _allowed[_indices[i]] = true;

        // Mask tokens outside nucleus
        for (int i = 0; i < n; i++)
        {
            if (!_allowed[i])
                scores[i] = float.NegativeInfinity;
        }
    }

    /// <summary>In-place softmax.</summary>
    private static void Softmax(float[] x)
    {
        float max = x[0];
        for (int i = 1; i < x.Length; i++) if (x[i] > max) max = x[i];
        float sum = 0f;
        for (int i = 0; i < x.Length; i++)
        {
            x[i] = MathF.Exp(x[i] - max);
            sum += x[i];
        }
        float inv = 1f / sum;
        for (int i = 0; i < x.Length; i++)
            x[i] *= inv;
    }

    /// <summary>Sample from a categorical distribution (array of probabilities).</summary>
    private int CategoricalSample(float[] probs)
    {
        float r = (float)_rng.NextDouble();
        float cumSum = 0f;
        for (int i = 0; i < probs.Length; i++)
        {
            cumSum += probs[i];
            if (r <= cumSum) return i;
        }
        return probs.Length - 1; // Safety fallback
    }
}
