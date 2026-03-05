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
/// </summary>
public class SamplingPipeline
{
    private readonly Random _rng;

    public SamplingPipeline(int seed = -1)
    {
        _rng = seed >= 0 ? new Random(seed) : new Random();
    }

    /// <summary>
    /// Sample the next token ID from logits using the configured strategy.
    /// </summary>
    public int Sample(ReadOnlySpan<float> logits, GenerationConfig config, List<int>? previousTokens = null)
    {
        // Copy logits so we can modify them
        var scores = new float[logits.Length];
        logits.CopyTo(scores);

        // 1. Repetition penalty
        if (previousTokens != null && config.RepetitionPenalty != 1.0f)
            ApplyRepetitionPenalty(scores, previousTokens, config.RepetitionPenalty);

        // 2. Temperature
        if (config.Temperature <= 0f)
            return ArgMax(scores);

        ApplyTemperature(scores, config.Temperature);

        // 3. Top-K
        if (config.TopK > 0)
            ApplyTopK(scores, config.TopK);

        // 4. Top-P
        if (config.TopP < 1.0f)
            ApplyTopP(scores, config.TopP);

        // 5. Convert to probabilities and sample
        Softmax(scores);
        return CategoricalSample(scores);
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
    /// </summary>
    private static void ApplyRepetitionPenalty(float[] scores, List<int> previous, float penalty)
    {
        var seen = new HashSet<int>(previous);
        foreach (int id in seen)
        {
            if (id < 0 || id >= scores.Length) continue;
            scores[id] = scores[id] > 0 ? scores[id] / penalty : scores[id] * penalty;
        }
    }

    /// <summary>
    /// Keep only the top-K highest scoring tokens, set all others to -inf.
    /// </summary>
    private static void ApplyTopK(float[] scores, int k)
    {
        if (k >= scores.Length) return;

        // Find the k-th largest value using partial sort
        var indices = Enumerable.Range(0, scores.Length).ToArray();
        Array.Sort(scores, indices);
        Array.Reverse(scores);
        Array.Reverse(indices);

        float threshold = scores[k - 1];

        // We sorted in-place so we need to "unsort"
        var result = new float[scores.Length];
        for (int i = 0; i < scores.Length; i++)
            result[indices[i]] = i < k ? scores[i] : float.NegativeInfinity;

        result.CopyTo(scores, 0);
    }

    /// <summary>
    /// Nucleus sampling: keep the smallest set of tokens whose cumulative
    /// probability exceeds topP, set all others to -inf.
    /// </summary>
    private static void ApplyTopP(float[] scores, float topP)
    {
        // Compute softmax to get probabilities
        var probs = new float[scores.Length];
        scores.CopyTo(probs, 0);
        Softmax(probs);

        // Sort by probability descending
        var indices = Enumerable.Range(0, probs.Length).ToArray();
        Array.Sort(probs, indices);
        Array.Reverse(probs);
        Array.Reverse(indices);

        // Find cutoff
        float cumSum = 0f;
        int cutoff = probs.Length;
        for (int i = 0; i < probs.Length; i++)
        {
            cumSum += probs[i];
            if (cumSum >= topP)
            {
                cutoff = i + 1;
                break;
            }
        }

        // Mask tokens outside nucleus
        var allowed = new HashSet<int>();
        for (int i = 0; i < cutoff; i++)
            allowed.Add(indices[i]);

        for (int i = 0; i < scores.Length; i++)
        {
            if (!allowed.Contains(i))
                scores[i] = float.NegativeInfinity;
        }
    }

    /// <summary>In-place softmax.</summary>
    private static void Softmax(float[] x)
    {
        float max = x.Max();
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
        return probs.Length - 1; // Shouldn't reach here, but safety fallback
    }
}
