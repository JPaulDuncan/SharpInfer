namespace SharpInfer.Core.Layers;

/// <summary>
/// Rotary Position Embeddings (RoPE).
/// Encodes positional information by rotating pairs of query/key dimensions
/// using sinusoidal functions. This is the position encoding used in LLaMA,
/// Mistral, and most modern open-weight models.
///
/// Reference: https://arxiv.org/abs/2104.09864
/// </summary>
public static class RotaryEmbedding
{
    /// <summary>
    /// Apply RoPE to query and key vectors at a given position.
    /// Modifies q and k in-place.
    /// </summary>
    /// <param name="q">Query vector for this position [numHeads * headDim]</param>
    /// <param name="k">Key vector for this position [numKvHeads * headDim]</param>
    /// <param name="position">Absolute position in the sequence</param>
    /// <param name="headDim">Dimension of each attention head</param>
    /// <param name="numHeads">Number of query heads</param>
    /// <param name="numKvHeads">Number of key-value heads</param>
    /// <param name="theta">Base frequency (default 10000)</param>
    public static void Apply(Span<float> q, Span<float> k,
        int position, int headDim, int numHeads, int numKvHeads, float theta = 10000f)
    {
        // Apply to each query head
        for (int h = 0; h < numHeads; h++)
        {
            var qHead = q.Slice(h * headDim, headDim);
            ApplyToHead(qHead, position, headDim, theta);
        }

        // Apply to each key head
        for (int h = 0; h < numKvHeads; h++)
        {
            var kHead = k.Slice(h * headDim, headDim);
            ApplyToHead(kHead, position, headDim, theta);
        }
    }

    private static void ApplyToHead(Span<float> head, int position, int headDim, float theta)
    {
        // Process pairs of dimensions
        for (int i = 0; i < headDim; i += 2)
        {
            // Frequency for this dimension pair
            float freq = 1.0f / MathF.Pow(theta, (float)i / headDim);
            float angle = position * freq;

            float cos = MathF.Cos(angle);
            float sin = MathF.Sin(angle);

            float x0 = head[i];
            float x1 = head[i + 1];

            // Rotation matrix application
            head[i]     = x0 * cos - x1 * sin;
            head[i + 1] = x0 * sin + x1 * cos;
        }
    }

    /// <summary>
    /// Precompute the RoPE frequency table for a given sequence length.
    /// Use this to amortize the trig computations when processing full prompts.
    /// </summary>
    public static (float[] Cos, float[] Sin) PrecomputeFreqs(int maxSeqLen, int headDim, float theta = 10000f)
    {
        int tableSize = maxSeqLen * (headDim / 2);
        var cos = new float[tableSize];
        var sin = new float[tableSize];

        for (int pos = 0; pos < maxSeqLen; pos++)
        {
            for (int i = 0; i < headDim / 2; i++)
            {
                float freq = 1.0f / MathF.Pow(theta, (float)(2 * i) / headDim);
                float angle = pos * freq;
                int idx = pos * (headDim / 2) + i;
                cos[idx] = MathF.Cos(angle);
                sin[idx] = MathF.Sin(angle);
            }
        }

        return (cos, sin);
    }
}
