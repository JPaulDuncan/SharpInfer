using SharpInfer.Core.Models;
using SharpInfer.Core.Tensors;
using SharpInfer.Core.Engine;

namespace SharpInfer.Core.Layers;

/// <summary>
/// Complete transformer model implementing the forward pass.
/// Supports the LLaMA/Mistral architecture: pre-norm, SwiGLU FFN, RoPE, GQA.
///
/// This is the central computation: given a token and position, produce logits
/// over the vocabulary for the next token.
/// </summary>
public class Transformer
{
    private readonly ModelConfig _config;
    private readonly ModelWeights _weights;
    private readonly IComputeBackend _compute;

    // Reusable scratch buffers (avoids allocation during inference)
    private readonly float[] _x;        // Current hidden state [hiddenSize]
    private readonly float[] _xNorm;    // Normalized hidden state
    private readonly float[] _q;        // Query projection
    private readonly float[] _k;        // Key projection
    private readonly float[] _v;        // Value projection
    private readonly float[] _attnOut;  // Attention output
    private readonly float[] _ffnGate;  // FFN gate projection
    private readonly float[] _ffnUp;    // FFN up projection
    private readonly float[] _ffnDown;  // FFN down projection
    private readonly float[] _logits;   // Output logits [vocabSize]

    public ModelConfig Config => _config;

    public Transformer(ModelConfig config, ModelWeights weights, IComputeBackend? compute = null)
    {
        _config = config;
        _weights = weights;
        _compute = compute ?? new CpuBackend();

        // Pre-allocate all scratch buffers
        int h = config.HiddenSize;
        int inter = config.IntermediateSize;
        _x = new float[h];
        _xNorm = new float[h];
        _q = new float[config.NumAttentionHeads * config.HeadDim];
        _k = new float[config.NumKvHeads * config.HeadDim];
        _v = new float[config.NumKvHeads * config.HeadDim];
        _attnOut = new float[h];
        _ffnGate = new float[inter];
        _ffnUp = new float[inter];
        _ffnDown = new float[h];
        _logits = new float[config.VocabSize];
    }

    /// <summary>
    /// Run the forward pass for a single token at a given position.
    /// Returns logits over the vocabulary.
    /// </summary>
    /// <param name="tokenId">Input token ID</param>
    /// <param name="position">Position in the sequence (0-indexed)</param>
    /// <param name="kvCache">KV cache for attention</param>
    /// <returns>Logits array [vocabSize] — do NOT modify, shared buffer.</returns>
    public ReadOnlySpan<float> Forward(int tokenId, int position, KVCache kvCache)
    {
        int h = _config.HiddenSize;
        int headDim = _config.HeadDim;
        int numHeads = _config.NumAttentionHeads;
        int numKvHeads = _config.NumKvHeads;
        int kvGroups = _config.NumKvGroups;

        // --- Token Embedding ---
        _weights.TokenEmbedding.AsSpan(tokenId * h, h).CopyTo(_x);

        // --- Transformer Layers ---
        for (int layer = 0; layer < _config.NumLayers; layer++)
        {
            var lw = _weights.Layers[layer];

            // === Self-Attention ===

            // Pre-norm
            _compute.RmsNorm(_xNorm, _x, lw.AttnNorm, _config.RmsNormEps);

            // QKV projections
            _compute.MatVecMul(lw.Wq, _xNorm, _q, numHeads * headDim, h);
            _compute.MatVecMul(lw.Wk, _xNorm, _k, numKvHeads * headDim, h);
            _compute.MatVecMul(lw.Wv, _xNorm, _v, numKvHeads * headDim, h);

            // Apply rotary position embeddings
            RotaryEmbedding.Apply(_q, _k, position, headDim, numHeads, numKvHeads, _config.RopeTheta);

            // Store K, V in cache for this layer and position
            kvCache.Store(layer, position, _k, _v);

            // Multi-head attention with grouped-query attention (GQA)
            Array.Clear(_attnOut);
            float scale = 1f / MathF.Sqrt(headDim);

            for (int head = 0; head < numHeads; head++)
            {
                int kvHead = head / kvGroups; // GQA: multiple query heads share one KV head
                var qHead = _q.AsSpan(head * headDim, headDim);

                // Compute attention scores for all cached positions
                Span<float> scores = stackalloc float[position + 1];
                for (int t = 0; t <= position; t++)
                {
                    var cachedK = kvCache.GetKey(layer, t, kvHead, headDim);
                    scores[t] = TensorOps.Dot(qHead, cachedK) * scale;
                }

                // Softmax over scores
                TensorOps.Softmax(scores);

                // Weighted sum of values
                var outHead = _attnOut.AsSpan(head * headDim, headDim);
                for (int t = 0; t <= position; t++)
                {
                    var cachedV = kvCache.GetValue(layer, t, kvHead, headDim);
                    TensorOps.AddScaled(outHead, cachedV, scores[t]);
                }
            }

            // Output projection
            float[] attnProjected = new float[h];
            _compute.MatVecMul(lw.Wo, _attnOut, attnProjected, h, h);

            // Residual connection
            _compute.AddInPlace(_x, attnProjected);

            // === Feed-Forward Network (SwiGLU) ===

            // Pre-norm
            _compute.RmsNorm(_xNorm, _x, lw.FfnNorm, _config.RmsNormEps);

            // Gate and up projections
            _compute.MatVecMul(lw.WGate, _xNorm, _ffnGate, _config.IntermediateSize, h);
            _compute.MatVecMul(lw.WUp, _xNorm, _ffnUp, _config.IntermediateSize, h);

            // SiLU activation on gate, then element-wise multiply with up
            _compute.SiLU(_ffnGate);
            TensorOps.MulInPlace(_ffnGate, _ffnUp);

            // Down projection
            _compute.MatVecMul(lw.WDown, _ffnGate, _ffnDown, h, _config.IntermediateSize);

            // Residual connection
            _compute.AddInPlace(_x, _ffnDown);
        }

        // --- Final Norm + LM Head ---
        _compute.RmsNorm(_xNorm, _x, _weights.FinalNorm, _config.RmsNormEps);
        _compute.MatVecMul(_weights.OutputWeight, _xNorm, _logits, _config.VocabSize, h);

        return _logits;
    }
}
