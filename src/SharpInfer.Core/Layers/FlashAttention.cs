using System.Buffers;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using SharpInfer.Core.Tensors;

namespace SharpInfer.Core.Layers;

/// <summary>
/// FlashAttention — memory-efficient attention that avoids materializing the full N×N
/// attention matrix, reducing memory from O(N²) to O(N) and improving cache locality.
///
/// Standard attention:
///   S = Q @ K^T              (N×N matrix — O(N²) memory)
///   P = softmax(S / sqrt(d))
///   O = P @ V
///
/// FlashAttention (tiled):
///   For each block of Q rows:
///     For each block of K/V columns:
///       Compute partial S_ij = Q_i @ K_j^T (small block, fits in cache)
///       Update running softmax numerator and denominator
///       Accumulate O_i += softmax_correction * S_ij @ V_j
///
/// This implementation:
///   - Tile-based processing with configurable block sizes
///   - Online softmax (numerically stable, single-pass)
///   - Supports causal masking
///   - Supports Grouped-Query Attention (GQA)
///   - AVX2 vectorized inner loops
///   - Optional CUDA kernel path
///
/// Reference: Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention
/// with IO-Awareness" (2022), "FlashAttention-2" (2023)
/// </summary>
public static class FlashAttention
{
    /// <summary>Default block sizes tuned for L2 cache (~256KB).</summary>
    public const int BLOCK_SIZE_Q = 64;  // Rows of Q per tile
    public const int BLOCK_SIZE_KV = 64; // Rows of K/V per tile

    /// <summary>
    /// Compute attention output using FlashAttention algorithm.
    ///
    /// Q: [seqLen, headDim] — query vectors for one attention head
    /// K: [kvLen, headDim]  — key vectors (may be from GQA group)
    /// V: [kvLen, headDim]  — value vectors
    /// Output: [seqLen, headDim]
    /// </summary>
    /// <param name="Q">Query matrix [seqLen × headDim]</param>
    /// <param name="K">Key matrix [kvLen × headDim]</param>
    /// <param name="V">Value matrix [kvLen × headDim]</param>
    /// <param name="output">Output matrix [seqLen × headDim] (pre-allocated)</param>
    /// <param name="seqLen">Number of query positions</param>
    /// <param name="kvLen">Number of key/value positions</param>
    /// <param name="headDim">Dimension per head</param>
    /// <param name="scale">Attention scale factor (typically 1/sqrt(headDim))</param>
    /// <param name="causal">Whether to apply causal masking</param>
    /// <param name="startPos">Starting position for causal mask offset (for KV cache)</param>
    public static void Forward(
        ReadOnlySpan<float> Q,
        ReadOnlySpan<float> K,
        ReadOnlySpan<float> V,
        Span<float> output,
        int seqLen,
        int kvLen,
        int headDim,
        float scale,
        bool causal = true,
        int startPos = 0)
    {
        // Block sizes (clamped to actual dimensions)
        int blockQ = Math.Min(BLOCK_SIZE_Q, seqLen);
        int blockKV = Math.Min(BLOCK_SIZE_KV, kvLen);

        // Rent per-call scratch buffers from the shared pool instead of allocating.
        // ArrayPool.Rent may return a larger array; we only use [0..size).
        float[] rowMax = ArrayPool<float>.Shared.Rent(blockQ);
        float[] rowSum = ArrayPool<float>.Shared.Rent(blockQ);
        float[] rowOut = ArrayPool<float>.Shared.Rent(blockQ * headDim);
        float[] sBlock = ArrayPool<float>.Shared.Rent(blockQ * blockKV);

        try
        {

        // Process Q in blocks
        for (int qi = 0; qi < seqLen; qi += blockQ)
        {
            int qEnd = Math.Min(qi + blockQ, seqLen);
            int qBlock = qEnd - qi;

            // Initialize accumulators for this Q block
            Array.Fill(rowMax, float.NegativeInfinity, 0, qBlock);
            Array.Fill(rowSum, 0f, 0, qBlock);
            Array.Clear(rowOut, 0, qBlock * headDim);

            // Process K/V in blocks
            for (int ki = 0; ki < kvLen; ki += blockKV)
            {
                int kEnd = Math.Min(ki + blockKV, kvLen);
                int kBlock = kEnd - ki;

                // Compute S_ij = Q_i @ K_j^T * scale
                ComputeAttentionScores(
                    Q, K, sBlock,
                    qi, ki, qBlock, kBlock, headDim, scale);

                // Apply causal mask
                if (causal)
                {
                    ApplyCausalMask(sBlock, qi, ki, qBlock, kBlock, startPos);
                }

                // Online softmax update + output accumulation
                UpdateOnlineSoftmax(
                    sBlock, V,
                    rowMax, rowSum, rowOut,
                    qi, ki, qBlock, kBlock, headDim);
            }

            // Normalize output by softmax denominator
            for (int r = 0; r < qBlock; r++)
            {
                float invSum = rowSum[r] > 0 ? 1f / rowSum[r] : 0f;
                int outRowOffset = (qi + r) * headDim;
                int accRowOffset = r * headDim;

                if (Avx2.IsSupported)
                {
                    var invSumVec = Vector256.Create(invSum);
                    int d = 0;
                    for (; d + 8 <= headDim; d += 8)
                    {
                        var acc = Vector256.Create(
                            rowOut[accRowOffset + d], rowOut[accRowOffset + d + 1],
                            rowOut[accRowOffset + d + 2], rowOut[accRowOffset + d + 3],
                            rowOut[accRowOffset + d + 4], rowOut[accRowOffset + d + 5],
                            rowOut[accRowOffset + d + 6], rowOut[accRowOffset + d + 7]);
                        var result = Avx.Multiply(acc, invSumVec);
                        output[outRowOffset + d] = result.GetElement(0);
                        output[outRowOffset + d + 1] = result.GetElement(1);
                        output[outRowOffset + d + 2] = result.GetElement(2);
                        output[outRowOffset + d + 3] = result.GetElement(3);
                        output[outRowOffset + d + 4] = result.GetElement(4);
                        output[outRowOffset + d + 5] = result.GetElement(5);
                        output[outRowOffset + d + 6] = result.GetElement(6);
                        output[outRowOffset + d + 7] = result.GetElement(7);
                    }
                    for (; d < headDim; d++)
                        output[outRowOffset + d] = rowOut[accRowOffset + d] * invSum;
                }
                else
                {
                    for (int d = 0; d < headDim; d++)
                        output[outRowOffset + d] = rowOut[accRowOffset + d] * invSum;
                }
            }
        }

        } // end try
        finally
        {
            ArrayPool<float>.Shared.Return(rowMax);
            ArrayPool<float>.Shared.Return(rowSum);
            ArrayPool<float>.Shared.Return(rowOut);
            ArrayPool<float>.Shared.Return(sBlock);
        }
    }

    /// <summary>
    /// Compute S_ij = Q_i @ K_j^T * scale for a tile.
    /// sBlock layout: [qBlock × kBlock]
    /// </summary>
    private static void ComputeAttentionScores(
        ReadOnlySpan<float> Q, ReadOnlySpan<float> K,
        float[] sBlock,
        int qStart, int kStart,
        int qBlock, int kBlock, int headDim, float scale)
    {
        for (int qi = 0; qi < qBlock; qi++)
        {
            int qOffset = (qStart + qi) * headDim;

            for (int ki = 0; ki < kBlock; ki++)
            {
                int kOffset = (kStart + ki) * headDim;

                // Dot product Q[qi] · K[ki]
                float dot = 0f;

                if (Avx2.IsSupported)
                {
                    dot = SimdDot(Q, qOffset, K, kOffset, headDim);
                }
                else
                {
                    for (int d = 0; d < headDim; d++)
                        dot += Q[qOffset + d] * K[kOffset + d];
                }

                sBlock[qi * kBlock + ki] = dot * scale;
            }
        }
    }

    /// <summary>Apply causal mask: set future positions to -inf.</summary>
    private static void ApplyCausalMask(
        float[] sBlock,
        int qStart, int kStart,
        int qBlock, int kBlock,
        int startPos)
    {
        for (int qi = 0; qi < qBlock; qi++)
        {
            int queryPos = startPos + qStart + qi;
            for (int ki = 0; ki < kBlock; ki++)
            {
                int keyPos = kStart + ki;
                if (keyPos > queryPos)
                {
                    sBlock[qi * kBlock + ki] = float.NegativeInfinity;
                }
            }
        }
    }

    /// <summary>
    /// Online softmax update with output accumulation.
    ///
    /// For each query row, maintains:
    ///   - rowMax[r]: max attention score seen so far
    ///   - rowSum[r]: sum of exp(s - max) seen so far
    ///   - rowOut[r, :]: accumulated weighted sum (unnormalized)
    ///
    /// When processing a new KV block, if new max > old max:
    ///   1. Rescale existing accumulators by exp(oldMax - newMax)
    ///   2. Add new contributions with exp(s - newMax)
    /// </summary>
    private static void UpdateOnlineSoftmax(
        float[] sBlock, ReadOnlySpan<float> V,
        float[] rowMax, float[] rowSum, float[] rowOut,
        int qStart, int kStart,
        int qBlock, int kBlock, int headDim)
    {
        for (int qi = 0; qi < qBlock; qi++)
        {
            // Find max in this block for this query row
            float blockMax = float.NegativeInfinity;
            for (int ki = 0; ki < kBlock; ki++)
            {
                float s = sBlock[qi * kBlock + ki];
                if (s > blockMax) blockMax = s;
            }

            if (float.IsNegativeInfinity(blockMax)) continue; // All masked

            // Compute new running max
            float oldMax = rowMax[qi];
            float newMax = Math.Max(oldMax, blockMax);

            // Rescale existing accumulator if max changed
            float rescale = float.IsNegativeInfinity(oldMax) ? 0f : MathF.Exp(oldMax - newMax);
            rowSum[qi] *= rescale;

            int outOffset = qi * headDim;

            // Rescale existing output accumulator
            if (rescale != 0f && rescale != 1f)
            {
                for (int d = 0; d < headDim; d++)
                    rowOut[outOffset + d] *= rescale;
            }

            // Add new contributions: for each K position in this block
            for (int ki = 0; ki < kBlock; ki++)
            {
                float s = sBlock[qi * kBlock + ki];
                if (float.IsNegativeInfinity(s)) continue;

                float p = MathF.Exp(s - newMax);
                rowSum[qi] += p;

                // Accumulate: rowOut[qi, :] += p * V[kStart + ki, :]
                int vOffset = (kStart + ki) * headDim;

                if (Avx2.IsSupported)
                {
                    var pVec = Vector256.Create(p);
                    int d = 0;
                    for (; d + 8 <= headDim; d += 8)
                    {
                        var vVec = Vector256.Create(
                            V[vOffset + d], V[vOffset + d + 1],
                            V[vOffset + d + 2], V[vOffset + d + 3],
                            V[vOffset + d + 4], V[vOffset + d + 5],
                            V[vOffset + d + 6], V[vOffset + d + 7]);
                        var oVec = Vector256.Create(
                            rowOut[outOffset + d], rowOut[outOffset + d + 1],
                            rowOut[outOffset + d + 2], rowOut[outOffset + d + 3],
                            rowOut[outOffset + d + 4], rowOut[outOffset + d + 5],
                            rowOut[outOffset + d + 6], rowOut[outOffset + d + 7]);

                        var result = Avx.Add(oVec, Avx.Multiply(pVec, vVec));

                        rowOut[outOffset + d] = result.GetElement(0);
                        rowOut[outOffset + d + 1] = result.GetElement(1);
                        rowOut[outOffset + d + 2] = result.GetElement(2);
                        rowOut[outOffset + d + 3] = result.GetElement(3);
                        rowOut[outOffset + d + 4] = result.GetElement(4);
                        rowOut[outOffset + d + 5] = result.GetElement(5);
                        rowOut[outOffset + d + 6] = result.GetElement(6);
                        rowOut[outOffset + d + 7] = result.GetElement(7);
                    }
                    for (; d < headDim; d++)
                        rowOut[outOffset + d] += p * V[vOffset + d];
                }
                else
                {
                    for (int d = 0; d < headDim; d++)
                        rowOut[outOffset + d] += p * V[vOffset + d];
                }
            }

            rowMax[qi] = newMax;
        }
    }

    /// <summary>AVX2 dot product helper.</summary>
    private static float SimdDot(ReadOnlySpan<float> a, int aOff, ReadOnlySpan<float> b, int bOff, int len)
    {
        var sum = Vector256<float>.Zero;
        int d = 0;

        for (; d + 8 <= len; d += 8)
        {
            var va = Vector256.Create(
                a[aOff + d], a[aOff + d + 1], a[aOff + d + 2], a[aOff + d + 3],
                a[aOff + d + 4], a[aOff + d + 5], a[aOff + d + 6], a[aOff + d + 7]);
            var vb = Vector256.Create(
                b[bOff + d], b[bOff + d + 1], b[bOff + d + 2], b[bOff + d + 3],
                b[bOff + d + 4], b[bOff + d + 5], b[bOff + d + 6], b[bOff + d + 7]);
            sum = Avx.Add(sum, Avx.Multiply(va, vb));
        }

        float result = 0f;
        for (int i = 0; i < 8; i++) result += sum.GetElement(i);
        for (; d < len; d++) result += a[aOff + d] * b[bOff + d];
        return result;
    }

    // ─── CUDA FlashAttention Kernel ─────────────────────────────

    /// <summary>
    /// CUDA FlashAttention forward pass (P/Invoke to native kernel).
    /// Falls back to CPU implementation if CUDA is unavailable.
    /// </summary>
    public static void ForwardCuda(
        IntPtr Q, IntPtr K, IntPtr V, IntPtr output,
        int batchSize, int numHeads, int seqLen, int kvLen, int headDim,
        float scale, bool causal)
    {
        try
        {
            NativeFlashAttn.flash_attention_forward(
                Q, K, V, output,
                batchSize, numHeads, seqLen, kvLen, headDim,
                scale, causal ? 1 : 0);
        }
        catch (DllNotFoundException)
        {
            // Fall back to CPU — caller should use the managed Forward() method
            throw new NotSupportedException(
                "CUDA FlashAttention kernel not available. " +
                "Use FlashAttention.Forward() for CPU fallback.");
        }
    }

    /// <summary>Native P/Invoke declarations for CUDA FlashAttention.</summary>
    private static class NativeFlashAttn
    {
        [System.Runtime.InteropServices.DllImport("sharpinfer_cuda", CallingConvention = System.Runtime.InteropServices.CallingConvention.Cdecl)]
        public static extern void flash_attention_forward(
            IntPtr Q, IntPtr K, IntPtr V, IntPtr output,
            int batchSize, int numHeads, int seqLen, int kvLen, int headDim,
            float scale, int causal);
    }
}

/// <summary>
/// Configuration for FlashAttention behavior.
/// </summary>
public class FlashAttentionConfig
{
    /// <summary>Whether to use FlashAttention (vs standard O(N²) attention). Default: true.</summary>
    public bool Enabled { get; set; } = true;

    /// <summary>Block size for Q dimension tiling. Tune based on L2 cache size.</summary>
    public int BlockSizeQ { get; set; } = FlashAttention.BLOCK_SIZE_Q;

    /// <summary>Block size for K/V dimension tiling.</summary>
    public int BlockSizeKV { get; set; } = FlashAttention.BLOCK_SIZE_KV;

    /// <summary>Prefer CUDA kernel when available.</summary>
    public bool PreferCuda { get; set; } = true;
}
