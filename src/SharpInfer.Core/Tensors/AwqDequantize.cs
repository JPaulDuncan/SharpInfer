namespace SharpInfer.Core.Tensors;

/// <summary>
/// AWQ (Activation-aware Weight Quantization) dequantization.
///
/// AWQ is similar to GPTQ in storage layout but uses a fundamentally different
/// quantization algorithm: it identifies "salient" weight channels (those that
/// matter most for activations) and protects them during quantization.
///
/// Tensor layout in safetensors for an AWQ-quantized linear layer:
///   {name}.qweight  — packed INT4 values [rows/pack_factor x cols], stored as int32
///   {name}.qzeros   — zero points per group [num_groups x cols/pack_factor], stored as int32
///   {name}.scales   — scale factors per group [num_groups x cols], stored as float16
///
/// AWQ does NOT use g_idx (no activation reordering / desc_act).
///
/// Two sub-variants:
///   - GEMM: Standard layout, same as GPTQ without desc_act
///   - GEMV: Reordered for vector-matrix multiply (used on some inference engines)
///
/// Config (from quantize_config.json or config.json):
///   quant_method: "awq"
///   bits: 4
///   group_size: 128
///   zero_point: true (asymmetric, typical)
///   version: "GEMM" or "GEMV"
///
/// Reference: https://arxiv.org/abs/2306.00978
/// </summary>
public static class AwqDequantize
{
    /// <summary>
    /// Dequantize an AWQ GEMM-format weight matrix.
    ///
    /// This is structurally identical to GPTQ without desc_act,
    /// but the quantization quality differs due to AWQ's activation-aware algorithm.
    ///
    /// Formula: weight[row, col] = (packed_int - zero_point[group, col]) * scale[group, col]
    /// </summary>
    public static float[] DequantizeGemm(
        int[] qweight, int[] qzeros, float[] scales,
        int bits, int groupSize, int rows, int cols)
    {
        int packFactor = 32 / bits;
        uint mask = (1u << bits) - 1;
        var output = new float[rows * cols];

        for (int row = 0; row < rows; row++)
        {
            int group = row / groupSize;
            int packRow = row / packFactor;
            int packOffset = (row % packFactor) * bits;

            for (int col = 0; col < cols; col++)
            {
                // Extract quantized value
                int packedVal = qweight[packRow * cols + col];
                int quantized = (int)((uint)packedVal >> packOffset) & (int)mask;

                // Extract zero point
                int zpPackCol = col / packFactor;
                int zpOffset = (col % packFactor) * bits;
                int zpPacked = qzeros[group * (cols / packFactor) + zpPackCol];
                int zeroPoint = (int)((uint)zpPacked >> zpOffset) & (int)mask;

                // Dequantize
                float scale = scales[group * cols + col];
                output[row * cols + col] = (quantized - zeroPoint) * scale;
            }
        }

        return output;
    }

    /// <summary>
    /// Dequantize an AWQ GEMV-format weight matrix.
    ///
    /// GEMV format reorders weights for efficient matrix-vector multiplication.
    /// Instead of packing consecutive rows into int32, it interleaves values
    /// to enable coalesced memory access during single-token generation.
    ///
    /// Packing layout (GEMV):
    ///   Within each int32, values are ordered for sequential access by
    ///   the dot-product dimension rather than the output dimension.
    ///   The interleave factor determines how values are grouped.
    /// </summary>
    public static float[] DequantizeGemv(
        int[] qweight, int[] qzeros, float[] scales,
        int bits, int groupSize, int rows, int cols)
    {
        int packFactor = 32 / bits;
        uint mask = (1u << bits) - 1;
        var output = new float[rows * cols];

        // GEMV interleave factor (typically 4 for AWQ GEMV INT4)
        int interleave = bits == 4 ? 4 : 2;

        for (int row = 0; row < rows; row++)
        {
            int group = row / groupSize;

            // GEMV reordering: values within a pack group are interleaved
            int baseRow = row / (packFactor * interleave);
            int interleaveIdx = (row % (packFactor * interleave));
            int packSlot = interleaveIdx / interleave;
            int interleaveOffset = interleaveIdx % interleave;

            for (int col = 0; col < cols; col++)
            {
                // GEMV packs values with an interleave stride
                int gemvCol = col * interleave + interleaveOffset;
                int packIdx = baseRow * (cols * interleave) + gemvCol;
                int packOffset = packSlot * bits;

                if (packIdx >= qweight.Length) continue;

                int quantized = (int)((uint)qweight[packIdx] >> packOffset) & (int)mask;

                // Zero points use the same GEMV reordering
                int zpPackCol = col / packFactor;
                int zpOffset = (col % packFactor) * bits;
                int zpIdx = group * (cols / packFactor) + zpPackCol;
                int zeroPoint = zpIdx < qzeros.Length
                    ? (int)((uint)qzeros[zpIdx] >> zpOffset) & (int)mask
                    : (1 << (bits - 1)); // fallback to symmetric

                float scale = scales[group * cols + col];
                output[row * cols + col] = (quantized - zeroPoint) * scale;
            }
        }

        return output;
    }

    /// <summary>
    /// Auto-detect GEMM vs GEMV format and dequantize accordingly.
    /// GEMV format can be detected by checking the qweight tensor shape:
    /// GEMM: [rows/pack_factor, cols]
    /// GEMV: [rows/pack_factor, cols * interleave] (wider)
    /// </summary>
    public static float[] DequantizeAuto(
        int[] qweight, int[] qzeros, float[] scales,
        int bits, int groupSize, int rows, int cols, AwqVersion version = AwqVersion.GEMM)
    {
        return version switch
        {
            AwqVersion.GEMM => DequantizeGemm(qweight, qzeros, scales, bits, groupSize, rows, cols),
            AwqVersion.GEMV => DequantizeGemv(qweight, qzeros, scales, bits, groupSize, rows, cols),
            _ => DequantizeGemm(qweight, qzeros, scales, bits, groupSize, rows, cols),
        };
    }
}

/// <summary>AWQ format variants.</summary>
public enum AwqVersion
{
    GEMM,
    GEMV,
}

/// <summary>
/// AWQ quantization configuration, parsed from quantize_config.json or config.json.
/// </summary>
public class AwqConfig
{
    public int Bits { get; set; } = 4;
    public int GroupSize { get; set; } = 128;
    public bool ZeroPoint { get; set; } = true;
    public AwqVersion Version { get; set; } = AwqVersion.GEMM;
    public string QuantMethod { get; set; } = "awq";
}
