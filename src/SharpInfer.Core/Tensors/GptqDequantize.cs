namespace SharpInfer.Core.Tensors;

/// <summary>
/// GPTQ (Post-Training Quantization) dequantization.
///
/// GPTQ stores weights as packed INT4 (or INT3/INT8) values with group-wise
/// scaling. Unlike GGUF's row-major block format, GPTQ uses column-oriented
/// packing where multiple 4-bit values are packed into 32-bit integers.
///
/// Tensor layout in safetensors for a GPTQ-quantized linear layer:
///   {name}.qweight  — packed INT4 values [rows/pack_factor x cols], stored as int32
///   {name}.qzeros   — zero points per group [num_groups x cols/pack_factor], stored as int32
///   {name}.scales   — scale factors per group [num_groups x cols], stored as float16
///   {name}.g_idx    — optional group index mapping [rows], for non-sequential grouping
///
/// Standard config (from quantize_config.json):
///   bits: 4 (or 2, 3, 8)
///   group_size: 128 (common) or -1 (per-column)
///   desc_act: true/false (whether activation reordering was used → g_idx present)
///
/// Reference: https://arxiv.org/abs/2210.17323
/// </summary>
public static class GptqDequantize
{
    /// <summary>
    /// Dequantize a GPTQ-quantized weight matrix.
    ///
    /// The formula for each weight is:
    ///   weight[row, col] = (packed_int - zero_point[group, col]) * scale[group, col]
    ///
    /// Where group = row / group_size (or g_idx[row] if desc_act is used).
    /// </summary>
    /// <param name="qweight">Packed int32 array [rows/pack_factor x cols]</param>
    /// <param name="qzeros">Packed zero points [num_groups x cols/pack_factor] as int32</param>
    /// <param name="scales">Scale factors [num_groups x cols] as float</param>
    /// <param name="gIdx">Optional group index mapping [rows]. Null if desc_act=false.</param>
    /// <param name="bits">Quantization bits (typically 4)</param>
    /// <param name="groupSize">Number of rows per quantization group</param>
    /// <param name="rows">Original (unquantized) number of rows</param>
    /// <param name="cols">Number of columns</param>
    /// <returns>Dequantized float array [rows x cols]</returns>
    public static float[] Dequantize(
        int[] qweight, int[] qzeros, float[] scales,
        int[]? gIdx, int bits, int groupSize, int rows, int cols)
    {
        int packFactor = 32 / bits;          // How many values fit in one int32
        uint mask = (1u << bits) - 1;        // Bit mask for extracting one value
        var output = new float[rows * cols];

        for (int row = 0; row < rows; row++)
        {
            // Determine which quantization group this row belongs to
            int group = gIdx != null ? gIdx[row] : row / groupSize;

            int packRow = row / packFactor;
            int packOffset = (row % packFactor) * bits;

            for (int col = 0; col < cols; col++)
            {
                // Extract the quantized integer from packed storage
                int packedVal = qweight[packRow * cols + col];
                int quantized = (int)((uint)packedVal >> packOffset) & (int)mask;

                // Extract zero point (also packed)
                int zpPackCol = col / packFactor;
                int zpOffset = (col % packFactor) * bits;
                int zpPacked = qzeros[group * (cols / packFactor) + zpPackCol];
                int zeroPoint = (int)((uint)zpPacked >> zpOffset) & (int)mask;

                // Apply scale and zero point
                float scale = scales[group * cols + col];
                output[row * cols + col] = (quantized - zeroPoint) * scale;
            }
        }

        return output;
    }

    /// <summary>
    /// Dequantize with Marlin-optimized layout.
    /// Marlin reorders weights for efficient GPU memory access patterns.
    /// The logical output is the same as standard GPTQ dequantization.
    ///
    /// Marlin packing: weights are stored in a tile-interleaved format
    /// optimized for warp-level matrix multiply on NVIDIA GPUs.
    /// This function handles the de-interleaving.
    /// </summary>
    public static float[] DequantizeMarlin(
        int[] bQweight, float[] scales,
        int bits, int groupSize, int rows, int cols)
    {
        // Marlin uses a specific tiling: 16x64 tiles with interleaved rows
        const int TileRows = 16;
        const int TileCols = 64;

        int packFactor = 32 / bits;
        uint mask = (1u << bits) - 1;
        var output = new float[rows * cols];

        int numGroups = groupSize > 0 ? rows / groupSize : 1;

        for (int row = 0; row < rows; row++)
        {
            int group = row / groupSize;

            // Marlin interleaves rows within each tile
            int tileRow = row / TileRows;
            int intraRow = row % TileRows;

            for (int col = 0; col < cols; col++)
            {
                int tileCol = col / TileCols;
                int intraCol = col % TileCols;

                // Calculate the Marlin-reordered index
                int marlinIdx = (tileRow * ((cols + TileCols - 1) / TileCols) + tileCol)
                    * TileRows * TileCols
                    + intraRow * TileCols + intraCol;

                int packIdx = marlinIdx / packFactor;
                int packOffset = (marlinIdx % packFactor) * bits;

                if (packIdx >= bQweight.Length) continue;

                int quantized = (int)((uint)bQweight[packIdx] >> packOffset) & (int)mask;

                // Marlin uses symmetric quantization (no zero point, centered at 2^(bits-1))
                int zeroPoint = 1 << (bits - 1);
                float scale = scales[group * cols + col];
                output[row * cols + col] = (quantized - zeroPoint) * scale;
            }
        }

        return output;
    }
}

/// <summary>
/// GPTQ quantization configuration, parsed from quantize_config.json.
/// </summary>
public class GptqConfig
{
    public int Bits { get; set; } = 4;
    public int GroupSize { get; set; } = 128;
    public bool DescAct { get; set; } = false;
    public bool SymmetricQuantization { get; set; } = false;
    public string QuantMethod { get; set; } = "gptq";
    public bool IsMarlin { get; set; } = false;
}
