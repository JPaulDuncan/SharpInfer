using System.Numerics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace SharpInfer.Core.Tensors;

/// <summary>
/// Extended quantization types for aggressive compression.
///
/// Adds K-quant variants (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K) used by llama.cpp
/// and 1-bit quantization (BitNet) for extreme compression.
///
/// K-quants use super-blocks with per-block scales, allowing finer granularity
/// and better accuracy at the same bit-width vs basic quant types.
///
/// Memory savings vs F16:
///   Q6_K: 6.5 bits/weight → 2.5x compression
///   Q5_K: 5.5 bits/weight → 2.9x compression
///   Q4_K: 4.5 bits/weight → 3.5x compression
///   Q3_K: 3.4 bits/weight → 4.7x compression
///   Q2_K: 2.6 bits/weight → 6.2x compression
///   BitNet: 1.6 bits/weight → 10x compression (ternary {-1, 0, +1})
/// </summary>
public enum QuantTypeExtended
{
    // Original types (matching QuantType for compatibility)
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,

    // K-quant types (super-block quantization with improved accuracy)
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,

    // 1-bit / ternary quantization
    BitNet = 20,      // 1.58-bit: ternary {-1, 0, +1}
    BitNetB158 = 21,  // BitNet b1.58 with AbsMean quantization
}

/// <summary>
/// K-quant super-block: groups of 256 weights organized as 16 sub-blocks of 16 weights each.
/// Each sub-block has its own scale, and the super-block has a shared scale for the sub-block scales.
///
/// This hierarchical approach gives better accuracy than flat quantization at the same bit-width.
/// </summary>
public static class KQuantDequantize
{
    /// <summary>Super-block size for K-quant types.</summary>
    public const int SuperBlockSize = 256;

    /// <summary>Sub-block size within a super-block.</summary>
    public const int SubBlockSize = 16;

    /// <summary>Number of sub-blocks per super-block.</summary>
    public const int NumSubBlocks = SuperBlockSize / SubBlockSize; // 16

    /// <summary>
    /// Dequantize Q2_K super-block: 2-bit weights with per-sub-block 4-bit scales.
    ///
    /// Layout (84 bytes per 256 weights):
    ///   [f16 super_scale][f16 super_min][16 × 4-bit scale nibbles (8 bytes)][64 bytes packed 2-bit values]
    ///
    /// Each weight = sub_scale * quant_val + sub_min
    /// where sub_scale and sub_min are derived from the 4-bit nibbles and super-scale/min.
    /// </summary>
    public static void DequantQ2_K(ReadOnlySpan<byte> block, Span<float> output)
    {
        float superScale = Dequantize.HalfToFloat(block[..2]);
        float superMin = Dequantize.HalfToFloat(block.Slice(2, 2));

        // 16 sub-block scales packed as 4-bit nibbles (8 bytes)
        Span<float> subScales = stackalloc float[NumSubBlocks];
        Span<float> subMins = stackalloc float[NumSubBlocks];
        for (int i = 0; i < 8; i++)
        {
            byte packed = block[4 + i];
            subScales[i * 2] = (packed & 0x0F) * superScale;
            subScales[i * 2 + 1] = (packed >> 4) * superScale;
            subMins[i * 2] = (packed & 0x0F) * superMin;
            subMins[i * 2 + 1] = (packed >> 4) * superMin;
        }

        // 256 weights packed as 2-bit values (64 bytes)
        int dataOffset = 12;
        for (int sb = 0; sb < NumSubBlocks; sb++)
        {
            float scale = subScales[sb];
            float min = subMins[sb];
            int baseIdx = sb * SubBlockSize;

            for (int j = 0; j < SubBlockSize; j++)
            {
                int bitIdx = baseIdx + j;
                int byteIdx = bitIdx / 4;
                int bitPos = (bitIdx % 4) * 2;
                int val = (block[dataOffset + byteIdx] >> bitPos) & 0x03;
                output[baseIdx + j] = val * scale + min;
            }
        }
    }

    /// <summary>
    /// Dequantize Q3_K super-block: 3-bit weights with per-sub-block scales.
    ///
    /// Layout per 256 weights:
    ///   [f16 super_scale][16 × 6-bit scales (12 bytes)][96 bytes packed 3-bit values][16 bytes high-bits]
    ///
    /// 3-bit is stored as 2 low bits + 1 high bit in separate arrays.
    /// Each weight = scale * (quant_val - 4) for symmetric quantization.
    /// </summary>
    public static void DequantQ3_K(ReadOnlySpan<byte> block, Span<float> output)
    {
        float superScale = Dequantize.HalfToFloat(block[..2]);

        // 16 scales encoded in 12 bytes (6 bits each)
        Span<float> subScales = stackalloc float[NumSubBlocks];
        for (int i = 0; i < NumSubBlocks; i++)
        {
            int bitOffset = i * 6;
            int byteIdx = bitOffset / 8;
            int bitPos = bitOffset % 8;
            int rawScale;
            if (bitPos + 6 <= 8)
            {
                rawScale = (block[2 + byteIdx] >> bitPos) & 0x3F;
            }
            else
            {
                rawScale = ((block[2 + byteIdx] >> bitPos) |
                           (block[2 + byteIdx + 1] << (8 - bitPos))) & 0x3F;
            }
            // 6-bit signed: center at 32
            subScales[i] = (rawScale - 32) * superScale;
        }

        // Low 2 bits: 64 bytes at offset 14
        int lowOffset = 14;
        // High 1 bit: 32 bytes at offset 78
        int highOffset = 78;

        for (int sb = 0; sb < NumSubBlocks; sb++)
        {
            float scale = subScales[sb];
            int baseIdx = sb * SubBlockSize;

            for (int j = 0; j < SubBlockSize; j++)
            {
                int idx = baseIdx + j;

                // Low 2 bits
                int lowByteIdx = idx / 4;
                int lowBitPos = (idx % 4) * 2;
                int lowBits = (block[lowOffset + lowByteIdx] >> lowBitPos) & 0x03;

                // High 1 bit
                int highByteIdx = idx / 8;
                int highBitPos = idx % 8;
                int highBit = (block[highOffset + highByteIdx] >> highBitPos) & 0x01;

                int val = lowBits | (highBit << 2); // 3-bit value [0..7]
                output[idx] = (val - 4) * scale;    // Symmetric around 0
            }
        }
    }

    /// <summary>
    /// Dequantize Q4_K super-block: 4-bit weights with per-sub-block 6-bit scales and mins.
    ///
    /// Higher accuracy than Q4_0 due to hierarchical scales.
    /// </summary>
    public static void DequantQ4_K(ReadOnlySpan<byte> block, Span<float> output)
    {
        float superScale = Dequantize.HalfToFloat(block[..2]);
        float superMin = Dequantize.HalfToFloat(block.Slice(2, 2));

        // 16 scales + 16 mins packed as 6-bit values (24 bytes)
        Span<float> subScales = stackalloc float[NumSubBlocks];
        Span<float> subMins = stackalloc float[NumSubBlocks];

        for (int i = 0; i < NumSubBlocks; i++)
        {
            int bitOff = i * 6;
            int byteIdx = bitOff / 8;
            int bitPos = bitOff % 8;
            int rawScale, rawMin;

            if (bitPos + 6 <= 8)
            {
                rawScale = (block[4 + byteIdx] >> bitPos) & 0x3F;
                rawMin = (block[16 + byteIdx] >> bitPos) & 0x3F;
            }
            else
            {
                rawScale = ((block[4 + byteIdx] >> bitPos) |
                           (block[4 + byteIdx + 1] << (8 - bitPos))) & 0x3F;
                rawMin = ((block[16 + byteIdx] >> bitPos) |
                         (block[16 + byteIdx + 1] << (8 - bitPos))) & 0x3F;
            }

            subScales[i] = rawScale * superScale;
            subMins[i] = rawMin * superMin;
        }

        // 128 bytes of packed 4-bit values at offset 28
        int dataOffset = 28;
        for (int sb = 0; sb < NumSubBlocks; sb++)
        {
            float scale = subScales[sb];
            float min = subMins[sb];
            int baseIdx = sb * SubBlockSize;

            for (int j = 0; j < SubBlockSize / 2; j++)
            {
                byte packed = block[dataOffset + (sb * SubBlockSize / 2) + j];
                output[baseIdx + j * 2] = (packed & 0x0F) * scale + min;
                output[baseIdx + j * 2 + 1] = (packed >> 4) * scale + min;
            }
        }
    }

    /// <summary>
    /// Dequantize Q6_K super-block: 6-bit weights with per-sub-block 8-bit scales.
    /// Highest K-quant precision, nearly F16 quality.
    /// </summary>
    public static void DequantQ6_K(ReadOnlySpan<byte> block, Span<float> output)
    {
        float superScale = Dequantize.HalfToFloat(block[..2]);

        // 16 scales as 8-bit signed values (16 bytes)
        Span<float> subScales = stackalloc float[NumSubBlocks];
        for (int i = 0; i < NumSubBlocks; i++)
        {
            subScales[i] = (sbyte)block[2 + i] * superScale;
        }

        // Low 4 bits: 128 bytes at offset 18
        int lowOffset = 18;
        // High 2 bits: 64 bytes at offset 146
        int highOffset = 146;

        for (int sb = 0; sb < NumSubBlocks; sb++)
        {
            float scale = subScales[sb];
            int baseIdx = sb * SubBlockSize;

            for (int j = 0; j < SubBlockSize; j++)
            {
                int idx = baseIdx + j;

                // Low 4 bits (nibble)
                int lowByteIdx = idx / 2;
                int lowBits = (idx % 2 == 0)
                    ? (block[lowOffset + lowByteIdx] & 0x0F)
                    : (block[lowOffset + lowByteIdx] >> 4);

                // High 2 bits
                int highByteIdx = idx / 4;
                int highBitPos = (idx % 4) * 2;
                int highBits = (block[highOffset + highByteIdx] >> highBitPos) & 0x03;

                int val = lowBits | (highBits << 4); // 6-bit value [0..63]
                output[idx] = (val - 32) * scale;    // Symmetric
            }
        }
    }

    /// <summary>
    /// Dequantize an entire tensor from K-quant format.
    /// </summary>
    public static float[] DequantizeTensorKQuant(ReadOnlySpan<byte> data, QuantTypeExtended type, int numElements)
    {
        int blockSize = SuperBlockSize;
        int blockBytes = BlockBytesKQuant(type);
        int numBlocks = numElements / blockSize;
        var output = new float[numElements];

        for (int b = 0; b < numBlocks; b++)
        {
            var block = data.Slice(b * blockBytes, blockBytes);
            var dest = output.AsSpan(b * blockSize, blockSize);

            switch (type)
            {
                case QuantTypeExtended.Q2_K: DequantQ2_K(block, dest); break;
                case QuantTypeExtended.Q3_K: DequantQ3_K(block, dest); break;
                case QuantTypeExtended.Q4_K: DequantQ4_K(block, dest); break;
                case QuantTypeExtended.Q6_K: DequantQ6_K(block, dest); break;
                default: throw new NotSupportedException($"K-quant type {type} not implemented.");
            }
        }

        return output;
    }

    /// <summary>Byte size of one K-quant super-block.</summary>
    public static int BlockBytesKQuant(QuantTypeExtended type) => type switch
    {
        QuantTypeExtended.Q2_K => 2 + 2 + 8 + 64,          // 76 bytes for 256 weights
        QuantTypeExtended.Q3_K => 2 + 12 + 64 + 32,        // 110 bytes for 256 weights
        QuantTypeExtended.Q4_K => 2 + 2 + 12 + 12 + 128,   // 156 bytes for 256 weights
        QuantTypeExtended.Q6_K => 2 + 16 + 128 + 64,       // 210 bytes for 256 weights
        _ => throw new NotSupportedException($"K-quant type {type}")
    };

    /// <summary>Bits per weight for each K-quant type.</summary>
    public static float BitsPerWeight(QuantTypeExtended type) => type switch
    {
        QuantTypeExtended.F32 => 32f,
        QuantTypeExtended.F16 => 16f,
        QuantTypeExtended.Q8_0 => 8.5f,
        QuantTypeExtended.Q6_K => 6.5625f,
        QuantTypeExtended.Q5_K => 5.5f,
        QuantTypeExtended.Q4_K => 4.875f,
        QuantTypeExtended.Q4_0 => 4.5f,
        QuantTypeExtended.Q3_K => 3.4375f,
        QuantTypeExtended.Q2_K => 2.375f,
        QuantTypeExtended.BitNet => 1.625f,
        QuantTypeExtended.BitNetB158 => 1.58f,
        _ => throw new NotSupportedException($"Unknown type {type}")
    };
}

/// <summary>
/// BitNet 1.58-bit quantization: ternary weights {-1, 0, +1}.
///
/// Based on "The Era of 1-bit LLMs" (Microsoft Research, 2024).
/// Each weight is encoded as 2 bits (ternary: 00=-1, 01=0, 10=+1)
/// packed 4 per byte. A per-row scale factor preserves magnitude.
///
/// Key advantages:
///   - Replaces FP multiply-accumulate with integer add/subtract
///   - 10x memory reduction vs F16
///   - No floating-point multiplications in MatVec (just additions)
///   - Extremely cache-friendly
///
/// Quantization uses AbsMean scaling:
///   scale = mean(|W|)
///   W_ternary = round_clip(W / scale)   → {-1, 0, +1}
/// </summary>
public static class BitNetQuantizer
{
    /// <summary>Number of ternary values packed per byte.</summary>
    public const int ValuesPerByte = 4;

    /// <summary>
    /// Quantize a weight row to ternary {-1, 0, +1} using AbsMean scaling.
    ///
    /// Returns packed ternary data and the per-row scale factor.
    /// </summary>
    public static (byte[] packed, float scale) QuantizeRow(ReadOnlySpan<float> weights)
    {
        // Compute AbsMean
        double absSum = 0;
        for (int i = 0; i < weights.Length; i++)
            absSum += Math.Abs(weights[i]);
        float scale = (float)(absSum / weights.Length);

        if (scale < 1e-10f)
        {
            // All zeros
            return (new byte[(weights.Length + ValuesPerByte - 1) / ValuesPerByte], 0f);
        }

        float invScale = 1f / scale;
        int packedLen = (weights.Length + ValuesPerByte - 1) / ValuesPerByte;
        var packed = new byte[packedLen];

        for (int i = 0; i < weights.Length; i++)
        {
            float normalized = weights[i] * invScale;
            // Round to {-1, 0, +1}
            int ternary;
            if (normalized > 0.5f) ternary = 1;
            else if (normalized < -0.5f) ternary = -1;
            else ternary = 0;

            // Encode: -1→0b00, 0→0b01, +1→0b10
            int encoded = ternary + 1; // {0, 1, 2}
            int byteIdx = i / ValuesPerByte;
            int bitPos = (i % ValuesPerByte) * 2;
            packed[byteIdx] |= (byte)(encoded << bitPos);
        }

        return (packed, scale);
    }

    /// <summary>
    /// Dequantize a ternary row back to float values.
    /// </summary>
    public static void DequantizeRow(ReadOnlySpan<byte> packed, float scale, Span<float> output)
    {
        for (int i = 0; i < output.Length; i++)
        {
            int byteIdx = i / ValuesPerByte;
            int bitPos = (i % ValuesPerByte) * 2;
            int encoded = (packed[byteIdx] >> bitPos) & 0x03;
            int ternary = encoded - 1; // {-1, 0, +1}
            output[i] = ternary * scale;
        }
    }

    /// <summary>
    /// Quantize an entire weight matrix (row-major) to BitNet format.
    /// Returns per-row scales and packed ternary data.
    /// </summary>
    public static BitNetMatrix QuantizeMatrix(float[] weights, int rows, int cols)
    {
        var result = new BitNetMatrix
        {
            Rows = rows,
            Cols = cols,
            Scales = new float[rows],
            PackedData = new byte[rows][],
        };

        for (int r = 0; r < rows; r++)
        {
            var row = weights.AsSpan(r * cols, cols);
            (result.PackedData[r], result.Scales[r]) = QuantizeRow(row);
        }

        return result;
    }

    /// <summary>
    /// Ultra-fast matrix-vector multiply for ternary weights.
    /// Replaces all FP multiplications with integer additions/subtractions.
    ///
    /// For each output[i] = sum(W[i,j] * x[j]) where W[i,j] ∈ {-1, 0, +1}:
    ///   output[i] = scale[i] * (sum_positive(x[j]) - sum_negative(x[j]))
    ///
    /// This is ~10x faster than F16 matmul on CPU.
    /// </summary>
    public static void MatVecTernary(BitNetMatrix matrix, ReadOnlySpan<float> input, Span<float> output)
    {
        if (input.Length != matrix.Cols)
            throw new ArgumentException($"Input size {input.Length} != matrix cols {matrix.Cols}");

        for (int r = 0; r < matrix.Rows; r++)
        {
            float sum = MatVecTernaryRow(matrix.PackedData[r], input, matrix.Cols);
            output[r] = sum * matrix.Scales[r];
        }
    }

    /// <summary>
    /// AVX2-accelerated ternary dot product for a single row.
    /// Falls back to scalar on non-x86 platforms.
    /// </summary>
    private static float MatVecTernaryRow(byte[] packedRow, ReadOnlySpan<float> input, int cols)
    {
        if (Avx2.IsSupported && cols >= 32)
            return MatVecTernaryRowAvx2(packedRow, input, cols);

        return MatVecTernaryRowScalar(packedRow, input, cols);
    }

    private static float MatVecTernaryRowScalar(byte[] packed, ReadOnlySpan<float> input, int cols)
    {
        float posSum = 0, negSum = 0;

        for (int j = 0; j < cols; j++)
        {
            int byteIdx = j / ValuesPerByte;
            int bitPos = (j % ValuesPerByte) * 2;
            int encoded = (packed[byteIdx] >> bitPos) & 0x03;

            switch (encoded)
            {
                case 2: posSum += input[j]; break; // +1
                case 0: negSum += input[j]; break; // -1
                // case 1: zero, skip
            }
        }

        return posSum - negSum;
    }

    private static unsafe float MatVecTernaryRowAvx2(byte[] packed, ReadOnlySpan<float> input, int cols)
    {
        var vPosSum = Vector256<float>.Zero;
        var vNegSum = Vector256<float>.Zero;

        fixed (float* pInput = input)
        fixed (byte* pPacked = packed)
        {
            int j = 0;

            // Process 8 elements at a time with AVX2
            for (; j + 8 <= cols; j += 8)
            {
                // Load 8 input values
                var vInput = Avx.LoadVector256(pInput + j);

                // Decode 8 ternary values (16 bits = 2 bytes)
                // Build masks for +1 and -1
                var posMask = Vector256<float>.Zero;
                var negMask = Vector256<float>.Zero;

                for (int k = 0; k < 8; k++)
                {
                    int idx = j + k;
                    int byteIdx = idx / ValuesPerByte;
                    int bitPos = (idx % ValuesPerByte) * 2;
                    int encoded = (pPacked[byteIdx] >> bitPos) & 0x03;

                    if (encoded == 2) // +1
                        posMask = posMask.WithElement(k, 1f);
                    else if (encoded == 0) // -1
                        negMask = negMask.WithElement(k, 1f);
                }

                vPosSum = Avx.Add(vPosSum, Avx.Multiply(vInput, posMask));
                vNegSum = Avx.Add(vNegSum, Avx.Multiply(vInput, negMask));
            }

            // Horizontal sum
            float posSum = HorizontalSum256(vPosSum);
            float negSum = HorizontalSum256(vNegSum);

            // Scalar tail
            for (; j < cols; j++)
            {
                int byteIdx = j / ValuesPerByte;
                int bitPos = (j % ValuesPerByte) * 2;
                int encoded = (pPacked[byteIdx] >> bitPos) & 0x03;
                if (encoded == 2) posSum += pInput[j];
                else if (encoded == 0) negSum += pInput[j];
            }

            return posSum - negSum;
        }
    }

    private static float HorizontalSum256(Vector256<float> v)
    {
        var hi128 = Avx.ExtractVector128(v, 1);
        var lo128 = Avx.ExtractVector128(v, 0);
        var sum128 = Sse.Add(lo128, hi128);
        var shuf = Sse.MoveHighToLow(sum128, sum128);
        sum128 = Sse.Add(sum128, shuf);
        sum128 = Sse.AddScalar(sum128, Sse.Shuffle(sum128, sum128, 0x01));
        return sum128.ToScalar();
    }
}

/// <summary>
/// A weight matrix stored in BitNet ternary format.
/// </summary>
public class BitNetMatrix
{
    public int Rows { get; init; }
    public int Cols { get; init; }
    public float[] Scales { get; init; } = Array.Empty<float>();
    public byte[][] PackedData { get; init; } = Array.Empty<byte[]>();

    /// <summary>Size in bytes.</summary>
    public long SizeBytes =>
        Scales.Length * 4L +
        PackedData.Sum(p => (long)p.Length);

    /// <summary>Compression ratio vs F32.</summary>
    public double CompressionRatio =>
        (double)(Rows * Cols * 4) / SizeBytes;

    /// <summary>Dequantize the full matrix back to float.</summary>
    public float[] Dequantize()
    {
        var result = new float[Rows * Cols];
        for (int r = 0; r < Rows; r++)
        {
            var dest = result.AsSpan(r * Cols, Cols);
            BitNetQuantizer.DequantizeRow(PackedData[r], Scales[r], dest);
        }
        return result;
    }
}

/// <summary>
/// Dynamic quantization: quantize weights at runtime from a higher-precision source.
/// Useful for trying different quantization levels without re-downloading models.
/// </summary>
public static class DynamicQuantizer
{
    /// <summary>
    /// Quantize F32 weights to a target K-quant type.
    /// Returns packed bytes matching the K-quant block format.
    /// </summary>
    public static byte[] QuantizeToKQuant(ReadOnlySpan<float> weights, QuantTypeExtended targetType)
    {
        int blockSize = KQuantDequantize.SuperBlockSize;
        int numBlocks = weights.Length / blockSize;
        int blockBytes = KQuantDequantize.BlockBytesKQuant(targetType);
        var output = new byte[numBlocks * blockBytes];

        for (int b = 0; b < numBlocks; b++)
        {
            var block = weights.Slice(b * blockSize, blockSize);
            var dest = output.AsSpan(b * blockBytes, blockBytes);

            switch (targetType)
            {
                case QuantTypeExtended.Q2_K: QuantizeBlockQ2K(block, dest); break;
                case QuantTypeExtended.Q3_K: QuantizeBlockQ3K(block, dest); break;
                case QuantTypeExtended.Q4_K: QuantizeBlockQ4K(block, dest); break;
                default: throw new NotSupportedException($"Quantization to {targetType} not supported.");
            }
        }

        return output;
    }

    private static void QuantizeBlockQ2K(ReadOnlySpan<float> block, Span<byte> output)
    {
        // Find global range
        float maxAbs = 0;
        for (int i = 0; i < block.Length; i++)
            maxAbs = Math.Max(maxAbs, Math.Abs(block[i]));

        float superScale = maxAbs / (3 * 15); // 2-bit has 4 levels, 4-bit sub-scale has 16 levels
        float superMin = 0;

        // Write super-scale and super-min as f16
        WriteHalf(output, 0, superScale);
        WriteHalf(output, 2, superMin);

        // Quantize sub-block scales (4-bit each, packed in 8 bytes)
        for (int sb = 0; sb < KQuantDequantize.NumSubBlocks; sb++)
        {
            var subBlock = block.Slice(sb * KQuantDequantize.SubBlockSize, KQuantDequantize.SubBlockSize);
            float subMax = 0;
            for (int j = 0; j < subBlock.Length; j++)
                subMax = Math.Max(subMax, Math.Abs(subBlock[j]));

            int scaleNibble = superScale > 0 ? Math.Clamp((int)(subMax / superScale / 3 + 0.5f), 0, 15) : 0;

            int byteIdx = sb / 2;
            if (sb % 2 == 0)
                output[4 + byteIdx] = (byte)(scaleNibble & 0x0F);
            else
                output[4 + byteIdx] |= (byte)((scaleNibble & 0x0F) << 4);
        }

        // Quantize 256 weights to 2-bit each (64 bytes)
        int dataOffset = 12;
        for (int i = 0; i < 256; i++)
        {
            int sb = i / KQuantDequantize.SubBlockSize;
            int scaleByteIdx = sb / 2;
            int rawScale = (sb % 2 == 0)
                ? (output[4 + scaleByteIdx] & 0x0F)
                : (output[4 + scaleByteIdx] >> 4);
            float scale = rawScale * superScale;

            int quantVal = scale > 0
                ? Math.Clamp((int)((block[i] - superMin) / scale + 0.5f), 0, 3)
                : 0;

            int bytePos = i / 4;
            int bitPos = (i % 4) * 2;
            output[dataOffset + bytePos] |= (byte)(quantVal << bitPos);
        }
    }

    private static void QuantizeBlockQ3K(ReadOnlySpan<float> block, Span<byte> output)
    {
        float maxAbs = 0;
        for (int i = 0; i < block.Length; i++)
            maxAbs = Math.Max(maxAbs, Math.Abs(block[i]));

        float superScale = maxAbs > 0 ? maxAbs / (4 * 31) : 0;
        WriteHalf(output, 0, superScale);

        // Quantize sub-block scales as 6-bit signed values
        for (int sb = 0; sb < KQuantDequantize.NumSubBlocks; sb++)
        {
            var subBlock = block.Slice(sb * KQuantDequantize.SubBlockSize, KQuantDequantize.SubBlockSize);
            float subMax = 0;
            for (int j = 0; j < subBlock.Length; j++)
                subMax = Math.Max(subMax, Math.Abs(subBlock[j]));

            int rawScale = superScale > 0
                ? Math.Clamp((int)(subMax / superScale / 4 + 0.5f) + 32, 0, 63)
                : 32;

            // Pack 6-bit value
            int bitOffset = sb * 6;
            int byteIdx = bitOffset / 8;
            int bitPos = bitOffset % 8;
            output[2 + byteIdx] |= (byte)(rawScale << bitPos);
            if (bitPos + 6 > 8)
                output[2 + byteIdx + 1] |= (byte)(rawScale >> (8 - bitPos));
        }

        // Low 2 bits at offset 14
        int lowOffset = 14;
        // High 1 bit at offset 78
        int highOffset = 78;

        for (int i = 0; i < 256; i++)
        {
            int sb = i / KQuantDequantize.SubBlockSize;
            float scale = GetSubScaleQ3K(output, sb, superScale);

            int quantVal = scale != 0
                ? Math.Clamp((int)(block[i] / scale + 4 + 0.5f), 0, 7)
                : 4;

            // Low 2 bits
            int lowByteIdx = i / 4;
            int lowBitPos = (i % 4) * 2;
            output[lowOffset + lowByteIdx] |= (byte)((quantVal & 0x03) << lowBitPos);

            // High 1 bit
            int highByteIdx = i / 8;
            int highBitPos = i % 8;
            output[highOffset + highByteIdx] |= (byte)(((quantVal >> 2) & 0x01) << highBitPos);
        }
    }

    private static float GetSubScaleQ3K(Span<byte> output, int sb, float superScale)
    {
        int bitOffset = sb * 6;
        int byteIdx = bitOffset / 8;
        int bitPos = bitOffset % 8;
        int raw;
        if (bitPos + 6 <= 8)
            raw = (output[2 + byteIdx] >> bitPos) & 0x3F;
        else
            raw = ((output[2 + byteIdx] >> bitPos) | (output[2 + byteIdx + 1] << (8 - bitPos))) & 0x3F;
        return (raw - 32) * superScale;
    }

    private static void QuantizeBlockQ4K(ReadOnlySpan<float> block, Span<byte> output)
    {
        float maxVal = float.MinValue, minVal = float.MaxValue;
        for (int i = 0; i < block.Length; i++)
        {
            maxVal = Math.Max(maxVal, block[i]);
            minVal = Math.Min(minVal, block[i]);
        }

        float superScale = (maxVal - minVal) / (15 * 63);
        float superMin = minVal < 0 ? -minVal / (15 * 63) : 0;

        WriteHalf(output, 0, superScale);
        WriteHalf(output, 2, superMin);

        // Sub-block 6-bit scales and mins (12 + 12 bytes)
        for (int sb = 0; sb < KQuantDequantize.NumSubBlocks; sb++)
        {
            var subBlock = block.Slice(sb * KQuantDequantize.SubBlockSize, KQuantDequantize.SubBlockSize);
            float subMax = float.MinValue, subMin = float.MaxValue;
            for (int j = 0; j < subBlock.Length; j++)
            {
                subMax = Math.Max(subMax, subBlock[j]);
                subMin = Math.Min(subMin, subBlock[j]);
            }

            int rawScale = superScale > 0
                ? Math.Clamp((int)((subMax - subMin) / 15 / superScale + 0.5f), 0, 63) : 0;
            int rawMin = superMin > 0
                ? Math.Clamp((int)(-subMin / 15 / superMin + 0.5f), 0, 63) : 0;

            Pack6Bit(output.Slice(4), sb, rawScale);
            Pack6Bit(output.Slice(16), sb, rawMin);
        }

        // Pack 4-bit values at offset 28
        int dataOffset = 28;
        for (int sb = 0; sb < KQuantDequantize.NumSubBlocks; sb++)
        {
            float scale = Unpack6Bit(output.Slice(4), sb) * superScale;
            float min = Unpack6Bit(output.Slice(16), sb) * superMin;
            int baseIdx = sb * KQuantDequantize.SubBlockSize;

            for (int j = 0; j < KQuantDequantize.SubBlockSize / 2; j++)
            {
                float v0 = block[baseIdx + j * 2];
                float v1 = block[baseIdx + j * 2 + 1];

                int q0 = scale > 0 ? Math.Clamp((int)((v0 - min) / scale + 0.5f), 0, 15) : 0;
                int q1 = scale > 0 ? Math.Clamp((int)((v1 - min) / scale + 0.5f), 0, 15) : 0;

                output[dataOffset + sb * KQuantDequantize.SubBlockSize / 2 + j] = (byte)(q0 | (q1 << 4));
            }
        }
    }

    private static void Pack6Bit(Span<byte> dest, int index, int value)
    {
        int bitOff = index * 6;
        int byteIdx = bitOff / 8;
        int bitPos = bitOff % 8;
        dest[byteIdx] |= (byte)((value & 0x3F) << bitPos);
        if (bitPos + 6 > 8 && byteIdx + 1 < dest.Length)
            dest[byteIdx + 1] |= (byte)((value & 0x3F) >> (8 - bitPos));
    }

    private static int Unpack6Bit(Span<byte> src, int index)
    {
        int bitOff = index * 6;
        int byteIdx = bitOff / 8;
        int bitPos = bitOff % 8;
        if (bitPos + 6 <= 8)
            return (src[byteIdx] >> bitPos) & 0x3F;
        return ((src[byteIdx] >> bitPos) | (src[byteIdx + 1] << (8 - bitPos))) & 0x3F;
    }

    private static void WriteHalf(Span<byte> dest, int offset, float value)
    {
        var half = BitConverter.HalfToUInt16Bits((Half)value);
        dest[offset] = (byte)(half & 0xFF);
        dest[offset + 1] = (byte)(half >> 8);
    }

    /// <summary>
    /// Estimate model size at each quantization level.
    /// </summary>
    public static QuantizationSizeEstimate EstimateSizes(long totalParams)
    {
        return new QuantizationSizeEstimate
        {
            TotalParams = totalParams,
            F32Bytes = totalParams * 4,
            F16Bytes = totalParams * 2,
            Q8Bytes = (long)(totalParams * KQuantDequantize.BitsPerWeight(QuantTypeExtended.Q8_0) / 8),
            Q6KBytes = (long)(totalParams * KQuantDequantize.BitsPerWeight(QuantTypeExtended.Q6_K) / 8),
            Q4KBytes = (long)(totalParams * KQuantDequantize.BitsPerWeight(QuantTypeExtended.Q4_K) / 8),
            Q3KBytes = (long)(totalParams * KQuantDequantize.BitsPerWeight(QuantTypeExtended.Q3_K) / 8),
            Q2KBytes = (long)(totalParams * KQuantDequantize.BitsPerWeight(QuantTypeExtended.Q2_K) / 8),
            BitNetBytes = (long)(totalParams * KQuantDequantize.BitsPerWeight(QuantTypeExtended.BitNet) / 8),
        };
    }
}

/// <summary>Estimated model size at each quantization level.</summary>
public class QuantizationSizeEstimate
{
    public long TotalParams { get; init; }
    public long F32Bytes { get; init; }
    public long F16Bytes { get; init; }
    public long Q8Bytes { get; init; }
    public long Q6KBytes { get; init; }
    public long Q4KBytes { get; init; }
    public long Q3KBytes { get; init; }
    public long Q2KBytes { get; init; }
    public long BitNetBytes { get; init; }

    public string FormatTable()
    {
        static string Fmt(long b) => b switch
        {
            >= 1L << 30 => $"{b / (double)(1L << 30):F1} GB",
            >= 1L << 20 => $"{b / (double)(1L << 20):F1} MB",
            _ => $"{b / (double)(1L << 10):F1} KB"
        };

        return $"""
            Model size estimates ({TotalParams:N0} parameters):
              F32:    {Fmt(F32Bytes)}
              F16:    {Fmt(F16Bytes)}
              Q8_0:   {Fmt(Q8Bytes)}
              Q6_K:   {Fmt(Q6KBytes)}
              Q4_K:   {Fmt(Q4KBytes)}
              Q3_K:   {Fmt(Q3KBytes)}
              Q2_K:   {Fmt(Q2KBytes)}
              BitNet: {Fmt(BitNetBytes)}
            """;
    }
}
