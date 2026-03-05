namespace SharpInfer.Core.Tensors;

/// <summary>
/// Quantization types matching GGUF/GGML format specifications.
/// </summary>
public enum QuantType
{
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    BF16 = 30,
}

/// <summary>
/// Dequantization routines for GGML quantized weight formats.
/// Each block encodes a group of weights with a shared scale factor.
/// </summary>
public static class Dequantize
{
    /// <summary>Block size for each quantization type.</summary>
    public static int BlockSize(QuantType type) => type switch
    {
        QuantType.Q4_0 => 32,
        QuantType.Q4_1 => 32,
        QuantType.Q5_0 => 32,
        QuantType.Q5_1 => 32,
        QuantType.Q8_0 => 32,
        QuantType.Q8_1 => 32,
        QuantType.Q2_K => 256,
        QuantType.Q3_K => 256,
        QuantType.Q4_K => 256,
        QuantType.Q5_K => 256,
        QuantType.Q6_K => 256,
        _ => 1
    };

    /// <summary>Byte size of one block for each quantization type.</summary>
    public static int BlockBytes(QuantType type) => type switch
    {
        QuantType.Q4_0 => 2 + 16,       // f16 scale + 16 bytes (32 x 4-bit)
        QuantType.Q4_1 => 2 + 2 + 16,   // f16 scale + f16 min + 16 bytes
        QuantType.Q5_0 => 2 + 4 + 16,   // f16 scale + 4 high-bit bytes + 16 packed bytes
        QuantType.Q5_1 => 2 + 2 + 4 + 16, // f16 scale + f16 min + 4 high-bit bytes + 16 packed bytes
        QuantType.Q8_0 => 2 + 32,       // f16 scale + 32 bytes (32 x 8-bit)
        QuantType.Q8_1 => 2 + 2 + 32,   // f16 scale + f16 sum + 32 bytes (32 x 8-bit)
        QuantType.Q2_K => 84,           // f16 d + f16 dmin + 16 scales + 64 qs
        QuantType.Q3_K => 110,          // f16 d + 32 hmask + 64 qs + 12 scales
        QuantType.Q4_K => 144,          // f16 d + f16 dmin + 12 scales + 128 qs
        QuantType.Q5_K => 176,          // f16 d + f16 dmin + 12 scales + 128 qs + 32 qh
        QuantType.Q6_K => 210,          // f16 d + 128 ql + 64 qh + 16 scales
        QuantType.F16 => 2,
        QuantType.BF16 => 2,
        QuantType.F32 => 4,
        _ => throw new NotSupportedException($"Quantization type {type} not yet implemented.")
    };

    /// <summary>
    /// Dequantize Q4_0 block: 32 weights packed as 4-bit integers with a shared f16 scale.
    /// Layout: [f16 scale][16 bytes of packed 4-bit values]
    /// </summary>
    public static void DequantQ4_0(ReadOnlySpan<byte> block, Span<float> output)
    {
        float scale = HalfToFloat(block[..2]);

        for (int j = 0; j < 16; j++)
        {
            byte packed = block[2 + j];
            // Low nibble (subtract 8 for zero-centering)
            output[j] = ((packed & 0x0F) - 8) * scale;
            // High nibble
            output[j + 16] = ((packed >> 4) - 8) * scale;
        }
    }

    /// <summary>
    /// Dequantize Q4_1 block: like Q4_0 but with a min value for asymmetric quantization.
    /// Layout: [f16 scale][f16 min][16 bytes of packed 4-bit values]
    /// </summary>
    public static void DequantQ4_1(ReadOnlySpan<byte> block, Span<float> output)
    {
        float scale = HalfToFloat(block[..2]);
        float min = HalfToFloat(block.Slice(2, 2));

        for (int j = 0; j < 16; j++)
        {
            byte packed = block[4 + j];
            output[j] = (packed & 0x0F) * scale + min;
            output[j + 16] = (packed >> 4) * scale + min;
        }
    }

    /// <summary>
    /// Dequantize Q8_0 block: 32 weights stored as signed 8-bit integers with a shared f16 scale.
    /// Layout: [f16 scale][32 signed bytes]
    /// </summary>
    public static void DequantQ8_0(ReadOnlySpan<byte> block, Span<float> output)
    {
        float scale = HalfToFloat(block[..2]);

        for (int j = 0; j < 32; j++)
        {
            output[j] = (sbyte)block[2 + j] * scale;
        }
    }

    /// <summary>
    /// Dequantize Q5_0 block: 32 weights with 5-bit quants (4 low + 1 high).
    /// Layout: [f16 scale][4 high-bit bytes][16 packed bytes (low 4-bit pairs)]
    /// </summary>
    public static void DequantQ5_0(ReadOnlySpan<byte> block, Span<float> output)
    {
        float scale = HalfToFloat(block[..2]);
        uint qh = BitConverter.ToUInt32(block.Slice(2, 4));

        for (int j = 0; j < 16; j++)
        {
            byte packed = block[6 + j];
            int x0 = (packed & 0x0F) | (((int)(qh >> j) & 1) << 4);
            int x1 = (packed >> 4) | (((int)(qh >> (j + 16)) & 1) << 4);
            output[j] = (x0 - 16) * scale;
            output[j + 16] = (x1 - 16) * scale;
        }
    }

    /// <summary>
    /// Dequantize Q5_1 block: like Q5_0 but with asymmetric min value.
    /// Layout: [f16 scale][f16 min][4 high-bit bytes][16 packed bytes]
    /// </summary>
    public static void DequantQ5_1(ReadOnlySpan<byte> block, Span<float> output)
    {
        float scale = HalfToFloat(block[..2]);
        float min = HalfToFloat(block.Slice(2, 2));
        uint qh = BitConverter.ToUInt32(block.Slice(4, 4));

        for (int j = 0; j < 16; j++)
        {
            byte packed = block[8 + j];
            int x0 = (packed & 0x0F) | (((int)(qh >> j) & 1) << 4);
            int x1 = (packed >> 4) | (((int)(qh >> (j + 16)) & 1) << 4);
            output[j] = x0 * scale + min;
            output[j + 16] = x1 * scale + min;
        }
    }

    /// <summary>
    /// Dequantize Q8_1 block: 32 weights as signed 8-bit with scale and sum (asymmetric).
    /// Layout: [f16 scale][f16 sum][32 signed bytes]
    /// The sum is used for dot product optimization but not needed for simple dequantization.
    /// </summary>
    public static void DequantQ8_1(ReadOnlySpan<byte> block, Span<float> output)
    {
        float scale = HalfToFloat(block[..2]);
        // block[2..4] is the sum, not needed for dequant
        for (int j = 0; j < 32; j++)
        {
            output[j] = (sbyte)block[4 + j] * scale;
        }
    }

    // ─── K-Quant Dequantization ─────────────────────────────────────

    /// <summary>
    /// Dequantize Q2_K block: 256 weights with 2-bit quants, per-16-element scales.
    /// Layout: [f16 d][f16 dmin][16 scales][64 qs]
    /// </summary>
    public static void DequantQ2_K(ReadOnlySpan<byte> block, Span<float> output)
    {
        float d = HalfToFloat(block[..2]);
        float dmin = HalfToFloat(block.Slice(2, 2));
        ReadOnlySpan<byte> scales = block.Slice(4, 16);
        ReadOnlySpan<byte> qs = block.Slice(20, 64);

        int outIdx = 0;
        for (int j = 0; j < 16; j++)
        {
            float sc = d * (scales[j] & 0xF);
            float m = dmin * (scales[j] >> 4);
            for (int l = 0; l < 4; l++)
            {
                byte q = qs[j * 4 + l];
                output[outIdx++] = sc * (q & 3) - m;
                output[outIdx++] = sc * ((q >> 2) & 3) - m;
                output[outIdx++] = sc * ((q >> 4) & 3) - m;
                output[outIdx++] = sc * ((q >> 6) & 3) - m;
            }
        }
    }

    /// <summary>
    /// Dequantize Q3_K block: 256 weights with 3-bit quants.
    /// Layout: [f16 d][32 hmask][64 qs][12 scales]
    /// Each value = d * scale * (low2bits + high_bit_contribution)
    /// where high_bit_contribution is 0 if hmask bit is set, -4 otherwise.
    /// </summary>
    public static void DequantQ3_K(ReadOnlySpan<byte> block, Span<float> output)
    {
        float dAll = HalfToFloat(block[..2]);
        ReadOnlySpan<byte> hmask = block.Slice(2, 32);
        ReadOnlySpan<byte> qs = block.Slice(34, 64);
        ReadOnlySpan<byte> scalesRaw = block.Slice(98, 12);

        // Decode 16 6-bit scales from 12 packed bytes using ggml bit layout:
        // bytes 0-3: low nibbles → low 4 bits of scales 0-3
        //            high nibbles → low 4 bits of scales 8-11
        // bytes 4-7: low nibbles → low 4 bits of scales 4-7
        //            high nibbles → low 4 bits of scales 12-15
        // bytes 8-11: high 2 bits packed for all 16 scales
        Span<int> scales = stackalloc int[16];
        for (int i = 0; i < 4; i++)
        {
            scales[i] = scalesRaw[i] & 0xF;
            scales[i + 4] = scalesRaw[i + 4] & 0xF;
            scales[i + 8] = scalesRaw[i] >> 4;
            scales[i + 12] = (scalesRaw[i + 4]) >> 4;
        }
        for (int i = 0; i < 4; i++)
        {
            byte hi = scalesRaw[8 + i];
            scales[i] |= ((hi >> 0) & 3) << 4;
            scales[i + 4] |= ((hi >> 2) & 3) << 4;
            scales[i + 8] |= ((hi >> 4) & 3) << 4;
            scales[i + 12] |= ((hi >> 6) & 3) << 4;
        }
        for (int i = 0; i < 16; i++)
            scales[i] -= 32;

        // Dequantize: 2 halves of 128 values, each half has 4 groups of 32
        int outIdx = 0;
        int qOff = 0;
        byte m = 1;   // hmask bit selector, cycles through bits 0-7
        int scIdx = 0;
        for (int n = 0; n < 2; n++)  // two halves
        {
            int shift = 0;
            for (int j = 0; j < 4; j++)  // 4 groups of 32 per half
            {
                float dl = dAll * scales[scIdx];
                for (int l = 0; l < 16; l++)
                {
                    int lo2 = (qs[qOff + l] >> shift) & 3;
                    int hb = (hmask[l] & m) != 0 ? 0 : -4;
                    output[outIdx++] = dl * (lo2 + hb);
                }
                float dl2 = dAll * scales[scIdx + 1];
                for (int l = 16; l < 32; l++)
                {
                    int lo2 = (qs[qOff + l] >> shift) & 3;
                    int hb = (hmask[l] & m) != 0 ? 0 : -4;
                    output[outIdx++] = dl2 * (lo2 + hb);
                }
                shift += 2;
                m <<= 1;
                scIdx += 2;
            }
            qOff += 32;
        }
    }

    /// <summary>
    /// Dequantize Q4_K block: 256 weights with 4-bit quants, 8 sub-blocks of 32.
    /// Layout: [f16 d][f16 dmin][12 packed scales][128 qs]
    /// </summary>
    public static void DequantQ4_K(ReadOnlySpan<byte> block, Span<float> output)
    {
        float d = HalfToFloat(block[..2]);
        float dmin = HalfToFloat(block.Slice(2, 2));
        ReadOnlySpan<byte> scales = block.Slice(4, 12);
        ReadOnlySpan<byte> qs = block.Slice(16, 128);

        int outIdx = 0;
        int qIdx = 0;
        for (int j = 0; j < 8; j += 2)
        {
            GetScaleMinK4(j, scales, out byte sc1, out byte m1);
            GetScaleMinK4(j + 1, scales, out byte sc2, out byte m2);
            float d1 = d * sc1;
            float m1f = dmin * m1;
            float d2 = d * sc2;
            float m2f = dmin * m2;

            for (int l = 0; l < 32; l++)
            {
                output[outIdx + l] = d1 * (qs[qIdx + l] & 0xF) - m1f;
                output[outIdx + l + 32] = d2 * (qs[qIdx + l] >> 4) - m2f;
            }
            outIdx += 64;
            qIdx += 32;
        }
    }

    /// <summary>Helper to decode packed 6-bit scale and min values for Q4_K/Q5_K.</summary>
    private static void GetScaleMinK4(int j, ReadOnlySpan<byte> q, out byte sc, out byte min)
    {
        if (j < 4)
        {
            sc = (byte)(q[j] & 63);
            min = (byte)(q[j + 4] & 63);
        }
        else
        {
            sc = (byte)((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
            min = (byte)((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
        }
    }

    /// <summary>
    /// Dequantize Q5_K block: 256 weights with 5-bit quants.
    /// Layout: [f16 d][f16 dmin][12 packed scales][128 qs][32 qh]
    /// The 32 qh bytes provide the 5th bit for all 256 values.
    /// Each byte's 8 bits cover the same index across all 4 groups of 64.
    /// </summary>
    public static void DequantQ5_K(ReadOnlySpan<byte> block, Span<float> output)
    {
        float d = HalfToFloat(block[..2]);
        float dmin = HalfToFloat(block.Slice(2, 2));
        ReadOnlySpan<byte> scales = block.Slice(4, 12);
        ReadOnlySpan<byte> qs = block.Slice(16, 128);
        ReadOnlySpan<byte> qh = block.Slice(144, 32);

        int outIdx = 0;
        int qIdx = 0;
        int scIdx = 0;
        byte u1 = 1, u2 = 2;  // bit masks for qh, shift left by 2 each iteration
        for (int j = 0; j < 4; j++)
        {
            GetScaleMinK4(scIdx, scales, out byte sc1, out byte m1);
            GetScaleMinK4(scIdx + 1, scales, out byte sc2, out byte m2);
            float d1 = d * sc1;
            float m1f = dmin * m1;
            float d2 = d * sc2;
            float m2f = dmin * m2;

            for (int l = 0; l < 32; l++)
            {
                int hi1 = (qh[l] & u1) != 0 ? 16 : 0;
                int hi2 = (qh[l] & u2) != 0 ? 16 : 0;
                output[outIdx + l] = d1 * ((qs[qIdx + l] & 0xF) + hi1) - m1f;
                output[outIdx + l + 32] = d2 * ((qs[qIdx + l] >> 4) + hi2) - m2f;
            }
            outIdx += 64;
            qIdx += 32;
            scIdx += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }

    /// <summary>
    /// Dequantize Q6_K block: 256 weights with 6-bit quants, 16 sub-blocks of 16.
    /// Layout: [128 ql][64 qh][16 int8 scales][f16 d]
    /// </summary>
    public static void DequantQ6_K(ReadOnlySpan<byte> block, Span<float> output)
    {
        ReadOnlySpan<byte> ql = block[..128];
        ReadOnlySpan<byte> qh = block.Slice(128, 64);
        ReadOnlySpan<sbyte> scales = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, sbyte>(block.Slice(192, 16));
        float d = HalfToFloat(block.Slice(208, 2));

        int outIdx = 0;
        for (int n = 0; n < 2; n++)  // two halves of 128
        {
            int qlOff = n * 64;
            int qhOff = n * 32;
            int scOff = n * 8;

            for (int l = 0; l < 32; l++)
            {
                int scIdx = scOff + l / 16;
                int q1 = (ql[qlOff + l] & 0xF) | (((qh[qhOff + l] >> 0) & 3) << 4);
                int q2 = (ql[qlOff + l + 32] & 0xF) | (((qh[qhOff + l] >> 2) & 3) << 4);
                int q3 = (ql[qlOff + l] >> 4) | (((qh[qhOff + l] >> 4) & 3) << 4);
                int q4 = (ql[qlOff + l + 32] >> 4) | (((qh[qhOff + l] >> 6) & 3) << 4);

                output[outIdx + l] = d * scales[scIdx] * (q1 - 32);
                output[outIdx + l + 32] = d * scales[scIdx + 2] * (q2 - 32);
                output[outIdx + l + 64] = d * scales[scIdx + 4] * (q3 - 32);
                output[outIdx + l + 96] = d * scales[scIdx + 6] * (q4 - 32);
            }
            outIdx += 128;
        }
    }

    /// <summary>Convert a 2-byte IEEE 754 half-precision float to a C# float.</summary>
    public static float HalfToFloat(ReadOnlySpan<byte> bytes)
    {
        ushort h = BitConverter.ToUInt16(bytes);
        return (float)BitConverter.UInt16BitsToHalf(h);
    }

    /// <summary>
    /// Convert a 2-byte bfloat16 to a C# float.
    /// BF16 is the upper 16 bits of a 32-bit float, so we just shift left by 16.
    /// </summary>
    public static float BFloat16ToFloat(ReadOnlySpan<byte> bytes)
    {
        ushort bf = BitConverter.ToUInt16(bytes);
        uint f32bits = (uint)bf << 16;
        return BitConverter.UInt32BitsToSingle(f32bits);
    }

    /// <summary>
    /// Dequantize an entire weight tensor from raw bytes.
    /// Returns a float array suitable for use in tensor operations.
    /// </summary>
    public static float[] DequantizeTensor(ReadOnlySpan<byte> data, QuantType type, int numElements)
    {
        if (type == QuantType.F32)
        {
            var result = new float[numElements];
            for (int i = 0; i < numElements; i++)
                result[i] = BitConverter.ToSingle(data.Slice(i * 4, 4));
            return result;
        }

        if (type == QuantType.F16)
        {
            var result = new float[numElements];
            for (int i = 0; i < numElements; i++)
                result[i] = HalfToFloat(data.Slice(i * 2, 2));
            return result;
        }

        if (type == QuantType.BF16)
        {
            var result = new float[numElements];
            for (int i = 0; i < numElements; i++)
                result[i] = BFloat16ToFloat(data.Slice(i * 2, 2));
            return result;
        }

        int blockSize = BlockSize(type);
        int blockBytes = BlockBytes(type);
        int numBlocks = numElements / blockSize;
        var output = new float[numElements];

        for (int b = 0; b < numBlocks; b++)
        {
            var block = data.Slice(b * blockBytes, blockBytes);
            var dest = output.AsSpan(b * blockSize, blockSize);

            switch (type)
            {
                case QuantType.Q4_0: DequantQ4_0(block, dest); break;
                case QuantType.Q4_1: DequantQ4_1(block, dest); break;
                case QuantType.Q5_0: DequantQ5_0(block, dest); break;
                case QuantType.Q5_1: DequantQ5_1(block, dest); break;
                case QuantType.Q8_0: DequantQ8_0(block, dest); break;
                case QuantType.Q8_1: DequantQ8_1(block, dest); break;
                case QuantType.Q2_K: DequantQ2_K(block, dest); break;
                case QuantType.Q3_K: DequantQ3_K(block, dest); break;
                case QuantType.Q4_K: DequantQ4_K(block, dest); break;
                case QuantType.Q5_K: DequantQ5_K(block, dest); break;
                case QuantType.Q6_K: DequantQ6_K(block, dest); break;
                default: throw new NotSupportedException($"Quantization type {type} not implemented.");
            }
        }

        return output;
    }
}
