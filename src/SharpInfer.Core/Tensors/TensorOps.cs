using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace SharpInfer.Core.Tensors;

/// <summary>
/// SIMD-accelerated tensor operations.
/// These are the hot-path math functions used throughout the transformer.
/// Provides AVX2 fast paths with scalar fallbacks.
/// </summary>
public static class TensorOps
{
    /// <summary>
    /// Computes the dot product of two float spans using SIMD.
    /// This is the most critical operation — called millions of times during inference.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Spans must have equal length for dot product.");

        float sum = 0f;
        int i = 0;

        if (Avx.IsSupported && a.Length >= 8)
        {
            var vsum = Vector256<float>.Zero;
            ref float aRef = ref MemoryMarshal.GetReference(a);
            ref float bRef = ref MemoryMarshal.GetReference(b);

            for (; i + 8 <= a.Length; i += 8)
            {
                var va = Vector256.LoadUnsafe(ref Unsafe.Add(ref aRef, i));
                var vb = Vector256.LoadUnsafe(ref Unsafe.Add(ref bRef, i));
                vsum = Avx.Add(vsum, Avx.Multiply(va, vb));
            }

            // Horizontal sum: reduce 8 floats to 1
            var hi = Avx.ExtractVector128(vsum, 1);
            var lo = Avx.ExtractVector128(vsum, 0);
            var sum128 = Sse.Add(hi, lo);
            sum128 = Sse3.IsSupported
                ? Sse3.HorizontalAdd(sum128, sum128)
                : Sse.Add(sum128, Sse.MoveHighToLow(sum128, sum128));
            sum128 = Sse.AddScalar(sum128, Sse.Shuffle(sum128, sum128, 0x01));
            sum = sum128.ToScalar();
        }

        // Scalar tail
        for (; i < a.Length; i++)
            sum += a[i] * b[i];

        return sum;
    }

    /// <summary>
    /// Matrix-vector multiplication: result = mat @ vec
    /// mat is [rows x cols], vec is [cols], result is [rows]
    /// </summary>
    public static void MatVecMul(ReadOnlySpan<float> mat, ReadOnlySpan<float> vec, Span<float> result, int rows, int cols)
    {
        if (mat.Length != rows * cols)
            throw new ArgumentException("Matrix dimensions mismatch.");

        for (int r = 0; r < rows; r++)
        {
            result[r] = Dot(mat.Slice(r * cols, cols), vec);
        }
    }

    /// <summary>
    /// Matrix-matrix multiplication: C = A @ B
    /// A is [M x K], B is [K x N], C is [M x N]
    /// Uses simple tiled approach — replace with BLAS for production.
    /// </summary>
    public static void MatMul(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> c,
        int M, int K, int N)
    {
        const int TileSize = 32;
        c.Clear();

        for (int i = 0; i < M; i += TileSize)
        {
            for (int j = 0; j < N; j += TileSize)
            {
                for (int k = 0; k < K; k += TileSize)
                {
                    int iEnd = Math.Min(i + TileSize, M);
                    int jEnd = Math.Min(j + TileSize, N);
                    int kEnd = Math.Min(k + TileSize, K);

                    for (int ii = i; ii < iEnd; ii++)
                    {
                        for (int kk = k; kk < kEnd; kk++)
                        {
                            float aVal = a[ii * K + kk];
                            int cBase = ii * N;
                            int bBase = kk * N;

                            for (int jj = j; jj < jEnd; jj++)
                            {
                                c[cBase + jj] += aVal * b[bBase + jj];
                            }
                        }
                    }
                }
            }
        }
    }

    /// <summary>In-place softmax over a span of floats.</summary>
    public static void Softmax(Span<float> x)
    {
        // Numerical stability: subtract max
        float max = float.MinValue;
        for (int i = 0; i < x.Length; i++)
            if (x[i] > max) max = x[i];

        float sum = 0f;
        for (int i = 0; i < x.Length; i++)
        {
            x[i] = MathF.Exp(x[i] - max);
            sum += x[i];
        }

        float invSum = 1f / sum;
        for (int i = 0; i < x.Length; i++)
            x[i] *= invSum;
    }

    /// <summary>In-place RMS normalization.</summary>
    public static void RmsNorm(Span<float> output, ReadOnlySpan<float> x, ReadOnlySpan<float> weight, float eps = 1e-5f)
    {
        float sumSq = 0f;
        for (int i = 0; i < x.Length; i++)
            sumSq += x[i] * x[i];

        float scale = 1f / MathF.Sqrt(sumSq / x.Length + eps);

        for (int i = 0; i < x.Length; i++)
            output[i] = x[i] * scale * weight[i];
    }

    /// <summary>Element-wise addition: a += b</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void AddInPlace(Span<float> a, ReadOnlySpan<float> b)
    {
        int i = 0;
        if (Avx.IsSupported)
        {
            ref float aRef = ref MemoryMarshal.GetReference(a);
            ref float bRef = ref MemoryMarshal.GetReference(b);
            for (; i + 8 <= a.Length; i += 8)
            {
                var va = Vector256.LoadUnsafe(ref Unsafe.Add(ref aRef, i));
                var vb = Vector256.LoadUnsafe(ref Unsafe.Add(ref bRef, i));
                Avx.Add(va, vb).StoreUnsafe(ref Unsafe.Add(ref aRef, i));
            }
        }
        for (; i < a.Length; i++)
            a[i] += b[i];
    }

    /// <summary>Element-wise multiplication: a *= b (Hadamard product)</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void MulInPlace(Span<float> a, ReadOnlySpan<float> b)
    {
        int i = 0;
        if (Avx.IsSupported)
        {
            ref float aRef = ref MemoryMarshal.GetReference(a);
            ref float bRef = ref MemoryMarshal.GetReference(b);
            for (; i + 8 <= a.Length; i += 8)
            {
                var va = Vector256.LoadUnsafe(ref Unsafe.Add(ref aRef, i));
                var vb = Vector256.LoadUnsafe(ref Unsafe.Add(ref bRef, i));
                Avx.Multiply(va, vb).StoreUnsafe(ref Unsafe.Add(ref aRef, i));
            }
        }
        for (; i < a.Length; i++)
            a[i] *= b[i];
    }

    /// <summary>
    /// SiLU (Swish) activation: x * sigmoid(x).
    /// Used in LLaMA/Mistral feed-forward networks.
    /// </summary>
    public static void SiLU(Span<float> x)
    {
        for (int i = 0; i < x.Length; i++)
            x[i] = x[i] / (1f + MathF.Exp(-x[i]));
    }

    /// <summary>GELU activation (approximate). Used in some model architectures.</summary>
    public static void GeLU(Span<float> x)
    {
        const float sqrt2OverPi = 0.7978845608f;
        const float coeff = 0.044715f;
        for (int i = 0; i < x.Length; i++)
        {
            float v = x[i];
            x[i] = 0.5f * v * (1f + MathF.Tanh(sqrt2OverPi * (v + coeff * v * v * v)));
        }
    }

    /// <summary>Apply a scaled addition: dest += src * scale</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void AddScaled(Span<float> dest, ReadOnlySpan<float> src, float scale)
    {
        for (int i = 0; i < dest.Length; i++)
            dest[i] += src[i] * scale;
    }

    /// <summary>
    /// Argmax over a span of floats. Returns the index of the largest element.
    /// Used in greedy decoding.
    /// </summary>
    public static int ArgMax(ReadOnlySpan<float> x)
    {
        int maxIdx = 0;
        float maxVal = x[0];
        for (int i = 1; i < x.Length; i++)
        {
            if (x[i] > maxVal)
            {
                maxVal = x[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }
}

/// <summary>
/// Interface for pluggable compute backends (CPU, CUDA, Vulkan, etc.)
/// All tensor math routes through this so GPU acceleration is a drop-in swap.
/// </summary>
public interface IComputeBackend
{
    void MatVecMul(ReadOnlySpan<float> mat, ReadOnlySpan<float> vec, Span<float> result, int rows, int cols);
    void MatMul(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> c, int M, int K, int N);
    void Softmax(Span<float> x);
    void RmsNorm(Span<float> output, ReadOnlySpan<float> x, ReadOnlySpan<float> weight, float eps);
    void SiLU(Span<float> x);
    void AddInPlace(Span<float> a, ReadOnlySpan<float> b);
    string Name { get; }
}

/// <summary>Default CPU backend using SIMD-accelerated TensorOps.</summary>
public class CpuBackend : IComputeBackend
{
    public string Name => "CPU (SIMD)";

    public void MatVecMul(ReadOnlySpan<float> mat, ReadOnlySpan<float> vec, Span<float> result, int rows, int cols)
        => TensorOps.MatVecMul(mat, vec, result, rows, cols);

    public void MatMul(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> c, int M, int K, int N)
        => TensorOps.MatMul(a, b, c, M, K, N);

    public void Softmax(Span<float> x) => TensorOps.Softmax(x);

    public void RmsNorm(Span<float> output, ReadOnlySpan<float> x, ReadOnlySpan<float> weight, float eps)
        => TensorOps.RmsNorm(output, x, weight, eps);

    public void SiLU(Span<float> x) => TensorOps.SiLU(x);

    public void AddInPlace(Span<float> a, ReadOnlySpan<float> b) => TensorOps.AddInPlace(a, b);
}
