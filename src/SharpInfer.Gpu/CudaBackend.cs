using System.Runtime.InteropServices;
using SharpInfer.Core.Tensors;

namespace SharpInfer.Gpu;

/// <summary>
/// CUDA compute backend using P/Invoke to call into a native CUDA library.
///
/// Architecture:
/// - This C# class is a thin wrapper that marshals data to/from GPU memory
/// - The actual CUDA kernels live in a native shared library (sharpinfer_cuda.so/.dll)
/// - Build the native library from the Kernels/ directory using nvcc
///
/// To build the native CUDA library:
///   cd Kernels/
///   nvcc -shared -o sharpinfer_cuda.so kernels.cu -O3
///
/// For development without a GPU, falls back to CPU automatically.
/// </summary>
public class CudaBackend : IComputeBackend, IDisposable
{
    private readonly IntPtr _handle;
    private readonly bool _available;

    public string Name => _available ? $"CUDA (Device: {DeviceName})" : "CUDA (unavailable, using CPU fallback)";
    public string DeviceName { get; private set; } = "none";
    public bool IsAvailable => _available;

    private readonly CpuBackend _cpuFallback = new();

    public CudaBackend(int deviceId = 0)
    {
        try
        {
            _handle = NativeMethods.cuda_init(deviceId);
            _available = _handle != IntPtr.Zero;
            if (_available)
            {
                var nameBuffer = new byte[256];
                NativeMethods.cuda_device_name(_handle, nameBuffer, nameBuffer.Length);
                DeviceName = System.Text.Encoding.UTF8.GetString(nameBuffer).TrimEnd('\0');
            }
        }
        catch (DllNotFoundException)
        {
            _available = false;
        }
    }

    public void MatVecMul(ReadOnlySpan<float> mat, ReadOnlySpan<float> vec, Span<float> result, int rows, int cols)
    {
        if (!_available) { _cpuFallback.MatVecMul(mat, vec, result, rows, cols); return; }

        unsafe
        {
            fixed (float* matPtr = mat, vecPtr = vec, resPtr = result)
            {
                NativeMethods.cuda_mat_vec_mul(_handle, matPtr, vecPtr, resPtr, rows, cols);
            }
        }
    }

    public void MatMul(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> c, int M, int K, int N)
    {
        if (!_available) { _cpuFallback.MatMul(a, b, c, M, K, N); return; }

        unsafe
        {
            fixed (float* aPtr = a, bPtr = b, cPtr = c)
            {
                NativeMethods.cuda_mat_mul(_handle, aPtr, bPtr, cPtr, M, K, N);
            }
        }
    }

    public void Softmax(Span<float> x)
    {
        if (!_available) { _cpuFallback.Softmax(x); return; }

        unsafe
        {
            fixed (float* ptr = x)
            {
                NativeMethods.cuda_softmax(_handle, ptr, x.Length);
            }
        }
    }

    public void RmsNorm(Span<float> output, ReadOnlySpan<float> x, ReadOnlySpan<float> weight, float eps)
    {
        if (!_available) { _cpuFallback.RmsNorm(output, x, weight, eps); return; }

        unsafe
        {
            fixed (float* outPtr = output, xPtr = x, wPtr = weight)
            {
                NativeMethods.cuda_rms_norm(_handle, outPtr, xPtr, wPtr, x.Length, eps);
            }
        }
    }

    public void SiLU(Span<float> x)
    {
        if (!_available) { _cpuFallback.SiLU(x); return; }

        unsafe
        {
            fixed (float* ptr = x)
            {
                NativeMethods.cuda_silu(_handle, ptr, x.Length);
            }
        }
    }

    public void AddInPlace(Span<float> a, ReadOnlySpan<float> b)
    {
        if (!_available) { _cpuFallback.AddInPlace(a, b); return; }

        unsafe
        {
            fixed (float* aPtr = a, bPtr = b)
            {
                NativeMethods.cuda_add_inplace(_handle, aPtr, bPtr, a.Length);
            }
        }
    }

    public void Dispose()
    {
        if (_available && _handle != IntPtr.Zero)
            NativeMethods.cuda_cleanup(_handle);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// P/Invoke declarations for the native CUDA library.
    /// The native library must export these C functions.
    /// </summary>
    private static class NativeMethods
    {
        private const string LibName = "sharpinfer_cuda";

        [DllImport(LibName)] public static extern IntPtr cuda_init(int deviceId);
        [DllImport(LibName)] public static extern void cuda_cleanup(IntPtr handle);
        [DllImport(LibName)] public static extern void cuda_device_name(IntPtr handle, byte[] buffer, int bufLen);

        [DllImport(LibName)] public static unsafe extern void cuda_mat_vec_mul(
            IntPtr handle, float* mat, float* vec, float* result, int rows, int cols);

        [DllImport(LibName)] public static unsafe extern void cuda_mat_mul(
            IntPtr handle, float* a, float* b, float* c, int M, int K, int N);

        [DllImport(LibName)] public static unsafe extern void cuda_softmax(
            IntPtr handle, float* x, int len);

        [DllImport(LibName)] public static unsafe extern void cuda_rms_norm(
            IntPtr handle, float* output, float* x, float* weight, int size, float eps);

        [DllImport(LibName)] public static unsafe extern void cuda_silu(
            IntPtr handle, float* x, int len);

        [DllImport(LibName)] public static unsafe extern void cuda_add_inplace(
            IntPtr handle, float* a, float* b, int len);
    }
}
