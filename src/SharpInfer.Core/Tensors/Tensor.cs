using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace SharpInfer.Core.Tensors;

/// <summary>
/// A dense tensor backed by a contiguous float array.
/// Supports SIMD-accelerated operations for hot-path inference math.
/// </summary>
public class Tensor : IDisposable
{
    private float[] _data;
    private bool _disposed;

    public int[] Shape { get; }
    public int Length => _data.Length;
    public ReadOnlySpan<float> Data => _data;
    public Span<float> MutableData => _data;

    public Tensor(int[] shape)
    {
        Shape = shape;
        _data = new float[shape.Aggregate(1, (a, b) => a * b)];
    }

    public Tensor(int[] shape, float[] data)
    {
        if (data.Length != shape.Aggregate(1, (a, b) => a * b))
            throw new ArgumentException("Data length does not match shape.");
        Shape = shape;
        _data = data;
    }

    /// <summary>Creates a tensor from a span, copying the data.</summary>
    public static Tensor FromSpan(int[] shape, ReadOnlySpan<float> data)
    {
        var tensor = new Tensor(shape);
        data.CopyTo(tensor._data);
        return tensor;
    }

    /// <summary>Creates a tensor filled with zeros.</summary>
    public static Tensor Zeros(params int[] shape) => new(shape);

    /// <summary>Creates a tensor filled with a constant value.</summary>
    public static Tensor Full(float value, params int[] shape)
    {
        var tensor = new Tensor(shape);
        Array.Fill(tensor._data, value);
        return tensor;
    }

    public float this[int index]
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => _data[index];
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        set => _data[index] = value;
    }

    /// <summary>Get a slice view into a specific row of a 2D tensor.</summary>
    public Span<float> Row(int row)
    {
        int cols = Shape[^1];
        return _data.AsSpan(row * cols, cols);
    }

    /// <summary>Reshape the tensor (must preserve total element count).</summary>
    public Tensor Reshape(params int[] newShape)
    {
        int newLen = newShape.Aggregate(1, (a, b) => a * b);
        if (newLen != Length)
            throw new ArgumentException($"Cannot reshape {Length} elements into shape [{string.Join(",", newShape)}].");
        return new Tensor(newShape, _data);
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _data = Array.Empty<float>();
            _disposed = true;
        }
        GC.SuppressFinalize(this);
    }
}
