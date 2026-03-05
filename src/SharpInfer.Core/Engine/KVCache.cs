using SharpInfer.Core.Models;

namespace SharpInfer.Core.Engine;

/// <summary>
/// Key-Value cache for transformer attention.
/// Stores the projected K and V vectors for each layer and position,
/// enabling autoregressive generation without recomputing attention
/// over the full sequence at each step.
///
/// Memory layout: [layer][position * numKvHeads * headDim]
/// </summary>
public class KVCache : IDisposable
{
    private readonly float[][] _keyCache;    // [layer] -> flat array [maxSeq * numKvHeads * headDim]
    private readonly float[][] _valueCache;  // [layer] -> flat array [maxSeq * numKvHeads * headDim]
    private readonly int _numKvHeads;
    private readonly int _headDim;
    private readonly int _maxSeqLen;
    private readonly int _stridePerPosition; // numKvHeads * headDim
    private int _currentLength;

    public int CurrentLength => _currentLength;
    public int MaxLength => _maxSeqLen;

    public KVCache(ModelConfig config)
    {
        _numKvHeads = config.NumKvHeads;
        _headDim = config.HeadDim;
        _maxSeqLen = config.MaxSequenceLength;
        _stridePerPosition = _numKvHeads * _headDim;

        _keyCache = new float[config.NumLayers][];
        _valueCache = new float[config.NumLayers][];

        for (int i = 0; i < config.NumLayers; i++)
        {
            _keyCache[i] = new float[_maxSeqLen * _stridePerPosition];
            _valueCache[i] = new float[_maxSeqLen * _stridePerPosition];
        }
    }

    /// <summary>
    /// Store K and V vectors for a given layer and position.
    /// </summary>
    public void Store(int layer, int position, ReadOnlySpan<float> key, ReadOnlySpan<float> value)
    {
        if (position >= _maxSeqLen)
            throw new InvalidOperationException($"Position {position} exceeds max sequence length {_maxSeqLen}.");

        int offset = position * _stridePerPosition;
        key[.._stridePerPosition].CopyTo(_keyCache[layer].AsSpan(offset, _stridePerPosition));
        value[.._stridePerPosition].CopyTo(_valueCache[layer].AsSpan(offset, _stridePerPosition));

        if (position >= _currentLength)
            _currentLength = position + 1;
    }

    /// <summary>Get the cached key vector for a specific layer, position, and KV head.</summary>
    public ReadOnlySpan<float> GetKey(int layer, int position, int kvHead, int headDim)
    {
        int offset = position * _stridePerPosition + kvHead * headDim;
        return _keyCache[layer].AsSpan(offset, headDim);
    }

    /// <summary>Get the cached value vector for a specific layer, position, and KV head.</summary>
    public ReadOnlySpan<float> GetValue(int layer, int position, int kvHead, int headDim)
    {
        int offset = position * _stridePerPosition + kvHead * headDim;
        return _valueCache[layer].AsSpan(offset, headDim);
    }

    /// <summary>Reset the cache (e.g., for a new conversation).</summary>
    public void Clear()
    {
        for (int i = 0; i < _keyCache.Length; i++)
        {
            Array.Clear(_keyCache[i]);
            Array.Clear(_valueCache[i]);
        }
        _currentLength = 0;
    }

    /// <summary>Write raw cache data to a binary writer (for prompt cache serialization).</summary>
    public void WriteRawData(BinaryWriter writer)
    {
        int dataLen = _currentLength * _stridePerPosition;
        for (int layer = 0; layer < _keyCache.Length; layer++)
        {
            var keyBytes = new byte[dataLen * sizeof(float)];
            Buffer.BlockCopy(_keyCache[layer], 0, keyBytes, 0, keyBytes.Length);
            writer.Write(keyBytes);

            var valBytes = new byte[dataLen * sizeof(float)];
            Buffer.BlockCopy(_valueCache[layer], 0, valBytes, 0, valBytes.Length);
            writer.Write(valBytes);
        }
    }

    /// <summary>Read raw cache data from a binary reader (for prompt cache deserialization).</summary>
    public void ReadRawData(BinaryReader reader, int length)
    {
        _currentLength = length;
        int dataLen = length * _stridePerPosition;
        for (int layer = 0; layer < _keyCache.Length; layer++)
        {
            var keyBytes = reader.ReadBytes(dataLen * sizeof(float));
            Buffer.BlockCopy(keyBytes, 0, _keyCache[layer], 0, keyBytes.Length);

            var valBytes = reader.ReadBytes(dataLen * sizeof(float));
            Buffer.BlockCopy(valBytes, 0, _valueCache[layer], 0, valBytes.Length);
        }
    }

    public void Dispose()
    {
        // Allow GC to reclaim the large arrays
        for (int i = 0; i < _keyCache.Length; i++)
        {
            _keyCache[i] = Array.Empty<float>();
            _valueCache[i] = Array.Empty<float>();
        }
        GC.SuppressFinalize(this);
    }
}
