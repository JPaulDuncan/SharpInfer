using System.IO.Compression;
using SharpInfer.Core.Models;

namespace SharpInfer.Core.Engine;

/// <summary>
/// Persistent KV cache serialization for prompt caching.
///
/// Saves the KV cache state to disk after processing a system prompt or
/// conversation prefix, and restores it instantly on subsequent runs.
/// This avoids re-processing expensive prefill operations.
///
/// Use cases:
///   - System prompts (same prompt used across many conversations)
///   - Long documents (process once, query multiple times)
///   - Multi-turn conversations (resume from last state)
///
/// File format: GZip-compressed binary
///   [4 bytes: magic "SIPC"]
///   [4 bytes: version]
///   [4 bytes: num_layers]
///   [4 bytes: kv_heads]
///   [4 bytes: head_dim]
///   [4 bytes: cached_length]
///   [32 bytes: SHA-256 hash of the token sequence (for validation)]
///   [N bytes: key cache data (float32)]
///   [N bytes: value cache data (float32)]
/// </summary>
public class PromptCache
{
    private const uint Magic = 0x43504953; // "SIPC" — SharpInfer Prompt Cache
    private const uint Version = 1;

    private readonly string _cacheDir;

    public PromptCache(string cacheDirectory)
    {
        _cacheDir = cacheDirectory;
        Directory.CreateDirectory(_cacheDir);
    }

    /// <summary>
    /// Save the current KV cache state to disk, associated with a token sequence hash.
    /// </summary>
    public void Save(KVCache cache, List<int> tokenSequence, string label = "default")
    {
        string path = GetCachePath(label);
        byte[] tokenHash = ComputeTokenHash(tokenSequence);

        using var fileStream = File.Create(path);
        using var gzip = new GZipStream(fileStream, CompressionLevel.Fastest);
        using var writer = new BinaryWriter(gzip);

        writer.Write(Magic);
        writer.Write(Version);
        writer.Write(cache.MaxLength);    // for validation
        writer.Write(cache.CurrentLength);
        writer.Write(tokenHash);

        // Serialize the raw cache arrays
        cache.SerializeTo(writer);

        writer.Flush();
    }

    /// <summary>
    /// Attempt to load a cached KV state. Returns true if successful.
    /// Validates that the token sequence matches the cached state.
    /// </summary>
    public bool TryLoad(KVCache cache, List<int> tokenSequence, string label = "default")
    {
        string path = GetCachePath(label);
        if (!File.Exists(path)) return false;

        try
        {
            byte[] expectedHash = ComputeTokenHash(tokenSequence);

            using var fileStream = File.OpenRead(path);
            using var gzip = new GZipStream(fileStream, CompressionMode.Decompress);
            using var reader = new BinaryReader(gzip);

            uint magic = reader.ReadUInt32();
            if (magic != Magic) return false;

            uint version = reader.ReadUInt32();
            if (version != Version) return false;

            int maxLength = reader.ReadInt32();
            if (maxLength != cache.MaxLength) return false;

            int cachedLength = reader.ReadInt32();
            byte[] storedHash = reader.ReadBytes(32);

            // Validate token sequence matches
            if (!storedHash.AsSpan().SequenceEqual(expectedHash))
                return false;

            cache.DeserializeFrom(reader, cachedLength);
            return true;
        }
        catch
        {
            return false;
        }
    }

    /// <summary>Check if a cache exists for the given label.</summary>
    public bool Exists(string label = "default") => File.Exists(GetCachePath(label));

    /// <summary>Delete a cached state.</summary>
    public void Delete(string label = "default")
    {
        var path = GetCachePath(label);
        if (File.Exists(path)) File.Delete(path);
    }

    /// <summary>List all cached prompt states.</summary>
    public string[] ListCaches() =>
        Directory.GetFiles(_cacheDir, "*.sipc")
            .Select(f => Path.GetFileNameWithoutExtension(f))
            .ToArray();

    private string GetCachePath(string label) =>
        Path.Combine(_cacheDir, $"{SanitizeLabel(label)}.sipc");

    private static string SanitizeLabel(string label) =>
        string.Concat(label.Select(c => char.IsLetterOrDigit(c) || c == '-' || c == '_' ? c : '_'));

    private static byte[] ComputeTokenHash(List<int> tokens)
    {
        using var sha = System.Security.Cryptography.SHA256.Create();
        var bytes = new byte[tokens.Count * 4];
        for (int i = 0; i < tokens.Count; i++)
            BitConverter.GetBytes(tokens[i]).CopyTo(bytes, i * 4);
        return sha.ComputeHash(bytes);
    }
}

// Extension methods for KVCache serialization
public static class KVCacheSerializationExtensions
{
    /// <summary>Write all cache data to a binary writer.</summary>
    public static void SerializeTo(this KVCache cache, BinaryWriter writer)
    {
        // We need access to the raw arrays — add this method to KVCache
        writer.Write(cache.CurrentLength);
        cache.WriteRawData(writer);
    }

    /// <summary>Read cache data from a binary reader.</summary>
    public static void DeserializeFrom(this KVCache cache, BinaryReader reader, int length)
    {
        cache.ReadRawData(reader, length);
    }
}
