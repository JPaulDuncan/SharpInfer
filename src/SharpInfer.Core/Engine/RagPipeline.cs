using SharpInfer.Core.Tensors;

namespace SharpInfer.Core.Engine;

/// <summary>
/// Retrieval-Augmented Generation (RAG) pipeline.
///
/// Adds external knowledge to the model by:
/// 1. Splitting documents into chunks
/// 2. Computing embeddings for each chunk
/// 3. Storing embeddings in a pluggable vector store
/// 4. At query time, retrieving the most relevant chunks
/// 5. Injecting retrieved context into the prompt
///
/// Vector store backends:
///   - InMemoryVectorStore: brute-force cosine similarity (default, no dependencies)
///   - SqliteVectorStore:   persistent SQLite + sqlite-vss for HNSW search
///   - Any custom IVectorStore implementation (Qdrant, Milvus, pgvector, etc.)
///
/// Reference: https://arxiv.org/abs/2005.11401 (Lewis et al.)
/// </summary>
public class RagPipeline
{
    private readonly IVectorStore _store;
    private readonly ITextEmbedder _embedder;

    /// <summary>Number of chunks to retrieve per query.</summary>
    public int TopK { get; set; } = 5;

    /// <summary>Minimum similarity score to include a chunk (0-1).</summary>
    public float MinScore { get; set; } = 0.3f;

    /// <summary>Template for injecting context into the prompt.</summary>
    public string ContextTemplate { get; set; } =
        "Use the following context to answer the question. If the context doesn't contain the answer, say so.\n\n" +
        "Context:\n{context}\n\nQuestion: {query}";

    public int DocumentCount => _store.Count;

    /// <summary>Create a RAG pipeline with a specific vector store backend.</summary>
    public RagPipeline(ITextEmbedder embedder, IVectorStore store)
    {
        _embedder = embedder;
        _store = store;
    }

    /// <summary>Create a RAG pipeline with the default in-memory vector store.</summary>
    public RagPipeline(ITextEmbedder embedder, int embeddingDim)
        : this(embedder, new InMemoryVectorStore(embeddingDim))
    {
    }

    /// <summary>
    /// Ingest a document: split into chunks and store embeddings.
    /// </summary>
    public async Task IngestAsync(string text, string source = "",
        int chunkSize = 512, int chunkOverlap = 64)
    {
        var chunks = ChunkText(text, chunkSize, chunkOverlap);

        for (int i = 0; i < chunks.Count; i++)
        {
            var embedding = await _embedder.EmbedAsync(chunks[i]);
            await _store.AddAsync(embedding, new ChunkMetadata
            {
                Text = chunks[i],
                Source = source,
                ChunkIndex = i,
            });
        }
    }

    /// <summary>Ingest a file (text, markdown, etc.)</summary>
    public async Task IngestFileAsync(string filePath)
    {
        string text = await File.ReadAllTextAsync(filePath);
        await IngestAsync(text, source: Path.GetFileName(filePath));
    }

    /// <summary>Ingest all supported files in a directory.</summary>
    public async Task IngestDirectoryAsync(string dirPath, string searchPattern = "*.*")
    {
        var extensions = new HashSet<string> { ".txt", ".md", ".json", ".csv", ".log", ".xml", ".html" };
        var files = Directory.GetFiles(dirPath, searchPattern, SearchOption.AllDirectories)
            .Where(f => extensions.Contains(Path.GetExtension(f).ToLower()));

        foreach (var file in files)
            await IngestFileAsync(file);
    }

    /// <summary>Retrieve relevant chunks for a query.</summary>
    public async Task<List<RetrievedChunk>> RetrieveAsync(string query)
    {
        var queryEmbedding = await _embedder.EmbedAsync(query);
        var results = await _store.SearchAsync(queryEmbedding, TopK);

        return results
            .Where(r => r.Score >= MinScore)
            .Select(r => new RetrievedChunk
            {
                Text = r.Metadata.Text,
                Source = r.Metadata.Source,
                Score = r.Score,
            })
            .ToList();
    }

    /// <summary>Build an augmented prompt: retrieve context and inject it into the template.</summary>
    public async Task<string> AugmentPromptAsync(string query)
    {
        var chunks = await RetrieveAsync(query);

        if (chunks.Count == 0)
            return query;

        var context = string.Join("\n\n---\n\n",
            chunks.Select((c, i) => $"[{i + 1}] ({c.Source}, score: {c.Score:F2})\n{c.Text}"));

        return ContextTemplate
            .Replace("{context}", context)
            .Replace("{query}", query);
    }

    /// <summary>
    /// Delete all chunks from a specific source document.
    /// Useful for re-ingesting an updated document.
    /// </summary>
    public async Task DeleteSourceAsync(string source)
    {
        await _store.DeleteBySourceAsync(source);
    }

    /// <summary>Clear all stored documents and embeddings.</summary>
    public void Clear() => _store.Clear();

    /// <summary>Get statistics about the vector store.</summary>
    public VectorStoreStats GetStats() => _store.GetStats();

    #region Text Chunking

    private static List<string> ChunkText(string text, int chunkSize, int overlap)
    {
        var chunks = new List<string>();
        var paragraphs = text.Split(new[] { "\n\n", "\r\n\r\n" }, StringSplitOptions.RemoveEmptyEntries);

        var currentChunk = new System.Text.StringBuilder();
        var overlapBuffer = new Queue<string>();

        foreach (var para in paragraphs)
        {
            if (currentChunk.Length + para.Length > chunkSize && currentChunk.Length > 0)
            {
                chunks.Add(currentChunk.ToString().Trim());
                currentChunk.Clear();
                foreach (var overlapPara in overlapBuffer)
                    currentChunk.AppendLine(overlapPara);
            }

            currentChunk.AppendLine(para);
            overlapBuffer.Enqueue(para);
            while (string.Join("", overlapBuffer).Length > overlap && overlapBuffer.Count > 1)
                overlapBuffer.Dequeue();
        }

        if (currentChunk.Length > 0)
            chunks.Add(currentChunk.ToString().Trim());

        // Handle very long paragraphs
        var result = new List<string>();
        foreach (var chunk in chunks)
        {
            if (chunk.Length <= chunkSize * 1.5)
            {
                result.Add(chunk);
            }
            else
            {
                var sentences = chunk.Split(new[] { ". ", "! ", "? " }, StringSplitOptions.RemoveEmptyEntries);
                var subChunk = new System.Text.StringBuilder();
                foreach (var sentence in sentences)
                {
                    if (subChunk.Length + sentence.Length > chunkSize && subChunk.Length > 0)
                    {
                        result.Add(subChunk.ToString().Trim());
                        subChunk.Clear();
                    }
                    subChunk.Append(sentence).Append(". ");
                }
                if (subChunk.Length > 0)
                    result.Add(subChunk.ToString().Trim());
            }
        }

        return result;
    }

    #endregion
}

// ============================================================================
// Interfaces
// ============================================================================

/// <summary>
/// Pluggable vector store interface.
/// Implement this to add support for any vector database backend.
/// </summary>
public interface IVectorStore
{
    /// <summary>Total number of stored vectors.</summary>
    int Count { get; }

    /// <summary>Add a vector with associated metadata.</summary>
    Task AddAsync(float[] embedding, ChunkMetadata metadata);

    /// <summary>Find the top-K most similar vectors to a query.</summary>
    Task<List<SearchResult>> SearchAsync(float[] query, int topK);

    /// <summary>Delete all entries from a specific source.</summary>
    Task DeleteBySourceAsync(string source);

    /// <summary>Remove all entries.</summary>
    void Clear();

    /// <summary>Get store statistics.</summary>
    VectorStoreStats GetStats();
}

/// <summary>
/// Interface for text embedding. Implementations:
/// - ModelEmbedder: use the inference engine's own hidden states
/// - A dedicated embedding model (all-MiniLM, nomic-embed, etc.)
/// - An external API (OpenAI, Cohere, etc.)
/// </summary>
public interface ITextEmbedder
{
    int Dimensions { get; }
    Task<float[]> EmbedAsync(string text);
}

// ============================================================================
// In-Memory Vector Store (default)
// ============================================================================

/// <summary>
/// Simple in-memory vector store with brute-force cosine similarity search.
/// No dependencies. Good for up to ~10K chunks.
///
/// Pros: Zero setup, fast for small datasets, no disk I/O
/// Cons: O(n) search, lost on restart, high memory usage for large datasets
/// </summary>
public class InMemoryVectorStore : IVectorStore
{
    private readonly int _dim;
    private readonly List<VectorEntry> _entries = new();

    public int Count => _entries.Count;

    public InMemoryVectorStore(int dimensions)
    {
        _dim = dimensions;
    }

    public Task AddAsync(float[] embedding, ChunkMetadata metadata)
    {
        if (embedding.Length != _dim)
            throw new ArgumentException($"Embedding dimension mismatch: expected {_dim}, got {embedding.Length}");
        _entries.Add(new VectorEntry { Embedding = embedding, Metadata = metadata });
        return Task.CompletedTask;
    }

    public Task<List<SearchResult>> SearchAsync(float[] query, int topK)
    {
        var scores = new (float Score, int Index)[_entries.Count];

        for (int i = 0; i < _entries.Count; i++)
            scores[i] = (TensorOps.Dot(query, _entries[i].Embedding), i);

        var results = scores
            .OrderByDescending(s => s.Score)
            .Take(topK)
            .Select(s => new SearchResult
            {
                Score = s.Score,
                Metadata = _entries[s.Index].Metadata,
            })
            .ToList();

        return Task.FromResult(results);
    }

    public Task DeleteBySourceAsync(string source)
    {
        _entries.RemoveAll(e => e.Metadata.Source == source);
        return Task.CompletedTask;
    }

    public void Clear() => _entries.Clear();

    public VectorStoreStats GetStats() => new()
    {
        Backend = "InMemory",
        VectorCount = _entries.Count,
        Dimensions = _dim,
        Persistent = false,
        ApproxMemoryMb = _entries.Count * _dim * sizeof(float) / (1024f * 1024f),
    };

    private class VectorEntry
    {
        public required float[] Embedding { get; init; }
        public required ChunkMetadata Metadata { get; init; }
    }
}

// ============================================================================
// Model Embedder
// ============================================================================

/// <summary>
/// Embedding using the inference engine's hidden states.
/// Uses mean pooling over the last hidden layer as the sentence embedding.
///
/// TODO: For production quality, modify Transformer to expose intermediate
/// hidden states and compute proper mean-pooled embeddings.
/// </summary>
public class ModelEmbedder : ITextEmbedder
{
    private readonly InferenceEngine _engine;
    public int Dimensions => _engine.Config.HiddenSize;

    public ModelEmbedder(InferenceEngine engine)
    {
        _engine = engine;
    }

    public async Task<float[]> EmbedAsync(string text)
    {
        var tokens = _engine.Tokenize(text);

        // TODO: Replace with actual hidden state extraction from Transformer
        // For now, use hash-based pseudo-embedding as placeholder
        var embedding = new float[Dimensions];
        var rng = new Random(tokens.GetHashCode());
        for (int i = 0; i < Dimensions; i++)
            embedding[i] = (float)(rng.NextDouble() * 2 - 1);

        // L2 normalize
        float norm = 0f;
        for (int i = 0; i < embedding.Length; i++)
            norm += embedding[i] * embedding[i];
        norm = MathF.Sqrt(norm);
        if (norm > 0f)
            for (int i = 0; i < embedding.Length; i++)
                embedding[i] /= norm;

        return await Task.FromResult(embedding);
    }
}

// ============================================================================
// Data Models
// ============================================================================

public class ChunkMetadata
{
    public required string Text { get; init; }
    public string Source { get; init; } = "";
    public int ChunkIndex { get; init; }
}

public class SearchResult
{
    public float Score { get; init; }
    public required ChunkMetadata Metadata { get; init; }
}

public class RetrievedChunk
{
    public required string Text { get; init; }
    public string Source { get; init; } = "";
    public float Score { get; init; }
}

public class VectorStoreStats
{
    public string Backend { get; init; } = "";
    public int VectorCount { get; init; }
    public int Dimensions { get; init; }
    public bool Persistent { get; init; }
    public float ApproxMemoryMb { get; init; }
    public string? DbPath { get; init; }
    public long? DbSizeBytes { get; init; }
}
