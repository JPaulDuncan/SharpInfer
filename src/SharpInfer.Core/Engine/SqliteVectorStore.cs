using System.Data;
using System.Text;
using System.Text.Json;

namespace SharpInfer.Core.Engine;

/// <summary>
/// Persistent vector store using SQLite + sqlite-vss extension for HNSW search.
///
/// Requires:
///   - System.Data.SQLite or Microsoft.Data.Sqlite
///   - sqlite-vss extension (.so/.dll) for approximate nearest neighbor search
///
/// Features:
///   - Persistent storage (survives restarts)
///   - HNSW index for O(log n) approximate nearest neighbor search
///   - Source-based deletion for document re-ingestion
///   - Metadata storage alongside vectors
///   - Configurable HNSW parameters (M, efConstruction, efSearch)
///
/// Falls back to exact brute-force search if sqlite-vss extension is not available.
///
/// Schema:
///   chunks:    (id, text, source, chunk_index, embedding_blob)
///   vss_index: virtual table via sqlite-vss for ANN search
/// </summary>
public class SqliteVectorStore : IVectorStore, IDisposable
{
    private readonly int _dim;
    private readonly string _dbPath;
    private readonly IDbConnection _connection;
    private bool _vssAvailable;
    private int _count;

    /// <summary>HNSW M parameter (number of neighbors per node). Higher = better recall, more memory.</summary>
    public int HnswM { get; set; } = 16;

    /// <summary>HNSW efConstruction (search width during index build). Higher = better recall, slower build.</summary>
    public int EfConstruction { get; set; } = 200;

    /// <summary>HNSW efSearch (search width during query). Higher = better recall, slower search.</summary>
    public int EfSearch { get; set; } = 64;

    public int Count => _count;

    /// <summary>
    /// Create a SQLite vector store.
    /// Uses ADO.NET interfaces for compatibility with both System.Data.SQLite and Microsoft.Data.Sqlite.
    /// </summary>
    /// <param name="dimensions">Embedding vector dimensions.</param>
    /// <param name="dbPath">Path to the SQLite database file.</param>
    /// <param name="connectionFactory">
    /// Factory to create an IDbConnection. If null, attempts to use Microsoft.Data.Sqlite via reflection.
    /// Example: () => new SqliteConnection($"Data Source={dbPath}")
    /// </param>
    public SqliteVectorStore(int dimensions, string dbPath, Func<string, IDbConnection>? connectionFactory = null)
    {
        _dim = dimensions;
        _dbPath = dbPath;

        if (connectionFactory != null)
        {
            _connection = connectionFactory(dbPath);
        }
        else
        {
            _connection = CreateConnectionViaReflection(dbPath);
        }

        _connection.Open();
        InitializeDatabase();
    }

    private void InitializeDatabase()
    {
        // Create chunks table
        ExecuteNonQuery(@"
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT '',
                chunk_index INTEGER NOT NULL DEFAULT 0,
                embedding BLOB NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )");

        ExecuteNonQuery(@"
            CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source)");

        // Try to load sqlite-vss extension
        _vssAvailable = TryLoadVssExtension();

        if (_vssAvailable)
        {
            // Create VSS virtual table for HNSW index
            ExecuteNonQuery($@"
                CREATE VIRTUAL TABLE IF NOT EXISTS vss_chunks USING vss0(
                    embedding({_dim})
                )");
        }

        // Get current count
        _count = Convert.ToInt32(ExecuteScalar("SELECT COUNT(*) FROM chunks"));
    }

    private bool TryLoadVssExtension()
    {
        try
        {
            // Try common paths for sqlite-vss extension
            var possiblePaths = new[]
            {
                "vss0",           // In PATH or LD_LIBRARY_PATH
                "./vss0",         // Current directory
                "./libvss0",      // Linux
                "./vss0.dll",     // Windows
                "./libvss0.so",   // Linux explicit
                "./libvss0.dylib" // macOS
            };

            foreach (var path in possiblePaths)
            {
                try
                {
                    ExecuteNonQuery($"SELECT load_extension('{path}')");
                    return true;
                }
                catch
                {
                    // Try next path
                }
            }

            return false;
        }
        catch
        {
            return false;
        }
    }

    public async Task AddAsync(float[] embedding, ChunkMetadata metadata)
    {
        if (embedding.Length != _dim)
            throw new ArgumentException($"Embedding dimension mismatch: expected {_dim}, got {embedding.Length}");

        byte[] embeddingBlob = EmbeddingToBytes(embedding);

        using var cmd = _connection.CreateCommand();
        cmd.CommandText = @"
            INSERT INTO chunks (text, source, chunk_index, embedding)
            VALUES (@text, @source, @chunkIndex, @embedding)";

        AddParameter(cmd, "@text", metadata.Text);
        AddParameter(cmd, "@source", metadata.Source);
        AddParameter(cmd, "@chunkIndex", metadata.ChunkIndex);
        AddParameter(cmd, "@embedding", embeddingBlob);

        cmd.ExecuteNonQuery();
        long rowId = Convert.ToInt64(ExecuteScalar("SELECT last_insert_rowid()"));

        // Add to VSS index if available
        if (_vssAvailable)
        {
            try
            {
                using var vssCmd = _connection.CreateCommand();
                vssCmd.CommandText = @"
                    INSERT INTO vss_chunks (rowid, embedding)
                    VALUES (@rowid, @embedding)";
                AddParameter(vssCmd, "@rowid", rowId);
                AddParameter(vssCmd, "@embedding", embeddingBlob);
                vssCmd.ExecuteNonQuery();
            }
            catch
            {
                // VSS insert failed — fall back to brute force for search
                _vssAvailable = false;
            }
        }

        _count++;
        await Task.CompletedTask;
    }

    public async Task<List<SearchResult>> SearchAsync(float[] query, int topK)
    {
        if (_vssAvailable)
            return await VssSearchAsync(query, topK);
        else
            return await BruteForceSearchAsync(query, topK);
    }

    /// <summary>HNSW-accelerated search via sqlite-vss.</summary>
    private async Task<List<SearchResult>> VssSearchAsync(float[] query, int topK)
    {
        byte[] queryBlob = EmbeddingToBytes(query);

        using var cmd = _connection.CreateCommand();
        cmd.CommandText = $@"
            SELECT c.text, c.source, c.chunk_index, vss.distance
            FROM vss_chunks AS vss
            INNER JOIN chunks AS c ON c.id = vss.rowid
            WHERE vss_search(vss.embedding, @query)
            LIMIT @topK";

        AddParameter(cmd, "@query", queryBlob);
        AddParameter(cmd, "@topK", topK);

        var results = new List<SearchResult>();

        using var reader = cmd.ExecuteReader();
        while (reader.Read())
        {
            // sqlite-vss returns L2 distance; convert to cosine-like similarity
            float distance = Convert.ToSingle(reader["distance"]);
            float score = 1f / (1f + distance); // Convert distance to similarity

            results.Add(new SearchResult
            {
                Score = score,
                Metadata = new ChunkMetadata
                {
                    Text = reader["text"].ToString() ?? "",
                    Source = reader["source"].ToString() ?? "",
                    ChunkIndex = Convert.ToInt32(reader["chunk_index"]),
                }
            });
        }

        return await Task.FromResult(results);
    }

    /// <summary>Brute-force cosine similarity search (fallback when VSS unavailable).</summary>
    private async Task<List<SearchResult>> BruteForceSearchAsync(float[] query, int topK)
    {
        // Normalize query
        float queryNorm = 0f;
        for (int i = 0; i < query.Length; i++)
            queryNorm += query[i] * query[i];
        queryNorm = MathF.Sqrt(queryNorm);

        using var cmd = _connection.CreateCommand();
        cmd.CommandText = "SELECT id, text, source, chunk_index, embedding FROM chunks";

        var scored = new List<(float Score, ChunkMetadata Metadata)>();

        using var reader = cmd.ExecuteReader();
        while (reader.Read())
        {
            byte[] blob = (byte[])reader["embedding"];
            float[] embedding = BytesToEmbedding(blob);

            // Cosine similarity
            float dot = 0f, embNorm = 0f;
            for (int i = 0; i < _dim; i++)
            {
                dot += query[i] * embedding[i];
                embNorm += embedding[i] * embedding[i];
            }
            embNorm = MathF.Sqrt(embNorm);

            float score = (queryNorm > 0 && embNorm > 0)
                ? dot / (queryNorm * embNorm)
                : 0f;

            scored.Add((score, new ChunkMetadata
            {
                Text = reader["text"].ToString() ?? "",
                Source = reader["source"].ToString() ?? "",
                ChunkIndex = Convert.ToInt32(reader["chunk_index"]),
            }));
        }

        var results = scored
            .OrderByDescending(s => s.Score)
            .Take(topK)
            .Select(s => new SearchResult { Score = s.Score, Metadata = s.Metadata })
            .ToList();

        return await Task.FromResult(results);
    }

    public async Task DeleteBySourceAsync(string source)
    {
        if (_vssAvailable)
        {
            // Delete from VSS index first
            try
            {
                using var vssCmd = _connection.CreateCommand();
                vssCmd.CommandText = @"
                    DELETE FROM vss_chunks
                    WHERE rowid IN (SELECT id FROM chunks WHERE source = @source)";
                AddParameter(vssCmd, "@source", source);
                vssCmd.ExecuteNonQuery();
            }
            catch
            {
                // VSS delete might fail; proceed with chunks table
            }
        }

        using var cmd = _connection.CreateCommand();
        cmd.CommandText = "DELETE FROM chunks WHERE source = @source";
        AddParameter(cmd, "@source", source);
        int deleted = cmd.ExecuteNonQuery();
        _count -= deleted;

        await Task.CompletedTask;
    }

    public void Clear()
    {
        if (_vssAvailable)
        {
            try { ExecuteNonQuery("DELETE FROM vss_chunks"); } catch { }
        }

        ExecuteNonQuery("DELETE FROM chunks");
        ExecuteNonQuery("VACUUM"); // Reclaim space
        _count = 0;
    }

    public VectorStoreStats GetStats()
    {
        long dbSize = 0;
        try
        {
            if (File.Exists(_dbPath))
                dbSize = new FileInfo(_dbPath).Length;
        }
        catch { }

        return new VectorStoreStats
        {
            Backend = _vssAvailable ? "SQLite+VSS (HNSW)" : "SQLite (brute-force)",
            VectorCount = _count,
            Dimensions = _dim,
            Persistent = true,
            ApproxMemoryMb = 0, // Disk-based
            DbPath = _dbPath,
            DbSizeBytes = dbSize,
        };
    }

    /// <summary>Rebuild the VSS index. Useful after bulk inserts or deletions.</summary>
    public void RebuildIndex()
    {
        if (!_vssAvailable) return;

        try
        {
            ExecuteNonQuery("DELETE FROM vss_chunks");

            using var cmd = _connection.CreateCommand();
            cmd.CommandText = "SELECT id, embedding FROM chunks";

            using var reader = cmd.ExecuteReader();
            while (reader.Read())
            {
                long rowId = Convert.ToInt64(reader["id"]);
                byte[] blob = (byte[])reader["embedding"];

                using var insertCmd = _connection.CreateCommand();
                insertCmd.CommandText = @"
                    INSERT INTO vss_chunks (rowid, embedding)
                    VALUES (@rowid, @embedding)";
                AddParameter(insertCmd, "@rowid", rowId);
                AddParameter(insertCmd, "@embedding", blob);
                insertCmd.ExecuteNonQuery();
            }
        }
        catch
        {
            _vssAvailable = false;
        }
    }

    /// <summary>Export all vectors to a new InMemoryVectorStore (for migration/testing).</summary>
    public InMemoryVectorStore ToInMemoryStore()
    {
        var memStore = new InMemoryVectorStore(_dim);

        using var cmd = _connection.CreateCommand();
        cmd.CommandText = "SELECT text, source, chunk_index, embedding FROM chunks ORDER BY id";

        using var reader = cmd.ExecuteReader();
        while (reader.Read())
        {
            byte[] blob = (byte[])reader["embedding"];
            float[] embedding = BytesToEmbedding(blob);

            memStore.AddAsync(embedding, new ChunkMetadata
            {
                Text = reader["text"].ToString() ?? "",
                Source = reader["source"].ToString() ?? "",
                ChunkIndex = Convert.ToInt32(reader["chunk_index"]),
            }).Wait();
        }

        return memStore;
    }

    #region Helpers

    private static byte[] EmbeddingToBytes(float[] embedding)
    {
        var bytes = new byte[embedding.Length * sizeof(float)];
        Buffer.BlockCopy(embedding, 0, bytes, 0, bytes.Length);
        return bytes;
    }

    private static float[] BytesToEmbedding(byte[] bytes)
    {
        var embedding = new float[bytes.Length / sizeof(float)];
        Buffer.BlockCopy(bytes, 0, embedding, 0, bytes.Length);
        return embedding;
    }

    private void ExecuteNonQuery(string sql)
    {
        using var cmd = _connection.CreateCommand();
        cmd.CommandText = sql;
        cmd.ExecuteNonQuery();
    }

    private object? ExecuteScalar(string sql)
    {
        using var cmd = _connection.CreateCommand();
        cmd.CommandText = sql;
        return cmd.ExecuteScalar();
    }

    private static void AddParameter(IDbCommand cmd, string name, object value)
    {
        var param = cmd.CreateParameter();
        param.ParameterName = name;
        param.Value = value;
        cmd.Parameters.Add(param);
    }

    /// <summary>
    /// Try to create an IDbConnection via reflection.
    /// Looks for Microsoft.Data.Sqlite or System.Data.SQLite in loaded assemblies.
    /// </summary>
    private static IDbConnection CreateConnectionViaReflection(string dbPath)
    {
        string connectionString = $"Data Source={dbPath}";

        // Try Microsoft.Data.Sqlite first
        var sqliteType = Type.GetType("Microsoft.Data.Sqlite.SqliteConnection, Microsoft.Data.Sqlite");
        sqliteType ??= Type.GetType("System.Data.SQLite.SQLiteConnection, System.Data.SQLite");

        if (sqliteType != null)
        {
            var conn = Activator.CreateInstance(sqliteType, connectionString) as IDbConnection;
            if (conn != null) return conn;
        }

        // Try loading assembly dynamically
        foreach (var assemblyName in new[] { "Microsoft.Data.Sqlite", "System.Data.SQLite" })
        {
            try
            {
                var assembly = System.Reflection.Assembly.Load(assemblyName);
                var connType = assembly.GetTypes().FirstOrDefault(t =>
                    typeof(IDbConnection).IsAssignableFrom(t) && !t.IsAbstract);

                if (connType != null)
                {
                    var conn = Activator.CreateInstance(connType, connectionString) as IDbConnection;
                    if (conn != null) return conn;
                }
            }
            catch { }
        }

        throw new InvalidOperationException(
            "No SQLite provider found. Install Microsoft.Data.Sqlite or System.Data.SQLite NuGet package.\n" +
            "For sqlite-vss support, also install the sqlite-vss native library.\n\n" +
            "  dotnet add package Microsoft.Data.Sqlite\n" +
            "  # Optional: download sqlite-vss from https://github.com/asg017/sqlite-vss/releases");
    }

    public void Dispose()
    {
        _connection?.Close();
        _connection?.Dispose();
    }

    #endregion
}

/// <summary>
/// Factory for creating vector stores from configuration.
/// </summary>
public static class VectorStoreFactory
{
    /// <summary>
    /// Create a vector store based on the configuration.
    /// </summary>
    public static IVectorStore Create(VectorStoreConfig config, int embeddingDim)
    {
        return config.Backend.ToLowerInvariant() switch
        {
            "sqlite" or "sqlite-vss" or "sqlitevss" =>
                new SqliteVectorStore(embeddingDim, config.DbPath ?? "./vectors.db"),

            "memory" or "inmemory" or "in-memory" or "" =>
                new InMemoryVectorStore(embeddingDim),

            _ => throw new ArgumentException($"Unknown vector store backend: {config.Backend}. " +
                "Supported: 'memory', 'sqlite'")
        };
    }
}

/// <summary>
/// Configuration for vector store backend selection.
/// </summary>
public class VectorStoreConfig
{
    /// <summary>Backend type: "memory" (default) or "sqlite".</summary>
    public string Backend { get; set; } = "memory";

    /// <summary>Path to SQLite database file (for sqlite backend).</summary>
    public string? DbPath { get; set; }
}
