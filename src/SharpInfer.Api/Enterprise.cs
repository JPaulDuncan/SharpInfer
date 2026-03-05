using System.Collections.Concurrent;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace SharpInfer.Api;

/// <summary>
/// Enterprise features for production deployments:
///   - API key authentication
///   - Per-key rate limiting (token bucket algorithm)
///   - Request/response audit logging
///   - Usage metering and billing-ready tracking
///   - IP allowlisting / blocklisting
///   - Request size limits
///   - Health monitoring with detailed metrics
///
/// All features are opt-in and disabled by default.
/// </summary>
public class EnterpriseMiddleware
{
    private readonly ApiKeyStore _keyStore;
    private readonly RateLimiter _rateLimiter;
    private readonly AuditLogger _auditLogger;
    private readonly UsageMeter _usageMeter;
    private readonly EnterpriseConfig _config;

    public EnterpriseMiddleware(EnterpriseConfig config)
    {
        _config = config;
        _keyStore = new ApiKeyStore();
        _rateLimiter = new RateLimiter(config.RateLimit);
        _auditLogger = new AuditLogger(config.Audit);
        _usageMeter = new UsageMeter();
    }

    public ApiKeyStore Keys => _keyStore;
    public RateLimiter RateLimiter => _rateLimiter;
    public AuditLogger AuditLogger => _auditLogger;
    public UsageMeter UsageMeter => _usageMeter;

    /// <summary>
    /// Validate an incoming request. Returns null if valid, or an error response if rejected.
    /// </summary>
    public AuthResult Authenticate(string? authHeader, string clientIp)
    {
        // IP filtering
        if (_config.IpAllowlist.Count > 0 && !_config.IpAllowlist.Contains(clientIp))
            return AuthResult.Denied("IP not in allowlist", 403);
        if (_config.IpBlocklist.Contains(clientIp))
            return AuthResult.Denied("IP blocked", 403);

        // Auth disabled — allow all
        if (!_config.AuthEnabled)
            return AuthResult.Allowed("anonymous", "default");

        // Extract bearer token
        if (string.IsNullOrEmpty(authHeader) || !authHeader.StartsWith("Bearer ", StringComparison.OrdinalIgnoreCase))
            return AuthResult.Denied("Missing or invalid Authorization header. Expected: Bearer <api_key>", 401);

        string apiKey = authHeader["Bearer ".Length..].Trim();

        // Validate key
        var keyInfo = _keyStore.Validate(apiKey);
        if (keyInfo == null)
            return AuthResult.Denied("Invalid API key", 401);

        if (!keyInfo.Enabled)
            return AuthResult.Denied("API key is disabled", 403);

        if (keyInfo.ExpiresAt.HasValue && keyInfo.ExpiresAt.Value < DateTime.UtcNow)
            return AuthResult.Denied("API key has expired", 401);

        // Rate limiting
        if (_config.RateLimitEnabled)
        {
            var limit = _rateLimiter.CheckLimit(keyInfo.Id, keyInfo.RateLimit);
            if (!limit.Allowed)
                return AuthResult.RateLimited(limit.RetryAfterSeconds, keyInfo.Id, keyInfo.Tier);
        }

        return AuthResult.Allowed(keyInfo.Id, keyInfo.Tier);
    }

    /// <summary>Record a completed request for metering and audit.</summary>
    public void RecordRequest(RequestRecord record)
    {
        if (_config.AuditEnabled)
            _auditLogger.Log(record);

        if (_config.MeteringEnabled)
            _usageMeter.Record(record.KeyId, record.PromptTokens, record.CompletionTokens, record.DurationMs);
    }
}

public class AuthResult
{
    public bool IsAllowed { get; set; }
    public string? KeyId { get; set; }
    public string? Tier { get; set; }
    public string? Error { get; set; }
    public int StatusCode { get; set; }
    public int? RetryAfterSeconds { get; set; }
    public Dictionary<string, string> Headers { get; set; } = new();

    public static AuthResult Allowed(string keyId, string tier) =>
        new() { IsAllowed = true, KeyId = keyId, Tier = tier };

    public static AuthResult Denied(string error, int statusCode) =>
        new() { IsAllowed = false, Error = error, StatusCode = statusCode };

    public static AuthResult RateLimited(int retryAfter, string keyId, string tier) =>
        new()
        {
            IsAllowed = false,
            Error = "Rate limit exceeded",
            StatusCode = 429,
            RetryAfterSeconds = retryAfter,
            KeyId = keyId,
            Tier = tier,
            Headers = { ["Retry-After"] = retryAfter.ToString() },
        };
}

// === API Key Store ===

public class ApiKeyStore
{
    private readonly ConcurrentDictionary<string, ApiKeyInfo> _keys = new(); // hash → info
    private readonly ConcurrentDictionary<string, string> _keyToHash = new(); // raw key → hash

    /// <summary>Generate a new API key with the given configuration.</summary>
    public (string ApiKey, ApiKeyInfo Info) CreateKey(string name, string tier = "default",
        int? requestsPerMinute = null, DateTime? expiresAt = null)
    {
        // Generate a secure random key
        var keyBytes = RandomNumberGenerator.GetBytes(32);
        string apiKey = $"si-{Convert.ToBase64String(keyBytes).Replace("+", "").Replace("/", "").Replace("=", "")[..48]}";
        string hash = HashKey(apiKey);

        var info = new ApiKeyInfo
        {
            Id = $"key_{Guid.NewGuid():N}"[..20],
            Name = name,
            Tier = tier,
            KeyHash = hash,
            KeyPrefix = apiKey[..10] + "...",
            RateLimit = requestsPerMinute ?? GetDefaultRateLimit(tier),
            ExpiresAt = expiresAt,
            CreatedAt = DateTime.UtcNow,
            Enabled = true,
        };

        _keys[hash] = info;
        _keyToHash[apiKey] = hash;

        return (apiKey, info);
    }

    /// <summary>Validate an API key and return its info if valid.</summary>
    public ApiKeyInfo? Validate(string apiKey)
    {
        string hash = HashKey(apiKey);
        return _keys.TryGetValue(hash, out var info) ? info : null;
    }

    /// <summary>Revoke an API key by its ID.</summary>
    public bool Revoke(string keyId)
    {
        var entry = _keys.Values.FirstOrDefault(k => k.Id == keyId);
        if (entry == null) return false;
        entry.Enabled = false;
        return true;
    }

    /// <summary>List all API keys (without revealing the actual keys).</summary>
    public List<ApiKeyInfo> ListKeys() =>
        _keys.Values.OrderByDescending(k => k.CreatedAt).ToList();

    /// <summary>Load keys from a JSON file.</summary>
    public void LoadFromFile(string path)
    {
        if (!File.Exists(path)) return;
        var json = File.ReadAllText(path);
        var keys = JsonSerializer.Deserialize<List<ApiKeyInfo>>(json) ?? new();
        foreach (var key in keys)
            _keys[key.KeyHash] = key;
    }

    /// <summary>Save keys to a JSON file.</summary>
    public void SaveToFile(string path)
    {
        var json = JsonSerializer.Serialize(_keys.Values.ToList(), new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(path, json);
    }

    private static string HashKey(string apiKey)
    {
        var bytes = SHA256.HashData(Encoding.UTF8.GetBytes(apiKey));
        return Convert.ToHexString(bytes).ToLower();
    }

    private static int GetDefaultRateLimit(string tier) => tier.ToLower() switch
    {
        "free" => 10,
        "starter" => 60,
        "pro" => 300,
        "enterprise" => 1000,
        _ => 60,
    };
}

public class ApiKeyInfo
{
    [JsonPropertyName("id")] public string Id { get; set; } = "";
    [JsonPropertyName("name")] public string Name { get; set; } = "";
    [JsonPropertyName("tier")] public string Tier { get; set; } = "default";
    [JsonPropertyName("key_hash")] public string KeyHash { get; set; } = "";
    [JsonPropertyName("key_prefix")] public string KeyPrefix { get; set; } = "";
    [JsonPropertyName("rate_limit")] public int RateLimit { get; set; } = 60; // requests per minute
    [JsonPropertyName("expires_at")] public DateTime? ExpiresAt { get; set; }
    [JsonPropertyName("created_at")] public DateTime CreatedAt { get; set; }
    [JsonPropertyName("enabled")] public bool Enabled { get; set; } = true;
}

// === Rate Limiter (Token Bucket) ===

public class RateLimiter
{
    private readonly ConcurrentDictionary<string, TokenBucket> _buckets = new();
    private readonly RateLimitConfig _config;

    public RateLimiter(RateLimitConfig config)
    {
        _config = config;
    }

    /// <summary>Check if a request is within rate limits.</summary>
    public RateLimitResult CheckLimit(string keyId, int? customLimit = null)
    {
        int maxPerMinute = customLimit ?? _config.DefaultRequestsPerMinute;

        var bucket = _buckets.GetOrAdd(keyId, _ => new TokenBucket(maxPerMinute));

        if (bucket.TryConsume())
        {
            return new RateLimitResult
            {
                Allowed = true,
                Remaining = bucket.Remaining,
                Limit = maxPerMinute,
                ResetAt = bucket.ResetAt,
            };
        }

        int retryAfter = Math.Max(1, (int)(bucket.ResetAt - DateTime.UtcNow).TotalSeconds);
        return new RateLimitResult
        {
            Allowed = false,
            Remaining = 0,
            Limit = maxPerMinute,
            ResetAt = bucket.ResetAt,
            RetryAfterSeconds = retryAfter,
        };
    }

    /// <summary>Get rate limit headers for a response.</summary>
    public Dictionary<string, string> GetHeaders(string keyId)
    {
        if (!_buckets.TryGetValue(keyId, out var bucket))
            return new Dictionary<string, string>();

        return new Dictionary<string, string>
        {
            ["X-RateLimit-Limit"] = bucket.MaxTokens.ToString(),
            ["X-RateLimit-Remaining"] = bucket.Remaining.ToString(),
            ["X-RateLimit-Reset"] = bucket.ResetAt.ToString("O"),
        };
    }
}

public class TokenBucket
{
    private int _tokens;
    private DateTime _lastRefill;
    private readonly object _lock = new();

    public int MaxTokens { get; }
    public DateTime ResetAt => _lastRefill.AddMinutes(1);
    public int Remaining => _tokens;

    public TokenBucket(int maxPerMinute)
    {
        MaxTokens = maxPerMinute;
        _tokens = maxPerMinute;
        _lastRefill = DateTime.UtcNow;
    }

    public bool TryConsume()
    {
        lock (_lock)
        {
            Refill();
            if (_tokens <= 0) return false;
            _tokens--;
            return true;
        }
    }

    private void Refill()
    {
        var now = DateTime.UtcNow;
        var elapsed = now - _lastRefill;
        if (elapsed.TotalMinutes >= 1.0)
        {
            _tokens = MaxTokens;
            _lastRefill = now;
        }
        else
        {
            // Gradual refill: add tokens proportionally
            int tokensToAdd = (int)(elapsed.TotalMinutes * MaxTokens);
            if (tokensToAdd > 0)
            {
                _tokens = Math.Min(MaxTokens, _tokens + tokensToAdd);
                _lastRefill = now;
            }
        }
    }
}

public class RateLimitResult
{
    public bool Allowed { get; set; }
    public int Remaining { get; set; }
    public int Limit { get; set; }
    public DateTime ResetAt { get; set; }
    public int RetryAfterSeconds { get; set; }
}

// === Audit Logger ===

public class AuditLogger
{
    private readonly AuditConfig _config;
    private readonly ConcurrentQueue<AuditEntry> _recentEntries = new();
    private readonly object _writeLock = new();
    private StreamWriter? _writer;
    private int _entryCount;

    public AuditLogger(AuditConfig config)
    {
        _config = config;
        if (config.Enabled && !string.IsNullOrEmpty(config.LogFile))
        {
            var dir = Path.GetDirectoryName(config.LogFile);
            if (!string.IsNullOrEmpty(dir))
                Directory.CreateDirectory(dir);
            _writer = new StreamWriter(config.LogFile, append: true)
            {
                AutoFlush = true
            };
        }
    }

    /// <summary>Log a request record.</summary>
    public void Log(RequestRecord record)
    {
        var entry = new AuditEntry
        {
            Timestamp = DateTime.UtcNow,
            RequestId = record.RequestId,
            KeyId = record.KeyId,
            Endpoint = record.Endpoint,
            Method = record.Method,
            ClientIp = record.ClientIp,
            StatusCode = record.StatusCode,
            PromptTokens = record.PromptTokens,
            CompletionTokens = record.CompletionTokens,
            DurationMs = record.DurationMs,
            Model = record.Model,
            UserAgent = record.UserAgent,
            // Only log prompt/response if configured (privacy consideration)
            Prompt = _config.LogPrompts ? record.Prompt : null,
            Response = _config.LogResponses ? record.Response : null,
        };

        // Keep recent entries in memory for /admin/audit endpoint
        _recentEntries.Enqueue(entry);
        while (_recentEntries.Count > _config.MaxMemoryEntries)
            _recentEntries.TryDequeue(out _);

        // Write to file
        if (_writer != null)
        {
            lock (_writeLock)
            {
                _writer.WriteLine(JsonSerializer.Serialize(entry));
                _entryCount++;

                // Rotate log file if needed
                if (_config.MaxFileSize > 0 && _entryCount > 0 && _entryCount % 1000 == 0)
                    RotateIfNeeded();
            }
        }
    }

    /// <summary>Get recent audit entries from memory.</summary>
    public List<AuditEntry> GetRecent(int count = 100, string? keyFilter = null)
    {
        var query = _recentEntries.AsEnumerable();
        if (keyFilter != null)
            query = query.Where(e => e.KeyId == keyFilter);
        return query.TakeLast(count).Reverse().ToList();
    }

    /// <summary>Get audit statistics.</summary>
    public AuditStats GetStats()
    {
        var entries = _recentEntries.ToArray();
        return new AuditStats
        {
            TotalEntries = entries.Length,
            FileEntries = _entryCount,
            UniqueKeys = entries.Select(e => e.KeyId).Distinct().Count(),
            TotalPromptTokens = entries.Sum(e => e.PromptTokens),
            TotalCompletionTokens = entries.Sum(e => e.CompletionTokens),
            AvgDurationMs = entries.Length > 0 ? entries.Average(e => e.DurationMs) : 0,
            ErrorRate = entries.Length > 0
                ? (float)entries.Count(e => e.StatusCode >= 400) / entries.Length
                : 0,
        };
    }

    private void RotateIfNeeded()
    {
        if (_config.LogFile == null || _writer == null) return;
        var fi = new FileInfo(_config.LogFile);
        if (!fi.Exists || fi.Length < _config.MaxFileSize) return;

        _writer.Close();
        string rotated = $"{_config.LogFile}.{DateTime.UtcNow:yyyyMMdd_HHmmss}";
        File.Move(_config.LogFile, rotated);
        _writer = new StreamWriter(_config.LogFile, append: true) { AutoFlush = true };
        _entryCount = 0;
    }

    public void Dispose()
    {
        _writer?.Dispose();
    }
}

public class AuditEntry
{
    [JsonPropertyName("timestamp")] public DateTime Timestamp { get; set; }
    [JsonPropertyName("request_id")] public string RequestId { get; set; } = "";
    [JsonPropertyName("key_id")] public string? KeyId { get; set; }
    [JsonPropertyName("endpoint")] public string Endpoint { get; set; } = "";
    [JsonPropertyName("method")] public string Method { get; set; } = "";
    [JsonPropertyName("client_ip")] public string? ClientIp { get; set; }
    [JsonPropertyName("status_code")] public int StatusCode { get; set; }
    [JsonPropertyName("prompt_tokens")] public int PromptTokens { get; set; }
    [JsonPropertyName("completion_tokens")] public int CompletionTokens { get; set; }
    [JsonPropertyName("duration_ms")] public long DurationMs { get; set; }
    [JsonPropertyName("model")] public string? Model { get; set; }
    [JsonPropertyName("user_agent")] public string? UserAgent { get; set; }
    [JsonPropertyName("prompt")] public string? Prompt { get; set; }
    [JsonPropertyName("response")] public string? Response { get; set; }
}

public class AuditStats
{
    [JsonPropertyName("total_entries")] public int TotalEntries { get; set; }
    [JsonPropertyName("file_entries")] public int FileEntries { get; set; }
    [JsonPropertyName("unique_keys")] public int UniqueKeys { get; set; }
    [JsonPropertyName("total_prompt_tokens")] public long TotalPromptTokens { get; set; }
    [JsonPropertyName("total_completion_tokens")] public long TotalCompletionTokens { get; set; }
    [JsonPropertyName("avg_duration_ms")] public double AvgDurationMs { get; set; }
    [JsonPropertyName("error_rate")] public float ErrorRate { get; set; }
}

public class RequestRecord
{
    public string RequestId { get; set; } = Guid.NewGuid().ToString("N");
    public string? KeyId { get; set; }
    public string Endpoint { get; set; } = "";
    public string Method { get; set; } = "POST";
    public string? ClientIp { get; set; }
    public int StatusCode { get; set; } = 200;
    public int PromptTokens { get; set; }
    public int CompletionTokens { get; set; }
    public long DurationMs { get; set; }
    public string? Model { get; set; }
    public string? UserAgent { get; set; }
    public string? Prompt { get; set; }
    public string? Response { get; set; }
}

// === Usage Metering ===

public class UsageMeter
{
    private readonly ConcurrentDictionary<string, UsageRecord> _usage = new();

    /// <summary>Record token usage for a key.</summary>
    public void Record(string? keyId, int promptTokens, int completionTokens, long durationMs)
    {
        string key = keyId ?? "anonymous";
        var record = _usage.GetOrAdd(key, _ => new UsageRecord { KeyId = key });

        Interlocked.Increment(ref record._requestCount);
        Interlocked.Add(ref record._promptTokens, promptTokens);
        Interlocked.Add(ref record._completionTokens, completionTokens);
        Interlocked.Add(ref record._totalDurationMs, durationMs);

        // Track daily usage
        string today = DateTime.UtcNow.ToString("yyyy-MM-dd");
        record.DailyUsage.AddOrUpdate(today,
            _ => new DailyUsage { Date = today, Requests = 1, PromptTokens = promptTokens, CompletionTokens = completionTokens },
            (_, existing) =>
            {
                Interlocked.Increment(ref existing._requests);
                Interlocked.Add(ref existing._promptTokens, promptTokens);
                Interlocked.Add(ref existing._completionTokens, completionTokens);
                return existing;
            });
    }

    /// <summary>Get usage for a specific key.</summary>
    public UsageRecord? GetUsage(string keyId) =>
        _usage.TryGetValue(keyId, out var record) ? record : null;

    /// <summary>Get usage for all keys.</summary>
    public List<UsageSummary> GetAllUsage() =>
        _usage.Values.Select(r => new UsageSummary
        {
            KeyId = r.KeyId,
            TotalRequests = r.RequestCount,
            TotalPromptTokens = r.PromptTokens,
            TotalCompletionTokens = r.CompletionTokens,
            TotalTokens = r.PromptTokens + r.CompletionTokens,
            AvgDurationMs = r.RequestCount > 0 ? r.TotalDurationMs / (double)r.RequestCount : 0,
            // Rough cost estimate: $0.001 per 1K prompt tokens, $0.002 per 1K completion tokens
            EstimatedCostUsd = (r.PromptTokens * 0.001f / 1000) + (r.CompletionTokens * 0.002f / 1000),
        }).ToList();

    /// <summary>Get daily usage breakdown for a key.</summary>
    public List<DailyUsage> GetDailyUsage(string keyId, int days = 30)
    {
        if (!_usage.TryGetValue(keyId, out var record))
            return new List<DailyUsage>();

        var cutoff = DateTime.UtcNow.AddDays(-days).ToString("yyyy-MM-dd");
        return record.DailyUsage.Values
            .Where(d => string.Compare(d.Date, cutoff, StringComparison.Ordinal) >= 0)
            .OrderByDescending(d => d.Date)
            .ToList();
    }

    /// <summary>Reset usage counters for a key.</summary>
    public void Reset(string keyId)
    {
        _usage.TryRemove(keyId, out _);
    }
}

public class UsageRecord
{
    public string KeyId { get; set; } = "";
    internal int _requestCount;
    internal int _promptTokens;
    internal int _completionTokens;
    internal long _totalDurationMs;

    public int RequestCount => _requestCount;
    public int PromptTokens => _promptTokens;
    public int CompletionTokens => _completionTokens;
    public long TotalDurationMs => _totalDurationMs;

    public ConcurrentDictionary<string, DailyUsage> DailyUsage { get; } = new();
}

public class DailyUsage
{
    [JsonPropertyName("date")] public string Date { get; set; } = "";

    [JsonPropertyName("requests")]
    public int Requests { get => _requests; set => _requests = value; }
    internal int _requests;

    [JsonPropertyName("prompt_tokens")]
    public int PromptTokens { get => _promptTokens; set => _promptTokens = value; }
    internal int _promptTokens;

    [JsonPropertyName("completion_tokens")]
    public int CompletionTokens { get => _completionTokens; set => _completionTokens = value; }
    internal int _completionTokens;
}

public class UsageSummary
{
    [JsonPropertyName("key_id")] public string KeyId { get; set; } = "";
    [JsonPropertyName("total_requests")] public int TotalRequests { get; set; }
    [JsonPropertyName("total_prompt_tokens")] public int TotalPromptTokens { get; set; }
    [JsonPropertyName("total_completion_tokens")] public int TotalCompletionTokens { get; set; }
    [JsonPropertyName("total_tokens")] public int TotalTokens { get; set; }
    [JsonPropertyName("avg_duration_ms")] public double AvgDurationMs { get; set; }
    [JsonPropertyName("estimated_cost_usd")] public float EstimatedCostUsd { get; set; }
}

// === Configuration ===

public class EnterpriseConfig
{
    [JsonPropertyName("auth_enabled")] public bool AuthEnabled { get; set; } = false;
    [JsonPropertyName("rate_limit_enabled")] public bool RateLimitEnabled { get; set; } = false;
    [JsonPropertyName("audit_enabled")] public bool AuditEnabled { get; set; } = false;
    [JsonPropertyName("metering_enabled")] public bool MeteringEnabled { get; set; } = false;
    [JsonPropertyName("api_keys_file")] public string? ApiKeysFile { get; set; }
    [JsonPropertyName("ip_allowlist")] public HashSet<string> IpAllowlist { get; set; } = new();
    [JsonPropertyName("ip_blocklist")] public HashSet<string> IpBlocklist { get; set; } = new();
    [JsonPropertyName("max_request_size_bytes")] public long MaxRequestSizeBytes { get; set; } = 10 * 1024 * 1024; // 10MB
    [JsonPropertyName("rate_limit")] public RateLimitConfig RateLimit { get; set; } = new();
    [JsonPropertyName("audit")] public AuditConfig Audit { get; set; } = new();
}

public class RateLimitConfig
{
    [JsonPropertyName("default_requests_per_minute")] public int DefaultRequestsPerMinute { get; set; } = 60;
    [JsonPropertyName("burst_multiplier")] public float BurstMultiplier { get; set; } = 1.5f;
}

public class AuditConfig
{
    [JsonPropertyName("enabled")] public bool Enabled { get; set; } = false;
    [JsonPropertyName("log_file")] public string? LogFile { get; set; } = "./logs/audit.jsonl";
    [JsonPropertyName("log_prompts")] public bool LogPrompts { get; set; } = false;
    [JsonPropertyName("log_responses")] public bool LogResponses { get; set; } = false;
    [JsonPropertyName("max_memory_entries")] public int MaxMemoryEntries { get; set; } = 10000;
    [JsonPropertyName("max_file_size")] public long MaxFileSize { get; set; } = 100 * 1024 * 1024; // 100MB
}
