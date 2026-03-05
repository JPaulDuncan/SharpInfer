using System.Net.Http.Headers;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace SharpInfer.Core.Engine;

/// <summary>
/// Cloud API fallback: seamlessly route requests to cloud LLM providers
/// when the local model is insufficient or unavailable.
///
/// Supported providers:
///   - OpenAI (GPT-4, GPT-3.5, etc.)
///   - Anthropic (Claude)
///   - Custom OpenAI-compatible endpoints (Together, Groq, Fireworks, etc.)
///
/// Fallback triggers:
///   1. Model not loaded / out of memory
///   2. Context length exceeded
///   3. User explicitly requests cloud model
///   4. Quality routing: complex queries → cloud, simple queries → local
///   5. Load balancing: local busy → overflow to cloud
///
/// All cloud requests use the OpenAI-compatible chat completions format,
/// with provider-specific adaptations where needed.
/// </summary>
public class CloudFallbackRouter
{
    private readonly List<CloudProvider> _providers = new();
    private readonly HttpClient _httpClient;
    private Func<string, bool>? _shouldFallback;
    private CloudFallbackConfig _config;

    public Action<string>? OnLog { get; set; }
    public IReadOnlyList<CloudProvider> Providers => _providers;

    public CloudFallbackRouter(CloudFallbackConfig config, HttpClient? httpClient = null)
    {
        _config = config;
        _httpClient = httpClient ?? new HttpClient { Timeout = TimeSpan.FromSeconds(config.TimeoutSeconds) };

        foreach (var providerConfig in config.Providers)
        {
            _providers.Add(new CloudProvider(providerConfig));
        }
    }

    /// <summary>
    /// Set a custom function to decide whether to fall back to cloud.
    /// Receives the prompt; returns true to use cloud.
    /// </summary>
    public void SetFallbackDecider(Func<string, bool> decider)
    {
        _shouldFallback = decider;
    }

    /// <summary>
    /// Determine whether this request should use cloud fallback.
    /// </summary>
    public bool ShouldUseFallback(string prompt, bool localAvailable, int? localContextLength = null)
    {
        // Always fall back if local model is unavailable
        if (!localAvailable) return true;

        // Context length exceeded
        if (localContextLength.HasValue && prompt.Length / 4 > localContextLength.Value)
        {
            OnLog?.Invoke($"Cloud fallback: prompt (~{prompt.Length / 4} tokens) exceeds local context ({localContextLength})");
            return true;
        }

        // Custom decider
        if (_shouldFallback != null)
            return _shouldFallback(prompt);

        // Keyword-based routing
        if (_config.FallbackKeywords.Count > 0)
        {
            var lower = prompt.ToLowerInvariant();
            foreach (var kw in _config.FallbackKeywords)
            {
                if (lower.Contains(kw.ToLowerInvariant()))
                {
                    OnLog?.Invoke($"Cloud fallback: keyword match '{kw}'");
                    return true;
                }
            }
        }

        return false;
    }

    /// <summary>
    /// Select the best available provider based on priority and health.
    /// </summary>
    public CloudProvider? SelectProvider(string? preferredProvider = null)
    {
        if (preferredProvider != null)
        {
            var preferred = _providers.FirstOrDefault(p =>
                p.Config.Name.Equals(preferredProvider, StringComparison.OrdinalIgnoreCase));
            if (preferred?.IsHealthy == true)
                return preferred;
        }

        return _providers
            .Where(p => p.IsHealthy && p.Config.Enabled)
            .OrderBy(p => p.Config.Priority)
            .ThenBy(p => p.RecentLatencyMs)
            .FirstOrDefault();
    }

    /// <summary>
    /// Send a chat completion request to a cloud provider (non-streaming).
    /// </summary>
    public async Task<CloudResponse> ChatAsync(
        List<ChatMessage> messages,
        CloudRequestOptions? options = null,
        string? preferredProvider = null)
    {
        var provider = SelectProvider(preferredProvider)
            ?? throw new InvalidOperationException("No healthy cloud providers available.");

        options ??= new CloudRequestOptions();
        var startTime = DateTime.UtcNow;

        try
        {
            OnLog?.Invoke($"Cloud request → {provider.Config.Name} ({provider.Config.Model})");

            var request = BuildRequest(provider, messages, options, stream: false);
            var httpResponse = await _httpClient.SendAsync(request);
            var responseBody = await httpResponse.Content.ReadAsStringAsync();

            if (!httpResponse.IsSuccessStatusCode)
            {
                provider.RecordFailure();
                throw new CloudProviderException(provider.Config.Name,
                    $"HTTP {(int)httpResponse.StatusCode}: {responseBody}");
            }

            var response = ParseResponse(provider, responseBody);
            var latency = (DateTime.UtcNow - startTime).TotalMilliseconds;
            provider.RecordSuccess(latency);

            OnLog?.Invoke($"Cloud response ← {provider.Config.Name} ({latency:F0}ms, {response.Usage?.TotalTokens ?? 0} tokens)");
            return response;
        }
        catch (Exception ex) when (ex is not CloudProviderException)
        {
            provider.RecordFailure();

            // Try next provider
            var fallback = _providers
                .Where(p => p != provider && p.IsHealthy && p.Config.Enabled)
                .OrderBy(p => p.Config.Priority)
                .FirstOrDefault();

            if (fallback != null)
            {
                OnLog?.Invoke($"Failing over to {fallback.Config.Name}...");
                var request = BuildRequest(fallback, messages, options, stream: false);
                var httpResponse = await _httpClient.SendAsync(request);
                var responseBody = await httpResponse.Content.ReadAsStringAsync();

                if (httpResponse.IsSuccessStatusCode)
                {
                    var response = ParseResponse(fallback, responseBody);
                    fallback.RecordSuccess((DateTime.UtcNow - startTime).TotalMilliseconds);
                    return response;
                }
            }

            throw;
        }
    }

    /// <summary>
    /// Stream a chat completion from a cloud provider, yielding tokens.
    /// </summary>
    public async IAsyncEnumerable<string> ChatStreamAsync(
        List<ChatMessage> messages,
        CloudRequestOptions? options = null,
        string? preferredProvider = null,
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        var provider = SelectProvider(preferredProvider)
            ?? throw new InvalidOperationException("No healthy cloud providers available.");

        options ??= new CloudRequestOptions();
        var startTime = DateTime.UtcNow;

        OnLog?.Invoke($"Cloud stream → {provider.Config.Name} ({provider.Config.Model})");

        var request = BuildRequest(provider, messages, options, stream: true);
        using var httpResponse = await _httpClient.SendAsync(request,
            HttpCompletionOption.ResponseHeadersRead, ct);

        if (!httpResponse.IsSuccessStatusCode)
        {
            provider.RecordFailure();
            var body = await httpResponse.Content.ReadAsStringAsync(ct);
            throw new CloudProviderException(provider.Config.Name,
                $"HTTP {(int)httpResponse.StatusCode}: {body}");
        }

        using var stream = await httpResponse.Content.ReadAsStreamAsync(ct);
        using var reader = new StreamReader(stream);
        int tokenCount = 0;

        while (!reader.EndOfStream && !ct.IsCancellationRequested)
        {
            var line = await reader.ReadLineAsync(ct);
            if (line == null) break;

            // SSE format: "data: {...}"
            if (!line.StartsWith("data: ")) continue;
            var data = line["data: ".Length..];
            if (data == "[DONE]") break;

            string? token = null;

            try
            {
                if (provider.Config.ProviderType == CloudProviderType.Anthropic)
                    token = ParseAnthropicStreamChunk(data);
                else
                    token = ParseOpenAIStreamChunk(data);
            }
            catch (JsonException) { continue; }

            if (token != null)
            {
                tokenCount++;
                yield return token;
            }
        }

        var latency = (DateTime.UtcNow - startTime).TotalMilliseconds;
        provider.RecordSuccess(latency);
        OnLog?.Invoke($"Cloud stream complete ← {provider.Config.Name} ({latency:F0}ms, {tokenCount} tokens)");
    }

    // ─── Request Building ──────────────────────────────────────

    private HttpRequestMessage BuildRequest(
        CloudProvider provider, List<ChatMessage> messages,
        CloudRequestOptions options, bool stream)
    {
        var config = provider.Config;
        HttpRequestMessage request;

        if (config.ProviderType == CloudProviderType.Anthropic)
        {
            request = BuildAnthropicRequest(config, messages, options, stream);
        }
        else
        {
            request = BuildOpenAIRequest(config, messages, options, stream);
        }

        return request;
    }

    private HttpRequestMessage BuildOpenAIRequest(
        CloudProviderConfig config, List<ChatMessage> messages,
        CloudRequestOptions options, bool stream)
    {
        var body = new
        {
            model = config.Model,
            messages = messages.Select(m => new { role = m.Role, content = m.Content }),
            max_tokens = options.MaxTokens ?? 1024,
            temperature = options.Temperature ?? 0.7f,
            top_p = options.TopP ?? 0.9f,
            stream,
        };

        var request = new HttpRequestMessage(HttpMethod.Post, $"{config.BaseUrl}/chat/completions");
        request.Content = new StringContent(
            JsonSerializer.Serialize(body, _jsonOptions),
            Encoding.UTF8, "application/json");
        request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", config.ApiKey);

        // Custom headers (e.g., for Anthropic via OpenAI-compatible proxy)
        foreach (var (key, value) in config.CustomHeaders)
            request.Headers.TryAddWithoutValidation(key, value);

        return request;
    }

    private HttpRequestMessage BuildAnthropicRequest(
        CloudProviderConfig config, List<ChatMessage> messages,
        CloudRequestOptions options, bool stream)
    {
        // Anthropic Messages API format
        var systemMsg = messages.FirstOrDefault(m => m.Role == "system");
        var chatMessages = messages.Where(m => m.Role != "system")
            .Select(m => new { role = m.Role, content = m.Content });

        var bodyDict = new Dictionary<string, object>
        {
            ["model"] = config.Model,
            ["messages"] = chatMessages,
            ["max_tokens"] = options.MaxTokens ?? 1024,
            ["stream"] = stream,
        };

        if (systemMsg != null)
            bodyDict["system"] = systemMsg.Content;
        if (options.Temperature.HasValue)
            bodyDict["temperature"] = options.Temperature.Value;
        if (options.TopP.HasValue)
            bodyDict["top_p"] = options.TopP.Value;

        var request = new HttpRequestMessage(HttpMethod.Post, $"{config.BaseUrl}/messages");
        request.Content = new StringContent(
            JsonSerializer.Serialize(bodyDict, _jsonOptions),
            Encoding.UTF8, "application/json");
        request.Headers.TryAddWithoutValidation("x-api-key", config.ApiKey);
        request.Headers.TryAddWithoutValidation("anthropic-version", "2023-06-01");

        return request;
    }

    // ─── Response Parsing ──────────────────────────────────────

    private CloudResponse ParseResponse(CloudProvider provider, string responseBody)
    {
        if (provider.Config.ProviderType == CloudProviderType.Anthropic)
            return ParseAnthropicResponse(responseBody);

        return ParseOpenAIResponse(responseBody);
    }

    private CloudResponse ParseOpenAIResponse(string json)
    {
        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;

        var choice = root.GetProperty("choices")[0];
        var message = choice.GetProperty("message");

        var response = new CloudResponse
        {
            Content = message.GetProperty("content").GetString() ?? "",
            FinishReason = choice.TryGetProperty("finish_reason", out var fr) ? fr.GetString() : null,
            Model = root.TryGetProperty("model", out var m) ? m.GetString() : null,
        };

        if (root.TryGetProperty("usage", out var usage))
        {
            response.Usage = new CloudUsage
            {
                PromptTokens = usage.TryGetProperty("prompt_tokens", out var pt) ? pt.GetInt32() : 0,
                CompletionTokens = usage.TryGetProperty("completion_tokens", out var ct) ? ct.GetInt32() : 0,
                TotalTokens = usage.TryGetProperty("total_tokens", out var tt) ? tt.GetInt32() : 0,
            };
        }

        return response;
    }

    private CloudResponse ParseAnthropicResponse(string json)
    {
        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;

        var content = root.GetProperty("content");
        var textParts = new StringBuilder();
        foreach (var block in content.EnumerateArray())
        {
            if (block.TryGetProperty("type", out var type) && type.GetString() == "text")
                textParts.Append(block.GetProperty("text").GetString());
        }

        var response = new CloudResponse
        {
            Content = textParts.ToString(),
            FinishReason = root.TryGetProperty("stop_reason", out var sr) ? sr.GetString() : null,
            Model = root.TryGetProperty("model", out var m) ? m.GetString() : null,
        };

        if (root.TryGetProperty("usage", out var usage))
        {
            response.Usage = new CloudUsage
            {
                PromptTokens = usage.TryGetProperty("input_tokens", out var it) ? it.GetInt32() : 0,
                CompletionTokens = usage.TryGetProperty("output_tokens", out var ot) ? ot.GetInt32() : 0,
                TotalTokens = (usage.TryGetProperty("input_tokens", out var it2) ? it2.GetInt32() : 0) +
                             (usage.TryGetProperty("output_tokens", out var ot2) ? ot2.GetInt32() : 0),
            };
        }

        return response;
    }

    private string? ParseOpenAIStreamChunk(string json)
    {
        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;
        var choices = root.GetProperty("choices");
        if (choices.GetArrayLength() == 0) return null;
        var delta = choices[0].GetProperty("delta");
        return delta.TryGetProperty("content", out var content) ? content.GetString() : null;
    }

    private string? ParseAnthropicStreamChunk(string json)
    {
        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;
        if (!root.TryGetProperty("type", out var type)) return null;

        return type.GetString() switch
        {
            "content_block_delta" =>
                root.TryGetProperty("delta", out var delta) &&
                delta.TryGetProperty("text", out var text)
                    ? text.GetString() : null,
            _ => null
        };
    }

    /// <summary>Get health status of all providers.</summary>
    public IEnumerable<(string Name, bool Healthy, double LatencyMs, int Failures)> GetProviderStatus()
    {
        foreach (var p in _providers)
        {
            yield return (p.Config.Name, p.IsHealthy, p.RecentLatencyMs, p.ConsecutiveFailures);
        }
    }

    private static readonly JsonSerializerOptions _jsonOptions = new()
    {
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
    };
}

/// <summary>
/// Tracks state and health for a single cloud provider.
/// </summary>
public class CloudProvider
{
    public CloudProviderConfig Config { get; }

    private int _consecutiveFailures;
    private double _recentLatencyMs;
    private DateTime _lastFailure = DateTime.MinValue;
    private readonly object _lock = new();

    public int ConsecutiveFailures => _consecutiveFailures;
    public double RecentLatencyMs => _recentLatencyMs;

    /// <summary>
    /// Provider is healthy if fewer than MaxRetries consecutive failures,
    /// or enough time has passed since last failure (circuit breaker reset).
    /// </summary>
    public bool IsHealthy
    {
        get
        {
            if (_consecutiveFailures < Config.MaxRetries) return true;
            // Circuit breaker: retry after cooldown
            return (DateTime.UtcNow - _lastFailure).TotalSeconds > Config.CooldownSeconds;
        }
    }

    public CloudProvider(CloudProviderConfig config)
    {
        Config = config;
    }

    public void RecordSuccess(double latencyMs)
    {
        lock (_lock)
        {
            _consecutiveFailures = 0;
            _recentLatencyMs = _recentLatencyMs * 0.7 + latencyMs * 0.3; // EMA
        }
    }

    public void RecordFailure()
    {
        lock (_lock)
        {
            _consecutiveFailures++;
            _lastFailure = DateTime.UtcNow;
        }
    }
}

// ─── Types ────────────────────────────────────────────────

public class ChatMessage
{
    [JsonPropertyName("role")] public string Role { get; set; } = "user";
    [JsonPropertyName("content")] public string Content { get; set; } = "";
}

public class CloudResponse
{
    public string Content { get; set; } = "";
    public string? FinishReason { get; set; }
    public string? Model { get; set; }
    public CloudUsage? Usage { get; set; }
}

public class CloudUsage
{
    public int PromptTokens { get; set; }
    public int CompletionTokens { get; set; }
    public int TotalTokens { get; set; }
}

public class CloudRequestOptions
{
    public int? MaxTokens { get; set; }
    public float? Temperature { get; set; }
    public float? TopP { get; set; }
    public List<string>? Stop { get; set; }
}

public class CloudProviderException : Exception
{
    public string ProviderName { get; }
    public CloudProviderException(string provider, string message)
        : base($"[{provider}] {message}")
    {
        ProviderName = provider;
    }
}

// ─── Config ────────────────────────────────────────────────

public enum CloudProviderType
{
    OpenAI,
    Anthropic,
    OpenAICompatible,
}

public class CloudProviderConfig
{
    [JsonPropertyName("name")] public string Name { get; set; } = "";
    [JsonPropertyName("provider_type")] public CloudProviderType ProviderType { get; set; } = CloudProviderType.OpenAICompatible;
    [JsonPropertyName("base_url")] public string BaseUrl { get; set; } = "https://api.openai.com/v1";
    [JsonPropertyName("api_key")] public string ApiKey { get; set; } = "";
    [JsonPropertyName("model")] public string Model { get; set; } = "gpt-4";
    [JsonPropertyName("enabled")] public bool Enabled { get; set; } = true;
    [JsonPropertyName("priority")] public int Priority { get; set; } = 10; // Lower = higher priority
    [JsonPropertyName("max_retries")] public int MaxRetries { get; set; } = 3;
    [JsonPropertyName("cooldown_seconds")] public int CooldownSeconds { get; set; } = 60;
    [JsonPropertyName("custom_headers")] public Dictionary<string, string> CustomHeaders { get; set; } = new();
}

public class CloudFallbackConfig
{
    [JsonPropertyName("enabled")] public bool Enabled { get; set; } = false;
    [JsonPropertyName("providers")] public List<CloudProviderConfig> Providers { get; set; } = new();
    [JsonPropertyName("timeout_seconds")] public int TimeoutSeconds { get; set; } = 60;
    [JsonPropertyName("fallback_keywords")] public List<string> FallbackKeywords { get; set; } = new();
    [JsonPropertyName("fallback_on_context_exceeded")] public bool FallbackOnContextExceeded { get; set; } = true;
    [JsonPropertyName("fallback_on_load")] public bool FallbackOnLoad { get; set; } = true;
}
