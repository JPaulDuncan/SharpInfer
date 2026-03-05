using System.Net.Http.Json;
using System.Text.Json;

namespace SharpInfer.Core.Tools;

/// <summary>
/// Example tool: Web search via a configurable search API.
/// Demonstrates the ITool interface for extending the engine with external capabilities.
///
/// To use, provide a search API endpoint (e.g., SearXNG, Brave Search, Tavily, etc.)
/// </summary>
public class WebSearchTool : ITool
{
    private readonly HttpClient _httpClient;
    private readonly string _apiEndpoint;
    private readonly string? _apiKey;
    private readonly int _maxResults;

    public string Name => "web_search";
    public string Description => "Search the web for current information. Use when you need up-to-date facts, news, or data.";
    public string ParameterSchema => """{"type":"object","properties":{"query":{"type":"string","description":"The search query"}},"required":["query"]}""";

    public WebSearchTool(string apiEndpoint, string? apiKey = null, int maxResults = 5)
    {
        _httpClient = new HttpClient();
        _apiEndpoint = apiEndpoint;
        _apiKey = apiKey;
        _maxResults = maxResults;
    }

    public async Task<string> ExecuteAsync(string arguments)
    {
        var args = JsonDocument.Parse(arguments).RootElement;
        string query = args.GetProperty("query").GetString()
            ?? throw new ArgumentException("Missing 'query' parameter");

        try
        {
            var requestUrl = BuildRequestUrl(query);

            if (_apiKey != null)
                _httpClient.DefaultRequestHeaders.Authorization =
                    new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", _apiKey);

            var response = await _httpClient.GetStringAsync(requestUrl);
            return FormatResults(response);
        }
        catch (Exception ex)
        {
            return $"Search failed: {ex.Message}";
        }
    }

    private string BuildRequestUrl(string query)
    {
        var encoded = Uri.EscapeDataString(query);
        return $"{_apiEndpoint}?q={encoded}&format=json&count={_maxResults}";
    }

    private string FormatResults(string rawResponse)
    {
        try
        {
            var doc = JsonDocument.Parse(rawResponse);
            var root = doc.RootElement;

            var results = new System.Text.StringBuilder();

            // Try common response formats (SearXNG, Brave, generic)
            JsonElement resultsArray;
            if (root.TryGetProperty("results", out resultsArray) ||
                root.TryGetProperty("web", out var web) && web.TryGetProperty("results", out resultsArray))
            {
                int count = 0;
                foreach (var item in resultsArray.EnumerateArray())
                {
                    if (count >= _maxResults) break;

                    string title = item.TryGetProperty("title", out var t) ? t.GetString() ?? "" : "";
                    string url = item.TryGetProperty("url", out var u) ? u.GetString() ?? "" : "";
                    string snippet = item.TryGetProperty("content", out var c) ? c.GetString() ?? ""
                        : item.TryGetProperty("description", out var d) ? d.GetString() ?? "" : "";

                    results.AppendLine($"[{count + 1}] {title}");
                    results.AppendLine($"    URL: {url}");
                    results.AppendLine($"    {snippet}");
                    results.AppendLine();
                    count++;
                }
            }

            return results.Length > 0 ? results.ToString() : "No results found.";
        }
        catch
        {
            return rawResponse.Length > 2000 ? rawResponse[..2000] + "..." : rawResponse;
        }
    }
}

/// <summary>
/// Example tool: Read the contents of a URL.
/// </summary>
public class UrlReaderTool : ITool
{
    private readonly HttpClient _httpClient = new();

    public string Name => "read_url";
    public string Description => "Fetch and read the text content of a web page.";
    public string ParameterSchema => """{"type":"object","properties":{"url":{"type":"string","description":"The URL to read"}},"required":["url"]}""";

    public async Task<string> ExecuteAsync(string arguments)
    {
        var args = JsonDocument.Parse(arguments).RootElement;
        string url = args.GetProperty("url").GetString()
            ?? throw new ArgumentException("Missing 'url' parameter");

        try
        {
            var content = await _httpClient.GetStringAsync(url);
            // Basic HTML stripping — replace with a real HTML-to-text converter for production
            content = System.Text.RegularExpressions.Regex.Replace(content, "<[^>]+>", " ");
            content = System.Text.RegularExpressions.Regex.Replace(content, @"\s+", " ");
            return content.Length > 4000 ? content[..4000] + "..." : content;
        }
        catch (Exception ex)
        {
            return $"Failed to fetch URL: {ex.Message}";
        }
    }
}
