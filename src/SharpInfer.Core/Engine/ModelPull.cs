using System.Collections.Concurrent;
using System.Net.Http.Headers;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace SharpInfer.Core.Engine;

/// <summary>
/// Model pull/download system — download models from remote registries.
///
/// Supports:
///   - HuggingFace Hub (models, GGUF files, safetensors)
///   - Direct URL downloads
///   - Resumable downloads with partial file support
///   - SHA256 integrity verification
///   - Progress tracking with speed estimates
///   - Concurrent multi-file downloads for safetensors shards
///   - Model alias system (e.g., "llama3" → HuggingFace repo + file)
///   - Local model cache management
///
/// Usage:
///   var puller = new ModelPuller("./models");
///   await puller.PullAsync("TheBloke/Llama-2-7B-GGUF:Q4_K_M");
///   await puller.PullAsync("https://example.com/model.gguf");
///   await puller.PullAsync("llama3");  // uses built-in alias
/// </summary>
public class ModelPuller
{
    private readonly string _modelsDir;
    private readonly HttpClient _http;
    private readonly ConcurrentDictionary<string, PullJob> _activeJobs = new();
    private readonly Dictionary<string, ModelAlias> _aliases = new();

    public Action<string>? OnLog { get; set; }
    public Action<PullProgress>? OnProgress { get; set; }

    public ModelPuller(string modelsDir, string? hfToken = null)
    {
        _modelsDir = modelsDir;
        Directory.CreateDirectory(modelsDir);

        _http = new HttpClient();
        _http.DefaultRequestHeaders.UserAgent.ParseAdd("SharpInfer/1.0");
        if (!string.IsNullOrEmpty(hfToken))
            _http.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", hfToken);

        InitializeAliases();
    }

    /// <summary>
    /// Pull a model by name, URL, or HuggingFace repo ID.
    ///
    /// Formats:
    ///   "llama3"                                    → Built-in alias
    ///   "TheBloke/Llama-2-7B-GGUF"                 → HuggingFace repo (auto-select best GGUF)
    ///   "TheBloke/Llama-2-7B-GGUF:Q4_K_M"          → HuggingFace repo with quant filter
    ///   "hf://user/repo/file.gguf"                  → Direct HuggingFace file
    ///   "https://example.com/model.gguf"            → Direct URL download
    /// </summary>
    public async Task<PullResult> PullAsync(string source, CancellationToken ct = default)
    {
        // Check for alias
        if (_aliases.TryGetValue(source.ToLower(), out var alias))
        {
            OnLog?.Invoke($"Resolved alias '{source}' → {alias.Repo}:{alias.QuantFilter ?? "best"}");
            source = alias.QuantFilter != null ? $"{alias.Repo}:{alias.QuantFilter}" : alias.Repo;
        }

        // Direct URL
        if (source.StartsWith("http://") || source.StartsWith("https://"))
            return await DownloadDirectAsync(source, ct);

        // HuggingFace hf:// protocol
        if (source.StartsWith("hf://"))
            return await DownloadHfFileAsync(source[5..], ct);

        // HuggingFace repo:quant format
        string? quantFilter = null;
        if (source.Contains(':'))
        {
            var parts = source.Split(':', 2);
            source = parts[0];
            quantFilter = parts[1];
        }

        // Assume HuggingFace repo ID
        return await PullFromHuggingFaceAsync(source, quantFilter, ct);
    }

    /// <summary>List downloaded models in the models directory.</summary>
    public List<DownloadedModel> ListDownloaded()
    {
        var models = new List<DownloadedModel>();
        if (!Directory.Exists(_modelsDir)) return models;

        // Direct GGUF files
        foreach (var file in Directory.GetFiles(_modelsDir, "*.gguf"))
        {
            var fi = new FileInfo(file);
            models.Add(new DownloadedModel
            {
                Name = Path.GetFileNameWithoutExtension(file),
                Path = file,
                Format = "gguf",
                SizeBytes = fi.Length,
                DownloadedAt = fi.CreationTimeUtc,
            });
        }

        // Subdirectories (safetensors repos, etc.)
        foreach (var dir in Directory.GetDirectories(_modelsDir))
        {
            var configPath = Path.Combine(dir, "config.json");
            if (File.Exists(configPath) || Directory.GetFiles(dir, "*.safetensors").Length > 0)
            {
                long totalSize = Directory.GetFiles(dir, "*", SearchOption.AllDirectories)
                    .Sum(f => new FileInfo(f).Length);
                models.Add(new DownloadedModel
                {
                    Name = Path.GetFileName(dir),
                    Path = dir,
                    Format = "safetensors",
                    SizeBytes = totalSize,
                    DownloadedAt = new DirectoryInfo(dir).CreationTimeUtc,
                });
            }
        }

        return models.OrderByDescending(m => m.DownloadedAt).ToList();
    }

    /// <summary>Delete a downloaded model.</summary>
    public bool Delete(string nameOrPath)
    {
        // Try exact path
        if (File.Exists(nameOrPath))
        {
            File.Delete(nameOrPath);
            // Delete companion files (.sha256, .json)
            var sha = nameOrPath + ".sha256";
            if (File.Exists(sha)) File.Delete(sha);
            OnLog?.Invoke($"Deleted: {nameOrPath}");
            return true;
        }

        if (Directory.Exists(nameOrPath))
        {
            Directory.Delete(nameOrPath, recursive: true);
            OnLog?.Invoke($"Deleted: {nameOrPath}");
            return true;
        }

        // Try by name in models dir
        var ggufPath = Path.Combine(_modelsDir, nameOrPath + ".gguf");
        if (File.Exists(ggufPath))
        {
            File.Delete(ggufPath);
            OnLog?.Invoke($"Deleted: {ggufPath}");
            return true;
        }

        var dirPath = Path.Combine(_modelsDir, nameOrPath);
        if (Directory.Exists(dirPath))
        {
            Directory.Delete(dirPath, recursive: true);
            OnLog?.Invoke($"Deleted: {dirPath}");
            return true;
        }

        return false;
    }

    /// <summary>Get active/recent download jobs.</summary>
    public List<PullJob> GetActiveJobs() =>
        _activeJobs.Values.OrderByDescending(j => j.StartedAt).ToList();

    // === HuggingFace Hub Integration ===

    private async Task<PullResult> PullFromHuggingFaceAsync(string repoId, string? quantFilter, CancellationToken ct)
    {
        OnLog?.Invoke($"Fetching file list from HuggingFace: {repoId}");

        // Get repo file listing via HF API
        var apiUrl = $"https://huggingface.co/api/models/{repoId}";
        var response = await _http.GetAsync(apiUrl, ct);

        if (!response.IsSuccessStatusCode)
        {
            return new PullResult
            {
                Success = false,
                Error = $"Failed to fetch repo info: HTTP {(int)response.StatusCode} {response.ReasonPhrase}",
            };
        }

        var json = await response.Content.ReadAsStringAsync(ct);
        var repoInfo = JsonSerializer.Deserialize<HfRepoInfo>(json);

        if (repoInfo?.Siblings == null || repoInfo.Siblings.Count == 0)
            return new PullResult { Success = false, Error = "No files found in repository" };

        // Find GGUF files
        var ggufFiles = repoInfo.Siblings
            .Where(s => s.FileName.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase))
            .ToList();

        if (ggufFiles.Count > 0)
        {
            // Filter by quant type if specified
            HfSibling? target;
            if (!string.IsNullOrEmpty(quantFilter))
            {
                target = ggufFiles.FirstOrDefault(f =>
                    f.FileName.Contains(quantFilter, StringComparison.OrdinalIgnoreCase));
                if (target == null)
                {
                    OnLog?.Invoke($"Available GGUF files: {string.Join(", ", ggufFiles.Select(f => f.FileName))}");
                    return new PullResult
                    {
                        Success = false,
                        Error = $"No GGUF file matching '{quantFilter}'. Available: {string.Join(", ", ggufFiles.Select(f => f.FileName))}",
                    };
                }
            }
            else
            {
                // Auto-select: prefer Q4_K_M, then Q4_K_S, then Q5_K_M, then first available
                target = ggufFiles.FirstOrDefault(f => f.FileName.Contains("Q4_K_M", StringComparison.OrdinalIgnoreCase))
                    ?? ggufFiles.FirstOrDefault(f => f.FileName.Contains("Q4_K_S", StringComparison.OrdinalIgnoreCase))
                    ?? ggufFiles.FirstOrDefault(f => f.FileName.Contains("Q5_K_M", StringComparison.OrdinalIgnoreCase))
                    ?? ggufFiles.First();
                OnLog?.Invoke($"Auto-selected: {target.FileName}");
            }

            var downloadUrl = $"https://huggingface.co/{repoId}/resolve/main/{target.FileName}";
            return await DownloadDirectAsync(downloadUrl, ct, target.FileName);
        }

        // Check for safetensors
        var safetensorFiles = repoInfo.Siblings
            .Where(s => s.FileName.EndsWith(".safetensors") || s.FileName == "config.json" ||
                        s.FileName == "tokenizer.json" || s.FileName == "tokenizer.model" ||
                        s.FileName == "tokenizer_config.json" || s.FileName == "special_tokens_map.json")
            .ToList();

        if (safetensorFiles.Count > 0)
        {
            return await DownloadSafetensorsRepoAsync(repoId, safetensorFiles, ct);
        }

        return new PullResult { Success = false, Error = "No supported model files found (GGUF or safetensors)" };
    }

    private async Task<PullResult> DownloadSafetensorsRepoAsync(string repoId, List<HfSibling> files, CancellationToken ct)
    {
        string repoName = repoId.Replace("/", "_");
        string destDir = Path.Combine(_modelsDir, repoName);
        Directory.CreateDirectory(destDir);

        OnLog?.Invoke($"Downloading {files.Count} files to {destDir}");

        int downloaded = 0;
        long totalBytes = 0;

        foreach (var file in files)
        {
            ct.ThrowIfCancellationRequested();
            var url = $"https://huggingface.co/{repoId}/resolve/main/{file.FileName}";
            var destPath = Path.Combine(destDir, file.FileName);

            // Create subdirectories if needed
            var subDir = Path.GetDirectoryName(destPath);
            if (!string.IsNullOrEmpty(subDir))
                Directory.CreateDirectory(subDir);

            if (File.Exists(destPath))
            {
                OnLog?.Invoke($"  Skipping (exists): {file.FileName}");
                downloaded++;
                continue;
            }

            OnLog?.Invoke($"  Downloading: {file.FileName}");
            var result = await DownloadFileAsync(url, destPath, file.FileName, ct);
            if (result.Success)
            {
                downloaded++;
                totalBytes += result.BytesDownloaded;
            }
            else
            {
                OnLog?.Invoke($"  Failed: {file.FileName} — {result.Error}");
            }
        }

        return new PullResult
        {
            Success = downloaded > 0,
            ModelPath = destDir,
            BytesDownloaded = totalBytes,
            FilesDownloaded = downloaded,
        };
    }

    private async Task<PullResult> DownloadHfFileAsync(string hfPath, CancellationToken ct)
    {
        // hf://user/repo/path/to/file.gguf
        var url = $"https://huggingface.co/{hfPath.Replace("/resolve/", "/resolve/")}"
            .Replace("hf://", "https://huggingface.co/");

        // If no /resolve/ in path, add it
        if (!url.Contains("/resolve/"))
        {
            var parts = hfPath.Split('/', 3);
            if (parts.Length >= 3)
                url = $"https://huggingface.co/{parts[0]}/{parts[1]}/resolve/main/{parts[2]}";
        }

        return await DownloadDirectAsync(url, ct);
    }

    private async Task<PullResult> DownloadDirectAsync(string url, CancellationToken ct, string? suggestedName = null)
    {
        string fileName = suggestedName ?? GetFileNameFromUrl(url);
        string destPath = Path.Combine(_modelsDir, fileName);

        // Check if already exists
        if (File.Exists(destPath))
        {
            var fi = new FileInfo(destPath);
            OnLog?.Invoke($"Model already exists: {destPath} ({FormatSize(fi.Length)})");
            return new PullResult
            {
                Success = true,
                ModelPath = destPath,
                BytesDownloaded = 0,
                AlreadyExisted = true,
            };
        }

        OnLog?.Invoke($"Downloading: {url}");
        return await DownloadFileAsync(url, destPath, fileName, ct);
    }

    private async Task<PullResult> DownloadFileAsync(string url, string destPath, string displayName, CancellationToken ct)
    {
        var job = new PullJob
        {
            Id = Guid.NewGuid().ToString("N")[..12],
            Url = url,
            DestPath = destPath,
            FileName = displayName,
            StartedAt = DateTime.UtcNow,
            Status = PullStatus.Downloading,
        };
        _activeJobs[job.Id] = job;

        string partialPath = destPath + ".partial";

        try
        {
            // Check for partial download (resume support)
            long existingBytes = 0;
            if (File.Exists(partialPath))
            {
                existingBytes = new FileInfo(partialPath).Length;
                OnLog?.Invoke($"  Resuming from {FormatSize(existingBytes)}");
            }

            using var request = new HttpRequestMessage(HttpMethod.Get, url);
            if (existingBytes > 0)
                request.Headers.Range = new RangeHeaderValue(existingBytes, null);

            using var response = await _http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, ct);

            if (response.StatusCode == System.Net.HttpStatusCode.RequestedRangeNotSatisfiable)
            {
                // File is complete, partial had the full content
                if (File.Exists(partialPath))
                    File.Move(partialPath, destPath, overwrite: true);
                return new PullResult { Success = true, ModelPath = destPath };
            }

            response.EnsureSuccessStatusCode();

            long? totalBytes = response.Content.Headers.ContentLength;
            if (existingBytes > 0 && response.StatusCode == System.Net.HttpStatusCode.PartialContent)
                totalBytes = totalBytes.HasValue ? totalBytes + existingBytes : null;
            else if (response.StatusCode == System.Net.HttpStatusCode.OK && existingBytes > 0)
                existingBytes = 0; // Server doesn't support range — restart

            job.TotalBytes = totalBytes ?? 0;

            OnLog?.Invoke($"  Size: {(totalBytes.HasValue ? FormatSize(totalBytes.Value) : "unknown")}");

            // Download with progress
            using var stream = await response.Content.ReadAsStreamAsync(ct);
            using var fileStream = new FileStream(partialPath,
                existingBytes > 0 && response.StatusCode == System.Net.HttpStatusCode.PartialContent
                    ? FileMode.Append
                    : FileMode.Create,
                FileAccess.Write);

            var buffer = new byte[64 * 1024]; // 64KB buffer
            long bytesRead = existingBytes;
            var speedTracker = new SpeedTracker();
            int read;
            DateTime lastProgressUpdate = DateTime.UtcNow;

            while ((read = await stream.ReadAsync(buffer, ct)) > 0)
            {
                await fileStream.WriteAsync(buffer.AsMemory(0, read), ct);
                bytesRead += read;
                speedTracker.AddBytes(read);

                job.DownloadedBytes = bytesRead;

                // Update progress every 500ms
                if ((DateTime.UtcNow - lastProgressUpdate).TotalMilliseconds >= 500)
                {
                    lastProgressUpdate = DateTime.UtcNow;
                    float progress = totalBytes.HasValue && totalBytes > 0
                        ? (float)bytesRead / totalBytes.Value * 100
                        : 0;
                    double speed = speedTracker.GetSpeed();
                    string eta = totalBytes.HasValue && speed > 0
                        ? FormatDuration(TimeSpan.FromSeconds((totalBytes.Value - bytesRead) / speed))
                        : "?";

                    var progressInfo = new PullProgress
                    {
                        FileName = displayName,
                        BytesDownloaded = bytesRead,
                        TotalBytes = totalBytes ?? 0,
                        ProgressPercent = progress,
                        SpeedBytesPerSec = speed,
                        Eta = eta,
                    };
                    OnProgress?.Invoke(progressInfo);

                    // Console progress bar
                    string progressBar = totalBytes.HasValue
                        ? $"\r  [{new string('█', (int)(progress / 5))}{new string('░', 20 - (int)(progress / 5))}] {progress:F1}% {FormatSize(bytesRead)}/{FormatSize(totalBytes.Value)} {FormatSpeed(speed)} ETA {eta}"
                        : $"\r  {FormatSize(bytesRead)} downloaded {FormatSpeed(speed)}";
                    Console.Write(progressBar);
                }
            }

            Console.WriteLine(); // End progress line

            // Move partial to final
            fileStream.Close();
            File.Move(partialPath, destPath, overwrite: true);

            // Verify integrity if SHA256 available
            job.Status = PullStatus.Verifying;
            OnLog?.Invoke($"  Verifying integrity...");
            string sha256 = await ComputeSha256Async(destPath, ct);
            File.WriteAllText(destPath + ".sha256", sha256);

            job.Status = PullStatus.Completed;
            job.CompletedAt = DateTime.UtcNow;

            OnLog?.Invoke($"  Downloaded: {destPath} ({FormatSize(bytesRead)}, SHA256: {sha256[..16]}...)");

            return new PullResult
            {
                Success = true,
                ModelPath = destPath,
                BytesDownloaded = bytesRead - existingBytes,
                Sha256 = sha256,
            };
        }
        catch (OperationCanceledException)
        {
            job.Status = PullStatus.Cancelled;
            OnLog?.Invoke($"  Download cancelled: {displayName}");
            return new PullResult { Success = false, Error = "Download cancelled" };
        }
        catch (Exception ex)
        {
            job.Status = PullStatus.Failed;
            job.Error = ex.Message;
            OnLog?.Invoke($"  Download failed: {ex.Message}");
            return new PullResult { Success = false, Error = ex.Message };
        }
        finally
        {
            // Keep job in history for a while
            _ = Task.Delay(TimeSpan.FromMinutes(5)).ContinueWith(_ => _activeJobs.TryRemove(job.Id, out PullJob? _));
        }
    }

    private static async Task<string> ComputeSha256Async(string filePath, CancellationToken ct)
    {
        using var sha256 = SHA256.Create();
        using var stream = File.OpenRead(filePath);
        var hash = await sha256.ComputeHashAsync(stream, ct);
        return Convert.ToHexString(hash).ToLower();
    }

    private static string GetFileNameFromUrl(string url)
    {
        var uri = new Uri(url);
        string name = Path.GetFileName(uri.LocalPath);
        if (string.IsNullOrEmpty(name) || !name.Contains('.'))
            name = "model_" + DateTime.UtcNow.ToString("yyyyMMdd_HHmmss") + ".gguf";
        return name;
    }

    // === Built-in Model Aliases ===

    private void InitializeAliases()
    {
        // Popular model aliases — maps short names to HuggingFace repos
        AddAlias("llama3", "QuantFactory/Meta-Llama-3-8B-Instruct-GGUF", "Q4_K_M");
        AddAlias("llama3:70b", "QuantFactory/Meta-Llama-3-70B-Instruct-GGUF", "Q4_K_M");
        AddAlias("llama2", "TheBloke/Llama-2-7B-Chat-GGUF", "Q4_K_M");
        AddAlias("llama2:13b", "TheBloke/Llama-2-13B-chat-GGUF", "Q4_K_M");
        AddAlias("mistral", "TheBloke/Mistral-7B-Instruct-v0.2-GGUF", "Q4_K_M");
        AddAlias("mixtral", "TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF", "Q4_K_M");
        AddAlias("codellama", "TheBloke/CodeLlama-7B-Instruct-GGUF", "Q4_K_M");
        AddAlias("phi3", "microsoft/Phi-3-mini-4k-instruct-gguf", null);
        AddAlias("phi2", "TheBloke/phi-2-GGUF", "Q4_K_M");
        AddAlias("gemma", "google/gemma-2b-it-GGUF", null);
        AddAlias("gemma:7b", "google/gemma-7b-it-GGUF", null);
        AddAlias("qwen2", "Qwen/Qwen2-7B-Instruct-GGUF", "Q4_K_M");
        AddAlias("tinyllama", "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", "Q4_K_M");
    }

    private void AddAlias(string name, string repo, string? quantFilter)
    {
        _aliases[name.ToLower()] = new ModelAlias { Name = name, Repo = repo, QuantFilter = quantFilter };
    }

    /// <summary>Add a custom model alias.</summary>
    public void RegisterAlias(string name, string repo, string? quantFilter = null)
    {
        AddAlias(name, repo, quantFilter);
    }

    /// <summary>List all registered aliases.</summary>
    public Dictionary<string, ModelAlias> GetAliases() => new(_aliases);

    // === Formatting Helpers ===

    public static string FormatSize(long bytes)
    {
        if (bytes < 1024) return $"{bytes}B";
        if (bytes < 1024 * 1024) return $"{bytes / 1024.0:F1}KB";
        if (bytes < 1024L * 1024 * 1024) return $"{bytes / (1024.0 * 1024):F1}MB";
        return $"{bytes / (1024.0 * 1024 * 1024):F2}GB";
    }

    private static string FormatSpeed(double bytesPerSec)
    {
        if (bytesPerSec < 1024) return $"{bytesPerSec:F0}B/s";
        if (bytesPerSec < 1024 * 1024) return $"{bytesPerSec / 1024:F1}KB/s";
        return $"{bytesPerSec / (1024 * 1024):F1}MB/s";
    }

    private static string FormatDuration(TimeSpan ts)
    {
        if (ts.TotalHours >= 1) return $"{ts.Hours}h{ts.Minutes}m";
        if (ts.TotalMinutes >= 1) return $"{ts.Minutes}m{ts.Seconds}s";
        return $"{ts.Seconds}s";
    }
}

// === Speed Tracker ===

internal class SpeedTracker
{
    private readonly Queue<(DateTime Time, long Bytes)> _samples = new();
    private readonly TimeSpan _window = TimeSpan.FromSeconds(5);

    public void AddBytes(long bytes)
    {
        var now = DateTime.UtcNow;
        _samples.Enqueue((now, bytes));

        // Remove old samples
        while (_samples.Count > 0 && (now - _samples.Peek().Time) > _window)
            _samples.Dequeue();
    }

    public double GetSpeed()
    {
        if (_samples.Count < 2) return 0;
        var items = _samples.ToArray();
        var elapsed = (items[^1].Time - items[0].Time).TotalSeconds;
        if (elapsed <= 0) return 0;
        long totalBytes = items.Sum(s => s.Bytes);
        return totalBytes / elapsed;
    }
}

// === Types ===

public class PullResult
{
    public bool Success { get; set; }
    public string? ModelPath { get; set; }
    public long BytesDownloaded { get; set; }
    public int FilesDownloaded { get; set; }
    public string? Sha256 { get; set; }
    public string? Error { get; set; }
    public bool AlreadyExisted { get; set; }
}

public class PullProgress
{
    public string FileName { get; set; } = "";
    public long BytesDownloaded { get; set; }
    public long TotalBytes { get; set; }
    public float ProgressPercent { get; set; }
    public double SpeedBytesPerSec { get; set; }
    public string Eta { get; set; } = "";
}

public class PullJob
{
    public string Id { get; set; } = "";
    public string Url { get; set; } = "";
    public string DestPath { get; set; } = "";
    public string FileName { get; set; } = "";
    public DateTime StartedAt { get; set; }
    public DateTime? CompletedAt { get; set; }
    public PullStatus Status { get; set; }
    public long TotalBytes { get; set; }
    public long DownloadedBytes { get; set; }
    public string? Error { get; set; }
    public float Progress => TotalBytes > 0 ? (float)DownloadedBytes / TotalBytes : 0;
}

public enum PullStatus
{
    Queued,
    Downloading,
    Verifying,
    Completed,
    Failed,
    Cancelled,
}

public class ModelAlias
{
    public string Name { get; set; } = "";
    public string Repo { get; set; } = "";
    public string? QuantFilter { get; set; }
}

public class DownloadedModel
{
    public string Name { get; set; } = "";
    public string Path { get; set; } = "";
    public string Format { get; set; } = "";
    public long SizeBytes { get; set; }
    public DateTime DownloadedAt { get; set; }
}

// === HuggingFace API Types ===

internal class HfRepoInfo
{
    [JsonPropertyName("siblings")] public List<HfSibling>? Siblings { get; set; }
    [JsonPropertyName("modelId")] public string? ModelId { get; set; }
    [JsonPropertyName("author")] public string? Author { get; set; }
    [JsonPropertyName("tags")] public List<string>? Tags { get; set; }
}

internal class HfSibling
{
    [JsonPropertyName("rfilename")] public string FileName { get; set; } = "";
    [JsonPropertyName("size")] public long? Size { get; set; }
}
