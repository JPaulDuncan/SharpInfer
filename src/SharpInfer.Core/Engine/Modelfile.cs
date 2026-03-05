using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.IO.Compression;
using SharpInfer.Core.Tensors;

namespace SharpInfer.Core.Engine;

/// <summary>
/// Modelfile: a declarative packaging format for SharpInfer models.
///
/// Inspired by Docker's Dockerfile, this provides
/// a human-readable way to configure and distribute models with all their
/// settings, system prompts, adapters, and templates bundled together.
///
/// File format example:
///
///   FROM ./models/llama-3.2-8b.gguf
///   PARAMETER temperature 0.7
///   PARAMETER top_p 0.9
///   PARAMETER top_k 40
///   PARAMETER context_length 8192
///   PARAMETER gpu_layers 99
///   SYSTEM "You are a helpful coding assistant specializing in C#."
///   TEMPLATE "[INST] {{.System}}\n{{.Prompt}} [/INST]"
///   ADAPTER ./adapters/code-lora.bin
///   LICENSE MIT
///   QUANTIZE Q4_K
///
/// Can also be bundled as a .simodel package (zip-based) containing
/// the Modelfile + all referenced files.
/// </summary>
public class Modelfile
{
    /// <summary>Base model path (FROM directive).</summary>
    public string FromModel { get; set; } = "";

    /// <summary>Parameters (PARAMETER directives).</summary>
    public Dictionary<string, string> Parameters { get; set; } = new();

    /// <summary>System prompt (SYSTEM directive).</summary>
    public string? SystemPrompt { get; set; }

    /// <summary>Chat template (TEMPLATE directive).</summary>
    public string? Template { get; set; }

    /// <summary>LoRA adapter paths (ADAPTER directives).</summary>
    public List<string> Adapters { get; set; } = new();

    /// <summary>License text (LICENSE directive).</summary>
    public string? License { get; set; }

    /// <summary>Target quantization (QUANTIZE directive).</summary>
    public string? QuantizeTarget { get; set; }

    /// <summary>Model description/notes (MESSAGE directive).</summary>
    public List<ModelfileMessage> Messages { get; set; } = new();

    /// <summary>Embedded files (for .simodel packages).</summary>
    public Dictionary<string, string> EmbeddedFiles { get; set; } = new();

    /// <summary>Custom metadata.</summary>
    public Dictionary<string, string> Metadata { get; set; } = new();

    // ─── Parsing ────────────────────────────────────────────────

    /// <summary>
    /// Parse a Modelfile from text content.
    /// </summary>
    public static Modelfile Parse(string content)
    {
        var mf = new Modelfile();
        var lines = content.Split('\n');
        var currentMultiLine = new StringBuilder();
        string? currentDirective = null;
        bool inMultiLine = false;

        foreach (var rawLine in lines)
        {
            var line = rawLine.TrimEnd('\r');

            // Handle multi-line strings (triple-quoted)
            if (inMultiLine)
            {
                if (line.TrimEnd() == "\"\"\"")
                {
                    SetMultiLineValue(mf, currentDirective!, currentMultiLine.ToString());
                    inMultiLine = false;
                    currentMultiLine.Clear();
                }
                else
                {
                    if (currentMultiLine.Length > 0) currentMultiLine.AppendLine();
                    currentMultiLine.Append(line);
                }
                continue;
            }

            // Skip comments and blank lines
            var trimmed = line.Trim();
            if (trimmed.Length == 0 || trimmed.StartsWith('#'))
                continue;

            // Parse directive
            var spaceIdx = trimmed.IndexOf(' ');
            if (spaceIdx < 0) continue;

            var directive = trimmed[..spaceIdx].ToUpperInvariant();
            var value = trimmed[(spaceIdx + 1)..].Trim();

            // Check for triple-quote start
            if (value.StartsWith("\"\"\""))
            {
                currentDirective = directive;
                inMultiLine = true;
                var afterQuotes = value[3..];
                if (afterQuotes.Length > 0)
                    currentMultiLine.Append(afterQuotes);
                continue;
            }

            // Strip surrounding quotes
            if (value.StartsWith('"') && value.EndsWith('"') && value.Length >= 2)
                value = value[1..^1];

            switch (directive)
            {
                case "FROM":
                    mf.FromModel = value;
                    break;
                case "PARAMETER":
                    var paramParts = value.Split(' ', 2);
                    if (paramParts.Length == 2)
                        mf.Parameters[paramParts[0]] = paramParts[1];
                    break;
                case "SYSTEM":
                    mf.SystemPrompt = Unescape(value);
                    break;
                case "TEMPLATE":
                    mf.Template = Unescape(value);
                    break;
                case "ADAPTER":
                    mf.Adapters.Add(value);
                    break;
                case "LICENSE":
                    mf.License = value;
                    break;
                case "QUANTIZE":
                    mf.QuantizeTarget = value;
                    break;
                case "MESSAGE":
                    var msgParts = value.Split(' ', 2);
                    if (msgParts.Length == 2)
                        mf.Messages.Add(new ModelfileMessage { Role = msgParts[0], Content = Unescape(msgParts[1]) });
                    break;
                case "META":
                    var metaParts = value.Split(' ', 2);
                    if (metaParts.Length == 2)
                        mf.Metadata[metaParts[0]] = metaParts[1];
                    break;
                case "EMBED":
                    var embedParts = value.Split(' ', 2);
                    if (embedParts.Length == 2)
                        mf.EmbeddedFiles[embedParts[0]] = embedParts[1];
                    break;
            }
        }

        return mf;
    }

    /// <summary>Parse a Modelfile from a file path.</summary>
    public static Modelfile ParseFile(string path)
    {
        return Parse(File.ReadAllText(path));
    }

    private static void SetMultiLineValue(Modelfile mf, string directive, string value)
    {
        switch (directive)
        {
            case "SYSTEM": mf.SystemPrompt = value; break;
            case "TEMPLATE": mf.Template = value; break;
            case "LICENSE": mf.License = value; break;
        }
    }

    // ─── Serialization ──────────────────────────────────────────

    /// <summary>
    /// Generate the Modelfile text content.
    /// </summary>
    public string ToModelfileText()
    {
        var sb = new StringBuilder();
        sb.AppendLine($"FROM {FromModel}");
        sb.AppendLine();

        // Parameters
        foreach (var (key, val) in Parameters)
            sb.AppendLine($"PARAMETER {key} {val}");

        if (Parameters.Count > 0) sb.AppendLine();

        // System prompt
        if (!string.IsNullOrEmpty(SystemPrompt))
        {
            if (SystemPrompt.Contains('\n'))
                sb.AppendLine($"SYSTEM \"\"\"\n{SystemPrompt}\n\"\"\"");
            else
                sb.AppendLine($"SYSTEM \"{Escape(SystemPrompt)}\"");
            sb.AppendLine();
        }

        // Template
        if (!string.IsNullOrEmpty(Template))
        {
            if (Template.Contains('\n'))
                sb.AppendLine($"TEMPLATE \"\"\"\n{Template}\n\"\"\"");
            else
                sb.AppendLine($"TEMPLATE \"{Escape(Template)}\"");
            sb.AppendLine();
        }

        // Adapters
        foreach (var adapter in Adapters)
            sb.AppendLine($"ADAPTER {adapter}");

        // Quantize
        if (!string.IsNullOrEmpty(QuantizeTarget))
            sb.AppendLine($"QUANTIZE {QuantizeTarget}");

        // Messages
        foreach (var msg in Messages)
            sb.AppendLine($"MESSAGE {msg.Role} \"{Escape(msg.Content)}\"");

        // Metadata
        foreach (var (key, val) in Metadata)
            sb.AppendLine($"META {key} {val}");

        // License
        if (!string.IsNullOrEmpty(License))
        {
            sb.AppendLine();
            if (License.Contains('\n'))
                sb.AppendLine($"LICENSE \"\"\"\n{License}\n\"\"\"");
            else
                sb.AppendLine($"LICENSE {License}");
        }

        return sb.ToString();
    }

    /// <summary>Save Modelfile to disk.</summary>
    public void SaveToFile(string path)
    {
        File.WriteAllText(path, ToModelfileText());
    }

    // ─── Engine Config Conversion ──────────────────────────────

    /// <summary>
    /// Convert this Modelfile to an EngineConfig for use with InferenceEngine.
    /// Resolves relative paths against a base directory.
    /// </summary>
    public EngineConfig ToEngineConfig(string? baseDir = null)
    {
        baseDir ??= Directory.GetCurrentDirectory();

        var config = new EngineConfig
        {
            ModelPath = ResolvePath(FromModel, baseDir),
        };

        // Map parameters
        if (Parameters.TryGetValue("temperature", out var temp))
            config.Generation.Temperature = float.Parse(temp);
        if (Parameters.TryGetValue("top_p", out var topP))
            config.Generation.TopP = float.Parse(topP);
        if (Parameters.TryGetValue("top_k", out var topK))
            config.Generation.TopK = int.Parse(topK);
        if (Parameters.TryGetValue("max_tokens", out var maxTok))
            config.Generation.MaxTokens = int.Parse(maxTok);
        if (Parameters.TryGetValue("context_length", out var ctx))
            config.ContextLength = int.Parse(ctx);
        if (Parameters.TryGetValue("repetition_penalty", out var rep))
            config.Generation.RepetitionPenalty = float.Parse(rep);
        if (Parameters.TryGetValue("seed", out var seed))
            config.Generation.Seed = int.Parse(seed);

        // GPU settings
        if (Parameters.TryGetValue("gpu_layers", out var gpuLayers))
        {
            config.Gpu.Enabled = true;
            config.Gpu.GpuLayers = int.Parse(gpuLayers);
        }
        if (Parameters.TryGetValue("gpu", out var gpu) && bool.TryParse(gpu, out var gpuEnabled))
            config.Gpu.Enabled = gpuEnabled;

        // System prompt
        if (!string.IsNullOrEmpty(SystemPrompt))
            config.Generation.SystemPrompt = SystemPrompt;

        // LoRA adapters
        if (Adapters.Count > 0)
        {
            config.Lora.Enabled = true;
            foreach (var adapter in Adapters)
            {
                var name = Path.GetFileNameWithoutExtension(adapter);
                config.Lora.Adapters.Add(new LoraAdapterEntry
                {
                    Name = name,
                    Path = ResolvePath(adapter, baseDir),
                });
            }
            config.Lora.ActiveAdapter = config.Lora.Adapters[0].Name;
        }

        // FlashAttention
        if (Parameters.TryGetValue("flash_attention", out var fa) && bool.TryParse(fa, out var faEnabled))
            config.FlashAttention.Enabled = faEnabled;

        // Batch scheduler
        if (Parameters.TryGetValue("batch_scheduler", out var bs) && bool.TryParse(bs, out var bsEnabled))
        {
            config.BatchScheduler.Enabled = bsEnabled;
        }
        if (Parameters.TryGetValue("max_batch_size", out var mbs))
            config.BatchScheduler.MaxBatchSize = int.Parse(mbs);

        return config;
    }

    /// <summary>
    /// Create a Modelfile from an existing EngineConfig.
    /// </summary>
    public static Modelfile FromEngineConfig(EngineConfig config)
    {
        var mf = new Modelfile
        {
            FromModel = config.ModelPath,
            SystemPrompt = config.Generation.SystemPrompt,
        };

        mf.Parameters["temperature"] = config.Generation.Temperature.ToString("F1");
        mf.Parameters["top_p"] = config.Generation.TopP.ToString("F2");
        mf.Parameters["top_k"] = config.Generation.TopK.ToString();
        mf.Parameters["max_tokens"] = config.Generation.MaxTokens.ToString();

        if (config.ContextLength.HasValue)
            mf.Parameters["context_length"] = config.ContextLength.Value.ToString();
        if (config.Generation.RepetitionPenalty != 1.1f)
            mf.Parameters["repetition_penalty"] = config.Generation.RepetitionPenalty.ToString("F2");
        if (config.Gpu.Enabled)
        {
            mf.Parameters["gpu"] = "true";
            if (config.Gpu.GpuLayers.HasValue)
                mf.Parameters["gpu_layers"] = config.Gpu.GpuLayers.Value.ToString();
        }

        foreach (var adapter in config.Lora.Adapters)
            mf.Adapters.Add(adapter.Path);

        if (!config.FlashAttention.Enabled)
            mf.Parameters["flash_attention"] = "false";

        return mf;
    }

    // ─── Package (.simodel) ────────────────────────────────────

    /// <summary>
    /// Bundle this Modelfile and all its referenced files into a .simodel package.
    /// A .simodel is a ZIP archive containing:
    ///   - Modelfile (the declarative config)
    ///   - manifest.json (package metadata)
    ///   - model weights file(s)
    ///   - adapter file(s)
    ///   - any embedded files
    /// </summary>
    public void PackageToFile(string outputPath, string? baseDir = null, Action<string>? onLog = null)
    {
        baseDir ??= Directory.GetCurrentDirectory();

        if (File.Exists(outputPath))
            File.Delete(outputPath);

        using var zip = ZipFile.Open(outputPath, ZipArchiveMode.Create);

        // 1. Write Modelfile (with relative paths)
        var modelfileEntry = zip.CreateEntry("Modelfile");
        using (var writer = new StreamWriter(modelfileEntry.Open()))
        {
            writer.Write(ToModelfileText());
        }
        onLog?.Invoke("Added: Modelfile");

        // 2. Write manifest
        var manifest = new PackageManifest
        {
            Version = "1.0",
            Created = DateTimeOffset.UtcNow,
            ModelName = Path.GetFileNameWithoutExtension(FromModel),
            QuantType = QuantizeTarget,
            Parameters = Parameters,
            HasSystemPrompt = !string.IsNullOrEmpty(SystemPrompt),
            HasTemplate = !string.IsNullOrEmpty(Template),
            AdapterCount = Adapters.Count,
        };

        var manifestEntry = zip.CreateEntry("manifest.json");
        using (var stream = manifestEntry.Open())
        {
            JsonSerializer.Serialize(stream, manifest, new JsonSerializerOptions { WriteIndented = true });
        }
        onLog?.Invoke("Added: manifest.json");

        // 3. Add model weights
        var modelPath = ResolvePath(FromModel, baseDir);
        if (File.Exists(modelPath))
        {
            var modelFileName = Path.GetFileName(modelPath);
            onLog?.Invoke($"Adding model: {modelFileName} ({new FileInfo(modelPath).Length / (1024 * 1024):F0} MB)...");
            zip.CreateEntryFromFile(modelPath, $"models/{modelFileName}");
            onLog?.Invoke($"Added: models/{modelFileName}");
        }

        // 4. Add adapters
        foreach (var adapter in Adapters)
        {
            var adapterPath = ResolvePath(adapter, baseDir);
            if (File.Exists(adapterPath))
            {
                var fileName = Path.GetFileName(adapterPath);
                zip.CreateEntryFromFile(adapterPath, $"adapters/{fileName}");
                onLog?.Invoke($"Added: adapters/{fileName}");
            }
        }

        // 5. Add embedded files
        foreach (var (name, path) in EmbeddedFiles)
        {
            var filePath = ResolvePath(path, baseDir);
            if (File.Exists(filePath))
            {
                zip.CreateEntryFromFile(filePath, $"files/{name}");
                onLog?.Invoke($"Added: files/{name}");
            }
        }

        onLog?.Invoke($"Package created: {outputPath}");
    }

    /// <summary>
    /// Extract a .simodel package and return the parsed Modelfile.
    /// </summary>
    public static Modelfile UnpackFromFile(string packagePath, string extractDir, Action<string>? onLog = null)
    {
        if (!File.Exists(packagePath))
            throw new FileNotFoundException($"Package not found: {packagePath}");

        Directory.CreateDirectory(extractDir);
        ZipFile.ExtractToDirectory(packagePath, extractDir, overwriteFiles: true);
        onLog?.Invoke($"Extracted to: {extractDir}");

        // Read Modelfile
        var modelfilePath = Path.Combine(extractDir, "Modelfile");
        if (!File.Exists(modelfilePath))
            throw new FileNotFoundException("Package does not contain a Modelfile.");

        var mf = ParseFile(modelfilePath);

        // Rewrite paths to point to extracted locations
        if (!string.IsNullOrEmpty(mf.FromModel))
        {
            var modelFileName = Path.GetFileName(mf.FromModel);
            var extracted = Path.Combine(extractDir, "models", modelFileName);
            if (File.Exists(extracted))
                mf.FromModel = extracted;
        }

        for (int i = 0; i < mf.Adapters.Count; i++)
        {
            var adapterFileName = Path.GetFileName(mf.Adapters[i]);
            var extracted = Path.Combine(extractDir, "adapters", adapterFileName);
            if (File.Exists(extracted))
                mf.Adapters[i] = extracted;
        }

        onLog?.Invoke("Package loaded successfully.");
        return mf;
    }

    /// <summary>
    /// Read the manifest from a .simodel package without extracting.
    /// </summary>
    public static PackageManifest? ReadManifest(string packagePath)
    {
        using var zip = ZipFile.OpenRead(packagePath);
        var entry = zip.GetEntry("manifest.json");
        if (entry == null) return null;

        using var stream = entry.Open();
        return JsonSerializer.Deserialize<PackageManifest>(stream);
    }

    /// <summary>
    /// List contents of a .simodel package.
    /// </summary>
    public static IEnumerable<(string Name, long Size)> ListPackageContents(string packagePath)
    {
        using var zip = ZipFile.OpenRead(packagePath);
        foreach (var entry in zip.Entries)
        {
            yield return (entry.FullName, entry.Length);
        }
    }

    // ─── Template Rendering ────────────────────────────────────

    /// <summary>
    /// Apply the chat template to a prompt, inserting system prompt and user message.
    ///
    /// Template variables:
    ///   {{.System}} - System prompt
    ///   {{.Prompt}} - User prompt
    ///   {{.First}}  - true for the first message in conversation
    /// </summary>
    public string ApplyTemplate(string userPrompt, bool isFirst = true)
    {
        if (string.IsNullOrEmpty(Template))
            return BuildDefaultPrompt(userPrompt);

        var result = Template
            .Replace("{{.System}}", SystemPrompt ?? "You are a helpful assistant.")
            .Replace("{{.Prompt}}", userPrompt)
            .Replace("{{.First}}", isFirst ? "true" : "false");

        // Handle conditional blocks: {{if .First}}...{{end}}
        if (!isFirst)
        {
            result = RemoveConditionalBlocks(result, "First");
        }

        return result;
    }

    /// <summary>
    /// Apply template for a multi-turn conversation.
    /// </summary>
    public string ApplyTemplateMultiTurn(IEnumerable<(string role, string content)> messages)
    {
        var sb = new StringBuilder();
        bool first = true;

        foreach (var (role, content) in messages)
        {
            if (role == "system" && first)
            {
                // System prompt handled by template
                first = false;
                continue;
            }

            if (role == "user")
            {
                sb.Append(ApplyTemplate(content, first));
                first = false;
            }
            else if (role == "assistant")
            {
                sb.Append(content);
            }
        }

        return sb.ToString();
    }

    private string BuildDefaultPrompt(string userPrompt)
    {
        var sb = new StringBuilder();
        if (!string.IsNullOrEmpty(SystemPrompt))
            sb.AppendLine(SystemPrompt);
        sb.AppendLine();
        sb.Append(userPrompt);
        return sb.ToString();
    }

    private static string RemoveConditionalBlocks(string template, string variable)
    {
        var startTag = $"{{{{if .{variable}}}}}";
        var endTag = "{{end}}";
        var result = new StringBuilder(template);

        while (true)
        {
            var str = result.ToString();
            int start = str.IndexOf(startTag);
            if (start < 0) break;
            int end = str.IndexOf(endTag, start);
            if (end < 0) break;
            result.Remove(start, end + endTag.Length - start);
        }

        return result.ToString();
    }

    // ─── Helpers ────────────────────────────────────────────────

    private static string ResolvePath(string path, string baseDir)
    {
        if (Path.IsPathRooted(path)) return path;
        return Path.GetFullPath(Path.Combine(baseDir, path));
    }

    private static string Escape(string s) =>
        s.Replace("\\", "\\\\").Replace("\"", "\\\"").Replace("\n", "\\n");

    private static string Unescape(string s) =>
        s.Replace("\\n", "\n").Replace("\\\"", "\"").Replace("\\\\", "\\");
}

/// <summary>A message in the Modelfile (few-shot example).</summary>
public class ModelfileMessage
{
    public string Role { get; set; } = "user";
    public string Content { get; set; } = "";
}

/// <summary>Manifest for .simodel packages.</summary>
public class PackageManifest
{
    [JsonPropertyName("version")] public string Version { get; set; } = "1.0";
    [JsonPropertyName("created")] public DateTimeOffset Created { get; set; }
    [JsonPropertyName("model_name")] public string ModelName { get; set; } = "";
    [JsonPropertyName("quant_type")] public string? QuantType { get; set; }
    [JsonPropertyName("parameters")] public Dictionary<string, string> Parameters { get; set; } = new();
    [JsonPropertyName("has_system_prompt")] public bool HasSystemPrompt { get; set; }
    [JsonPropertyName("has_template")] public bool HasTemplate { get; set; }
    [JsonPropertyName("adapter_count")] public int AdapterCount { get; set; }

    public override string ToString()
    {
        var sb = new StringBuilder();
        sb.AppendLine($"Model: {ModelName}");
        if (QuantType != null) sb.AppendLine($"Quantization: {QuantType}");
        sb.AppendLine($"Created: {Created:u}");
        sb.AppendLine($"System prompt: {(HasSystemPrompt ? "Yes" : "No")}");
        sb.AppendLine($"Chat template: {(HasTemplate ? "Yes" : "No")}");
        sb.AppendLine($"Adapters: {AdapterCount}");
        return sb.ToString();
    }
}

/// <summary>
/// Model registry: discover and manage models from a local directory.
/// </summary>
public class ModelRegistry
{
    private readonly string _modelsDir;
    private readonly Dictionary<string, ModelRegistryEntry> _models = new();

    public IReadOnlyDictionary<string, ModelRegistryEntry> Models => _models;

    public ModelRegistry(string modelsDir)
    {
        _modelsDir = modelsDir;
        Directory.CreateDirectory(modelsDir);
    }

    /// <summary>Scan the models directory for available models.</summary>
    public void Scan()
    {
        _models.Clear();

        // GGUF files
        foreach (var file in Directory.GetFiles(_modelsDir, "*.gguf", SearchOption.AllDirectories))
        {
            var name = Path.GetFileNameWithoutExtension(file);
            _models[name] = new ModelRegistryEntry
            {
                Name = name,
                Path = file,
                Format = "gguf",
                SizeBytes = new FileInfo(file).Length,
            };
        }

        // .simodel packages
        foreach (var file in Directory.GetFiles(_modelsDir, "*.simodel", SearchOption.AllDirectories))
        {
            var name = Path.GetFileNameWithoutExtension(file);
            var manifest = Modelfile.ReadManifest(file);
            _models[name] = new ModelRegistryEntry
            {
                Name = name,
                Path = file,
                Format = "simodel",
                SizeBytes = new FileInfo(file).Length,
                Manifest = manifest,
            };
        }

        // Safetensors directories
        foreach (var dir in Directory.GetDirectories(_modelsDir))
        {
            if (Directory.GetFiles(dir, "*.safetensors").Length > 0)
            {
                var name = Path.GetFileName(dir);
                long totalSize = Directory.GetFiles(dir).Sum(f => new FileInfo(f).Length);
                _models[name] = new ModelRegistryEntry
                {
                    Name = name,
                    Path = dir,
                    Format = "safetensors",
                    SizeBytes = totalSize,
                };
            }
        }

        // Modelfiles (look for files named "Modelfile" or *.modelfile)
        foreach (var file in Directory.GetFiles(_modelsDir, "Modelfile", SearchOption.AllDirectories))
        {
            var dir = Path.GetDirectoryName(file)!;
            var name = Path.GetFileName(dir);
            if (!_models.ContainsKey(name))
            {
                _models[name] = new ModelRegistryEntry
                {
                    Name = name,
                    Path = file,
                    Format = "modelfile",
                    SizeBytes = 0,
                };
            }
        }
    }

    /// <summary>Get a model by name.</summary>
    public ModelRegistryEntry? Get(string name) =>
        _models.TryGetValue(name, out var entry) ? entry : null;

    /// <summary>List all models with sizes.</summary>
    public string ListFormatted()
    {
        if (_models.Count == 0) return "No models found.";

        var sb = new StringBuilder();
        sb.AppendLine($"{"NAME",-30} {"FORMAT",-12} {"SIZE",-12}");
        sb.AppendLine(new string('─', 54));

        foreach (var (name, entry) in _models.OrderBy(kv => kv.Key))
        {
            string size = entry.SizeBytes switch
            {
                >= 1L << 30 => $"{entry.SizeBytes / (double)(1L << 30):F1} GB",
                >= 1L << 20 => $"{entry.SizeBytes / (double)(1L << 20):F1} MB",
                _ => $"{entry.SizeBytes / 1024.0:F1} KB"
            };
            sb.AppendLine($"{name,-30} {entry.Format,-12} {size,-12}");
        }

        return sb.ToString();
    }
}

/// <summary>An entry in the model registry.</summary>
public class ModelRegistryEntry
{
    public string Name { get; init; } = "";
    public string Path { get; init; } = "";
    public string Format { get; init; } = "";
    public long SizeBytes { get; init; }
    public PackageManifest? Manifest { get; init; }
}
