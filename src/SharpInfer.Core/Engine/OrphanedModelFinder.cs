namespace SharpInfer.Core.Engine;

/// <summary>
/// Finds model files in the models directory that are not referenced by any
/// Modelfile's FROM or ADAPTER directives. These "orphaned" models waste disk
/// space and can be safely cleaned up.
/// </summary>
public class OrphanedModelFinder
{
    private readonly string _modelsDir;

    public OrphanedModelFinder(string modelsDir)
    {
        _modelsDir = Path.GetFullPath(modelsDir);
    }

    /// <summary>Find all orphaned models not referenced by any Modelfile.</summary>
    public List<OrphanedModel> FindOrphaned()
    {
        if (!Directory.Exists(_modelsDir))
            return new List<OrphanedModel>();

        var allModels = GetAllModelFiles();
        var referenced = GetReferencedModels();

        return allModels
            .Where(m => !referenced.Contains(NormalizePath(m.Path)))
            .ToList();
    }

    /// <summary>
    /// Delete orphaned models. If dryRun is true, only simulates deletion.
    /// Optionally filter to specific model names.
    /// </summary>
    public CleanupResult Cleanup(bool dryRun = true, List<string>? specificModels = null)
    {
        var orphaned = FindOrphaned();

        // Filter to specific models if requested
        if (specificModels != null && specificModels.Count > 0)
        {
            orphaned = orphaned
                .Where(m => specificModels.Any(s =>
                    m.Name.Contains(s, StringComparison.OrdinalIgnoreCase) ||
                    Path.GetFileNameWithoutExtension(m.Name)
                        .Contains(s, StringComparison.OrdinalIgnoreCase)))
                .ToList();
        }

        var result = new CleanupResult
        {
            DryRun = dryRun,
            DeletedCount = orphaned.Count,
            BytesFreed = orphaned.Sum(m => m.SizeBytes),
        };

        foreach (var model in orphaned)
        {
            if (dryRun)
            {
                result.DeletedPaths.Add(model.Path);
                continue;
            }

            try
            {
                if (model.Format == "directory")
                {
                    Directory.Delete(model.Path, recursive: true);
                }
                else
                {
                    File.Delete(model.Path);

                    // Also delete companion files (.sha256, .json, etc.)
                    foreach (var ext in new[] { ".sha256", ".json", ".metadata" })
                    {
                        string companion = model.Path + ext;
                        if (File.Exists(companion))
                            File.Delete(companion);
                    }
                }

                result.DeletedPaths.Add(model.Path);
            }
            catch (Exception ex)
            {
                result.Errors.Add($"{model.Name}: {ex.Message}");
                result.DeletedCount--;
                result.BytesFreed -= model.SizeBytes;
            }
        }

        return result;
    }

    /// <summary>Collect all model file paths referenced by Modelfiles in the directory.</summary>
    private HashSet<string> GetReferencedModels()
    {
        var referenced = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        // Find all Modelfile files (exact name "Modelfile" or *.modelfile)
        var modelfiles = new List<string>();

        foreach (var file in Directory.GetFiles(_modelsDir, "Modelfile", SearchOption.AllDirectories))
            modelfiles.Add(file);

        foreach (var file in Directory.GetFiles(_modelsDir, "*.modelfile", SearchOption.AllDirectories))
            modelfiles.Add(file);

        // Parse each Modelfile and collect referenced paths
        foreach (var mfPath in modelfiles.Distinct())
        {
            try
            {
                var modelfile = Modelfile.ParseFile(mfPath);
                string mfDir = Path.GetDirectoryName(mfPath) ?? _modelsDir;

                // FROM directive — the base model
                if (!string.IsNullOrWhiteSpace(modelfile.FromModel))
                {
                    string resolved = Path.IsPathRooted(modelfile.FromModel)
                        ? modelfile.FromModel
                        : Path.GetFullPath(Path.Combine(mfDir, modelfile.FromModel));
                    referenced.Add(NormalizePath(resolved));
                }

                // ADAPTER directives — LoRA adapters
                foreach (var adapter in modelfile.Adapters)
                {
                    if (!string.IsNullOrWhiteSpace(adapter))
                    {
                        string resolved = Path.IsPathRooted(adapter)
                            ? adapter
                            : Path.GetFullPath(Path.Combine(mfDir, adapter));
                        referenced.Add(NormalizePath(resolved));
                    }
                }
            }
            catch
            {
                // Skip unparseable Modelfiles
            }
        }

        return referenced;
    }

    /// <summary>Scan the models directory for all model files.</summary>
    private List<OrphanedModel> GetAllModelFiles()
    {
        var models = new List<OrphanedModel>();

        // .gguf files
        foreach (var file in Directory.GetFiles(_modelsDir, "*.gguf", SearchOption.AllDirectories))
        {
            var info = new FileInfo(file);
            models.Add(new OrphanedModel
            {
                Name = info.Name,
                Path = info.FullName,
                Format = "gguf",
                SizeBytes = info.Length,
                CreatedAt = info.CreationTime,
            });
        }

        // Safetensors directories (contain *.safetensors files + config.json)
        foreach (var dir in Directory.GetDirectories(_modelsDir))
        {
            var safetensorsFiles = Directory.GetFiles(dir, "*.safetensors");
            if (safetensorsFiles.Length > 0)
            {
                long totalSize = safetensorsFiles.Sum(f => new FileInfo(f).Length);
                var dirInfo = new DirectoryInfo(dir);
                models.Add(new OrphanedModel
                {
                    Name = dirInfo.Name,
                    Path = dirInfo.FullName,
                    Format = "directory",
                    SizeBytes = totalSize,
                    CreatedAt = dirInfo.CreationTime,
                });
            }
        }

        return models;
    }

    /// <summary>Normalize a path for comparison (full path, consistent separators).</summary>
    private static string NormalizePath(string path)
    {
        try
        {
            return Path.GetFullPath(path).Replace('\\', '/');
        }
        catch
        {
            return path.Replace('\\', '/');
        }
    }
}

/// <summary>A model file that is not referenced by any Modelfile.</summary>
public class OrphanedModel
{
    public string Name { get; set; } = "";
    public string Path { get; set; } = "";
    public string Format { get; set; } = "";
    public long SizeBytes { get; set; }
    public DateTime CreatedAt { get; set; }
}

/// <summary>Result of an orphan cleanup operation.</summary>
public class CleanupResult
{
    public int DeletedCount { get; set; }
    public long BytesFreed { get; set; }
    public List<string> DeletedPaths { get; set; } = new();
    public List<string> Errors { get; set; } = new();
    public bool DryRun { get; set; }
}
