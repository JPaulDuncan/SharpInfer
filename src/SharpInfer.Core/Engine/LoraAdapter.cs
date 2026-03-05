using System.Text.Json;
using SharpInfer.Core.Models;
using SharpInfer.Core.Tensors;

namespace SharpInfer.Core.Engine;

/// <summary>
/// LoRA (Low-Rank Adaptation) hot-loading support.
///
/// LoRA works by adding small rank-decomposed delta matrices to the base model weights:
///   W_effective = W_base + (alpha/rank) * (B @ A)
///
/// Where:
///   W_base: original frozen weight [out_dim x in_dim]
///   A: low-rank down-projection [rank x in_dim]
///   B: low-rank up-projection [out_dim x rank]
///   alpha: scaling factor (typically = rank)
///   rank: typically 8, 16, 32, or 64
///
/// LoRA adapters are typically 0.1-1% the size of the base model,
/// enabling task-specific customization without reloading the full model.
///
/// Supports loading from:
///   - Hugging Face PEFT format (adapter_model.safetensors + adapter_config.json)
///   - Raw safetensors with standard naming conventions
///
/// Reference: https://arxiv.org/abs/2106.09685
/// </summary>
public class LoraAdapter : IDisposable
{
    public string Name { get; }
    public int Rank { get; }
    public float Alpha { get; }
    public float Scale => Alpha / Rank;
    public string[] TargetModules { get; }

    /// <summary>Per-layer LoRA weight pairs. Key: layer index, Value: the delta weights.</summary>
    private readonly Dictionary<int, LoraLayerWeights> _layers = new();

    /// <summary>Whether this adapter is currently applied to the base weights.</summary>
    public bool IsApplied { get; private set; }

    public LoraAdapter(string name, int rank, float alpha, string[] targetModules)
    {
        Name = name;
        Rank = rank;
        Alpha = alpha;
        TargetModules = targetModules;
    }

    /// <summary>
    /// Load a LoRA adapter from a Hugging Face PEFT directory.
    /// Expected files: adapter_config.json, adapter_model.safetensors
    /// </summary>
    public static LoraAdapter LoadFromDirectory(string path, Action<string>? progress = null)
    {
        var configPath = Path.Combine(path, "adapter_config.json");
        if (!File.Exists(configPath))
            throw new FileNotFoundException("adapter_config.json not found.", configPath);

        var configJson = JsonDocument.Parse(File.ReadAllText(configPath)).RootElement;

        int rank = configJson.GetProperty("r").GetInt32();
        float alpha = configJson.TryGetProperty("lora_alpha", out var a) ? a.GetSingle() : rank;
        var targetModules = configJson.GetProperty("target_modules")
            .EnumerateArray().Select(m => m.GetString()!).ToArray();
        string adapterName = Path.GetFileName(path);

        progress?.Invoke($"LoRA '{adapterName}': rank={rank}, alpha={alpha}, targets=[{string.Join(",", targetModules)}]");

        var adapter = new LoraAdapter(adapterName, rank, alpha, targetModules);

        // Load adapter weights
        var modelPath = Path.Combine(path, "adapter_model.safetensors");
        if (File.Exists(modelPath))
        {
            adapter.LoadSafetensors(modelPath, progress);
        }
        else
        {
            // Try .bin format (older PEFT)
            var binPath = Path.Combine(path, "adapter_model.bin");
            if (File.Exists(binPath))
                throw new NotSupportedException("PyTorch .bin LoRA files not yet supported. Convert to safetensors first.");
            else
                throw new FileNotFoundException("No adapter_model.safetensors found.", modelPath);
        }

        return adapter;
    }

    private void LoadSafetensors(string path, Action<string>? progress)
    {
        using var stream = File.OpenRead(path);
        using var reader = new BinaryReader(stream);

        ulong headerLen = reader.ReadUInt64();
        var headerBytes = reader.ReadBytes((int)headerLen);
        var header = JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(headerBytes)!;

        long dataStart = 8 + (long)headerLen;

        // PEFT naming convention:
        //   base_model.model.model.layers.{N}.self_attn.q_proj.lora_A.weight
        //   base_model.model.model.layers.{N}.self_attn.q_proj.lora_B.weight
        var tensorMap = new Dictionary<string, float[]>();

        foreach (var (name, info) in header)
        {
            if (name == "__metadata__") continue;

            var dtype = info.GetProperty("dtype").GetString()!;
            var shape = info.GetProperty("shape").EnumerateArray().Select(e => e.GetInt32()).ToArray();
            var offsets = info.GetProperty("data_offsets").EnumerateArray().Select(e => e.GetInt64()).ToArray();

            long start = dataStart + offsets[0];
            long end = dataStart + offsets[1];
            int byteLen = (int)(end - start);

            stream.Seek(start, SeekOrigin.Begin);
            var rawBytes = reader.ReadBytes(byteLen);
            int numElements = shape.Aggregate(1, (a, b) => a * b);

            // Convert F16/BF16/F32 to float
            var data = new float[numElements];
            switch (dtype)
            {
                case "F32":
                    Buffer.BlockCopy(rawBytes, 0, data, 0, rawBytes.Length);
                    break;
                case "F16":
                    for (int i = 0; i < numElements; i++)
                        data[i] = (float)BitConverter.UInt16BitsToHalf(BitConverter.ToUInt16(rawBytes, i * 2));
                    break;
                case "BF16":
                    for (int i = 0; i < numElements; i++)
                    {
                        ushort raw = BitConverter.ToUInt16(rawBytes, i * 2);
                        data[i] = BitConverter.UInt32BitsToSingle((uint)raw << 16);
                    }
                    break;
            }

            tensorMap[name] = data;
        }

        // Parse tensor names to build per-layer LoRA weights
        ParsePeftTensors(tensorMap, progress);
    }

    private void ParsePeftTensors(Dictionary<string, float[]> tensors, Action<string>? progress)
    {
        // Group by layer number and projection type
        // Name format: base_model.model.model.layers.{N}.self_attn.{proj}.lora_{A|B}.weight
        // Or shorter: model.layers.{N}.self_attn.{proj}.lora_{A|B}.weight

        var layerPattern = new System.Text.RegularExpressions.Regex(
            @"layers\.(\d+)\.(self_attn|mlp)\.(\w+)\.lora_([AB])\.weight");

        var grouped = new Dictionary<(int Layer, string Module), (float[]? A, float[]? B)>();

        foreach (var (name, data) in tensors)
        {
            var match = layerPattern.Match(name);
            if (!match.Success) continue;

            int layerIdx = int.Parse(match.Groups[1].Value);
            string block = match.Groups[2].Value;    // self_attn or mlp
            string proj = match.Groups[3].Value;     // q_proj, k_proj, etc.
            string ab = match.Groups[4].Value;       // A or B

            string module = $"{block}.{proj}";
            var key = (layerIdx, module);

            if (!grouped.ContainsKey(key))
                grouped[key] = (null, null);

            var current = grouped[key];
            grouped[key] = ab == "A" ? (data, current.B) : (current.A, data);
        }

        // Build LoRA layer weights
        foreach (var ((layerIdx, module), (loraA, loraB)) in grouped)
        {
            if (loraA == null || loraB == null)
            {
                progress?.Invoke($"  Warning: Incomplete LoRA pair for layer {layerIdx}.{module}");
                continue;
            }

            if (!_layers.ContainsKey(layerIdx))
                _layers[layerIdx] = new LoraLayerWeights();

            _layers[layerIdx].SetProjection(module, loraA, loraB);
        }

        progress?.Invoke($"  Loaded LoRA weights for {_layers.Count} layers");
    }

    /// <summary>
    /// Apply the LoRA adapter to base model weights (additive merge).
    /// W_effective = W_base + scale * (B @ A)
    /// </summary>
    public void Apply(ModelWeights baseWeights)
    {
        if (IsApplied)
            throw new InvalidOperationException($"LoRA '{Name}' is already applied. Unapply first.");

        foreach (var (layerIdx, loraWeights) in _layers)
        {
            if (layerIdx >= baseWeights.Layers.Length) continue;
            var layer = baseWeights.Layers[layerIdx];

            ApplyToProjection(layer.Wq, loraWeights.QProj, "q_proj", layerIdx);
            ApplyToProjection(layer.Wk, loraWeights.KProj, "k_proj", layerIdx);
            ApplyToProjection(layer.Wv, loraWeights.VProj, "v_proj", layerIdx);
            ApplyToProjection(layer.Wo, loraWeights.OProj, "o_proj", layerIdx);
            ApplyToProjection(layer.WGate, loraWeights.GateProj, "gate_proj", layerIdx);
            ApplyToProjection(layer.WUp, loraWeights.UpProj, "up_proj", layerIdx);
            ApplyToProjection(layer.WDown, loraWeights.DownProj, "down_proj", layerIdx);
        }

        IsApplied = true;
    }

    /// <summary>
    /// Remove the LoRA adapter from base model weights (subtractive).
    /// W_base = W_effective - scale * (B @ A)
    /// </summary>
    public void Unapply(ModelWeights baseWeights)
    {
        if (!IsApplied)
            throw new InvalidOperationException($"LoRA '{Name}' is not currently applied.");

        foreach (var (layerIdx, loraWeights) in _layers)
        {
            if (layerIdx >= baseWeights.Layers.Length) continue;
            var layer = baseWeights.Layers[layerIdx];

            UnapplyFromProjection(layer.Wq, loraWeights.QProj);
            UnapplyFromProjection(layer.Wk, loraWeights.KProj);
            UnapplyFromProjection(layer.Wv, loraWeights.VProj);
            UnapplyFromProjection(layer.Wo, loraWeights.OProj);
            UnapplyFromProjection(layer.WGate, loraWeights.GateProj);
            UnapplyFromProjection(layer.WUp, loraWeights.UpProj);
            UnapplyFromProjection(layer.WDown, loraWeights.DownProj);
        }

        IsApplied = false;
    }

    private void ApplyToProjection(float[] baseWeight, LoraProjection? proj, string name, int layer)
    {
        if (proj == null) return;

        // Compute delta = scale * (B @ A) and add to base weight
        // A: [rank x in_dim], B: [out_dim x rank]
        // delta: [out_dim x in_dim]
        var delta = ComputeDelta(proj.LoraA, proj.LoraB, proj.OutDim, proj.InDim, Rank);

        for (int i = 0; i < baseWeight.Length && i < delta.Length; i++)
            baseWeight[i] += delta[i] * Scale;
    }

    private void UnapplyFromProjection(float[] baseWeight, LoraProjection? proj)
    {
        if (proj == null) return;

        var delta = ComputeDelta(proj.LoraA, proj.LoraB, proj.OutDim, proj.InDim, Rank);

        for (int i = 0; i < baseWeight.Length && i < delta.Length; i++)
            baseWeight[i] -= delta[i] * Scale;
    }

    /// <summary>Compute B @ A (matrix multiply of the two low-rank factors).</summary>
    private static float[] ComputeDelta(float[] a, float[] b, int outDim, int inDim, int rank)
    {
        // B: [outDim x rank], A: [rank x inDim]
        // Result: [outDim x inDim]
        var result = new float[outDim * inDim];
        TensorOps.MatMul(b, a, result, outDim, rank, inDim);
        return result;
    }

    public void Dispose()
    {
        _layers.Clear();
        GC.SuppressFinalize(this);
    }
}

/// <summary>LoRA weight pairs for a single transformer layer.</summary>
public class LoraLayerWeights
{
    public LoraProjection? QProj { get; set; }
    public LoraProjection? KProj { get; set; }
    public LoraProjection? VProj { get; set; }
    public LoraProjection? OProj { get; set; }
    public LoraProjection? GateProj { get; set; }
    public LoraProjection? UpProj { get; set; }
    public LoraProjection? DownProj { get; set; }

    public void SetProjection(string module, float[] loraA, float[] loraB)
    {
        // A: [rank x in_dim], B: [out_dim x rank]
        int rank = loraA.Length / (loraA.Length > loraB.Length
            ? loraA.Length / (loraB.Length / (loraB.Length > loraA.Length ? 1 : 1))
            : 1);

        // Infer dimensions: A is [rank x in_dim], B is [out_dim x rank]
        // We need to figure out rank from the tensor sizes
        // If A has shape [r, in], then A.Length = r * in
        // If B has shape [out, r], then B.Length = out * r
        // rank is the shared dimension: gcd approach or just try common values
        int inferredRank = InferRank(loraA.Length, loraB.Length);
        int inDim = loraA.Length / inferredRank;
        int outDim = loraB.Length / inferredRank;

        var proj = new LoraProjection
        {
            LoraA = loraA,
            LoraB = loraB,
            InDim = inDim,
            OutDim = outDim,
        };

        switch (module)
        {
            case "self_attn.q_proj": QProj = proj; break;
            case "self_attn.k_proj": KProj = proj; break;
            case "self_attn.v_proj": VProj = proj; break;
            case "self_attn.o_proj": OProj = proj; break;
            case "mlp.gate_proj": GateProj = proj; break;
            case "mlp.up_proj": UpProj = proj; break;
            case "mlp.down_proj": DownProj = proj; break;
        }
    }

    private static int InferRank(int aSize, int bSize)
    {
        // Common LoRA ranks
        int[] commonRanks = { 4, 8, 16, 32, 64, 128, 256 };
        foreach (int r in commonRanks)
        {
            if (aSize % r == 0 && bSize % r == 0)
                return r;
        }
        // Fallback: GCD
        return GCD(aSize, bSize);
    }

    private static int GCD(int a, int b) => b == 0 ? a : GCD(b, a % b);
}

/// <summary>A single LoRA low-rank projection pair (A and B matrices).</summary>
public class LoraProjection
{
    public required float[] LoraA { get; init; }  // [rank x in_dim]
    public required float[] LoraB { get; init; }  // [out_dim x rank]
    public int InDim { get; init; }
    public int OutDim { get; init; }
}
