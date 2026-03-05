using System.Text.Json;
using System.Text.Json.Serialization;

namespace SharpInfer.Core.Engine;

/// <summary>
/// Unified configuration for all SharpInfer engine features.
/// Can be loaded from a JSON config file, environment variables, or CLI flags.
///
/// All advanced features are disabled by default and opt-in via configuration.
/// </summary>
public class EngineConfig
{
    // === Model ===
    [JsonPropertyName("model_path")]
    public string ModelPath { get; set; } = "";

    [JsonPropertyName("context_length")]
    public int? ContextLength { get; set; }

    // === Compute ===
    [JsonPropertyName("gpu")]
    public GpuConfig Gpu { get; set; } = new();

    // === Generation Defaults ===
    [JsonPropertyName("generation")]
    public GenerationDefaults Generation { get; set; } = new();

    // === Speculative Decoding ===
    [JsonPropertyName("speculative")]
    public SpeculativeConfig Speculative { get; set; } = new();

    // === LoRA Adapters ===
    [JsonPropertyName("lora")]
    public LoraConfig Lora { get; set; } = new();

    // === Prompt Caching ===
    [JsonPropertyName("prompt_cache")]
    public PromptCacheConfig PromptCaching { get; set; } = new();

    // === Classifier-Free Guidance ===
    [JsonPropertyName("cfg")]
    public CfgConfig Cfg { get; set; } = new();

    // === Self-Consistency ===
    [JsonPropertyName("self_consistency")]
    public SelfConsistencyConfig SelfConsistency { get; set; } = new();

    // === RAG ===
    [JsonPropertyName("rag")]
    public RagConfig Rag { get; set; } = new();

    // === Beam Search ===
    [JsonPropertyName("beam_search")]
    public BeamSearchConfig BeamSearch { get; set; } = new();

    // === Tools ===
    [JsonPropertyName("tools")]
    public ToolsConfig Tools { get; set; } = new();

    // === MCP Servers ===
    [JsonPropertyName("mcp_servers")]
    public List<Mcp.McpServerConfig> McpServers { get; set; } = new();

    // === Multi-Agent ===
    [JsonPropertyName("agents")]
    public AgentsConfig MultiAgent { get; set; } = new();

    // === Structured Output ===
    [JsonPropertyName("structured_output")]
    public StructuredOutputEngineConfig StructuredOutput { get; set; } = new();

    // === Batch Scheduler ===
    [JsonPropertyName("batch_scheduler")]
    public BatchSchedulerConfig BatchScheduler { get; set; } = new();

    // === FlashAttention ===
    [JsonPropertyName("flash_attention")]
    public FlashAttentionEngineConfig FlashAttention { get; set; } = new();

    // === Multimodal / Vision ===
    [JsonPropertyName("multimodal")]
    public MultimodalConfig Multimodal { get; set; } = new();

    // === Hardware Backend ===
    [JsonPropertyName("hardware_backend")]
    public HardwareBackendConfig HardwareBackend { get; set; } = new();

    // === Quantization ===
    [JsonPropertyName("quantization")]
    public QuantizationConfig Quantization { get; set; } = new();

    // === Model Packaging ===
    [JsonPropertyName("modelfile")]
    public string? ModelfilePath { get; set; }

    // === Model Registry ===
    [JsonPropertyName("models_dir")]
    public string ModelsDir { get; set; } = "./models";

    // === Distributed Inference ===
    [JsonPropertyName("distributed")]
    public DistributedConfig Distributed { get; set; } = new();

    // === Cloud Fallback ===
    [JsonPropertyName("cloud_fallback")]
    public CloudFallbackConfig CloudFallback { get; set; } = new();

    // === WebSocket ===
    [JsonPropertyName("websocket")]
    public WebSocketConfig WebSocket { get; set; } = new();

    // === Batch Processing ===
    [JsonPropertyName("batch_processing")]
    public BatchProcessingConfig BatchProcessing { get; set; } = new();

    // === Enterprise ===
    [JsonPropertyName("enterprise")]
    public EnterpriseEngineConfig Enterprise { get; set; } = new();

    // === Model Pull ===
    [JsonPropertyName("model_pull")]
    public ModelPullConfig ModelPull { get; set; } = new();

    // === API Server ===
    [JsonPropertyName("api")]
    public ApiConfig Api { get; set; } = new();

    /// <summary>Load config from a JSON file.</summary>
    public static EngineConfig LoadFromFile(string path)
    {
        var json = File.ReadAllText(path);
        return JsonSerializer.Deserialize<EngineConfig>(json, new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true,
            ReadCommentHandling = JsonCommentHandling.Skip,
            AllowTrailingCommas = true,
        }) ?? new EngineConfig();
    }

    /// <summary>Save config to a JSON file.</summary>
    public void SaveToFile(string path)
    {
        var json = JsonSerializer.Serialize(this, new JsonSerializerOptions
        {
            WriteIndented = true,
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingDefault,
        });
        File.WriteAllText(path, json);
    }

    /// <summary>Generate a default config file for reference.</summary>
    public static void GenerateDefaultConfig(string path)
    {
        var config = new EngineConfig
        {
            ModelPath = "./models/your-model.gguf",
            Generation = new GenerationDefaults
            {
                Temperature = 0.7f,
                TopP = 0.9f,
                TopK = 40,
                MaxTokens = 512,
            }
        };
        config.SaveToFile(path);
    }
}

public class GpuConfig
{
    [JsonPropertyName("enabled")] public bool Enabled { get; set; } = false;
    [JsonPropertyName("device_id")] public int DeviceId { get; set; } = 0;
    [JsonPropertyName("gpu_layers")] public int? GpuLayers { get; set; } // null = all layers
}

public class GenerationDefaults
{
    [JsonPropertyName("temperature")] public float Temperature { get; set; } = 0.7f;
    [JsonPropertyName("top_p")] public float TopP { get; set; } = 0.9f;
    [JsonPropertyName("top_k")] public int TopK { get; set; } = 40;
    [JsonPropertyName("max_tokens")] public int MaxTokens { get; set; } = 512;
    [JsonPropertyName("repetition_penalty")] public float RepetitionPenalty { get; set; } = 1.1f;
    [JsonPropertyName("seed")] public int Seed { get; set; } = -1;
    [JsonPropertyName("system_prompt")] public string SystemPrompt { get; set; } = "You are a helpful AI assistant.";
}

public class SpeculativeConfig
{
    [JsonPropertyName("enabled")] public bool Enabled { get; set; } = false;
    [JsonPropertyName("draft_model_path")] public string DraftModelPath { get; set; } = "";
    [JsonPropertyName("lookahead_tokens")] public int LookaheadTokens { get; set; } = 5;
}

public class LoraConfig
{
    [JsonPropertyName("enabled")] public bool Enabled { get; set; } = false;
    [JsonPropertyName("adapters")] public List<LoraAdapterEntry> Adapters { get; set; } = new();
    [JsonPropertyName("active_adapter")] public string? ActiveAdapter { get; set; }
}

public class LoraAdapterEntry
{
    [JsonPropertyName("name")] public string Name { get; set; } = "";
    [JsonPropertyName("path")] public string Path { get; set; } = "";
}

public class PromptCacheConfig
{
    [JsonPropertyName("enabled")] public bool Enabled { get; set; } = false;
    [JsonPropertyName("cache_dir")] public string CacheDir { get; set; } = "./.cache/prompts";
}

public class CfgConfig
{
    [JsonPropertyName("enabled")] public bool Enabled { get; set; } = false;
    [JsonPropertyName("guidance_scale")] public float GuidanceScale { get; set; } = 1.5f;
    [JsonPropertyName("negative_prompt")] public string NegativePrompt { get; set; } = "";
}

public class SelfConsistencyConfig
{
    [JsonPropertyName("enabled")] public bool Enabled { get; set; } = false;
    [JsonPropertyName("num_candidates")] public int NumCandidates { get; set; } = 5;
    [JsonPropertyName("candidate_temperature")] public float CandidateTemperature { get; set; } = 0.8f;
    [JsonPropertyName("strategy")] public string Strategy { get; set; } = "majority_vote";
}

public class RagConfig
{
    [JsonPropertyName("enabled")] public bool Enabled { get; set; } = false;
    [JsonPropertyName("documents")] public List<string> Documents { get; set; } = new();
    [JsonPropertyName("chunk_size")] public int ChunkSize { get; set; } = 512;
    [JsonPropertyName("chunk_overlap")] public int ChunkOverlap { get; set; } = 64;
    [JsonPropertyName("top_k")] public int TopK { get; set; } = 5;
    [JsonPropertyName("min_score")] public float MinScore { get; set; } = 0.3f;
    [JsonPropertyName("vector_store")] public string VectorStore { get; set; } = "memory";
    [JsonPropertyName("vector_db_path")] public string? VectorDbPath { get; set; }
}

public class BeamSearchConfig
{
    [JsonPropertyName("enabled")] public bool Enabled { get; set; } = false;
    [JsonPropertyName("num_beams")] public int NumBeams { get; set; } = 4;
    [JsonPropertyName("length_penalty")] public float LengthPenalty { get; set; } = 1.0f;
    [JsonPropertyName("diversity_penalty")] public float DiversityPenalty { get; set; } = 0f;
    [JsonPropertyName("early_stopping")] public bool EarlyStopping { get; set; } = true;
}

public class ToolsConfig
{
    [JsonPropertyName("enabled")] public bool Enabled { get; set; } = false;
    [JsonPropertyName("search_api")] public string? SearchApiEndpoint { get; set; }
    [JsonPropertyName("search_api_key")] public string? SearchApiKey { get; set; }
    [JsonPropertyName("url_reader")] public bool UrlReader { get; set; } = true;
}

public class AgentsConfig
{
    [JsonPropertyName("enabled")] public bool Enabled { get; set; } = false;
    [JsonPropertyName("agents")] public List<AgentDefinition> Agents { get; set; } = new();
    [JsonPropertyName("flows")] public List<FlowDefinition> Flows { get; set; } = new();
}

public class AgentDefinition
{
    [JsonPropertyName("name")] public string Name { get; set; } = "";
    [JsonPropertyName("role")] public string Role { get; set; } = "";
    [JsonPropertyName("system_prompt")] public string SystemPrompt { get; set; } = "You are a helpful assistant.";
    [JsonPropertyName("temperature")] public float Temperature { get; set; } = 0.7f;
    [JsonPropertyName("top_p")] public float TopP { get; set; } = 0.9f;
    [JsonPropertyName("max_tokens")] public int MaxTokens { get; set; } = 1024;
    [JsonPropertyName("tools")] public List<string> Tools { get; set; } = new();
    [JsonPropertyName("model_path")] public string? ModelPath { get; set; }
    [JsonPropertyName("can_handoff")] public bool CanHandoff { get; set; } = true;
    [JsonPropertyName("handoff_targets")] public List<string> HandoffTargets { get; set; } = new();
}

public class FlowDefinition
{
    [JsonPropertyName("name")] public string Name { get; set; } = "";
    [JsonPropertyName("type")] public string Type { get; set; } = "chain"; // chain, parallel, debate, router, handoff, map-reduce
    [JsonPropertyName("agents")] public List<string> Agents { get; set; } = new();
    [JsonPropertyName("synthesizer")] public string? Synthesizer { get; set; }
    [JsonPropertyName("judge")] public string? Judge { get; set; }
    [JsonPropertyName("entry_agent")] public string? EntryAgent { get; set; }
    [JsonPropertyName("routes")] public Dictionary<string, string>? Routes { get; set; }
    [JsonPropertyName("fallback")] public string? Fallback { get; set; }
    [JsonPropertyName("rounds")] public int Rounds { get; set; } = 3;
    [JsonPropertyName("max_turns")] public int MaxTurns { get; set; } = 50;
    [JsonPropertyName("pass_full_history")] public bool PassFullHistory { get; set; } = false;
}

public class StructuredOutputEngineConfig
{
    [JsonPropertyName("enabled")] public bool Enabled { get; set; } = false;
    [JsonPropertyName("default_type")] public string DefaultType { get; set; } = "none";
    [JsonPropertyName("default_schema")] public string? DefaultSchema { get; set; }
}

public class BatchSchedulerConfig
{
    [JsonPropertyName("enabled")] public bool Enabled { get; set; } = false;
    [JsonPropertyName("max_batch_size")] public int MaxBatchSize { get; set; } = 8;
    [JsonPropertyName("max_sequences")] public int MaxSequences { get; set; } = 16;
}

public class FlashAttentionEngineConfig
{
    [JsonPropertyName("enabled")] public bool Enabled { get; set; } = true;
    [JsonPropertyName("block_size_q")] public int BlockSizeQ { get; set; } = 64;
    [JsonPropertyName("block_size_kv")] public int BlockSizeKV { get; set; } = 64;
    [JsonPropertyName("prefer_cuda")] public bool PreferCuda { get; set; } = true;
}

public class MultimodalConfig
{
    [JsonPropertyName("enabled")] public bool Enabled { get; set; } = false;
    [JsonPropertyName("vision_model")] public string? VisionModelPath { get; set; }
    [JsonPropertyName("image_size")] public int ImageSize { get; set; } = 336;
    [JsonPropertyName("patch_size")] public int PatchSize { get; set; } = 14;
    [JsonPropertyName("max_images")] public int MaxImages { get; set; } = 4;
    [JsonPropertyName("high_res_tiling")] public bool HighResTiling { get; set; } = true;
    [JsonPropertyName("max_tiles")] public int MaxTilesPerImage { get; set; } = 6;
}

public class HardwareBackendConfig
{
    [JsonPropertyName("backend")] public string Backend { get; set; } = "auto"; // auto, cuda, metal, vulkan, neon, cpu
    [JsonPropertyName("device_id")] public int DeviceId { get; set; } = 0;
}

public class QuantizationConfig
{
    [JsonPropertyName("type")] public string Type { get; set; } = "auto"; // auto, F32, F16, Q8_0, Q6_K, Q4_K, Q3_K, Q2_K, BitNet
    [JsonPropertyName("dynamic_requantize")] public bool DynamicRequantize { get; set; } = false;
    [JsonPropertyName("target_type")] public string? TargetType { get; set; }
}

public class WebSocketConfig
{
    [JsonPropertyName("enabled")] public bool Enabled { get; set; } = false;
    [JsonPropertyName("max_connections_per_ip")] public int MaxConnectionsPerIp { get; set; } = 5;
    [JsonPropertyName("max_concurrent_requests")] public int MaxConcurrentRequests { get; set; } = 50;
}

public class ApiConfig
{
    [JsonPropertyName("port")] public int Port { get; set; } = 8080;
    [JsonPropertyName("host")] public string Host { get; set; } = "0.0.0.0";
    [JsonPropertyName("cors")] public bool Cors { get; set; } = true;
}

public class BatchProcessingConfig
{
    [JsonPropertyName("enabled")] public bool Enabled { get; set; } = false;
    [JsonPropertyName("max_concurrency")] public int MaxConcurrency { get; set; } = 4;
    [JsonPropertyName("max_batch_size")] public int MaxBatchSize { get; set; } = 1000;
    [JsonPropertyName("result_expiry_hours")] public int ResultExpiryHours { get; set; } = 24;
}

public class EnterpriseEngineConfig
{
    [JsonPropertyName("auth_enabled")] public bool AuthEnabled { get; set; } = false;
    [JsonPropertyName("rate_limit_enabled")] public bool RateLimitEnabled { get; set; } = false;
    [JsonPropertyName("audit_enabled")] public bool AuditEnabled { get; set; } = false;
    [JsonPropertyName("metering_enabled")] public bool MeteringEnabled { get; set; } = false;
    [JsonPropertyName("api_keys_file")] public string? ApiKeysFile { get; set; }
    [JsonPropertyName("audit_log_file")] public string? AuditLogFile { get; set; } = "./logs/audit.jsonl";
    [JsonPropertyName("requests_per_minute")] public int RequestsPerMinute { get; set; } = 60;
    [JsonPropertyName("ip_allowlist")] public List<string> IpAllowlist { get; set; } = new();
    [JsonPropertyName("ip_blocklist")] public List<string> IpBlocklist { get; set; } = new();
}

public class ModelPullConfig
{
    [JsonPropertyName("enabled")] public bool Enabled { get; set; } = true;
    [JsonPropertyName("hf_token")] public string? HuggingFaceToken { get; set; }
    [JsonPropertyName("default_quant")] public string DefaultQuant { get; set; } = "Q4_K_M";
    [JsonPropertyName("auto_download")] public bool AutoDownload { get; set; } = false;
}
