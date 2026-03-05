using System.IO.Pipes;
using System.Text;
using System.Text.Json;
using SharpInfer.Core.Engine;
using SharpInfer.Core.Sampling;

namespace SharpInfer.VsCode;

/// <summary>
/// VS Code extension bridge using JSON-RPC over stdin/stdout.
///
/// This process is spawned by a VS Code extension and communicates via
/// the Language Server Protocol (LSP)-style JSON-RPC messages.
///
/// The VS Code extension (TypeScript side) sends requests like:
///   {"jsonrpc":"2.0","id":1,"method":"generate","params":{"prompt":"...","stream":true}}
///
/// And receives responses:
///   {"jsonrpc":"2.0","id":1,"result":{"text":"..."}}
///
/// For streaming, notifications are sent:
///   {"jsonrpc":"2.0","method":"token","params":{"text":"..."}}
///
/// Supported methods:
///   initialize     — Load model, return capabilities
///   generate       — Generate text (streaming or non-streaming)
///   complete       — Code completion (inline suggestions)
///   chat           — Multi-turn chat
///   tokenize       — Tokenize text (debugging)
///   reset          — Reset KV cache
///   shutdown       — Clean shutdown
/// </summary>
public class LanguageServerBridge : IDisposable
{
    private InferenceEngine? _engine;
    private readonly Stream _input;
    private readonly Stream _output;
    private readonly object _writeLock = new();

    public LanguageServerBridge(Stream input, Stream output)
    {
        _input = input;
        _output = output;
    }

    /// <summary>Main message loop — reads JSON-RPC requests from stdin and dispatches them.</summary>
    public async Task RunAsync(CancellationToken ct = default)
    {
        using var reader = new StreamReader(_input, Encoding.UTF8);

        while (!ct.IsCancellationRequested)
        {
            // Read Content-Length header (LSP framing)
            string? headerLine = await reader.ReadLineAsync(ct);
            if (headerLine == null) break;

            int contentLength = 0;
            while (!string.IsNullOrEmpty(headerLine))
            {
                if (headerLine.StartsWith("Content-Length:"))
                    contentLength = int.Parse(headerLine["Content-Length:".Length..].Trim());
                headerLine = await reader.ReadLineAsync(ct);
            }

            if (contentLength == 0) continue;

            // Read message body
            var buffer = new char[contentLength];
            int read = await reader.ReadBlockAsync(buffer, 0, contentLength);
            string messageJson = new string(buffer, 0, read);

            try
            {
                var message = JsonDocument.Parse(messageJson).RootElement;
                await DispatchAsync(message, ct);
            }
            catch (Exception ex)
            {
                LogError($"Error processing message: {ex.Message}");
            }
        }
    }

    private async Task DispatchAsync(JsonElement message, CancellationToken ct)
    {
        string method = message.GetProperty("method").GetString()!;
        int? id = message.TryGetProperty("id", out var idProp) ? idProp.GetInt32() : null;
        var @params = message.TryGetProperty("params", out var p) ? p : default;

        switch (method)
        {
            case "initialize":
                await HandleInitialize(id, @params);
                break;
            case "generate":
                await HandleGenerate(id, @params, ct);
                break;
            case "complete":
                await HandleComplete(id, @params, ct);
                break;
            case "chat":
                await HandleChat(id, @params, ct);
                break;
            case "tokenize":
                HandleTokenize(id, @params);
                break;
            case "reset":
                _engine?.ResetContext();
                SendResult(id, new { success = true });
                break;
            case "shutdown":
                _engine?.Dispose();
                SendResult(id, new { success = true });
                return;
            default:
                SendError(id, -32601, $"Method not found: {method}");
                break;
        }
    }

    private async Task HandleInitialize(int? id, JsonElement @params)
    {
        string modelPath = @params.GetProperty("modelPath").GetString()!;
        int? contextLength = @params.TryGetProperty("contextLength", out var cl) ? cl.GetInt32() : null;

        var options = new EngineOptions
        {
            ContextLength = contextLength,
            ProgressCallback = msg => SendNotification("log", new { message = msg }),
        };

        _engine = InferenceEngine.Load(modelPath, options);

        SendResult(id, new
        {
            success = true,
            model = _engine.Config.ToString(),
            vocabSize = _engine.Config.VocabSize,
            contextLength = _engine.Config.MaxSequenceLength,
        });
    }

    private async Task HandleGenerate(int? id, JsonElement @params, CancellationToken ct)
    {
        if (_engine == null) { SendError(id, -32002, "Engine not initialized"); return; }

        string prompt = @params.GetProperty("prompt").GetString()!;
        bool stream = @params.TryGetProperty("stream", out var s) && s.GetBoolean();

        var config = ParseGenConfig(@params);

        if (stream)
        {
            var sb = new StringBuilder();
            await foreach (var token in _engine.GenerateAsync(prompt, config))
            {
                if (ct.IsCancellationRequested) break;
                sb.Append(token);
                SendNotification("token", new { text = token });
            }
            SendResult(id, new { text = sb.ToString(), done = true });
        }
        else
        {
            string result = await _engine.GenerateCompleteAsync(prompt, config);
            SendResult(id, new { text = result });
        }
    }

    private async Task HandleComplete(int? id, JsonElement @params, CancellationToken ct)
    {
        if (_engine == null) { SendError(id, -32002, "Engine not initialized"); return; }

        string prefix = @params.GetProperty("prefix").GetString()!;
        string? suffix = @params.TryGetProperty("suffix", out var suf) ? suf.GetString() : null;

        // For code completion, use lower temperature and shorter max tokens
        var config = new GenerationConfig
        {
            Temperature = 0.2f,
            TopP = 0.95f,
            MaxTokens = @params.TryGetProperty("maxTokens", out var mt) ? mt.GetInt32() : 128,
            StopStrings = { "\n\n", "```" },
        };

        // Build fill-in-the-middle prompt if suffix is provided
        string prompt = suffix != null
            ? $"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>"
            : prefix;

        string completion = await _engine.GenerateCompleteAsync(prompt, config);
        SendResult(id, new { text = completion });
    }

    private async Task HandleChat(int? id, JsonElement @params, CancellationToken ct)
    {
        if (_engine == null) { SendError(id, -32002, "Engine not initialized"); return; }

        var messages = @params.GetProperty("messages");
        var sb = new StringBuilder();

        foreach (var msg in messages.EnumerateArray())
        {
            string role = msg.GetProperty("role").GetString()!;
            string content = msg.GetProperty("content").GetString()!;
            sb.AppendLine($"<|{role}|>\n{content}\n<|end|>");
        }
        sb.Append("<|assistant|>\n");

        var config = ParseGenConfig(@params);
        bool stream = @params.TryGetProperty("stream", out var s) && s.GetBoolean();

        if (stream)
        {
            var response = new StringBuilder();
            await foreach (var token in _engine.GenerateAsync(sb.ToString(), config))
            {
                if (ct.IsCancellationRequested) break;
                response.Append(token);
                SendNotification("token", new { text = token });
            }
            SendResult(id, new { text = response.ToString(), done = true });
        }
        else
        {
            string result = await _engine.GenerateCompleteAsync(sb.ToString(), config);
            SendResult(id, new { text = result });
        }
    }

    private void HandleTokenize(int? id, JsonElement @params)
    {
        if (_engine == null) { SendError(id, -32002, "Engine not initialized"); return; }

        string text = @params.GetProperty("text").GetString()!;
        var tokens = _engine.Tokenize(text);
        SendResult(id, new { tokens, count = tokens.Count });
    }

    #region JSON-RPC Helpers

    private void SendResult(int? id, object result)
    {
        if (id == null) return;
        var response = JsonSerializer.Serialize(new { jsonrpc = "2.0", id, result });
        WriteMessage(response);
    }

    private void SendError(int? id, int code, string message)
    {
        var response = JsonSerializer.Serialize(new
        {
            jsonrpc = "2.0",
            id,
            error = new { code, message }
        });
        WriteMessage(response);
    }

    private void SendNotification(string method, object @params)
    {
        var notification = JsonSerializer.Serialize(new { jsonrpc = "2.0", method, @params });
        WriteMessage(notification);
    }

    private void WriteMessage(string json)
    {
        var bytes = Encoding.UTF8.GetBytes(json);
        var header = Encoding.UTF8.GetBytes($"Content-Length: {bytes.Length}\r\n\r\n");

        lock (_writeLock)
        {
            _output.Write(header);
            _output.Write(bytes);
            _output.Flush();
        }
    }

    private void LogError(string message)
    {
        SendNotification("log", new { level = "error", message });
    }

    #endregion

    private static GenerationConfig ParseGenConfig(JsonElement @params) => new()
    {
        Temperature = @params.TryGetProperty("temperature", out var t) ? t.GetSingle() : 0.7f,
        TopP = @params.TryGetProperty("topP", out var tp) ? tp.GetSingle() : 0.9f,
        TopK = @params.TryGetProperty("topK", out var tk) ? tk.GetInt32() : 40,
        MaxTokens = @params.TryGetProperty("maxTokens", out var mt) ? mt.GetInt32() : 512,
    };

    public void Dispose()
    {
        _engine?.Dispose();
        GC.SuppressFinalize(this);
    }
}
