---
layout: default
title: JSON-RPC Protocol
parent: VS Code Extension
nav_order: 4
---

# JSON-RPC Protocol Reference
{: .no_toc }

Full specification for the stdin/stdout communication protocol.
{: .fs-6 .fw-300 }

<details open markdown="block">
  <summary>Table of contents</summary>
  {: .text-delta }
- TOC
{:toc}
</details>

---

## Wire Format

Messages use **LSP-style content-length framing** — the same format as the Language Server Protocol. Each message is preceded by a header:

```
Content-Length: {byte length of JSON body}\r\n
\r\n
{JSON body}
```

Example:
```
Content-Length: 82\r\n
\r\n
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"modelPath":"/models/codellama.gguf"}}
```

---

## Message Types

### Request (client → server)

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "initialize",
  "params": { ... }
}
```

### Response (server → client)

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": { ... }
}
```

Or on error:
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "error": { "code": -32002, "message": "Engine not initialized" }
}
```

### Notification (server → client, streaming only)

Notifications have no `id` and expect no response:
```json
{
  "jsonrpc": "2.0",
  "method": "token",
  "params": { "text": "Hello" }
}
```

---

## Methods

### `initialize`

Load a model into the inference engine. Must be called before any other inference requests.

**Request params:**
```json
{
  "modelPath": "/path/to/codellama-7b.gguf",
  "contextLength": 4096
}
```

`contextLength` is optional — defaults to the model's built-in context length.

**Result:**
```json
{
  "success": true,
  "model": "CodeLlama 7B Q4_K_M",
  "vocabSize": 32000,
  "contextLength": 4096
}
```

---

### `complete`

Generate a code completion from a prefix (and optional suffix for FIM).

**Request params:**
```json
{
  "prefix": "def calculate_sum(items):\n    total = 0\n    ",
  "suffix": "\n    return total",
  "maxTokens": 128,
  "temperature": 0.2,
  "language": "python"
}
```

`suffix`, `temperature`, and `language` are optional.

**Result:**
```json
{
  "text": "for item in items:\n        total += item"
}
```

---

### `generate`

Generate text from a free-form prompt.

**Request params:**
```json
{
  "prompt": "Write a Python function that sorts a list of dicts by a key.",
  "maxTokens": 256,
  "temperature": 0.7,
  "stream": true
}
```

With `"stream": true`, tokens arrive as `token` notifications before the final response.

**Result:**
```json
{
  "text": "def sort_by_key(data, key):\n    return sorted(data, key=lambda x: x[key])",
  "done": true
}
```

---

### `chat`

Multi-turn conversation with a messages array.

**Request params:**
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful coding assistant."},
    {"role": "user", "content": "How do I reverse a string in Python?"}
  ],
  "maxTokens": 512,
  "temperature": 0.7,
  "stream": true
}
```

**Result:**
```json
{
  "text": "You can reverse a string in Python using slicing: `s[::-1]`",
  "done": true
}
```

---

### `tokenize`

Convert text to token IDs. Useful for debugging and measuring context usage.

**Request params:**
```json
{
  "text": "Hello, world!"
}
```

**Result:**
```json
{
  "tokens": [15043, 29892, 3186, 29991],
  "count": 4
}
```

---

### `reset`

Clear the KV cache and conversation history. Use between unrelated tasks to free memory.

**Request params:** `{}`

**Result:** `{"success": true}`

---

### `shutdown`

Gracefully shut down the engine and release resources.

**Request params:** `{}`

**Result:** `{"success": true}`

The process exits after responding.

---

## Streaming Notifications

When `"stream": true` is set on a `generate` or `chat` request:

**Token notification:**
```json
{"jsonrpc":"2.0","method":"token","params":{"text":"Hello"}}
```

**Log notification (debug info):**
```json
{"jsonrpc":"2.0","method":"log","params":{"message":"Loading model weights..."}}
```

The final `result` response signals that streaming is complete.

---

## Error Codes

| Code | Meaning |
|---|---|
| `-32601` | Method not found |
| `-32602` | Invalid params |
| `-32002` | Engine not initialized — send `initialize` first |
| `-32603` | Internal error during inference |

---

## TypeScript Helper

```typescript
import * as cp from "child_process";

class SharpInferBackend {
  private process: cp.ChildProcess;
  private pendingRequests = new Map<number, (result: any) => void>();
  private nextId = 1;
  private buffer = "";

  constructor(executablePath: string, modelPath: string) {
    this.process = cp.spawn("dotnet", ["run", "--project", executablePath]);
    this.process.stdout?.on("data", (data: Buffer) => this.onData(data));
    this.initialize(modelPath);
  }

  private onData(data: Buffer) {
    this.buffer += data.toString();
    // Parse Content-Length framed messages
    while (true) {
      const headerEnd = this.buffer.indexOf("\r\n\r\n");
      if (headerEnd === -1) break;
      const header = this.buffer.slice(0, headerEnd);
      const match = header.match(/Content-Length: (\d+)/);
      if (!match) break;
      const length = parseInt(match[1]);
      const bodyStart = headerEnd + 4;
      if (this.buffer.length < bodyStart + length) break;
      const body = this.buffer.slice(bodyStart, bodyStart + length);
      this.buffer = this.buffer.slice(bodyStart + length);
      this.handleMessage(JSON.parse(body));
    }
  }

  private handleMessage(msg: any) {
    if (msg.id && this.pendingRequests.has(msg.id)) {
      this.pendingRequests.get(msg.id)!(msg.result || msg.error);
      this.pendingRequests.delete(msg.id);
    }
    // Handle notifications (msg.method without msg.id) here
  }

  send(method: string, params: any): Promise<any> {
    const id = this.nextId++;
    const msg = JSON.stringify({ jsonrpc: "2.0", id, method, params });
    const frame = `Content-Length: ${Buffer.byteLength(msg)}\r\n\r\n${msg}`;
    this.process.stdin?.write(frame);
    return new Promise(resolve => this.pendingRequests.set(id, resolve));
  }

  initialize(modelPath: string) {
    return this.send("initialize", { modelPath, contextLength: 4096 });
  }

  complete(prefix: string, suffix?: string, maxTokens = 128) {
    return this.send("complete", { prefix, suffix, maxTokens });
  }

  chat(messages: any[], maxTokens = 512) {
    return this.send("chat", { messages, maxTokens, stream: true });
  }
}
```
