---
layout: default
title: Streaming
parent: API Reference
nav_order: 2
---

# Streaming
{: .no_toc }

Real-time token delivery via Server-Sent Events or WebSocket.
{: .fs-6 .fw-300 }

<details open markdown="block">
  <summary>Table of contents</summary>
  {: .text-delta }
- TOC
{:toc}
</details>

---

## Server-Sent Events (SSE)

SSE is the standard streaming mode and is compatible with all OpenAI client libraries.

### Enabling SSE

Add `"stream": true` to your chat completion request:

```bash
curl http://localhost:3512/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sharpinfer",
    "messages": [{"role": "user", "content": "Tell me a story."}],
    "stream": true
  }'
```

### SSE Event Format

The response is a stream of `text/event-stream` events. Each token is one event:

```
data: {"id":"chatcmpl-abc","model":"sharpinfer","choices":[{"delta":{"role":"assistant"},"index":0}]}

data: {"id":"chatcmpl-abc","model":"sharpinfer","choices":[{"delta":{"content":"Once"},"index":0}]}

data: {"id":"chatcmpl-abc","model":"sharpinfer","choices":[{"delta":{"content":" upon"},"index":0}]}

data: {"id":"chatcmpl-abc","model":"sharpinfer","choices":[{"delta":{"content":" a"},"index":0}]}

data: {"id":"chatcmpl-abc","model":"sharpinfer","choices":[{"delta":{},"index":0,"finish_reason":"stop"}]}

data: [DONE]
```

### Parsing Rules

1. Each event is a line beginning with `data: `
2. Events are separated by blank lines (`\n\n`)
3. The first chunk delivers `delta.role = "assistant"`
4. Subsequent chunks deliver one token each in `delta.content`
5. The final chunk has an empty `delta` with `finish_reason: "stop"` (or `"length"` if `max_tokens` was reached)
6. The stream ends with the literal `data: [DONE]`
7. Accumulate all `delta.content` values to reconstruct the full response

### JavaScript Example

```javascript
const response = await fetch("http://localhost:3512/v1/chat/completions", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    model: "sharpinfer",
    messages: [{ role: "user", content: "Hello!" }],
    stream: true
  })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();
let fullText = "";

while (true) {
  const { done, value } = await reader.read();
  if (done) break;

  const chunk = decoder.decode(value);
  for (const line of chunk.split("\n")) {
    if (!line.startsWith("data: ")) continue;
    const data = line.slice(6);
    if (data === "[DONE]") break;

    const parsed = JSON.parse(data);
    const token = parsed.choices?.[0]?.delta?.content;
    if (token) {
      fullText += token;
      updateUI(token); // display token immediately
    }
  }
}
```

### Python (openai library)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:3512/v1", api_key="not-needed")

stream = client.chat.completions.create(
    model="sharpinfer",
    messages=[{"role": "user", "content": "Tell me a story."}],
    stream=True
)

for chunk in stream:
    token = chunk.choices[0].delta.content
    if token:
        print(token, end="", flush=True)
```

---

## WebSocket Streaming

WebSocket provides lower-latency streaming and a persistent connection. Use this if your frontend specifically needs bidirectional communication or wants to avoid per-request HTTP overhead.

{: .note }
For most integrations, SSE over HTTP is simpler and fully adequate. WebSocket uses a custom JSON-RPC protocol that is not OpenAI-compatible.

### Connecting

```
ws://localhost:3512/ws
```

### Sending a Request

Send a JSON message with `type` set to `generate` or `chat`:

```json
{
  "type": "chat",
  "id": "req-001",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ],
  "max_tokens": 256,
  "temperature": 0.7
}
```

| Field | Description |
|---|---|
| `type` | `generate` (prompt), `chat` (messages array), `cancel`, `ping`, `status` |
| `id` | Request ID string — returned in all responses to match replies to requests |
| `messages` | For `chat` type: array of `{role, content}` objects |
| `prompt` | For `generate` type: raw prompt string |
| `max_tokens` | Maximum tokens to generate |
| `temperature` | Sampling temperature |

### Receiving Tokens

As the model generates, you'll receive token notifications (no `id` field — they're notifications, not responses):

```json
{"jsonrpc":"2.0","method":"token","params":{"id":"req-001","token":"Hello","index":0}}
{"jsonrpc":"2.0","method":"token","params":{"id":"req-001","token":"!","index":1}}
```

When generation completes:

```json
{
  "jsonrpc": "2.0",
  "method": "done",
  "params": {
    "id": "req-001",
    "finish_reason": "stop",
    "usage": {
      "total_tokens": 12,
      "tokens_per_second": 45.3,
      "duration_ms": 265
    }
  }
}
```

### Cancelling a Request

```json
{"type": "cancel", "id": "req-001"}
```

The server confirms:
```json
{"jsonrpc":"2.0","method":"cancelled","params":{"id":"req-001"}}
```

### Connection Limits

Each IP address is limited to 5 concurrent WebSocket connections and 50 concurrent generation requests across all connections.

### Keepalive

Send a `ping` to check connection health:
```json
{"type": "ping"}
```
The server responds with `{"jsonrpc":"2.0","method":"pong","params":{}}`.
