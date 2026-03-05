---
layout: default
title: Code Examples
parent: Integration Guide
nav_order: 2
---

# Code Examples
{: .no_toc }

Copy-paste ready examples for Python, JavaScript/TypeScript, and curl.
{: .fs-6 .fw-300 }

<details open markdown="block">
  <summary>Table of contents</summary>
  {: .text-delta }
- TOC
{:toc}
</details>

---

## Python

### Non-Streaming (openai library)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:3512/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="sharpinfer",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is async/await in C#?"}
    ],
    temperature=0.7,
    max_tokens=512
)

print(response.choices[0].message.content)
print(f"Tokens used: {response.usage.total_tokens}")
```

### Streaming (openai library)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:3512/v1",
    api_key="not-needed"
)

stream = client.chat.completions.create(
    model="sharpinfer",
    messages=[{"role": "user", "content": "Write a short poem about C#."}],
    stream=True
)

for chunk in stream:
    token = chunk.choices[0].delta.content
    if token:
        print(token, end="", flush=True)
print()
```

### Non-Streaming (requests)

```python
import requests

response = requests.post(
    "http://localhost:3512/v1/chat/completions",
    json={
        "model": "sharpinfer",
        "messages": [
            {"role": "user", "content": "Explain dependency injection."}
        ],
        "temperature": 0.7,
        "max_tokens": 256
    }
)

data = response.json()
print(data["choices"][0]["message"]["content"])
```

### Streaming (requests)

```python
import requests
import json

with requests.post(
    "http://localhost:3512/v1/chat/completions",
    json={"model": "sharpinfer", "messages": [{"role": "user", "content": "Hello!"}], "stream": True},
    stream=True
) as response:
    for line in response.iter_lines():
        if not line:
            continue
        line = line.decode("utf-8")
        if not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            break
        chunk = json.loads(data)
        token = chunk["choices"][0]["delta"].get("content", "")
        print(token, end="", flush=True)
```

### Multi-Turn Conversation

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:3512/v1", api_key="not-needed")
history = [{"role": "system", "content": "You are a helpful coding assistant."}]

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ("quit", "exit"):
        break

    history.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model="sharpinfer",
        messages=history,
        max_tokens=512
    )

    assistant_reply = response.choices[0].message.content
    history.append({"role": "assistant", "content": assistant_reply})
    print(f"Assistant: {assistant_reply}\n")
```

---

## JavaScript / TypeScript

### Non-Streaming (fetch)

```javascript
const response = await fetch("http://localhost:3512/v1/chat/completions", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    model: "sharpinfer",
    messages: [
      { role: "system", content: "You are a helpful assistant." },
      { role: "user", content: "What is TypeScript?" }
    ],
    temperature: 0.7,
    max_tokens: 256
  })
});

const data = await response.json();
console.log(data.choices[0].message.content);
console.log(`Tokens: ${data.usage.total_tokens}`);
```

### Streaming (fetch + ReadableStream)

```javascript
async function streamChat(messages) {
  const response = await fetch("http://localhost:3512/v1/chat/completions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: "sharpinfer", messages, stream: true })
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let fullText = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const chunk = decoder.decode(value, { stream: true });
    for (const line of chunk.split("\n")) {
      if (!line.startsWith("data: ")) continue;
      const data = line.slice(6);
      if (data === "[DONE]") return fullText;

      try {
        const parsed = JSON.parse(data);
        const token = parsed.choices?.[0]?.delta?.content ?? "";
        fullText += token;
        process.stdout.write(token); // or update DOM
      } catch {}
    }
  }

  return fullText;
}

// Usage
const text = await streamChat([
  { role: "user", content: "Explain React hooks." }
]);
```

### TypeScript with openai package

```typescript
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://localhost:3512/v1",
  apiKey: "not-needed",
  dangerouslyAllowBrowser: true // only for browser use
});

const stream = await client.chat.completions.create({
  model: "sharpinfer",
  messages: [{ role: "user", content: "Hello from TypeScript!" }],
  stream: true
});

for await (const chunk of stream) {
  const token = chunk.choices[0]?.delta?.content ?? "";
  process.stdout.write(token);
}
```

### Model Management (pull + load)

```javascript
// Pull a model with progress tracking
const pullResponse = await fetch("http://localhost:3512/api/pull", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ name: "hf.co/bartowski/Phi-3-mini-4k-instruct-GGUF:Q4_K_M", stream: true })
});

const reader = pullResponse.body.getReader();
const decoder = new TextDecoder();
while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  for (const line of decoder.decode(value).split("\n").filter(Boolean)) {
    const progress = JSON.parse(line);
    if (progress.total > 0) {
      const pct = Math.round((progress.completed / progress.total) * 100);
      console.log(`Downloading: ${pct}%`);
    }
    if (progress.status === "success") console.log("Download complete!");
  }
}

// Load the model
const loadRes = await fetch("http://localhost:3512/api/load", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ name: "Phi-3-mini-4k-instruct" })
});
const loadData = await loadRes.json();
console.log(loadData.status); // "success"
```

---

## curl

### Basic Request

```bash
curl http://localhost:3512/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sharpinfer",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 128
  }'
```

### Streaming

```bash
curl http://localhost:3512/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"sharpinfer","messages":[{"role":"user","content":"Count to 5 slowly."}],"stream":true}'
```

### Pull and Load a Model

```bash
# Pull
curl -X POST http://localhost:3512/api/pull \
  -H "Content-Type: application/json" \
  -d '{"name": "hf.co/bartowski/Phi-3-mini-4k-instruct-GGUF:Q4_K_M"}'

# Load
curl -X POST http://localhost:3512/api/load \
  -H "Content-Type: application/json" \
  -d '{"name": "Phi-3-mini-4k-instruct"}'

# Check health
curl http://localhost:3512/health
```
