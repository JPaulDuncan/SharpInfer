---
layout: default
title: Chat
parent: VS Code Extension
nav_order: 3
---

# Chat
{: .no_toc }

Multi-turn coding assistant in a VS Code panel.
{: .fs-6 .fw-300 }

---

## Overview

The chat feature lets you have a conversation with the model about your code — without leaving VS Code. Ask it to explain unfamiliar code, suggest refactors, write tests, or debug issues.

---

## The `chat` Request

```json
{
  "jsonrpc": "2.0",
  "id": 4,
  "method": "chat",
  "params": {
    "messages": [
      {"role": "system", "content": "You are an expert C# and .NET developer."},
      {"role": "user", "content": "How do I implement IDisposable correctly?"}
    ],
    "maxTokens": 512,
    "stream": true
  }
}
```

With `"stream": true`, tokens arrive as notifications:

```json
{"jsonrpc":"2.0","method":"token","params":{"text":"Implement"}}
{"jsonrpc":"2.0","method":"token","params":{"text":" IDisposable"}}
...
{"jsonrpc":"2.0","id":4,"result":{"text":"...full text...","done":true}}
```

---

## Multi-Turn Conversations

Include all previous messages in the `messages` array to maintain context:

```json
{
  "method": "chat",
  "params": {
    "messages": [
      {"role": "system", "content": "You are a helpful C# developer."},
      {"role": "user", "content": "What is the difference between IEnumerable and IQueryable?"},
      {"role": "assistant", "content": "IEnumerable executes queries in memory..."},
      {"role": "user", "content": "When would I prefer one over the other?"}
    ],
    "maxTokens": 512,
    "stream": true
  }
}
```

The server is stateless — it doesn't remember previous turns. Your extension must maintain the full conversation history and send it with each request.

---

## Common Use Cases

**Explaining code** — Select a code block, pass it in the user message:
```
"Please explain what this method does:\n\n```csharp\n{selected code}\n```"
```

**Debugging** — Share the error and relevant code:
```
"I'm getting this exception: {error}. Here's the relevant code: {code}. What could be wrong?"
```

**Writing tests** — Ask the model to generate unit tests:
```
"Write xUnit tests for this C# method: {code}"
```

**Refactoring** — Ask for improvement suggestions:
```
"Can you refactor this to be more idiomatic .NET? {code}"
```

**Documentation** — Generate XML doc comments:
```
"Write XML documentation comments for this method: {code}"
```

---

## Implementing a Chat Panel

```typescript
import * as vscode from "vscode";

class ChatPanel {
  private panel: vscode.WebviewPanel;
  private history: Array<{role: string, content: string}> = [];

  constructor(context: vscode.ExtensionContext) {
    this.panel = vscode.window.createWebviewPanel(
      "sharpinferChat",
      "SharpInfer Chat",
      vscode.ViewColumn.Beside,
      { enableScripts: true }
    );
    this.history.push({
      role: "system",
      content: "You are an expert software developer. Help the user with their code."
    });
  }

  async sendMessage(userMessage: string): Promise<void> {
    this.history.push({ role: "user", content: userMessage });

    let fullReply = "";

    // Stream tokens back as they arrive
    await backend.chat({
      messages: this.history,
      maxTokens: 512,
      stream: true,
      onToken: (token: string) => {
        fullReply += token;
        this.panel.webview.postMessage({ type: "token", text: token });
      }
    });

    this.history.push({ role: "assistant", content: fullReply });
  }
}
```
