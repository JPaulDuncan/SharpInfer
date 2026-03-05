---
layout: default
title: Code Completion
parent: VS Code Extension
nav_order: 2
---

# Code Completion
{: .no_toc }

Inline AI suggestions as you type.
{: .fs-6 .fw-300 }

---

## How It Works

When you pause while typing, the extension captures:
- **Prefix** — everything in the file before your cursor
- **Suffix** — everything after your cursor (optional)

It sends a `complete` request to the SharpInfer backend, which uses **Fill-in-the-Middle (FIM)** prompting to generate code that fits naturally between the two halves. The result appears as ghost text in the editor.

Press `Tab` to accept, or keep typing to dismiss.

---

## The `complete` Request

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "complete",
  "params": {
    "prefix": "public static int Fibonacci(int n)\n{\n    if (n <= 1) return n;\n    ",
    "suffix": "\n}",
    "maxTokens": 128,
    "language": "csharp"
  }
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "text": "return Fibonacci(n - 1) + Fibonacci(n - 2);"
  }
}
```

---

## Completion Settings

Code completion uses conservative defaults for speed and accuracy:

| Setting | Default | Notes |
|---|---|---|
| `temperature` | `0.2` | Low = predictable, focused code |
| `topP` | `0.95` | Slightly restrictive |
| `maxTokens` | `128` | Short — keeps suggestions fast |
| Stop strings | `\n\n`, ` ``` ` | Stops at block breaks |

---

## Fill-in-the-Middle (FIM)

When a suffix is provided, the backend constructs a FIM prompt:

```
<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>
```

This tells the model to generate code that connects the prefix to the suffix — enabling accurate mid-function completions and even cross-line fills.

{: .note }
FIM requires a model with built-in FIM support. **CodeLlama**, **DeepSeek Coder**, **StarCoder2**, and **Qwen2.5 Coder** all support FIM natively. General-purpose models (LLaMA, Mistral) will still complete from the prefix but won't use the suffix context.

---

## Registering a Completion Provider in VS Code

```typescript
import * as vscode from "vscode";

class SharpInferCompletionProvider implements vscode.InlineCompletionItemProvider {
  async provideInlineCompletionItems(
    document: vscode.TextDocument,
    position: vscode.Position,
    context: vscode.InlineCompletionContext,
    token: vscode.CancellationToken
  ): Promise<vscode.InlineCompletionList> {
    const prefix = document.getText(
      new vscode.Range(new vscode.Position(0, 0), position)
    );
    const suffix = document.getText(
      new vscode.Range(position, new vscode.Position(document.lineCount, 0))
    );

    const completion = await backend.complete({ prefix, suffix, maxTokens: 128 });

    return {
      items: [
        new vscode.InlineCompletionItem(
          completion.text,
          new vscode.Range(position, position)
        )
      ]
    };
  }
}

// Register in your activate() function
context.subscriptions.push(
  vscode.languages.registerInlineCompletionItemProvider(
    { pattern: "**" },
    new SharpInferCompletionProvider()
  )
);
```

---

## Performance Tips

- **Use a small model** — 1–3B parameter models respond in under a second on most hardware
- **Enable GPU** — Even a modest GPU cuts completion latency significantly
- **Reduce `maxTokens`** — For single-line completions, `64` tokens is sufficient and much faster
- **Debounce requests** — Wait 300–500ms after the user stops typing before sending a request to avoid unnecessary calls
