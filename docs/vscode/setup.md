---
layout: default
title: Setup
parent: VS Code Extension
nav_order: 1
---

# Setup
{: .no_toc }

Build the backend and connect it to VS Code.
{: .fs-6 .fw-300 }

<details open markdown="block">
  <summary>Table of contents</summary>
  {: .text-delta }
- TOC
{:toc}
</details>

---

## Step 1: Build the Backend

```bash
dotnet build src/SharpInfer.VsCode/SharpInfer.VsCode.csproj
```

This compiles the `SharpInfer.VsCode` process that VS Code will spawn.

---

## Step 2: Download a Code Model

Code-specific models produce significantly better completions than general-purpose models:

| Model | Size (Q4_K_M) | Notes |
|---|---|---|
| **CodeLlama 7B** | ~4.1 GB | Excellent code completion, FIM support |
| **DeepSeek Coder 6.7B** | ~4.0 GB | Strong across many languages |
| **StarCoder2 7B** | ~4.2 GB | Good for Python, JavaScript, TypeScript |
| **Qwen2.5 Coder 7B** | ~4.6 GB | Excellent, modern codebase |
| **Phi-3 Mini** | ~2.2 GB | Fast, good for quick completions |

Download via the CLI or API:

```bash
dotnet run --project src/SharpInfer.Cli -- \
  --models-dir ./models
# Then: /pull hf.co/bartowski/CodeLlama-7B-instruct-GGUF:Q4_K_M
```

---

## Step 3: Test the Backend

Run the backend manually to verify it starts correctly:

```bash
dotnet run --project src/SharpInfer.VsCode
```

The process will wait silently for JSON-RPC messages on stdin. Send an initialize request to confirm it responds:

```bash
echo 'Content-Length: 82\r\n\r\n{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"modelPath":"./models/codellama-7b.gguf"}}' | \
  dotnet run --project src/SharpInfer.VsCode
```

You should see a response with `"success": true`.

---

## Step 4: Configure the Extension

The TypeScript extension needs two paths:

1. **Path to the `SharpInfer.VsCode` executable** (or `dotnet run` command)
2. **Path to your model file**

Configure these in the extension's settings or companion `extension.ts`:

```typescript
// In your VS Code extension settings contribution
const backendPath = vscode.workspace.getConfiguration("sharpinfer").get("backendPath");
const modelPath = vscode.workspace.getConfiguration("sharpinfer").get("modelPath");
```

---

## Step 5: Spawn the Backend from the Extension

```typescript
import * as cp from "child_process";

const backend = cp.spawn("dotnet", [
  "run",
  "--project",
  "/path/to/src/SharpInfer.VsCode",
  "--",
  "--model", "/path/to/codellama-7b.gguf"
]);

// Send initialize
sendMessage(backend, {
  jsonrpc: "2.0",
  id: 1,
  method: "initialize",
  params: {
    modelPath: "/path/to/codellama-7b.gguf",
    contextLength: 4096
  }
});
```

The backend is now ready to handle `complete`, `chat`, and `generate` requests.

---

## GPU Acceleration

Enable GPU for faster completions by passing backend flags when spawning the process. Configure in your extension settings and pass as parameters:

```typescript
const backend = cp.spawn("dotnet", [
  "run", "--project", "path/to/SharpInfer.VsCode",
  "--", "--gpu", "--model", modelPath
]);
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| Backend won't start | Ensure .NET 8 is installed: `dotnet --version`. Try running manually to see error output. |
| "Engine not initialized" errors | Always send an `initialize` request before any `complete`, `chat`, or `generate` requests. |
| Very slow completions | Use a smaller model (1–3B). Enable GPU with `--gpu`. Reduce `maxTokens`. |
| No suggestions in editor | Check the VS Code Output panel for the extension. Verify the backend process is running. |
| High memory usage | Switch to a more quantized model (Q4_K_M instead of Q8_0) or a smaller model size. |
