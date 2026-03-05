---
layout: default
title: GPU Acceleration
parent: CLI Guide
nav_order: 6
---

# GPU Acceleration
{: .no_toc }

Dramatically faster inference with CUDA, Metal, Vulkan, or ARM NEON.
{: .fs-6 .fw-300 }

<details open markdown="block">
  <summary>Table of contents</summary>
  {: .text-delta }
- TOC
{:toc}
</details>

---

## Auto-Detection

With `--backend auto` (the default when no backend flag is passed), SharpInfer probes your system at startup and selects the best available backend. The priority order is:

**CUDA → Metal → Vulkan → ARM NEON → CPU**

Use `/hardware` inside the CLI to see what was detected and which backend is active.

---

## NVIDIA CUDA

Add `--gpu` to enable CUDA acceleration:

```bash
dotnet run --project src/SharpInfer.Cli -- --model model.gguf --gpu
```

SharpInfer offloads matrix multiplication, softmax, normalization, and activations to the GPU. For 7B models this typically yields **5–15× speedup** over CPU.

### Partial Layer Offload

If your GPU doesn't have enough VRAM to hold the full model, offload only some layers:

```bash
# Offload 20 of 32 layers — rest runs on CPU
dotnet run --project src/SharpInfer.Cli -- --model model.gguf --gpu --gpu-layers 20
```

Token generation with partial offload is slower than full GPU but faster than full CPU.

### Multi-GPU

Select a specific GPU by device ID:

```bash
dotnet run --project src/SharpInfer.Cli -- --model model.gguf --gpu --gpu-device 1
```

### Building the CUDA Kernels

The CUDA backend requires compiling the native kernels once:

```bash
cd src/SharpInfer.Gpu/Kernels
nvcc -shared -o sharpinfer_cuda.so kernels.cu -O3
```

On Windows, compile to `sharpinfer_cuda.dll`. Place the output file alongside the application DLLs. Verify CUDA is installed with `nvidia-smi`.

---

## Apple Metal (macOS)

On Apple Silicon (M1, M2, M3, M4), use the Metal backend for GPU acceleration:

```bash
dotnet run --project src/SharpInfer.Cli -- --model model.gguf --backend metal
```

Or just use `--backend auto` — Metal is detected automatically on macOS with Apple Silicon.

**Expected speedup:** 4–8× over CPU for most models.

---

## Vulkan (Cross-Platform GPU)

Vulkan works on NVIDIA, AMD, Intel, and mobile GPUs. It's the best option for AMD GPUs on Linux/Windows:

```bash
dotnet run --project src/SharpInfer.Cli -- --model model.gguf --backend vulkan
```

Requirements: Vulkan 1.2+ drivers. On Linux, ensure `libvulkan` is installed.

---

## ARM NEON

On ARM64 processors (Apple M-series, Qualcomm Snapdragon, AWS Graviton), SharpInfer uses NEON SIMD intrinsics for accelerated CPU inference. This is detected automatically with `--backend auto`.

---

## FlashAttention

Regardless of the backend, `--flash-attention` enables block-wise attention computation that reduces memory bandwidth:

```bash
dotnet run --project src/SharpInfer.Cli -- --model model.gguf --gpu --flash-attention
```

FlashAttention is especially beneficial for long context lengths (4K+ tokens), reducing both memory usage and inference time.

---

## VRAM Requirements

Approximate GPU VRAM needed for common model sizes at Q4_K_M:

| Model Size | VRAM (Q4_K_M) | VRAM (F16) |
|---|---|---|
| 1B params | ~1 GB | ~2 GB |
| 3B params | ~2.5 GB | ~6 GB |
| 7B params | ~5 GB | ~14 GB |
| 13B params | ~8.5 GB | ~26 GB |
| 70B params | ~42 GB | — |

If VRAM is insufficient, use `--gpu-layers` for partial offload.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| "CUDA not available" | Install NVIDIA drivers and CUDA Toolkit. Check with `nvidia-smi`. |
| GPU detected but not used | Ensure `sharpinfer_cuda.so` / `.dll` is in the executable directory. |
| Out of VRAM | Use `--gpu-layers N` to offload fewer layers. Try a more quantized model (Q4_K_M instead of Q8_0). |
| Vulkan errors | Update GPU drivers. Ensure `vulkan-icd-loader` is installed on Linux. |
| Metal not detected | Ensure you're on Apple Silicon (M1/M2/M3/M4), not Intel. Check macOS is 13+. |
