/**
 * SharpInfer CUDA Kernels
 *
 * Build: nvcc -shared -o sharpinfer_cuda.so kernels.cu -O3 -arch=sm_70
 * (Windows: nvcc -shared -o sharpinfer_cuda.dll kernels.cu -O3 -arch=sm_70)
 *
 * These kernels are called from C# via P/Invoke.
 * Each function manages its own GPU memory transfers.
 *
 * For production use, consider:
 * - Persistent GPU memory allocation (avoid repeated cudaMalloc/cudaFree)
 * - Using cuBLAS for matrix operations
 * - Half-precision (FP16) computation
 * - Flash Attention kernels
 */

#include <cuda_runtime.h>
#include <cmath>
#include <cstring>

struct CudaContext {
    int device_id;
    cudaStream_t stream;
};

extern "C" {

// --- Initialization ---

void* cuda_init(int device_id) {
    if (cudaSetDevice(device_id) != cudaSuccess) return nullptr;

    auto* ctx = new CudaContext();
    ctx->device_id = device_id;
    cudaStreamCreate(&ctx->stream);
    return ctx;
}

void cuda_cleanup(void* handle) {
    auto* ctx = (CudaContext*)handle;
    cudaStreamDestroy(ctx->stream);
    delete ctx;
}

void cuda_device_name(void* handle, char* buffer, int buf_len) {
    auto* ctx = (CudaContext*)handle;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, ctx->device_id);
    strncpy(buffer, prop.name, buf_len - 1);
    buffer[buf_len - 1] = '\0';
}

// --- Kernels ---

__global__ void mat_vec_mul_kernel(const float* mat, const float* vec,
                                    float* result, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    float sum = 0.0f;
    for (int j = 0; j < cols; j++) {
        sum += mat[row * cols + j] * vec[j];
    }
    result[row] = sum;
}

void cuda_mat_vec_mul(void* handle, const float* h_mat, const float* h_vec,
                      float* h_result, int rows, int cols) {
    auto* ctx = (CudaContext*)handle;

    float *d_mat, *d_vec, *d_result;
    cudaMalloc(&d_mat, rows * cols * sizeof(float));
    cudaMalloc(&d_vec, cols * sizeof(float));
    cudaMalloc(&d_result, rows * sizeof(float));

    cudaMemcpyAsync(d_mat, h_mat, rows * cols * sizeof(float), cudaMemcpyHostToDevice, ctx->stream);
    cudaMemcpyAsync(d_vec, h_vec, cols * sizeof(float), cudaMemcpyHostToDevice, ctx->stream);

    int threads = 256;
    int blocks = (rows + threads - 1) / threads;
    mat_vec_mul_kernel<<<blocks, threads, 0, ctx->stream>>>(d_mat, d_vec, d_result, rows, cols);

    cudaMemcpyAsync(h_result, d_result, rows * sizeof(float), cudaMemcpyDeviceToHost, ctx->stream);
    cudaStreamSynchronize(ctx->stream);

    cudaFree(d_mat); cudaFree(d_vec); cudaFree(d_result);
}

__global__ void mat_mul_kernel(const float* a, const float* b, float* c,
                                int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += a[row * K + k] * b[k * N + col];
    }
    c[row * N + col] = sum;
}

void cuda_mat_mul(void* handle, const float* h_a, const float* h_b,
                  float* h_c, int M, int K, int N) {
    auto* ctx = (CudaContext*)handle;

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, M * K * sizeof(float));
    cudaMalloc(&d_b, K * N * sizeof(float));
    cudaMalloc(&d_c, M * N * sizeof(float));

    cudaMemcpyAsync(d_a, h_a, M * K * sizeof(float), cudaMemcpyHostToDevice, ctx->stream);
    cudaMemcpyAsync(d_b, h_b, K * N * sizeof(float), cudaMemcpyHostToDevice, ctx->stream);

    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    mat_mul_kernel<<<blocks, threads, 0, ctx->stream>>>(d_a, d_b, d_c, M, K, N);

    cudaMemcpyAsync(h_c, d_c, M * N * sizeof(float), cudaMemcpyDeviceToHost, ctx->stream);
    cudaStreamSynchronize(ctx->stream);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
}

__global__ void softmax_kernel(float* x, int len) {
    // Single-block implementation for moderate vocabulary sizes
    extern __shared__ float shared[];

    int tid = threadIdx.x;

    // Find max (reduction)
    float local_max = -INFINITY;
    for (int i = tid; i < len; i += blockDim.x) {
        local_max = fmaxf(local_max, x[i]);
    }
    shared[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] = fmaxf(shared[tid], shared[tid + s]);
        __syncthreads();
    }
    float max_val = shared[0];

    // Compute exp and sum
    float local_sum = 0.0f;
    for (int i = tid; i < len; i += blockDim.x) {
        x[i] = expf(x[i] - max_val);
        local_sum += x[i];
    }
    shared[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }
    float sum = shared[0];

    // Normalize
    for (int i = tid; i < len; i += blockDim.x) {
        x[i] /= sum;
    }
}

void cuda_softmax(void* handle, float* h_x, int len) {
    auto* ctx = (CudaContext*)handle;

    float* d_x;
    cudaMalloc(&d_x, len * sizeof(float));
    cudaMemcpyAsync(d_x, h_x, len * sizeof(float), cudaMemcpyHostToDevice, ctx->stream);

    int threads = 256;
    softmax_kernel<<<1, threads, threads * sizeof(float), ctx->stream>>>(d_x, len);

    cudaMemcpyAsync(h_x, d_x, len * sizeof(float), cudaMemcpyDeviceToHost, ctx->stream);
    cudaStreamSynchronize(ctx->stream);
    cudaFree(d_x);
}

__global__ void rms_norm_kernel(float* output, const float* x, const float* weight,
                                 int size, float eps) {
    int tid = threadIdx.x;
    extern __shared__ float shared[];

    // Compute sum of squares
    float local_ss = 0.0f;
    for (int i = tid; i < size; i += blockDim.x) {
        local_ss += x[i] * x[i];
    }
    shared[tid] = local_ss;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }

    float scale = rsqrtf(shared[0] / size + eps);

    for (int i = tid; i < size; i += blockDim.x) {
        output[i] = x[i] * scale * weight[i];
    }
}

void cuda_rms_norm(void* handle, float* h_output, const float* h_x,
                   const float* h_weight, int size, float eps) {
    auto* ctx = (CudaContext*)handle;

    float *d_output, *d_x, *d_weight;
    cudaMalloc(&d_output, size * sizeof(float));
    cudaMalloc(&d_x, size * sizeof(float));
    cudaMalloc(&d_weight, size * sizeof(float));

    cudaMemcpyAsync(d_x, h_x, size * sizeof(float), cudaMemcpyHostToDevice, ctx->stream);
    cudaMemcpyAsync(d_weight, h_weight, size * sizeof(float), cudaMemcpyHostToDevice, ctx->stream);

    int threads = 256;
    rms_norm_kernel<<<1, threads, threads * sizeof(float), ctx->stream>>>(
        d_output, d_x, d_weight, size, eps);

    cudaMemcpyAsync(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost, ctx->stream);
    cudaStreamSynchronize(ctx->stream);

    cudaFree(d_output); cudaFree(d_x); cudaFree(d_weight);
}

__global__ void silu_kernel(float* x, int len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) {
        x[i] = x[i] / (1.0f + expf(-x[i]));
    }
}

void cuda_silu(void* handle, float* h_x, int len) {
    auto* ctx = (CudaContext*)handle;

    float* d_x;
    cudaMalloc(&d_x, len * sizeof(float));
    cudaMemcpyAsync(d_x, h_x, len * sizeof(float), cudaMemcpyHostToDevice, ctx->stream);

    int threads = 256;
    int blocks = (len + threads - 1) / threads;
    silu_kernel<<<blocks, threads, 0, ctx->stream>>>(d_x, len);

    cudaMemcpyAsync(h_x, d_x, len * sizeof(float), cudaMemcpyDeviceToHost, ctx->stream);
    cudaStreamSynchronize(ctx->stream);
    cudaFree(d_x);
}

__global__ void add_inplace_kernel(float* a, const float* b, int len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) a[i] += b[i];
}

void cuda_add_inplace(void* handle, float* h_a, const float* h_b, int len) {
    auto* ctx = (CudaContext*)handle;

    float *d_a, *d_b;
    cudaMalloc(&d_a, len * sizeof(float));
    cudaMalloc(&d_b, len * sizeof(float));

    cudaMemcpyAsync(d_a, h_a, len * sizeof(float), cudaMemcpyHostToDevice, ctx->stream);
    cudaMemcpyAsync(d_b, h_b, len * sizeof(float), cudaMemcpyHostToDevice, ctx->stream);

    int threads = 256;
    int blocks = (len + threads - 1) / threads;
    add_inplace_kernel<<<blocks, threads, 0, ctx->stream>>>(d_a, d_b, len);

    cudaMemcpyAsync(h_a, d_a, len * sizeof(float), cudaMemcpyDeviceToHost, ctx->stream);
    cudaStreamSynchronize(ctx->stream);

    cudaFree(d_a); cudaFree(d_b);
}

} // extern "C"
