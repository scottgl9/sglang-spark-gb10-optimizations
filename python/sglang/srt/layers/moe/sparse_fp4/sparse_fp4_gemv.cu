/*
 * Sparse FP4 GEMV — exploits 2:4 structured sparsity in NVFP4 MoE weights.
 *
 * For bandwidth-limited decode (BS=1 per expert), reads only the 2 non-zero
 * values per group-of-4, cutting DRAM traffic ~25% vs dense FP4.
 *
 * Weight dequantization:
 *   value = fp4_lut[nibble] * fp8_to_float(block_scale) * global_scale
 *
 * Adapted from dgx-vllm sparse_fp4_v7.cu for SGLang integration.
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

// FP4 E2M1 LUT (16 entries)
__constant__ float c_fp4_lut[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// FP8 E4M3FN -> float32 LUT (256 entries, initialized at module load)
__constant__ float c_fp8_lut[256];

static float fp8_e4m3fn_to_float_host(uint8_t x) {
    uint32_t sign = (x >> 7) & 1;
    uint32_t exp = (x >> 3) & 0xF;
    uint32_t mant = x & 0x7;

    if (exp == 0 && mant == 0) return sign ? -0.0f : 0.0f;

    float val;
    if (exp == 0) {
        val = ldexpf((float)mant / 8.0f, -6);
    } else {
        val = ldexpf(1.0f + (float)mant / 8.0f, (int)exp - 7);
    }
    return sign ? -val : val;
}

static bool fp8_lut_initialized = false;
static void init_fp8_lut() {
    if (fp8_lut_initialized) return;
    float lut[256];
    for (int i = 0; i < 256; i++)
        lut[i] = fp8_e4m3fn_to_float_host((uint8_t)i);
    cudaMemcpyToSymbol(c_fp8_lut, lut, 256 * sizeof(float));
    fp8_lut_initialized = true;
}


// =====================================================================
// Batched Sparse FP4 GEMV with FP8 block scales
// =====================================================================
template <int THREADS, int THREAD_N, int TILE_K>
__global__ void batched_sparse_gemv_kernel(
    const half* __restrict__ A,               // [E_active, K]
    const uint8_t* __restrict__ B_comp_T,     // [E_total, K/4, N]
    const uint8_t* __restrict__ Meta_T_pk,    // [E_total, K/8, N]
    const uint8_t* __restrict__ scales_T,     // [E_total, n_scale_groups, N] FP8 E4M3FN
    const float* __restrict__ g_scales,       // [E_total] global scales
    half* __restrict__ C,                     // [E_active, K_blocks, N] partials
    const int* __restrict__ expert_ids,
    int N, int K, int n_scale_groups, int E_active
) {
    constexpr int N_PER_BLOCK = THREADS * THREAD_N;

    const int expert_active = blockIdx.z;
    if (expert_active >= E_active) return;
    const int expert_total = expert_ids[expert_active];
    const int tid = threadIdx.x;
    const int n_base = blockIdx.x * N_PER_BLOCK + tid * THREAD_N;
    const bool valid_n = (n_base + THREAD_N <= N);

    const int k_start = blockIdx.y * TILE_K;
    const int k_end = min(k_start + TILE_K, K);

    const long b_comp_off = (long)expert_total * (K / 4) * N;
    const long meta_off = (long)expert_total * (K / 8) * N;
    const long scale_off = (long)expert_total * (long)n_scale_groups * N;
    const half* A_row = A + expert_active * K;
    const float g_scale = g_scales[expert_total];
    const int pairs_per_scale = (K / 8) / n_scale_groups;

    extern __shared__ float sh_data[];
    float* sh_lut = sh_data;
    float* sh_A = sh_data + 16;

    if (tid < 16) sh_lut[tid] = c_fp4_lut[tid];
    for (int i = tid; i < k_end - k_start; i += THREADS)
        sh_A[i] = __half2float(A_row[k_start + i]);
    __syncthreads();

    if (!valid_n) return;

    float acc[THREAD_N] = {};
    const uint8_t* B_ptr = B_comp_T + b_comp_off + n_base;
    const uint8_t* M_ptr = Meta_T_pk + meta_off + n_base;
    const uint8_t* S_ptr = scales_T + scale_off + n_base;

    int g_start = k_start / 4;
    int g_end = k_end / 4;
    int m_start = k_start / 8;

    for (int gi = g_start; gi < g_end; gi += 2) {
        int k_local = (gi - g_start) * 4;
        int mi = m_start + (gi - g_start) / 2;
        int si = (k_start / 8 + (gi - g_start) / 2) / pairs_per_scale;

        uint32_t comp0_4 = *reinterpret_cast<const uint32_t*>(&B_ptr[gi * N]);
        uint32_t comp1_4 = *reinterpret_cast<const uint32_t*>(&B_ptr[(gi + 1) * N]);
        uint32_t meta_4  = *reinterpret_cast<const uint32_t*>(&M_ptr[mi * N]);
        uint32_t scale_4 = *reinterpret_cast<const uint32_t*>(&S_ptr[si * N]);

        #pragma unroll
        for (int j = 0; j < THREAD_N; j++) {
            uint8_t comp0 = (comp0_4 >> (j * 8)) & 0xFF;
            uint8_t comp1 = (comp1_4 >> (j * 8)) & 0xFF;
            uint8_t meta  = (meta_4  >> (j * 8)) & 0xFF;
            float scale = c_fp8_lut[(scale_4 >> (j * 8)) & 0xFF] * g_scale;

            float sum = sh_lut[comp0 & 0x0F] * sh_A[k_local + (meta & 3)]
                      + sh_lut[(comp0 >> 4) & 0x0F] * sh_A[k_local + ((meta >> 2) & 3)]
                      + sh_lut[comp1 & 0x0F] * sh_A[k_local + 4 + ((meta >> 4) & 3)]
                      + sh_lut[(comp1 >> 4) & 0x0F] * sh_A[k_local + 4 + ((meta >> 6) & 3)];
            acc[j] += sum * scale;
        }
    }

    long out_base = (long)expert_active * ((K + TILE_K - 1) / TILE_K) * N;
    int out_offset = blockIdx.y * N + n_base;
    #pragma unroll
    for (int j = 0; j < THREAD_N; j++)
        C[out_base + out_offset + j] = __float2half(acc[j]);
}


// =====================================================================
// SiLU + multiply kernel: out = silu(gate) * up
// gate = input[:, :N], up = input[:, N:]  (input is [M, 2*N])
// =====================================================================
__global__ void silu_mul_kernel(
    const half* __restrict__ input,  // [M, 2*N]
    half* __restrict__ output,       // [M, N]
    int M, int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    int m = idx / N;
    int n = idx % N;
    float gate = __half2float(input[m * 2 * N + n]);
    float up = __half2float(input[m * 2 * N + N + n]);
    float sig = 1.0f / (1.0f + expf(-gate));
    output[idx] = __float2half(gate * sig * up);
}

// =====================================================================
// BF16 -> FP16 conversion + token replication kernel
// Input: [M, K] bfloat16, Output: [M*topk, K] float16
// =====================================================================
__global__ void bf16_to_fp16_replicate_kernel(
    const __nv_bfloat16* __restrict__ input,  // [M, K]
    half* __restrict__ output,                // [M*topk, K]
    int M, int K, int topk
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * topk * K;
    if (idx >= total) return;
    int mk = idx / K;
    int k = idx % K;
    int m = mk / topk;
    float val = __bfloat162float(input[m * K + k]);
    output[idx] = __float2half(val);
}

// =====================================================================
// Weighted reduction kernel: output[m] = sum_t(down[m*topk+t] * weight[m][t])
// =====================================================================
__global__ void weighted_reduce_fp16_kernel(
    const half* __restrict__ down,       // [M*topk, K] float16
    const float* __restrict__ weights,   // [M, topk]
    half* __restrict__ output,           // [M, K]
    int M, int K, int topk,
    bool apply_router_weight_on_input
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * K) return;
    int m = idx / K;
    int k = idx % K;
    float acc = 0.0f;
    for (int t = 0; t < topk; t++) {
        float val = __half2float(down[(m * topk + t) * K + k]);
        if (apply_router_weight_on_input) {
            acc += val;
        } else {
            acc += val * weights[m * topk + t];
        }
    }
    output[idx] = __float2half(acc);
}


// =====================================================================
// Fused sparse MoE forward
// =====================================================================
constexpr int T = 128;
constexpr int TN = 4;
constexpr int TK = 64;
constexpr int NPB = T * TN;  // 512

static int smem_size(int tile_k) {
    return (16 + tile_k) * sizeof(float);
}

torch::Tensor fused_sparse_moe(
    torch::Tensor hidden_states,       // [M, K] bfloat16 or float16
    torch::Tensor topk_weights,        // [M, topk] float32
    torch::Tensor topk_ids,            // [M, topk] int32
    torch::Tensor expert_map,          // [global_E] int32 or empty
    // W13 (gate+up) weights
    torch::Tensor w13_comp,            // [E_total, K/4, 2*N] uint8
    torch::Tensor w13_meta,            // [E_total, K/8, 2*N] uint8
    torch::Tensor w13_scale,           // [E_total, n_groups, 2*N] uint8
    torch::Tensor w13_g_scales,        // [E_total] float32
    // W2 (down) weights
    torch::Tensor w2_comp,             // [E_total, N/4, K] uint8
    torch::Tensor w2_meta,             // [E_total, N/8, K] uint8
    torch::Tensor w2_scale,            // [E_total, n_groups2, K] uint8
    torch::Tensor w2_g_scales,         // [E_total] float32
    int inter_size,                    // N (moe_intermediate_size)
    bool apply_router_weight_on_input
) {
    init_fp8_lut();

    int M = hidden_states.size(0);
    int K = hidden_states.size(1);
    int topk = topk_ids.size(1);
    int N = inter_size;
    int E_active = M * topk;

    auto stream = at::cuda::getCurrentCUDAStream();
    c10::cuda::CUDAStreamGuard stream_guard(stream);
    auto opts_fp16 = torch::TensorOptions().dtype(torch::kFloat16).device(hidden_states.device());

    // 1. Map expert IDs
    torch::Tensor mapped_ids;
    if (expert_map.numel() > 0) {
        mapped_ids = expert_map.index({topk_ids}).reshape(-1).to(torch::kInt32);
    } else {
        mapped_ids = topk_ids.reshape(-1).to(torch::kInt32);
    }

    // 2. BF16->FP16 + token replication
    auto A = torch::empty({E_active, K}, opts_fp16);
    if (hidden_states.scalar_type() == torch::kBFloat16) {
        int total = E_active * K;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        bf16_to_fp16_replicate_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(hidden_states.data_ptr()),
            reinterpret_cast<half*>(A.data_ptr<at::Half>()),
            M, K, topk);
    } else {
        A = hidden_states.to(torch::kFloat16).repeat_interleave(topk, 0);
    }

    // 3. GEMM1: gate_up projection [E_active, 2*N]
    int N2 = 2 * N;
    int n_scale_groups_13 = w13_scale.size(1);
    int n_blocks_13 = (N2 + NPB - 1) / NPB;
    int k_blocks_13 = (K + TK - 1) / TK;
    dim3 grid_13(n_blocks_13, k_blocks_13, E_active);

    auto C13 = torch::empty({E_active, k_blocks_13, N2}, opts_fp16);
    batched_sparse_gemv_kernel<T, TN, TK>
        <<<grid_13, T, smem_size(TK), stream>>>(
            reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
            w13_comp.data_ptr<uint8_t>(),
            w13_meta.data_ptr<uint8_t>(),
            w13_scale.data_ptr<uint8_t>(),
            w13_g_scales.data_ptr<float>(),
            reinterpret_cast<half*>(C13.data_ptr<at::Half>()),
            mapped_ids.data_ptr<int>(),
            N2, K, n_scale_groups_13, E_active);

    torch::Tensor gate_up = k_blocks_13 == 1 ? C13.squeeze(1) : C13.sum(1);

    // 4. SiLU activation
    auto intermediate = torch::empty({E_active, N}, opts_fp16);
    {
        int total = E_active * N;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        silu_mul_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const half*>(gate_up.data_ptr<at::Half>()),
            reinterpret_cast<half*>(intermediate.data_ptr<at::Half>()),
            E_active, N);
    }

    // 5. GEMM2: down projection [E_active, K]
    int K2 = N;
    int N2_down = K;
    int n_scale_groups_2 = w2_scale.size(1);
    int n_blocks_2 = (N2_down + NPB - 1) / NPB;
    int k_blocks_2 = (K2 + TK - 1) / TK;
    dim3 grid_2(n_blocks_2, k_blocks_2, E_active);

    auto C2 = torch::empty({E_active, k_blocks_2, N2_down}, opts_fp16);
    batched_sparse_gemv_kernel<T, TN, TK>
        <<<grid_2, T, smem_size(TK), stream>>>(
            reinterpret_cast<const half*>(intermediate.data_ptr<at::Half>()),
            w2_comp.data_ptr<uint8_t>(),
            w2_meta.data_ptr<uint8_t>(),
            w2_scale.data_ptr<uint8_t>(),
            w2_g_scales.data_ptr<float>(),
            reinterpret_cast<half*>(C2.data_ptr<at::Half>()),
            mapped_ids.data_ptr<int>(),
            N2_down, K2, n_scale_groups_2, E_active);

    torch::Tensor down = k_blocks_2 == 1 ? C2.squeeze(1) : C2.sum(1);

    // 6. Weighted reduction [M, K]
    auto output = torch::empty({M, K}, opts_fp16);
    {
        int total = M * K;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        weighted_reduce_fp16_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const half*>(down.data_ptr<at::Half>()),
            topk_weights.data_ptr<float>(),
            reinterpret_cast<half*>(output.data_ptr<at::Half>()),
            M, K, topk, apply_router_weight_on_input);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_sparse_moe", &fused_sparse_moe,
          "Fused sparse FP4 MoE forward (GEMV with 2:4 sparsity)");
}
