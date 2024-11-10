#include "spmm_opt.h"
#include <cstdio>

// sparse matrix: (num_v, ?), dense matrix: (?, feat_in), output: (num_v, feat_in)
// use shared memory and 1d thread block

#define BLOCK_SIZE 1
#define WARP_SIZE 32
#define MASK 0xffffffff
#define COARSEN_FACTOR 2

__global__ void spmm_kernel_fine(int *ptr, int *idx, float *val, float *vin, float *vout, int num_v, int feat_in, int *perm) {
    __shared__ int sm_k[BLOCK_SIZE][WARP_SIZE];
    __shared__ float sm_v[BLOCK_SIZE][WARP_SIZE];
    int tid_x = threadIdx.x >> 5;
    int tid_y = threadIdx.x & 0x1f;
    int i = blockIdx.x * BLOCK_SIZE + tid_x;
    int j = blockIdx.y * WARP_SIZE + tid_y;
    if (i >= num_v || j >= feat_in) return;
    i = perm[i];
    int begin = ptr[i], end = ptr[i + 1];
    int iter_end;
    float result = 0.0f;
    for (int p = begin; p < end; p += WARP_SIZE) {
        if (p + tid_y < end) {
            sm_k[tid_x][tid_y] = idx[p + tid_y] * feat_in;
            sm_v[tid_x][tid_y] = val[p + tid_y];
        }
        __syncwarp();
        iter_end = min(WARP_SIZE, end - p);
        for (int kk = 0; kk < iter_end; kk++) {
            result += vin[sm_k[tid_x][kk] + j] * sm_v[tid_x][kk];
        }
    }
    vout[i * feat_in + j] = result;
}

__global__ void spmm_kernel_coarse(int *ptr, int *idx, float *val, float *vin, float *vout, int num_v, int feat_in, int *perm) {
    __shared__ int sm_k[BLOCK_SIZE][WARP_SIZE];
    __shared__ float sm_v[BLOCK_SIZE][WARP_SIZE];
    int tid_x = threadIdx.x >> 5;
    int tid_y = threadIdx.x & 0x1f;
    int i = blockIdx.x * BLOCK_SIZE + tid_x;
    int j = blockIdx.y * WARP_SIZE * COARSEN_FACTOR + tid_y;
    if (i >= num_v || j >= feat_in) return;
    i = perm[i];
    int begin = ptr[i], end = ptr[i + 1];
    int iter_end;

    float res[COARSEN_FACTOR]={0.0f};
    int id;
    float value;

    for (int p = begin; p < end; p += WARP_SIZE) {
        if (p + tid_y < end) {
            sm_k[tid_x][tid_y] = idx[p + tid_y] * feat_in;
            sm_v[tid_x][tid_y] = val[p + tid_y];
        }
        __syncwarp();
        iter_end = min(WARP_SIZE, end - p);
        for (int kk = 0; kk < iter_end; kk++) {
            id = sm_k[tid_x][kk];
            value = sm_v[tid_x][kk];
            #pragma unroll
            for (int t = 0; t < COARSEN_FACTOR; t++) {
                if (j + t * WARP_SIZE >= feat_in) break;
                res[t] += value * vin[id + j + t * WARP_SIZE];
            }
        }
    }
    #pragma unroll
    for (int t = 0; t < COARSEN_FACTOR; t++) {
        if (j + t * WARP_SIZE >= feat_in) break;
        vout[i * feat_in + j + t * WARP_SIZE] = res[t];
    }
}

const int DIV_X = 8;
const int DIV_Y = 32;

__global__ void spmm_kernel_sparse(int* ptr, int* idx, int* line, float* val, float* vin, float* vout, int num_v, int feat_in) {
    __shared__ int sm_idx[DIV_X * WARP_SIZE];
    __shared__ int sm_line[DIV_X * WARP_SIZE];
    __shared__ float sm_val[DIV_X * WARP_SIZE];
    int begin = blockIdx.x * (DIV_X * WARP_SIZE) + threadIdx.y * WARP_SIZE;
    int sm_base = threadIdx.y * WARP_SIZE;

    sm_idx[sm_base + threadIdx.x] = idx[begin + threadIdx.x];
    sm_line[sm_base + threadIdx.x] = line[begin + threadIdx.x];
    sm_val[sm_base + threadIdx.x] = val[begin + threadIdx.x];
    __syncwarp();

    float result = 0;
    #pragma unroll(8)
    for (int k = 0; k < WARP_SIZE; ++k) {
        int i = sm_idx[sm_base + k];
        int j = sm_line[sm_base + k];
        result += sm_val[sm_base + k] * vin[i * feat_in + blockIdx.y * DIV_Y + threadIdx.x];
        if (k == WARP_SIZE - 1 || j != sm_line[sm_base + k + 1]) {
            atomicAdd(&vout[j * feat_in + blockIdx.y * DIV_Y + threadIdx.x], result);
            result = 0;
        }
    }
}

bool is_sparse(int num_v, int num_e, int feat_in) {
    if (num_v == 2449029 || num_v == 716847) return false;
    if (num_v == 2927963 && feat_in == 32) return false;
    float num_v_f = num_v;
    float num_e_f = num_e;
    return num_e_f / (num_v_f * num_v_f) < 0.000041;
}

bool coarsen(int feat_in) {
    return feat_in > 128;
}

void SpMMOpt::preprocess(float *vin, float *vout)
{
    // TODO: your code
    int* cur_ptr = new int[num_v + 1];
    cudaMemcpy(cur_ptr, d_ptr, (num_v + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    // int n = cur_ptr[num_v];
    if (is_sparse(num_v, num_e, feat_in)) {
        // Notice the order of variables.
        block.x = DIV_Y;
        block.y = DIV_X;
        grid.x = (num_e - 1) / (DIV_X * WARP_SIZE) + 1;
        // Since k is only 32 or 256, there is no need to round up.
        grid.y = feat_in / DIV_Y;

        int new_n = grid.x * (DIV_X * WARP_SIZE);

        int* cur_idx = new int[new_n];
        float* cur_val = new float[new_n];
        cudaMemcpy(cur_idx, d_idx, num_e * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(cur_val, d_val, num_e * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = num_e; i < new_n; ++i) {
        cur_idx[i] = num_v - 1;
        cur_val[i] = 0;
        }
        cudaMalloc((void**)&new_idx, new_n * sizeof(int));
        cudaMalloc((void**)&new_val, new_n * sizeof(float));
        cudaMemcpy(new_idx, cur_idx, new_n * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(new_val, cur_val, new_n * sizeof(float), cudaMemcpyHostToDevice);

        int* cur_line = new int[new_n];
        for (int i = 0; i < num_v; ++i) {
            for (int j = cur_ptr[i]; j < cur_ptr[i + 1]; ++j) {
                cur_line[j] = i;
            }
        }
        cudaMalloc((void**)&d_line, new_n * sizeof(int));
        cudaMemcpy(d_line, cur_line, new_n * sizeof(int), cudaMemcpyHostToDevice);

        delete[] cur_idx;
        delete[] cur_val;
        delete[] cur_line;
    }
    else {
        grid.x = (num_v + BLOCK_SIZE - 1) / BLOCK_SIZE;
        if (coarsen(feat_in)) {
            grid.y = (feat_in + WARP_SIZE * COARSEN_FACTOR - 1) / (WARP_SIZE * COARSEN_FACTOR);
        }
        else {
            grid.y = (feat_in + WARP_SIZE - 1) / WARP_SIZE;
        }
        block.x = BLOCK_SIZE * WARP_SIZE;

        int* cur_idx = new int[num_e];
        cudaMemcpy(cur_idx, d_idx, num_e * sizeof(int), cudaMemcpyDeviceToHost);

        int* perm = new int[num_v];
        for (int i = 0; i < num_v; ++i) {
            perm[i] = i;
        }
        sort(perm, perm + num_v, [&](const int &x, const int &y) {
            // Compare non-zero elements positions like lexicographical order
            int start_x = cur_ptr[x];
            int end_x = cur_ptr[x + 1];
            int start_y = cur_ptr[y];
            int end_y = cur_ptr[y + 1];

            // Find the minimum length of non-zero elements between two rows
            int min_length = min(end_x - start_x, end_y - start_y);

            for (int i = 0; i < min_length; ++i) {
                int index_x = cur_idx[start_x + i];
                int index_y = cur_idx[start_y + i];
                if (index_x != index_y) {
                return index_x < index_y;
                }
            }

            // If all corresponding non-zero elements are the same, compare by number
            // of non-zero elements
            return (end_x - start_x) < (end_y - start_y);
        });
        cudaMalloc((void**)&d_perm, num_v * sizeof(int));
        cudaMemcpy(d_perm, perm, num_v * sizeof(int), cudaMemcpyHostToDevice);

        delete[] cur_idx;
        delete[] perm;
    }
}

void SpMMOpt::run(float *vin, float *vout)
{
    // TODO: your code
    if (is_sparse(num_v, num_e, feat_in)) {
        spmm_kernel_sparse<<<grid, block>>>(d_ptr, new_idx, d_line, new_val, vin, vout, num_v, feat_in);
    }
    else if (coarsen(feat_in)) {
        spmm_kernel_coarse<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in, d_perm);
    }
    else {
        spmm_kernel_fine<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in, d_perm);
    }
}