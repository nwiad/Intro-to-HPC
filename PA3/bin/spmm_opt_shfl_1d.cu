#include "spmm_opt.h"
#include <cstdio>

// sparse matrix: (num_v, ?), dense matrix: (?, INFEATURE), output: (num_v, INFEATURE)
// use __shfl_sync and 1d thread block

#define BLOCK_SIZE 16
#define WARP_SIZE 32
#define MASK 0xffffffff

__global__ void spmm_kernel_opt(int *ptr, int *idx, float *val, float *vin, float *vout, int num_v, int feat_in) {
    int tid_x = threadIdx.x >> 5;
    int tid_y = threadIdx.x & 0x1f;
    int i = blockIdx.x * BLOCK_SIZE + tid_x;
    int j = blockIdx.y * WARP_SIZE + tid_y;
    if (i >= num_v || j >= feat_in) return;
    int begin = ptr[i], end = ptr[i + 1];
    float result = 0.0;
    int sm_k[WARP_SIZE], id;
    float sm_v[WARP_SIZE], value;
    for (int p = begin; p < end; p += WARP_SIZE) {
        if (p + tid_y < end) {
            id = idx[p + tid_y];
            value = val[p + tid_y];
        }
        else {
            id = 0;
            value = 0.0;
        }
        #pragma unroll
        for (int kk = 0; kk < WARP_SIZE; kk++) {
            sm_k[kk] = __shfl_sync(MASK, id, kk);
            sm_v[kk] = __shfl_sync(MASK, value, kk);
            result += vin[sm_k[kk] * feat_in + j] * sm_v[kk];
        }
    }
    vout[i * feat_in + j] = result;
}

void SpMMOpt::preprocess(float *vin, float *vout)
{
    // TODO: your code
    grid.x = (num_v + BLOCK_SIZE - 1) / BLOCK_SIZE;
    grid.y = (feat_in + WARP_SIZE - 1) / WARP_SIZE;
    block.x = BLOCK_SIZE * WARP_SIZE;
}

void SpMMOpt::run(float *vin, float *vout)
{
    // TODO: your code
    spmm_kernel_opt<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in);
}