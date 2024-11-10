#include "spmm_opt.h"
#include <cstdio>

// sparse matrix: (num_v, ?), dense matrix: (?, INFEATURE), output: (num_v, INFEATURE)
// use shared memory and 2d thread block

#define BLOCK_SIZE 16
#define WARP_SIZE 32
#define MASK 0xffffffff

__global__ void spmm_kernel_opt(int *ptr, int *idx, float *val, float *vin, float *vout, int num_v, int feat_in) {
    __shared__ int sm_k[BLOCK_SIZE][WARP_SIZE];
    __shared__ float sm_v[BLOCK_SIZE][WARP_SIZE];
    int tid_x = threadIdx.y;
    int tid_y = threadIdx.x;
    int i = blockIdx.x * BLOCK_SIZE + tid_x;
    int j = blockIdx.y * WARP_SIZE + tid_y;
    if (i >= num_v || j >= feat_in) return;
    int begin = ptr[i], end = ptr[i + 1];
    float result = 0.0;
    for (int p = begin; p < end; p += WARP_SIZE) {
        if (p + tid_y < end) {
            sm_k[tid_x][tid_y] = idx[p + tid_y];
            sm_v[tid_x][tid_y] = val[p + tid_y];
        }
        else {
            sm_k[tid_x][tid_y] = 0;
            sm_v[tid_x][tid_y] = 0.0;
        }
        __syncwarp();
        #pragma unroll
        for (int kk = 0; kk < WARP_SIZE; kk++) {
            result += vin[sm_k[tid_x][kk] * feat_in + j] * sm_v[tid_x][kk];
        }
    }
    vout[i * feat_in + j] = result;
}

void SpMMOpt::preprocess(float *vin, float *vout)
{
    // TODO: your code
    grid.x = (num_v + BLOCK_SIZE - 1) / BLOCK_SIZE;
    grid.y = (feat_in + WARP_SIZE - 1) / WARP_SIZE;
    block.y = BLOCK_SIZE;
    block.x = WARP_SIZE;
}

void SpMMOpt::run(float *vin, float *vout)
{
    // TODO: your code
    spmm_kernel_opt<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in);
}