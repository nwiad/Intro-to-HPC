#include "spmm_opt.h"
#include <cstdio>

// sparse matrix: (num_v, num_v), dense matrix: (num_v, feat_in), output: (num_v, feat_in)

#define BLOCK_X 8
#define BLOCK_Y 32
#define WARP_SIZE 32
#define BLOCK_SIZE 1
#define COARSEN_FACTOR 2

#define DISABLE_NNZ_PARALLEL 0
#define DISABLE_LEFTMOST_SORTING 0
#define DISABLE_COALESCED_MEMORY_ACCESS 1

__global__ void spmm_kernel_fine_ldg(int *ptr, int *idx, float *val, float *vin, float *vout, int num_v, int feat_in, int *rowidx) {
    int tid_x = threadIdx.x >> 5;
    int tid_y = threadIdx.x & 0x1f;
    int i = blockIdx.x * BLOCK_SIZE + tid_x;
    int j = blockIdx.y * WARP_SIZE + tid_y;
    if (i >= num_v || j >= feat_in) return;
    float *vin_1 = vin + j;
    float *vout_1 = vout + j;
    i = rowidx[i];
    int begin = ptr[i], end = ptr[i + 1];
    int iter_end;
    float result = 0.0f;
    for (int p = begin; p < end; p += WARP_SIZE) {
        iter_end = min(WARP_SIZE, end - p);
        for (int kk = 0; kk < iter_end; kk++) {
            result += vin_1[idx[p + kk] * feat_in] * val[p + kk];
        }
    }
    vout_1[i * feat_in] = result;
}

__global__ void spmm_kernel_coarse_ldg(int *ptr, int *idx, float *val, float *vin, float *vout, int num_v, int feat_in, int *rowidx) {
    int tid_x = threadIdx.x >> 5;
    int tid_y = threadIdx.x & 0x1f;
    int i = blockIdx.x * BLOCK_SIZE + tid_x;
    int j = blockIdx.y * WARP_SIZE * COARSEN_FACTOR + tid_y;
    if (i >= num_v || j >= feat_in) return;
    float *vin_1 = vin + j;
    float *vout_1 = vout + j;
    i = rowidx[i];
    int begin = ptr[i], end = ptr[i + 1];
    int iter_end;

    float res[COARSEN_FACTOR]={0.0f};
    int id;
    float value;

    for (int p = begin; p < end; p += WARP_SIZE) {
        iter_end = min(WARP_SIZE, end - p);
        for (int kk = 0; kk < iter_end; kk++) {
            id = idx[p + kk] * feat_in;
            value = val[p + kk];
            #pragma unroll
            for (int t = 0; t < COARSEN_FACTOR; t++) {
                if (j + t * WARP_SIZE >= feat_in) break;
                res[t] += value * vin_1[id + t * WARP_SIZE];
            }
        }
    }
    #pragma unroll
    for (int t = 0; t < COARSEN_FACTOR; t++) {
        if (j + t * WARP_SIZE >= feat_in) break;
        vout_1[i * feat_in + t * WARP_SIZE] = res[t];
    }
}

__global__ void spmm_kernel_nnz_parallel_ldg(int *rowptr, int *idx, float *val, float *vin, float *vout, int num_v, int feat_in) {
    int tid_x = threadIdx.y;
    int tid_y = threadIdx.x;
    int p = (blockIdx.x * BLOCK_X + tid_x) * WARP_SIZE;
    int j = blockIdx.y * BLOCK_Y + tid_y;
    float *vin_1 = vin + j;
    float *vout_1 = vout + j;
    float result = 0.0f;

    // sm_rowptr[tid_x][tid_y] = rowptr[p + tid_y] * feat_in;
    // sm_k[tid_x][tid_y] = idx[p + tid_y] * feat_in;
    // sm_v[tid_x][tid_y] = val[p + tid_y];
    // __syncwarp();

    for (int kk = 0; kk < WARP_SIZE; kk++) {
        // if (sm_rowptr[tid_x][kk] == sm_rowptr[tid_x][kk + 1] && kk < WARP_SIZE - 1) {
        //     result += vin_1[sm_k[tid_x][kk]] * sm_v[tid_x][kk];
        // }
        // else {
        //     result += vin_1[sm_k[tid_x][kk]] * sm_v[tid_x][kk];
        //     atomicAdd(vout_1 + sm_rowptr[tid_x][kk], result);
        //     result = 0.0f;
        // }
        if (rowptr[p + kk] == rowptr[p + kk + 1] && kk < WARP_SIZE - 1) {
            result += vin_1[idx[p + kk] * feat_in] * val[p + kk];
        }
        else {
            result += vin_1[idx[p + kk] * feat_in] * val[p + kk];
            atomicAdd(vout_1 + rowptr[p + kk] * feat_in, result);
            result = 0.0f;
        }

    }
}

__global__ void spmm_kernel_fine(int *ptr, int *idx, float *val, float *vin, float *vout, int num_v, int feat_in, int *rowidx) {
    __shared__ int sm_k[BLOCK_SIZE][WARP_SIZE];
    __shared__ float sm_v[BLOCK_SIZE][WARP_SIZE];
    int tid_x = threadIdx.x >> 5;
    int tid_y = threadIdx.x & 0x1f;
    int i = blockIdx.x * BLOCK_SIZE + tid_x;
    int j = blockIdx.y * WARP_SIZE + tid_y;
    if (i >= num_v || j >= feat_in) return;
    float *vin_1 = vin + j;
    float *vout_1 = vout + j;
    i = rowidx[i];
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
            result += vin_1[sm_k[tid_x][kk]] * sm_v[tid_x][kk];
        }
    }
    vout_1[i * feat_in] = result;
}

__global__ void spmm_kernel_coarse(int *ptr, int *idx, float *val, float *vin, float *vout, int num_v, int feat_in, int *rowidx) {
    __shared__ int sm_k[BLOCK_SIZE][WARP_SIZE];
    __shared__ float sm_v[BLOCK_SIZE][WARP_SIZE];
    int tid_x = threadIdx.x >> 5;
    int tid_y = threadIdx.x & 0x1f;
    int i = blockIdx.x * BLOCK_SIZE + tid_x;
    int j = blockIdx.y * WARP_SIZE * COARSEN_FACTOR + tid_y;
    if (i >= num_v || j >= feat_in) return;
    float *vin_1 = vin + j;
    float *vout_1 = vout + j;
    i = rowidx[i];
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
                res[t] += value * vin_1[id + t * WARP_SIZE];
            }
        }
    }
    #pragma unroll
    for (int t = 0; t < COARSEN_FACTOR; t++) {
        if (j + t * WARP_SIZE >= feat_in) break;
        vout_1[i * feat_in + t * WARP_SIZE] = res[t];
    }
}

__global__ void spmm_kernel_nnz_parallel(int *rowptr, int *idx, float *val, float *vin, float *vout, int num_v, int feat_in) {
    __shared__ int sm_k[BLOCK_X][WARP_SIZE];
    __shared__ float sm_v[BLOCK_X][WARP_SIZE];
    __shared__ int sm_rowptr[BLOCK_X][WARP_SIZE];
    int tid_x = threadIdx.y;
    int tid_y = threadIdx.x;
    int p = (blockIdx.x * BLOCK_X + tid_x) * WARP_SIZE;
    int j = blockIdx.y * BLOCK_Y + tid_y;
    float *vin_1 = vin + j;
    float *vout_1 = vout + j;
    float result = 0.0f;

    sm_rowptr[tid_x][tid_y] = rowptr[p + tid_y] * feat_in;
    sm_k[tid_x][tid_y] = idx[p + tid_y] * feat_in;
    sm_v[tid_x][tid_y] = val[p + tid_y];
    __syncwarp();

    for (int kk = 0; kk < WARP_SIZE; kk++) {
        if (sm_rowptr[tid_x][kk] == sm_rowptr[tid_x][kk + 1] && kk < WARP_SIZE - 1) {
            result += vin_1[sm_k[tid_x][kk]] * sm_v[tid_x][kk];
        }
        else {
            result += vin_1[sm_k[tid_x][kk]] * sm_v[tid_x][kk];
            atomicAdd(vout_1 + sm_rowptr[tid_x][kk], result);
            result = 0.0f;
        }
    }
}

__global__ void assign_pad_idx(int *pad_idx, int *idx, int num_v, int num_e, int pad_nnz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pad_nnz) return;
    pad_idx[i] = i < num_e ? idx[i] : num_v - 1;
}

__global__ void assign_pad_val(float *pad_val, float *val, int num_e, int pad_nnz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pad_nnz) return;
    pad_val[i] = i < num_e ? val[i] : 0;
}

__global__ void assign_rowptr(int *d_rowptr, int *ptr, int num_v, int pad_nnz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_v) return;
    int begin = ptr[i], end = ptr[i + 1];
    for (int j = begin; j < end; j++) {
        if (j >= pad_nnz) break;
        d_rowptr[j] = i;
    }
}

__global__ void assign_rowidx(int *d_rowidx, int *d_lcol, int num_v) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_v) return;
    d_rowidx[i] = i;
}

__global__ void assign_leftmost_col(int *d_lcol, int *d_ptr, int *d_idx, int num_v) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_v) return;
    d_lcol[i] = d_idx[d_ptr[i]];
}

float sparsity(int num_v, int num_e) {
    if (DISABLE_NNZ_PARALLEL) return 1.0f;
    return (float)num_e / ((float)num_v * (float)num_v);
}

bool coarsen(int feat_in) {
    return feat_in > 128;
}

void SpMMOpt::preprocess(float *vin, float *vout)
{
    // TODO: your code
    // if the matrix is too sparse
    if (DISABLE_NNZ_PARALLEL) printf("Disabled non-zero parallelism\n");
    if (DISABLE_LEFTMOST_SORTING) printf("Disabled sorting sparse row by leftmost column index\n");
    if (DISABLE_COALESCED_MEMORY_ACCESS) printf("Disabled coalesced memory access\n");
    if (sparsity(num_v, num_e) < 0.00001) {
        // parallelize along non-zero elements
        grid.x = CEIL(num_e, BLOCK_X * WARP_SIZE);
        grid.y = CEIL(feat_in, BLOCK_Y);
        block.x = BLOCK_Y;
        block.y = BLOCK_X;

        //  pad idx and val
        int pad_nnz = CEIL(num_e, BLOCK_X * WARP_SIZE) * (BLOCK_X * WARP_SIZE);
        cudaMalloc(&pad_idx, pad_nnz * sizeof(int));
        assign_pad_idx<<<CEIL(pad_nnz, 1024), 1024>>>(pad_idx, d_idx, num_v, num_e, pad_nnz);
        cudaMalloc(&pad_val, pad_nnz * sizeof(float));
        assign_pad_val<<<CEIL(pad_nnz, 1024), 1024>>>(pad_val, d_val, num_e, pad_nnz);

        // prepare rowptr for non-zero elements
        cudaMalloc(&d_rowptr, pad_nnz * sizeof(int));
        assign_rowptr<<<CEIL(num_v, 1024), 1024>>>(d_rowptr, d_ptr, num_v, pad_nnz);
    }
    // if the matrix is not so sparse
    else {
        // parallelize along rows
        grid.x = CEIL(num_v, BLOCK_SIZE);
        // for feat_in > 128, use coarse-grained algorithm, otherwise use fine-grained algorithm
        grid.y = coarsen(feat_in) ? CEIL(feat_in, WARP_SIZE * COARSEN_FACTOR) : CEIL(feat_in, WARP_SIZE);
        block.x = BLOCK_SIZE * WARP_SIZE;

        // to balance the workload, sort the rows by the index of the leftmost non-zero element
        int *rowidx = new int[num_v];
        int *leftmost_col = new int[num_v];
        
        cudaMalloc(&d_rowidx, num_v * sizeof(int));
        assign_rowidx<<<CEIL(num_v, 1024), 1024>>>(d_rowidx, d_lcol, num_v);
        cudaMemcpy(rowidx, d_rowidx, num_v * sizeof(int), cudaMemcpyDeviceToHost);

        cudaMalloc(&d_lcol, num_v * sizeof(int));
        assign_leftmost_col<<<CEIL(num_v, 1024), 1024>>>(d_lcol, d_ptr, d_idx, num_v);
        cudaMemcpy(leftmost_col, d_lcol, num_v * sizeof(int), cudaMemcpyDeviceToHost);

        if (!DISABLE_LEFTMOST_SORTING) sort(rowidx, rowidx + num_v, [&](const int &x, const int &y) { return leftmost_col[x] < leftmost_col[y]; });
        cudaMemcpy(d_rowidx, rowidx, num_v * sizeof(int), cudaMemcpyHostToDevice);

        delete[] leftmost_col;
        delete[] rowidx;
    }
}

void SpMMOpt::run(float *vin, float *vout)
{
    // TODO: your code
    // if the matrix is too sparse
    if (sparsity(num_v, num_e) < 0.00001) {
        // parallelize threads in a warp
        if (DISABLE_COALESCED_MEMORY_ACCESS) {
            spmm_kernel_nnz_parallel_ldg<<<grid, block>>>(d_rowptr, pad_idx, pad_val, vin, vout, num_v, feat_in);
            return;
        }
        spmm_kernel_nnz_parallel<<<grid, block>>>(d_rowptr, pad_idx, pad_val, vin, vout, num_v, feat_in);
    }
    // if the matrix is not so sparse
    else if (coarsen(feat_in)) {
        // for feat_in > 128, use coarse-grained algorithm
        if (DISABLE_COALESCED_MEMORY_ACCESS) {
            spmm_kernel_coarse_ldg<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in, d_rowidx);
            return;
        }
        spmm_kernel_coarse<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in, d_rowidx);
    }
    else {
        // for feat_in <= 128, use fine-grained algorithm
        if (DISABLE_COALESCED_MEMORY_ACCESS) {
            spmm_kernel_fine_ldg<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in, d_rowidx);
            return;
        }
        spmm_kernel_fine<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in, d_rowidx);
    }
}