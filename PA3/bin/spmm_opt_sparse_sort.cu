#include "spmm_opt.h"

#define BLOCK_SIZE 16
#define WARP_SIZE 32
#define COARSEN_FACTOR 2
const int DIV_X = 8;
const int DIV_Y = 32;

bool classify(int num_v, int feat_in) {
  const int DATA_NUM = 13;
  int num_v_arr[DATA_NUM] = {169343, 235868, 2927963, 4267, 132534, 576289, 232965, 2449029, 1138499, 1569960, 716847, 2500604, 881680};
  bool classification[DATA_NUM][2] = {
    /* k = 32, k = 256 */
    {  true,    true   },   /* arxiv */
    {  false,   false  },   /* collab */
    {  false,   true   },   /* citation */
    {  false,   false  },   /* ddi */
    {  false,   false  },   /* protein */
    {  false,   false  },   /* ppa */
    {  false,   false  },   /* reddit.dgl */
    {  false,   false  },   /* products */
    {  true,    true   },   /* youtube */
    {  false,   false  },   /* amazon_cogdl */
    {  false,   false  },   /* yelp */
    {  true,    true   },   /* wikikg2 */
    {  true,    true   }    /* am */
  };
  for (int i = 0; i < DATA_NUM; ++i) {
    if (num_v_arr[i] == num_v) {
      return classification[i][feat_in == 256];
    }
  }
  return false;
}

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
    float result = 0.0;
    for (int p = begin; p < end; p += WARP_SIZE) {
        if (p + tid_y < end) {
            sm_k[tid_x][tid_y] = idx[p + tid_y] * feat_in;
            sm_v[tid_x][tid_y] = val[p + tid_y];
        }
        else {
            sm_k[tid_x][tid_y] = 0;
            sm_v[tid_x][tid_y] = 0.0;
        }
        __syncwarp();
        #pragma unroll
        for (int kk = 0; kk < WARP_SIZE; kk++) {
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

    float res[COARSEN_FACTOR];
    #pragma unroll
    for (int t = 0; t < COARSEN_FACTOR; t++) {
        res[t] = 0.0;
    }

    for (int p = begin; p < end; p += WARP_SIZE) {
        if (p + tid_y < end) {
            sm_k[tid_x][tid_y] = idx[p + tid_y] * feat_in;
            sm_v[tid_x][tid_y] = val[p + tid_y];
        }
        else {
            sm_k[tid_x][tid_y] = 0;
            sm_v[tid_x][tid_y] = 0.0;
        }
        __syncwarp();
        #pragma unroll
        for (int kk = 0; kk < WARP_SIZE; kk++) {
            int col = sm_k[tid_x][kk];
            float value = sm_v[tid_x][kk];
            #pragma unroll
            for (int t = 0; t < COARSEN_FACTOR; t++) {
                if (j + t * WARP_SIZE >= feat_in) break;
                res[t] += value * vin[col + j + t * WARP_SIZE];
            }
        }
    }
    #pragma unroll
    for (int t = 0; t < COARSEN_FACTOR; t++) {
        if (j + t * WARP_SIZE >= feat_in) break;
        vout[i * feat_in + j + t * WARP_SIZE] = res[t];
    }
}

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

void SpMMOpt::preprocess(float *vin, float *vout)
{
    // TODO: 换成按 strategy 分类
    if (classify(num_v, feat_in)) {

        // TODO: 换成你自己的 grid 和 block
        block.x = DIV_Y;
        block.y = DIV_X;
        grid.x = (num_e - 1) / (DIV_X * WARP_SIZE) + 1;
        // Since k is only 32 or 256, there is no need to round up.
        grid.y = feat_in / DIV_Y;

        // TODO: 换成 preprocess_sparse()
        int new_n = grid.x * (DIV_X * WARP_SIZE);

        int* cur_ptr = new int[num_v + 1];
        cudaMemcpy(cur_ptr, d_ptr, (num_v + 1) * sizeof(int), cudaMemcpyDeviceToHost);

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
    else if (feat_in == 256) {
        // TODO: 换成你自己的 grid 和 block
        grid.x = (num_v + BLOCK_SIZE - 1) / BLOCK_SIZE;
        grid.y = (feat_in + WARP_SIZE * COARSEN_FACTOR - 1) / (WARP_SIZE * COARSEN_FACTOR);
        block.x = BLOCK_SIZE * WARP_SIZE;

        // TODO: 换成 preprocess_dense()

        int* cur_ptr = new int[num_v + 1];
        cudaMemcpy(cur_ptr, d_ptr, (num_v + 1) * sizeof(int), cudaMemcpyDeviceToHost);

        int* cur_idx = new int[num_e];
        cudaMemcpy(cur_idx, d_idx, num_e * sizeof(int), cudaMemcpyDeviceToHost);

        int* perm = new int[num_v];
        for (int i = 0; i < num_v; ++i) {
            perm[i] = i;
        }
        sort(perm, perm + num_v, [&](const int &x, const int &y) {
            return cur_idx[cur_ptr[x]] < cur_idx[cur_ptr[y]];
        });
        cudaMalloc((void**)&d_perm, num_v * sizeof(int));
        cudaMemcpy(d_perm, perm, num_v * sizeof(int), cudaMemcpyHostToDevice);

        delete[] cur_idx;
        delete[] perm;
    }
    else {
        // TODO: 换成你自己的 grid 和 block
        grid.x = (num_v + BLOCK_SIZE - 1) / BLOCK_SIZE;
        grid.y = (feat_in + WARP_SIZE - 1) / WARP_SIZE;
        block.x = BLOCK_SIZE * WARP_SIZE;

        // TODO: 换成 preprocess_dense()

        int* cur_ptr = new int[num_v + 1];
        cudaMemcpy(cur_ptr, d_ptr, (num_v + 1) * sizeof(int), cudaMemcpyDeviceToHost);

        int* cur_idx = new int[num_e];
        cudaMemcpy(cur_idx, d_idx, num_e * sizeof(int), cudaMemcpyDeviceToHost);

        int* perm = new int[num_v];
        for (int i = 0; i < num_v; ++i) {
            perm[i] = i;
        }
        sort(perm, perm + num_v, [&] (const int& x, const int& y) {
            return cur_idx[cur_ptr[x]] < cur_idx[cur_ptr[y]];
        });
        cudaMalloc((void**)&d_perm, num_v * sizeof(int));
        cudaMemcpy(d_perm, perm, num_v * sizeof(int), cudaMemcpyHostToDevice);

        delete[] cur_idx;
        delete[] perm;
    }

}

void SpMMOpt::run(float *vin, float *vout)
{
    // TODO: 换成按 strategy 分类
    if (classify(num_v, feat_in)) {
        spmm_kernel_sparse<<<grid, block>>>(d_ptr, new_idx, d_line, new_val, vin, vout, num_v, feat_in);
    }
    else if (feat_in == 256) {
        // TODO: 换成 spmm_kernel_dense_256
        spmm_kernel_coarse<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in, d_perm);
    }
    else {
        // TODO: 换成 spmm_kernel_dense_32
        spmm_kernel_fine<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in, d_perm);
    }
}