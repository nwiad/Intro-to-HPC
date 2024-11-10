// PLEASE MODIFY THIS FILE TO IMPLEMENT YOUR SOLUTION

// divide n * n adjacency matrix into b * b blocks
// use 1 thread block to handle 1 matrix block
// combine kernel launch for phase 2, 3
// further combine
// use shared memory
// stride = 2
// beautify the bounding conditions
// use local variables to avoid redundant memory access
// pad n * n to ((n + b - 1) / b * b) * ((n + b - 1) / b * b) to avoid branching
// unroll the k-loop
// switch __syncthreads() and local variable assignment

#include "apsp.h"
#include <stdio.h>
#include "cuda_utils.h"

constexpr int b = 64; // n * n adjacency matrix is divided into b * b blocks
constexpr int stride = 2; // stride of matrix block
constexpr int INF = 0x3fffffff; // use (INT_MAX + 1)/2 - 1 = 0x3fffffff to avoid overflow

namespace {


__global__ void kernel_phase_1(int n, int p, int *graph) {
    // blockId.x = blockId.y = 0
    // blockDim.x = blockDim.y = b/2
    // threadIdx.y in [0, b/2 - 1], threadIdx.x in [0, b/2 - 1]
    __shared__ int shared_diag[b][b];
    int y = threadIdx.y * stride; // y in [0, b - 2]
    int x = threadIdx.x * stride; // x in [0, b - 2]
    int i = y + p * b; // i in [p * b, (p + 1) * b - 2]
    int j = x + p * b; // j in [p * b, (p + 1) * b - 2]
    // no branching
    shared_diag[y][x]         = graph[i * n + j];
    shared_diag[y][x + 1]     = graph[i * n + j + 1];
    shared_diag[y + 1][x]     = graph[(i + 1) * n + j];
    shared_diag[y + 1][x + 1] = graph[(i + 1) * n + j + 1];
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < b; k++) {
        shared_diag[y][x]         = min(shared_diag[y][x]        , shared_diag[y][k] + shared_diag[k][x]);
        shared_diag[y][x + 1]     = min(shared_diag[y][x + 1]    , shared_diag[y][k] + shared_diag[k][x + 1]);
        shared_diag[y + 1][x]     = min(shared_diag[y + 1][x]    , shared_diag[y + 1][k] + shared_diag[k][x]);
        shared_diag[y + 1][x + 1] = min(shared_diag[y + 1][x + 1], shared_diag[y + 1][k] + shared_diag[k][x + 1]);
        __syncthreads();
    }
    // no branching
    graph[i * n + j]           = shared_diag[y][x];
    graph[i * n + j + 1]       = shared_diag[y][x + 1];
    graph[(i + 1) * n + j]     = shared_diag[y + 1][x];
    graph[(i + 1) * n + j + 1] = shared_diag[y + 1][x + 1];
}

__global__ void kernel_phase_2_row(int n, int p, int *graph) {
    if (blockIdx.x == p) return;
    // blockId.y = 0
    // blockDim.x = blockDim.y = b/2
    // threadIdx.y in [0, b/2 - 1], threadIdx.x in [0, b/2 - 1],
    __shared__ int shared_blk[b][b];
    __shared__ int shared_diag[b][b];
    auto y = threadIdx.y * stride; // y in [0, b - 2]
    auto x = threadIdx.x * stride; // x in [0, b - 2]
    auto i = y + p * b; // i in [p * b, (p + 1) * b - 2]
    auto j = blockIdx.x * b + x; // j in [blockIdx.x * b, (blockIdx.x + 1) * b - 2]
    auto i_diag = y + p * b; // i_diag = i in [p * b, (p + 1) * b - 2]
    auto j_diag = x + p * b; // j_diag in [p * b, (p + 1) * b - 2]
    // no branching
    shared_blk[y][x]          = graph[i * n + j];
    shared_blk[y][x + 1]      = graph[i * n + j + 1];
    shared_blk[y + 1][x]      = graph[(i + 1) * n + j];
    shared_blk[y + 1][x + 1]  = graph[(i + 1) * n + j + 1];
    shared_diag[y][x]         = graph[i_diag * n + j_diag];
    shared_diag[y][x + 1]     = graph[i_diag * n + j_diag + 1];
    shared_diag[y + 1][x]     = graph[(i_diag + 1) * n + j_diag];
    shared_diag[y + 1][x + 1] = graph[(i_diag + 1) * n + j_diag + 1];
    int blk[2][2];
    blk[0][0] = shared_blk[y][x];
    blk[0][1] = shared_blk[y][x + 1];
    blk[1][0] = shared_blk[y + 1][x];
    blk[1][1] = shared_blk[y + 1][x + 1];
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < b; k++) {
        blk[0][0] = min(blk[0][0], shared_diag[y][k] + shared_blk[k][x]);
        blk[0][1] = min(blk[0][1], shared_diag[y][k] + shared_blk[k][x + 1]);
        blk[1][0] = min(blk[1][0], shared_diag[y + 1][k] + shared_blk[k][x]);
        blk[1][1] = min(blk[1][1], shared_diag[y + 1][k] + shared_blk[k][x + 1]);
        // no revision of shared memory, no need to run __syncthreads()
    }
    // no branching
    graph[i * n + j]           = blk[0][0];
    graph[i * n + j + 1]       = blk[0][1];
    graph[(i + 1) * n + j]     = blk[1][0];
    graph[(i + 1) * n + j + 1] = blk[1][1];
}

__global__ void kernel_phase_2_col(int n, int p, int *graph) {
    if (blockIdx.y == p) return;
    // blockIdx.x = 0
    // blockDim.x = blockDim.y = b/2
    // threadIdx.y in [0, b/2 - 1], threadIdx.x in [0, b/2 - 1],
    __shared__ int shared_blk[b][b];
    __shared__ int shared_diag[b][b];
    auto y = threadIdx.y * stride; // y in [0, b - 2]
    auto x = threadIdx.x * stride; // x in [0, b - 2]
    auto i = blockIdx.y * b + y; // i in [blockIdx.y * b, (blockIdx.y + 1) * b - 2]
    auto j = x + p * b; // j in [p * b, (p + 1) * b - 2]
    auto i_diag = y + p * b; // i_diag in [p * b, (p + 1) * b - 2]
    auto j_diag = x + p * b; // j_diag = j in [p * b, (p + 1) * b - 2]
    // no branching
    shared_blk[y][x]          = graph[i * n + j];
    shared_blk[y][x + 1]      = graph[i * n + j + 1];
    shared_blk[y + 1][x]      = graph[(i + 1) * n + j];
    shared_blk[y + 1][x + 1]  = graph[(i + 1) * n + j + 1];
    shared_diag[y][x]         = graph[i_diag * n + j_diag];
    shared_diag[y][x + 1]     = graph[i_diag * n + j_diag + 1];
    shared_diag[y + 1][x]     = graph[(i_diag + 1) * n + j_diag];
    shared_diag[y + 1][x + 1] = graph[(i_diag + 1) * n + j_diag + 1];
    int blk[2][2];
    blk[0][0] = shared_blk[y][x];
    blk[0][1] = shared_blk[y][x + 1];
    blk[1][0] = shared_blk[y + 1][x];
    blk[1][1] = shared_blk[y + 1][x + 1];
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < b; k++) {
        blk[0][0] = min(blk[0][0], shared_blk[y][k] + shared_diag[k][x]);
        blk[0][1] = min(blk[0][1], shared_blk[y][k] + shared_diag[k][x + 1]);
        blk[1][0] = min(blk[1][0], shared_blk[y + 1][k] + shared_diag[k][x]);
        blk[1][1] = min(blk[1][1], shared_blk[y + 1][k] + shared_diag[k][x + 1]);
        // no revision of shared memory, no need to run __syncthreads()
    }
    // no branching
    graph[i * n + j]           = blk[0][0];
    graph[i * n + j + 1]       = blk[0][1];
    graph[(i + 1) * n + j]     = blk[1][0];
    graph[(i + 1) * n + j + 1] = blk[1][1];
}

__global__ void kernel_phase_3(int n, int p, int *graph){
    if (blockIdx.x == p || blockIdx.y == p) return;
    // blockDim.x = blockDim.y = b/2
    // threadIdx.y in [0, b/2 - 1], threadIdx.x in [0, b/2 - 1]
    __shared__ int shared_row[b][b];
    __shared__ int shared_col[b][b];
    auto y = threadIdx.y * stride; // y in [0, b - 2]
    auto x = threadIdx.x * stride; // x in [0, b - 2]
    auto i = blockIdx.y * b + y; // i in [blockIdx.y * b, (blockIdx.y + 1) * b - 2]
    auto j = blockIdx.x * b + x; // j in [blockIdx.x * b, (blockIdx.x + 1) * b - 2]
    auto i_row = blockIdx.y * b + y; // i_row = i in [blockIdx.y * b, (blockIdx.y + 1) * b - 2]
    auto j_row = x + p * b; // j_row in [p * b, (p + 1) * b - 2]
    auto i_col = y + p * b; // i_col in [p * b, (p + 1) * b - 2]
    auto j_col = blockIdx.x * b + x; // j_col = j in [blockIdx.x * b, (blockIdx.x + 1) * b - 2]
    // no branching
    shared_row[y][x]         = graph[i_row * n + j_row];
    shared_row[y][x + 1]     = graph[i_row * n + j_row + 1];
    shared_row[y + 1][x]     = graph[(i_row + 1) * n + j_row];
    shared_row[y + 1][x + 1] = graph[(i_row + 1) * n + j_row + 1];
    shared_col[y][x]         = graph[i_col * n + j_col];
    shared_col[y][x + 1]     = graph[i_col * n + j_col + 1];
    shared_col[y + 1][x]     = graph[(i_col + 1) * n + j_col];
    shared_col[y + 1][x + 1] = graph[(i_col + 1) * n + j_col + 1];
    int blk[2][2];
    // since graph[i * n + j] is not used by other threads, no need to store it in shared memory
    blk[0][0] = graph[i * n + j];
    blk[0][1] = graph[i * n + j + 1];
    blk[1][0] = graph[(i + 1) * n + j];
    blk[1][1] = graph[(i + 1) * n + j + 1];
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < b; k++) {
        blk[0][0] = min(blk[0][0], shared_row[y][k] + shared_col[k][x]);
        blk[0][1] = min(blk[0][1], shared_row[y][k] + shared_col[k][x + 1]);
        blk[1][0] = min(blk[1][0], shared_row[y + 1][k] + shared_col[k][x]);
        blk[1][1] = min(blk[1][1], shared_row[y + 1][k] + shared_col[k][x + 1]);
        // no revision of shared memory, no need to run __syncthreads()
    }
    // no branching
    graph[i * n + j]           = blk[0][0];
    graph[i * n + j + 1]       = blk[0][1];
    graph[(i + 1) * n + j]     = blk[1][0];
    graph[(i + 1) * n + j + 1] = blk[1][1];
}

__global__ void copyAndPadGraph(int n, int padded_len, /* device */ int *graph, /* device */ int *padded_graph) {
    // blockDim.x = blockDim.y = 32
    // threadIdx.y in [0, 31], threadIdx.x in [0, 31]
    auto i = blockIdx.y * blockDim.y + threadIdx.y;
    auto j = blockIdx.x * blockDim.x + threadIdx.x;
    padded_graph[i * padded_len + j] = (i < n && j < n) ? graph[i * n + j] : INF;
}

__global__ void copyPaddedGraph(int n, int padded_len, /* device */ int *graph, /* device */ int *padded_graph) {
    // blockDim.x = blockDim.y = 32
    // threadIdx.y in [0, 31], threadIdx.x in [0, 31]
    auto i = blockIdx.y * blockDim.y + threadIdx.y;
    auto j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && j < n) {
        graph[i * n + j] = padded_graph[i * padded_len + j];
    }
}

}

void apsp(int n, /* device */ int *graph) {
    int iters = (n + b - 1) / b;
    int *padded_graph;
    int padded_len = iters * b;
    CHK_CUDA_ERR(cudaMalloc(&padded_graph, padded_len * padded_len * sizeof(int)));
    dim3 pad_thr(32, 32);
    dim3 pad_blk((padded_len - 1) / 32 + 1, (padded_len - 1) / 32 + 1);
    copyAndPadGraph<<<pad_blk, pad_thr>>>(n, padded_len, graph, padded_graph);
    dim3 thr(b/stride, b/stride); // (b/2, b/2)
    dim3 blk_1(1);
    dim3 blk_row(iters, 1);
    dim3 blk_col(1, iters);
    dim3 blk_3(iters, iters);
    for (int p = 0; p < iters; p++) {
        kernel_phase_1<<<blk_1, thr>>>(padded_len, p, padded_graph);
        kernel_phase_2_row<<<blk_row, thr>>>(padded_len, p, padded_graph);
        kernel_phase_2_col<<<blk_col, thr>>>(padded_len, p, padded_graph);
        kernel_phase_3<<<blk_3, thr>>>(padded_len, p, padded_graph);
    }
    dim3 cpy_thr(32, 32);
    dim3 cpy_blk((n - 1) / 32 + 1, (n - 1) / 32 + 1);
    copyPaddedGraph<<<cpy_blk, cpy_thr>>>(n, padded_len, graph, padded_graph);
    CHK_CUDA_ERR(cudaFree(padded_graph));
}
