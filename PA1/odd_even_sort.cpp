#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <mpi.h>

#include "worker.h"

void partial_merge(float *first, float *second, int first_len, int second_len, bool keeps_smaller, float *temp) {
  // require half size of temp
  if (keeps_smaller) {
    int i = 0, j = 0, k = 0;
    while (i < first_len && j < second_len && k < first_len) {
      if (first[i] < second[j]) {
        temp[k++] = first[i++];
      } else {
        temp[k++] = second[j++];
      }
    }
    while (i < first_len && k < first_len) {
      temp[k++] = first[i++];
    }
    while (j < second_len && k < first_len) {
      temp[k++] = second[j++];
    }
  } else {
    int i = first_len - 1, j = second_len - 1, k = second_len - 1;
    while (i >= 0 && j >= 0 && k >= 0) {
      if (first[i] > second[j]) {
        temp[k--] = first[i--];
      } else {
        temp[k--] = second[j--];
      }
    }
    while (i >= 0 && k >= 0) {
      temp[k--] = first[i--];
    }
    while (j >= 0 && k >= 0) {
      temp[k--] = second[j--];
    }
  }
}

int get_recv_len(int n, int nprocs, int rank) {
  int block_size = ceiling(n, nprocs);
  int IO_offset = block_size * rank;
  return (IO_offset >= n) ? 0 : std::min(block_size, n - IO_offset);
}

void Worker::sort() {
  /** Your code ... */
  // you can use variables in class Worker: n, nprocs, rank, block_len, data
  // n: number of elements
  // nprocs: number of processes
  // block_len = n / nprocs: number of elements in each process, except the last one
  // rank: rank of current process
  // data: pointer to the array of floats
  if (out_of_range) return;

  int iters = nprocs;
  int odd_even = 0; // 0 for switching (2k, 2k+1), 1 for switching (2k+1, 2k+2)
  float send_info[2], recv_info[2]; // info[0]: min, info[1]: max
  int max_neighbour_len = ceiling(n, nprocs); // maximum length of the neighbour's block
  float *recv_buf = new float[max_neighbour_len]; // buffer for receiving
  float *merge_buf = new float[max_neighbour_len]; // temporary buffer for merging， partial_merge
  MPI_Request req_send, req_recv, req_send_buf, req_recv_buf;
  std::sort(data, data + block_len);

  send_info[0] = data[0];
  send_info[1] = data[block_len - 1];
  if (rank % 2 == 0 && !last_rank) {
    MPI_Isend(&send_info[0], 2, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, &req_send);
    MPI_Irecv(&recv_info[0], 2, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, &req_recv);
  } else if (rank % 2 == 1 && rank > 0) {
    MPI_Isend(&send_info[0], 2, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, &req_send);
    MPI_Irecv(&recv_info[0], 2, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, &req_recv);
  }
  while (iters--) { // maximum number of iterations
    if (rank % 2 == odd_even) { // merge and keep the smaller part: rank is even while switching (2k, 2k+1) or rank is odd while switching (2k+1, 2k+2)
      bool need_send = true;
      if (!last_rank) {
        MPI_Wait(&req_send, MPI_STATUS_IGNORE);
        MPI_Wait(&req_recv, MPI_STATUS_IGNORE);
        if (recv_info[0] < data[block_len - 1]) { // if the min of the next rank is smaller than the max of the current rank
          int recv_len = get_recv_len(n, nprocs, rank + 1);
          MPI_Isend(data, block_len, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, &req_send_buf);
          MPI_Irecv(recv_buf, recv_len, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, &req_recv_buf);
          MPI_Wait(&req_send_buf, MPI_STATUS_IGNORE);
          MPI_Wait(&req_recv_buf, MPI_STATUS_IGNORE);
          // overlap communication and merge
          if (rank > 0 && iters > 0) {
            need_send = false;
            send_info[0] = std::min(data[0], recv_buf[0]);
            MPI_Isend(&send_info[0], 2, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, &req_send);
            MPI_Irecv(&recv_info[0], 2, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, &req_recv);
          }
          partial_merge(data, recv_buf, block_len, recv_len, true, merge_buf); // merge and keep the smaller part
          std::swap(data, merge_buf);
        }
        // else don't send or receive data
      }
      // if info has not been sent or received
      if (rank > 0 && iters > 0 && need_send) {
        send_info[0] = data[0];
        send_info[1] = data[block_len - 1];
        MPI_Isend(&send_info[0], 2, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, &req_send);
        MPI_Irecv(&recv_info[0], 2, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, &req_recv);
      }      
      odd_even = 1 - odd_even;
    } else { // merge and keep the larger part: rank is odd while switching (2k, 2k+1) or rank is even while switching (2k+1, 2k+2)
      bool need_send = true;
      if (rank > 0) {
        MPI_Wait(&req_send, MPI_STATUS_IGNORE);
        MPI_Wait(&req_recv, MPI_STATUS_IGNORE);
        if (recv_info[1] > data[0]) { // if the max of the previous rank is larger than the min of the current rank
          int recv_len = get_recv_len(n, nprocs, rank - 1);
          MPI_Isend(data, block_len, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, &req_send_buf);
          MPI_Irecv(recv_buf, recv_len, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, &req_recv_buf);
          MPI_Wait(&req_send_buf, MPI_STATUS_IGNORE);
          MPI_Wait(&req_recv_buf, MPI_STATUS_IGNORE);
          // overlap communication and merge
          if (!last_rank && iters > 0) {
            need_send = false;
            send_info[1] = std::max(data[block_len - 1], recv_buf[recv_len - 1]);
            MPI_Isend(&send_info[0], 2, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, &req_send);
            MPI_Irecv(&recv_info[0], 2, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, &req_recv);
          }
          partial_merge(recv_buf, data, recv_len, block_len, false, merge_buf); // merge and keep the larger part
          std::swap(data, merge_buf);
        }
        // else don't send or receive data
      }
      // if info has not been sent or received
      if (!last_rank && iters > 0 && need_send) {
        send_info[0] = data[0];
        send_info[1] = data[block_len - 1];
        MPI_Isend(&send_info[0], 2, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, &req_send);
        MPI_Irecv(&recv_info[0], 2, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, &req_recv);
      }
      odd_even = 1 - odd_even;
    }
  }
  delete[] merge_buf;
  delete[] recv_buf;
}
