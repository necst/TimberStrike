//
// Created by liqinbin on 10/14/20.
// ThunderGBM device_lambda.h: https://github.com/Xtra-Computing/thundergbm/blob/master/include/thundergbm/util/device_lambda.h
// Under Apache-2.0 license
// copyright (c) 2020 jiashuai
//

#ifndef FEDTREE_DEVICE_LAMBDA_H
#define FEDTREE_DEVICE_LAMBDA_H

#ifdef USE_CUDA

#include "FedTree/common.h"

template<typename L>
__global__ void lambda_kernel(size_t len, L lambda) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < len; i += blockDim.x * gridDim.x) {
        lambda(i);
    }
}

template<typename L>
__global__ void anonymous_kernel_k(L lambda) {
    lambda();
}

template<typename L>
__global__ void lambda_2d_sparse_kernel(const int *len2, L lambda) {
    int i = blockIdx.x;
    int begin = len2[i];
    int end = len2[i + 1];
    for (int j = begin + blockIdx.y * blockDim.x + threadIdx.x; j < end; j += blockDim.x * gridDim.y) {
        lambda(i, j);
    }
}

///p100 has 56 MPs, using 32*56 thread blocks
template<int NUM_BLOCK = 32 * 56, int BLOCK_SIZE = 256, typename L>
//template<int NUM_BLOCK = 30345/4, int BLOCK_SIZE = 128, typename L>
inline void device_loop(int len, L lambda) {
    if (len > 0) {
        lambda_kernel << < NUM_BLOCK, BLOCK_SIZE >> > (len, lambda);
        cudaDeviceSynchronize();
        CUDA_CHECK(cudaPeekAtLastError());
    }
}

template<typename L>
inline void anonymous_kernel(L lambda, size_t smem_size = 0, int NUM_BLOCK = 32 * 56, int BLOCK_SIZE = 256) {
    anonymous_kernel_k<< < NUM_BLOCK, BLOCK_SIZE, smem_size >> > (lambda);
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaPeekAtLastError());
}

/**
 * @brief: (len1 x NUM_BLOCK) is the total number of blocks; len2 is an array of lengths.
 */
template<typename L>
void device_loop_2d(int len1, const int *len2, L lambda, unsigned int NUM_BLOCK = 4 * 56,
                    unsigned int BLOCK_SIZE = 256) {
    if (len1 > 0) {
        lambda_2d_sparse_kernel << < dim3(len1, NUM_BLOCK), BLOCK_SIZE >> > (len2, lambda);
        cudaDeviceSynchronize();
        CUDA_CHECK(cudaPeekAtLastError());
    }
}
#endif

#endif //FEDTREE_DEVICE_LAMBDA_H
