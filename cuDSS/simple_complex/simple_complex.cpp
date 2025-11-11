/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cuComplex.h>

#include "cudss.h"

/*
    This example demonstrates basic usage of cuDSS APIs for solving
    a system of linear algebraic equations with a sparse matrix:
                                Ax = b,
    where:
        A is the sparse input matrix,
        b is the (dense) right-hand side vector (or a matrix),
        x is the (dense) solution vector (or a matrix).
*/

#define CUDSS_EXAMPLE_FREE                                      \
  do {                                                          \
    /* 释放保存在主机端的 CSR 结构、解向量和右端项等缓冲区。 */ \
    free(csr_offsets_h);                                        \
    free(csr_columns_h);                                        \
    free(csr_values_h);                                         \
    free(x_values_h);                                           \
    free(b_values_h);                                           \
    /* 释放 GPU 端与上述缓冲区一一对应的设备内存，避免泄漏。 */ \
    cudaFree(csr_offsets_d);                                    \
    cudaFree(csr_columns_d);                                    \
    cudaFree(csr_values_d);                                     \
    cudaFree(x_values_d);                                       \
    cudaFree(b_values_d);                                       \
  } while (0);

#define CUDA_CALL_AND_CHECK(call, msg) \
    do { \
        cuda_error = call; \
        if (cuda_error != cudaSuccess) { \
            printf("Example FAILED: CUDA API returned error = %d, details: " #msg "\n", cuda_error); \
            CUDSS_EXAMPLE_FREE; \
            return -1; \
        } \
    } while(0);


#define CUDSS_CALL_AND_CHECK(call, status, msg) \
    do { \
        status = call; \
        if (status != CUDSS_STATUS_SUCCESS) { \
            printf("Example FAILED: CUDSS call ended unsuccessfully with status = %d, details: " #msg "\n", status); \
            CUDSS_EXAMPLE_FREE; \
            return -2; \
        } \
    } while(0);


int main (int argc, char *argv[]) {
    printf("---------------------------------------------------------\n");
    printf("cuDSS example: solving a complex linear 5x5 system\n"
           "with a symmetric positive-definite matrix \n");
    printf("---------------------------------------------------------\n");
    cudaError_t cuda_error = cudaSuccess;
    cudssStatus_t status = CUDSS_STATUS_SUCCESS;

    int n = 5;     // 问题规模
    int nnz = 8;   // 非零元数量
    int nrhs = 1;  // 右端项数量

    /* 主机端用于保存稀疏矩阵 A 的 CSR 结构以及稠密向量的缓冲区。 */
    int* csr_offsets_h = NULL;  // CSR 行偏移数组
    int* csr_columns_h = NULL;  // CSR 列索引数组
    cuComplex* csr_values_h =
        NULL;  // CSR 非零元数组, cuComplex 类型表示float32复数
    cuComplex *x_values_h = NULL, *b_values_h = NULL;  // 解向量和右端项

    /* 设备端（GPU）上与主机缓冲区镜像的数据，用于实际计算。 */
    int *csr_offsets_d = NULL;
    int *csr_columns_d = NULL;
    cuComplex *csr_values_d = NULL;
    cuComplex *x_values_d = NULL, *b_values_d = NULL;

    /* Allocate host memory for the sparse input matrix A,
       right-hand side x and solution b*/
    /*分配主机内存*/
    csr_offsets_h = (int*)malloc((n + 1) * sizeof(int));
    csr_columns_h = (int*)malloc(nnz * sizeof(int));
    csr_values_h = (cuComplex*)malloc(nnz * sizeof(cuComplex));
    x_values_h = (cuComplex*)malloc(nrhs * n * sizeof(cuComplex));
    b_values_h = (cuComplex*)malloc(nrhs * n * sizeof(cuComplex));

    if (!csr_offsets_h || ! csr_columns_h || !csr_values_h ||
        !x_values_h || !b_values_h) {
        printf("Error: host memory allocation failed\n");
        return -1;
    }

    /* Initialize host memory for A and b */
    int i = 0;
    /* CSR 行偏移数组标识每一行在列索引与数值数组中的范围。 */
    csr_offsets_h[i++] = 0;
    csr_offsets_h[i++] = 2;
    csr_offsets_h[i++] = 4;
    csr_offsets_h[i++] = 6;
    csr_offsets_h[i++] = 7;
    csr_offsets_h[i++] = 8;

    i = 0;
    /* 非零元对应的列索引，与下面的 csr_values_h 一一对应。 */
    csr_columns_h[i++] = 0; csr_columns_h[i++] = 2;
    csr_columns_h[i++] = 1; csr_columns_h[i++] = 2;
    csr_columns_h[i++] = 2; csr_columns_h[i++] = 4;
    csr_columns_h[i++] = 3;
    csr_columns_h[i++] = 4;

    i = 0;
    /* 非零元的实部（存放在结构体成员 x 中），虚部稍后设置。 */
    csr_values_h[i++].x = 4.0; csr_values_h[i++].x = 1.0;
    csr_values_h[i++].x = 3.0; csr_values_h[i++].x = 2.0;
    csr_values_h[i++].x = 5.0; csr_values_h[i++].x = 1.0;
    csr_values_h[i++].x = 1.0;
    csr_values_h[i++].x = 2.0;

    i = 0;
    /* 非零元的虚部（结构体成员 y）。本示例矩阵为实矩阵，因此设为 0。 */
    csr_values_h[i++].y = 0.0; csr_values_h[i++].y = 0.0;
    csr_values_h[i++].y = 0.0; csr_values_h[i++].y = 0.0;
    csr_values_h[i++].y = 0.0; csr_values_h[i++].y = 0.0;
    csr_values_h[i++].y = 0.0;
    csr_values_h[i++].y = 0.0;

    /* Note: Right-hand side b is initialized with values which correspond
       to the exact solution vector {1, 2, 3, 4, 5} */
    i = 0;
    b_values_h[i++].x = 7.0;
    b_values_h[i++].x = 12.0;
    b_values_h[i++].x = 25.0;
    b_values_h[i++].x = 4.0;
    b_values_h[i++].x = 13.0;

    i = 0;
    b_values_h[i++].y = 0.0;
    b_values_h[i++].y = 0.0;
    b_values_h[i++].y = 0.0;
    b_values_h[i++].y = 0.0;
    b_values_h[i++].y = 0.0;

    /* Allocate device memory for A, x and b */
    /*分配设备内存*/
    CUDA_CALL_AND_CHECK(cudaMalloc(&csr_offsets_d, (n + 1) * sizeof(int)),
                        "cudaMalloc for csr_offsets");
    CUDA_CALL_AND_CHECK(cudaMalloc(&csr_columns_d, nnz * sizeof(int)),
                        "cudaMalloc for csr_columns");
    CUDA_CALL_AND_CHECK(cudaMalloc(&csr_values_d, nnz * sizeof(cuComplex)),
                        "cudaMalloc for csr_values");
    CUDA_CALL_AND_CHECK(cudaMalloc(&b_values_d, nrhs * n * sizeof(cuComplex)),
                        "cudaMalloc for b_values");
    CUDA_CALL_AND_CHECK(cudaMalloc(&x_values_d, nrhs * n * sizeof(cuComplex)),
                        "cudaMalloc for x_values");

    /* Copy host memory to device for A and b */
    /*将主机内存复制到设备内存*/
    CUDA_CALL_AND_CHECK(cudaMemcpy(csr_offsets_d, csr_offsets_h, (n + 1) * sizeof(int),
                        cudaMemcpyHostToDevice), "cudaMemcpy for csr_offsets");
    CUDA_CALL_AND_CHECK(cudaMemcpy(csr_columns_d, csr_columns_h, nnz * sizeof(int),
                        cudaMemcpyHostToDevice), "cudaMemcpy for csr_columns");
    CUDA_CALL_AND_CHECK(cudaMemcpy(csr_values_d, csr_values_h, nnz * sizeof(cuComplex),
                        cudaMemcpyHostToDevice), "cudaMemcpy for csr_values");
    CUDA_CALL_AND_CHECK(cudaMemcpy(b_values_d, b_values_h, nrhs * n * sizeof(cuComplex),
                        cudaMemcpyHostToDevice), "cudaMemcpy for b_values");

    /* Create a CUDA stream */
    cudaStream_t stream = NULL;
    CUDA_CALL_AND_CHECK(cudaStreamCreate(&stream), "cudaStreamCreate");

    /* Creating the cuDSS library handle */
    cudssHandle_t handle;

    CUDSS_CALL_AND_CHECK(cudssCreate(&handle), status, "cudssCreate");

    /* (optional) Setting the custom stream for the library handle */
    CUDSS_CALL_AND_CHECK(cudssSetStream(handle, stream), status, "cudssSetStream");

    /* Creating cuDSS solver configuration and data objects */
    cudssConfig_t solverConfig;
    cudssData_t solverData;

    CUDSS_CALL_AND_CHECK(cudssConfigCreate(&solverConfig), status, "cudssConfigCreate");
    CUDSS_CALL_AND_CHECK(cudssDataCreate(handle, &solverData), status, "cudssDataCreate");

    /* Create matrix objects for the right-hand side b and solution x (as dense matrices). */
    cudssMatrix_t x, b;

    /*
    因为解向量 x 和右端项 b 都是大小为 n×nrhs 的稠密矩阵（此例中 nrhs =
    1），这里把矩阵的行数和列数都设置为 n（5）。使用 int64_t 是为了满足 cuDSS
    接口对索引/尺寸使用 64 位整数的要求，以支持大规模问题。
    */
    int64_t nrows = n, ncols = n;
    /*
    ldb 和 ldx 是列主序稠密矩阵的 leading
    dimension（每列之间的内存跨度）。列主序下，一列有 nrows 个元素；而此处 nrows
    = ncols = n，所以两者都是 n。后续创建描述符时会用到这两个步长，让 cuDSS
    正确解释底层内存布局。
    */
    int ldb = ncols, ldx = nrows;
    /* 使用稠密矩阵描述符包裹裸指针，便于 cuDSS 统一访问。 */
    CUDSS_CALL_AND_CHECK(cudssMatrixCreateDn(&b, ncols, nrhs, ldb, b_values_d, CUDA_C_32F,
                         CUDSS_LAYOUT_COL_MAJOR), status, "cudssMatrixCreateDn for b");
    CUDSS_CALL_AND_CHECK(cudssMatrixCreateDn(&x, nrows, nrhs, ldx, x_values_d, CUDA_C_32F,
                         CUDSS_LAYOUT_COL_MAJOR), status, "cudssMatrixCreateDn for x");

    /* Create a matrix object for the sparse input matrix. */
    cudssMatrix_t A;
    cudssMatrixType_t mtype = CUDSS_MTYPE_SPD;        // 矩阵类型：对称正定
    cudssMatrixViewType_t mview = CUDSS_MVIEW_UPPER;  // 只存储上三角部分
    cudssIndexBase_t base = CUDSS_BASE_ZERO;          // 索引从 0 开始

    /*解释每个变量的含义:
    mtype: 指定矩阵 A 的类型，这里是对称正定矩阵（SPD），
           因为该类型矩阵可以使用更高效的 Cholesky 分解。
    mview: 指定矩阵 A 存储的部分，这里是上三角部分，
           因为对称矩阵的下三角部分与上三角部分是镜像关系，存储一半即可节省空间。
    base:  指定索引的起始位置，这里是从 0 开始，符合 C/C++ 语言的习惯。
    通过合理设置这些参数，cuDSS 能够正确理解矩阵 A 的结构和存储方式，
    并选择合适的算法进行求解。

    */
    CUDSS_CALL_AND_CHECK(cudssMatrixCreateCsr(&A, nrows, ncols, nnz, csr_offsets_d, NULL,
                         csr_columns_d, csr_values_d, CUDA_R_32I, CUDA_C_32F, mtype, mview,
                         base), status, "cudssMatrixCreateCsr");

    /* 符号分解阶段：分析稀疏模式并准备内部数据结构。 */
    CUDSS_CALL_AND_CHECK(cudssExecute(handle, CUDSS_PHASE_ANALYSIS, solverConfig, solverData,
                         A, x, b), status, "cudssExecute for analysis");

    /* 数值分解阶段：基于分析结果进行数值分解（SPD 矩阵对应 Cholesky）。 */
    CUDSS_CALL_AND_CHECK(cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, solverConfig,
                         solverData, A, x, b), status, "cudssExecute for factor");

    /* 求解阶段：利用分解结果求解 Ax = b，写回解向量 x。 */
    CUDSS_CALL_AND_CHECK(cudssExecute(handle, CUDSS_PHASE_SOLVE, solverConfig, solverData,
                         A, x, b), status, "cudssExecute for solve");

    /* 按照创建的逆序销毁矩阵描述符、数据对象以及库句柄，释放内部资源。 */
    CUDSS_CALL_AND_CHECK(cudssMatrixDestroy(A), status, "cudssMatrixDestroy for A");
    CUDSS_CALL_AND_CHECK(cudssMatrixDestroy(b), status, "cudssMatrixDestroy for b");
    CUDSS_CALL_AND_CHECK(cudssMatrixDestroy(x), status, "cudssMatrixDestroy for x");
    CUDSS_CALL_AND_CHECK(cudssDataDestroy(handle, solverData), status, "cudssDataDestroy");
    CUDSS_CALL_AND_CHECK(cudssConfigDestroy(solverConfig), status, "cudssConfigDestroy");
    CUDSS_CALL_AND_CHECK(cudssDestroy(handle), status, "cudssHandleDestroy");

    CUDA_CALL_AND_CHECK(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

    /* Print the solution and compare against the exact solution */
    CUDA_CALL_AND_CHECK(cudaMemcpy(x_values_h, x_values_d, nrhs * n * sizeof(cuComplex),
                        cudaMemcpyDeviceToHost), "cudaMemcpy for x_values");

    int passed = 1;
    for (int i = 0; i < n; i++) {
        printf("x[%d] = (%1.4f, %1.4f) expected (%1.4f, 0)\n", i,
               x_values_h[i].x, x_values_h[i].y, double(i+1));
        if (fabs(x_values_h[i].x - (i + 1)) + fabs(x_values_h[i].y) > 2.e-6)
            passed = 0;
    }

    /* Release the data allocated on the user side */

    CUDSS_EXAMPLE_FREE;

    if (status == CUDSS_STATUS_SUCCESS && passed) {
        printf("Example PASSED\n");
        return 0;
    } else {
        printf("Example FAILED\n");
        return -1;
    }
}