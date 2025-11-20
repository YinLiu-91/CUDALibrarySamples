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
#include <string.h>
#include <limits.h>
#include <stdint.h>
#include <math.h>
#include <assert.h>
#include <cuda_runtime.h>

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

#define CUDSS_EXAMPLE_FREE \
    do { \
        free(csr_offsets_h); \
        free(csr_columns_h); \
        free(csr_values_h); \
        free(x_values_h); \
        free(b_values_h); \
        cudaFree(csr_offsets_d); \
        cudaFree(csr_columns_d); \
        cudaFree(csr_values_d); \
        cudaFree(x_values_d); \
        cudaFree(b_values_d); \
    } while(0);

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

static bool parse_arg_value(const char *name, const char *text, int min_value, int *out_value) {
    char *end = NULL;
    long parsed = strtol(text, &end, 10);
    if (end == text || *end != '\0') {
        fprintf(stderr, "Error: %s expects an integer, got '%s'\n", name, text);
        return false;
    }
    if (parsed < min_value) {
        fprintf(stderr, "Error: %s must be >= %d\n", name, min_value);
        return false;
    }
    if (parsed > INT_MAX) {
        fprintf(stderr, "Error: %s exceeds INT_MAX\n", name);
        return false;
    }
    *out_value = (int)parsed;
    return true;
}

static int64_t compute_upper_band_nnz(int n, int bandwidth) {
    if (bandwidth < 0) {
        bandwidth = 0;
    }
    int64_t total = 0;
    for (int row = 0; row < n; ++row) {
        int max_col = row + bandwidth;
        if (max_col >= n) {
            max_col = n - 1;
        }
        total += (max_col - row + 1);
    }
    return total;
}


int main (int argc, char *argv[]) {
    printf("---------------------------------------------------------\n");
    printf("cuDSS example: solving a real linear system\n"
           "with a configurable banded SPD matrix\n");
    printf("---------------------------------------------------------\n");

    if (argc >= 2 && (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0)) {
            printf("Usage: %s [n] [bandwidth]\n", argv[0]);
            printf("       n         : matrix dimension (default 5)\n");
            printf("       bandwidth : number of super-diagonals to keep (default 2)\n");
        return 0;
    }

    cudaError_t cuda_error = cudaSuccess;
    cudssStatus_t status = CUDSS_STATUS_SUCCESS;

    int n = 5;
    int bandwidth = 2;
    int nrhs = 1;

    if (argc > 1) {
        if (!parse_arg_value("n", argv[1], 1, &n)) {
            return -1;
        }
    }
    if (argc > 2) {
        if (!parse_arg_value("bandwidth", argv[2], 0, &bandwidth)) {
            return -1;
        }
    }
    if (argc > 3) {
        printf("Warning: ignoring %d extra argument(s) starting with '%s'\n", argc - 3, argv[3]);
    }

    if (bandwidth >= n) {
        int clamped = (n > 0) ? (n - 1) : 0;
        if (bandwidth != clamped) {
            printf("Info: bandwidth %d truncated to %d for n=%d\n", bandwidth, clamped, n);
        }
        bandwidth = clamped;
    }

    int64_t max_band_nnz = compute_upper_band_nnz(n, bandwidth);
    if (max_band_nnz > INT_MAX) {
        fprintf(stderr, "Error: matrix too large (nnz=%lld exceeds INT_MAX)\n", (long long)max_band_nnz);
        return -1;
    }
    int nnz = (int)max_band_nnz;

    printf("Matrix configuration => n=%d, bandwidth=%d, stored nnz=%d\n", n, bandwidth, nnz);

    int *csr_offsets_h = NULL;
    int *csr_columns_h = NULL;
    double *csr_values_h = NULL;
    double *x_values_h = NULL, *b_values_h = NULL;

    int *csr_offsets_d = NULL;
    int *csr_columns_d = NULL;
    double *csr_values_d = NULL;
    double *x_values_d = NULL, *b_values_d = NULL;

    /* Allocate host memory for the sparse input matrix A,
       right-hand side x and solution b*/

    csr_offsets_h = (int*)malloc((n + 1) * sizeof(int));
    csr_columns_h = (int*)malloc(nnz * sizeof(int));
    csr_values_h = (double*)malloc(nnz * sizeof(double));
    x_values_h = (double*)malloc(nrhs * n * sizeof(double));
    b_values_h = (double*)malloc(nrhs * n * sizeof(double));

    if (!csr_offsets_h || ! csr_columns_h || !csr_values_h ||
        !x_values_h || !b_values_h) {
        printf("Error: host memory allocation failed\n");
        return -1;
    }

    /* Initialize host memory for A and b using a banded SPD matrix */
    for (int idx = 0; idx < nrhs * n; ++idx) {
        x_values_h[idx] = 0.0;
        b_values_h[idx] = 0.0;
    }

    const double diag_value = 2.0 * (bandwidth + 1);
    int values_written = 0;
    for (int row = 0; row < n; ++row) {
        csr_offsets_h[row] = values_written;
        csr_columns_h[values_written] = row;
        csr_values_h[values_written] = diag_value;
        ++values_written;

        for (int offset = 1; offset <= bandwidth; ++offset) {
            int col = row + offset;
            if (col >= n) {
                break;
            }
            csr_columns_h[values_written] = col;
            csr_values_h[values_written] = 1.0;
            ++values_written;
        }
    }
    csr_offsets_h[n] = values_written;
    assert(values_written == nnz);

    /* Construct RHS corresponding to the exact solution vector {1, 2, ..., n} */
    for (int row = 0; row < n; ++row) {
        for (int idx = csr_offsets_h[row]; idx < csr_offsets_h[row + 1]; ++idx) {
            int col = csr_columns_h[idx];
            double val = csr_values_h[idx];
            double x_col = double(col + 1);
            b_values_h[row] += val * x_col;
            if (col != row) {
                b_values_h[col] += val * double(row + 1);
            }
        }
    }

    /* Allocate device memory for A, x and b */
    CUDA_CALL_AND_CHECK(cudaMalloc(&csr_offsets_d, (n + 1) * sizeof(int)),
                        "cudaMalloc for csr_offsets");
    CUDA_CALL_AND_CHECK(cudaMalloc(&csr_columns_d, nnz * sizeof(int)),
                        "cudaMalloc for csr_columns");
    CUDA_CALL_AND_CHECK(cudaMalloc(&csr_values_d, nnz * sizeof(double)),
                        "cudaMalloc for csr_values");
    CUDA_CALL_AND_CHECK(cudaMalloc(&b_values_d, nrhs * n * sizeof(double)),
                        "cudaMalloc for b_values");
    CUDA_CALL_AND_CHECK(cudaMalloc(&x_values_d, nrhs * n * sizeof(double)),
                        "cudaMalloc for x_values");

    /* Copy host memory to device for A and b */
    CUDA_CALL_AND_CHECK(cudaMemcpy(csr_offsets_d, csr_offsets_h, (n + 1) * sizeof(int),
                        cudaMemcpyHostToDevice), "cudaMemcpy for csr_offsets");
    CUDA_CALL_AND_CHECK(cudaMemcpy(csr_columns_d, csr_columns_h, nnz * sizeof(int),
                        cudaMemcpyHostToDevice), "cudaMemcpy for csr_columns");
    CUDA_CALL_AND_CHECK(cudaMemcpy(csr_values_d, csr_values_h, nnz * sizeof(double),
                        cudaMemcpyHostToDevice), "cudaMemcpy for csr_values");
    CUDA_CALL_AND_CHECK(cudaMemcpy(b_values_d, b_values_h, nrhs * n * sizeof(double),
                        cudaMemcpyHostToDevice), "cudaMemcpy for b_values");

    /* Create a CUDA stream */
    cudaStream_t stream = NULL;
    CUDA_CALL_AND_CHECK(cudaStreamCreate(&stream), "cudaStreamCreate");

    /* Creating the cuDSS library handle */
    cudssHandle_t handle;

    CUDSS_CALL_AND_CHECK(cudssCreate(&handle), status, "cudssCreate");

    /* (optional) Setting the custom stream for the library handle */
    CUDSS_CALL_AND_CHECK(cudssSetStream(handle, stream), status, "cudssSetStream");

    /* Set the full name of the cuDSS threading layer library.
  Note: if threading_layer_libname = NULL then cudssSetThreadingLayer takes
  the threading layer library name from the environment variable
  "CUDSS_THREADING_LIB"*/
    bool use_MT = false;
    if (use_MT) {
#if USE_OPENMP
      CUDSS_CALL_AND_CHECK(cudssSetThreadingLayer(handle, NULL), status,
                           "cudssSetThreadingLayer");
#endif
    }

    /* Creating cuDSS solver configuration and data objects */
    cudssConfig_t solverConfig;
    cudssData_t solverData;

    CUDSS_CALL_AND_CHECK(cudssConfigCreate(&solverConfig), status, "cudssConfigCreate");
    CUDSS_CALL_AND_CHECK(cudssDataCreate(handle, &solverData), status, "cudssDataCreate");

    /* Create matrix objects for the right-hand side b and solution x (as dense matrices). */
    cudssMatrix_t x, b;

    int64_t nrows = n, ncols = n;
    int ldb = ncols, ldx = nrows;
    CUDSS_CALL_AND_CHECK(cudssMatrixCreateDn(&b, ncols, nrhs, ldb, b_values_d, CUDA_R_64F,
                         CUDSS_LAYOUT_COL_MAJOR), status, "cudssMatrixCreateDn for b");
    CUDSS_CALL_AND_CHECK(cudssMatrixCreateDn(&x, nrows, nrhs, ldx, x_values_d, CUDA_R_64F,
                         CUDSS_LAYOUT_COL_MAJOR), status, "cudssMatrixCreateDn for x");

    /* Create a matrix object for the sparse input matrix. */
    cudssMatrix_t A;
    cudssMatrixType_t mtype     = CUDSS_MTYPE_SPD;
    cudssMatrixViewType_t mview = CUDSS_MVIEW_UPPER;
    cudssIndexBase_t base       = CUDSS_BASE_ZERO;
    CUDSS_CALL_AND_CHECK(cudssMatrixCreateCsr(&A, nrows, ncols, nnz, csr_offsets_d, NULL,
                         csr_columns_d, csr_values_d, CUDA_R_32I, CUDA_R_64F, mtype, mview,
                         base), status, "cudssMatrixCreateCsr");

    /* Symbolic factorization */
    CUDSS_CALL_AND_CHECK(cudssExecute(handle, CUDSS_PHASE_ANALYSIS, solverConfig, solverData,
                         A, x, b), status, "cudssExecute for analysis");

    /* Factorization */
    CUDSS_CALL_AND_CHECK(cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, solverConfig,
                         solverData, A, x, b), status, "cudssExecute for factor");

    /* Solving */
    CUDSS_CALL_AND_CHECK(cudssExecute(handle, CUDSS_PHASE_SOLVE, solverConfig, solverData,
                         A, x, b), status, "cudssExecute for solve");

    /* Destroying opaque objects, matrix wrappers and the cuDSS library handle */
    CUDSS_CALL_AND_CHECK(cudssMatrixDestroy(A), status, "cudssMatrixDestroy for A");
    CUDSS_CALL_AND_CHECK(cudssMatrixDestroy(b), status, "cudssMatrixDestroy for b");
    CUDSS_CALL_AND_CHECK(cudssMatrixDestroy(x), status, "cudssMatrixDestroy for x");
    CUDSS_CALL_AND_CHECK(cudssDataDestroy(handle, solverData), status, "cudssDataDestroy");
    CUDSS_CALL_AND_CHECK(cudssConfigDestroy(solverConfig), status, "cudssConfigDestroy");
    CUDSS_CALL_AND_CHECK(cudssDestroy(handle), status, "cudssHandleDestroy");

    CUDA_CALL_AND_CHECK(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

    /* Print the solution and compare against the exact solution */
    CUDA_CALL_AND_CHECK(cudaMemcpy(x_values_h, x_values_d, nrhs * n * sizeof(double),
                        cudaMemcpyDeviceToHost), "cudaMemcpy for x_values");

    const double kSolutionTolerance = 1e-9;
    int passed = 1;
    for (int i = 0; i < n; i++) {
        printf("x[%d] = %1.4f expected %1.4f\n", i, x_values_h[i], double(i+1));
        if (fabs(x_values_h[i] - (i + 1)) > kSolutionTolerance)
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