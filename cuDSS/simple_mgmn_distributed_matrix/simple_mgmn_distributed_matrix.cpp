/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <math.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

#include "cudss.h"
#include "../simple_matrix_market/matrix_market_reader.h"

#ifdef USE_MPI
#  include "mpi.h"
#  ifdef USE_NCCL
#    include "nccl.h"
#  endif
#  if !defined(USE_OPENMPI) && !defined(USE_NCCL)
#    error "With USE_MPI, either USE_OPENMPI or USE_NCCL must be defined"
#  endif
#  if defined(USE_OPENMPI) && defined(USE_NCCL)
#    error "With USE_MPI, exactly one of USE_OPENMPI and USE_NCCL must be defined"
#  endif
#else
#  error "This example needs to be compiled with MPI (and optionally with NCCL as well)"
#endif

/*
    This example demonstrates usage of MGMN mode in cuDSS for solving
    a system of linear algebraic equations with a sparse matrix:
                                Ax = b,
    where:
        A is the sparse input matrix distributed among 2 mpi ranks,
        b is the (dense) right-hand side vector (or a matrix),
        x is the (dense) solution vector (or a matrix).
    Note: in this example A, b and x are distributed between 2 ranks,
    the rest of the processes have empty input. However all ranks are assumed
    to have correct global matrix shapes (n, nnz, nrhs).

    Note: The MGMN mode is intended to be used for solving large systems.
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
            return -3; \
        } \
    } while(0);

#ifdef USE_MPI
#  define MPI_CALL_AND_CHECK(call, msg) \
    do { \
        mpi_error = call; \
        if (mpi_error != 0) { \
            printf("Example FAILED: MPI call returned error = %d, details: " #msg "\n", mpi_error); \
            return -4; \
        } \
    } while(0);
#  ifdef USE_NCCL
#    define NCCL_CALL_AND_CHECK(call, msg) \
    do { \
        nccl_result = call; \
        if (nccl_result != ncclSuccess) { \
            printf("Example FAILED: NCCL call returned error = %d, details: " #msg "\n", nccl_result); \
            return -5; \
        } \
    } while(0);
#  endif
#endif


#define CUDSS_CALL_AND_CHECK(call, status, msg) \
    do { \
        status = call; \
        if (status != CUDSS_STATUS_SUCCESS) { \
            printf("Example FAILED: CUDSS call ended unsuccessfully with status = %d, details: " #msg "\n", status); \
            if (rank == 0 || rank == 1) { \
                CUDSS_EXAMPLE_FREE; \
            } \
            return -6; \
        } \
    } while(0);

// Helper function to extract local CSR data for a rank from global matrix
void extract_local_csr(int n_global, int *csr_offsets_global, int *csr_columns_global, 
                       double *csr_values_global, int row_start, int row_end,
                       int **csr_offsets_local, int **csr_columns_local, 
                       double **csr_values_local, int *local_nnz) {
    int local_n = row_end - row_start + 1;
    if (local_n <= 0) {
        *csr_offsets_local = NULL;
        *csr_columns_local = NULL;
        *csr_values_local = NULL;
        *local_nnz = 0;
        return;
    }
    
    // Count local nnz
    *local_nnz = 0;
    for (int i = row_start; i <= row_end; i++) {
        *local_nnz += csr_offsets_global[i + 1] - csr_offsets_global[i];
    }
    
    // Allocate local memory
    *csr_offsets_local = (int*)malloc((local_n + 1) * sizeof(int));
    *csr_columns_local = (int*)malloc(*local_nnz * sizeof(int));
    *csr_values_local = (double*)malloc(*local_nnz * sizeof(double));
    
    // Extract local data
    int local_idx = 0;
    (*csr_offsets_local)[0] = 0;
    for (int i = 0; i < local_n; i++) {
        int global_row = row_start + i;
        int row_start_idx = csr_offsets_global[global_row];
        int row_end_idx = csr_offsets_global[global_row + 1];
        int row_nnz = row_end_idx - row_start_idx;
        
        for (int j = 0; j < row_nnz; j++) {
            (*csr_columns_local)[local_idx] = csr_columns_global[row_start_idx + j];
            (*csr_values_local)[local_idx] = csr_values_global[row_start_idx + j];
            local_idx++;
        }
        (*csr_offsets_local)[i + 1] = local_idx;
    }
}

/**
 * 0. 需要支持cuda aware的openmpi，apt-get安装的openmpi不支持cuda
 * aware，需要单独编译
 * 1. cmake 命令:  cmake ../ -DOPENMPI_PATH=/home/cudss/cudaa-mpi-install
 * -DNCCL_PATH=/usr/lib/x86_64-linux-gnu
 * 2. 执行命令openmpi: UDA_VISIBLE_DEVICES=0,1 mpirun -np 2 --allow-run-as-root
 * ./simple_mgmn_distributed_matrix_example_openmpi  openmpi
 * /usr/lib/x86_64-linux-gnu/libcudss_commlayer_openmpi.so general full
 * /home/AMGX/build/p9_100_10000.mtx
 * 3. nccl还不支持，可能由于没有nvlink？
 */
int main (int argc, char *argv[]) {
    cudaError_t   cuda_error = cudaSuccess;
    cudssStatus_t status     = CUDSS_STATUS_SUCCESS;
#ifdef USE_MPI
    int           mpi_error  = 0;
#  ifdef USE_NCCL
    ncclResult_t  nccl_result = ncclSuccess;
#  endif
#endif

    /* Initializing the communication backend
       Note: cuDSS can work with any CUDA-aware communication backend through the
       user-defined communication layers (see the documentation), but this example
       demonstrates the mode using pre-built communication layers for OpenMPI and
       NCCL.
       Therefore, as both of them rely on MPI_Init/MPI_Finalize, we call it here.
       For a different communication backend, a different initialization/cleanup
       APIs might be needed. */
#ifdef USE_MPI
    MPI_CALL_AND_CHECK(MPI_Init(&argc, &argv), "MPI_Init");
#endif

    /* Identifying the root process as the one with rank equal to 0 */
    int rank = 0, size = 1;
#ifdef USE_MPI
    MPI_CALL_AND_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "MPI_Comm_rank");
    MPI_CALL_AND_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &size), "MPI_Comm_size");
#endif
    if (rank == 0) {
        printf("---------------------------------------------------------\n");
        printf("cuDSS example: this example will be run with %d processes\n", size);
        printf("Note: number of processes must not exceed the number of\n"
               "GPU devices available\n");
        printf("---------------------------------------------------------\n");
        printf("cuDSS example: solving a real linear system\n"
               "with a distributed symmetric positive-definite matrix using \n"
               "the MGMN mode.\n");
        printf("Matrix will be read from file and distributed among ranks.\n");
        printf("---------------------------------------------------------\n");
        fflush(0);
    }

    /* Binding each process to a specific GPU device under the assumption
       that the number of processes does not exceed the number of devices */
    int device_count = 0;
    CUDA_CALL_AND_CHECK(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count == 0) {
        printf("Error: no GPU devices have been found\n");
        fflush(0);
#ifdef USE_MPI
        MPI_Finalize();
#endif
        return -1;
    }
    int device_id = rank % device_count;
    CUDA_CALL_AND_CHECK(cudaSetDevice(device_id), "cudaSetDevice");

    /*timers*/
    cudaEvent_t start, stop;
    float time_ms = 0.0f, total_ms = 0.0f;
    // Create CUDA events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    /* Parsing the communication layer information from the input parameters */
    char comm_backend_name[1024];
    char comm_layer_libname[1024];
    const char *matrix_filename = nullptr;
    const char *vector_filename = nullptr;

    if (argc < 6) {
      if (rank == 0) {
        printf("Error: this example requires at least 6 arguments:\n"
               "1. the communication backend name (openmpi or nccl)\n"
               "2. the communication layer library (full name with the path, "
               "or empty string to use CUDSS_COMM_LIB)\n"
               "3. the matrix type (general|symmetric|hermitian|spd|hpd)\n"
               "4. the matrix view (full|lower|upper)\n"
               "5. the matrix filename (Matrix Market format)\n"
               "6. (optional) the right-hand side vector filename (Matrix "
               "Market format)\n"
               "Note: backend should match the communication layer library\n");
        fflush(0);
      }
#ifdef USE_MPI
        MPI_Finalize();
#endif
        return -2;
    }

    strncpy(comm_backend_name, argv[1], sizeof(comm_backend_name));
    if (rank == 0) printf("Communication backend name is: %s\n", comm_backend_name);
    
    if (strlen(argv[2]) > 0) {
        strncpy(comm_layer_libname, argv[2], sizeof(comm_layer_libname));
        if (rank == 0) printf("Communication layer library name is: %s\n", comm_layer_libname);
    } else {
        if (rank == 0)
            printf("Since communication layer library name was not provided as input \n"
                    "argument for the executable, the layer library name will be \n"
                    "taken from CUDSS_COMM_LIB environment variable\n");
    }

    // Parse matrix type
    cudssMatrixType_t mtype;
    if (strcmp(argv[3], "general") == 0)
      mtype = CUDSS_MTYPE_GENERAL;
    else if (strcmp(argv[3], "symmetric") == 0)
      mtype = CUDSS_MTYPE_SYMMETRIC;
    else if (strcmp(argv[3], "hermitian") == 0)
      mtype = CUDSS_MTYPE_HERMITIAN;
    else if (strcmp(argv[3], "spd") == 0)
      mtype = CUDSS_MTYPE_SPD;
    else if (strcmp(argv[3], "hpd") == 0)
      mtype = CUDSS_MTYPE_HPD;
    else {
      if (rank == 0) {
        fprintf(stderr,
                "Error: Invalid matrix type '%s'. Supported types are general, "
                "symmetric, hermitian, spd, hpd\n",
                argv[3]);
      }
#ifdef USE_MPI
      MPI_Finalize();
#endif
      return EXIT_FAILURE;
    }

    // Parse matrix view
    cudssMatrixViewType_t mview;
    if (strcmp(argv[4], "full") == 0)
      mview = CUDSS_MVIEW_FULL;
    else if (strcmp(argv[4], "lower") == 0)
      mview = CUDSS_MVIEW_LOWER;
    else if (strcmp(argv[4], "upper") == 0)
      mview = CUDSS_MVIEW_UPPER;
    else {
      if (rank == 0) {
        fprintf(
            stderr,
            "Error: Invalid matrix view type '%s'. Supported types are: full, "
            "lower, upper.\n",
            argv[4]);
      }
#ifdef USE_MPI
      MPI_Finalize();
#endif
      return EXIT_FAILURE;
    }

    if ((mview != CUDSS_MVIEW_FULL) && (mtype == CUDSS_MTYPE_GENERAL)) {
      if (rank == 0) {
        fprintf(
            stderr,
            "Error: you chose a lower/upper view of the matrix but you also "
            "specified that it is general (not symmetric).\n"
            "If your matrix file is truly non-symmetric, half of the "
            "elements would not be used, as the lower/upper part of the\n"
            "matrix will be mirrored and you will be solving for the wrong "
            "matrix. Or the reader will throw an error.\n");
      }
#ifdef USE_MPI
      MPI_Finalize();
#endif
      return EXIT_FAILURE;
    }

    matrix_filename = argv[5];
    if (argc > 6) {
      vector_filename = argv[6];
    }
    if (rank == 0) {
      printf("Matrix type: %s\n", argv[3]);
      printf("Matrix view: %s\n", argv[4]);
      printf("Matrix filename: %s\n", matrix_filename);
      if (vector_filename) {
        printf("Vector filename: %s\n", vector_filename);
      } else {
        printf("No vector filename provided, will use default values\n");
      }
    }
    fflush(0);


    /* Creating a communicator of the type matching the communication backend */
#ifdef USE_MPI
    MPI_Comm *mpi_comm = NULL;

    #if USE_OPENMPI
    if (strcmp(comm_backend_name,"openmpi") == 0) {
        mpi_comm = (MPI_Comm*) malloc(sizeof(MPI_Comm));
        mpi_comm[0] = MPI_COMM_WORLD;
    }
    #endif
    #if USE_NCCL
    ncclComm_t *nccl_comm = NULL;
    if (strcmp(comm_backend_name,"nccl") == 0) {
        nccl_comm = (ncclComm_t*) malloc(sizeof(ncclComm_t));
        if (nccl_comm == NULL) {
          printf("Error: failed to allocate memory for nccl_comm\n");
          fflush(0);
          return -2;
        }
        ncclUniqueId id;
        if (rank == 0) {
            NCCL_CALL_AND_CHECK(ncclGetUniqueId(&id), "ncclGetUniqueId");
        }
        MPI_CALL_AND_CHECK(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD),
                           "MPI_Bcast for id");
        NCCL_CALL_AND_CHECK(ncclCommInitRank(nccl_comm, size, id, rank),
                            "ncclCommInitRank");
    }
    #endif
#endif

    int n = 0;
    int nnz = 0;
    int nrhs = 1;

    // Global matrix data (only on rank 0)
    int *csr_offsets_global = NULL;
    int *csr_columns_global = NULL;
    double *csr_values_global = NULL;
    double *b_values_global = NULL;

    // Local matrix data
    int *csr_offsets_h = NULL;
    int *csr_columns_h = NULL;
    double *csr_values_h = NULL;
    double *x_values_h = NULL, *b_values_h = NULL;

    int *csr_offsets_d = NULL;
    int *csr_columns_d = NULL;
    double *csr_values_d = NULL;
    double *x_values_d = NULL, *b_values_d = NULL;

    int matrix_row_start = 0;
    int matrix_row_end   = 0;
    int b_row_start = 0;
    int b_row_end   = 0;

    /* Read input matrix from file on rank 0 */
    if (rank == 0) {
        printf("--- Reading matrix from file ---\n");
        cudaEventRecord(start);
        int failed = matrix_reader(matrix_filename, n, nnz, &csr_offsets_global,
                                   &csr_columns_global, &csr_values_global, mview);
        if (failed) {
            fprintf(stderr, "Reader failed.\n");
#ifdef USE_MPI
            MPI_Finalize();
#endif
            return EXIT_FAILURE;
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time_ms, start, stop);
        printf("Matrix reading time: %.4f ms\n", time_ms);
        total_ms += time_ms;
        printf("Matrix size: %dx%d, nnz: %d\n", n, n, nnz);
        fflush(0);
    }

    /* Broadcast global matrix size to all ranks */
#ifdef USE_MPI
    MPI_CALL_AND_CHECK(MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD), "MPI_Bcast for n");
    MPI_CALL_AND_CHECK(MPI_Bcast(&nnz, 1, MPI_INT, 0, MPI_COMM_WORLD), "MPI_Bcast for nnz");
#endif

    //Note that distribution of the solution can be different from the matrix
    //    and right-hand side
    //In this example we set full overlapping for the solution vector that means
    //    all ranks will have full results
    int x_row_start = 0;
    int x_row_end   = n - 1;
    x_values_h = (double*)malloc(nrhs * n * sizeof(double));
    if (!x_values_h) {
        printf("Error: host memory allocation failed\n");fflush(0);
        return -2;
    }
    /* Allocate device memory for x */
    CUDA_CALL_AND_CHECK(cudaMalloc(&x_values_d, nrhs * n * sizeof(double)),
        "cudaMalloc for x_values");

    /* Distribute matrix rows among ranks */
    if (rank == 0) {
        //Matrix row start and end should be always in 0-based
        // Simple distribution: rank 0 gets first half, rank 1 gets second half (with overlap)
        matrix_row_start = 0;
        matrix_row_end   = size == 1 ? n - 1 : (n / 2);

        //Note that distribution of right-hand can be different from the matrix.
        b_row_start = 0;
        b_row_end   = size == 1 ? n - 1 : (n / 2);
        int local_b_n = b_row_end - b_row_start + 1;

        int local_n = matrix_row_end - matrix_row_start + 1;
        int local_nnz = 0;

        /* Extract local CSR data from global matrix */
        extract_local_csr(n, csr_offsets_global, csr_columns_global, csr_values_global,
                         matrix_row_start, matrix_row_end,
                         &csr_offsets_h, &csr_columns_h, &csr_values_h, &local_nnz);

        if (!csr_offsets_h || !csr_columns_h || !csr_values_h) {
            printf("Error: host memory allocation failed\n");fflush(0);
            return -2;
        }

        /* Read or allocate right-hand side b */
        if (vector_filename != nullptr) {
            int failed = rhs_reader(vector_filename, n, &b_values_global);
            if (failed) {
                fprintf(stderr, "RHS reader failed.\n");
#ifdef USE_MPI
                MPI_Finalize();
#endif
                return EXIT_FAILURE;
            }
            // Extract local b values
            b_values_h = (double*)malloc(nrhs * local_b_n * sizeof(double));
            if (!b_values_h) {
                printf("Error: host memory allocation failed\n");fflush(0);
                return -2;
            }
            for (int i = 0; i < local_b_n; i++) {
                b_values_h[i] = b_values_global[b_row_start + i];
            }
        } else {
            /* Allocate host memory for the right-hand side b and fill with 1.0 */
            b_values_h = (double*)malloc(nrhs * local_b_n * sizeof(double));
            if (!b_values_h) {
                printf("Error: host memory allocation failed\n");fflush(0);
                return -2;
            }
            for (int i = 0; i < local_b_n; i++) {
                b_values_h[i] = 1.0;
            }
        }

        /* Allocate device memory for A */
        CUDA_CALL_AND_CHECK(cudaMalloc(&csr_offsets_d, (local_n + 1) * sizeof(int)),
                            "cudaMalloc for csr_offsets");
        CUDA_CALL_AND_CHECK(cudaMalloc(&csr_columns_d, local_nnz * sizeof(int)),
                            "cudaMalloc for csr_columns");
        CUDA_CALL_AND_CHECK(cudaMalloc(&csr_values_d, local_nnz * sizeof(double)),
                            "cudaMalloc for csr_values");
        /* Allocate device memory for b */
        CUDA_CALL_AND_CHECK(cudaMalloc(&b_values_d, nrhs * local_b_n * sizeof(double)),
                            "cudaMalloc for b_values");

        /* Copy host memory to device for A*/
        CUDA_CALL_AND_CHECK(cudaMemcpy(csr_offsets_d, csr_offsets_h, (local_n + 1) * sizeof(int),
                            cudaMemcpyHostToDevice), "cudaMemcpy for csr_offsets");
        CUDA_CALL_AND_CHECK(cudaMemcpy(csr_columns_d, csr_columns_h, local_nnz * sizeof(int),
                            cudaMemcpyHostToDevice), "cudaMemcpy for csr_columns");
        CUDA_CALL_AND_CHECK(cudaMemcpy(csr_values_d, csr_values_h, local_nnz * sizeof(double),
                            cudaMemcpyHostToDevice), "cudaMemcpy for csr_values");
        /* Copy host memory to device for b */
        CUDA_CALL_AND_CHECK(cudaMemcpy(b_values_d, b_values_h,
            nrhs * local_b_n * sizeof(double), cudaMemcpyHostToDevice),
            "cudaMemcpy for b_values");

        /* Send global matrix data to rank 1 if needed */
        if (size > 1) {
#ifdef USE_MPI
            MPI_CALL_AND_CHECK(MPI_Send(csr_offsets_global, n + 1, MPI_INT, 1, 0, MPI_COMM_WORLD),
                               "MPI_Send for csr_offsets_global");
            MPI_CALL_AND_CHECK(MPI_Send(csr_columns_global, nnz, MPI_INT, 1, 1, MPI_COMM_WORLD),
                               "MPI_Send for csr_columns_global");
            MPI_CALL_AND_CHECK(MPI_Send(csr_values_global, nnz, MPI_DOUBLE, 1, 2, MPI_COMM_WORLD),
                               "MPI_Send for csr_values_global");
            if (vector_filename != nullptr && b_values_global != NULL) {
                MPI_CALL_AND_CHECK(MPI_Send(b_values_global, n, MPI_DOUBLE, 1, 3, MPI_COMM_WORLD),
                                   "MPI_Send for b_values_global");
            }
#endif
        }

        /* Free global matrix data after sending */
        if (csr_offsets_global) free(csr_offsets_global);
        if (csr_columns_global) free(csr_columns_global);
        if (csr_values_global) free(csr_values_global);
        if (b_values_global) free(b_values_global);
    } else if (rank == 1 && size > 1) {
        //Matrix row start and end should always be 0-based
        // Rank 1 gets second half (with overlap from rank 0)
        matrix_row_start = (n / 2);
        matrix_row_end   = n - 1;
        //Note that distribution of right-hand can be different from the matrix.
        b_row_start = (n / 2);
        b_row_end   = n - 1;
        int local_b_n = b_row_end - b_row_start + 1;

        int local_n = matrix_row_end - matrix_row_start + 1;
        int local_nnz = 0;

        /* Receive global matrix data from rank 0 */
        // First, receive the global CSR arrays
        csr_offsets_global = (int*)malloc((n + 1) * sizeof(int));
        csr_columns_global = (int*)malloc(nnz * sizeof(int));
        csr_values_global = (double*)malloc(nnz * sizeof(double));
#ifdef USE_MPI
        MPI_CALL_AND_CHECK(MPI_Recv(csr_offsets_global, n + 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE),
                           "MPI_Recv for csr_offsets_global");
        MPI_CALL_AND_CHECK(MPI_Recv(csr_columns_global, nnz, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE),
                           "MPI_Recv for csr_columns_global");
        MPI_CALL_AND_CHECK(MPI_Recv(csr_values_global, nnz, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE),
                           "MPI_Recv for csr_values_global");
#endif

        /* Extract local CSR data from global matrix */
        extract_local_csr(n, csr_offsets_global, csr_columns_global, csr_values_global,
                         matrix_row_start, matrix_row_end,
                         &csr_offsets_h, &csr_columns_h, &csr_values_h, &local_nnz);

        if (!csr_offsets_h || !csr_columns_h || !csr_values_h) {
            printf("Error: host memory allocation failed\n");fflush(0);
            return -2;
        }

        /* Receive or allocate right-hand side b */
        if (vector_filename != nullptr) {
            // Receive global b from rank 0
            b_values_global = (double*)malloc(n * sizeof(double));
#ifdef USE_MPI
            MPI_CALL_AND_CHECK(MPI_Recv(b_values_global, n, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE),
                               "MPI_Recv for b_values_global");
#endif
            // Extract local b values
            b_values_h = (double*)malloc(nrhs * local_b_n * sizeof(double));
            if (!b_values_h) {
                printf("Error: host memory allocation failed\n");fflush(0);
                return -2;
            }
            for (int i = 0; i < local_b_n; i++) {
                b_values_h[i] = b_values_global[b_row_start + i];
            }
        } else {
            /* Allocate host memory for the right-hand side b and fill with 1.0 */
            b_values_h = (double*)malloc(nrhs * local_b_n * sizeof(double));
            if (!b_values_h) {
                printf("Error: host memory allocation failed\n");fflush(0);
                return -2;
            }
            for (int i = 0; i < local_b_n; i++) {
                b_values_h[i] = 1.0;
            }
        }

        /* Allocate device memory for A */
        CUDA_CALL_AND_CHECK(cudaMalloc(&csr_offsets_d, (local_n + 1) * sizeof(int)),
                            "cudaMalloc for csr_offsets");
        CUDA_CALL_AND_CHECK(cudaMalloc(&csr_columns_d, local_nnz * sizeof(int)),
                            "cudaMalloc for csr_columns");
        CUDA_CALL_AND_CHECK(cudaMalloc(&csr_values_d, local_nnz * sizeof(double)),
                            "cudaMalloc for csr_values");
        /* Allocate device memory for b */
        CUDA_CALL_AND_CHECK(cudaMalloc(&b_values_d, nrhs * local_b_n * sizeof(double)),
                            "cudaMalloc for b_values");

        /* Copy host memory to device for A*/
        CUDA_CALL_AND_CHECK(cudaMemcpy(csr_offsets_d, csr_offsets_h, (local_n + 1) * sizeof(int),
                            cudaMemcpyHostToDevice), "cudaMemcpy for csr_offsets");
        CUDA_CALL_AND_CHECK(cudaMemcpy(csr_columns_d, csr_columns_h, local_nnz * sizeof(int),
                            cudaMemcpyHostToDevice), "cudaMemcpy for csr_columns");
        CUDA_CALL_AND_CHECK(cudaMemcpy(csr_values_d, csr_values_h, local_nnz * sizeof(double),
                            cudaMemcpyHostToDevice), "cudaMemcpy for csr_values");
        /* Copy host memory to device for b */
        CUDA_CALL_AND_CHECK(cudaMemcpy(b_values_d, b_values_h,
            nrhs * local_b_n * sizeof(double), cudaMemcpyHostToDevice),
            "cudaMemcpy for b_values");

        /* Free global matrix data after extraction */
        if (csr_offsets_global) free(csr_offsets_global);
        if (csr_columns_global) free(csr_columns_global);
        if (csr_values_global) free(csr_values_global);
        if (b_values_global) free(b_values_global);
    } else {
        //Ranks > 1 in this example do not have a piece of the input matrix and.
        //    right-hand side. Thus setting any pair of numbers such that
        //    matrix_row_start > matrix_row_end will do.
        matrix_row_start = 1;
        matrix_row_end   = 0;
        b_row_start = 1;
        b_row_end   = 0;
        // For empty ranks, pointers can be NULL as per original example
    }

    /* Create a CUDA stream */
    cudaStream_t stream = NULL;
    CUDA_CALL_AND_CHECK(cudaStreamCreate(&stream), "cudaStreamCreate");

    /* Creating the cuDSS library handle */
    cudssHandle_t handle;

    CUDSS_CALL_AND_CHECK(cudssCreate(&handle), status, "cudssCreate");

    /* Set the full name of the cuDSS communication layer library.
      Note: if comm_layer_libname = NULL then cudssSetCommLayer takes
      the communication layer library name from the environment variable
      "CUDSS_COMM_LIB" */
    cudssSetCommLayer(
        handle, strlen(comm_layer_libname) > 0 ? comm_layer_libname : NULL);

    /* (optional) Setting the custom stream for the library handle */
    CUDSS_CALL_AND_CHECK(cudssSetStream(handle, stream), status, "cudssSetStream");

    /* Creating cuDSS solver configuration and data objects */
    cudssConfig_t solverConfig;
    cudssData_t solverData;

    CUDSS_CALL_AND_CHECK(cudssConfigCreate(&solverConfig), status, "cudssConfigCreate");
    CUDSS_CALL_AND_CHECK(cudssDataCreate(handle, &solverData), status, "cudssDataCreate");

    /* Create matrix objects for the right-hand side b and solution x (as dense matrices).
       Note: currently, solution and right-hand side arrays must be fully present only on
       the root process (rank = 0). */
    cudssMatrix_t x, b;

    int64_t nrows = n, ncols = n;
    int ldb = ncols, ldx = nrows;
    CUDSS_CALL_AND_CHECK(cudssMatrixCreateDn(&b, ncols, nrhs, ldb, b_values_d, CUDA_R_64F,
                         CUDSS_LAYOUT_COL_MAJOR), status, "cudssMatrixCreateDn for b");
    /* Set row-wise 1D distribution for  X */
    CUDSS_CALL_AND_CHECK(cudssMatrixSetDistributionRow1d(b,
        b_row_start, b_row_end), status, "cudssMatrixSetDistributionRow1d");

    CUDSS_CALL_AND_CHECK(cudssMatrixCreateDn(&x, nrows, nrhs, ldx, x_values_d, CUDA_R_64F,
                         CUDSS_LAYOUT_COL_MAJOR), status, "cudssMatrixCreateDn for x");
    /* Set row-wise 1D distribution for  X */
    CUDSS_CALL_AND_CHECK(cudssMatrixSetDistributionRow1d(x,
        x_row_start, x_row_end), status, "cudssMatrixSetDistributionRow1d");

    /* Create a matrix object for the sparse input matrix.
       Note: matrix A is distributed between rank = 0 and rank = 1,
       and the rest of the processes should have correct shape of the matrix
       and matrix_row_start > matrix_row_end (and it is
       fine to have NULL pointers for the data arrays). */
    cudssMatrix_t A;
    cudssIndexBase_t base = CUDSS_BASE_ZERO;
    
    if (rank == 0) {
        printf("--- Start solving--- \n");
    }
    cudaEventRecord(start);
    CUDSS_CALL_AND_CHECK(cudssMatrixCreateCsr(&A, nrows, ncols, nnz, csr_offsets_d,
        NULL, csr_columns_d, csr_values_d, CUDA_R_32I, CUDA_R_64F, mtype, mview,
        base), status, "cudssMatrixCreateCsr");
    /* Set row-wise 1D distribution for matrix A */
    CUDSS_CALL_AND_CHECK(cudssMatrixSetDistributionRow1d(A,
        matrix_row_start, matrix_row_end), status, "cudssMatrixSetDistributionRow1d");
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    if (rank == 0) {
        printf("cuDSS MatrixCreateCsr time: %.4f ms\n", time_ms);
        total_ms += time_ms;
    }

    /* Setting communicator to be used by MGMN mode of cuDSS */
#ifdef USE_MPI
    #if USE_OPENMPI
    if (strcmp(comm_backend_name, "openmpi") == 0) {
      CUDSS_CALL_AND_CHECK(cudssDataSet(handle, solverData, CUDSS_DATA_COMM,
                                        mpi_comm, sizeof(MPI_Comm *)),
                           status, "cudssDataSet for OpenMPI comm");
    }
#endif
#if USE_NCCL
    if (strcmp(comm_backend_name,"nccl") == 0) {
      if (nccl_comm == NULL) {
        printf("RANK %d: Error: nccl_comm is NULL before setting\n", rank);
        fflush(0);
        return -2;
      }
      CUDSS_CALL_AND_CHECK(cudssDataSet(handle, solverData, CUDSS_DATA_COMM,
                                        nccl_comm, sizeof(ncclComm_t *)),
                           status, "cudssDataSet for NCCL comm");
    }
#endif
#endif

    /* Symbolic factorization */
    cudaEventRecord(start);
    CUDSS_CALL_AND_CHECK(cudssExecute(handle, CUDSS_PHASE_ANALYSIS, solverConfig, solverData,
                         A, x, b), status, "cudssExecute for analysis");

    // Query CUDSS_DATA_HYBRID_DEVICE_MEMORY_MIN for current device
    int current_device = 0;
    cudaGetDevice(&current_device);
    int64_t hybrid_device_memory_limit = 0;
    size_t sizeWritten;
    CUDSS_CALL_AND_CHECK(
        cudssDataGet(handle, solverData, CUDSS_DATA_HYBRID_DEVICE_MEMORY_MIN,
                     &hybrid_device_memory_limit,
                     sizeof(hybrid_device_memory_limit), &sizeWritten),
        status, "cudssDataGet for the memory estimates");
    printf(
        "RANK %d: dev_id = %d CUDSS_DATA_HYBRID_DEVICE_MEMORY_MIN %ld bytes\n",
        rank, current_device, hybrid_device_memory_limit);

    int64_t memory_estimates[16] = {0};
    CUDSS_CALL_AND_CHECK(
        cudssDataGet(handle, solverData, CUDSS_DATA_MEMORY_ESTIMATES,
                     &memory_estimates, sizeof(memory_estimates), &sizeWritten),
        status, "cudssDataGet for the memory estimates");
    printf("RANK %d: memory estimates: device: %lf (stable) %lf (peak) MB\n",
           rank, 1.0 * memory_estimates[0] / 1024 / 1024,
           1.0 * memory_estimates[1] / 1024 / 1024);
    printf("RANK %d: memory estimates: host: %lf (stable) %lf (peak) MB\n",
           rank, 1.0 * memory_estimates[2] / 1024 / 1024,
           1.0 * memory_estimates[3] / 1024 / 1024);
    printf("RANK %d: memory estimates: hybrid peak: %lf MB = (GPU) %lf MB + "
           "(CPU) %lf MB\n",
           rank, 1.0 * memory_estimates[4] / 1024 / 1024,
           1.0 * memory_estimates[5] / 1024 / 1024,
           1.0 * (memory_estimates[4] - memory_estimates[5]) / 1024 / 1024);
    fflush(0);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    double all_3phase_time = 0.0;
    if (rank == 0) {
      printf("cuDSS Analysis time: %.4f ms\n", time_ms);
      total_ms += time_ms;
      all_3phase_time += time_ms;
    }

    /* Factorization */
    cudaEventRecord(start);
    CUDSS_CALL_AND_CHECK(cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, solverConfig,
                         solverData, A, x, b), status, "cudssExecute for factor");
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    if (rank == 0) {
        printf("cuDSS Factorization time: %.4f ms\n", time_ms);
        total_ms += time_ms;
        all_3phase_time += time_ms;
    }

    /* Solving */
    cudaEventRecord(start);
    CUDSS_CALL_AND_CHECK(cudssExecute(handle, CUDSS_PHASE_SOLVE, solverConfig, solverData,
                         A, x, b), status, "cudssExecute for solve");
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    if (rank == 0) {
        printf("cuDSS Solve time: %.4f ms\n", time_ms);
        total_ms += time_ms;
        // printf("program Total time: %.4f ms\n", total_ms);
        all_3phase_time += time_ms;
        printf("cuDSS Total time: %.4f ms\n", all_3phase_time);
    }

    /* Destroying opaque objects, matrix wrappers and the cuDSS library handle */
    CUDSS_CALL_AND_CHECK(cudssMatrixDestroy(A), status, "cudssMatrixDestroy for A");
    CUDSS_CALL_AND_CHECK(cudssMatrixDestroy(b), status, "cudssMatrixDestroy for b");
    CUDSS_CALL_AND_CHECK(cudssMatrixDestroy(x), status, "cudssMatrixDestroy for x");
    CUDSS_CALL_AND_CHECK(cudssDataDestroy(handle, solverData), status, "cudssDataDestroy");
    CUDSS_CALL_AND_CHECK(cudssConfigDestroy(solverConfig), status, "cudssConfigDestroy");
    CUDSS_CALL_AND_CHECK(cudssDestroy(handle), status, "cudssHandleDestroy");

    CUDA_CALL_AND_CHECK(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

    /* (optional) Print the solution */
    CUDA_CALL_AND_CHECK(cudaMemcpy(x_values_h, x_values_d, nrhs * n * sizeof(double),
        cudaMemcpyDeviceToHost), "cudaMemcpy for x_values");

    if (rank == 0) {
        printf("--- Solving complete ! --- \n");
        printf("Solution vector (first 10 and last 10 elements):\n");
        for (int i = 0; i < n && i < 10; i++) {
            printf("x[%d] = %e\n", i, x_values_h[i]);
        }
        if (n > 20) {
            printf("...\n");
            for (int i = n - 10; i < n; i++) {
                printf("x[%d] = %e\n", i, x_values_h[i]);
            }
        } else if (n > 10) {
            for (int i = 10; i < n; i++) {
                printf("x[%d] = %e\n", i, x_values_h[i]);
            }
        }
    }

    /* (optional) For the root process, print the solution and compare against the exact solution */
    if (rank == 0 || rank == 1) {
        /* Release the data allocated on the user side */
        CUDSS_EXAMPLE_FREE;
    }

    /* Deleting the memory allocated for the communicator */
#if USE_MPI
    if (mpi_comm != NULL) free(mpi_comm);
#endif
#if USE_NCCL
    if (nccl_comm != NULL) NCCL_CALL_AND_CHECK(ncclCommDestroy(*nccl_comm),"ncclCommDestroy");
    if (nccl_comm != NULL) free(nccl_comm);
#endif

    /* Cleanup for the communication backend
       See comments about calling MPI_Init() above */
#ifdef USE_MPI
    MPI_Finalize();
#endif

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    if (status == CUDSS_STATUS_SUCCESS) {
        if (rank == 0)
            printf("Example PASSED\n");
        return 0;
    } else {
        if (rank == 0)
            printf("Example FAILED\n");
        return -3;
    }
}