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

#include <assert.h>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

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

static inline cuComplex conjugate_if_needed(const cuComplex& value,
                                            bool conjugate) {
  if (!conjugate) {
    return value;
  }
  return make_cuComplex(value.x, -value.y);
}

static float compute_complex_residual_norm(int n, const int* csr_offsets_h,
                                           const int* csr_columns_h,
                                           const cuComplex* csr_values_h,
                                           const cuComplex* x_values_h,
                                           const cuComplex* b_values_h,
                                           cudssMatrixViewType_t mview,
                                           bool isHermitian) {
  std::vector<cuComplex> Ax(n, make_cuComplex(0.0f, 0.0f));

  for (int row = 0; row < n; ++row) {
    for (int idx = csr_offsets_h[row]; idx < csr_offsets_h[row + 1]; ++idx) {
      int col = csr_columns_h[idx];
      cuComplex val = csr_values_h[idx];

      switch (mview) {
        case CUDSS_MVIEW_FULL: {
          cuComplex product = cuCmulf(val, x_values_h[col]);
          Ax[row] = cuCaddf(Ax[row], product);
          break;
        }
        case CUDSS_MVIEW_UPPER: {
          if (col >= row) {
            cuComplex product = cuCmulf(val, x_values_h[col]);
            Ax[row] = cuCaddf(Ax[row], product);
            if (col != row) {
              cuComplex mirrored = conjugate_if_needed(val, isHermitian);
              cuComplex mirror_product = cuCmulf(mirrored, x_values_h[row]);
              Ax[col] = cuCaddf(Ax[col], mirror_product);
            }
          }
          break;
        }
        case CUDSS_MVIEW_LOWER: {
          if (col <= row) {
            cuComplex product = cuCmulf(val, x_values_h[col]);
            Ax[row] = cuCaddf(Ax[row], product);
            if (col != row) {
              cuComplex mirrored = conjugate_if_needed(val, isHermitian);
              cuComplex mirror_product = cuCmulf(mirrored, x_values_h[row]);
              Ax[col] = cuCaddf(Ax[col], mirror_product);
            }
          }
          break;
        }
      }
    }
  }

  float norm_squared = 0.0f;
  for (int i = 0; i < n; ++i) {
    cuComplex diff =
        make_cuComplex(Ax[i].x - b_values_h[i].x, Ax[i].y - b_values_h[i].y);
    norm_squared += diff.x * diff.x + diff.y * diff.y;
  }

  return sqrtf(norm_squared);
}

static int read_matrix_market_complex(
    const std::string& filename, int& n, int& nnz, int** csr_offsets_h,
    int** csr_columns_h, cuComplex** csr_values_h, cudssMatrixType_t& mtype,
    cudssMatrixViewType_t& mview, bool& isHermitian) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    fprintf(stderr, "Error: Could not open matrix file %s\n", filename.c_str());
    return EXIT_FAILURE;
  }

  std::string line;
  bool headerFound = false;
  bool sizeFound = false;
  int declaredNnz = 0;
  bool valuesAreComplex = false;
  std::string symmetry = "general";

  std::vector<std::tuple<int, int, cuComplex>> entries;
  bool foundLower = false;
  bool foundUpper = false;

  while (std::getline(file, line)) {
    if (line.empty()) {
      continue;
    }

    if (!headerFound) {
      if (line.rfind("%%MatrixMarket", 0) == 0) {
        headerFound = true;
        std::istringstream headerStream(line);
        std::string marker, object, format, valueType, symmetryToken;
        headerStream >> marker >> object >> format >> valueType >>
            symmetryToken;
        if (object != "matrix" || format != "coordinate") {
          fprintf(stderr,
                  "Error: Unsupported Matrix Market header in %s. Expected "
                  "'matrix coordinate'.\n",
                  filename.c_str());
          return EXIT_FAILURE;
        }
        valuesAreComplex = (valueType == "complex");
        if (!valuesAreComplex && valueType != "real") {
          fprintf(stderr,
                  "Error: Unsupported value type '%s' in %s. Expected real or "
                  "complex.\n",
                  valueType.c_str(), filename.c_str());
          return EXIT_FAILURE;
        }
        symmetry = symmetryToken;
        continue;
      }
      if (line[0] == '%') {
        continue;
      }
    }

    if (line[0] == '%') {
      continue;
    }

    std::istringstream dataStream(line);
    if (!sizeFound) {
      int ncols = 0;
      dataStream >> n >> ncols >> declaredNnz;
      if (!dataStream || ncols != n) {
        fprintf(stderr,
                "Error: Matrix in %s must be square. Parsed n=%d, m=%d.\n",
                filename.c_str(), n, ncols);
        return EXIT_FAILURE;
      }
      sizeFound = true;
    } else {
      int row = 0;
      int col = 0;
      double realPart = 0.0;
      double imagPart = 0.0;
      dataStream >> row >> col >> realPart;
      if (!dataStream) {
        fprintf(stderr, "Error: Invalid entry in %s.\n", filename.c_str());
        return EXIT_FAILURE;
      }
      if (valuesAreComplex) {
        if (!(dataStream >> imagPart)) {
          fprintf(stderr, "Error: Missing imaginary part in %s.\n",
                  filename.c_str());
          return EXIT_FAILURE;
        }
      } else {
        if (!(dataStream >> imagPart)) {
          imagPart = 0.0;
        }
      }

      row -= 1;
      col -= 1;
      entries.emplace_back(row, col,
                           make_cuComplex(static_cast<float>(realPart),
                                          static_cast<float>(imagPart)));
      if (row < col)
        foundUpper = true;
      else if (row > col)
        foundLower = true;
    }
  }
  file.close();

  if (!headerFound || !sizeFound) {
    fprintf(stderr, "Error: Incomplete Matrix Market file %s.\n",
            filename.c_str());
    return EXIT_FAILURE;
  }

  if (declaredNnz != static_cast<int>(entries.size())) {
    fprintf(stderr,
            "Warning: Declared nnz=%d but read %zu entries in %s. Continuing "
            "with read data.\n",
            declaredNnz, entries.size(), filename.c_str());
  }

  nnz = static_cast<int>(entries.size());
  if (nnz == 0) {
    fprintf(stderr, "Error: Matrix file %s contains no entries.\n",
            filename.c_str());
    return EXIT_FAILURE;
  }

  std::string symmetryLower = symmetry;
  std::transform(symmetryLower.begin(), symmetryLower.end(),
                 symmetryLower.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  if (symmetryLower == "general") {
    mview = CUDSS_MVIEW_FULL;
    mtype = CUDSS_MTYPE_GENERAL;
    isHermitian = false;
  } else if (symmetryLower == "symmetric") {
    isHermitian = false;
    mview = foundLower && !foundUpper ? CUDSS_MVIEW_LOWER : CUDSS_MVIEW_UPPER;
    mtype = CUDSS_MTYPE_SYMMETRIC;
  } else if (symmetryLower == "hermitian") {
    isHermitian = true;
    mview = foundLower && !foundUpper ? CUDSS_MVIEW_LOWER : CUDSS_MVIEW_UPPER;
    mtype = CUDSS_MTYPE_HERMITIAN;
  } else {
    fprintf(stderr, "Error: Unsupported symmetry '%s' in %s.\n",
            symmetry.c_str(), filename.c_str());
    return EXIT_FAILURE;
  }

  *csr_offsets_h = (int*)malloc((n + 1) * sizeof(int));
  *csr_columns_h = (int*)malloc(nnz * sizeof(int));
  *csr_values_h = (cuComplex*)malloc(nnz * sizeof(cuComplex));

  if (!(*csr_offsets_h) || !(*csr_columns_h) || !(*csr_values_h)) {
    fprintf(stderr, "Error: Host memory allocation failed while reading %s.\n",
            filename.c_str());
    free(*csr_offsets_h);
    free(*csr_columns_h);
    free(*csr_values_h);
    *csr_offsets_h = NULL;
    *csr_columns_h = NULL;
    *csr_values_h = NULL;
    return EXIT_FAILURE;
  }

  std::fill(*csr_offsets_h, *csr_offsets_h + (n + 1), 0);

  std::sort(entries.begin(), entries.end(),
            [](const std::tuple<int, int, cuComplex>& a,
               const std::tuple<int, int, cuComplex>& b) {
              if (std::get<0>(a) != std::get<0>(b))
                return std::get<0>(a) < std::get<0>(b);
              return std::get<1>(a) < std::get<1>(b);
            });

  int currentIdx = 0;
  for (const auto& entry : entries) {
    int row = std::get<0>(entry);
    int col = std::get<1>(entry);
    if (row < 0 || row >= n || col < 0 || col >= n) {
      fprintf(stderr, "Error: Entry (%d,%d) out of bounds in %s.\n", row, col,
              filename.c_str());
      free(*csr_offsets_h);
      free(*csr_columns_h);
      free(*csr_values_h);
      *csr_offsets_h = NULL;
      *csr_columns_h = NULL;
      *csr_values_h = NULL;
      return EXIT_FAILURE;
    }
    (*csr_offsets_h)[row + 1]++;
    (*csr_columns_h)[currentIdx] = col;
    (*csr_values_h)[currentIdx] = std::get<2>(entry);
    currentIdx++;
  }

  for (int i = 0; i < n; ++i) {
    (*csr_offsets_h)[i + 1] += (*csr_offsets_h)[i];
  }

  return EXIT_SUCCESS;
}

static int read_rhs_matrix_market_complex(const std::string& filename,
                                          int expected_n,
                                          cuComplex** b_values_h) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    fprintf(stderr, "Error: Could not open RHS file %s\n", filename.c_str());
    return EXIT_FAILURE;
  }

  std::string line;
  bool headerFound = false;
  bool sizeFound = false;
  bool valuesAreComplex = false;
  int declaredRows = 0;
  int declaredCols = 0;

  std::vector<cuComplex> values;

  while (std::getline(file, line)) {
    if (line.empty()) {
      continue;
    }

    if (!headerFound) {
      if (line.rfind("%%MatrixMarket", 0) == 0) {
        headerFound = true;
        std::istringstream headerStream(line);
        std::string marker, object, format, valueType, symmetry;
        headerStream >> marker >> object >> format >> valueType >> symmetry;
        if (object != "matrix" || format != "array") {
          fprintf(
              stderr,
              "Error: Unsupported RHS header in %s. Expected 'matrix array'.\n",
              filename.c_str());
          return EXIT_FAILURE;
        }
        valuesAreComplex = (valueType == "complex");
        if (!valuesAreComplex && valueType != "real") {
          fprintf(stderr, "Error: Unsupported RHS value type '%s' in %s.\n",
                  valueType.c_str(), filename.c_str());
          return EXIT_FAILURE;
        }
        continue;
      }
      if (line[0] == '%') {
        continue;
      }
    }

    if (line[0] == '%') {
      continue;
    }

    std::istringstream dataStream(line);
    if (!sizeFound) {
      dataStream >> declaredRows >> declaredCols;
      if (!dataStream || declaredCols != 1) {
        fprintf(stderr, "Error: RHS in %s must be a single column vector.\n",
                filename.c_str());
        return EXIT_FAILURE;
      }
      if (declaredRows != expected_n) {
        fprintf(
            stderr,
            "Error: RHS size mismatch: matrix has %d rows but RHS has %d.\n",
            expected_n, declaredRows);
        return EXIT_FAILURE;
      }
      sizeFound = true;
    } else {
      double realPart = 0.0;
      double imagPart = 0.0;
      dataStream >> realPart;
      if (!dataStream) {
        fprintf(stderr, "Error: Invalid RHS entry in %s.\n", filename.c_str());
        return EXIT_FAILURE;
      }
      if (valuesAreComplex) {
        if (!(dataStream >> imagPart)) {
          fprintf(stderr, "Error: Missing RHS imaginary part in %s.\n",
                  filename.c_str());
          return EXIT_FAILURE;
        }
      } else {
        if (!(dataStream >> imagPart)) {
          imagPart = 0.0;
        }
      }
      values.push_back(make_cuComplex(static_cast<float>(realPart),
                                      static_cast<float>(imagPart)));
    }
  }

  file.close();

  if (!sizeFound || static_cast<int>(values.size()) != expected_n) {
    fprintf(stderr, "Error: RHS file %s does not have %d entries.\n",
            filename.c_str(), expected_n);
    return EXIT_FAILURE;
  }

  *b_values_h = (cuComplex*)malloc(expected_n * sizeof(cuComplex));
  if (!(*b_values_h)) {
    fprintf(stderr, "Error: Host memory allocation failed for RHS.\n");
    return EXIT_FAILURE;
  }

  std::copy(values.begin(), values.end(), *b_values_h);
  return EXIT_SUCCESS;
}

int main (int argc, char *argv[]) {
    printf("---------------------------------------------------------\n");
    printf("cuDSS example: solving a complex linear 5x5 system\n"
           "with a symmetric positive-definite matrix \n");
    printf("---------------------------------------------------------\n");
    cudaError_t cuda_error = cudaSuccess;
    cudssStatus_t status = CUDSS_STATUS_SUCCESS;

    int n = 0;     // 问题规模 (可能由文件决定)
    int nnz = 0;   // 非零元数量 (可能由文件决定)
    int nrhs = 1;  // 右端项数量

    /* 主机端用于保存稀疏矩阵 A 的 CSR 结构以及稠密向量的缓冲区。 */
    int* csr_offsets_h = NULL;  // CSR 行偏移数组
    int* csr_columns_h = NULL;  // CSR 列索引数组
    cuComplex* csr_values_h =
        NULL;  // CSR 非零元数组, cuComplex 类型表示 float32 复数
    cuComplex *x_values_h = NULL, *b_values_h = NULL;  // 解向量和右端项

    /* 设备端（GPU）上与主机缓冲区镜像的数据，用于实际计算。 */
    int* csr_offsets_d = NULL;
    int* csr_columns_d = NULL;
    cuComplex* csr_values_d = NULL;
    cuComplex *x_values_d = NULL, *b_values_d = NULL;

    /* 尝试从 Matrix Market (.mtx) 文件读取复数稀疏矩阵和
       RHS。如果命令行提供了文件名则优先使用： 用法: ./simple_complex_example
       <matrix_file.mtx> [rhs_file.mtx]
       若读取失败，则回退到内置的小示例矩阵（5x5）。 */
    bool loaded_from_file = false;
    std::string matrix_file;
    std::string rhs_file;
    if (argc >= 2) {
      matrix_file = argv[1];
      if (argc >= 3) rhs_file = argv[2];
    } else {
      /* 默认尝试查找构建目录中可能存在的示例文件名 */
      matrix_file = "A_1762908628060293_3_matrix.mtx";
      rhs_file = "A_1762908628060293_3_rhs.mtx";
    }

    cudssMatrixType_t file_mtype = CUDSS_MTYPE_GENERAL;
    cudssMatrixViewType_t file_mview = CUDSS_MVIEW_FULL;
    bool file_isHermitian = false;

    if (!matrix_file.empty()) {
      std::ifstream f(matrix_file);
      if (f.good()) {
        f.close();
        if (read_matrix_market_complex(
                matrix_file, n, nnz, &csr_offsets_h, &csr_columns_h,
                &csr_values_h, file_mtype, file_mview, file_isHermitian) == 0) {
          /* 如果 RHS 文件存在则读取，否则在之后填充为 1+0i */
          if (!rhs_file.empty()) {
            std::ifstream fr(rhs_file);
            if (fr.good()) {
              fr.close();
              if (read_rhs_matrix_market_complex(rhs_file, n, &b_values_h) !=
                  0) {
                fprintf(
                    stderr,
                    "Warning: failed to read RHS file %s, using default RHS.\n",
                    rhs_file.c_str());
                free(b_values_h);
                b_values_h = NULL;
              }
            }
          }

          if (!b_values_h) {
            /* 如果没有 RHS，则用 1+0i 填充 */
            b_values_h = (cuComplex*)malloc(n * sizeof(cuComplex));
            for (int ii = 0; ii < n; ++ii)
              b_values_h[ii] = make_cuComplex(1.0f, 0.0f);
          }

          /* 为解分配主机内存 */
          x_values_h = (cuComplex*)malloc(nrhs * n * sizeof(cuComplex));
          if (!x_values_h) {
            fprintf(stderr, "Error: host memory allocation failed for x.\n");
            return -1;
          }

          /* 从文件读取成功 */
          loaded_from_file = true;
        } else {
          fprintf(stderr,
                  "Warning: failed to read matrix file %s, falling back to "
                  "built-in example.\n",
                  matrix_file.c_str());
        }
      }
    }

    /* 如果没有从文件加载成功，使用内置示例数据（保持原有行为） */
    if (!loaded_from_file) {
      n = 5;     // 问题规模
      nnz = 8;   // 非零元数量
      nrhs = 1;  // 右端项数量

      /* 分配并初始化内置示例数据（与原示例一致） */
      csr_offsets_h = (int*)malloc((n + 1) * sizeof(int));
      csr_columns_h = (int*)malloc(nnz * sizeof(int));
      csr_values_h = (cuComplex*)malloc(nnz * sizeof(cuComplex));
      x_values_h = (cuComplex*)malloc(nrhs * n * sizeof(cuComplex));
      b_values_h = (cuComplex*)malloc(nrhs * n * sizeof(cuComplex));

      if (!csr_offsets_h || !csr_columns_h || !csr_values_h || !x_values_h ||
          !b_values_h) {
        printf("Error: host memory allocation failed\n");
        return -1;
      }

      int ii = 0;
      csr_offsets_h[ii++] = 0;
      csr_offsets_h[ii++] = 2;
      csr_offsets_h[ii++] = 4;
      csr_offsets_h[ii++] = 6;
      csr_offsets_h[ii++] = 7;
      csr_offsets_h[ii++] = 8;

      ii = 0;
      csr_columns_h[ii++] = 0;
      csr_columns_h[ii++] = 2;
      csr_columns_h[ii++] = 1;
      csr_columns_h[ii++] = 2;
      csr_columns_h[ii++] = 2;
      csr_columns_h[ii++] = 4;
      csr_columns_h[ii++] = 3;
      csr_columns_h[ii++] = 4;

      ii = 0;
      csr_values_h[ii++].x = 4.0;
      csr_values_h[ii++].x = 1.0;
      csr_values_h[ii++].x = 3.0;
      csr_values_h[ii++].x = 2.0;
      csr_values_h[ii++].x = 5.0;
      csr_values_h[ii++].x = 1.0;
      csr_values_h[ii++].x = 1.0;
      csr_values_h[ii++].x = 2.0;
      ii = 0;
      csr_values_h[ii++].y = 0.0;
      csr_values_h[ii++].y = 0.0;
      csr_values_h[ii++].y = 0.0;
      csr_values_h[ii++].y = 0.0;
      csr_values_h[ii++].y = 0.0;
      csr_values_h[ii++].y = 0.0;
      csr_values_h[ii++].y = 0.0;
      csr_values_h[ii++].y = 0.0;

      ii = 0;
      b_values_h[ii++].x = 7.0;
      b_values_h[ii++].x = 12.0;
      b_values_h[ii++].x = 25.0;
      b_values_h[ii++].x = 4.0;
      b_values_h[ii++].x = 13.0;
      ii = 0;
      b_values_h[ii++].y = 0.0;
      b_values_h[ii++].y = 0.0;
      b_values_h[ii++].y = 0.0;
      b_values_h[ii++].y = 0.0;
      b_values_h[ii++].y = 0.0;
    }

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
    /* 如果从文件加载，使用文件提供的类型和视图；否则保持原示例的默认设置 */
    cudssMatrixType_t mtype =
        loaded_from_file ? file_mtype : CUDSS_MTYPE_SPD;  // 矩阵类型
    cudssMatrixViewType_t mview =
        loaded_from_file ? file_mview : CUDSS_MVIEW_UPPER;  // 存储视图
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

    /* Print the solution. If we loaded the matrix from a file, compute the
       residual norm ||Ax - b|| to validate the solution; otherwise compare
       against the small built-in exact solution {1,2,..,n}. */
    CUDA_CALL_AND_CHECK(
        cudaMemcpy(x_values_h, x_values_d, nrhs * n * sizeof(cuComplex),
                   cudaMemcpyDeviceToHost),
        "cudaMemcpy for x_values");

    int passed = 1;
    if (loaded_from_file) {
      float residual = compute_complex_residual_norm(
          n, csr_offsets_h, csr_columns_h, csr_values_h, x_values_h, b_values_h,
          mview, file_isHermitian);
      printf("Residual (absolute) L2 norm ||Ax-b|| = %e\n", (double)residual);
      passed = (residual < 1e-3f); /* tolerance for sample; adjust as needed */
    } else {
      for (int i = 0; i < n; i++) {
        printf("x[%d] = (%1.4f, %1.4f) expected (%1.4f, 0)\n", i,
               x_values_h[i].x, x_values_h[i].y, double(i + 1));
        if (fabs(x_values_h[i].x - (i + 1)) + fabs(x_values_h[i].y) > 2.e-6)
          passed = 0;
      }
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