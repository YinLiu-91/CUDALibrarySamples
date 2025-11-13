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
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

#include "cudss.h"
#include "matrix_market_reader_complex.h"

namespace {

// 用于封装命令行输入的辅助结构体，记录稀疏矩阵、右端向量来源
struct ExampleInput {
  std::string matrix_path;
  std::string rhs_path;
  bool matrix_from_cli = false;
  bool rhs_from_cli = false;
};

// 运行时支持的精度模式，目前包括单精度和双精度复数
enum class PrecisionMode { kComplex64, kComplex128 };

template <typename ComplexT>
struct PrecisionTraits;

// 针对 cuComplex（单精度复数）的 traits，用于屏蔽不同精度下 API 差异
template <>
struct PrecisionTraits<cuComplex> {
  using ComplexType = cuComplex;
  using RealType = float;

  static constexpr cudaDataType_t kCudaType = CUDA_C_32F;
  static constexpr const char* kName = "complex64";

  static ComplexType make(double real, double imag) {
    return make_cuComplex(static_cast<float>(real), static_cast<float>(imag));
  }
  static ComplexType zero() { return make_cuComplex(0.0f, 0.0f); }
  static ComplexType add(ComplexType a, ComplexType b) { return cuCaddf(a, b); }
  static ComplexType sub(ComplexType a, ComplexType b) {
    return make_cuComplex(a.x - b.x, a.y - b.y);
  }
  static ComplexType mul(ComplexType a, ComplexType b) { return cuCmulf(a, b); }
  static ComplexType conj(ComplexType a) { return cuConjf(a); }
  static double norm_sq(ComplexType a) {
    double rx = static_cast<double>(a.x);
    double ry = static_cast<double>(a.y);
    return rx * rx + ry * ry;
  }
  static double real(ComplexType a) { return static_cast<double>(a.x); }
  static double imag(ComplexType a) { return static_cast<double>(a.y); }
  static double residual_tolerance() { return 1e-3; }
  static double solution_tolerance() { return 2e-6; }
};

// 针对 cuDoubleComplex（双精度复数）的 traits，与单精度实现对应
template <>
struct PrecisionTraits<cuDoubleComplex> {
  using ComplexType = cuDoubleComplex;
  using RealType = double;

  static constexpr cudaDataType_t kCudaType = CUDA_C_64F;
  static constexpr const char* kName = "complex128";

  static ComplexType make(double real, double imag) {
    return make_cuDoubleComplex(real, imag);
  }
  static ComplexType zero() { return make_cuDoubleComplex(0.0, 0.0); }
  static ComplexType add(ComplexType a, ComplexType b) { return cuCadd(a, b); }
  static ComplexType sub(ComplexType a, ComplexType b) {
    return make_cuDoubleComplex(cuCreal(a) - cuCreal(b),
                                cuCimag(a) - cuCimag(b));
  }
  static ComplexType mul(ComplexType a, ComplexType b) { return cuCmul(a, b); }
  static ComplexType conj(ComplexType a) { return cuConj(a); }
  static double norm_sq(ComplexType a) {
    double rx = cuCreal(a);
    double ry = cuCimag(a);
    return rx * rx + ry * ry;
  }
  static double real(ComplexType a) { return cuCreal(a); }
  static double imag(ComplexType a) { return cuCimag(a); }
  static double residual_tolerance() { return 1e-9; }
  static double solution_tolerance() { return 1e-12; }
};

// 计算稀疏矩阵 A 与解向量 x 的乘积 Ax，并与 b 做差得到残差范数
template <typename ComplexT>
double compute_complex_residual_norm(
    int n, const int* csr_offsets_h, const int* csr_columns_h,
    const ComplexT* csr_values_h, const ComplexT* x_values_h,
    const ComplexT* b_values_h, cudssMatrixViewType_t view, bool is_hermitian) {
  using Traits = PrecisionTraits<ComplexT>;
  std::vector<ComplexT> Ax(n, Traits::zero());

  for (int row = 0; row < n; ++row) {
    for (int idx = csr_offsets_h[row]; idx < csr_offsets_h[row + 1]; ++idx) {
      int col = csr_columns_h[idx];
      ComplexT val = csr_values_h[idx];
      Ax[row] = Traits::add(Ax[row], Traits::mul(val, x_values_h[col]));

      if (view == CUDSS_MVIEW_UPPER && col != row) {
        // 如果矩阵是厄米特矩阵，那么存储的上三角元素被镜像到对应的下三角位置时
        // 需要取共轭，保证恢复出的完整矩阵满足 A =
        // A^H；非厄米特矩阵则直接复制即可。
        ComplexT mirrored = is_hermitian ? Traits::conj(val) : val;
        Ax[col] = Traits::add(Ax[col], Traits::mul(mirrored, x_values_h[row]));
      } else if (view == CUDSS_MVIEW_LOWER && col != row) {
        // 同理，当只存储了下三角视图时，镜像到上三角也要根据是否厄米特决定是否取共轭。
        ComplexT mirrored = is_hermitian ? Traits::conj(val) : val;
        Ax[col] = Traits::add(Ax[col], Traits::mul(mirrored, x_values_h[row]));
      }
    }
  }

  double norm_sq = 0.0;
  for (int i = 0; i < n; ++i) {
    ComplexT diff = Traits::sub(Ax[i], b_values_h[i]);
    norm_sq += Traits::norm_sq(diff);
  }
  return std::sqrt(norm_sq);
}

// 根据输入矩阵路径生成解向量输出文件名，默认追加 "_solution.mtx"
std::string make_solution_output_path(const std::string& matrix_path) {
  if (matrix_path.empty()) {
    return "solution_x.mtx";
  }

  std::string result = matrix_path;
  size_t slash = result.find_last_of("/\\");
  size_t dot = result.find_last_of('.');
  if (dot != std::string::npos && (slash == std::string::npos || dot > slash)) {
    result = result.substr(0, dot);
  }

  return result + "_solution.mtx";
}

// 尝试依据矩阵路径推导黄金解（golden solution）文件位置
std::string make_golden_solution_path(const std::string& matrix_path) {
  if (matrix_path.empty()) {
    return std::string();
  }

  std::string result = matrix_path;
  size_t slash = result.find_last_of("/\\");
  size_t dot = result.find_last_of('.');
  std::string prefix;
  if (dot != std::string::npos && (slash == std::string::npos || dot > slash)) {
    prefix = result.substr(0, dot);
  } else {
    prefix = result;
  }

  const std::string matrix_suffix = "_matrix";
  if (prefix.size() >= matrix_suffix.size()) {
    size_t pos = prefix.rfind(matrix_suffix);
    if (pos != std::string::npos &&
        pos + matrix_suffix.size() == prefix.size()) {
      prefix = prefix.substr(0, pos);
    }
  }

  return prefix + "_golden_sol.mtx";
}

// 依据 L2 范数评估当前解与参考解的差异，同时可以返回相对误差
template <typename ComplexT>
double compute_solution_error_norm(int n, const ComplexT* solution,
                                   const ComplexT* reference,
                                   double* relative_norm_out) {
  using Traits = PrecisionTraits<ComplexT>;
  double diff_norm_sq = 0.0;
  double ref_norm_sq = 0.0;

  for (int i = 0; i < n; ++i) {
    ComplexT diff = Traits::sub(solution[i], reference[i]);
    diff_norm_sq += Traits::norm_sq(diff);
    ref_norm_sq += Traits::norm_sq(reference[i]);
  }

  double diff_norm = std::sqrt(diff_norm_sq);
  double ref_norm = std::sqrt(ref_norm_sq);
  if (relative_norm_out) {
    if (ref_norm > 0.0) {
      *relative_norm_out = diff_norm / ref_norm;
    } else {
      *relative_norm_out = 0.0;
    }
  }
  return diff_norm;
}

// 将列主序存储的解向量写成 Matrix Market array 格式，便于后处理
template <typename ComplexT>
bool write_solution_matrix_market(const std::string& path, int nrows, int ncols,
                                  const ComplexT* values) {
  std::ofstream out(path.c_str());
  if (!out.is_open()) {
    return false;
  }

  out << "%%MatrixMarket matrix array complex general\n";
  out << "% Generated by cuDSS simple_complex example\n";
  out << nrows << " " << ncols << "\n";
  out << std::scientific << std::setprecision(16);

  using Traits = PrecisionTraits<ComplexT>;
  for (int col = 0; col < ncols; ++col) {
    for (int row = 0; row < nrows; ++row) {
      const ComplexT& value = values[col * nrows + row];
      out << Traits::real(value) << " " << Traits::imag(value) << "\n";
    }
  }

  return true;
}

// 单次求解样例的主流程：加载数据、拷贝到 GPU、执行 cuDSS 三阶段并回收结果
template <typename ComplexT>
int run_example(const ExampleInput& input) {
  using Traits = PrecisionTraits<ComplexT>;

  int n = 0;
  int nnz = 0;
  int nrhs = 1;

  int* csr_offsets_h = nullptr;
  int* csr_columns_h = nullptr;
  ComplexT* csr_values_h = nullptr;
  ComplexT* x_values_h = nullptr;
  ComplexT* b_values_h = nullptr;

  int* csr_offsets_d = nullptr;
  int* csr_columns_d = nullptr;
  ComplexT* csr_values_d = nullptr;
  ComplexT* x_values_d = nullptr;
  ComplexT* b_values_d = nullptr;

  cudaStream_t stream = nullptr;
  cudaEvent_t event_start = nullptr;
  cudaEvent_t event_stop = nullptr;
  cudssHandle_t handle = nullptr;
  cudssConfig_t solver_config = nullptr;
  cudssData_t solver_data = nullptr;
  cudssMatrix_t A = nullptr;
  cudssMatrix_t x = nullptr;
  cudssMatrix_t b = nullptr;

  bool loaded_from_file = false;
  cudssMatrixType_t matrix_type = CUDSS_MTYPE_SPD;
  cudssMatrixViewType_t matrix_view = CUDSS_MVIEW_UPPER;
  bool matrix_is_hermitian = false;

  double analysis_ms = 0.0;
  double factor_ms = 0.0;
  double solve_ms = 0.0;

  // RAII 风格的清理函数，任何错误返回都会触发资源释放
  auto cleanup = [&]() {
    if (A) {
      cudssMatrixDestroy(A);
      A = nullptr;
    }
    if (x) {
      cudssMatrixDestroy(x);
      x = nullptr;
    }
    if (b) {
      cudssMatrixDestroy(b);
      b = nullptr;
    }
    if (solver_data) {
      cudssDataDestroy(handle, solver_data);
      solver_data = nullptr;
    }
    if (solver_config) {
      cudssConfigDestroy(solver_config);
      solver_config = nullptr;
    }
    if (handle) {
      cudssDestroy(handle);
      handle = nullptr;
    }
    if (stream) {
      cudaStreamDestroy(stream);
      stream = nullptr;
    }
    if (event_start) {
      cudaEventDestroy(event_start);
      event_start = nullptr;
    }
    if (event_stop) {
      cudaEventDestroy(event_stop);
      event_stop = nullptr;
    }

    if (csr_offsets_d) {
      cudaFree(csr_offsets_d);
      csr_offsets_d = nullptr;
    }
    if (csr_columns_d) {
      cudaFree(csr_columns_d);
      csr_columns_d = nullptr;
    }
    if (csr_values_d) {
      cudaFree(csr_values_d);
      csr_values_d = nullptr;
    }
    if (x_values_d) {
      cudaFree(x_values_d);
      x_values_d = nullptr;
    }
    if (b_values_d) {
      cudaFree(b_values_d);
      b_values_d = nullptr;
    }

    free(csr_offsets_h);
    free(csr_columns_h);
    free(csr_values_h);
    free(x_values_h);
    free(b_values_h);

    csr_offsets_h = nullptr;
    csr_columns_h = nullptr;
    csr_values_h = nullptr;
    x_values_h = nullptr;
    b_values_h = nullptr;
  };

#define CUDA_CALL_AND_CHECK(call, msg)                                      \
  do {                                                                      \
    cudaError_t _cuda_error = (call);                                       \
    if (_cuda_error != cudaSuccess) {                                       \
      printf("Example FAILED: CUDA API returned error = %d, details: %s\n", \
             static_cast<int>(_cuda_error), (msg));                         \
      cleanup();                                                            \
      return -1;                                                            \
    }                                                                       \
  } while (0)

#define CUDSS_CALL_AND_CHECK(call, msg)                                    \
  do {                                                                     \
    cudssStatus_t _cudss_status = (call);                                  \
    if (_cudss_status != CUDSS_STATUS_SUCCESS) {                           \
      printf(                                                              \
          "Example FAILED: CUDSS call ended unsuccessfully with status = " \
          "%d, details: %s\n",                                             \
          static_cast<int>(_cudss_status), (msg));                         \
      cleanup();                                                           \
      return -2;                                                           \
    }                                                                      \
  } while (0)

  if (!input.matrix_path.empty()) {
    std::ifstream matrix_stream(input.matrix_path);
    if (matrix_stream.good()) {
      matrix_stream.close();
      // 读取外部 Matrix Market 稀疏矩阵，支持从命令行指定视图与 RHS
      matrix_market::MatrixMetadata metadata;
      matrix_market::MatrixReadOptions options;
      options.requested_view = CUDSS_MVIEW_FULL;
      options.allow_real_as_complex = true;

      matrix_market::Status status =
          matrix_market::read_matrix_coordinate<ComplexT>(
              input.matrix_path, n, nnz, &csr_offsets_h, &csr_columns_h,
              &csr_values_h, &metadata, options);

      if (status == matrix_market::Status::kSuccess) {
        matrix_type = metadata.mtype;
        matrix_view = metadata.detected_view;
        matrix_is_hermitian = metadata.is_hermitian;

        if (!input.rhs_path.empty()) {
          std::ifstream rhs_stream(input.rhs_path);
          if (rhs_stream.good()) {
            rhs_stream.close();
            matrix_market::Status rhs_status =
                matrix_market::read_rhs_array<ComplexT>(input.rhs_path, n,
                                                        &b_values_h, true);
            if (rhs_status != matrix_market::Status::kSuccess) {
              fprintf(stderr,
                      "Warning: failed to read RHS file %s (%s); using default "
                      "RHS.\n",
                      input.rhs_path.c_str(),
                      matrix_market::StatusToString(rhs_status));
              free(b_values_h);
              b_values_h = nullptr;
            }
          } else if (input.rhs_from_cli) {
            fprintf(stderr,
                    "Warning: RHS file %s not found, using default RHS.\n",
                    input.rhs_path.c_str());
          }
        }

        if (!b_values_h) {
          b_values_h = static_cast<ComplexT*>(
              malloc(static_cast<size_t>(n) * sizeof(ComplexT)));
          if (!b_values_h) {
            fprintf(stderr, "Error: host allocation failed for RHS.\n");
            cleanup();
            return -1;
          }
          for (int i = 0; i < n; ++i) {
            b_values_h[i] = Traits::make(1.0, 0.0);
          }
        }

        x_values_h = static_cast<ComplexT*>(
            malloc(static_cast<size_t>(nrhs) * n * sizeof(ComplexT)));
        if (!x_values_h) {
          fprintf(stderr, "Error: host allocation failed for solution.\n");
          cleanup();
          return -1;
        }
        std::fill(x_values_h, x_values_h + static_cast<size_t>(nrhs) * n,
                  Traits::zero());

        loaded_from_file = true;
      } else {
        fprintf(stderr,
                "Warning: failed to read matrix file %s (%s); falling back to "
                "built-in example.\n",
                input.matrix_path.c_str(),
                matrix_market::StatusToString(status));
        free(csr_offsets_h);
        free(csr_columns_h);
        free(csr_values_h);
        csr_offsets_h = nullptr;
        csr_columns_h = nullptr;
        csr_values_h = nullptr;
      }
    } else if (input.matrix_from_cli) {
      fprintf(stderr,
              "Warning: matrix file %s not found; using built-in data.\n",
              input.matrix_path.c_str());
    }
  }

  if (!loaded_from_file) {
    // 未提供外部文件时，构造一个 5x5 对称正定矩阵及对应 RHS 用于示例演示
    n = 5;
    nnz = 8;
    nrhs = 1;
    matrix_type = CUDSS_MTYPE_SPD;
    matrix_view = CUDSS_MVIEW_UPPER;
    matrix_is_hermitian = false;

    csr_offsets_h = static_cast<int*>(malloc((n + 1) * sizeof(int)));
    csr_columns_h = static_cast<int*>(malloc(nnz * sizeof(int)));
    csr_values_h = static_cast<ComplexT*>(malloc(nnz * sizeof(ComplexT)));
    x_values_h = static_cast<ComplexT*>(malloc(nrhs * n * sizeof(ComplexT)));
    b_values_h = static_cast<ComplexT*>(malloc(nrhs * n * sizeof(ComplexT)));

    if (!csr_offsets_h || !csr_columns_h || !csr_values_h || !x_values_h ||
        !b_values_h) {
      fprintf(stderr, "Error: host allocation failed for built-in data.\n");
      cleanup();
      return -1;
    }

    const int offsets[] = {0, 2, 4, 6, 7, 8};
    const int columns[] = {0, 2, 1, 2, 2, 4, 3, 4};
    const double values[] = {4.0, 1.0, 3.0, 2.0, 5.0, 1.0, 1.0, 2.0};
    const double rhs_values[] = {7.0, 12.0, 25.0, 4.0, 13.0};

    for (int i = 0; i < n + 1; ++i) {
      csr_offsets_h[i] = offsets[i];
    }
    for (int i = 0; i < nnz; ++i) {
      csr_columns_h[i] = columns[i];
      csr_values_h[i] = Traits::make(values[i], 0.0);
    }
    for (int i = 0; i < n; ++i) {
      b_values_h[i] = Traits::make(rhs_values[i], 0.0);
      x_values_h[i] = Traits::zero();
    }
  }

  // === 设备端资源分配与数据拷贝 ===
  CUDA_CALL_AND_CHECK(
      cudaMalloc(&csr_offsets_d, static_cast<size_t>(n + 1) * sizeof(int)),
      "cudaMalloc csr_offsets");
  CUDA_CALL_AND_CHECK(
      cudaMalloc(&csr_columns_d, static_cast<size_t>(nnz) * sizeof(int)),
      "cudaMalloc csr_columns");
  CUDA_CALL_AND_CHECK(
      cudaMalloc(&csr_values_d, static_cast<size_t>(nnz) * sizeof(ComplexT)),
      "cudaMalloc csr_values");
  CUDA_CALL_AND_CHECK(
      cudaMalloc(&b_values_d, static_cast<size_t>(nrhs) * n * sizeof(ComplexT)),
      "cudaMalloc b_values");
  CUDA_CALL_AND_CHECK(
      cudaMalloc(&x_values_d, static_cast<size_t>(nrhs) * n * sizeof(ComplexT)),
      "cudaMalloc x_values");

  CUDA_CALL_AND_CHECK(cudaMemcpy(csr_offsets_d, csr_offsets_h,
                                 static_cast<size_t>(n + 1) * sizeof(int),
                                 cudaMemcpyHostToDevice),
                      "cudaMemcpy csr_offsets");
  CUDA_CALL_AND_CHECK(cudaMemcpy(csr_columns_d, csr_columns_h,
                                 static_cast<size_t>(nnz) * sizeof(int),
                                 cudaMemcpyHostToDevice),
                      "cudaMemcpy csr_columns");
  CUDA_CALL_AND_CHECK(cudaMemcpy(csr_values_d, csr_values_h,
                                 static_cast<size_t>(nnz) * sizeof(ComplexT),
                                 cudaMemcpyHostToDevice),
                      "cudaMemcpy csr_values");
  CUDA_CALL_AND_CHECK(
      cudaMemcpy(b_values_d, b_values_h,
                 static_cast<size_t>(nrhs) * n * sizeof(ComplexT),
                 cudaMemcpyHostToDevice),
      "cudaMemcpy b_values");
  CUDA_CALL_AND_CHECK(
      cudaMemset(x_values_d, 0,
                 static_cast<size_t>(nrhs) * n * sizeof(ComplexT)),
      "cudaMemset x_values");

  // 创建独立的 CUDA stream 与事件，用于测量各阶段耗时
  CUDA_CALL_AND_CHECK(cudaStreamCreate(&stream), "cudaStreamCreate");

  CUDA_CALL_AND_CHECK(cudaEventCreate(&event_start), "cudaEventCreate (start)");
  CUDA_CALL_AND_CHECK(cudaEventCreate(&event_stop), "cudaEventCreate (stop)");

  // === cuDSS 句柄与矩阵描述子的初始化 ===
  CUDSS_CALL_AND_CHECK(cudssCreate(&handle), "cudssCreate");
  CUDSS_CALL_AND_CHECK(cudssSetStream(handle, stream), "cudssSetStream");
  CUDSS_CALL_AND_CHECK(cudssConfigCreate(&solver_config), "cudssConfigCreate");
  CUDSS_CALL_AND_CHECK(cudssDataCreate(handle, &solver_data),
                       "cudssDataCreate");

  int64_t nrows = n;
  int64_t ncols = n;
  int ldb = ncols;
  int ldx = nrows;

  CUDSS_CALL_AND_CHECK(
      cudssMatrixCreateDn(&b, ncols, nrhs, ldb, b_values_d, Traits::kCudaType,
                          CUDSS_LAYOUT_COL_MAJOR),
      "cudssMatrixCreateDn (b)");
  CUDSS_CALL_AND_CHECK(
      cudssMatrixCreateDn(&x, nrows, nrhs, ldx, x_values_d, Traits::kCudaType,
                          CUDSS_LAYOUT_COL_MAJOR),
      "cudssMatrixCreateDn (x)");

  cudssIndexBase_t base = CUDSS_BASE_ZERO;
  CUDSS_CALL_AND_CHECK(
      cudssMatrixCreateCsr(&A, nrows, ncols, nnz, csr_offsets_d, nullptr,
                           csr_columns_d, csr_values_d, CUDA_R_32I,
                           Traits::kCudaType, matrix_type, matrix_view, base),
      "cudssMatrixCreateCsr");

  // === cuDSS 三阶段求解：分析 / 分解 / 求解，并采集耗时 ===
  CUDA_CALL_AND_CHECK(cudaEventRecord(event_start, stream),
                      "cudaEventRecord (analysis start)");
  CUDSS_CALL_AND_CHECK(cudssExecute(handle, CUDSS_PHASE_ANALYSIS, solver_config,
                                    solver_data, A, x, b),
                       "cudssExecute (analysis)");
  CUDA_CALL_AND_CHECK(cudaEventRecord(event_stop, stream),
                      "cudaEventRecord (analysis stop)");
  CUDA_CALL_AND_CHECK(cudaEventSynchronize(event_stop),
                      "cudaEventSynchronize (analysis)");
  {
    float elapsed_ms = 0.0f;
    CUDA_CALL_AND_CHECK(
        cudaEventElapsedTime(&elapsed_ms, event_start, event_stop),
        "cudaEventElapsedTime (analysis)");
    analysis_ms = static_cast<double>(elapsed_ms);
  }
  CUDA_CALL_AND_CHECK(cudaEventRecord(event_start, stream),
                      "cudaEventRecord (factor start)");
  CUDSS_CALL_AND_CHECK(cudssExecute(handle, CUDSS_PHASE_FACTORIZATION,
                                    solver_config, solver_data, A, x, b),
                       "cudssExecute (factor)");
  CUDA_CALL_AND_CHECK(cudaEventRecord(event_stop, stream),
                      "cudaEventRecord (factor stop)");
  CUDA_CALL_AND_CHECK(cudaEventSynchronize(event_stop),
                      "cudaEventSynchronize (factor)");
  {
    float elapsed_ms = 0.0f;
    CUDA_CALL_AND_CHECK(
        cudaEventElapsedTime(&elapsed_ms, event_start, event_stop),
        "cudaEventElapsedTime (factor)");
    factor_ms = static_cast<double>(elapsed_ms);
  }
  CUDA_CALL_AND_CHECK(cudaEventRecord(event_start, stream),
                      "cudaEventRecord (solve start)");
  CUDSS_CALL_AND_CHECK(cudssExecute(handle, CUDSS_PHASE_SOLVE, solver_config,
                                    solver_data, A, x, b),
                       "cudssExecute (solve)");
  CUDA_CALL_AND_CHECK(cudaEventRecord(event_stop, stream),
                      "cudaEventRecord (solve stop)");
  CUDA_CALL_AND_CHECK(cudaEventSynchronize(event_stop),
                      "cudaEventSynchronize (solve)");
  {
    float elapsed_ms = 0.0f;
    CUDA_CALL_AND_CHECK(
        cudaEventElapsedTime(&elapsed_ms, event_start, event_stop),
        "cudaEventElapsedTime (solve)");
    solve_ms = static_cast<double>(elapsed_ms);
  }

  // 等待所有 GPU 任务完成，随后将解向量拷贝回主机
  CUDA_CALL_AND_CHECK(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

  CUDA_CALL_AND_CHECK(
      cudaMemcpy(x_values_h, x_values_d,
                 static_cast<size_t>(nrhs) * n * sizeof(ComplexT),
                 cudaMemcpyDeviceToHost),
      "cudaMemcpy x_values");

  printf("Precision: %s\n", Traits::kName);
  double total_ms = analysis_ms + factor_ms + solve_ms;
  printf(
      "Timing (ms): analysis=%0.3f, factorization=%0.3f, solve=%0.3f, "
      "total=%0.3f\n",
      analysis_ms, factor_ms, solve_ms, total_ms);
  bool passed = true;

  if (loaded_from_file) {
    // 使用读入的稀疏矩阵进行残差验证，确认 Ax ≈ b
    double residual = compute_complex_residual_norm<ComplexT>(
        n, csr_offsets_h, csr_columns_h, csr_values_h, x_values_h, b_values_h,
        matrix_view, matrix_is_hermitian);
    printf("Residual (absolute) L2 norm ||Ax-b|| = %e\n", residual);
    if (residual >= Traits::residual_tolerance()) {
      passed = false;
    }

    // 若存在黄金解文件，进一步打印与黄金解之间的绝对/相对误差
    std::string golden_path = make_golden_solution_path(input.matrix_path);
    if (!golden_path.empty()) {
      std::ifstream golden_stream(golden_path.c_str());
      if (golden_stream.good()) {
        golden_stream.close();
        ComplexT* golden_values = nullptr;
        matrix_market::Status golden_status =
            matrix_market::read_rhs_array<ComplexT>(golden_path, n,
                                                    &golden_values, true);
        if (golden_status == matrix_market::Status::kSuccess) {
          double relative_error = 0.0;
          double abs_error = compute_solution_error_norm<ComplexT>(
              n, x_values_h, golden_values, &relative_error);
          printf("Solution error vs golden (L2): absolute=%e relative=%e\n",
                 abs_error, relative_error);
          free(golden_values);
        } else {
          fprintf(stderr, "Warning: failed to read golden solution %s (%s)\n",
                  golden_path.c_str(),
                  matrix_market::StatusToString(golden_status));
        }
      }
    }
  } else {
    // 内置测试数据下，直接对比期望向量 (1,2,3,4,5)
    for (int i = 0; i < n; ++i) {
      double xr = Traits::real(x_values_h[i]);
      double xi = Traits::imag(x_values_h[i]);
      printf("x[%d] = (%1.6f, %1.6f) expected (%1.6f, 0)\n", i, xr, xi,
             static_cast<double>(i + 1));
      double error = std::fabs(xr - static_cast<double>(i + 1)) + std::fabs(xi);
      if (error > Traits::solution_tolerance()) {
        passed = false;
      }
    }
  }

  if (loaded_from_file) {
    std::string solution_path = make_solution_output_path(input.matrix_path);
    if (write_solution_matrix_market<ComplexT>(solution_path, n, nrhs,
                                               x_values_h)) {
      printf("Solution written to %s\n", solution_path.c_str());
    } else {
      fprintf(stderr, "Warning: failed to write solution to %s\n",
              solution_path.c_str());
    }
  }

  cleanup();

#undef CUDSS_CALL_AND_CHECK
#undef CUDA_CALL_AND_CHECK

  if (passed) {
    printf("Example PASSED\n");
    return 0;
  }

  printf("Example FAILED\n");
  return -1;
}

}  // namespace

//
// 终端示例：
//   ./simple_complex_example -d ../cfm56-case/A_1762995034677701_3_matrix.mtx \
//       ../cfm56-case/A_1762995034677701_3_rhs.mtx
//   其中 -s / -d 控制单精度（complex64）或双精度（complex128）求解；
//   当提供 matrix.mtx 时，程序会自动寻找同名 *_golden_sol.mtx 进行对比。
int main(int argc, char* argv[]) {
  printf("---------------------------------------------------------\n");
  printf(
      "cuDSS example: solving a complex linear system (single or double\n"
      "precision)\n");
  printf("---------------------------------------------------------\n");

  ExampleInput input;
  input.matrix_path = "A_1762908628060293_3_matrix.mtx";
  input.rhs_path = "A_1762908628060293_3_rhs.mtx";

  PrecisionMode precision = PrecisionMode::kComplex64;

  int index = 1;
  while (index < argc) {
    const char* arg = argv[index];
    if (std::strncmp(arg, "--precision=", 12) == 0) {
      const char* value = arg + 12;
      if (std::strcmp(value, "double") == 0 ||
          std::strcmp(value, "complex128") == 0) {
        precision = PrecisionMode::kComplex128;
      } else {
        precision = PrecisionMode::kComplex64;
      }
      ++index;
    } else if (std::strcmp(arg, "--double") == 0 ||
               std::strcmp(arg, "-d") == 0) {
      precision = PrecisionMode::kComplex128;
      ++index;
    } else if (std::strcmp(arg, "--single") == 0 ||
               std::strcmp(arg, "--float") == 0 ||
               std::strcmp(arg, "-s") == 0) {
      precision = PrecisionMode::kComplex64;
      ++index;
    } else if (std::strcmp(arg, "--help") == 0 || std::strcmp(arg, "-h") == 0) {
      printf("Usage: %s [--precision=single|double] [matrix.mtx [rhs.mtx]]\n",
             argv[0]);
      return 0;
    } else {
      break;
    }
  }

  if (index < argc) {
    input.matrix_path = argv[index++];
    input.matrix_from_cli = true;
  }
  if (index < argc) {
    input.rhs_path = argv[index++];
    input.rhs_from_cli = true;
  }
  if (index < argc) {
    fprintf(stderr,
            "Warning: ignoring %d extra argument(s) starting with '%s'.\n",
            argc - index, argv[index]);
  }

  if (precision == PrecisionMode::kComplex128) {
    return run_example<cuDoubleComplex>(input);
  }

  return run_example<cuComplex>(input);
}
