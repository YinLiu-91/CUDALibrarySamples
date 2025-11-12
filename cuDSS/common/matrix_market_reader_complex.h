/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
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

#ifndef CUDALIBRARYSAMPLES_CUDSS_COMMON_MATRIX_MARKET_READER_COMPLEX_H_
#define CUDALIBRARYSAMPLES_CUDSS_COMMON_MATRIX_MARKET_READER_COMPLEX_H_

#include <cuComplex.h>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "cudss.h"

namespace matrix_market {

enum class Status {
  kSuccess = 0,
  kFileNotFound,
  kInvalidHeader,
  kInvalidSizeLine,
  kInvalidEntry,
  kHostAllocationFailed,
  kDeclaredNnzMismatch,
  kUnexpectedLowerEntry,
  kUnexpectedUpperEntry,
  kRowIndexOutOfBounds,
  kColumnIndexOutOfBounds,
  kUnsupportedValueType,
  kRhsSizeMismatch,
  kRhsInvalidEntry
};

inline const char* StatusToString(Status status) {
  switch (status) {
    case Status::kSuccess:
      return "success";
    case Status::kFileNotFound:
      return "file not found";
    case Status::kInvalidHeader:
      return "invalid or unsupported header";
    case Status::kInvalidSizeLine:
      return "invalid matrix size line";
    case Status::kInvalidEntry:
      return "invalid matrix entry";
    case Status::kHostAllocationFailed:
      return "host allocation failed";
    case Status::kDeclaredNnzMismatch:
      return "declared nnz mismatch";
    case Status::kUnexpectedLowerEntry:
      return "unexpected lower triangle entry";
    case Status::kUnexpectedUpperEntry:
      return "unexpected upper triangle entry";
    case Status::kRowIndexOutOfBounds:
      return "row index out of bounds";
    case Status::kColumnIndexOutOfBounds:
      return "column index out of bounds";
    case Status::kUnsupportedValueType:
      return "unsupported value type";
    case Status::kRhsSizeMismatch:
      return "rhs size mismatch";
    case Status::kRhsInvalidEntry:
      return "invalid rhs entry";
  }
  return "unknown error";
}

struct MatrixMetadata {
  cudssMatrixType_t mtype = CUDSS_MTYPE_GENERAL;
  cudssMatrixViewType_t detected_view = CUDSS_MVIEW_FULL;
  bool is_hermitian = false;
  bool values_complex = false;
  bool has_lower = false;
  bool has_upper = false;
};

struct MatrixReadOptions {
  cudssMatrixViewType_t requested_view = CUDSS_MVIEW_FULL;
  bool allow_real_as_complex = true;
  bool enforce_declared_nnz = false;
};

template <typename Value>
struct ValueTraits;

template <>
struct ValueTraits<double> {
  using value_type = double;
  using real_type = double;

  static constexpr bool kIsComplex = false;
  static constexpr const char* kExpectedToken = "real";
  static constexpr const char* kAltToken = nullptr;

  static value_type make(real_type real_part, real_type /*imag_part*/) {
    return real_part;
  }
};

template <>
struct ValueTraits<float> {
  using value_type = float;
  using real_type = float;

  static constexpr bool kIsComplex = false;
  static constexpr const char* kExpectedToken = "real";
  static constexpr const char* kAltToken = nullptr;

  static value_type make(real_type real_part, real_type /*imag_part*/) {
    return real_part;
  }
};

template <>
struct ValueTraits<cuComplex> {
  using value_type = cuComplex;
  using real_type = float;

  static constexpr bool kIsComplex = true;
  static constexpr const char* kExpectedToken = "complex";
  static constexpr const char* kAltToken = "real";

  static value_type make(double real_part, double imag_part) {
    return make_cuComplex(static_cast<float>(real_part),
                          static_cast<float>(imag_part));
  }
};

template <>
struct ValueTraits<cuDoubleComplex> {
  using value_type = cuDoubleComplex;
  using real_type = double;

  static constexpr bool kIsComplex = true;
  static constexpr const char* kExpectedToken = "complex";
  static constexpr const char* kAltToken = "real";

  static value_type make(double real_part, double imag_part) {
    return make_cuDoubleComplex(real_part, imag_part);
  }
};

template <typename Value>
inline Status read_matrix_coordinate(const std::string& filename, int& n,
                                     int& nnz, int** csr_offsets_h,
                                     int** csr_columns_h, Value** csr_values_h,
                                     MatrixMetadata* metadata = nullptr,
                                     MatrixReadOptions options = {}) {
  using Traits = ValueTraits<Value>;

  *csr_offsets_h = nullptr;
  *csr_columns_h = nullptr;
  *csr_values_h = nullptr;
  n = 0;
  nnz = 0;

  std::ifstream file(filename);
  if (!file.is_open()) {
    std::fprintf(stderr, "Error: Could not open matrix file %s\n",
                 filename.c_str());
    return Status::kFileNotFound;
  }

  std::string line;
  bool header_found = false;
  bool size_found = false;
  int declared_nnz = 0;
  std::string value_token = "";
  std::string symmetry = "general";

  std::vector<std::tuple<int, int, Value>> entries;
  bool found_lower = false;
  bool found_upper = false;

  while (std::getline(file, line)) {
    if (line.empty()) {
      continue;
    }

    if (!header_found) {
      if (line.rfind("%%MatrixMarket", 0) == 0) {
        header_found = true;
        std::istringstream header_stream(line);
        std::string marker, object, format;
        header_stream >> marker >> object >> format >> value_token >> symmetry;
        if (object != "matrix" || format != "coordinate") {
          std::fprintf(stderr,
                       "Error: Unsupported Matrix Market header in %s. "
                       "Expected 'matrix coordinate'.\n",
                       filename.c_str());
          return Status::kInvalidHeader;
        }

        std::string value_token_lower = value_token;
        std::transform(value_token_lower.begin(), value_token_lower.end(),
                       value_token_lower.begin(),
                       [](unsigned char c) { return std::tolower(c); });

        const bool token_matches_expected =
            value_token_lower == Traits::kExpectedToken;
        const bool token_matches_alt = Traits::kAltToken != nullptr &&
                                       value_token_lower == Traits::kAltToken;

        if (!token_matches_expected &&
            !(Traits::kIsComplex && options.allow_real_as_complex &&
              token_matches_alt)) {
          std::fprintf(stderr, "Error: Unsupported value type '%s' in %s.\n",
                       value_token.c_str(), filename.c_str());
          return Status::kUnsupportedValueType;
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

    std::istringstream data_stream(line);
    if (!size_found) {
      int ncols = 0;
      data_stream >> n >> ncols >> declared_nnz;
      if (!data_stream || ncols != n) {
        std::fprintf(stderr,
                     "Error: Matrix in %s must be square. Parsed n=%d, m=%d.\n",
                     filename.c_str(), n, ncols);
        return Status::kInvalidSizeLine;
      }
      size_found = true;
    } else {
      int row = 0;
      int col = 0;
      double real_part = 0.0;
      double imag_part = 0.0;
      data_stream >> row >> col >> real_part;
      if (!data_stream) {
        std::fprintf(stderr, "Error: Invalid entry in %s.\n", filename.c_str());
        return Status::kInvalidEntry;
      }

      std::string value_token_lower = value_token;
      std::transform(value_token_lower.begin(), value_token_lower.end(),
                     value_token_lower.begin(),
                     [](unsigned char c) { return std::tolower(c); });

      if (Traits::kIsComplex && value_token_lower == "complex") {
        if (!(data_stream >> imag_part)) {
          std::fprintf(stderr, "Error: Missing imaginary part in %s.\n",
                       filename.c_str());
          return Status::kInvalidEntry;
        }
      } else if (Traits::kIsComplex && value_token_lower == "real") {
        if (!(data_stream >> imag_part)) {
          imag_part = 0.0;
        }
      } else {
        if (!(data_stream >> imag_part)) {
          imag_part = 0.0;
        }
      }

      row -= 1;
      col -= 1;
      entries.emplace_back(row, col, Traits::make(real_part, imag_part));
      if (row < col) {
        found_upper = true;
      } else if (row > col) {
        found_lower = true;
      }
    }
  }
  file.close();

  if (!header_found || !size_found) {
    std::fprintf(stderr, "Error: Incomplete Matrix Market file %s.\n",
                 filename.c_str());
    return Status::kInvalidHeader;
  }

  if (options.enforce_declared_nnz &&
      declared_nnz != static_cast<int>(entries.size())) {
    std::fprintf(stderr, "Error: Declared nnz=%d but read %zu entries in %s.\n",
                 declared_nnz, entries.size(), filename.c_str());
    return Status::kDeclaredNnzMismatch;
  } else if (declared_nnz != static_cast<int>(entries.size())) {
    std::fprintf(
        stderr,
        "Warning: Declared nnz=%d but read %zu entries in %s. Continuing.\n",
        declared_nnz, entries.size(), filename.c_str());
  }

  nnz = static_cast<int>(entries.size());
  if (nnz == 0) {
    std::fprintf(stderr, "Error: Matrix file %s contains no entries.\n",
                 filename.c_str());
    return Status::kInvalidEntry;
  }

  const bool request_upper = options.requested_view == CUDSS_MVIEW_UPPER;
  const bool request_lower = options.requested_view == CUDSS_MVIEW_LOWER;

  if (request_upper && found_lower) {
    std::fprintf(
        stderr,
        "Error: Requested upper view but lower triangle entries found in %s.\n",
        filename.c_str());
    return Status::kUnexpectedLowerEntry;
  }

  if (request_lower && found_upper) {
    std::fprintf(
        stderr,
        "Error: Requested lower view but upper triangle entries found in %s.\n",
        filename.c_str());
    return Status::kUnexpectedUpperEntry;
  }

  std::string symmetry_lower = symmetry;
  std::transform(symmetry_lower.begin(), symmetry_lower.end(),
                 symmetry_lower.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  MatrixMetadata meta;
  meta.values_complex = (value_token == "complex");
  meta.has_lower = found_lower;
  meta.has_upper = found_upper;

  if (symmetry_lower == "general") {
    meta.mtype = CUDSS_MTYPE_GENERAL;
    meta.is_hermitian = false;
    meta.detected_view = CUDSS_MVIEW_FULL;
  } else if (symmetry_lower == "symmetric") {
    meta.mtype = CUDSS_MTYPE_SYMMETRIC;
    meta.is_hermitian = false;
    meta.detected_view =
        (found_lower && !found_upper) ? CUDSS_MVIEW_LOWER : CUDSS_MVIEW_UPPER;
  } else if (symmetry_lower == "hermitian") {
    meta.mtype = CUDSS_MTYPE_HERMITIAN;
    meta.is_hermitian = true;
    meta.detected_view =
        (found_lower && !found_upper) ? CUDSS_MVIEW_LOWER : CUDSS_MVIEW_UPPER;
  } else {
    std::fprintf(stderr, "Error: Unsupported symmetry '%s' in %s.\n",
                 symmetry.c_str(), filename.c_str());
    return Status::kInvalidHeader;
  }

  if (options.requested_view == CUDSS_MVIEW_FULL) {
    meta.detected_view = meta.detected_view;
  } else {
    meta.detected_view = options.requested_view;
  }

  if (metadata) {
    *metadata = meta;
  }

  *csr_offsets_h = static_cast<int*>(std::malloc((n + 1) * sizeof(int)));
  *csr_columns_h = static_cast<int*>(std::malloc(nnz * sizeof(int)));
  *csr_values_h = static_cast<Value*>(std::malloc(nnz * sizeof(Value)));

  if (!(*csr_offsets_h) || !(*csr_columns_h) || !(*csr_values_h)) {
    std::fprintf(stderr,
                 "Error: Host memory allocation failed while reading %s.\n",
                 filename.c_str());
    std::free(*csr_offsets_h);
    std::free(*csr_columns_h);
    std::free(*csr_values_h);
    *csr_offsets_h = nullptr;
    *csr_columns_h = nullptr;
    *csr_values_h = nullptr;
    return Status::kHostAllocationFailed;
  }

  std::fill(*csr_offsets_h, *csr_offsets_h + (n + 1), 0);

  std::sort(entries.begin(), entries.end(),
            [](const std::tuple<int, int, Value>& a,
               const std::tuple<int, int, Value>& b) {
              if (std::get<0>(a) != std::get<0>(b))
                return std::get<0>(a) < std::get<0>(b);
              return std::get<1>(a) < std::get<1>(b);
            });

  int current_idx = 0;
  for (const auto& entry : entries) {
    const int row = std::get<0>(entry);
    const int col = std::get<1>(entry);
    if (row < 0 || row >= n) {
      std::fprintf(stderr, "Error: Entry (%d,%d) out of bounds in %s.\n", row,
                   col, filename.c_str());
      std::free(*csr_offsets_h);
      std::free(*csr_columns_h);
      std::free(*csr_values_h);
      *csr_offsets_h = nullptr;
      *csr_columns_h = nullptr;
      *csr_values_h = nullptr;
      return Status::kRowIndexOutOfBounds;
    }
    if (col < 0 || col >= n) {
      std::fprintf(stderr, "Error: Entry (%d,%d) out of bounds in %s.\n", row,
                   col, filename.c_str());
      std::free(*csr_offsets_h);
      std::free(*csr_columns_h);
      std::free(*csr_values_h);
      *csr_offsets_h = nullptr;
      *csr_columns_h = nullptr;
      *csr_values_h = nullptr;
      return Status::kColumnIndexOutOfBounds;
    }
    (*csr_offsets_h)[row + 1]++;
    (*csr_columns_h)[current_idx] = col;
    (*csr_values_h)[current_idx] = std::get<2>(entry);
    ++current_idx;
  }

  for (int i = 0; i < n; ++i) {
    (*csr_offsets_h)[i + 1] += (*csr_offsets_h)[i];
  }

  return Status::kSuccess;
}

template <typename Value>
inline Status read_rhs_array(const std::string& filename, int expected_n,
                             Value** b_values_h,
                             bool allow_real_as_complex = true) {
  using Traits = ValueTraits<Value>;

  *b_values_h = nullptr;

  std::ifstream file(filename);
  if (!file.is_open()) {
    std::fprintf(stderr, "Error: Could not open RHS file %s\n",
                 filename.c_str());
    return Status::kFileNotFound;
  }

  std::string line;
  bool header_found = false;
  bool size_found = false;
  std::string value_token = "";

  std::vector<Value> values;

  while (std::getline(file, line)) {
    if (line.empty()) {
      continue;
    }
    if (!header_found) {
      if (line.rfind("%%MatrixMarket", 0) == 0) {
        header_found = true;
        std::istringstream header_stream(line);
        std::string marker, object, format, symmetry;
        header_stream >> marker >> object >> format >> value_token >> symmetry;
        if (object != "matrix" || format != "array") {
          std::fprintf(
              stderr,
              "Error: Unsupported RHS header in %s. Expected 'matrix array'.\n",
              filename.c_str());
          return Status::kInvalidHeader;
        }

        std::string value_token_lower = value_token;
        std::transform(value_token_lower.begin(), value_token_lower.end(),
                       value_token_lower.begin(),
                       [](unsigned char c) { return std::tolower(c); });

        const bool token_matches_expected =
            value_token_lower == Traits::kExpectedToken;
        const bool token_matches_alt = Traits::kAltToken != nullptr &&
                                       value_token_lower == Traits::kAltToken;

        if (!token_matches_expected &&
            !(Traits::kIsComplex && allow_real_as_complex &&
              token_matches_alt)) {
          std::fprintf(stderr,
                       "Error: Unsupported RHS value type '%s' in %s.\n",
                       value_token.c_str(), filename.c_str());
          return Status::kUnsupportedValueType;
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

    std::istringstream data_stream(line);
    if (!size_found) {
      int rows = 0;
      int cols = 0;
      data_stream >> rows >> cols;
      if (!data_stream || cols != 1) {
        std::fprintf(stderr,
                     "Error: RHS in %s must be a single column vector.\n",
                     filename.c_str());
        return Status::kInvalidSizeLine;
      }
      if (rows != expected_n) {
        std::fprintf(
            stderr,
            "Error: RHS size mismatch: matrix has %d rows but RHS has %d.\n",
            expected_n, rows);
        return Status::kRhsSizeMismatch;
      }
      values.reserve(rows);
      size_found = true;
    } else {
      double real_part = 0.0;
      double imag_part = 0.0;
      data_stream >> real_part;
      if (!data_stream) {
        std::fprintf(stderr, "Error: Invalid RHS entry in %s.\n",
                     filename.c_str());
        return Status::kRhsInvalidEntry;
      }

      std::string value_token_lower = value_token;
      std::transform(value_token_lower.begin(), value_token_lower.end(),
                     value_token_lower.begin(),
                     [](unsigned char c) { return std::tolower(c); });

      if (Traits::kIsComplex && value_token_lower == "complex") {
        if (!(data_stream >> imag_part)) {
          std::fprintf(stderr, "Error: Missing RHS imaginary part in %s.\n",
                       filename.c_str());
          return Status::kRhsInvalidEntry;
        }
      } else if (Traits::kIsComplex && value_token_lower == "real") {
        if (!(data_stream >> imag_part)) {
          imag_part = 0.0;
        }
      } else {
        if (!(data_stream >> imag_part)) {
          imag_part = 0.0;
        }
      }

      values.push_back(ValueTraits<Value>::make(real_part, imag_part));
    }
  }
  file.close();

  if (!size_found || static_cast<int>(values.size()) != expected_n) {
    std::fprintf(stderr, "Error: RHS file %s does not have %d entries.\n",
                 filename.c_str(), expected_n);
    return Status::kRhsSizeMismatch;
  }

  *b_values_h = static_cast<Value*>(std::malloc(expected_n * sizeof(Value)));
  if (!(*b_values_h)) {
    std::fprintf(stderr, "Error: Host memory allocation failed for RHS.\n");
    return Status::kHostAllocationFailed;
  }

  std::copy(values.begin(), values.end(), *b_values_h);
  return Status::kSuccess;
}

}  // namespace matrix_market

#endif  // CUDALIBRARYSAMPLES_CUDSS_COMMON_MATRIX_MARKET_READER_COMPLEX_H_
