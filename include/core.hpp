/**
   @file include/core.hpp

   @brief Core components

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <config.hpp>

#include <array>
#include <fstream>
#include <iostream>
#include <tuple>
#include <vector>

#include <utils/getenv.hpp>

#ifdef IGANET_WITH_OPENMP
#include <omp.h>
#endif

#ifdef IGANET_WITH_MPI
#ifndef USE_C10D_MPI
#error "Torch must be compiled with USE_DISTRIBUTED=1, USE_MPI=1, USE_C10_MPI=1"
#endif
#include <torch/csrc/distributed/c10d/ProcessGroupMPI.hpp>
#endif

#include <torch/csrc/api/include/torch/types.h>
#include <torch/torch.h>

#ifdef IGANET_WITH_GISMO
#include <gismo.h>
#endif

#undef real_t
#undef index_t
#undef short_t

#ifdef IGANET_WITH_MATPLOT
#ifdef __CUDACC__
#pragma nv_diag_suppress 611
#endif
#include <matplot/matplot.h>
#ifdef __CUDACC__
#pragma nv_diag_default 611
#endif
#endif

#include <sysinfo.hpp>

namespace iganet {

using short_t = short int;

namespace literals {

/// @brief User-defined literals for integer values
/// @{
inline short_t operator""_s(unsigned long long value) { return value; };
inline int8_t operator""_i8(unsigned long long value) { return value; };
inline int16_t operator""_i16(unsigned long long value) { return value; };
inline int32_t operator""_i32(unsigned long long value) { return value; };
inline int64_t operator""_i64(unsigned long long value) { return value; };
/// @}
} // namespace literals

//  clang-format off
/// @brief Enumerator for specifying the initialization of B-spline coefficients
enum class log : short_t {
  none = 0,    /*!< no logging */
  fatal = 1,   /*!< log fatal errors */
  error = 2,   /*!< log errors */
  warning = 3, /*!< log warnings */
  info = 4,    /*!< log information */
  debug = 5,   /*!< log debug information */
  verbose = 6  /*!< log everything */
};
//  clang-format on

namespace logging {
/// @brief Dummy stream buffer
class NullStreamBuffer : public std::streambuf {
public:
  /// @brief Dummy output
  int overflow(int c) override { return traits_type::not_eof(c); }
};

/// @brief Dummy output stream
class NullOStream : public std::ostream {
public:
  /// @brief Constructor
  NullOStream() : std::ostream(&nullStreamBuffer) {}

private:
  NullStreamBuffer nullStreamBuffer;
};
} // namespace logging

/// @brief Logger
struct {
private:
  /// @brief Output stream
  std::ostream &outputStream = std::cout;

  /// @brief Dummy output stream
  logging::NullOStream nullStream;

  /// @brief Output file
  std::ofstream outputFile;

  /// @brief Log level
  enum log level = log::info;

public:
  /// @brief Sets the log level
  void setLogLevel(enum log level) { this->level = level; }

  /// @brief Sets the log file
  void setLogFile(std::string filename) {
    outputFile = std::ofstream(filename);
    outputStream.rdbuf(outputFile.rdbuf());
  }

  /// @brief Returns the output stream
  std::ostream &operator()(enum log level = log::info) {
    if (this->level >= level)
      switch (level) {
      case (log::fatal):
        return outputStream << "[FATAL ERROR] ";
      case (log::error):
        return outputStream << "[ERROR] ";
      case (log::warning):
        return outputStream << "[WARNING] ";
      case (log::info):
        return outputStream << "[INFO] ";
      case (log::debug):
        return outputStream << "[DEBUG] ";
      case (log::verbose):
        return outputStream << "[VERBOSE] ";
      default:
        return nullStream;
      }
    else
      return nullStream;
  }
} Log;

/// @brief Initializes the library
inline void init(std::ostream &os = Log(log::info)) {
  torch::manual_seed(1);

  // Set number of intraop thread pool threads
#ifdef IGANET_WITH_OPENMP
  at::set_num_threads(
      utils::getenv("IGANET_INTRAOP_NUM_THREADS", omp_get_max_threads()));
#else
  at::set_num_threads(utils::getenv("IGANET_INTRAOP_NUM_THREADS", 1));
#endif

  // Set number of interop thread pool threads
  at::set_num_interop_threads(utils::getenv("IGANET_INTEROP_NUM_THREADS", 1));

#ifdef IGANET_WITH_MPI
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0)
#endif
    // Output version information
    os << getVersion();
}

/// Stream manipulator
/// @{
inline int get_iomanip() {
  static int i = std::ios_base::xalloc();
  return i;
}

inline std::ostream &verbose(std::ostream &os) {
  os.iword(get_iomanip()) = 1;
  return os;
}
inline std::ostream &regular(std::ostream &os) {
  os.iword(get_iomanip()) = 0;
  return os;
}

inline bool is_verbose(std::ostream &os) {
  return os.iword(get_iomanip()) != 0;
}
/// @}

} // namespace iganet

namespace std {

/// Print (as string) an std::array of generic objects
template <typename T, std::size_t N>
inline std::ostream &operator<<(std::ostream &os, const std::array<T, N> &obj) {
  at::optional<std::string> name_ = c10::demangle(typeid(obj).name());

#if defined(_WIN32)
  // Windows adds "struct" or "class" as a prefix.
  if (name_->find("struct ") == 0) {
    name_->erase(name_->begin(), name_->begin() + 7);
  } else if (name_->find("class ") == 0) {
    name_->erase(name_->begin(), name_->begin() + 6);
  }
#endif // defined(_WIN32)

  os << *name_ << "(";
  for (const auto &i : obj)
    os << i << (&i == &(*obj.rbegin()) ? "" : ",");
  os << ")";

  return os;
}

namespace detail {
template <typename... Ts, std::size_t... Is>
inline std::ostream &output_tuple(std::ostream &os,
                                  const std::tuple<Ts...> &obj,
                                  std::index_sequence<Is...>) {
  (..., (os << std::get<Is>(obj) << "\n"));
  return os;
}

} // namespace detail

/// Print (as string) an std::tuple of generic objects
template <typename... Ts>
inline std::ostream &operator<<(std::ostream &os,
                                const std::tuple<Ts...> &obj) {
  at::optional<std::string> name_ = c10::demangle(typeid(obj).name());

#if defined(_WIN32)
  // Windows adds "struct" or "class" as a prefix.
  if (name_->find("struct ") == 0) {
    name_->erase(name_->begin(), name_->begin() + 7);
  } else if (name_->find("class ") == 0) {
    name_->erase(name_->begin(), name_->begin() + 6);
  }
#endif // defined(_WIN32)

  os << *name_ << "(\n";
  detail::output_tuple(os, obj, std::make_index_sequence<sizeof...(Ts)>());
  os << "\n)";

  return os;
}

} // namespace std
