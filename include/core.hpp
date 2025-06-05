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

#ifdef __CUDACC__
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#endif

#ifdef __HIPCC__
#include <c10/hip/HIPCachingAllocator.h>
#include <c10/hip/HIPFunctions.h>
#endif

#ifdef IGANET_WITH_GISMO
#include <gismo.h>
#include <gsModeling/gsSurfaceReparameterization.h>

#ifdef gsElasticity_ENABLED
#include <gsElasticity/src/gsElasticityAssembler.h>
#include <gsElasticity/src/gsGeoUtils.h>
#include <gsElasticity/src/gsMassAssembler.h>
#endif
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
/// @brief Enumerator for specifying the logging level
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

/// @brief Return a human-readable printout of the current memory allocator
/// statistics for a given device
inline std::string memory_summary(c10::DeviceIndex device =
#ifdef __CUDACC__
                                      c10::cuda::current_device()
#elif __HIPCC__
                                      c10::hip::current_device()
#else
                                      0
#endif
) {

  std::ostringstream os;

#if defined(__CUDACC__) || defined(__HIPCC__)

  auto _format_size = [](int64_t bytes) -> std::string {
    if (bytes == 0)
      return "0 B";

    std::array<std::string, 6> prefixes{"B", "KiB", "MiB", "GiB", "TiB", "PiB"};
    int64_t n = std::floor(std::max(0.0, std::log2(static_cast<double>(bytes) /
                                                   static_cast<double>(768))) /
                           static_cast<double>(10));

    return std::to_string((int64_t)(bytes / std::pow(1024, n))) + " " +
           prefixes[n];
  };

#if TORCH_VERSION_MAJOR > 2 ||                                                 \
    (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR > 4)
  using namespace c10::CachingDeviceAllocator;
#endif

#ifdef __CUDACC__
  using namespace c10::cuda::CUDACachingAllocator;
#elif __HIPCC__
  using namespace c10::hip::HIPCachingAllocator;
#endif

  DeviceStats deviceStats = getDeviceStats(device);

  os << "|====================================================================="
        "======|\n"
#ifdef __CUDACC__
     << "|                 LibTorch CUDA memory summary, device ID "
#elif __HIPCC__
     << "|                 LibTorch ROCm memory summary, device ID "
#endif
     << std::setw(18) << std::left << static_cast<int>(device) << "|\n"
     << "|---------------------------------------------------------------------"
        "------|\n"
#ifdef __CUDACC__
     << "|            CUDA OOMs: "
#elif __HIPCC__
     << "|            ROCm OOMs: "
#endif
     << std::setw(13) << std::left << deviceStats.num_ooms
#ifdef __CUDACC__
     << "|        cudaMalloc retries: "
#elif __HIPCC__
     << "|         hipMalloc retries: "
#endif
     << std::setw(10) << std::left << deviceStats.num_alloc_retries << "|\n"
     << "|====================================================================="
        "======|\n"
     << "|        Metric         | Cur Usage  | Peak Usage | Tot Alloc  | Tot "
        "Freed  |\n"
     << "|---------------------------------------------------------------------"
        "------|\n"
     << "| Allocated memory      | " << std::setw(10) << std::right
     << _format_size(
            deviceStats
                .allocated_bytes[static_cast<std::size_t>(StatType::AGGREGATE)]
                .current)
     << " | " << std::setw(10) << std::right
     << _format_size(
            deviceStats
                .allocated_bytes[static_cast<std::size_t>(StatType::AGGREGATE)]
                .peak)
     << " | " << std::setw(10) << std::right
     << _format_size(
            deviceStats
                .allocated_bytes[static_cast<std::size_t>(StatType::AGGREGATE)]
                .allocated)
     << " | " << std::setw(10) << std::right
     << _format_size(
            deviceStats
                .allocated_bytes[static_cast<std::size_t>(StatType::AGGREGATE)]
                .freed)
     << " |\n"
     << "|       from large pool | " << std::setw(10) << std::right
     << _format_size(
            deviceStats
                .allocated_bytes[static_cast<std::size_t>(StatType::LARGE_POOL)]
                .current)
     << " | " << std::setw(10) << std::right
     << _format_size(
            deviceStats
                .allocated_bytes[static_cast<std::size_t>(StatType::LARGE_POOL)]
                .peak)
     << " | " << std::setw(10) << std::right
     << _format_size(
            deviceStats
                .allocated_bytes[static_cast<std::size_t>(StatType::LARGE_POOL)]
                .allocated)
     << " | " << std::setw(10) << std::right
     << _format_size(
            deviceStats
                .allocated_bytes[static_cast<std::size_t>(StatType::LARGE_POOL)]
                .freed)
     << " |\n"
     << "|       from small pool | " << std::setw(10) << std::right
     << _format_size(
            deviceStats
                .allocated_bytes[static_cast<std::size_t>(StatType::SMALL_POOL)]
                .current)
     << " | " << std::setw(10) << std::right
     << _format_size(
            deviceStats
                .allocated_bytes[static_cast<std::size_t>(StatType::SMALL_POOL)]
                .peak)
     << " | " << std::setw(10) << std::right
     << _format_size(
            deviceStats
                .allocated_bytes[static_cast<std::size_t>(StatType::SMALL_POOL)]
                .allocated)
     << " | " << std::setw(10) << std::right
     << _format_size(
            deviceStats
                .allocated_bytes[static_cast<std::size_t>(StatType::SMALL_POOL)]
                .freed)
     << " |\n"
     << "|---------------------------------------------------------------------"
        "------|\n"
     << "| Active memory         | " << std::setw(10) << std::right
     << _format_size(
            deviceStats
                .active_bytes[static_cast<std::size_t>(StatType::AGGREGATE)]
                .current)
     << " | " << std::setw(10) << std::right
     << _format_size(
            deviceStats
                .active_bytes[static_cast<std::size_t>(StatType::AGGREGATE)]
                .peak)
     << " | " << std::setw(10) << std::right
     << _format_size(
            deviceStats
                .active_bytes[static_cast<std::size_t>(StatType::AGGREGATE)]
                .allocated)
     << " | " << std::setw(10) << std::right
     << _format_size(
            deviceStats
                .active_bytes[static_cast<std::size_t>(StatType::AGGREGATE)]
                .freed)
     << " |\n"
     << "|       from large pool | " << std::setw(10) << std::right
     << _format_size(
            deviceStats
                .active_bytes[static_cast<std::size_t>(StatType::LARGE_POOL)]
                .current)
     << " | " << std::setw(10) << std::right
     << _format_size(
            deviceStats
                .active_bytes[static_cast<std::size_t>(StatType::LARGE_POOL)]
                .peak)
     << " | " << std::setw(10) << std::right
     << _format_size(
            deviceStats
                .active_bytes[static_cast<std::size_t>(StatType::LARGE_POOL)]
                .allocated)
     << " | " << std::setw(10) << std::right
     << _format_size(
            deviceStats
                .active_bytes[static_cast<std::size_t>(StatType::LARGE_POOL)]
                .freed)
     << " |\n"
     << "|       from small pool | " << std::setw(10) << std::right
     << _format_size(
            deviceStats
                .active_bytes[static_cast<std::size_t>(StatType::SMALL_POOL)]
                .current)
     << " | " << std::setw(10) << std::right
     << _format_size(
            deviceStats
                .active_bytes[static_cast<std::size_t>(StatType::SMALL_POOL)]
                .peak)
     << " | " << std::setw(10) << std::right
     << _format_size(
            deviceStats
                .active_bytes[static_cast<std::size_t>(StatType::SMALL_POOL)]
                .allocated)
     << " | " << std::setw(10) << std::right
     << _format_size(
            deviceStats
                .active_bytes[static_cast<std::size_t>(StatType::SMALL_POOL)]
                .freed)
     << " |\n"
     << "|---------------------------------------------------------------------"
        "------|\n"
     << "| Requested memory      | " << std::setw(10) << std::right
     << _format_size(
            deviceStats
                .requested_bytes[static_cast<std::size_t>(StatType::AGGREGATE)]
                .current)
     << " | " << std::setw(10) << std::right
     << _format_size(
            deviceStats
                .requested_bytes[static_cast<std::size_t>(StatType::AGGREGATE)]
                .peak)
     << " | " << std::setw(10) << std::right
     << _format_size(
            deviceStats
                .requested_bytes[static_cast<std::size_t>(StatType::AGGREGATE)]
                .allocated)
     << " | " << std::setw(10) << std::right
     << _format_size(
            deviceStats
                .requested_bytes[static_cast<std::size_t>(StatType::AGGREGATE)]
                .freed)
     << " |\n"
     << "|       from large pool | " << std::setw(10) << std::right
     << _format_size(
            deviceStats
                .requested_bytes[static_cast<std::size_t>(StatType::LARGE_POOL)]
                .current)
     << " | " << std::setw(10) << std::right
     << _format_size(
            deviceStats
                .requested_bytes[static_cast<std::size_t>(StatType::LARGE_POOL)]
                .peak)
     << " | " << std::setw(10) << std::right
     << _format_size(
            deviceStats
                .requested_bytes[static_cast<std::size_t>(StatType::LARGE_POOL)]
                .allocated)
     << " | " << std::setw(10) << std::right
     << _format_size(
            deviceStats
                .requested_bytes[static_cast<std::size_t>(StatType::LARGE_POOL)]
                .freed)
     << " |\n"
     << "|       from small pool | " << std::setw(10) << std::right
     << _format_size(
            deviceStats
                .requested_bytes[static_cast<std::size_t>(StatType::SMALL_POOL)]
                .current)
     << " | " << std::setw(10) << std::right
     << _format_size(
            deviceStats
                .requested_bytes[static_cast<std::size_t>(StatType::SMALL_POOL)]
                .peak)
     << " | " << std::setw(10) << std::right
     << _format_size(
            deviceStats
                .requested_bytes[static_cast<std::size_t>(StatType::SMALL_POOL)]
                .allocated)
     << " | " << std::setw(10) << std::right
     << _format_size(
            deviceStats
                .requested_bytes[static_cast<std::size_t>(StatType::SMALL_POOL)]
                .freed)
     << " |\n"
     << "|---------------------------------------------------------------------"
        "------|\n"
     << "| GPU reserved memory   | " << std::setw(10) << std::right
     << _format_size(
            deviceStats
                .reserved_bytes[static_cast<std::size_t>(StatType::AGGREGATE)]
                .current)
     << " | " << std::setw(10) << std::right
     << _format_size(
            deviceStats
                .reserved_bytes[static_cast<std::size_t>(StatType::AGGREGATE)]
                .peak)
     << " | " << std::setw(10) << std::right
     << _format_size(
            deviceStats
                .reserved_bytes[static_cast<std::size_t>(StatType::AGGREGATE)]
                .allocated)
     << " | " << std::setw(10) << std::right
     << _format_size(
            deviceStats
                .reserved_bytes[static_cast<std::size_t>(StatType::AGGREGATE)]
                .freed)
     << " |\n"
     << "|       from large pool | " << std::setw(10) << std::right
     << _format_size(
            deviceStats
                .reserved_bytes[static_cast<std::size_t>(StatType::LARGE_POOL)]
                .current)
     << " | " << std::setw(10) << std::right
     << _format_size(
            deviceStats
                .reserved_bytes[static_cast<std::size_t>(StatType::LARGE_POOL)]
                .peak)
     << " | " << std::setw(10) << std::right
     << _format_size(
            deviceStats
                .reserved_bytes[static_cast<std::size_t>(StatType::LARGE_POOL)]
                .allocated)
     << " | " << std::setw(10) << std::right
     << _format_size(
            deviceStats
                .reserved_bytes[static_cast<std::size_t>(StatType::LARGE_POOL)]
                .freed)
     << " |\n"
     << "|       from small pool | " << std::setw(10) << std::right
     << _format_size(
            deviceStats
                .reserved_bytes[static_cast<std::size_t>(StatType::SMALL_POOL)]
                .current)
     << " | " << std::setw(10) << std::right
     << _format_size(
            deviceStats
                .reserved_bytes[static_cast<std::size_t>(StatType::SMALL_POOL)]
                .peak)
     << " | " << std::setw(10) << std::right
     << _format_size(
            deviceStats
                .reserved_bytes[static_cast<std::size_t>(StatType::SMALL_POOL)]
                .allocated)
     << " | " << std::setw(10) << std::right
     << _format_size(
            deviceStats
                .reserved_bytes[static_cast<std::size_t>(StatType::SMALL_POOL)]
                .freed)
     << " |\n"
     << "|---------------------------------------------------------------------"
        "------|\n"
     << "| Non-releasable memory | " << std::setw(10) << std::right
     << _format_size(deviceStats
                         .inactive_split_bytes[static_cast<std::size_t>(
                             StatType::AGGREGATE)]
                         .current)
     << " | " << std::setw(10) << std::right
     << _format_size(deviceStats
                         .inactive_split_bytes[static_cast<std::size_t>(
                             StatType::AGGREGATE)]
                         .peak)
     << " | " << std::setw(10) << std::right
     << _format_size(deviceStats
                         .inactive_split_bytes[static_cast<std::size_t>(
                             StatType::AGGREGATE)]
                         .allocated)
     << " | " << std::setw(10) << std::right
     << _format_size(deviceStats
                         .inactive_split_bytes[static_cast<std::size_t>(
                             StatType::AGGREGATE)]
                         .freed)
     << " |\n"
     << "|       from large pool | " << std::setw(10) << std::right
     << _format_size(deviceStats
                         .inactive_split_bytes[static_cast<std::size_t>(
                             StatType::LARGE_POOL)]
                         .current)
     << " | " << std::setw(10) << std::right
     << _format_size(deviceStats
                         .inactive_split_bytes[static_cast<std::size_t>(
                             StatType::LARGE_POOL)]
                         .peak)
     << " | " << std::setw(10) << std::right
     << _format_size(deviceStats
                         .inactive_split_bytes[static_cast<std::size_t>(
                             StatType::LARGE_POOL)]
                         .allocated)
     << " | " << std::setw(10) << std::right
     << _format_size(deviceStats
                         .inactive_split_bytes[static_cast<std::size_t>(
                             StatType::LARGE_POOL)]
                         .freed)
     << " |\n"
     << "|       from small pool | " << std::setw(10) << std::right
     << _format_size(deviceStats
                         .inactive_split_bytes[static_cast<std::size_t>(
                             StatType::SMALL_POOL)]
                         .current)
     << " | " << std::setw(10) << std::right
     << _format_size(deviceStats
                         .inactive_split_bytes[static_cast<std::size_t>(
                             StatType::SMALL_POOL)]
                         .peak)
     << " | " << std::setw(10) << std::right
     << _format_size(deviceStats
                         .inactive_split_bytes[static_cast<std::size_t>(
                             StatType::SMALL_POOL)]
                         .allocated)
     << " | " << std::setw(10) << std::right
     << _format_size(deviceStats
                         .inactive_split_bytes[static_cast<std::size_t>(
                             StatType::SMALL_POOL)]
                         .freed)
     << " |\n"
     << "|---------------------------------------------------------------------"
        "------|\n"
     << "| Allocations           | " << std::setw(10) << std::right
     << deviceStats.allocation[static_cast<std::size_t>(StatType::AGGREGATE)]
            .current
     << " | " << std::setw(10) << std::right
     << deviceStats.allocation[static_cast<std::size_t>(StatType::AGGREGATE)]
            .peak
     << " | " << std::setw(10) << std::right
     << deviceStats.allocation[static_cast<std::size_t>(StatType::AGGREGATE)]
            .allocated
     << " | " << std::setw(10) << std::right
     << deviceStats.allocation[static_cast<std::size_t>(StatType::AGGREGATE)]
            .freed
     << " |\n"
     << "|---------------------------------------------------------------------"
        "------|\n"
     << "|       from large pool | " << std::setw(10) << std::right
     << deviceStats.allocation[static_cast<std::size_t>(StatType::LARGE_POOL)]
            .current
     << " | " << std::setw(10) << std::right
     << deviceStats.allocation[static_cast<std::size_t>(StatType::LARGE_POOL)]
            .peak
     << " | " << std::setw(10) << std::right
     << deviceStats.allocation[static_cast<std::size_t>(StatType::LARGE_POOL)]
            .allocated
     << " | " << std::setw(10) << std::right
     << deviceStats.allocation[static_cast<std::size_t>(StatType::LARGE_POOL)]
            .freed
     << " |\n"
     << "|---------------------------------------------------------------------"
        "------|\n"
     << "|       from small pool | " << std::setw(10) << std::right
     << deviceStats.allocation[static_cast<std::size_t>(StatType::SMALL_POOL)]
            .current
     << " | " << std::setw(10) << std::right
     << deviceStats.allocation[static_cast<std::size_t>(StatType::SMALL_POOL)]
            .peak
     << " | " << std::setw(10) << std::right
     << deviceStats.allocation[static_cast<std::size_t>(StatType::SMALL_POOL)]
            .allocated
     << " | " << std::setw(10) << std::right
     << deviceStats.allocation[static_cast<std::size_t>(StatType::SMALL_POOL)]
            .freed
     << " |\n"
     << "|---------------------------------------------------------------------"
        "------|\n"
     << "| Active allocs         | " << std::setw(10) << std::right
     << deviceStats.active[static_cast<std::size_t>(StatType::AGGREGATE)]
            .current
     << " | " << std::setw(10) << std::right
     << deviceStats.active[static_cast<std::size_t>(StatType::AGGREGATE)].peak
     << " | " << std::setw(10) << std::right
     << deviceStats.active[static_cast<std::size_t>(StatType::AGGREGATE)]
            .allocated
     << " | " << std::setw(10) << std::right
     << deviceStats.active[static_cast<std::size_t>(StatType::AGGREGATE)].freed
     << " |\n"
     << "|---------------------------------------------------------------------"
        "------|\n"
     << "|       from large pool | " << std::setw(10) << std::right
     << deviceStats.active[static_cast<std::size_t>(StatType::LARGE_POOL)]
            .current
     << " | " << std::setw(10) << std::right
     << deviceStats.active[static_cast<std::size_t>(StatType::LARGE_POOL)].peak
     << " | " << std::setw(10) << std::right
     << deviceStats.active[static_cast<std::size_t>(StatType::LARGE_POOL)]
            .allocated
     << " | " << std::setw(10) << std::right
     << deviceStats.active[static_cast<std::size_t>(StatType::LARGE_POOL)].freed
     << " |\n"
     << "|---------------------------------------------------------------------"
        "------|\n"
     << "|       from small pool | " << std::setw(10) << std::right
     << deviceStats.active[static_cast<std::size_t>(StatType::SMALL_POOL)]
            .current
     << " | " << std::setw(10) << std::right
     << deviceStats.active[static_cast<std::size_t>(StatType::SMALL_POOL)].peak
     << " | " << std::setw(10) << std::right
     << deviceStats.active[static_cast<std::size_t>(StatType::SMALL_POOL)]
            .allocated
     << " | " << std::setw(10) << std::right
     << deviceStats.active[static_cast<std::size_t>(StatType::SMALL_POOL)].freed
     << " |\n"
     << "|---------------------------------------------------------------------"
        "------|\n"
     << "| GPU reserved segments | " << std::setw(10) << std::right
     << deviceStats.segment[static_cast<std::size_t>(StatType::AGGREGATE)]
            .current
     << " | " << std::setw(10) << std::right
     << deviceStats.segment[static_cast<std::size_t>(StatType::AGGREGATE)].peak
     << " | " << std::setw(10) << std::right
     << deviceStats.segment[static_cast<std::size_t>(StatType::AGGREGATE)]
            .allocated
     << " | " << std::setw(10) << std::right
     << deviceStats.segment[static_cast<std::size_t>(StatType::AGGREGATE)].freed
     << " |\n"
     << "|---------------------------------------------------------------------"
        "------|\n"
     << "|       from large pool | " << std::setw(10) << std::right
     << deviceStats.segment[static_cast<std::size_t>(StatType::LARGE_POOL)]
            .current
     << " | " << std::setw(10) << std::right
     << deviceStats.segment[static_cast<std::size_t>(StatType::LARGE_POOL)].peak
     << " | " << std::setw(10) << std::right
     << deviceStats.segment[static_cast<std::size_t>(StatType::LARGE_POOL)]
            .allocated
     << " | " << std::setw(10) << std::right
     << deviceStats.segment[static_cast<std::size_t>(StatType::LARGE_POOL)]
            .freed
     << " |\n"
     << "|---------------------------------------------------------------------"
        "------|\n"
     << "|       from small pool | " << std::setw(10) << std::right
     << deviceStats.segment[static_cast<std::size_t>(StatType::SMALL_POOL)]
            .current
     << " | " << std::setw(10) << std::right
     << deviceStats.segment[static_cast<std::size_t>(StatType::SMALL_POOL)].peak
     << " | " << std::setw(10) << std::right
     << deviceStats.segment[static_cast<std::size_t>(StatType::SMALL_POOL)]
            .allocated
     << " | " << std::setw(10) << std::right
     << deviceStats.segment[static_cast<std::size_t>(StatType::SMALL_POOL)]
            .freed
     << " |\n"
     << "|---------------------------------------------------------------------"
        "------|\n"
     << "| Non-releasable allocs | " << std::setw(10) << std::right
     << deviceStats
            .inactive_split[static_cast<std::size_t>(StatType::AGGREGATE)]
            .current
     << " | " << std::setw(10) << std::right
     << deviceStats
            .inactive_split[static_cast<std::size_t>(StatType::AGGREGATE)]
            .peak
     << " | " << std::setw(10) << std::right
     << deviceStats
            .inactive_split[static_cast<std::size_t>(StatType::AGGREGATE)]
            .allocated
     << " | " << std::setw(10) << std::right
     << deviceStats
            .inactive_split[static_cast<std::size_t>(StatType::AGGREGATE)]
            .freed
     << " |\n"
     << "|---------------------------------------------------------------------"
        "------|\n"
     << "|       from large pool | " << std::setw(10) << std::right
     << deviceStats
            .inactive_split[static_cast<std::size_t>(StatType::LARGE_POOL)]
            .current
     << " | " << std::setw(10) << std::right
     << deviceStats
            .inactive_split[static_cast<std::size_t>(StatType::LARGE_POOL)]
            .peak
     << " | " << std::setw(10) << std::right
     << deviceStats
            .inactive_split[static_cast<std::size_t>(StatType::LARGE_POOL)]
            .allocated
     << " | " << std::setw(10) << std::right
     << deviceStats
            .inactive_split[static_cast<std::size_t>(StatType::LARGE_POOL)]
            .freed
     << " |\n"
     << "|---------------------------------------------------------------------"
        "------|\n"
     << "|       from small pool | " << std::setw(10) << std::right
     << deviceStats
            .inactive_split[static_cast<std::size_t>(StatType::SMALL_POOL)]
            .current
     << " | " << std::setw(10) << std::right
     << deviceStats
            .inactive_split[static_cast<std::size_t>(StatType::SMALL_POOL)]
            .peak
     << " | " << std::setw(10) << std::right
     << deviceStats
            .inactive_split[static_cast<std::size_t>(StatType::SMALL_POOL)]
            .allocated
     << " | " << std::setw(10) << std::right
     << deviceStats
            .inactive_split[static_cast<std::size_t>(StatType::SMALL_POOL)]
            .freed
     << " |\n"
     << "|---------------------------------------------------------------------"
        "------|\n"
     << "| Oversize allocations  | " << std::setw(10) << std::right
     << deviceStats.oversize_allocations.current << " | " << std::setw(10)
     << std::right << deviceStats.oversize_allocations.peak << " | "
     << std::setw(10) << std::right
     << deviceStats.oversize_allocations.allocated << " | " << std::setw(10)
     << std::right << deviceStats.oversize_allocations.freed << " |\n"
     << "|---------------------------------------------------------------------"
        "------|\n"
     << "| Oversize GPU segments | " << std::setw(10) << std::right
     << deviceStats.oversize_segments.current << " | " << std::setw(10)
     << std::right << deviceStats.oversize_segments.peak << " | "
     << std::setw(10) << std::right << deviceStats.oversize_segments.allocated
     << " | " << std::setw(10) << std::right
     << deviceStats.oversize_segments.freed << " |\n"
     << "|====================================================================="
        "======|";
#else
  os << "Memory summary is only available for CUDA/HIP devices";
#endif

  return os.str();
}

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
  int flag;
  MPI_Initialized(&flag);

  if (flag == 0)
    if (MPI_Init(NULL, NULL) != MPI_SUCESS)
      throw std::runtime_error("An error occured during MPI initialization");

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0)
#endif
    // Output version information
    os << getVersion();
}

/// @brief Finalizes the library
inline void finalize(std::ostream &os = Log(log::info)) {

#if defined(__CUDACC__) || defined(__HIPCC__)
  std::cout << "\n" << memory_summary() << std::endl;
#endif

#ifdef IGANET_WITH_MPI
  if (MPI_Finalize() != MPI_SUCCESS)
    throw std::runtime_error("An error occured during MPI finalization");
#endif

  os << "Succeeded\n";
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
