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
#include <tuple>
#include <vector>

#if _OPENMP
#include <omp.h>
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
inline short_t operator""_s(unsigned long long value) { return value; };
inline int8_t operator""_i8(unsigned long long value) { return value; };
inline int16_t operator""_i16(unsigned long long value) { return value; };
inline int32_t operator""_i32(unsigned long long value) { return value; };
inline int64_t operator""_i64(unsigned long long value) { return value; };
} // namespace literals

/// @brief Get environment variable
template <typename T>
inline T getenv(const std::string &variable_name, const T &default_value = {}) {
  const char *value = std::getenv(variable_name.c_str());

  if (value) {
    T v;
    std::istringstream(value) >> v;
    return v;
  } else
    return default_value;
}

/// @brief Initializes the library
inline void init(std::ostream &os = std::clog) {
  torch::manual_seed(1);

  // Set number of intraop thread pool threads
#if _OPENMP
  at::set_num_threads(
      getenv("IGANET_INTRAOP_NUM_THREADS", omp_get_max_threads()));
#else
  at::set_num_threads(getenv("IGANET_INTRAOP_NUM_THREADS", 1));
#endif

  // Set number of interop thread pool threads
  at::set_num_interop_threads(getenv("IGANET_INTEROP_NUM_THREADS", 1));

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

/// Print (as string) an std::array of torch::Tensor objects
template <std::size_t N>
inline std::ostream &operator<<(std::ostream &os,
                                const std::array<torch::Tensor, N> &obj) {
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
  for (const auto &i : obj)
    if (!i.numel())
      os << "{}\n";
    else
      os << ((i.sizes().size() == 1) ? i.view({1, i.size(0)}) : i) << "\n";
  os << ")";

  return os;
}

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
