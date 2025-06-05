/**
   @file include/utils/tensorarray.hpp

   @brief TensorArray utility functions

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <array>
#include <initializer_list>

#include <options.hpp>
#include <utils/container.hpp>

#include <torch/torch.h>

namespace iganet {
namespace utils {

template <std::size_t N> using TensorArray = std::array<torch::Tensor, N>;

using TensorArray0 = TensorArray<0>;
using TensorArray1 = TensorArray<1>;
using TensorArray2 = TensorArray<2>;
using TensorArray3 = TensorArray<3>;
using TensorArray4 = TensorArray<4>;

/// @brief Converts a set of std::initializer_list objects to a TensorArray
/// object
/// @{
template <typename... Ts>
inline constexpr TensorArray<sizeof...(Ts)>
to_tensorArray(std::initializer_list<Ts> &&...lists) {
  return {to_tensor(std::forward<std::initializer_list<Ts>>(lists),
                    torch::IntArrayRef{-1}, Options<Ts>{})...};
}

template <typename... Ts>
inline constexpr TensorArray<sizeof...(Ts)>
to_tensorArray(torch::IntArrayRef sizes, std::initializer_list<Ts> &&...lists) {
  return {to_tensor(std::forward<std::initializer_list<Ts>>(lists), sizes,
                    Options<Ts>{})...};
}

template <typename... Ts, typename T>
inline constexpr TensorArray<sizeof...(Ts)>
to_tensorArray(const iganet::Options<T> &options,
               std::initializer_list<Ts> &&...lists) {
  static_assert(
      (std::is_same_v<T, Ts> && ...),
      "Type mismatch between Options<T> and std::initializer_list<Ts>");
  return {to_tensor(std::forward<std::initializer_list<Ts>>(lists),
                    torch::IntArrayRef{-1}, options)...};
}

template <typename... Ts, typename T>
inline constexpr TensorArray<sizeof...(Ts)>
to_tensorArray(torch::IntArrayRef sizes, const iganet::Options<T> &options,
               std::initializer_list<Ts> &&...lists) {
  static_assert(
      (std::is_same_v<T, Ts> && ...),
      "Type mismatch between Options<T> and std::initializer_list<Ts>");
  return {to_tensor(std::forward<std::initializer_list<Ts>>(lists), sizes,
                    options)...};
}
/// @}

/// @brief Converts a torch::Tensor object to a
/// torch::TensorAccessor object
/// @{
template <typename T, std::size_t N>
auto to_tensorAccessor(const torch::Tensor &tensor) {
  return tensor.template accessor<T, N>();
}

template <typename T, std::size_t N>
auto to_tensorAccessor(const torch::Tensor &tensor,
                       c10::DeviceType deviceType) {

  if (deviceType != tensor.device().type()) {
    auto tensor_device = tensor.to(deviceType);
    auto accessor = tensor_device.template accessor<T, N>();
    return std::tuple(tensor_device, accessor);
  } else {
    auto accessor = tensor.template accessor<T, N>();
    return std::tuple(tensor, accessor);
  }
}
/// @}

namespace detail {
/// @brief Converts an std::array of torch::Tensor objects to an
/// array of torch::TensorAccessor objects
/// @{
template <typename T, std::size_t N, std::size_t... Is>
auto to_tensorAccessor(const TensorArray<sizeof...(Is)> &tensorArray,
                       std::index_sequence<Is...>) {
  return std::array<torch::TensorAccessor<T, N>, sizeof...(Is)>{
      tensorArray[Is].template accessor<T, N>()...};
}

template <typename T, std::size_t N, std::size_t... Is>
auto to_tensorAccessor(const TensorArray<sizeof...(Is)> &tensorArray,
                       c10::DeviceType deviceType, std::index_sequence<Is...>) {
  std::array<torch::TensorBase, sizeof...(Is)> tensorArray_device{
      tensorArray[Is].to(deviceType)...};
  std::array<torch::TensorAccessor<T, N>, sizeof...(Is)> accessors{
      tensorArray_device[Is].template accessor<T, N>()...};
  return std::tuple(tensorArray_device, accessors);
}

template <typename T, std::size_t N, size_t... Dims, std::size_t... Is>
auto to_tensorAccessor(const BlockTensor<torch::Tensor, Dims...> &blocktensor,
                       c10::DeviceType deviceType, std::index_sequence<Is...>) {
  std::array<torch::TensorBase, sizeof...(Is)> tensorArray_device{
      blocktensor[Is]->to(deviceType)...};
  std::array<torch::TensorAccessor<T, N>, sizeof...(Is)> accessors{
      tensorArray_device[Is].template accessor<T, N>()...};
  return std::tuple(tensorArray_device, accessors);
}
/// @}
} // namespace detail

/// @brief Converts an std::array of torch::Tensor objects to an
/// array of torch::TensorAccessor objects
/// @{
template <typename T, std::size_t N, std::size_t M>
auto to_tensorAccessor(const TensorArray<M> &tensorArray) {
  return detail::to_tensorAccessor<T, N>(tensorArray,
                                         std::make_index_sequence<M>());
}

template <typename T, std::size_t N, std::size_t M>
auto to_tensorAccessor(const TensorArray<M> &tensorArray,
                       c10::DeviceType deviceType) {
  return detail::to_tensorAccessor<T, N>(tensorArray, deviceType,
                                         std::make_index_sequence<M>());
}

template <typename T, std::size_t N, std::size_t... Dims>
auto to_tensorAccessor(const BlockTensor<torch::Tensor, Dims...> &blocktensor,
                       c10::DeviceType deviceType) {
  return detail::to_tensorAccessor<T, N, Dims...>(
      blocktensor, deviceType, std::make_index_sequence<(Dims * ...)>());
}
/// @}

} // namespace utils
} // namespace iganet

#define TENSORARRAY_FORALL(obj, func, ...)                                     \
  []<std::size_t N>(const ::iganet::utils::TensorArray<N> &tensorArray) {      \
    ::iganet::utils::TensorArray<N> result;                                    \
    for (std::size_t i = 0; i < N; ++i)                                        \
      result[i] = tensorArray[i].func(__VA_ARGS__);                            \
    return result;                                                             \
  }(obj)

namespace std {

/// Print (as string) a TensorArray object
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

#ifdef __CUDACC__
#pragma nv_diag_suppress 186
#endif

  os << *name_ << "(\n";
  for (std::size_t i = 0; i < N; ++i) {
    os << obj[i] << "\n";

    if (iganet::is_verbose(os))
      os << "[ " << obj[i].options() << " ]\n";
  }

#ifdef __CUDACC__
#pragma nv_diag_default 186
#endif

  os << ")";

  return os;
}

} // namespace std
