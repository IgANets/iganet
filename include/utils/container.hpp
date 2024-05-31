/**
   @file include/utils/container.hpp

   @brief Container utility functions

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <array>
#include <initializer_list>
#include <vector>

#include <options.hpp>

#include <torch/torch.h>

namespace iganet {
namespace utils {

/// @brief Converts an std::vector object into std::array
template <std::size_t N, typename T>
inline std::array<T, N> to_array(std::vector<T> &&vector) {
  std::array<T, N> array;
  std::move(vector.begin(), vector.end(), array.begin());
  return array;
}

/// @brief Converts an std::array object into std::vector
template <typename T, std::size_t N>
inline std::vector<T> to_vector(std::array<T, N> &&array) {
  std::vector<T> vector;
  std::move(array.begin(), array.end(), vector.begin());
  return vector;
}

/// @brief Converts a list of arguments into std::array
template <typename... Args> inline auto to_array(Args &&...args) {
  return std::array<typename std::common_type<Args...>::type, sizeof...(Args)>{
      std::move(args)...};
}

/// @brief Converts a list of arguments into std::vector
template <typename... Args> inline auto to_vector(Args &&...args) {
  return std::vector<typename std::common_type<Args...>::type>{
      std::move(args)...};
}

/// @brief Converts an std::array to torch::Tensor
/// @{
template <typename T, std::size_t N>
inline auto
to_tensor(const std::array<T, N> &array,
          torch::IntArrayRef sizes = torch::IntArrayRef{-1},
          const iganet::Options<T> &options = iganet::Options<T>{}) {
  if (options.device() == torch::kCPU)
    return torch::from_blob(const_cast<T *>(std::data(array)),
                            (sizes == torch::IntArrayRef{-1}) ? array.size()
                                                              : sizes,
                            options)
        .detach()
        .clone()
        .requires_grad_(options.requires_grad());
  else
    return torch::from_blob(const_cast<T *>(std::data(array)),
                            (sizes == torch::IntArrayRef{-1}) ? array.size()
                                                              : sizes,
                            options.device(torch::kCPU))
        .detach()
        .clone()
        .to(options.device())
        .requires_grad_(options.requires_grad());
}

template <typename T, std::size_t N>
inline auto to_tensor(const std::array<T, N> &array,
                      const iganet::Options<T> &options) {
  if (options.device() == torch::kCPU)
    return torch::from_blob(const_cast<T *>(std::data(array)), array.size(),
                            options)
        .detach()
        .clone()
        .requires_grad_(options.requires_grad());
  else
    return torch::from_blob(const_cast<T *>(std::data(array)), array.size(),
                            options.device(torch::kCPU))
        .detach()
        .clone()
        .to(options.device())
        .requires_grad_(options.requires_grad());
}
/// @}

/// @brief Converts an std::initializer_list to torch::Tensor
/// @{
template <typename T>
inline auto
to_tensor(std::initializer_list<T> list,
          torch::IntArrayRef sizes = torch::IntArrayRef{-1},
          const iganet::Options<T> &options = iganet::Options<T>{}) {
  if (options.device() == torch::kCPU)
    return torch::from_blob(
               const_cast<T *>(std::data(list)),
               (sizes == torch::IntArrayRef{-1}) ? list.size() : sizes, options)
        .detach()
        .clone()
        .requires_grad_(options.requires_grad());
  else
    return torch::from_blob(const_cast<T *>(std::data(list)),
                            (sizes == torch::IntArrayRef{-1}) ? list.size()
                                                              : sizes,
                            options.device(torch::kCPU))
        .detach()
        .clone()
        .to(options.device())
        .requires_grad_(options.requires_grad());
}

template <typename T>
inline auto to_tensor(std::initializer_list<T> &list,
                      const iganet::Options<T> &options) {
  if (options.device() == torch::kCPU)
    return torch::from_blob(const_cast<T *>(std::data(list)), list.size(),
                            options)
        .detach()
        .clone()
        .requires_grad_(options.requires_grad());
  else
    return torch::from_blob(const_cast<T *>(std::data(list)), list.size(),
                            options.device(torch::kCPU))
        .detach()
        .clone()
        .to(options.device())
        .requires_grad_(options.requires_grad());
}
/// @}

/// @brief Converts an std::vector to torch::Tensor
/// @{
template <typename T>
inline auto
to_tensor(const std::vector<T> &vector,
          torch::IntArrayRef sizes = torch::IntArrayRef{-1},
          const iganet::Options<T> &options = iganet::Options<T>{}) {
  if (options.device() == torch::kCPU)
    return torch::from_blob(const_cast<T *>(std::data(vector)),
                            (sizes == torch::IntArrayRef{-1}) ? vector.size()
                                                              : sizes,
                            options)
        .detach()
        .clone()
        .requires_grad_(options.requires_grad());
  else
    return torch::from_blob(const_cast<T *>(std::data(vector)),
                            (sizes == torch::IntArrayRef{-1}) ? vector.size()
                                                              : sizes,
                            options.device(torch::kCPU))
        .detach()
        .clone()
        .to(options.device())
        .requires_grad_(options.requires_grad());
}

template <typename T>
inline auto to_tensor(const std::vector<T> &vector,
                      const iganet::Options<T> &options) {
  if (options.device() == torch::kCPU)
    return torch::from_blob(const_cast<T *>(std::data(vector)), vector.size(),
                            options)
        .detach()
        .clone()
        .requires_grad_(options.requires_grad());
  else
    return torch::from_blob(const_cast<T *>(std::data(vector)), vector.size(),
                            options.device(torch::kCPU))
        .detach()
        .clone()
        .to(options.device())
        .requires_grad_(options.requires_grad());
}
/// @}

/// @brief Converts an std::array<int64_t, N> to a at::IntArrayRef object
template <typename T, std::size_t N>
inline auto to_ArrayRef(const std::array<T, N> &array) {
  return at::ArrayRef<T>{array};
}

/// @brief Concatenates multiple std::array objects
/// @{
template <typename T, std::size_t... N>
inline auto concat(const std::array<T, N> &...arrays) {
  std::array<T, (N + ...)> result;
  std::size_t index{};

  ((std::copy_n(arrays.begin(), N, result.begin() + index), index += N), ...);

  return result;
}

template <typename T, std::size_t... N>
inline auto concat(std::array<T, N> &&...arrays) {
  std::array<T, (N + ...)> result;
  std::size_t index{};

  ((std::copy_n(std::make_move_iterator(arrays.begin()), N,
                result.begin() + index),
    index += N),
   ...);

  return result;
}
/// @}

/// @brief Concatenates multiple std::vector objects
/// @{
template <typename... Ts>
inline auto concat(const std::vector<Ts> &...vectors) {
  std::vector<typename std::common_type<Ts...>::type> result;

  (result.insert(result.end(), vectors.begin(), vectors.end()), ...);

  return result;
}

template <typename... Ts> inline auto concat(std::vector<Ts> &&...vectors) {
  std::vector<typename std::common_type<Ts...>::type> result;

  (result.insert(result.end(), std::make_move_iterator(vectors.begin()),
                 std::make_move_iterator(vectors.end())),
   ...);

  return result;
}
/// @}

/// @brief Adds two std::arrays
template <typename T, std::size_t size>
inline constexpr std::array<T, size> operator+(std::array<T, size> lhs,
                                               std::array<T, size> rhs) {
  std::array<T, size> result;

  for (std::size_t i = 0; i < size; ++i)
    result[i] = lhs[i] + rhs[i];

  return result;
}

/// @brief Appends data to a torch::ArrayRef object
template <typename T>
inline constexpr auto operator+(torch::ArrayRef<T> array, T data) {
  std::vector<T> result{array.vec()};
  result.push_back(data);
  return result;
}

/// @brief Appends data to a std::array object
template <typename T, std::size_t N>
inline constexpr auto operator+(std::array<T, N> array, T data) {
  std::array<T, N + 1> result;
  for (std::size_t i = 0; i < N; ++i)
    result[i] = array[i];
  result[N] = data;
  return result;
}

/// @brief Appends data to a std::vector object
template <typename T>
inline constexpr auto operator+(std::vector<T> vector, T data) {
  std::vector<T> result{vector};
  result.push_back(data);
  return result;
}

/// @brief Prepends data to a torch::ArrayRef object
template <typename T>
inline constexpr auto operator+(T data, torch::ArrayRef<T> array) {
  std::vector<T> result{array.vec()};
  result.insert(result.begin(), data);
  return result;
}

/// @brief Prepends data to a std::array object
template <typename T, std::size_t N>
inline constexpr auto operator+(T data, std::array<T, N> array) {
  std::array<T, N + 1> result;
  result[0] = data;
  for (std::size_t i = 0; i < N; ++i)
    result[i + 1] = array[i];
  return result;
}

/// @brief Prepends data to a std::vector object
template <typename T>
inline constexpr auto operator+(T data, std::vector<T> vector) {
  std::vector<T> result{vector};
  result.insert(result.begin(), data);
  return result;
}

} // namespace utils
} // namespace iganet
