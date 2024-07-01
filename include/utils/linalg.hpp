/**
   @file include/utils/linalg.hpp

   @brief Linear algebra utility functions

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <torch/torch.h>

namespace iganet {
namespace utils {

/// @brief Computes the directional dot-product between two tensors
/// with summation along the given dimension
///
/// @tparam dim Dimension along which the sum is computed
///
/// @tparam T0 Type of the first argument
///
/// @tparam T1 Type of the second argument
///
/// @param[in] t0  First argument
///
/// @param[in] t1  Second argument
///
/// @result Tensor containing the directional dot-product
template <short_t dim = 0, typename T0, typename T1>
inline auto dotproduct(T0 &&t0, T1 &&t1) {
  return torch::sum(torch::mul(t0, t1), dim);
}

/// @brief Computes the directional Kronecker-product between two
/// tensors along the given dimension
///
/// @tparam dim Dimension along which the Kronecker-product is computed
///
/// @tparam T0 Type of the first argument
///
/// @tparam T1 Type of the second argument
///
/// @param[in] t0  First argument
///
/// @param[in] t1  Second argument
///
/// @result Tensor containing the dimensional Kronecker-product
///
/// @note This is not the regular Kronecker-product but a
/// directional variant, that is, the Kronecker-product is computed
/// along the given direction. All other directions are left
/// unchanged. For the regular Kronecker-product use `utils::kron`.
template <short_t dim = 0, typename T0, typename T1>
inline auto kronproduct(T0 &&t0, T1 &&t1) {
  switch (t1.sizes().size()) {
  case 1:
    return torch::mul(t0.repeat_interleave(t1.size(dim), 0),
                      t1.repeat({t0.size(dim)}));
  case 2:
    if constexpr (dim == 0)
      return torch::mul(t0.repeat_interleave(t1.size(dim), 0),
                        t1.repeat({t0.size(dim), 1}));
    else if constexpr (dim == 1)
      return torch::mul(t0.repeat_interleave(t1.size(dim), 1),
                        t1.repeat({1, t0.size(dim)}));
  case 3:
    if constexpr (dim == 0)
      return torch::mul(t0.repeat_interleave(t1.size(dim), 0),
                        t1.repeat({t0.size(dim), 1, 1}));
    else if constexpr (dim == 1)
      return torch::mul(t0.repeat_interleave(t1.size(dim), 1),
                        t1.repeat({1, t0.size(dim), 1}));
    else if constexpr (dim == 2)
      return torch::mul(t0.repeat_interleave(t1.size(dim), 0),
                        t1.repeat({1, 1, t0.size(dim)}));
  case 4:
    if constexpr (dim == 0)
      return torch::mul(t0.repeat_interleave(t1.size(dim), 0),
                        t1.repeat({t0.size(dim), 1, 1, 1}));
    else if constexpr (dim == 1)
      return torch::mul(t0.repeat_interleave(t1.size(dim), 1),
                        t1.repeat({1, t0.size(dim), 1, 1}));
    else if constexpr (dim == 2)
      return torch::mul(t0.repeat_interleave(t1.size(dim), 0),
                        t1.repeat({1, 1, t0.size(dim), 1}));
    else if constexpr (dim == 3)
      return torch::mul(t0.repeat_interleave(t1.size(dim), 0),
                        t1.repeat({1, 1, 1, t0.size(dim)}));
  case 5:
    if constexpr (dim == 0)
      return torch::mul(t0.repeat_interleave(t1.size(dim), 0),
                        t1.repeat({t0.size(dim), 1, 1, 1, 1}));
    else if constexpr (dim == 1)
      return torch::mul(t0.repeat_interleave(t1.size(dim), 1),
                        t1.repeat({1, t0.size(dim), 1, 1, 1}));
    else if constexpr (dim == 2)
      return torch::mul(t0.repeat_interleave(t1.size(dim), 0),
                        t1.repeat({1, 1, t0.size(dim), 1, 1}));
    else if constexpr (dim == 3)
      return torch::mul(t0.repeat_interleave(t1.size(dim), 0),
                        t1.repeat({1, 1, 1, t0.size(dim), 1}));
    else if constexpr (dim == 4)
      return torch::mul(t0.repeat_interleave(t1.size(dim), 0),
                        t1.repeat({1, 1, 1, 1, t0.size(dim)}));
  case 6:
    if constexpr (dim == 0)
      return torch::mul(t0.repeat_interleave(t1.size(dim), 0),
                        t1.repeat({t0.size(dim), 1, 1, 1, 1, 1}));
    else if constexpr (dim == 1)
      return torch::mul(t0.repeat_interleave(t1.size(dim), 1),
                        t1.repeat({1, t0.size(dim), 1, 1, 1, 1}));
    else if constexpr (dim == 2)
      return torch::mul(t0.repeat_interleave(t1.size(dim), 0),
                        t1.repeat({1, 1, t0.size(dim), 1, 1, 1}));
    else if constexpr (dim == 3)
      return torch::mul(t0.repeat_interleave(t1.size(dim), 0),
                        t1.repeat({1, 1, 1, t0.size(dim), 1, 1}));
    else if constexpr (dim == 4)
      return torch::mul(t0.repeat_interleave(t1.size(dim), 0),
                        t1.repeat({1, 1, 1, 1, t0.size(dim), 1}));
    else if constexpr (dim == 5)
      return torch::mul(t0.repeat_interleave(t1.size(dim), 0),
                        t1.repeat({1, 1, 1, 1, 1, t0.size(dim)}));
  case 7:
    if constexpr (dim == 0)
      return torch::mul(t0.repeat_interleave(t1.size(dim), 0),
                        t1.repeat({t0.size(dim), 1, 1, 1, 1, 1, 1}));
    else if constexpr (dim == 1)
      return torch::mul(t0.repeat_interleave(t1.size(dim), 1),
                        t1.repeat({1, t0.size(dim), 1, 1, 1, 1, 1}));
    else if constexpr (dim == 2)
      return torch::mul(t0.repeat_interleave(t1.size(dim), 0),
                        t1.repeat({1, 1, t0.size(dim), 1, 1, 1, 1}));
    else if constexpr (dim == 3)
      return torch::mul(t0.repeat_interleave(t1.size(dim), 0),
                        t1.repeat({1, 1, 1, t0.size(dim), 1, 1, 1}));
    else if constexpr (dim == 4)
      return torch::mul(t0.repeat_interleave(t1.size(dim), 0),
                        t1.repeat({1, 1, 1, 1, t0.size(dim), 1, 1}));
    else if constexpr (dim == 5)
      return torch::mul(t0.repeat_interleave(t1.size(dim), 0),
                        t1.repeat({1, 1, 1, 1, 1, t0.size(dim), 1}));
    else if constexpr (dim == 6)
      return torch::mul(t0.repeat_interleave(t1.size(dim), 0),
                        t1.repeat({1, 1, 1, 1, 1, 1, t0.size(dim)}));
  case 8:
    if constexpr (dim == 0)
      return torch::mul(t0.repeat_interleave(t1.size(dim), 0),
                        t1.repeat({t0.size(dim), 1, 1, 1, 1, 1, 1, 1}));
    else if constexpr (dim == 1)
      return torch::mul(t0.repeat_interleave(t1.size(dim), 1),
                        t1.repeat({1, t0.size(dim), 1, 1, 1, 1, 1, 1}));
    else if constexpr (dim == 2)
      return torch::mul(t0.repeat_interleave(t1.size(dim), 0),
                        t1.repeat({1, 1, t0.size(dim), 1, 1, 1, 1, 1}));
    else if constexpr (dim == 3)
      return torch::mul(t0.repeat_interleave(t1.size(dim), 0),
                        t1.repeat({1, 1, 1, t0.size(dim), 1, 1, 1, 1}));
    else if constexpr (dim == 4)
      return torch::mul(t0.repeat_interleave(t1.size(dim), 0),
                        t1.repeat({1, 1, 1, 1, t0.size(dim), 1, 1, 1}));
    else if constexpr (dim == 5)
      return torch::mul(t0.repeat_interleave(t1.size(dim), 0),
                        t1.repeat({1, 1, 1, 1, 1, t0.size(dim), 1, 1}));
    else if constexpr (dim == 6)
      return torch::mul(t0.repeat_interleave(t1.size(dim), 0),
                        t1.repeat({1, 1, 1, 1, 1, 1, t0.size(dim), 1}));
    else if constexpr (dim == 7)
      return torch::mul(t0.repeat_interleave(t1.size(dim), 0),
                        t1.repeat({1, 1, 1, 1, 1, 1, 1, t0.size(dim)}));
  default:
    throw std::runtime_error("Unsupported tensor dimension");
  }
}

/// @brief Computes the directional Kronecker-product between two or more
/// tensors along the given dimension
///
/// @tparam dim Dimension along which the Kronecker-product is computed
///
/// @tparam T Type of the first argument
///
/// @tparam Ts Types of the variadic arguments
///
/// @param[in] t  First argument
///
/// @param[in] ts  Variadic arguments
///
/// @result Tensor containing the dimensional Kronecker-product
///
/// @note This is not the regular Kronecker-product but a
/// directional variant, that is, the Kronecker-product is computed
/// along the given direction. All other directions are left
/// unchanged. For the regular Kronecker-product use `utils::kron`.
template <short_t dim = 0, typename T, typename... Ts>
inline auto kronproduct(T &&t, Ts &&...ts) {
  return kronproduct<dim>(std::forward<T>(t),
                          kronproduct<dim>(std::forward<Ts>(ts)...));
}

/// @brief Computes the Kronecker-product between two or more tensors
/// @{
template <typename T0, typename T1> inline auto kron(T0 &&t0, T1 &&t1) {
  return torch::kron(std::forward<T0>(t0), std::forward<T1>(t1));
}

template <typename T, typename... Ts> inline auto kron(T &&t, Ts &&...ts) {
  return kron(std::forward<T>(t), kron(std::forward<Ts>(ts)...));
}
/// @}

/// @brief Computes the (partial) product of all std::array entries
template <typename T, std::size_t N>
inline T prod(std::array<T, N> array, std::size_t start_index = 0,
              std::size_t stop_index = N - 1) {
  T result{1};

  for (std::size_t i = start_index; i <= stop_index; ++i)
    result *= array[i];

  return result;
}

/// @brief Computes the (partial) sum of all std::array entries
template <typename T, std::size_t N>
inline T sum(std::array<T, N> array, std::size_t start_index = 0,
             std::size_t stop_index = N - 1) {
  T result{0};

  for (std::size_t i = start_index; i <= stop_index; ++i)
    result += array[i];

  return result;
}

} // namespace utils
} // namespace iganet
