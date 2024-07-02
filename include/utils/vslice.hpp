/**
   @file include/utils/vslice.hpp

   @brief VSlice utility functions

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <array>

#include <utils/linalg.hpp>
#include <utils/tensorarray.hpp>

#include <torch/torch.h>

namespace iganet {
namespace utils {

/// @brief Vectorized version of `torch::indexing::Slice` (see
/// https://pytorch.org/cppdocs/notes/tensor_indexing.html)
///
/// Creates a one-dimensional `torch::Tensor` object of
/// size `index.numel() * (stop_offset-start_offset)` with the
/// following content
///
/// \code
/// [ index[0]+start_offset,   ..., index[N-1]+start_offset,
///   index[0]+start_offset+1, ..., index[N-1]+start_offset+1,
///                            ...
///   index[0]+stop_offset-1,  ...  index[N-1]+stop_offset-1 ]
/// \endcode
///
/// @param[in] index        Tensor of indices
///
/// @param[in] start_offset Starting value of the offset
///
/// @param[in] stop_offset  Stopping value of the offset
template <bool transpose = false>
inline auto VSlice(torch::Tensor index, int64_t start_offset,
                   int64_t stop_offset) {
  if constexpr (transpose)
    return index.repeat_interleave(stop_offset - start_offset) +
           torch::linspace(start_offset, stop_offset - 1,
                           stop_offset - start_offset, index.options())
               .repeat(index.numel());
  else
    return index.repeat(stop_offset - start_offset) +
           torch::linspace(start_offset, stop_offset - 1,
                           stop_offset - start_offset, index.options())
               .repeat_interleave(index.numel());
}

/// @brief Vectorized version of `torch::indexing::Slice` (see
/// https://pytorch.org/cppdocs/notes/tensor_indexing.html)
///
/// @param[in] index        array of tensors of indices
///
/// @param[in] start_offset array of starting value of the offset
///
/// @param[in] stop_offset  array of stopping value of the offset
///
/// @param[in] leading_dim  array of leading dimension
template <bool transpose = false, std::size_t N>
inline auto VSlice(const utils::TensorArray<N> &index,
                   const std::array<int64_t, N> &start_offset,
                   const std::array<int64_t, N> &stop_offset,
                   const std::array<int64_t, N - 1> &leading_dim =
                       make_array<int64_t, N - 1>(1)) {

  // Check compatibility of arguments
  for (std::size_t i = 1; i < N; ++i)
    assert(index[i - 1].numel() == index[i].numel());

  auto dist = stop_offset - start_offset;

  if constexpr (transpose) {

    // Lambda expression to evaluate the k-th summand of the vslice
    auto vslice_summand_ = [&](std::size_t k) {
      if (k == N - 1) {
        return (index[k].repeat_interleave(utils::prod(dist, 0, k)) +
                torch::linspace(start_offset[k], stop_offset[k] - 1, dist[k],
                                index[0].options())
                    .repeat_interleave(utils::prod(dist, 0, k - 1))
                    .repeat(index[0].numel())) *
               utils::prod(leading_dim, 0, k - 1);
      } else if (k == 0) {
        if constexpr (N == 2) {
          return index[0].repeat_interleave(dist[0]).repeat_interleave(
                     dist[1]) +
                 torch::linspace(start_offset[0], stop_offset[0] - 1, dist[0],
                                 index[0].options())
                     .repeat(index[1].numel())
                     .repeat(dist[1]);
        } else { // N > 2
          return index[0].repeat_interleave(dist[0]).repeat_interleave(
                     utils::prod(dist, 1, N - 1)) +
                 torch::linspace(start_offset[0], stop_offset[0] - 1, dist[0],
                                 index[0].options())
                     .repeat(index[0].numel())
                     .repeat(utils::prod(dist, 1, N - 1));
        }
      } else {
        return (index[k]
                    .repeat_interleave(utils::prod(dist, 0, k))
                    .repeat_interleave(utils::prod(dist, k + 1, N - 1)) +
                torch::linspace(start_offset[k], stop_offset[k] - 1, dist[k],
                                index[0].options())
                    .repeat_interleave(utils::prod(dist, 0, k - 1))
                    .repeat(index[0].numel())
                    .repeat(utils::prod(dist, k + 1, N - 1))) *
               utils::prod(leading_dim, 0, k - 1);
      }
    };

    // Lambda expression to evaluate the vslice
    auto vslice_ = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
      return (vslice_summand_(Is) + ...);
    };

    return vslice_(std::make_index_sequence<N>{});
  } else {

    // Lambda expression to evaluate the k-th summand of the vslice
    auto vslice_summand_ = [&](std::size_t k) {
      if (k == N - 1) {
        return (index[k].repeat(utils::prod(dist, 0, k)) +
                torch::linspace(start_offset[k], stop_offset[k] - 1, dist[k],
                                index[0].options())
                    .repeat_interleave(index[0].numel() *
                                       utils::prod(dist, 0, k - 1))) *
               utils::prod(leading_dim, 0, k - 1);
      } else if (k == 0) {
        if constexpr (N == 2) {
          return (index[0].repeat(dist[0]) +
                  torch::linspace(start_offset[0], stop_offset[0] - 1, dist[0],
                                  index[0].options())
                      .repeat_interleave(index[0].numel()))
              .repeat(utils::prod(dist, k + 1, N - 1));
        } else { // N > 2
          return (index[0].repeat(dist[0]) +
                  torch::linspace(start_offset[0], stop_offset[0] - 1, dist[0],
                                  index[0].options())
                      .repeat_interleave(index[0].numel()))
              .repeat(utils::prod(dist, k + 1, N - 1));
        }
      } else {
        return (index[k].repeat(utils::prod(dist, 0, k)) +
                torch::linspace(start_offset[k], stop_offset[k] - 1, dist[k],
                                index[0].options())
                    .repeat_interleave(index[0].numel() *
                                       utils::prod(dist, 0, k - 1)))
                   .repeat(utils::prod(dist, k + 1, N - 1)) *
               utils::prod(leading_dim, 0, k - 1);
      }
    };

    // Lambda expression to evaluate the vslice
    auto vslice_ = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
      return (vslice_summand_(Is) + ...);
    };

    return vslice_(std::make_index_sequence<N>{});
  }
}

} // namespace utils
} // namespace iganet
