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
/// @param[in] index        2d array of tensors of indices
///
/// @param[in] start_offset 2d array of starting value of the offset
///
/// @param[in] stop_offset  2d array of stopping value of the offset
///
/// @param[in] leading_dim  Leading dimension
template <bool transpose = false>
inline auto VSlice(const utils::TensorArray<2> &index,
                   const std::array<int64_t, 2> &start_offset,
                   const std::array<int64_t, 2> &stop_offset,
                   int64_t leading_dim = 1) {
  assert(index[0].numel() == index[1].numel());

  auto dist0 = stop_offset[0] - start_offset[0];
  auto dist1 = stop_offset[1] - start_offset[1];
  auto dist01 = dist0 * dist1;

  if constexpr (transpose)
    return (index[1].repeat_interleave(dist01) +
            torch::linspace(start_offset[1], stop_offset[1] - 1, dist1,
                            index[0].options())
                .repeat_interleave(dist0)
                .repeat(index[0].numel())) *
               leading_dim +
           index[0].repeat_interleave(dist0).repeat_interleave(dist1) +
           torch::linspace(start_offset[0], stop_offset[0] - 1, dist0,
                           index[0].options())
               .repeat(index[1].numel())
               .repeat(dist1);
  else
    return (index[1].repeat(dist01) +
            torch::linspace(start_offset[1], stop_offset[1] - 1, dist1,
                            index[0].options())
                .repeat_interleave(index[0].numel() * dist0)) *
               leading_dim +
           (index[0].repeat(dist0) + torch::linspace(start_offset[0],
                                                     stop_offset[0] - 1, dist0,
                                                     index[0].options())
                                         .repeat_interleave(index[1].numel()))
               .repeat(dist1);
}

/// @brief Vectorized version of `torch::indexing::Slice` (see
/// https://pytorch.org/cppdocs/notes/tensor_indexing.html)
///
/// @param[in] index        3d array of tensors of indices
///
/// @param[in] start_offset 3d array of starting value of the offset
///
/// @param[in] stop_offset  3d array of stopping value of the offset
///
/// @param[in] leading_dim  2d array of leading dimension
template <bool transpose = false>
inline auto VSlice(const utils::TensorArray<3> &index,
                   const std::array<int64_t, 3> &start_offset,
                   const std::array<int64_t, 3> &stop_offset,
                   const std::array<int64_t, 2> &leading_dim = {1, 1}) {
  assert(index[0].numel() == index[1].numel() &&
         index[1].numel() == index[2].numel());

  auto dist0 = stop_offset[0] - start_offset[0];
  auto dist1 = stop_offset[1] - start_offset[1];
  auto dist2 = stop_offset[2] - start_offset[2];
  auto dist01 = dist0 * dist1;
  auto dist12 = dist1 * dist2;
  auto dist012 = dist0 * dist12;

  if constexpr (transpose)
    return (index[2].repeat_interleave(dist012) +
            torch::linspace(start_offset[2], stop_offset[2] - 1, dist2,
                            index[0].options())
                .repeat_interleave(dist01)
                .repeat(index[0].numel())) *
               leading_dim[0] * leading_dim[1] +
           (index[1].repeat_interleave(dist01).repeat_interleave(dist2) +
            torch::linspace(start_offset[1], stop_offset[1] - 1, dist1,
                            index[0].options())
                .repeat_interleave(dist0)
                .repeat(index[0].numel())
                .repeat(dist2)) *
               leading_dim[0] +
           index[0].repeat_interleave(dist0).repeat_interleave(dist12) +
           torch::linspace(start_offset[0], stop_offset[0] - 1, dist0,
                           index[0].options())
               .repeat(index[0].numel())
               .repeat(dist12);
  else
    return (index[2].repeat(dist012) +
            torch::linspace(start_offset[2], stop_offset[2] - 1, dist2,
                            index[0].options())
                .repeat_interleave(index[0].numel() * dist01)) *
               leading_dim[0] * leading_dim[1] +
           (index[1].repeat(dist01) +
            torch::linspace(start_offset[1], stop_offset[1] - 1, dist1,
                            index[0].options())
                .repeat_interleave(index[0].numel() * dist0))
                   .repeat(dist2) *
               leading_dim[0] +
           (index[0].repeat(dist0) + torch::linspace(start_offset[0],
                                                     stop_offset[0] - 1, dist0,
                                                     index[0].options())
                                         .repeat_interleave(index[0].numel()))
               .repeat(dist12);
}

/// @brief Vectorized version of `torch::indexing::Slice` (see
/// https://pytorch.org/cppdocs/notes/tensor_indexing.html)
///
/// @param[in] index        4d array of tensors of indices
///
/// @param[in] start_offset 4d array of starting value of the offset
///
/// @param[in] stop_offset  4d array of stopping value of the offset
///
/// @param[in] leading_dim  3d array of leading dimension
template <bool transpose = false>
inline auto VSlice(const utils::TensorArray<4> &index,
                   const std::array<int64_t, 4> &start_offset,
                   const std::array<int64_t, 4> &stop_offset,
                   const std::array<int64_t, 3> &leading_dim = {1, 1, 1}) {
  assert(index[0].numel() == index[1].numel() &&
         index[1].numel() == index[2].numel() &&
         index[2].numel() == index[3].numel());

  auto dist0 = stop_offset[0] - start_offset[0];
  auto dist1 = stop_offset[1] - start_offset[1];
  auto dist2 = stop_offset[2] - start_offset[2];
  auto dist3 = stop_offset[3] - start_offset[3];
  auto dist01 = dist0 * dist1;
  auto dist12 = dist1 * dist2;
  auto dist23 = dist2 * dist3;
  auto dist012 = dist0 * dist12;
  auto dist123 = dist1 * dist23;
  auto dist0123 = dist01 * dist23;

  if constexpr (transpose)
    return (index[3].repeat_interleave(dist0123) +
            torch::linspace(start_offset[3], stop_offset[3] - 1, dist3,
                            index[0].options())
                .repeat_interleave(dist012)
                .repeat(index[0].numel())) *
               leading_dim[0] * leading_dim[1] * leading_dim[2] +
           (index[2].repeat_interleave(dist012).repeat_interleave(dist3) +
            torch::linspace(start_offset[2], stop_offset[2] - 1, dist2,
                            index[0].options())
                .repeat_interleave(dist01)
                .repeat(index[0].numel())
                .repeat(dist3)) *
               leading_dim[0] * leading_dim[1] +
           (index[1].repeat_interleave(dist01).repeat_interleave(dist23) +
            torch::linspace(start_offset[1], stop_offset[1] - 1, dist1,
                            index[0].options())
                .repeat_interleave(dist0)
                .repeat(index[0].numel())
                .repeat(dist23)) *
               leading_dim[0] +
           index[0].repeat_interleave(dist0).repeat_interleave(dist123) +
           torch::linspace(start_offset[0], stop_offset[0] - 1, dist0,
                           index[0].options())
               .repeat(index[0].numel())
               .repeat(dist123);
  else
    return (index[3].repeat(dist0123) +
            torch::linspace(start_offset[3], stop_offset[3] - 1, dist3,
                            index[0].options())
                .repeat_interleave(index[0].numel() * dist012)) *
               leading_dim[0] * leading_dim[1] * leading_dim[2] +
           (index[2].repeat(dist012) +
            torch::linspace(start_offset[2], stop_offset[2] - 1, dist2,
                            index[0].options())
                .repeat_interleave(index[0].numel() * dist01))
                   .repeat(dist3) *
               leading_dim[0] * leading_dim[1] +
           (index[1].repeat(dist01) +
            torch::linspace(start_offset[1], stop_offset[1] - 1, dist1,
                            index[0].options())
                .repeat_interleave(index[0].numel() * dist0))
                   .repeat(dist23) *
               leading_dim[0] +
           (index[0].repeat(dist0) + torch::linspace(start_offset[0],
                                                     stop_offset[0] - 1, dist0,
                                                     index[0].options())
                                         .repeat_interleave(index[0].numel()))
               .repeat(dist123);
}

} // namespace utils
} // namespace iganet
