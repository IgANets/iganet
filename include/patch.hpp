/**
   @file include/bspline.hpp

   @brief Abstract patch function base class

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <core.hpp>

#include <utils/blocktensor.hpp>
#include <utils/tensorarray.hpp>

namespace iganet {

/// @brief Abstract patch function base class
template <typename real_t, short_t GeoDim, short_t ParDim> class BSplinePatch {
public:
  /// @brief Returns the `device` property
  virtual torch::Device device() const noexcept = 0;

  /// @brief Returns the `device_index` property
  virtual int32_t device_index() const noexcept = 0;

  /// @brief Returns the `dtype` property
  virtual torch::Dtype dtype() const noexcept = 0;

  /// @brief Returns the `layout` property
  virtual torch::Layout layout() const noexcept = 0;

  /// @brief Returns the `requires_grad` property
  virtual bool requires_grad() const noexcept = 0;

  /// @brief Returns the `pinned_memory` property
  virtual bool pinned_memory() const noexcept = 0;

  /// @brief Returns if the layout is sparse
  virtual bool is_sparse() const noexcept = 0;

  /// @brief Sets the B-spline object's `requires_grad` property
  virtual BSplinePatch &set_requires_grad(bool requires_grad) noexcept = 0;

  // @brief Returns all coefficients as a single tensor
  virtual torch::Tensor as_tensor() const noexcept = 0;

  /// @brief Sets all coefficients from a single tensor
  virtual BSplinePatch &from_tensor(const torch::Tensor &tensor) noexcept = 0;

  /// @brief Returns the size of the single tensor representation of
  /// all coefficients
  virtual int64_t as_tensor_size() const noexcept = 0;

  /// @brief Returns the value of the spline function from precomputed
  /// basis function
  /// @{
  virtual utils::BlockTensor<torch::Tensor, 1, GeoDim>
  eval_from_precomputed(const torch::Tensor &basfunc,
                        const torch::Tensor &coeff_indices, int64_t numeval,
                        torch::IntArrayRef sizes) const = 0;

  virtual utils::BlockTensor<torch::Tensor, 1, GeoDim>
  eval_from_precomputed(const utils::TensorArray<ParDim> &basfunc,
                        const torch::Tensor &coeff_indices, int64_t numeval,
                        torch::IntArrayRef sizes) const = 0;
  /// @}

  /// @brief Returns a string representation
  virtual void
  pretty_print(std::ostream &os = Log(log::info)) const noexcept = 0;
};

/// @brief Print (as string) a BSplinePatch object
template <typename real_t, short_t GeoDim, short_t ParDim>
inline std::ostream &
operator<<(std::ostream &os, const BSplinePatch<real_t, GeoDim, ParDim> &obj) {
  obj.pretty_print(os);
  return os;
}

} // namespace iganet
