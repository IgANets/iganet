/**
   @file include/nurbs.hpp

   @brief Multivariate non-uniform rational B-splines

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <bspline.hpp>

namespace iganet {

/// @brief Tensor-product non-uniform rational B-spline with uniform
/// knot vector (core functionality)
///
/// This class extends the base class UniformBSplineCore to
/// non-uniform rational B-splines (NURBS) with uniform knot
/// vectors. Like its base class it only implements the core
/// functionality of non-uniform rational B-splines.
///
/// This implementation exploits the fact that NURBS in \f$d\f$ space
/// dimensions can be realized through perspective projection from
/// B-splines in \f$d+1\f$ space dimensions. That is, a NURBS object
/// with control points
//
/// \f[
///   \mathbf{c}_i = \left(x_i, y_i, z_i\right)
/// \f]
///
/// is extended to a B-spline object with homogeneous coordinates
//
/// \f[
///   \mathbf{c}_i = \left(w_i x_i, w_i y_i, w_i z_i, w_i\right)
/// \f]
///
/// with non-negative weights \f$w_i\f$. All operations are performed
/// on the B-spline object which is afterwards converted to a NURBS
/// object through perspective projection onto the coordinates
/// \f$\left(x/w, y/w, z/w\right)\f$.
template <typename real_t, short_t GeoDim, short_t... Degrees>
class UniformNurbsCore
    : public UniformBSplineCore<real_t, GeoDim + 1, Degrees...> {
private:
  /// @brief Base type
  using Base = UniformBSplineCore<real_t, GeoDim + 1, Degrees...>;

public:
  /// @brief Value type
  using value_type = real_t;

  /// @brief Deduces the type of the template template parameter `BSpline`
  /// when exposed to the class template parameters `real_t` and
  /// `GeoDim`, and the `Degrees` parameter pack. The optional
  /// template parameter `degree_elevate` can be used to
  /// (de-)elevate the degrees by an additive constant
  template <template <typename, short_t, short_t...> class BSpline,
            std::make_signed<short_t>::type degree_elevate = 0>
  using derived_type =
      BSpline<real_t, GeoDim + 1, (Degrees + degree_elevate)...>;

  /// @brief Deduces the self-type possibly degrees (de-)elevated by
  /// the additive constant `degree_elevate`
  template <std::make_signed<short_t>::type degree_elevate = 0>
  using self_type =
      typename Base::template derived_type<UniformNurbsCore, degree_elevate>;

  /// @brief Deduces the derived self-type when exposed to different
  /// class template parameters `real_t` and `GeoDim`, and the
  /// `Degrees` parameter pack
  template <typename other_t, short_t GeoDim_, short_t... Degrees_>
  using derived_self_type = UniformNurbsCore<other_t, GeoDim_, Degrees_...>;

  /// @brief Deduces the derived self-type when exposed to a
  /// different class template parameter `real_t`
  template <typename other_t>
  using real_derived_self_type = UniformNurbsCore<other_t, GeoDim, Degrees...>;

  /// @brief Returns true if the B-spline is uniform
  static constexpr bool is_uniform() { return true; }

  /// @brief Returns true if the B-spline is non-uniform
  static constexpr bool is_nonuniform() { return false; }

  /// @brief Default constructor
  ///
  /// @param[in] options Options configuration
  UniformNurbsCore(Options<real_t> options = Options<real_t>{})
      : Base(options) {}

  /// @brief Constructor for equidistant knot vectors
  ///
  /// @param[in] ncoeffs Number of coefficients per parametric dimension
  ///
  /// @param[in] init Type of initialization
  ///
  /// @param[in] options Options configuration
  UniformNurbsCore(const std::array<int64_t, Base::parDim_> &ncoeffs,
                   enum init init = init::greville,
                   Options<real_t> options = Options<real_t>{})
      : Base(ncoeffs, init, options) {
    if (Base::coeffs_[GeoDim].defined())
      Base::coeffs_[GeoDim] = torch::ones_like(Base::coeffs_[GeoDim]);
  }

  /// @brief Constructor for equidistant knot vectors
  ///
  /// @param[in] ncoeffs Number of coefficients per parametric dimension
  ///
  /// @param[in] coeffs Vectors of coefficients per parametric dimension
  ///
  /// @param[in] clone  If true, coefficients will be cloned. Otherwise,
  /// coefficients will be aliased
  ///
  /// @param[in] options Options configuration
  ///
  /// @note It is not checked whether vectors of coefficients are
  /// compatible with the given Options object if clone is false.
  UniformNurbsCore(const std::array<int64_t, Base::parDim_> &ncoeffs,
                   const utils::TensorArray<Base::geoDim_> &coeffs,
                   bool clone = false,
                   Options<real_t> options = Options<real_t>{})
      : Base(ncoeffs, coeffs, clone, options) {}

  /// @brief Constructor for equidistant knot vectors
  ///
  /// @param[in] ncoeffs Number of coefficients per parametric dimension
  ///
  /// @param[in] coeffs Vectors of coefficients per parametric dimension
  ///
  /// @param[in] options Options configuration
  ///
  /// @note It is not checked whether vectors of coefficients are
  /// compatible with the given Options object if clone is false.
  UniformNurbsCore(const std::array<int64_t, Base::parDim_> &ncoeffs,
                   utils::TensorArray<Base::geoDim_> &&coeffs,
                   Options<real_t> options = Options<real_t>{})
      : Base(ncoeffs, coeffs, options) {}

  /// @brief Copy constructor
  ///
  /// @param[in] other Uniform Nurbs object to copy
  ///
  /// @param[in] options Options configuration
  template <typename other_t>
  UniformNurbsCore(
      const UniformNurbsCore<other_t, GeoDim + 1, Degrees...> &other,
      Options<real_t> options = Options<real_t>{})
      : Base(static_cast<UniformBSplineCore<other_t, GeoDim + 1, Degrees...>>(
            other)) {}

  /// @result Number of geometric dimensions
  ///
  /// @note This override of the `geoDim()` function makes sure that
  /// the geometric dimension is reported correctly with respect to
  /// the NURBS object
  inline static constexpr short_t geoDim() noexcept { return GeoDim; }

  /// @brief Returns a constant reference to the weights
  ///
  /// @note Since the weights are the entry of the homogeneous
  /// coordinates which are stored in coeff one can likewise retrieve
  /// them using `coeffs(GeoDim)`
  inline const torch::Tensor &weights() const noexcept {
    return Base::coeffs_[GeoDim];
  }

  /// @brief Returns a non-constant reference to the weights
  ///
  /// @note Since the weights are the entry of the homogeneous
  /// coordinates which are stored in coeff one can likewise retrieve
  /// them using `coeffs(GeoDim)`
  inline torch::Tensor &weights() noexcept { return Base::coeffs_[GeoDim]; }
};

/// @brief Tensor-product non-uniform rational B-spline with
/// non-uniform knot vectors (core functionality)
///
/// This class extends the base class NonUniformBSplineCore to
/// non-uniform B-splines. Like its base class it only implements
/// the core functionality of non-uniform B-splines
template <typename real_t, short_t GeoDim, short_t... Degrees>
class NonUniformNurbsCore
    : public NonUniformBSplineCore<real_t, GeoDim, Degrees...> {
private:
  /// @brief Base type
  using Base = NonUniformBSplineCore<real_t, GeoDim, Degrees...>;

protected:
  /// @brief Tensor storing the weights
  torch::Tensor weights_;

public:
  /// @brief Value type
  using value_type = real_t;

  /// @brief Deduces the type of the template template parameter `BSpline`
  /// when exposed to the class template parameters `real_t` and
  /// `GeoDim`, and the `Degrees` parameter pack. The optional
  /// template parameter `degree_elevate` can be used to
  /// (de-)elevate the degrees by an additive constant
  template <template <typename, short_t, short_t...> class BSpline,
            std::make_signed<short_t>::type degree_elevate = 0>
  using derived_type = BSpline<real_t, GeoDim, (Degrees + degree_elevate)...>;

  /// @brief Deduces the self-type possibly degrees (de-)elevated by
  /// the additive constant `degree_elevate`
  template <std::make_signed<short_t>::type degree_elevate = 0>
  using self_type =
      typename Base::template derived_type<NonUniformNurbsCore, degree_elevate>;

  /// @brief Deduces the derived self-type when exposed to different
  /// class template parameters `real_t` and `GeoDim`, and the
  /// `Degrees` parameter pack
  template <typename other_t, short_t GeoDim_, short_t... Degrees_>
  using derived_self_type = NonUniformNurbsCore<other_t, GeoDim_, Degrees_...>;

  /// @brief Deduces the derived self-type when exposed to a
  /// different class template parameter `real_t`
  template <typename other_t>
  using real_derived_self_type =
      NonUniformNurbsCore<other_t, GeoDim, Degrees...>;

  /// @brief Returns true if the B-spline is uniform
  static constexpr bool is_uniform() { return false; }

  /// @brief Returns true if the B-spline is non-uniform
  static constexpr bool is_nonuniform() { return true; }
};

/// @brief Tensor-product uniform Nurbs
template <typename real_t, short_t GeoDim, short_t... Degrees>
using UniformNurbs =
    BSplineCommon<UniformNurbsCore<real_t, GeoDim, Degrees...>>;

/// @brief Print (as string) a UniformNurbs object
template <typename real_t, short_t GeoDim, short_t... Degrees>
inline std::ostream &
operator<<(std::ostream &os,
           const UniformNurbs<real_t, GeoDim, Degrees...> &obj) {
  obj.pretty_print(os);
  return os;
}

/// @brief Tensor-product non-uniform Nurbs
template <typename real_t, short_t GeoDim, short_t... Degrees>
using NonUniformNurbs =
    BSplineCommon<NonUniformNurbsCore<real_t, GeoDim, Degrees...>>;

/// @brief Print (as string) a NonUniformNurbs object
template <typename real_t, short_t GeoDim, short_t... Degrees>
inline std::ostream &
operator<<(std::ostream &os,
           const NonUniformNurbs<real_t, GeoDim, Degrees...> &obj) {
  obj.pretty_print(os);
  return os;
}
} // namespace iganet
