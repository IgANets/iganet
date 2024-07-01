/**
   @file include/boundary.hpp

   @brief Boundary treatment

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/seq/for_each.hpp>

#include <bspline.hpp>

namespace iganet {

/// @brief Identifiers for topological sides
enum side {
  west = 1,
  east = 2,
  south = 3,
  north = 4,
  front = 5,
  back = 6,
  stime = 7,
  etime = 8,
  left = 1,
  right = 2,
  down = 3,
  up = 4,
  none = 0
};

/// @brief BoundaryCore
template <typename Spline, short_t> class BoundaryCore;

/// @brief BoundaryCore (1d specialization)
///
/// This specialization has 2 sides
/// - west (u=0)
/// - east (u=1)
template <typename Spline>
class BoundaryCore<Spline, /* parDim */ 1> : public utils::Serializable,
                                             private utils::FullQualifiedName {

  // Safeguard that template parameter is a valid SplineType
  static_assert(is_SplineType_v<Spline>, "Spline must be a valid SplineType");

  /// @brief Enable access to private members
  template <typename BoundaryCore> friend class BoundaryCommon;

protected:
  /// @brief Spline type
  using spline_type = Spline;

  /// @brief Boundary spline type
  using boundary_spline_type =
      typename Spline::template derived_self_type<typename Spline::value_type,
                                                  Spline::geoDim()>;

  /// @brief Deduces the derived boundary spline type when exposed
  /// to a different class template parameter `real_t`
  template <typename real_t>
  using real_derived_boundary_spline_type =
      typename Spline::template derived_self_type<real_t, Spline::geoDim()>;

  /// @brief Tuple of splines
  std::tuple<boundary_spline_type, boundary_spline_type> bdr_;

public:
  /// @brief Boundary type
  using boundary_type = decltype(bdr_);

  /// @brief Evaluation type
  using eval_type = std::tuple<torch::Tensor, torch::Tensor>;

  /// @brief Default constructor
  BoundaryCore(Options<typename Spline::value_type> options =
                   Options<typename Spline::value_type>{})
      : bdr_({boundary_spline_type(options), boundary_spline_type(options)}) {}

  /// @brief Copy constructor
  BoundaryCore(const boundary_type &bdr_) : bdr_(bdr_) {}

  /// @brief Move constructor
  BoundaryCore(boundary_type &&bdr_) : bdr_(bdr_) {}

  /// @brief Copy/clone constructor
  BoundaryCore(const BoundaryCore &other, bool clone)
      : bdr_(clone ? std::apply(
                         [](const auto &...bspline) {
                           return std::make_tuple(bspline.clone()...);
                         },
                         other.coeffs())
                   : other.coeffs()) {}

  /// @brief Constructor
  BoundaryCore(const std::array<int64_t, 1> &, enum init init = init::zeros,
               Options<typename Spline::value_type> options =
                   Options<typename Spline::value_type>{})
      : bdr_({boundary_spline_type(std::array<int64_t, 0>{}, init, options),
              boundary_spline_type(std::array<int64_t, 0>{}, init, options)}) {}

  /// @brief Constructor
  BoundaryCore(const std::array<std::vector<typename Spline::value_type>, 1> &,
               enum init init = init::zeros,
               Options<typename Spline::value_type> options =
                   Options<typename Spline::value_type>{})
      : bdr_({boundary_spline_type(std::array<int64_t, 0>{}, init, options),
              boundary_spline_type(std::array<int64_t, 0>{}, init, options)}) {}

  /// @brief Sets the coefficients of all spline objects from a
  /// single tensor that holds both boundary and inner coefficients
  ///
  /// @param[in] tensor Tensor from which to extract the coefficients
  ///
  /// @result Updates spline object
  inline auto &from_full_tensor(const torch::Tensor &tensor) {

    if (tensor.dim() > 1) {
      auto tensor_view = tensor.view({Spline::geoDim(), -1, tensor.size(-1)});

      side<west>().from_tensor(tensor_view.index({torch::indexing::Slice(), 0})
                                   .reshape({-1, tensor.size(-1)}));
      side<east>().from_tensor(tensor_view.index({torch::indexing::Slice(), -1})
                                   .reshape({-1, tensor.size(-1)}));
    } else {
      auto tensor_view = tensor.view({Spline::geoDim(), -1});

      side<west>().from_tensor(
          tensor_view.index({torch::indexing::Slice(), 0}).flatten());
      side<east>().from_tensor(
          tensor_view.index({torch::indexing::Slice(), -1}).flatten());
    }
    return *this;
  }

  /// @brief Returns the number of sides
  inline static constexpr short_t nsides() { return side::east; }

  /// @brief Returns constant reference to side-th Spline
  template <short_t s> inline constexpr auto &side() const {
    static_assert(s > none && s <= nsides());
    return std::get<s - 1>(bdr_);
  }

  /// @brief Returns non-constant reference to side-th Spline
  template <short_t s> inline constexpr auto &side() {
    static_assert(s > none && s <= nsides());
    return std::get<s - 1>(bdr_);
  }

  /// @brief Returns a constant reference to the array of
  /// coefficients for all boundary segments.
  inline constexpr auto &coeffs() const { return bdr_; }

  /// @brief Returns a non-constant reference to the array of
  /// coefficients for all boundary segments.
  inline constexpr auto &coeffs() { return bdr_; }

  /// @brief Returns the total number of coefficients
  inline int64_t ncumcoeffs() const {
    int64_t s = 0;
    s += side<west>().ncumcoeffs();
    s += side<east>().ncumcoeffs();

    return s;
  }

  /// @brief Returns a string representation of the Boundary object
  inline virtual void
  pretty_print(std::ostream &os = Log(log::info)) const noexcept override {
    os << name() << "(\n"
       << "west = " << side<west>() << "\n"
       << "east = " << side<east>() << "\n)";
  }

  /// @brief Returns the boundary object as JSON object
  inline nlohmann::json to_json() const override {
    nlohmann::json json;
    json["west"] = side<west>().to_json();
    json["east"] = side<east>().to_json();

    return json;
  }

  /// @brief Updates the boundary object from JSON object
  inline BoundaryCore &from_json(const nlohmann::json &json) {
    side<west>().from_json(json["west"]);
    side<east>().from_json(json["east"]);

    return *this;
  }

  /// @brief Returns the Greville abscissae
  inline eval_type greville() const {
    return eval_type{side<west>().greville(), side<east>().greville()};
  }
};

/// @brief BoundaryCore (2d specialization)
///
/// This specialization has 4 sides
/// - west  (u=0, v  )
/// - east  (u=1, v  )
/// - south (u,   v=0)
/// - north (u,   v=1)
template <typename Spline>
class BoundaryCore<Spline, /* parDim */ 2> : public utils::Serializable,
                                             private utils::FullQualifiedName {

  // Safeguard that template parameter is a valid SplineType
  static_assert(is_SplineType_v<Spline>, "Spline must be a valid SplineType");

  /// @brief Enable access to private members
  template <typename BoundaryCore> friend class BoundaryCommon;

protected:
  /// @brief Spline type
  using spline_type = Spline;

  /// @brief Boundary spline type
  using boundary_spline_type = std::tuple<
      typename Spline::template derived_self_type<
          typename Spline::value_type, Spline::geoDim(), Spline::degree(1)>,
      typename Spline::template derived_self_type<
          typename Spline::value_type, Spline::geoDim(), Spline::degree(0)>>;

  /// @brief Deduces the derived boundary spline type when exposed
  /// to a different class template parameter `real_t`
  template <typename real_t>
  using real_derived_boundary_spline_type =
      std::tuple<typename Spline::template derived_self_type<
                     real_t, Spline::geoDim(), Spline::degree(1)>,
                 typename Spline::template derived_self_type<
                     real_t, Spline::geoDim(), Spline::degree(0)>>;

  /// @brief Tuple of splines
  std::tuple<typename std::tuple_element_t<0, boundary_spline_type>,
             typename std::tuple_element_t<0, boundary_spline_type>,
             typename std::tuple_element_t<1, boundary_spline_type>,
             typename std::tuple_element_t<1, boundary_spline_type>>
      bdr_;

public:
  /// @brief Boundary type
  using boundary_type = decltype(bdr_);

  /// @brief Evaluation type
  using eval_type = std::tuple<utils::TensorArray<1>, utils::TensorArray<1>,
                               utils::TensorArray<1>, utils::TensorArray<1>>;

  /// @brief Default constructor
  BoundaryCore(Options<typename Spline::value_type> options =
                   Options<typename Spline::value_type>{})
      : bdr_({std::tuple_element_t<0, boundary_spline_type>(options),
              std::tuple_element_t<0, boundary_spline_type>(options),
              std::tuple_element_t<1, boundary_spline_type>(options),
              std::tuple_element_t<1, boundary_spline_type>(options)}) {}

  /// @brief Copy constructor
  BoundaryCore(const boundary_type &bdr_) : bdr_(bdr_) {}

  /// @brief Move constructor
  BoundaryCore(boundary_type &&bdr_) : bdr_(bdr_) {}

  /// @brief Copy/clone constructor
  BoundaryCore(const BoundaryCore &other, bool clone)
      : bdr_(clone ? std::apply(
                         [](const auto &...bspline) {
                           return std::make_tuple(bspline.clone()...);
                         },
                         other.coeffs())
                   : other.coeffs()) {}

  /// @brief Constructor
  BoundaryCore(const std::array<int64_t, 2> &ncoeffs,
               enum init init = init::zeros,
               Options<typename Spline::value_type> options =
                   Options<typename Spline::value_type>{})
      : bdr_({std::tuple_element_t<0, boundary_spline_type>(
                  std::array<int64_t, 1>({ncoeffs[1]}), init, options),
              std::tuple_element_t<0, boundary_spline_type>(
                  std::array<int64_t, 1>({ncoeffs[1]}), init, options),
              std::tuple_element_t<1, boundary_spline_type>(
                  std::array<int64_t, 1>({ncoeffs[0]}), init, options),
              std::tuple_element_t<1, boundary_spline_type>(
                  std::array<int64_t, 1>({ncoeffs[0]}), init, options)}) {}

  /// @brief Constructor
  BoundaryCore(
      const std::array<std::vector<typename Spline::value_type>, 2> &kv,
      enum init init = init::zeros,
      Options<typename Spline::value_type> options =
          Options<typename Spline::value_type>{})
      : bdr_({std::tuple_element_t<0, boundary_spline_type>(
                  std::array<std::vector<typename Spline::value_type>, 1>(
                      {kv[1]}),
                  init, options),
              std::tuple_element_t<0, boundary_spline_type>(
                  std::array<std::vector<typename Spline::value_type>, 1>(
                      {kv[1]}),
                  init, options),
              std::tuple_element_t<1, boundary_spline_type>(
                  std::array<std::vector<typename Spline::value_type>, 1>(
                      {kv[0]}),
                  init, options),
              std::tuple_element_t<1, boundary_spline_type>(
                  std::array<std::vector<typename Spline::value_type>, 1>(
                      {kv[0]}),
                  init, options)}) {}

  /// @brief Sets the coefficients of all spline objects from a
  /// single tensor that holds both boundary and inner coefficients
  ///
  /// @param[in] tensor Tensor from which to extract the coefficients
  ///
  /// @result Updates spline object
  inline auto &from_full_tensor(const torch::Tensor &tensor) {

    if (tensor.dim() > 1) {
      auto tensor_view =
          tensor.view({-1, side<west>().ncoeffs(0), side<south>().ncoeffs(0),
                       tensor.size(-1)});

      side<west>().from_tensor(
          tensor_view
              .index({torch::indexing::Slice(), torch::indexing::Slice(), 0})
              .reshape({-1, tensor.size(-1)}));
      side<east>().from_tensor(
          tensor_view
              .index({torch::indexing::Slice(), torch::indexing::Slice(), -1})
              .reshape({-1, tensor.size(-1)}));
      side<south>().from_tensor(
          tensor_view
              .index({torch::indexing::Slice(), 0, torch::indexing::Slice()})
              .reshape({-1, tensor.size(-1)}));
      side<north>().from_tensor(
          tensor_view
              .index({torch::indexing::Slice(), -1, torch::indexing::Slice()})
              .reshape({-1, tensor.size(-1)}));
    } else {
      auto tensor_view =
          tensor.view({-1, side<west>().ncoeffs(0), side<south>().ncoeffs(0)});

      side<west>().from_tensor(
          tensor_view
              .index({torch::indexing::Slice(), torch::indexing::Slice(), 0})
              .flatten());
      side<east>().from_tensor(
          tensor_view
              .index({torch::indexing::Slice(), torch::indexing::Slice(), -1})
              .flatten());
      side<south>().from_tensor(
          tensor_view
              .index({torch::indexing::Slice(), 0, torch::indexing::Slice()})
              .flatten());
      side<north>().from_tensor(
          tensor_view
              .index({torch::indexing::Slice(), -1, torch::indexing::Slice()})
              .flatten());
    }
    return *this;
  }

  /// @brief Returns the number of sides
  inline static constexpr short_t nsides() { return side::north; }

  /// @brief Returns constant reference to the s-th side's spline
  template <short_t s> inline constexpr auto &side() const {
    static_assert(s > none && s <= nsides());
    return std::get<s - 1>(bdr_);
  }

  /// @brief Returns non-constant reference to the s-th side's spline
  template <short_t s> inline constexpr auto &side() {
    static_assert(s > none && s <= nsides());
    return std::get<s - 1>(bdr_);
  }

  /// @brief Returns a constant reference to the array of
  /// coefficients for all boundary segments.
  inline constexpr auto &coeffs() const { return bdr_; }

  /// @brief Returns a non-constant reference to the array of
  /// coefficients for all boundary segments.
  inline constexpr auto &coeffs() { return bdr_; }

  /// @brief Returns the total number of coefficients
  inline int64_t ncumcoeffs() const {
    int64_t s = 0;
    s += side<west>().ncumcoeffs();
    s += side<east>().ncumcoeffs();
    s += side<south>().ncumcoeffs();
    s += side<north>().ncumcoeffs();

    return s;
  }

  /// @brief Returns a string representation of the Boundary object
  inline virtual void
  pretty_print(std::ostream &os = Log(log::info)) const noexcept override {
    os << name() << "(\n"
       << "west = " << side<west>() << "\n"
       << "east = " << side<east>() << "\n"
       << "south = " << side<south>() << "\n"
       << "north = " << side<north>() << "\n)";
  }

  /// @brief Returns the boundary object as JSON object
  inline nlohmann::json to_json() const override {
    nlohmann::json json;
    json["west"] = side<west>().to_json();
    json["east"] = side<east>().to_json();
    json["south"] = side<south>().to_json();
    json["north"] = side<north>().to_json();

    return json;
  }

  /// @brief Updates the boundary object from JSON object
  inline BoundaryCore &from_json(const nlohmann::json &json) {
    side<west>().from_json(json["west"]);
    side<east>().from_json(json["east"]);
    side<south>().from_json(json["south"]);
    side<north>().from_json(json["north"]);

    return *this;
  }

  /// @brief Returns the Greville abscissae
  inline eval_type greville() const {
    return eval_type{side<west>().greville(), side<east>().greville(),
                     side<south>().greville(), side<north>().greville()};
  }
};

/// @brief BoundaryCore (3d specialization)
///
/// This specialization has 6 sides
/// - west  (u=0, v,   w)
/// - east  (u=1, v,   w)
/// - south (u,   v=0, w)
/// - north (u,   v=1, w)
/// - front (u,   v,   w=0)
/// - back  (u,   v,   w=1)
template <typename Spline>
class BoundaryCore<Spline, /* parDim */ 3> : public utils::Serializable,
                                             private utils::FullQualifiedName {

  // Safeguard that template parameter is a valid SplineType
  static_assert(is_SplineType_v<Spline>, "Spline must be a valid SplineType");

  /// @brief Enable access to private members
  template <typename BoundaryCore> friend class BoundaryCommon;

protected:
  /// @brief Spline type
  using spline_type = Spline;

  /// @brief Boundary spline type
  using boundary_spline_type =
      std::tuple<typename Spline::template derived_self_type<
                     typename Spline::value_type, Spline::geoDim(),
                     Spline::degree(1), Spline::degree(2)>,
                 typename Spline::template derived_self_type<
                     typename Spline::value_type, Spline::geoDim(),
                     Spline::degree(0), Spline::degree(2)>,
                 typename Spline::template derived_self_type<
                     typename Spline::value_type, Spline::geoDim(),
                     Spline::degree(0), Spline::degree(1)>>;

  /// @brief Deduces the derived boundary spline type when exposed
  /// to a different class template parameter `real_t`
  template <typename real_t>
  using real_derived_boundary_spline_type = std::tuple<
      typename Spline::template derived_self_type<
          real_t, Spline::geoDim(), Spline::degree(1), Spline::degree(2)>,
      typename Spline::template derived_self_type<
          real_t, Spline::geoDim(), Spline::degree(0), Spline::degree(2)>,
      typename Spline::template derived_self_type<
          real_t, Spline::geoDim(), Spline::degree(0), Spline::degree(1)>>;

  /// @brief Tuple of splines
  std::tuple<typename std::tuple_element_t<0, boundary_spline_type>,
             typename std::tuple_element_t<0, boundary_spline_type>,
             typename std::tuple_element_t<1, boundary_spline_type>,
             typename std::tuple_element_t<1, boundary_spline_type>,
             typename std::tuple_element_t<2, boundary_spline_type>,
             typename std::tuple_element_t<2, boundary_spline_type>>
      bdr_;

public:
  /// @brief Boundary type
  using boundary_type = decltype(bdr_);

  /// @brief Evaluation type
  using eval_type = std::tuple<utils::TensorArray<2>, utils::TensorArray<2>,
                               utils::TensorArray<2>, utils::TensorArray<2>,
                               utils::TensorArray<2>, utils::TensorArray<2>>;

  /// @brief Default constructor
  BoundaryCore(Options<typename Spline::value_type> options =
                   Options<typename Spline::value_type>{})
      : bdr_({std::tuple_element_t<0, boundary_spline_type>(options),
              std::tuple_element_t<0, boundary_spline_type>(options),
              std::tuple_element_t<1, boundary_spline_type>(options),
              std::tuple_element_t<1, boundary_spline_type>(options),
              std::tuple_element_t<2, boundary_spline_type>(options),
              std::tuple_element_t<2, boundary_spline_type>(options)}) {}

  /// @brief Copy constructor
  BoundaryCore(const boundary_type &bdr_) : bdr_(bdr_) {}

  /// @brief Move constructor
  BoundaryCore(boundary_type &&bdr_) : bdr_(bdr_) {}

  /// @brief Copy/clone constructor
  BoundaryCore(const BoundaryCore &other, bool clone)
      : bdr_(clone ? std::apply(
                         [](const auto &...bspline) {
                           return std::make_tuple(bspline.clone()...);
                         },
                         other.coeffs())
                   : other.coeffs()) {}

  /// @brief Constructor
  BoundaryCore(const std::array<int64_t, 3> &ncoeffs,
               enum init init = init::zeros,
               Options<typename Spline::value_type> options =
                   Options<typename Spline::value_type>{})
      : bdr_({std::tuple_element_t<0, boundary_spline_type>(
                  std::array<int64_t, 2>({ncoeffs[1], ncoeffs[2]}), init,
                  options),
              std::tuple_element_t<0, boundary_spline_type>(
                  std::array<int64_t, 2>({ncoeffs[1], ncoeffs[2]}), init,
                  options),
              std::tuple_element_t<1, boundary_spline_type>(
                  std::array<int64_t, 2>({ncoeffs[0], ncoeffs[2]}), init,
                  options),
              std::tuple_element_t<1, boundary_spline_type>(
                  std::array<int64_t, 2>({ncoeffs[0], ncoeffs[2]}), init,
                  options),
              std::tuple_element_t<2, boundary_spline_type>(
                  std::array<int64_t, 2>({ncoeffs[0], ncoeffs[1]}), init,
                  options),
              std::tuple_element_t<2, boundary_spline_type>(
                  std::array<int64_t, 2>({ncoeffs[0], ncoeffs[1]}), init,
                  options)}) {}

  /// @brief Constructor
  BoundaryCore(
      const std::array<std::vector<typename Spline::value_type>, 3> &kv,
      enum init init = init::zeros,
      Options<typename Spline::value_type> options =
          Options<typename Spline::value_type>{})
      : bdr_({std::tuple_element_t<0, boundary_spline_type>(
                  std::array<std::vector<typename Spline::value_type>, 2>(
                      {kv[1], kv[2]}),
                  init, options),
              std::tuple_element_t<0, boundary_spline_type>(
                  std::array<std::vector<typename Spline::value_type>, 2>(
                      {kv[1], kv[2]}),
                  init, options),
              std::tuple_element_t<1, boundary_spline_type>(
                  std::array<std::vector<typename Spline::value_type>, 2>(
                      {kv[0], kv[2]}),
                  init, options),
              std::tuple_element_t<1, boundary_spline_type>(
                  std::array<std::vector<typename Spline::value_type>, 2>(
                      {kv[0], kv[2]}),
                  init, options),
              std::tuple_element_t<2, boundary_spline_type>(
                  std::array<std::vector<typename Spline::value_type>, 2>(
                      {kv[0], kv[1]}),
                  init, options),
              std::tuple_element_t<2, boundary_spline_type>(
                  std::array<std::vector<typename Spline::value_type>, 2>(
                      {kv[0], kv[1]}),
                  init, options)}) {}

  /// @brief Sets the coefficients of all spline objects from a
  /// single tensor that holds both boundary and inner coefficients
  ///
  /// @param[in] tensor Tensor from which to extract the coefficients
  ///
  /// @result Updates spline object
  inline auto &from_full_tensor(const torch::Tensor &tensor) {

    if (tensor.dim() > 1) {
      auto tensor_view =
          tensor.view({-1, side<west>().ncoeffs(1), side<west>().ncoeffs(0),
                       side<south>().ncoeffs(0), tensor.size(-1)});

      side<west>().from_tensor(
          tensor_view
              .index({torch::indexing::Slice(), torch::indexing::Slice(),
                      torch::indexing::Slice(), 0})
              .reshape({-1, tensor.size(-1)}));
      side<east>().from_tensor(
          tensor_view
              .index({torch::indexing::Slice(), torch::indexing::Slice(),
                      torch::indexing::Slice(), -1})
              .reshape({-1, tensor.size(-1)}));
      side<south>().from_tensor(
          tensor_view
              .index({torch::indexing::Slice(), torch::indexing::Slice(), 0,
                      torch::indexing::Slice()})
              .reshape({-1, tensor.size(-1)}));
      side<north>().from_tensor(
          tensor_view
              .index({torch::indexing::Slice(), torch::indexing::Slice(), -1,
                      torch::indexing::Slice()})
              .reshape({-1, tensor.size(-1)}));
      side<front>().from_tensor(
          tensor_view
              .index({torch::indexing::Slice(), 0, torch::indexing::Slice(),
                      torch::indexing::Slice()})
              .reshape({-1, tensor.size(-1)}));
      side<back>().from_tensor(
          tensor_view
              .index({torch::indexing::Slice(), -1, torch::indexing::Slice(),
                      torch::indexing::Slice()})
              .reshape({-1, tensor.size(-1)}));
    } else {
      auto tensor_view =
          tensor.view({-1, side<west>().ncoeffs(1), side<west>().ncoeffs(0),
                       side<south>().ncoeffs(0)});

      side<west>().from_tensor(
          tensor_view
              .index({torch::indexing::Slice(), torch::indexing::Slice(),
                      torch::indexing::Slice(), 0})
              .flatten());
      side<east>().from_tensor(
          tensor_view
              .index({torch::indexing::Slice(), torch::indexing::Slice(),
                      torch::indexing::Slice(), -1})
              .flatten());

      side<south>().from_tensor(
          tensor_view
              .index({torch::indexing::Slice(), torch::indexing::Slice(), 0,
                      torch::indexing::Slice()})
              .flatten());
      side<north>().from_tensor(
          tensor_view
              .index({torch::indexing::Slice(), torch::indexing::Slice(), -1,
                      torch::indexing::Slice()})
              .flatten());

      side<front>().from_tensor(
          tensor_view
              .index({torch::indexing::Slice(), 0, torch::indexing::Slice(),
                      torch::indexing::Slice()})
              .flatten());
      side<back>().from_tensor(
          tensor_view
              .index({torch::indexing::Slice(), -1, torch::indexing::Slice(),
                      torch::indexing::Slice()})
              .flatten());
    }
    return *this;
  }

  /// @brief Returns the number of sides
  inline static constexpr short_t nsides() { return side::back; }

  /// @brief Returns constant reference to side-th spline
  template <short_t s> inline constexpr auto &side() const {
    static_assert(s > none && s <= nsides());
    return std::get<s - 1>(bdr_);
  }

  /// @brief Returns non-constant reference to side-th spline
  template <short_t s> inline constexpr auto &side() {
    static_assert(s > none && s <= nsides());
    return std::get<s - 1>(bdr_);
  }

  /// @brief Returns a constant reference to the array of
  /// coefficients for all boundary segments.
  inline constexpr auto &coeffs() const { return bdr_; }

  /// @brief Returns a non-constant reference to the array of
  /// coefficients for all boundary segments.
  inline constexpr auto &coeffs() { return bdr_; }

  /// @brief Returns the total number of coefficients
  inline int64_t ncumcoeffs() const {
    int64_t s = 0;
    s += side<west>().ncumcoeffs();
    s += side<east>().ncumcoeffs();
    s += side<south>().ncumcoeffs();
    s += side<north>().ncumcoeffs();
    s += side<front>().ncumcoeffs();
    s += side<back>().ncumcoeffs();

    return s;
  }

  /// @brief Returns a string representation of the Boundary object
  inline virtual void
  pretty_print(std::ostream &os = Log(log::info)) const noexcept override {
    os << name() << "(\n"
       << "west = " << side<west>() << "\n"
       << "east = " << side<east>() << "\n"
       << "south = " << side<south>() << "\n"
       << "north = " << side<north>() << "\n"
       << "front = " << side<front>() << "\n"
       << "back = " << side<back>() << "\n)";
  }

  /// @brief Returns the boundary object as JSON object
  inline nlohmann::json to_json() const override {
    nlohmann::json json;
    json["west"] = side<west>().to_json();
    json["east"] = side<east>().to_json();
    json["south"] = side<south>().to_json();
    json["north"] = side<north>().to_json();
    json["front"] = side<front>().to_json();
    json["back"] = side<back>().to_json();

    return json;
  }

  /// @brief Updates the boundary object from JSON object
  inline BoundaryCore &from_json(const nlohmann::json &json) {
    side<west>().from_json(json["west"]);
    side<east>().from_json(json["east"]);
    side<south>().from_json(json["south"]);
    side<north>().from_json(json["north"]);
    side<front>().from_json(json["front"]);
    side<back>().from_json(json["back"]);

    return *this;
  }

  /// @brief Returns the Greville abscissae
  inline eval_type greville() const {
    return eval_type{side<west>().greville(),  side<east>().greville(),
                     side<south>().greville(), side<north>().greville(),
                     side<front>().greville(), side<back>().greville()};
  }
};

/// @brief BoundaryCore (4d specialization)
///
/// This specialization has 8 sides
/// - west  (u=0, v,   w,   t)
/// - east  (u=1, v,   w,   t)
/// - south (u,   v=0, w,   t)
/// - north (u,   v=1, w,   t)
/// - front (u,   v,   w=0, t)
/// - back  (u,   v,   w=1, t)
/// - stime (u,   v,   w,   t=0)
/// - etime (u,   v,   w,   t=1)
template <typename Spline>
class BoundaryCore<Spline, /* parDim */ 4> : public utils::Serializable,
                                             private utils::FullQualifiedName {

  // Safeguard that template parameter is a valid SplineType
  static_assert(is_SplineType_v<Spline>, "Spline must be a valid SplineType");

  /// @brief Enable access to private members
  template <typename BoundaryCore> friend class BoundaryCommon;

protected:
  /// @brief Spline type
  using spline_type = Spline;

  /// @brief Array storing the degrees
  using boundary_spline_type =
      std::tuple<typename Spline::template derived_self_type<
                     typename Spline::value_type, Spline::geoDim(),
                     Spline::degree(1), Spline::degree(2), Spline::degree(3)>,
                 typename Spline::template derived_self_type<
                     typename Spline::value_type, Spline::geoDim(),
                     Spline::degree(0), Spline::degree(2), Spline::degree(3)>,
                 typename Spline::template derived_self_type<
                     typename Spline::value_type, Spline::geoDim(),
                     Spline::degree(0), Spline::degree(1), Spline::degree(3)>,
                 typename Spline::template derived_self_type<
                     typename Spline::value_type, Spline::geoDim(),
                     Spline::degree(0), Spline::degree(1), Spline::degree(2)>>;

  /// @brief Deduces the derived boundary spline type when exposed
  /// to a different class template parameter `real_t`
  template <typename real_t>
  using real_derived_boundary_spline_type =
      std::tuple<typename Spline::template derived_self_type<
                     real_t, Spline::geoDim(), Spline::degree(1),
                     Spline::degree(2), Spline::degree(3)>,
                 typename Spline::template derived_self_type<
                     real_t, Spline::geoDim(), Spline::degree(0),
                     Spline::degree(2), Spline::degree(3)>,
                 typename Spline::template derived_self_type<
                     real_t, Spline::geoDim(), Spline::degree(0),
                     Spline::degree(1), Spline::degree(3)>,
                 typename Spline::template derived_self_type<
                     real_t, Spline::geoDim(), Spline::degree(0),
                     Spline::degree(1), Spline::degree(2)>>;

  /// @brief Tuple of splines
  std::tuple<typename std::tuple_element_t<0, boundary_spline_type>,
             typename std::tuple_element_t<0, boundary_spline_type>,
             typename std::tuple_element_t<1, boundary_spline_type>,
             typename std::tuple_element_t<1, boundary_spline_type>,
             typename std::tuple_element_t<2, boundary_spline_type>,
             typename std::tuple_element_t<2, boundary_spline_type>,
             typename std::tuple_element_t<3, boundary_spline_type>,
             typename std::tuple_element_t<3, boundary_spline_type>>
      bdr_;

public:
  /// @brief Boundary type
  using boundary_type = decltype(bdr_);

  /// @brief Evaluation type
  using eval_type = std::tuple<utils::TensorArray<3>, utils::TensorArray<3>,
                               utils::TensorArray<3>, utils::TensorArray<3>,
                               utils::TensorArray<3>, utils::TensorArray<3>,
                               utils::TensorArray<3>, utils::TensorArray<3>>;

  /// @brief Default constructor
  BoundaryCore(Options<typename Spline::value_type> options =
                   Options<typename Spline::value_type>{})
      : bdr_({std::tuple_element_t<0, boundary_spline_type>(options),
              std::tuple_element_t<0, boundary_spline_type>(options),
              std::tuple_element_t<1, boundary_spline_type>(options),
              std::tuple_element_t<1, boundary_spline_type>(options),
              std::tuple_element_t<2, boundary_spline_type>(options),
              std::tuple_element_t<2, boundary_spline_type>(options),
              std::tuple_element_t<3, boundary_spline_type>(options),
              std::tuple_element_t<3, boundary_spline_type>(options)}) {}

  /// @brief Copy constructor
  BoundaryCore(const boundary_type &bdr_) : bdr_(bdr_) {}

  /// @brief Move constructor
  BoundaryCore(boundary_type &&bdr_) : bdr_(bdr_) {}

  /// @brief Copy/clone constructor
  BoundaryCore(const BoundaryCore &other, bool clone)
      : bdr_(clone ? std::apply(
                         [](const auto &...bspline) {
                           return std::make_tuple(bspline.clone()...);
                         },
                         other.coeffs())
                   : other.coeffs()) {}

  /// @brief Constructor
  BoundaryCore(const std::array<int64_t, 4> &ncoeffs,
               enum init init = init::zeros,
               Options<typename Spline::value_type> options =
                   Options<typename Spline::value_type>{})
      : bdr_({std::tuple_element_t<0, boundary_spline_type>(
                  std::array<int64_t, 3>({ncoeffs[1], ncoeffs[2], ncoeffs[3]}),
                  init, options),
              std::tuple_element_t<0, boundary_spline_type>(
                  std::array<int64_t, 3>({ncoeffs[1], ncoeffs[2], ncoeffs[3]}),
                  init, options),
              std::tuple_element_t<1, boundary_spline_type>(
                  std::array<int64_t, 3>({ncoeffs[0], ncoeffs[2], ncoeffs[3]}),
                  init, options),
              std::tuple_element_t<1, boundary_spline_type>(
                  std::array<int64_t, 3>({ncoeffs[0], ncoeffs[2], ncoeffs[3]}),
                  init, options),
              std::tuple_element_t<2, boundary_spline_type>(
                  std::array<int64_t, 3>({ncoeffs[0], ncoeffs[1], ncoeffs[3]}),
                  init, options),
              std::tuple_element_t<2, boundary_spline_type>(
                  std::array<int64_t, 3>({ncoeffs[0], ncoeffs[1], ncoeffs[3]}),
                  init, options),
              std::tuple_element_t<3, boundary_spline_type>(
                  std::array<int64_t, 3>({ncoeffs[0], ncoeffs[1], ncoeffs[2]}),
                  init, options),
              std::tuple_element_t<3, boundary_spline_type>(
                  std::array<int64_t, 3>({ncoeffs[0], ncoeffs[1], ncoeffs[2]}),
                  init, options)}) {}

  /// @brief Constructor
  BoundaryCore(
      const std::array<std::vector<typename Spline::value_type>, 4> &kv,
      enum init init = init::zeros,
      Options<typename Spline::value_type> options =
          Options<typename Spline::value_type>{})
      : bdr_({std::tuple_element_t<0, boundary_spline_type>(
                  std::array<std::vector<typename Spline::value_type>, 3>(
                      {kv[1], kv[2], kv[3]}),
                  init, options),
              std::tuple_element_t<0, boundary_spline_type>(
                  std::array<std::vector<typename Spline::value_type>, 3>(
                      {kv[1], kv[2], kv[3]}),
                  init, options),
              std::tuple_element_t<1, boundary_spline_type>(
                  std::array<std::vector<typename Spline::value_type>, 3>(
                      {kv[0], kv[2], kv[3]}),
                  init, options),
              std::tuple_element_t<1, boundary_spline_type>(
                  std::array<std::vector<typename Spline::value_type>, 3>(
                      {kv[0], kv[2], kv[3]}),
                  init, options),
              std::tuple_element_t<2, boundary_spline_type>(
                  std::array<std::vector<typename Spline::value_type>, 3>(
                      {kv[0], kv[1], kv[3]}),
                  init, options),
              std::tuple_element_t<2, boundary_spline_type>(
                  std::array<std::vector<typename Spline::value_type>, 3>(
                      {kv[0], kv[1], kv[3]}),
                  init, options),
              std::tuple_element_t<3, boundary_spline_type>(
                  std::array<std::vector<typename Spline::value_type>, 3>(
                      {kv[0], kv[1], kv[2]}),
                  init, options),
              std::tuple_element_t<3, boundary_spline_type>(
                  std::array<std::vector<typename Spline::value_type>, 3>(
                      {kv[0], kv[1], kv[2]}),
                  init, options)}) {}

  /// @brief Sets the coefficients of all spline objects from a
  /// single tensor that holds both boundary and inner coefficients
  ///
  /// @param[in] tensor Tensor from which to extract the coefficients
  ///
  /// @result Updates spline object
  inline auto &from_full_tensor(const torch::Tensor &tensor) {

    if (tensor.dim() > 1) {
      auto tensor_view = tensor.view(
          {-1, side<west>().ncoeffs(2), side<west>().ncoeffs(1),
           side<west>().ncoeffs(0), side<south>().ncoeffs(0), tensor.size(-1)});

      side<west>().from_tensor(
          tensor_view
              .index({torch::indexing::Slice(), torch::indexing::Slice(),
                      torch::indexing::Slice(), torch::indexing::Slice(), 0})
              .reshape({-1, tensor.size(-1)}));
      side<east>().from_tensor(
          tensor_view
              .index({torch::indexing::Slice(), torch::indexing::Slice(),
                      torch::indexing::Slice(), torch::indexing::Slice(), -1})
              .reshape({-1, tensor.size(-1)}));
      side<south>().from_tensor(
          tensor_view
              .index({torch::indexing::Slice(), torch::indexing::Slice(),
                      torch::indexing::Slice(), 0, torch::indexing::Slice()})
              .reshape({-1, tensor.size(-1)}));
      side<north>().from_tensor(
          tensor_view
              .index({torch::indexing::Slice(), torch::indexing::Slice(),
                      torch::indexing::Slice(), -1, torch::indexing::Slice()})
              .reshape({-1, tensor.size(-1)}));
      side<front>().from_tensor(
          tensor_view
              .index({torch::indexing::Slice(), torch::indexing::Slice(), 0,
                      torch::indexing::Slice(), torch::indexing::Slice()})
              .reshape({-1, tensor.size(-1)}));
      side<back>().from_tensor(
          tensor_view
              .index({torch::indexing::Slice(), torch::indexing::Slice(), -1,
                      torch::indexing::Slice(), torch::indexing::Slice()})
              .reshape({-1, tensor.size(-1)}));
      side<stime>().from_tensor(
          tensor_view
              .index({torch::indexing::Slice(), 0, torch::indexing::Slice(),
                      torch::indexing::Slice(), torch::indexing::Slice()})
              .reshape({-1, tensor.size(-1)}));
      side<etime>().from_tensor(
          tensor_view
              .index({torch::indexing::Slice(), -1, torch::indexing::Slice(),
                      torch::indexing::Slice(), torch::indexing::Slice()})
              .reshape({-1, tensor.size(-1)}));
    } else {
      auto tensor_view =
          tensor.view({-1, side<west>().ncoeffs(2), side<west>().ncoeffs(1),
                       side<west>().ncoeffs(0), side<south>().ncoeffs(0)});

      side<west>().from_tensor(
          tensor_view
              .index({torch::indexing::Slice(), torch::indexing::Slice(),
                      torch::indexing::Slice(), torch::indexing::Slice(), 0})
              .flatten());
      side<east>().from_tensor(
          tensor_view
              .index({torch::indexing::Slice(), torch::indexing::Slice(),
                      torch::indexing::Slice(), torch::indexing::Slice(), -1})
              .flatten());

      side<south>().from_tensor(
          tensor_view
              .index({torch::indexing::Slice(), torch::indexing::Slice(),
                      torch::indexing::Slice(), 0, torch::indexing::Slice()})
              .flatten());
      side<north>().from_tensor(
          tensor_view
              .index({torch::indexing::Slice(), torch::indexing::Slice(),
                      torch::indexing::Slice(), -1, torch::indexing::Slice()})
              .flatten());

      side<front>().from_tensor(
          tensor_view
              .index({torch::indexing::Slice(), torch::indexing::Slice(), 0,
                      torch::indexing::Slice(), torch::indexing::Slice()})
              .flatten());
      side<back>().from_tensor(
          tensor_view
              .index({torch::indexing::Slice(), torch::indexing::Slice(), -1,
                      torch::indexing::Slice(), torch::indexing::Slice()})
              .flatten());

      side<stime>().from_tensor(
          tensor_view
              .index({torch::indexing::Slice(), 0, torch::indexing::Slice(),
                      torch::indexing::Slice(), torch::indexing::Slice()})
              .flatten());
      side<etime>().from_tensor(
          tensor_view
              .index({torch::indexing::Slice(), -1, torch::indexing::Slice(),
                      torch::indexing::Slice(), torch::indexing::Slice()})
              .flatten());
    }
    return *this;
  }

  /// @brief Returns the number of sides
  inline static constexpr short_t nsides() { return side::etime; }

  /// @brief Returns constant reference to side-th spline
  template <short_t s> inline constexpr auto &side() const {
    static_assert(s > none && s <= nsides());
    return std::get<s - 1>(bdr_);
  }

  /// @brief Returns non-constant reference to side-th spline
  template <short_t s> inline constexpr auto &side() {
    static_assert(s > none && s <= nsides());
    return std::get<s - 1>(bdr_);
  }

  /// @brief Returns a constant reference to the array of
  /// coefficients for all boundary segments.
  inline constexpr auto &coeffs() const { return bdr_; }

  /// @brief Returns a non-constant reference to the array of
  /// coefficients for all boundary segments.
  inline constexpr auto &coeffs() { return bdr_; }

  /// @brief Returns the total number of coefficients
  inline int64_t ncumcoeffs() const {
    int64_t s = 0;
    s += side<west>().ncumcoeffs();
    s += side<east>().ncumcoeffs();
    s += side<south>().ncumcoeffs();
    s += side<north>().ncumcoeffs();
    s += side<front>().ncumcoeffs();
    s += side<back>().ncumcoeffs();
    s += side<stime>().ncumcoeffs();
    s += side<etime>().ncumcoeffs();

    return s;
  }

  /// @brief Returns a string representation of the Boundary object
  inline virtual void
  pretty_print(std::ostream &os = Log(log::info)) const noexcept override {
    os << name() << "(\n"
       << "west = " << side<west>() << "\n"
       << "east = " << side<east>() << "\n"
       << "south = " << side<south>() << "\n"
       << "north = " << side<north>() << "\n"
       << "front = " << side<front>() << "\n"
       << "back = " << side<back>() << "\n"
       << "stime = " << side<stime>() << "\n"
       << "etime = " << side<etime>() << "\n)";
  }

  /// @brief Returns the boundary object as JSON object
  inline nlohmann::json to_json() const override {
    nlohmann::json json;
    json["west"] = side<west>().to_json();
    json["east"] = side<east>().to_json();
    json["south"] = side<south>().to_json();
    json["north"] = side<north>().to_json();
    json["front"] = side<front>().to_json();
    json["back"] = side<back>().to_json();
    json["stime"] = side<stime>().to_json();
    json["etime"] = side<etime>().to_json();

    return json;
  }

  /// @brief Updates the boundary object from JSON object
  inline BoundaryCore &from_json(const nlohmann::json &json) {
    side<west>().from_json(json["west"]);
    side<east>().from_json(json["east"]);
    side<south>().from_json(json["south"]);
    side<north>().from_json(json["north"]);
    side<front>().from_json(json["front"]);
    side<back>().from_json(json["back"]);
    side<stime>().from_json(json["stime"]);
    side<etime>().from_json(json["etime"]);

    return *this;
  }

  /// @brief Returns the Greville abscissae
  inline eval_type greville() const {
    return eval_type{side<west>().greville(),  side<east>().greville(),
                     side<south>().greville(), side<north>().greville(),
                     side<front>().greville(), side<back>().greville(),
                     side<stime>().greville(), side<etime>().greville()};
  }
};

/// @brief

namespace detail {
/// @brief Boundary type
class BoundaryType {};
} // namespace detail

/// @brief Type trait to check if T is a valid Boundary type
template <typename... T>
using is_BoundaryType =
    std::conjunction<std::is_base_of<detail::BoundaryType, T>...>;

/// @brief Alias to the value of is_BoundaryType
template <typename... T>
inline constexpr bool is_BoundaryType_v = is_BoundaryType<T...>::value;

/// @brief Boundary (common high-level functionality)
template <typename BoundaryCore>
class BoundaryCommon : private detail::BoundaryType, public BoundaryCore {
public:
  /// @brief Constructors from the base class
  using BoundaryCore::BoundaryCore;

  /// @brief Returns a clone of the boundary object
  BoundaryCommon clone() const { return BoundaryCommon(*this); }

private:
  /// @brief Returns all coefficients of all spline objects as a
  /// single tensor
  ///
  /// @result Tensor of coefficients
  template <std::size_t... Is>
  inline torch::Tensor as_tensor_(std::index_sequence<Is...>) const {
    return torch::cat({std::get<Is>(BoundaryCore::bdr_).as_tensor()...});
  }

public:
  /// @brief Returns all coefficients of all spline objects as a
  /// single tensor
  ///
  /// @result Tensor of coefficients
  inline torch::Tensor as_tensor() const {
    return as_tensor_(std::make_index_sequence<BoundaryCore::nsides()>{});
  }

private:
  /// @brief Returns the size of the single tensor representation of
  /// all spline objects
  ///
  /// @result Size of the tensor
  template <std::size_t... Is>
  inline int64_t as_tensor_size_(std::index_sequence<Is...>) const {
    return std::apply(
        [](auto... size) { return (size + ...); },
        std::make_tuple(std::get<Is>(BoundaryCore::bdr_).as_tensor_size()...));
  }

public:
  /// @brief Returns the size of the single tensor representation of
  /// all spline objects
  //
  /// @result Size of the tensor
  inline int64_t as_tensor_size() const {
    return as_tensor_size_(std::make_index_sequence<BoundaryCore::nsides()>{});
  }

private:
  /// @brief Sets the coefficients of all spline objects from a
  /// single tensor
  ///
  /// @param[in] tensor Tensor from which to extract the coefficients
  ///
  /// @result Updates spline object
  template <std::size_t... Is>
  inline auto &from_tensor_(std::index_sequence<Is...>,
                            const torch::Tensor &tensor) {

    std::size_t start(0);
    auto end = [&start](std::size_t inc) { return start += inc; };

    (std::get<Is>(BoundaryCore::bdr_)
         .from_tensor(tensor.index({torch::indexing::Slice(
             start, end(std::get<Is>(BoundaryCore::bdr_).ncumcoeffs() *
                        std::get<Is>(BoundaryCore::bdr_).geoDim()))})),
     ...);

    return *this;
  }

public:
  /// @brief Sets the coefficients of all spline objects from a
  /// single tensor
  ///
  /// @param[in] tensor Tensor from which to extract the coefficients
  ///
  /// @result Updated spline objects
  inline auto &from_tensor(const torch::Tensor &tensor) {
    return from_tensor_(std::make_index_sequence<BoundaryCore::nsides()>{},
                        tensor);
  }

private:
  /// @brief Returns the values of the boundary spline objects in
  /// the points `xi` @{
  template <deriv deriv = deriv::func, bool memory_optimized = false,
            size_t... Is, typename... Xi>
  inline auto eval_(std::index_sequence<Is...>,
                    const std::tuple<Xi...> &xi) const {
    return std::tuple(
        std::get<Is>(BoundaryCore::bdr_)
            .template eval<deriv, memory_optimized>(std::get<Is>(xi))...);
  }

  template <deriv deriv = deriv::func, bool memory_optimized = false,
            size_t... Is, typename... Xi, typename... Indices>
  inline auto eval_(std::index_sequence<Is...>, const std::tuple<Xi...> &xi,
                    const std::tuple<Indices...> &indices) const {
    return std::tuple(std::get<Is>(BoundaryCore::bdr_)
                          .template eval<deriv, memory_optimized>(
                              std::get<Is>(xi), std::get<Is>(indices))...);
  }

  template <deriv deriv = deriv::func, bool memory_optimized = false,
            size_t... Is, typename... Xi, typename... Indices,
            typename... Coeff_Indices>
  inline auto eval_(std::index_sequence<Is...>, const std::tuple<Xi...> &xi,
                    const std::tuple<Indices...> &indices,
                    const std::tuple<Coeff_Indices...> &coeff_indices) const {
    return std::tuple(std::get<Is>(BoundaryCore::bdr_)
                          .template eval<deriv, memory_optimized>(
                              std::get<Is>(xi), std::get<Is>(indices),
                              std::get<Is>(coeff_indices))...);
  }
  /// @}

public:
  /// @brief Returns the values of the spline objects in the points `xi`
  /// @{
  template <deriv deriv = deriv::func, bool memory_optimized = false,
            typename... Xi>
  inline auto eval(const std::tuple<Xi...> &xi) const {
    return eval_<deriv, memory_optimized>(
        std::make_index_sequence<BoundaryCore::nsides()>{}, xi);
  }

  template <deriv deriv = deriv::func, bool memory_optimized = false,
            typename... Xi, typename... Indices>
  inline auto eval(const std::tuple<Xi...> &xi,
                   const std::tuple<Indices...> &indices) const {
    static_assert(sizeof...(Xi) == sizeof...(Indices));
    return eval_<deriv, memory_optimized>(
        std::make_index_sequence<BoundaryCore::nsides()>{}, xi, indices);
  }

  template <deriv deriv = deriv::func, bool memory_optimized = false,
            typename... Xi, typename... Indices, typename... Coeff_Indices>
  inline auto eval(const std::tuple<Xi...> &xi,
                   const std::tuple<Indices...> &indices,
                   const std::tuple<Coeff_Indices...> &coeff_indices) const {
    static_assert(sizeof...(Xi) == sizeof...(Indices) &&
                  sizeof...(Xi) == sizeof...(Coeff_Indices));
    return eval_<deriv, memory_optimized>(
        std::make_index_sequence<BoundaryCore::nsides()>{}, xi, indices,
        coeff_indices);
  }
  /// @}

private:
  /// @brief Returns the value of the boundary spline objects from
  /// precomputed basis function @{
  template <size_t... Is, typename... Basfunc, typename... Coeff_Indices,
            typename... Numeval, typename... Sizes>
  inline auto
  eval_from_precomputed_(std::index_sequence<Is...>,
                         const std::tuple<Basfunc...> &basfunc,
                         const std::tuple<Coeff_Indices...> &coeff_indices,
                         const std::tuple<Numeval...> &numeval,
                         const std::tuple<Sizes...> &sizes) const {
    return std::tuple(std::get<Is>(BoundaryCore::bdr_)
                          .eval_from_precomputed(std::get<Is>(basfunc),
                                                 std::get<Is>(coeff_indices),
                                                 std::get<Is>(numeval),
                                                 std::get<Is>(sizes))...);
  }

  template <size_t... Is, typename... Basfunc, typename... Coeff_Indices,
            typename... Xi>
  inline auto
  eval_from_precomputed_(std::index_sequence<Is...>,
                         const std::tuple<Basfunc...> &basfunc,
                         const std::tuple<Coeff_Indices...> &coeff_indices,
                         const std::tuple<Xi...> &xi) const {
    return std::tuple(
        std::get<Is>(BoundaryCore::bdr_)
            .eval_from_precomputed(
                std::get<Is>(basfunc), std::get<Is>(coeff_indices),
                std::get<Is>(xi)[0].numel(), std::get<Is>(xi)[0].sizes())...);
  }
  /// @}

public:
  /// @brief Returns the value of the spline objects from
  /// precomputed basis function @{
  template <typename... Basfunc, typename... Coeff_Indices, typename... Numeval,
            typename... Sizes>
  inline auto
  eval_from_precomputed(const std::tuple<Basfunc...> &basfunc,
                        const std::tuple<Coeff_Indices...> &coeff_indices,
                        const std::tuple<Numeval...> &numeval,
                        const std::tuple<Sizes...> &sizes) const {
    static_assert(sizeof...(Basfunc) == sizeof...(Coeff_Indices) &&
                  sizeof...(Basfunc) == sizeof...(Numeval) &&
                  sizeof...(Basfunc) == sizeof...(Sizes));
    return eval_from_precomputed_(
        std::make_index_sequence<BoundaryCore::nsides()>{}, basfunc,
        coeff_indices, numeval, sizes);
  }

  template <typename... Basfunc, typename... Coeff_Indices, typename... Xi>
  inline auto
  eval_from_precomputed(const std::tuple<Basfunc...> &basfunc,
                        const std::tuple<Coeff_Indices...> &coeff_indices,
                        const std::tuple<Xi...> &xi) const {
    static_assert(sizeof...(Basfunc) == sizeof...(Coeff_Indices) &&
                  sizeof...(Basfunc) == sizeof...(Xi));
    return eval_from_precomputed_(
        std::make_index_sequence<BoundaryCore::nsides()>{}, basfunc,
        coeff_indices, xi);
  }
  /// @}

private:
  /// @brief Returns the knot indicies of boundary spline object's
  /// knot spans containing `xi`
  template <size_t... Is, typename... Xi>
  inline auto find_knot_indices_(std::index_sequence<Is...>,
                                 const std::tuple<Xi...> &xi) const {
    return std::tuple(std::get<Is>(BoundaryCore::bdr_)
                          .find_knot_indices(std::get<Is>(xi))...);
  }

public:
  /// @brief Returns the knot indicies of knot spans containing `xi`
  template <typename... Xi>
  inline auto find_knot_indices(const std::tuple<Xi...> &xi) const {
    return find_knot_indices_(
        std::make_index_sequence<BoundaryCore::nsides()>{}, xi);
  }

private:
  /// @brief Returns the values of the boundary spline spline
  /// object's basis functions in the points `xi`
  /// @{
  template <deriv deriv = deriv::func, bool memory_optimized = false,
            size_t... Is, typename... Xi>
  inline auto eval_basfunc_(std::index_sequence<Is...>,
                            const std::tuple<Xi...> &xi) const {
    return std::tuple(std::get<Is>(BoundaryCore::bdr_)
                          .template eval_basfunc<deriv, memory_optimized>(
                              std::get<Is>(xi))...);
  }

  template <deriv deriv = deriv::func, bool memory_optimized = false,
            size_t... Is, typename... Xi, typename... Indices>
  inline auto eval_basfunc_(std::index_sequence<Is...>,
                            const std::tuple<Xi...> &xi,
                            const std::tuple<Indices...> &indices) const {
    return std::tuple(std::get<Is>(BoundaryCore::bdr_)
                          .template eval_basfunc<deriv, memory_optimized>(
                              std::get<Is>(xi), std::get<Is>(indices))...);
  }
  /// @}

public:
  /// @brief Returns the values of the spline objects' basis
  /// functions in the points `xi` @{
  template <deriv deriv = deriv::func, bool memory_optimized = false,
            typename... Xi>
  inline auto eval_basfunc(const std::tuple<Xi...> &xi) const {
    return eval_basfunc_<deriv, memory_optimized>(
        std::make_index_sequence<BoundaryCore::nsides()>{}, xi);
  }

  template <deriv deriv = deriv::func, bool memory_optimized = false,
            typename... Xi, typename... Indices>
  inline auto eval_basfunc(const std::tuple<Xi...> &xi,
                           const std::tuple<Indices...> &indices) const {
    static_assert(sizeof...(Xi) == sizeof...(Indices));
    return eval_basfunc_<deriv, memory_optimized>(
        std::make_index_sequence<BoundaryCore::nsides()>{}, xi, indices);
  }
  /// @}

private:
  /// @brief Returns the indices of the boundary spline object's
  /// coefficients corresponding to the knot indices `indices`
  template <bool memory_optimized = false, size_t... Is, typename... Indices>
  inline auto find_coeff_indices_(std::index_sequence<Is...>,
                                  const std::tuple<Indices...> &indices) const {
    return std::tuple(std::get<Is>(BoundaryCore::bdr_)
                          .template find_coeff_indices<memory_optimized>(
                              std::get<Is>(indices))...);
  }

public:
  /// @brief Returns the indices of the spline objects'
  /// coefficients corresponding to the knot indices `indices`
  template <bool memory_optimized = false, typename... Indices>
  inline auto find_coeff_indices(const std::tuple<Indices...> &indices) const {
    return find_coeff_indices_<memory_optimized>(
        std::make_index_sequence<BoundaryCore::nsides()>{}, indices);
  }

private:
  /// @brief Returns the boundary spline object with uniformly
  /// refined knot and coefficient vectors
  template <size_t... Is>
  inline auto &uniform_refine_(std::index_sequence<Is...>, int numRefine = 1,
                               int dim = -1) {
    (std::get<Is>(BoundaryCore::bdr_).uniform_refine(numRefine, dim), ...);
    return *this;
  }

public:
  /// @brief Returns the spline objects with uniformly refined
  /// knot and coefficient vectors
  inline auto &uniform_refine(int numRefine = 1, int dim = -1) {
    if (dim == -1) {
      if constexpr (BoundaryCore::spline_type::parDim() > 1)
        uniform_refine_(std::make_index_sequence<BoundaryCore::nsides()>{},
                        numRefine, dim);
    } else if (dim == 0) {
      if constexpr (BoundaryCore::nsides() == 2) {
        // We do not refine the boundary of a curve
      } else if constexpr (BoundaryCore::nsides() == 4) {
        std::get<side::south - 1>(BoundaryCore::bdr_)
            .uniform_refine(numRefine, 0);
        std::get<side::north - 1>(BoundaryCore::bdr_)
            .uniform_refine(numRefine, 0);
      } else if constexpr (BoundaryCore::nsides() == 6) {
        std::get<side::south - 1>(BoundaryCore::bdr_)
            .uniform_refine(numRefine, 0);
        std::get<side::north - 1>(BoundaryCore::bdr_)
            .uniform_refine(numRefine, 0);
        std::get<side::front - 1>(BoundaryCore::bdr_)
            .uniform_refine(numRefine, 0);
        std::get<side::back - 1>(BoundaryCore::bdr_)
            .uniform_refine(numRefine, 0);
      } else if constexpr (BoundaryCore::nsides() == 8) {
        std::get<side::south - 1>(BoundaryCore::bdr_)
            .uniform_refine(numRefine, 0);
        std::get<side::north - 1>(BoundaryCore::bdr_)
            .uniform_refine(numRefine, 0);
        std::get<side::front - 1>(BoundaryCore::bdr_)
            .uniform_refine(numRefine, 0);
        std::get<side::back - 1>(BoundaryCore::bdr_)
            .uniform_refine(numRefine, 0);
        std::get<side::stime - 1>(BoundaryCore::bdr_)
            .uniform_refine(numRefine, 0);
        std::get<side::etime - 1>(BoundaryCore::bdr_)
            .uniform_refine(numRefine, 0);
      } else
        throw std::runtime_error("Invalid dimension");
    } else if (dim == 1) {
      if constexpr (BoundaryCore::nsides() == 4) {
        std::get<side::east - 1>(BoundaryCore::bdr_)
            .uniform_refine(numRefine, 0);
        std::get<side::west - 1>(BoundaryCore::bdr_)
            .uniform_refine(numRefine, 0);
      } else if constexpr (BoundaryCore::nsides() == 6) {
        std::get<side::east - 1>(BoundaryCore::bdr_)
            .uniform_refine(numRefine, 0);
        std::get<side::west - 1>(BoundaryCore::bdr_)
            .uniform_refine(numRefine, 0);
        std::get<side::front - 1>(BoundaryCore::bdr_)
            .uniform_refine(numRefine, 1);
        std::get<side::back - 1>(BoundaryCore::bdr_)
            .uniform_refine(numRefine, 1);

      } else if constexpr (BoundaryCore::nsides() == 8) {
        std::get<side::east - 1>(BoundaryCore::bdr_)
            .uniform_refine(numRefine, 0);
        std::get<side::west - 1>(BoundaryCore::bdr_)
            .uniform_refine(numRefine, 0);
        std::get<side::front - 1>(BoundaryCore::bdr_)
            .uniform_refine(numRefine, 1);
        std::get<side::back - 1>(BoundaryCore::bdr_)
            .uniform_refine(numRefine, 1);
        std::get<side::stime - 1>(BoundaryCore::bdr_)
            .uniform_refine(numRefine, 1);
        std::get<side::etime - 1>(BoundaryCore::bdr_)
            .uniform_refine(numRefine, 1);
      } else
        throw std::runtime_error("Invalid dimension");
    } else if (dim == 2) {
      if constexpr (BoundaryCore::nsides() == 6) {
        std::get<side::east - 1>(BoundaryCore::bdr_)
            .uniform_refine(numRefine, 1);
        std::get<side::west - 1>(BoundaryCore::bdr_)
            .uniform_refine(numRefine, 1);
        std::get<side::north - 1>(BoundaryCore::bdr_)
            .uniform_refine(numRefine, 1);
        std::get<side::south - 1>(BoundaryCore::bdr_)
            .uniform_refine(numRefine, 1);
      } else if constexpr (BoundaryCore::nsides() == 8) {
        std::get<side::west - 1>(BoundaryCore::bdr_)
            .uniform_refine(numRefine, 1);
        std::get<side::east - 1>(BoundaryCore::bdr_)
            .uniform_refine(numRefine, 1);
        std::get<side::south - 1>(BoundaryCore::bdr_)
            .uniform_refine(numRefine, 1);
        std::get<side::north - 1>(BoundaryCore::bdr_)
            .uniform_refine(numRefine, 1);
        std::get<side::stime - 1>(BoundaryCore::bdr_)
            .uniform_refine(numRefine, 2);
        std::get<side::etime - 1>(BoundaryCore::bdr_)
            .uniform_refine(numRefine, 2);
      } else
        throw std::runtime_error("Invalid dimension");
    } else if (dim == 3) {
      if constexpr (BoundaryCore::nsides() == 8) {
        std::get<side::west - 1>(BoundaryCore::bdr_)
            .uniform_refine(numRefine, 2);
        std::get<side::east - 1>(BoundaryCore::bdr_)
            .uniform_refine(numRefine, 2);
        std::get<side::south - 1>(BoundaryCore::bdr_)
            .uniform_refine(numRefine, 2);
        std::get<side::north - 1>(BoundaryCore::bdr_)
            .uniform_refine(numRefine, 2);
        std::get<side::front - 1>(BoundaryCore::bdr_)
            .uniform_refine(numRefine, 2);
        std::get<side::back - 1>(BoundaryCore::bdr_)
            .uniform_refine(numRefine, 2);
      } else
        throw std::runtime_error("Invalid dimension");
    } else
      throw std::runtime_error("Invalid dimension");
    return *this;
  }

private:
  /// @brief Writes the boundary spline object into a
  /// torch::serialize::OutputArchive object
  template <size_t... Is>
  inline torch::serialize::OutputArchive &
  write_(std::index_sequence<Is...>, torch::serialize::OutputArchive &archive,
         const std::string &key = "boundary") const {
    (std::get<Is>(BoundaryCore::bdr_)
         .write(archive, key + ".bdr[" + std::to_string(Is) + "]"),
     ...);
    return archive;
  }

public:
  /// @brief Saves the boundary spline to file
  inline void save(const std::string &filename,
                   const std::string &key = "boundary") const {
    torch::serialize::OutputArchive archive;
    write(archive, key).save_to(filename);
  }

  /// @brief Writes the boundary spline object into a
  /// torch::serialize::OutputArchive object
  inline torch::serialize::OutputArchive &
  write(torch::serialize::OutputArchive &archive,
        const std::string &key = "boundary") const {
    write_(std::make_index_sequence<BoundaryCore::nsides()>{}, archive, key);
    return archive;
  }

private:
  /// @brief Loads the function space object from a
  /// torch::serialize::InputArchive object
  template <size_t... Is>
  inline torch::serialize::InputArchive &
  read_(std::index_sequence<Is...>, torch::serialize::InputArchive &archive,
        const std::string &key = "boundary") {
    (std::get<Is>(BoundaryCore::bdr_)
         .read(archive, key + ".bdr[" + std::to_string(Is) + "]"),
     ...);
    return archive;
  }

public:
  /// @brief Loads the boundary spline object from file
  inline void load(const std::string &filename,
                   const std::string &key = "boundary") {
    torch::serialize::InputArchive archive;
    archive.load_from(filename);
    read(archive, key);
  }

  /// @brief Loads the boundary spline object from a
  /// torch::serialize::InputArchive object
  inline torch::serialize::InputArchive &
  read(torch::serialize::InputArchive &archive,
       const std::string &key = "boundary") {
    read_(std::make_index_sequence<BoundaryCore::nsides()>{}, archive, key);
    return archive;
  }

  /// @brief Returns the boundary object as XML object
  inline pugi::xml_document to_xml(int id = 0, std::string label = "",
                                   int index = -1) const {
    pugi::xml_document doc;
    pugi::xml_node root = doc.append_child("xml");
    to_xml(root, id, label, index);

    return doc;
  }

  /// @brief Returns the boundary object as XML node
  inline pugi::xml_node &to_xml(pugi::xml_node &root, int id = 0,
                                std::string label = "", int index = -1) const {
    // add Boundary node
    pugi::xml_node bdr = root.append_child("Boundary");

    if (id >= 0)
      bdr.append_attribute("id") = id;

    if (index >= 0)
      bdr.append_attribute("index") = index;

    if (!label.empty())
      bdr.append_attribute("label") = label.c_str();

    int index_ = 0;
    std::apply(
        [&bdr, &index_](const auto &...bspline) {
          (bspline.to_xml(bdr, -1, "", index_++), ...);
        },
        BoundaryCore::bdr_);

    return root;
  }

  /// @brief Updates the boundary object from XML object
  inline BoundaryCommon &from_xml(const pugi::xml_document &doc, int id = 0,
                                  std::string label = "", int index = -1) {
    return from_xml(doc.child("xml"), id, label, index);
  }

  /// @brief Updates the boundary object from XML node
  inline BoundaryCommon &from_xml(const pugi::xml_node &root, int id = 0,
                                  std::string label = "", int index = -1) {

    // Loop through all boundary nodes
    for (pugi::xml_node bdr : root.children("Boundary")) {

      // Check for "Boundary" with given id, index, label
      if ((id >= 0 ? bdr.attribute("id").as_int() == id : true) &&
          (index >= 0 ? bdr.attribute("index").as_int() == index : true) &&
          (!label.empty() ? bdr.attribute("label").value() == label : true)) {

        int index_ = 0;
        std::apply(
            [&bdr, &index_](auto &...bspline) {
              (bspline.from_xml(bdr, -1, "", index_++), ...);
            },
            BoundaryCore::bdr_);

        return *this;
      } else
        continue; // try next "Boundary"
    }

    throw std::runtime_error("XML object does not provide geometry with given "
                             "id, index, and/or label");
    return *this;
  }

private:
  /// @brief Returns true if both boundary spline objects are the
  /// same
  template <typename BoundaryCore_, size_t... Is>
  inline bool isequal_(std::index_sequence<Is...>,
                       const BoundaryCommon<BoundaryCore_> &other) const {
    return (
        (std::get<Is>(BoundaryCore::bdr_) == std::get<Is>(other.coeffs())) &&
        ...);
  }

public:
  /// @brief Returns true if both boundary objects are the same
  template <typename BoundaryCore_>
  inline bool operator==(const BoundaryCommon<BoundaryCore_> &other) const {
    return isequal_(std::make_index_sequence<BoundaryCore::nsides()>{}, other);
  }

  /// @brief Returns true if both boundary objects are different
  template <typename BoundaryCore_>
  inline bool operator!=(const BoundaryCommon<BoundaryCore_> &other) const {
    return !(
        *this ==
        other); // Do not change this to (*this != other) is it does not work
  }

private:
  /// @brief Returns true if both boundary spline objects are close up to the
  /// given tolerances
  template <typename BoundaryCore_, size_t... Is>
  inline bool
  isclose_(std::index_sequence<Is...>,
           const BoundaryCommon<BoundaryCore_> &other,
           typename BoundaryCore::spline_type::value_type rtol,
           typename BoundaryCore::spline_type::value_type atol) const {
    return ((std::get<Is>(BoundaryCore::bdr_)
                 .isclose(std::get<Is>(other.coeffs()))) &&
            ...);
  }

public:
  /// @brief Returns true if both boundary objects are close up to the given
  /// tolerances
  template <typename BoundaryCore_>
  inline bool
  isclose(const BoundaryCommon<BoundaryCore_> &other,
          typename BoundaryCore::spline_type::value_type rtol =
              typename BoundaryCore::spline_type::value_type{1e-5},
          typename BoundaryCore::spline_type::value_type atol =
              typename BoundaryCore::spline_type::value_type{1e-8}) const {
    return isclose_(std::make_index_sequence<BoundaryCore::nsides()>{}, other,
                    rtol, atol);
  }

#define GENERATE_EXPR_MACRO(r, data, name)                                     \
private:                                                                       \
  template <bool memory_optimized = false, size_t... Is, typename... Xi>       \
  inline auto BOOST_PP_CAT(name, _)(std::index_sequence<Is...>,                \
                                    const std::tuple<Xi...> &xi) const {       \
    return std::tuple(                                                         \
        std::get<Is>(BoundaryCore::bdr_)                                       \
            .template name<memory_optimized>(std::get<Is>(xi))...);            \
  }                                                                            \
                                                                               \
  template <bool memory_optimized = false, size_t... Is, typename... Xi,       \
            typename... Indices>                                               \
  inline auto BOOST_PP_CAT(name, _)(std::index_sequence<Is...>,                \
                                    const std::tuple<Xi...> &xi,               \
                                    const std::tuple<Indices...> &indices)     \
      const {                                                                  \
    return std::tuple(std::get<Is>(BoundaryCore::bdr_)                         \
                          .template name<memory_optimized>(                    \
                              std::get<Is>(xi), std::get<Is>(indices))...);    \
  }                                                                            \
                                                                               \
  template <bool memory_optimized = false, size_t... Is, typename... Xi,       \
            typename... Indices, typename... Coeff_Indices>                    \
  inline auto BOOST_PP_CAT(name, _)(                                           \
      std::index_sequence<Is...>, const std::tuple<Xi...> &xi,                 \
      const std::tuple<Indices...> &indices,                                   \
      const std::tuple<Coeff_Indices...> &coeff_indices) const {               \
    return std::tuple(std::get<Is>(BoundaryCore::bdr_)                         \
                          .template name<memory_optimized>(                    \
                              std::get<Is>(xi), std::get<Is>(indices),         \
                              std::get<Is>(coeff_indices))...);                \
  }                                                                            \
                                                                               \
public:                                                                        \
  template <bool memory_optimized = false, typename... Args>                   \
  inline auto name(const Args &...args) const {                                \
    return BOOST_PP_CAT(name, _)<memory_optimized>(                            \
        std::make_index_sequence<BoundaryCore::nsides()>{}, args...);          \
  }

  /// @brief Auto-generated functions
  /// @{
  BOOST_PP_SEQ_FOR_EACH(GENERATE_EXPR_MACRO, _, GENERATE_EXPR_SEQ)
  /// @}
#undef GENERATE_EXPR_MACRO

#define GENERATE_IEXPR_MACRO(r, data, name)                                    \
private:                                                                       \
  template <bool memory_optimized = false, size_t... Is, typename... Geometry, \
            typename... Xi>                                                    \
  inline auto BOOST_PP_CAT(name, _)(std::index_sequence<Is...>,                \
                                    const std::tuple<Geometry...> &G,          \
                                    const std::tuple<Xi...> &xi) const {       \
    return std::tuple(std::get<Is>(BoundaryCore::bdr_)                         \
                          .template name<memory_optimized>(                    \
                              std::get<Is>(G), std::get<Is>(xi))...);          \
  }                                                                            \
                                                                               \
  template <bool memory_optimized = false, size_t... Is, typename... Geometry, \
            typename... Xi, typename... Indices>                               \
  inline auto BOOST_PP_CAT(name, _)(                                           \
      std::index_sequence<Is...>, const std::tuple<Geometry...> &G,            \
      const std::tuple<Xi...> &xi, const std::tuple<Indices...> &indices)      \
      const {                                                                  \
    return std::tuple(                                                         \
        std::get<Is>(BoundaryCore::bdr_)                                       \
            .template name<memory_optimized>(                                  \
                std::get<Is>(G), std::get<Is>(xi), std::get<Is>(indices))...); \
  }                                                                            \
                                                                               \
  template <bool memory_optimized = false, size_t... Is, typename... Geometry, \
            typename... Xi, typename... Indices, typename... Coeff_Indices>    \
  inline auto BOOST_PP_CAT(name, _)(                                           \
      std::index_sequence<Is...>, const std::tuple<Geometry...> &G,            \
      const std::tuple<Xi...> &xi, const std::tuple<Indices...> &indices,      \
      const std::tuple<Coeff_Indices...> &coeff_indices) const {               \
    return std::tuple(std::get<Is>(BoundaryCore::bdr_)                         \
                          .template name<memory_optimized>(                    \
                              std::get<Is>(G), std::get<Is>(xi),               \
                              std::get<Is>(indices),                           \
                              std::get<Is>(coeff_indices))...);                \
  }                                                                            \
                                                                               \
public:                                                                        \
  template <bool memory_optimized = false, typename... Args>                   \
  inline auto name(const Args &...args) const {                                \
    return BOOST_PP_CAT(name, _)<memory_optimized>(                            \
        std::make_index_sequence<BoundaryCore::nsides()>{}, args...);          \
  }

  /// @brief Auto-generated functions
  /// @{
  BOOST_PP_SEQ_FOR_EACH(GENERATE_IEXPR_MACRO, _, GENERATE_IEXPR_SEQ)
  /// @}
#undef GENERATE_IEXPR_MACRO

  /// @brief Returns the `device` property of all splines
  auto device() const noexcept {
    return std::apply(
        [](const auto &...bspline) {
          return std::make_tuple(bspline.device()...);
        },
        BoundaryCore::bdr_);
  }

  /// @brief Returns the `device_index` property of all splines
  auto device_index() const noexcept {
    return std::apply(
        [](const auto &...bspline) {
          return std::make_tuple(bspline.device_index()...);
        },
        BoundaryCore::bdr_);
  }

  /// @brief Returns the `dtype` property of all splines
  auto dtype() const noexcept {
    return std::apply(
        [](const auto &...bspline) {
          return std::make_tuple(bspline.dtype()...);
        },
        BoundaryCore::bdr_);
  }

  /// @brief Returns the `layout` property of all splines
  auto layout() const noexcept {
    return std::apply(
        [](const auto &...bspline) {
          return std::make_tuple(bspline.layout()...);
        },
        BoundaryCore::bdr_);
  }

  /// @brief Returns the `requires_grad` property of all splines
  auto requires_grad() const noexcept {
    return std::apply(
        [](const auto &...bspline) {
          return std::make_tuple(bspline.requires_grad()...);
        },
        BoundaryCore::bdr_);
  }

  /// @brief Returns the `pinned_memory` property of all splines
  auto pinned_memory() const noexcept {
    return std::apply(
        [](const auto &...bspline) {
          return std::make_tuple(bspline.pinned_memory()...);
        },
        BoundaryCore::bdr_);
  }

  /// @brief Returns if the layout is sparse of all splines
  auto is_sparse() const noexcept {
    return std::apply(
        [](const auto &...bspline) {
          return std::make_tuple(bspline.is_sparse()...);
        },
        BoundaryCore::bdr_);
  }

  /// @brief Returns true if the B-spline is uniform of all splines
  auto is_uniform() const noexcept {
    return std::apply(
        [](const auto &...bspline) {
          return std::make_tuple(bspline.is_uniform()...);
        },
        BoundaryCore::bdr_);
  }

  /// @brief Returns true if the B-spline is non-uniform if all splines
  auto is_nonuniform() const noexcept {
    return std::apply(
        [](const auto &...bspline) {
          return std::make_tuple(bspline.is_nonuniform()...);
        },
        BoundaryCore::bdr_);
  }

  /// @brief Sets the boundary object's `requires_grad` property
  BoundaryCommon &set_requires_grad(bool requires_grad) {
    std::apply(
        [requires_grad](const auto &...bspline) {
          (bspline.set_requires_grad(requires_grad), ...);
        },
        BoundaryCore::bdr_);

    return *this;
  }

  /// @brief Returns a copy of the boundary object with settings from options
  template <typename real_t> inline auto to(Options<real_t> options) const {
    using boundary_type = BoundaryCommon<iganet::BoundaryCore<
        decltype(typename BoundaryCore::spline_type{}.template to(options)),
        BoundaryCore::spline_type::parDim()>>;

    return boundary_type(std::apply(
        [&options](const auto &...bspline) {
          return std::make_tuple(bspline.template to(options)...);
        },
        BoundaryCore::bdr_));
  }

  /// @brief Returns a copy of the boundary object with settings from device
  inline auto to(torch::Device device) const {
    return BoundaryCommon(std::apply(
        [&device](const auto &...bspline) {
          return std::make_tuple(bspline.to(device)...);
        },
        BoundaryCore::bdr_));
  }

  /// @brief Returns a copy of the boundary object with real_t type
  template <typename real_t> inline auto to() const {
    using boundary_type = BoundaryCommon<iganet::BoundaryCore<
        decltype(typename BoundaryCore::spline_type{}.template to<real_t>()),
        BoundaryCore::spline_type::parDim()>>;

    return boundary_type(std::apply(
        [](const auto &...bspline) {
          return std::make_tuple(bspline.template to<real_t>()...);
        },
        BoundaryCore::bdr_));
  }
};

/// @brief Boundary
template <typename Spline>
using Boundary = BoundaryCommon<BoundaryCore<Spline, Spline::parDim()>>;

/// @brief Print (as string) a Boundary object
template <typename Spline>
inline std::ostream &operator<<(std::ostream &os, const Boundary<Spline> &obj) {
  obj.pretty_print(os);
  return os;
}

} // namespace iganet
