/**
   @file include/functionspace.hpp

   @brief Function spaces

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <boost/preprocessor/seq/for_each.hpp>

#include <bspline.hpp>
#include <utils/container.hpp>
#include <utils/type_traits.hpp>
#include <utils/zip.hpp>

namespace iganet {

using namespace literals;
using utils::operator+;

/// @brief Enumerator for the function space component
enum class functionspace : short_t {
  interior = 0, /*!< interior component */
  boundary = 1  /*!< boundary component */
};

/// @brief Macro: Implements the default methods of a function space
#define IGANET_FUNCTIONSPACE_DEFAULT_OPS(FunctionSpace)                        \
  FunctionSpace() = default;                                                   \
  FunctionSpace(FunctionSpace &&) = default;                                   \
  FunctionSpace(const FunctionSpace &) = default;

namespace detail {

// Forward declaration
template <typename, typename> class FunctionSpace;

/// @brief Tensor-product function space
///
/// @note This class is not meant for direct use in
/// applications. Instead use S, TH, NE, or RT.
template <typename... Splines, typename... Boundaries>
class FunctionSpace<std::tuple<Splines...>, std::tuple<Boundaries...>>
    : public utils::Serializable, private utils::FullQualifiedName {

  static_assert(is_SplineType_v<Splines...>,
                "Splines must be valid SplineTypes");
  static_assert(is_BoundaryType_v<Boundaries...>,
                "Boundaries must be valid BoundaryTypes");

public:
  /// @brief Value type
  using value_type = std::common_type_t<typename Splines::value_type...>;

  /// @brief Spline type
  using spline_type = std::tuple<Splines...>;

  /// @brief Spline evaluation type
  using eval_type = std::tuple<utils::TensorArray<Splines::parDim()>...>;

  /// @brief Boundary type
  using boundary_type = std::tuple<Boundaries...>;

  /// @brief Boundary evaluation type
  using boundary_eval_type = std::tuple<typename Boundaries::eval_type...>;

protected:
  /// @brief Splines
  spline_type spline_;

  /// @brief Boundaries
  boundary_type boundary_;

public:
  /// @brief Default constructor
  FunctionSpace() = default;

  /// @brief Copy constructor
  FunctionSpace(const FunctionSpace &) = default;

  /// @brief Move constructor
  FunctionSpace(FunctionSpace &&) = default;

  /// @brief Constructor
  /// @{
  FunctionSpace(const std::array<int64_t, Splines::parDim()> &...ncoeffs,
                enum init init = init::greville,
                Options<value_type> options = iganet::Options<value_type>{})
      : spline_({ncoeffs, init, options}...),
        boundary_({ncoeffs, init::none, options}...) {
    boundary_from_full_tensor(this->as_tensor());
  }

  FunctionSpace(const std::array<std::vector<typename Splines::value_type>,
                                 Splines::parDim()> &...kv,
                enum init init = init::greville,
                Options<value_type> options = iganet::Options<value_type>{})
      : spline_({kv, init, options}...),
        boundary_({kv, init::none, options}...) {

    static_assert((Splines::is_nonuniform() && ... && true),
                  "Constructor is only available for non-uniform splines");
    boundary_from_full_tensor(this->as_tensor());
  }

  explicit FunctionSpace(const std::tuple<Splines...> &splines)
      : spline_(splines) { //,
    //        boundary_({splines.ncoeffs(), init::none, splines.options()}...) {
    boundary_from_full_tensor(this->as_tensor());
  }

  explicit FunctionSpace(std::tuple<Splines...> &&splines)
      : spline_(splines) { //,
    //        boundary_({splines.ncoeffs(), init::none, splines.options()}...) {
    boundary_from_full_tensor(this->as_tensor());
  }
  /// @}

  /// @brief Returns the number of function spaces
  inline static constexpr short_t nspaces() noexcept {
    return sizeof...(Splines);
  }

  /// @brief Returns the number of boundaries
  inline static constexpr short_t nboundaries() noexcept {
    return sizeof...(Boundaries);
  }

  /// @brief Returns a constant reference to the \f$s\f$-th function space
  template <short_t s> inline const auto &space() const noexcept {
    static_assert(s >= 0 && s < nspaces());
    return std::get<s>(spline_);
  }

  /// @brief Returns a non-constant reference to the \f$s\f$-th space
  template <short_t s> inline auto &space() noexcept {
    static_assert(s >= 0 && s < nspaces());
    return std::get<s>(spline_);
  }

  /// @brief Returns a constant reference to the \f$s\f$-th boundary object
  template <short_t s> inline const auto &boundary() const noexcept {
    static_assert(s >= 0 && s < nboundaries());
    return std::get<s>(boundary_);
  }

  /// @brief Returns a non-constant reference to the \f$s\f$-th
  /// boundary object
  template <short_t s> inline auto &boundary() noexcept {
    static_assert(s >= 0 && s < nboundaries());
    return std::get<s>(boundary_);
  }

  /// @brief Returns a clone of the function space
  inline FunctionSpace clone() const noexcept { return FunctionSpace(*this); }

  /// @brief Returns a clone of a subset of the function space
  template <short_t... s> inline auto clone() const noexcept {

    static_assert(((s >= 0 && s < nspaces()) && ... && true));

    return FunctionSpace<std::tuple<std::tuple_element_t<s, spline_type>...>,
                         std::tuple<std::tuple_element_t<s, boundary_type>...>>(
        std::get<s>(spline_)..., std::get<s>(boundary_)...);
  }

private:
  /// @brief Returns a single-tensor representation of the
  /// tuple of spaces
  template <std::size_t... Is>
  inline torch::Tensor
  spaces_as_tensor_(std::index_sequence<Is...>) const noexcept {
    return torch::cat({std::get<Is>(spline_).as_tensor()...});
  }

public:
  /// @brief Returns a single-tensor representation of the
  /// tuple of spaces
  virtual inline torch::Tensor spaces_as_tensor() const noexcept {
    return spaces_as_tensor_(
        std::make_index_sequence<FunctionSpace::nspaces()>{});
  }

private:
  /// @brief Returns a single-tensor representation of the
  /// tuple of boundaries
  template <std::size_t... Is>
  inline torch::Tensor
  boundary_as_tensor_(std::index_sequence<Is...>) const noexcept {
    return torch::cat({std::get<Is>(boundary_).as_tensor()...});
  }

public:
  /// @brief Returns a single-tensor representation of the
  /// tuple of boundaries
  virtual inline torch::Tensor boundary_as_tensor() const noexcept {
    return boundary_as_tensor_(
        std::make_index_sequence<FunctionSpace::nboundaries()>{});
  }

  /// @brief Returns a single-tensor representation of the
  /// function space object
  ///
  /// @note The default implementation behaves identical to
  /// spaces_as_tensor() but can be overridden in a derived class
  virtual inline torch::Tensor as_tensor() const noexcept {
    return spaces_as_tensor();
  }

private:
  /// @brief Returns the size of the single-tensor representation of
  /// the tuple of function spaces
  template <std::size_t... Is>
  inline int64_t
  spaces_as_tensor_size_(std::index_sequence<Is...>) const noexcept {
    return std::apply(
        [](auto... v) { return (v + ...); },
        std::make_tuple(std::get<Is>(spline_).as_tensor_size()...));
  }

public:
  /// @brief Returns the size of the single-tensor representation of
  /// the tuple of function spaces
  virtual inline int64_t spaces_as_tensor_size() const noexcept {
    return spaces_as_tensor_size_(
        std::make_index_sequence<FunctionSpace::nspaces()>{});
  }

private:
  /// @brief Returns the size of the single-tensor representation of
  /// the tuple of boundaries
  template <std::size_t... Is>
  inline int64_t
  boundary_as_tensor_size_(std::index_sequence<Is...>) const noexcept {
    return std::apply(
        [](auto... v) { return (v + ...); },
        std::make_tuple(std::get<Is>(boundary_).as_tensor_size()...));
  }

public:
  /// @brief Returns the size of the single-tensor representation of
  /// the tuple of boundaries
  virtual inline int64_t boundary_as_tensor_size() const noexcept {
    return boundary_as_tensor_size_(
        std::make_index_sequence<FunctionSpace::nboundaries()>{});
  }

  /// @brief Returns the size of the single-tensor representation of
  /// the function space object
  ///
  /// @note The default implementation behaves identical to
  /// spaces_as_tensor_size() but can be overridden in a derived class
  virtual inline int64_t as_tensor_size() const noexcept {
    return spaces_as_tensor_size();
  }

private:
  /// @brief Sets the tuple of spaces from a single-tensor representation
  template <std::size_t... Is>
  inline FunctionSpace &spaces_from_tensor_(std::index_sequence<Is...>,
                                            const torch::Tensor &tensor) {

    // Compute the partial sums of all function spaces
    std::array<int64_t, sizeof...(Is)> partialSums{0};
    auto partial_sums = [&partialSums,
                         this]<std::size_t... Js>(std::index_sequence<Js...>) {
      ((std::get<Js + 1>(partialSums) =
            std::get<Js>(partialSums) + std::get<Js>(spline_).as_tensor_size()),
       ...);
    };
    partial_sums(std::make_index_sequence<FunctionSpace::nspaces() - 1>{});

    // Call from_tensor for all function spaces
    ((std::get<Is>(spline_).from_tensor(tensor.index(
         {torch::indexing::Slice(partialSums[Is],
                                 partialSums[Is] +
                                     std::get<Is>(spline_).as_tensor_size()),
          "..."}))),
     ...);

    return *this;
  }

public:
  /// @brief Sets the tuple of spaces from a single-tensor representation
  virtual inline FunctionSpace &
  spaces_from_tensor(const torch::Tensor &tensor) {
    return spaces_from_tensor_(
        std::make_index_sequence<FunctionSpace::nspaces()>{}, tensor);
  }

private:
  /// @brief Sets the tuple of boundaries from a single-tensor representation of
  /// the boundaries only
  template <std::size_t... Is>
  inline FunctionSpace &boundary_from_tensor_(std::index_sequence<Is...>,
                                              const torch::Tensor &tensor) {
    (std::get<Is>(boundary_).from_tensor(std::get<Is>(spline_).as_tensor()),
     ...);

    return *this;
  }

public:
  /// @brief Sets the tuple of boundaries from a single-tensor representation of
  /// the boundaries only
  virtual inline FunctionSpace &
  boundary_from_tensor(const torch::Tensor &tensor) {
    return boundary_from_tensor_(
        std::make_index_sequence<FunctionSpace::nboundaries()>{}, tensor);
  }

private:
  /// @brief Sets the tuple of boundaries from a single-tensor representation
  template <std::size_t... Is>
  inline FunctionSpace &
  boundary_from_full_tensor_(std::index_sequence<Is...>,
                             const torch::Tensor &tensor) {
    (std::get<Is>(boundary_).from_full_tensor(
         std::get<Is>(spline_).as_tensor()),
     ...);

    return *this;
  }

public:
  /// @brief Sets the tuple of boundaries from a single-tensor representation
  virtual inline FunctionSpace &
  boundary_from_full_tensor(const torch::Tensor &tensor) {
    return boundary_from_full_tensor_(
        std::make_index_sequence<FunctionSpace::nboundaries()>{}, tensor);
  }

  /// @brief Sets the function space object from a single-tensor representation
  virtual inline FunctionSpace &from_tensor(const torch::Tensor &tensor) {
    spaces_from_tensor_(std::make_index_sequence<FunctionSpace::nspaces()>{},
                        tensor);
    boundary_from_full_tensor_(
        std::make_index_sequence<FunctionSpace::nboundaries()>{}, tensor);
    return *this;
  }

private:
  /// @brief Returns the function space object as XML node
  template <std::size_t... Is>
  inline pugi::xml_node &to_xml_(std::index_sequence<Is...>,
                                 pugi::xml_node &root, int id = 0,
                                 std::string label = "") const {

    (std::get<Is>(spline_).to_xml(root, id, label, Is), ...);
    return root;
  }

public:
  /// @brief Returns the function space object as XML object
  inline pugi::xml_document to_xml(int id = 0, std::string label = "") const {
    pugi::xml_document doc;
    pugi::xml_node root = doc.append_child("xml");
    to_xml(root, id, label);

    return doc;
  }

  /// @brief Returns the function space object as XML node
  inline pugi::xml_node &to_xml(pugi::xml_node &root, int id = 0,
                                std::string label = "") const {
    return to_xml_(std::make_index_sequence<FunctionSpace::nspaces()>{}, root,
                   id, label);
  }

private:
  /// @brief Updates the function space object from XML object
  template <std::size_t... Is>
  inline FunctionSpace &from_xml_(std::index_sequence<Is...>,
                                  const pugi::xml_node &root, int id = 0,
                                  std::string label = "") {

    (std::get<Is>(spline_).from_xml(root, id, label, Is), ...);
    return *this;
  }

public:
  /// @brief Updates the function space object from XML object
  inline FunctionSpace &from_xml(const pugi::xml_document &doc, int id = 0,
                                 std::string label = "") {
    return from_xml(doc.child("xml"), id, label);
  }

  /// @brief Updates the function space object from XML node
  inline FunctionSpace &from_xml(const pugi::xml_node &root, int id = 0,
                                 std::string label = "") {
    return from_xml_(std::make_index_sequence<FunctionSpace::nspaces()>{}, root,
                     id, label);
  }

private:
  /// @brief Serialization to JSON
  template <std::size_t... Is>
  nlohmann::json to_json_(std::index_sequence<Is...>) const {
    auto json_this = nlohmann::json::array();
    auto json_boundary = nlohmann::json::array();
    (json_this.push_back(std::get<Is>(spline_).to_json()), ...);
    (json_boundary.push_back(std::get<Is>(boundary_).to_json()), ...);

    auto json = nlohmann::json::array();
    for (auto [t, b] : utils::zip(json_this, json_boundary)) {
      auto json_inner = nlohmann::json::array();
      json_inner.push_back(t);
      json_inner.push_back(b);
      json.push_back(json_inner);
    }

    return json;
  }

public:
  /// @brief Serialization to JSON
  nlohmann::json to_json() const override {
    return to_json_(std::make_index_sequence<FunctionSpace::nspaces()>{});
  }

private:
  /// @brief Returns the values of the spline objects in the points `xi`
  /// @{
  template <functionspace comp = functionspace::interior,
            deriv deriv = deriv::func, bool memory_optimized = false,
            std::size_t... Is, typename... Xi>
  inline auto eval_(std::index_sequence<Is...>,
                    const std::tuple<Xi...> &xi) const {
    if constexpr (comp == functionspace::interior)
      return std::tuple(
          std::get<Is>(spline_).template eval<deriv, memory_optimized>(
              std::get<Is>(xi))...);
    else if constexpr (comp == functionspace::boundary)
      return std::tuple(
          std::get<Is>(boundary_).template eval<deriv, memory_optimized>(
              std::get<Is>(xi))...);
  }

  template <functionspace comp = functionspace::interior,
            deriv deriv = deriv::func, bool memory_optimized = false,
            std::size_t... Is, typename... Xi, typename... Knot_Indices>
  inline auto eval_(std::index_sequence<Is...>, const std::tuple<Xi...> &xi,
                    const std::tuple<Knot_Indices...> &knot_indices) const {
    if constexpr (comp == functionspace::interior)
      return std::tuple(
          std::get<Is>(spline_).template eval<deriv, memory_optimized>(
              std::get<Is>(xi), std::get<Is>(knot_indices))...);
    else if constexpr (comp == functionspace::boundary)
      return std::tuple(
          std::get<Is>(boundary_).template eval<deriv, memory_optimized>(
              std::get<Is>(xi), std::get<Is>(knot_indices))...);
  }

  template <functionspace comp = functionspace::interior,
            deriv deriv = deriv::func, bool memory_optimized = false,
            std::size_t... Is, typename... Xi, typename... Knot_Indices,
            typename... Coeff_Indices>
  inline auto eval_(std::index_sequence<Is...>, const std::tuple<Xi...> &xi,
                    const std::tuple<Knot_Indices...> &knot_indices,
                    const std::tuple<Coeff_Indices...> &coeff_indices) const {
    if constexpr (comp == functionspace::interior)
      return std::tuple(
          std::get<Is>(spline_).template eval<deriv, memory_optimized>(
              std::get<Is>(xi), std::get<Is>(knot_indices),
              std::get<Is>(coeff_indices))...);
    else if constexpr (comp == functionspace::boundary)
      return std::tuple(
          std::get<Is>(boundary_).template eval<deriv, memory_optimized>(
              std::get<Is>(xi), std::get<Is>(knot_indices),
              std::get<Is>(coeff_indices))...);
  }
  /// @}

public:
  /// @brief Returns the values of the spline objects in the points `xi`
  /// @{
  template <functionspace comp = functionspace::interior,
            deriv deriv = deriv::func, bool memory_optimized = false,
            typename... Xi>
  inline auto eval(const std::tuple<Xi...> &xi) const {
    static_assert(FunctionSpace::nspaces() == sizeof...(Xi),
                  "Size of Xi mismatches functionspace dimension");
    return eval_<comp, deriv, memory_optimized>(
        std::make_index_sequence<FunctionSpace::nspaces()>{}, xi);
  }

  template <functionspace comp = functionspace::interior,
            deriv deriv = deriv::func, bool memory_optimized = false,
            typename... Xi, typename... Knot_Indices>
  inline auto eval(const std::tuple<Xi...> &xi,
                   const std::tuple<Knot_Indices...> &knot_indices) const {
    static_assert(
        (FunctionSpace::nspaces() == sizeof...(Xi)) &&
            (FunctionSpace::nspaces() == sizeof...(Knot_Indices)),
        "Sizes of Xi and Knot_Indices mismatch functionspace dimension");
    return eval_<comp, deriv, memory_optimized>(
        std::make_index_sequence<FunctionSpace::nspaces()>{}, xi, knot_indices);
  }

  template <functionspace comp = functionspace::interior,
            deriv deriv = deriv::func, bool memory_optimized = false,
            typename... Xi, typename... Knot_Indices, typename... Coeff_Indices>
  inline auto eval(const std::tuple<Xi...> &xi,
                   const std::tuple<Knot_Indices...> &knot_indices,
                   const std::tuple<Coeff_Indices...> &coeff_indices) const {
    static_assert((FunctionSpace::nspaces() == sizeof...(Xi)) &&
                      (FunctionSpace::nspaces() == sizeof...(Knot_Indices)) &&
                      (FunctionSpace::nspaces() == sizeof...(Coeff_Indices)),
                  "Sizes of Xi, Knot_Indices and Coeff_Indices mismatch "
                  "functionspace dimension");
    return eval_<comp, deriv, memory_optimized>(
        std::make_index_sequence<FunctionSpace::nspaces()>{}, xi, knot_indices,
        coeff_indices);
  }
  /// @}

private:
  /// @brief Returns the value of the spline objects from
  /// precomputed basis function
  /// @{
  template <functionspace comp = functionspace::interior, std::size_t... Is,
            typename... Basfunc, typename... Coeff_Indices, typename... Numeval,
            typename... Sizes>
  inline auto
  eval_from_precomputed_(std::index_sequence<Is...>,
                         const std::tuple<Basfunc...> &basfunc,
                         const std::tuple<Coeff_Indices...> &coeff_indices,
                         const std::tuple<Numeval...> &numeval,
                         const std::tuple<Sizes...> &sizes) const {
    if constexpr (comp == functionspace::interior)
      return std::tuple(std::get<Is>(spline_).eval_from_precomputed(
          std::get<Is>(basfunc), std::get<Is>(coeff_indices),
          std::get<Is>(numeval), std::get<Is>(sizes))...);
    else if constexpr (comp == functionspace::boundary)
      return std::tuple(std::get<Is>(boundary_).eval_from_precomputed(
          std::get<Is>(basfunc), std::get<Is>(coeff_indices),
          std::get<Is>(numeval), std::get<Is>(sizes))...);
  }

  template <functionspace comp = functionspace::interior, std::size_t... Is,
            typename... Basfunc, typename... Coeff_Indices, typename... Xi>
  inline auto
  eval_from_precomputed_(std::index_sequence<Is...>,
                         const std::tuple<Basfunc...> &basfunc,
                         const std::tuple<Coeff_Indices...> &coeff_indices,
                         const std::tuple<Xi...> &xi) const {
    if constexpr (comp == functionspace::interior)
      return std::tuple(std::get<Is>(spline_).eval_from_precomputed(
          std::get<Is>(basfunc), std::get<Is>(coeff_indices),
          std::get<Is>(xi)[0].numel(), std::get<Is>(xi)[0].sizes())...);
    else if constexpr (comp == functionspace::boundary)
      return std::tuple(std::get<Is>(boundary_).eval_from_precomputed(
          std::get<Is>(basfunc), std::get<Is>(coeff_indices),
          std::get<Is>(xi))...);
  }
  /// @}

public:
  /// @brief Returns the value of the spline objects from
  /// precomputed basis function
  /// @{
  template <functionspace comp = functionspace::interior, typename... Basfunc,
            typename... Coeff_Indices, typename... Numeval, typename... Sizes>
  inline auto
  eval_from_precomputed(const std::tuple<Basfunc...> &basfunc,
                        const std::tuple<Coeff_Indices...> &coeff_indices,
                        const std::tuple<Numeval...> &numeval,
                        const std::tuple<Sizes...> &sizes) const {
    return eval_from_precomputed_<comp>(
        std::make_index_sequence<FunctionSpace::nspaces()>{}, basfunc,
        coeff_indices, numeval, sizes);
  }

  template <functionspace comp = functionspace::interior, typename... Basfunc,
            typename... Coeff_Indices, typename... Xi>
  inline auto
  eval_from_precomputed(const std::tuple<Basfunc...> &basfunc,
                        const std::tuple<Coeff_Indices...> &coeff_indices,
                        const std::tuple<Xi...> &xi) const {
    return eval_from_precomputed_<comp>(
        std::make_index_sequence<FunctionSpace::nspaces()>{}, basfunc,
        coeff_indices, xi);
  }
  /// @}

private:
  /// @brief Returns the knot indicies of knot spans containing `xi`
  template <functionspace comp = functionspace::interior, std::size_t... Is,
            typename... Xi>
  inline auto find_knot_indices_(std::index_sequence<Is...>,
                                 const std::tuple<Xi...> &xi) const {
    if constexpr (comp == functionspace::interior)
      return std::tuple(
          std::get<Is>(spline_).find_knot_indices(std::get<Is>(xi))...);
    else if constexpr (comp == functionspace::boundary)
      return std::tuple(
          std::get<Is>(boundary_).find_knot_indices(std::get<Is>(xi))...);
  }

public:
  /// @brief Returns the knot indicies of knot spans containing `xi`
  template <functionspace comp = functionspace::interior, typename... Xi>
  inline auto find_knot_indices(const std::tuple<Xi...> &xi) const {
    return find_knot_indices_<comp>(
        std::make_index_sequence<FunctionSpace::nspaces()>{}, xi);
  }

private:
  /// @brief Returns the values of the spline objects' basis functions in the
  /// points `xi`
  /// @{
  template <functionspace comp = functionspace::interior,
            deriv deriv = deriv::func, bool memory_optimized = false,
            std::size_t... Is, typename... Xi>
  inline auto eval_basfunc_(std::index_sequence<Is...>,
                            const std::tuple<Xi...> &xi) const {
    if constexpr (comp == functionspace::interior)
      return std::tuple(
          std::get<Is>(spline_).template eval_basfunc<deriv, memory_optimized>(
              std::get<Is>(xi))...);
    else if constexpr (comp == functionspace::boundary)
      return std::tuple(std::get<Is>(boundary_)
                            .template eval_basfunc<deriv, memory_optimized>(
                                std::get<Is>(xi))...);
  }

  template <functionspace comp = functionspace::interior,
            deriv deriv = deriv::func, bool memory_optimized = false,
            std::size_t... Is, typename... Xi, typename... Knot_Indices>
  inline auto
  eval_basfunc_(std::index_sequence<Is...>, const std::tuple<Xi...> &xi,
                const std::tuple<Knot_Indices...> &knot_indices) const {
    if constexpr (comp == functionspace::interior)
      return std::tuple(
          std::get<Is>(spline_).template eval_basfunc<deriv, memory_optimized>(
              std::get<Is>(xi), std::get<Is>(knot_indices))...);
    else if constexpr (comp == functionspace::boundary)
      return std::tuple(
          std::get<Is>(boundary_)
              .template eval_basfunc<deriv, memory_optimized>(
                  std::get<Is>(xi), std::get<Is>(knot_indices))...);
  }
  /// @}

public:
  /// @brief Returns the values of the spline objects' basis
  /// functions in the points `xi` @{
  template <functionspace comp = functionspace::interior,
            deriv deriv = deriv::func, bool memory_optimized = false,
            typename... Xi>
  inline auto eval_basfunc(const std::tuple<Xi...> &xi) const {
    return eval_basfunc_<comp, deriv, memory_optimized>(
        std::make_index_sequence<FunctionSpace::nspaces()>{}, xi);
  }

  template <functionspace comp = functionspace::interior,
            deriv deriv = deriv::func, bool memory_optimized = false,
            typename... Xi, typename... Knot_Indices>
  inline auto
  eval_basfunc(const std::tuple<Xi...> &xi,
               const std::tuple<Knot_Indices...> &knot_indices) const {
    return eval_basfunc_<comp, deriv, memory_optimized>(
        std::make_index_sequence<FunctionSpace::nspaces()>{}, xi, knot_indices);
  }
  /// @}

private:
  /// @brief Returns the indices of the spline objects'
  /// coefficients corresponding to the knot indices `indices`
  template <functionspace comp = functionspace::interior,
            bool memory_optimized = false, std::size_t... Is,
            typename... Knot_Indices>
  inline auto
  find_coeff_indices_(std::index_sequence<Is...>,
                      const std::tuple<Knot_Indices...> &knot_indices) const {
    if constexpr (comp == functionspace::interior)
      return std::tuple(
          std::get<Is>(spline_).template find_coeff_indices<memory_optimized>(
              std::get<Is>(knot_indices))...);
    else if constexpr (comp == functionspace::boundary)
      return std::tuple(
          std::get<Is>(boundary_).template find_coeff_indices<memory_optimized>(
              std::get<Is>(knot_indices))...);
  }

public:
  /// @brief Returns the indices of the spline objects'
  /// coefficients corresponding to the knot indices `indices`
  template <functionspace comp = functionspace::interior,
            bool memory_optimized = false, typename... Knot_Indices>
  inline auto
  find_coeff_indices(const std::tuple<Knot_Indices...> &knot_indices) const {
    return find_coeff_indices_<comp, memory_optimized>(
        std::make_index_sequence<FunctionSpace::nspaces()>{}, knot_indices);
  }

private:
  /// @brief Returns the spline objects with uniformly refined
  /// knot and coefficient vectors
  template <std::size_t... Is, std::size_t... Js>
  inline auto &uniform_refine_(std::index_sequence<Is...>,
                               std::index_sequence<Js...>, int numRefine = 1,
                               int dimRefine = -1) {
    (std::get<Is>(spline_).uniform_refine(numRefine, dimRefine), ...);
    (std::get<Js>(boundary_).uniform_refine(numRefine, dimRefine), ...);
    return *this;
  }

public:
  /// @brief Returns the spline objects with uniformly refined
  /// knot and coefficient vectors
  inline auto &uniform_refine(int numRefine = 1, int dimRefine = -1) {
    return uniform_refine_(
        std::make_index_sequence<FunctionSpace::nspaces()>{},
        std::make_index_sequence<FunctionSpace::nboundaries()>{}, numRefine,
        dimRefine);
  }

private:
  /// @brief Returns a copy of the function space object with settings from
  /// options
  template <typename real_t, std::size_t... Is, std::size_t... Js>
  inline auto to_(std::index_sequence<Is...>, std::index_sequence<Js...>,
                  Options<real_t> options) const {
    return FunctionSpace<
        typename Splines::template real_derived_self_type<real_t>...,
        typename Boundaries::template real_derived_self_type<real_t>...>(
        std::get<Is>(spline_).to(options)...,
        std::get<Js>(boundary_).to(options)...);
  }

public:
  /// @brief Returns a copy of the function space object with settings from
  /// options
  template <typename real_t> inline auto to(Options<real_t> options) const {
    return to_(std::make_index_sequence<FunctionSpace::nspaces()>{},
               std::make_index_sequence<FunctionSpace::nboundaries()>{},
               options);
  }

private:
  /// @brief Returns a copy of the function space object with settings from
  /// device
  template <std::size_t... Is, std::size_t... Js>
  inline auto to_(std::index_sequence<Is...>, std::index_sequence<Js...>,
                  torch::Device device) const {
    return FunctionSpace(std::get<Is>(spline_).to(device)...,
                         std::get<Js>(boundary_).to(device)...);
  }

public:
  /// @brief Returns a copy of the function space object with settings from
  /// device
  inline auto to(torch::Device device) const {
    return to_(std::make_index_sequence<FunctionSpace::nspaces()>{},
               std::make_index_sequence<FunctionSpace::nboundaries()>{},
               device);
  }

private:
  /// @brief Returns a copy of the function space object with real_t type
  template <typename real_t, std::size_t... Is, std::size_t... Js>
  inline auto to_(std::index_sequence<Is...>,
                  std::index_sequence<Js...>) const {
    return FunctionSpace<
        typename Splines::template real_derived_self_type<real_t>...,
        typename Boundaries::template real_derived_self_type<real_t>...>(
        std::get<Is>(spline_).template to<real_t>()...,
        std::get<Js>(boundary_).template to<real_t>()...);
  }

public:
  /// @brief Returns a copy of the function space object with real_t type
  template <typename real_t> inline auto to() const {
    return to_<real_t>(
        std::make_index_sequence<FunctionSpace::nspaces()>{},
        std::make_index_sequence<FunctionSpace::nboundaries()>{});
  }

private:
  /// @brief Scales the function space object by a scalar
  template <std::size_t... Is>
  inline auto scale_(std::index_sequence<Is...>, value_type s, int dim = -1) {
    (std::get<Is>(spline_).scale(s, dim), ...);
    boundary_from_full_tensor(this->as_tensor());
    return *this;
  }

public:
  /// @brief Scales the function space object by a scalar
  inline auto scale(value_type s, int dim = -1) {
    return scale_(std::make_index_sequence<FunctionSpace::nspaces()>{}, s, dim);
  }

private:
  /// @brief Scales the function space object by a vector
  template <std::size_t N, std::size_t... Is>
  inline auto scale_(std::index_sequence<Is...>, std::array<value_type, N> v) {
    (std::get<Is>(spline_).scale(v), ...);
    (std::get<Is>(boundary_).from_full_tensor(
         std::get<Is>(spline_).as_tensor()),
     ...);
    return *this;
  }

public:
  /// @brief Scales the function space object by a vector
  template <size_t N> inline auto scale(std::array<value_type, N> v) {
    return scale_(std::make_index_sequence<FunctionSpace::nspaces()>{}, v);
  }

private:
  /// @brief Translates the function space object by a vector
  template <std::size_t N, std::size_t... Is>
  inline auto translate_(std::index_sequence<Is...>,
                         std::array<value_type, N> v) {
    (std::get<Is>(spline_).translate(v), ...);
    (std::get<Is>(boundary_).from_full_tensor(
         std::get<Is>(spline_).as_tensor()),
     ...);
    return *this;
  }

public:
  /// @brief Translates the function space object by a vector
  template <size_t N> inline auto translate(std::array<value_type, N> v) {
    return translate_(std::make_index_sequence<FunctionSpace::nspaces()>{}, v);
  }

private:
  /// @brief Rotates the function space object by an angle in 2d
  template <std::size_t... Is>
  inline auto rotate_(std::index_sequence<Is...>, value_type angle) {
    (std::get<Is>(spline_).rotate(angle), ...);
    (std::get<Is>(boundary_).from_full_tensor(
         std::get<Is>(spline_).as_tensor()),
     ...);
    return *this;
  }

public:
  /// @brief Rotates the function space object by an angle in 2d
  inline auto rotate(value_type angle) {
    return rotate_(std::make_index_sequence<FunctionSpace::nspaces()>{}, angle);
  }

private:
  /// @brief Rotates the function space object by three angles in 3d
  template <std::size_t... Is>
  inline auto rotate_(std::index_sequence<Is...>,
                      std::array<value_type, 3> angle) {
    (std::get<Is>(spline_).rotate(angle), ...);
    (std::get<Is>(boundary_).from_full_tensor(
         std::get<Is>(spline_).as_tensor()),
     ...);
    return *this;
  }

public:
  /// @brief Rotates the function space object by three angles in 3d
  inline auto rotate(std::array<value_type, 3> angle) {
    return rotate_(std::make_index_sequence<FunctionSpace::nspaces()>{}, angle);
  }

private:
  /// @brief Computes the bounding boxes of the function space object
  template <std::size_t... Is>
  inline auto boundingBox_(std::index_sequence<Is...>) const {
    return std::tuple(std::get<Is>(spline_).boundingBox()...);
  }

public:
  /// @brief Computes the bounding boxes of the function space object
  inline auto boundingBox() const {
    return boundingBox_(std::make_index_sequence<FunctionSpace::nspaces()>{});
  }

private:
  /// @brief Writes the function space object into a
  /// torch::serialize::OutputArchive object
  template <std::size_t... Is>
  inline torch::serialize::OutputArchive &
  write_(std::index_sequence<Is...>, torch::serialize::OutputArchive &archive,
         const std::string &key = "functionspace") const {
    (std::get<Is>(spline_).write(
         archive, key + ".fspace[" + std::to_string(Is) + "].interior"),
     ...);
    (std::get<Is>(boundary_).write(
         archive, key + ".fspace[" + std::to_string(Is) + "].boundary"),
     ...);
    return archive;
  }

public:
  /// @brief Writes the function space object into a
  /// torch::serialize::OutputArchive object
  inline torch::serialize::OutputArchive &
  write(torch::serialize::OutputArchive &archive,
        const std::string &key = "functionspace") const {
    write_(std::make_index_sequence<FunctionSpace::nspaces()>{}, archive, key);
    return archive;
  }

private:
  /// @brief Loads the function space object from a
  /// torch::serialize::InputArchive object
  template <std::size_t... Is>
  inline torch::serialize::InputArchive &
  read_(std::index_sequence<Is...>, torch::serialize::InputArchive &archive,
        const std::string &key = "functionspace") {
    (std::get<Is>(spline_).read(archive, key + ".fspace[" + std::to_string(Is) +
                                             "].interior"),
     ...);
    (std::get<Is>(boundary_).read(
         archive, key + ".fspace[" + std::to_string(Is) + "].boundary"),
     ...);
    return archive;
  }

public:
  /// @brief Loads the function space object from a
  /// torch::serialize::InputArchive object
  inline torch::serialize::InputArchive &
  read(torch::serialize::InputArchive &archive,
       const std::string &key = "functionspace") {
    read_(std::make_index_sequence<FunctionSpace::nspaces()>{}, archive, key);
    return archive;
  }

  /// @brief Returns a string representation of the function space object
  inline virtual void
  pretty_print(std::ostream &os = Log(log::info)) const noexcept override {

    auto pretty_print_ = [this,
                          &os]<std::size_t... Is>(std::index_sequence<Is...>) {
      ((os << "\ninterior = ", std::get<Is>(spline_).pretty_print(os),
        os << "\nboundary = ", std::get<Is>(boundary_).pretty_print(os)),
       ...);
    };

    pretty_print_(std::make_index_sequence<nspaces()>{});
  }

  //  clang-format off
  /// @brief Returns a block-tensor with the curl of the
  /// function space object with respect to the parametric variables
  ///
  /// @param[in] xi Point(s) where to evaluate the curl
  ///
  /// @result Block-tensor with the curl with respect to the
  /// parametric variables `xi`
  /// \f[
  ///     \nabla_{\boldsymbol{\xi}} \times \mathbf{u}
  ///        =
  ///     \begin{bmatrix}
  ///        \mathbf{i}_0 & \cdots & \mathbf{i}_{d_\text{par}} \\
  ///        \frac{\partial}{\partial\xi_0} & \cdots &
  ///        \frac{\partial}{\partial\xi_{d_\text{par}}} \\
  ///        u_0 & \cdots & u_{d_\text{par}}
  ///     \end{bmatrix}
  /// \f]
  //  clang-format off
  ///
  /// @{
  template <functionspace comp = functionspace::interior,
            bool memory_optimized = false>
  inline auto curl(const utils::TensorArray1 &xi,
                   const std::tuple<utils::TensorArray1> &knot_indices,
                   const std::tuple<torch::Tensor> &coeff_indices) const {

    assert(xi[0].sizes() == std::get<0>(knot_indices)[0].sizes());

    throw std::runtime_error("Unsupported parametric/geometric dimension");

    return utils::BlockTensor<torch::Tensor, 1, 1>{};
  }

  template <functionspace comp = functionspace::interior,
            bool memory_optimized = false>
  inline auto
  curl(const utils::TensorArray2 &xi,
       const std::tuple<utils::TensorArray2, utils::TensorArray2> &knot_indices,
       const std::tuple<torch::Tensor, torch::Tensor> &coeff_indices) const {

    assert(xi[0].sizes() == std::get<0>(knot_indices)[0].sizes() &&
           xi[1].sizes() == std::get<1>(knot_indices)[0].sizes() &&
           xi[0].sizes() == xi[1].sizes());

    /// curl = 0,
    ///        0,
    ///        du_y / dx - du_x / dy
    ///
    /// Only the third component is returned
    return utils::BlockTensor<torch::Tensor, 1, 1>(
        *std::get<1>(spline_).template eval<deriv::dx, memory_optimized>(
            xi, std::get<1>(knot_indices), std::get<1>(coeff_indices))[0] -
        *std::get<0>(spline_).template eval<deriv::dx, memory_optimized>(
            xi, std::get<0>(knot_indices), std::get<0>(coeff_indices))[0]);
  }

  template <functionspace comp = functionspace::interior,
            bool memory_optimized = false>
  inline auto curl(const utils::TensorArray3 &xi,
                   const std::tuple<utils::TensorArray3, utils::TensorArray3,
                                    utils::TensorArray3> &knot_indices,
                   const std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
                       &coeff_indices) const {

    assert(xi[0].sizes() == std::get<0>(knot_indices)[0].sizes() &&
           xi[1].sizes() == std::get<1>(knot_indices)[0].sizes() &&
           xi[2].sizes() == std::get<2>(knot_indices)[0].sizes() &&
           xi[0].sizes() == xi[1].sizes() && xi[1].sizes() == xi[2].sizes());

    /// curl = du_z / dy - du_y / dz,
    ///        du_x / dz - du_z / dx,
    ///        du_y / dx - du_x / dy
    return utils::BlockTensor<torch::Tensor, 1, 3>(
        *std::get<2>(spline_).template eval<deriv::dy, memory_optimized>(
            xi, std::get<2>(knot_indices), std::get<2>(coeff_indices))[0] -
            *std::get<1>(spline_).template eval<deriv::dz, memory_optimized>(
                xi, std::get<1>(knot_indices), std::get<1>(coeff_indices))[0],
        *std::get<0>(spline_).template eval<deriv::dz, memory_optimized>(
            xi, std::get<0>(knot_indices), std::get<0>(coeff_indices))[0] -
            *std::get<2>(spline_).template eval<deriv::dx, memory_optimized>(
                xi, std::get<2>(knot_indices), std::get<2>(coeff_indices))[0],
        *std::get<1>(spline_).template eval<deriv::dx, memory_optimized>(
            xi, std::get<1>(knot_indices), std::get<1>(coeff_indices))[0] -
            *std::get<0>(spline_).template eval<deriv::dy, memory_optimized>(
                xi, std::get<0>(knot_indices), std::get<0>(coeff_indices))[0]);
  }

  template <functionspace comp = functionspace::interior,
            bool memory_optimized = false>
  inline auto
  curl(const utils::TensorArray4 &xi,
       const std::tuple<utils::TensorArray4, utils::TensorArray4,
                        utils::TensorArray4, utils::TensorArray4> &knot_indices,
       const std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
                        torch::Tensor> &coeff_indices) const {

    assert(xi[0].sizes() == std::get<0>(knot_indices)[0].sizes() &&
           xi[1].sizes() == std::get<1>(knot_indices)[0].sizes() &&
           xi[2].sizes() == std::get<2>(knot_indices)[0].sizes() &&
           xi[3].sizes() == std::get<3>(knot_indices)[0].sizes() &&
           xi[0].sizes() == xi[1].sizes() && xi[1].sizes() == xi[2].sizes() &&
           xi[2].sizes() == xi[3].sizes());

    throw std::runtime_error("Unsupported parametric/geometric dimension");

    return utils::BlockTensor<torch::Tensor, 1, 3>{};
  }
  /// @}

  /// @brief Returns a block-tensor with the divergence of the
  /// function space object with respect to the parametric variables
  ///
  /// @param[in] xi Point(s) where to evaluate the divergence
  ///
  /// @param[in] knot_indices Knot indices where to evaluate the divergence
  ///
  /// @param[in] coeff_indices Coefficient indices where to evaluate the
  /// divergence
  ///
  /// @result Block-tensor with the divergence of the function space with
  /// respect to the parametric variables
  /// \f[
  ///     \nabla_{\boldsymbol{\xi}} \cdot \mathbf{u}
  ///        =
  ///     \text{trace} ( J_{\boldsymbol{\xi}}(u) )
  ///        =
  ///     \frac{\partial u_0}{\partial \xi_0} +
  ///     \frac{\partial u_1}{\partial \xi_1} +
  ///        \dots
  ///     \frac{\partial u_{d_\text{geo}}}{\partial \xi_{d_\text{par}}}
  /// \f]
  ///
  /// @{
  template <functionspace comp = functionspace::interior,
            bool memory_optimized = false>
  inline auto div(const utils::TensorArray1 &xi,
                  const std::tuple<utils::TensorArray1> &knot_indices,
                  const std::tuple<torch::Tensor> &coeff_indices) const {

    assert(xi[0].sizes() == std::get<0>(knot_indices)[0].sizes());

    if constexpr (comp == functionspace::interior) {
      static_assert(std::tuple_element_t<0, spline_type>::geoDim() == 1,
                    "div(.) for vector-valued spaces requires 1D variables");

      return utils::BlockTensor<torch::Tensor, 1, 1>(
          std::get<0>(spline_).template eval<deriv::dx, memory_optimized>(
              xi, std::get<0>(knot_indices), std::get<0>(coeff_indices))[0]);
    }
  }

  template <functionspace comp = functionspace::interior,
            bool memory_optimized = false>
  inline auto
  div(const utils::TensorArray2 &xi,
      const std::tuple<utils::TensorArray2, utils::TensorArray2> &knot_indices,
      const std::tuple<torch::Tensor, torch::Tensor> &coeff_indices) const {

    assert(xi[0].sizes() == std::get<0>(knot_indices)[0].sizes() &&
           xi[1].sizes() == std::get<1>(knot_indices)[0].sizes() &&
           xi[0].sizes() == xi[1].sizes());

    if constexpr (comp == functionspace::interior) {
      static_assert(std::tuple_element_t<0, spline_type>::geoDim() == 1 &&
                        std::tuple_element_t<1, spline_type>::geoDim() == 1,
                    "div(.) for vector-valued spaces requires 1D variables");

      return utils::BlockTensor<torch::Tensor, 1, 1>(
          *std::get<0>(spline_).template eval<deriv::dx, memory_optimized>(
              xi, std::get<0>(knot_indices), std::get<0>(coeff_indices))[0] +
          *std::get<1>(spline_).template eval<deriv::dy, memory_optimized>(
              xi, std::get<1>(knot_indices), std::get<1>(coeff_indices))[0]);
    }
  }

  template <functionspace comp = functionspace::interior,
            bool memory_optimized = false>
  inline auto div(const utils::TensorArray3 &xi,
                  const std::tuple<utils::TensorArray3, utils::TensorArray3,
                                   utils::TensorArray3> &knot_indices,
                  const std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
                      &coeff_indices) const {

    assert(xi[0].sizes() == std::get<0>(knot_indices)[0].sizes() &&
           xi[1].sizes() == std::get<1>(knot_indices)[0].sizes() &&
           xi[2].sizes() == std::get<2>(knot_indices)[0].sizes() &&
           xi[0].sizes() == xi[1].sizes() && xi[1].sizes() == xi[2].sizes());

    if constexpr (comp == functionspace::interior) {
      static_assert(std::tuple_element_t<0, spline_type>::geoDim() == 1 &&
                        std::tuple_element_t<1, spline_type>::geoDim() == 1 &&
                        std::tuple_element_t<2, spline_type>::geoDim() == 1,
                    "div(.) for vector-valued spaces requires 1D variables");

      return utils::BlockTensor<torch::Tensor, 1, 1>(
          *std::get<0>(spline_).template eval<deriv::dx, memory_optimized>(
              xi, std::get<0>(knot_indices), std::get<0>(coeff_indices))[0] +
          *std::get<1>(spline_).template eval<deriv::dy, memory_optimized>(
              xi, std::get<1>(knot_indices), std::get<1>(coeff_indices))[0] +
          *std::get<2>(spline_).template eval<deriv::dz, memory_optimized>(
              xi, std::get<2>(knot_indices), std::get<2>(coeff_indices))[0]);
    }
  }

  template <functionspace comp = functionspace::interior,
            bool memory_optimized = false>
  inline auto
  div(const utils::TensorArray4 &xi,
      const std::tuple<utils::TensorArray4, utils::TensorArray4,
                       utils::TensorArray4, utils::TensorArray4> &knot_indices,
      const std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
                       torch::Tensor> &coeff_indices) const {

    assert(xi[0].sizes() == std::get<0>(knot_indices)[0].sizes() &&
           xi[1].sizes() == std::get<1>(knot_indices)[0].sizes() &&
           xi[2].sizes() == std::get<2>(knot_indices)[0].sizes() &&
           xi[3].sizes() == std::get<3>(knot_indices)[0].sizes() &&
           xi[0].sizes() == xi[1].sizes() && xi[1].sizes() == xi[2].sizes() &&
           xi[2].sizes() == xi[3].sizes());

    if constexpr (comp == functionspace::interior) {
      static_assert(std::tuple_element_t<0, spline_type>::geoDim() == 1 &&
                        std::tuple_element_t<1, spline_type>::geoDim() == 1 &&
                        std::tuple_element_t<2, spline_type>::geoDim() == 1 &&
                        std::tuple_element_t<3, spline_type>::geoDim() == 1,
                    "div(.) for vector-valued spaces requires 1D variables");

      return utils::BlockTensor<torch::Tensor, 1, 1>(
          *std::get<0>(spline_).template eval<deriv::dx, memory_optimized>(
              xi, std::get<0>(knot_indices), std::get<0>(coeff_indices))[0] +
          *std::get<1>(spline_).template eval<deriv::dy, memory_optimized>(
              xi, std::get<1>(knot_indices), std::get<1>(coeff_indices))[0] +
          *std::get<2>(spline_).template eval<deriv::dz, memory_optimized>(
              xi, std::get<2>(knot_indices), std::get<2>(coeff_indices))[0] +
          *std::get<3>(spline_).template eval<deriv::dt, memory_optimized>(
              xi, std::get<3>(knot_indices), std::get<3>(coeff_indices))[0]);
    }
  }
  /// @}

  /// @brief Returns a block-tensor with the gradient of the function space
  /// object in the points `xi` with respect to the parametric variables
  ///
  /// @param[in] xi Point(s) where to evaluate the gradient
  ///
  /// @result Block-tensor with the gradient with respect to the
  /// parametric variables `xi`
  /// \f[
  ///     \nabla_{\boldsymbol{\xi}}u
  ///        =
  ///     \left(\frac{\partial u_0}{\partial \xi_0},
  ///           \frac{\partial u_1}{\partial \xi_1},
  ///           \dots
  ///           \frac{\partial u_d}{\partial \xi_{d_\text{par}}}\right)
  /// \f]
  ///
  /// @{
  template <functionspace comp = functionspace::interior,
            bool memory_optimized = false>
  inline auto grad(const utils::TensorArray1 &xi,
                   const std::tuple<utils::TensorArray1> &knot_indices,
                   const std::tuple<torch::Tensor> &coeff_indices) const {

    assert(xi[0].sizes() == std::get<0>(knot_indices)[0].sizes());

    if constexpr (comp == functionspace::interior) {
      static_assert(std::tuple_element_t<0, spline_type>::geoDim() == 1,
                    "grid(.) for vector-valued spaces requires 1D variables");

      return utils::BlockTensor<torch::Tensor, 1, 1>(
          std::get<0>(spline_).template eval<deriv::dx, memory_optimized>(
              xi, std::get<0>(knot_indices), std::get<0>(coeff_indices))[0]);
    }
  }

  template <functionspace comp = functionspace::interior,
            bool memory_optimized = false>
  inline auto
  grad(const utils::TensorArray2 &xi,
       const std::tuple<utils::TensorArray2, utils::TensorArray2> &knot_indices,
       const std::tuple<torch::Tensor, torch::Tensor> &coeff_indices) const {

    assert(xi[0].sizes() == std::get<0>(knot_indices)[0].sizes() &&
           xi[1].sizes() == std::get<1>(knot_indices)[0].sizes() &&
           xi[0].sizes() == xi[1].sizes());

    if constexpr (comp == functionspace::interior) {
      static_assert(std::tuple_element_t<0, spline_type>::geoDim() == 1 &&
                        std::tuple_element_t<1, spline_type>::geoDim() == 1,
                    "grad(.) for vector-valued spaces requires 1D variables");

      return utils::BlockTensor<torch::Tensor, 1, 2>(
          std::get<0>(spline_).template eval<deriv::dx, memory_optimized>(
              xi, std::get<0>(knot_indices), std::get<0>(coeff_indices))[0],
          std::get<1>(spline_).template eval<deriv::dy, memory_optimized>(
              xi, std::get<1>(knot_indices), std::get<1>(coeff_indices))[0]);
    }
  }

  template <functionspace comp = functionspace::interior,
            bool memory_optimized = false>
  inline auto grad(const utils::TensorArray3 &xi,
                   const std::tuple<utils::TensorArray3, utils::TensorArray3,
                                    utils::TensorArray3> &knot_indices,
                   const std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
                       &coeff_indices) const {

    assert(xi[0].sizes() == std::get<0>(knot_indices)[0].sizes() &&
           xi[1].sizes() == std::get<1>(knot_indices)[0].sizes() &&
           xi[2].sizes() == std::get<2>(knot_indices)[0].sizes() &&
           xi[0].sizes() == xi[1].sizes() && xi[1].sizes() == xi[2].sizes());

    if constexpr (comp == functionspace::interior) {
      static_assert(std::tuple_element_t<0, spline_type>::geoDim() == 1 &&
                        std::tuple_element_t<1, spline_type>::geoDim() == 1 &&
                        std::tuple_element_t<2, spline_type>::geoDim() == 1,
                    "div(.) for vector-valued spaces requires 1D variables");

      return utils::BlockTensor<torch::Tensor, 1, 3>(
          std::get<0>(spline_).template eval<deriv::dx, memory_optimized>(
              xi, std::get<0>(knot_indices), std::get<0>(coeff_indices))[0],
          std::get<1>(spline_).template eval<deriv::dy, memory_optimized>(
              xi, std::get<1>(knot_indices), std::get<1>(coeff_indices))[0],
          std::get<2>(spline_).template eval<deriv::dz, memory_optimized>(
              xi, std::get<2>(knot_indices), std::get<2>(coeff_indices))[0]);
    }
  }

  template <functionspace comp = functionspace::interior,
            bool memory_optimized = false>
  inline auto
  grad(const utils::TensorArray4 &xi,
       const std::tuple<utils::TensorArray4, utils::TensorArray4,
                        utils::TensorArray4, utils::TensorArray4> &knot_indices,
       const std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
                        torch::Tensor> &coeff_indices) const {

    assert(xi[0].sizes() == std::get<0>(knot_indices)[0].sizes() &&
           xi[1].sizes() == std::get<1>(knot_indices)[0].sizes() &&
           xi[2].sizes() == std::get<2>(knot_indices)[0].sizes() &&
           xi[3].sizes() == std::get<3>(knot_indices)[0].sizes() &&
           xi[0].sizes() == xi[1].sizes() && xi[1].sizes() == xi[2].sizes() &&
           xi[2].sizes() == xi[3].sizes());

    if constexpr (comp == functionspace::interior) {
      static_assert(std::tuple_element_t<0, spline_type>::geoDim() == 1 &&
                        std::tuple_element_t<1, spline_type>::geoDim() == 1 &&
                        std::tuple_element_t<2, spline_type>::geoDim() == 1 &&
                        std::tuple_element_t<3, spline_type>::geoDim() == 1,
                    "grad(.) for vector-valued spaces requires 1D variables");

      return utils::BlockTensor<torch::Tensor, 1, 4>(
          std::get<0>(spline_).template eval<deriv::dx, memory_optimized>(
              xi, std::get<0>(knot_indices), std::get<0>(coeff_indices))[0],
          std::get<1>(spline_).template eval<deriv::dy, memory_optimized>(
              xi, std::get<1>(knot_indices), std::get<1>(coeff_indices))[0],
          std::get<2>(spline_).template eval<deriv::dz, memory_optimized>(
              xi, std::get<2>(knot_indices), std::get<2>(coeff_indices))[0],
          std::get<3>(spline_).template eval<deriv::dt, memory_optimized>(
              xi, std::get<3>(knot_indices), std::get<3>(coeff_indices))[0]);
    }
  }
  /// @}

  //  clang-format off
  /// @brief Returns a block-tensor with the Hessian of the function space
  /// object in the points `xi` with respect to the parametric
  /// variables
  ///
  /// @param[in] xi Point(s) where to evaluate the Hessian
  ///
  /// @result Block-tensor with the Hessian with respect to the
  /// parametric variables `xi`
  /// \f[
  ///     H_{\boldsymbol{\xi}}(u)
  ///        =
  ///     \begin{bmatrix}
  ///           \frac{\partial^2 u}{\partial^2 \xi_0}&
  ///           \frac{\partial^2 u}{\partial \xi_0\partial \xi_1}&
  ///           \dots&
  ///           \frac{\partial^2 u}{\partial \xi_0\partial \xi_{d_\text{par}}}
  ///           \\ \frac{\partial^2 u}{\partial \xi_1\partial \xi_0}&
  ///           \frac{\partial^2 u}{\partial^2 \xi_1}&
  ///           \dots&
  ///           \frac{\partial^2 u}{\partial \xi_1\partial \xi_{d_\text{par}}}
  ///           \\
  ///           \vdots& \vdots & \ddots & \vdots \\
  ///           \frac{\partial^2 u}{\partial \xi_{d_\text{par}}\partial \xi_0}&
  ///           \frac{\partial^2 u}{\partial \xi_{d_\text{par}}\partial \xi_1}&
  ///           \dots&
  ///           \frac{\partial^2 u}{\partial^2 \xi_{d_\text{par}}}
  ///     \end{bmatrix}
  /// \f]
  ///
  /// @note If the function space object has geometric dimension larger
  /// then one then all Hessian matrices are returned as slices of a
  /// rank-3 tensor.
  //  clang-format on
  ///
  /// @{
  template <functionspace comp = functionspace::interior,
            bool memory_optimized = false>
  inline auto hess(const utils::TensorArray1 &xi,
                   const std::tuple<utils::TensorArray1> &knot_indices,
                   const std::tuple<torch::Tensor> &coeff_indices) const {

    assert(xi[0].sizes() == std::get<0>(knot_indices)[0].sizes());

    if constexpr (comp == functionspace::interior) {
      static_assert(std::tuple_element_t<0, spline_type>::geoDim() == 1,
                    "hess(.) for vector-valued spaces requires 1D variables");

      return utils::BlockTensor<torch::Tensor, 1, 1>(
          std::get<0>(spline_).template eval<deriv::dx ^ 2, memory_optimized>(
              xi, std::get<0>(knot_indices), std::get<0>(coeff_indices)));
    }
  }

  template <functionspace comp = functionspace::interior,
            bool memory_optimized = false>
  inline auto
  hess(const utils::TensorArray2 &xi,
       const std::tuple<utils::TensorArray2, utils::TensorArray2> &knot_indices,
       const std::tuple<torch::Tensor, torch::Tensor> &coeff_indices) const {

    assert(xi[0].sizes() == std::get<0>(knot_indices)[0].sizes() &&
           xi[1].sizes() == std::get<1>(knot_indices)[0].sizes() &&
           xi[0].sizes() == xi[1].sizes());

    if constexpr (comp == functionspace::interior) {
      static_assert(std::tuple_element_t<0, spline_type>::geoDim() == 1 &&
                        std::tuple_element_t<1, spline_type>::geoDim() == 1,
                    "hess(.) for vector-valued spaces requires 1D variables");

      return utils::BlockTensor<torch::Tensor, 2, 2, 2>(
          std::get<0>(spline_).template eval<deriv::dx ^ 2, memory_optimized>(
              xi, std::get<0>(knot_indices), std::get<0>(coeff_indices)),
          std::get<0>(spline_)
              .template eval<deriv::dx + deriv::dy, memory_optimized>(
                  xi, std::get<0>(knot_indices), std::get<0>(coeff_indices)),
          std::get<0>(spline_)
              .template eval<deriv::dy + deriv::dx, memory_optimized>(
                  xi, std::get<0>(knot_indices), std::get<0>(coeff_indices)),
          std::get<0>(spline_).template eval<deriv::dy ^ 2, memory_optimized>(
              xi, std::get<0>(knot_indices), std::get<0>(coeff_indices)),

          std::get<1>(spline_).template eval<deriv::dx ^ 2, memory_optimized>(
              xi, std::get<1>(knot_indices), std::get<1>(coeff_indices)),
          std::get<1>(spline_)
              .template eval<deriv::dx + deriv::dy, memory_optimized>(
                  xi, std::get<1>(knot_indices), std::get<1>(coeff_indices)),
          std::get<1>(spline_)
              .template eval<deriv::dy + deriv::dx, memory_optimized>(
                  xi, std::get<1>(knot_indices), std::get<1>(coeff_indices)),
          std::get<1>(spline_).template eval<deriv::dy ^ 2, memory_optimized>(
              xi, std::get<1>(knot_indices), std::get<1>(coeff_indices)));
    }
  }

  template <functionspace comp = functionspace::interior,
            bool memory_optimized = false>
  inline auto hess(const utils::TensorArray3 &xi,
                   const std::tuple<utils::TensorArray3, utils::TensorArray3,
                                    utils::TensorArray3> &knot_indices,
                   const std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
                       &coeff_indices) const {

    assert(xi[0].sizes() == std::get<0>(knot_indices)[0].sizes() &&
           xi[1].sizes() == std::get<1>(knot_indices)[0].sizes() &&
           xi[2].sizes() == std::get<2>(knot_indices)[0].sizes() &&
           xi[0].sizes() == xi[1].sizes() && xi[1].sizes() == xi[2].sizes());

    if constexpr (comp == functionspace::interior) {
      static_assert(std::tuple_element_t<0, spline_type>::geoDim() == 1 &&
                        std::tuple_element_t<1, spline_type>::geoDim() == 1 &&
                        std::tuple_element_t<2, spline_type>::geoDim() == 1,
                    "hess(.) for vector-valued spaces requires 1D variables");

      return utils::BlockTensor<torch::Tensor, 3, 3, 3>(
          std::get<0>(spline_).template eval<deriv::dx ^ 2, memory_optimized>(
              xi, std::get<0>(knot_indices), std::get<0>(coeff_indices)),
          std::get<0>(spline_)
              .template eval<deriv::dx + deriv::dy, memory_optimized>(
                  xi, std::get<0>(knot_indices), std::get<0>(coeff_indices)),
          std::get<0>(spline_)
              .template eval<deriv::dx + deriv::dz, memory_optimized>(
                  xi, std::get<0>(knot_indices), std::get<0>(coeff_indices)),
          std::get<0>(spline_)
              .template eval<deriv::dy + deriv::dx, memory_optimized>(
                  xi, std::get<0>(knot_indices), std::get<0>(coeff_indices)),
          std::get<0>(spline_).template eval<deriv::dy ^ 2, memory_optimized>(
              xi, std::get<0>(knot_indices), std::get<0>(coeff_indices)),
          std::get<0>(spline_)
              .template eval<deriv::dy + deriv::dz, memory_optimized>(
                  xi, std::get<0>(knot_indices), std::get<0>(coeff_indices)),
          std::get<0>(spline_)
              .template eval<deriv::dz + deriv::dx, memory_optimized>(
                  xi, std::get<0>(knot_indices), std::get<0>(coeff_indices)),
          std::get<0>(spline_)
              .template eval<deriv::dz + deriv::dy, memory_optimized>(
                  xi, std::get<0>(knot_indices), std::get<0>(coeff_indices)),
          std::get<0>(spline_).template eval<deriv::dz ^ 2, memory_optimized>(
              xi, std::get<0>(knot_indices), std::get<0>(coeff_indices)),

          std::get<1>(spline_).template eval<deriv::dx ^ 2, memory_optimized>(
              xi, std::get<1>(knot_indices), std::get<1>(coeff_indices)),
          std::get<1>(spline_)
              .template eval<deriv::dx + deriv::dy, memory_optimized>(
                  xi, std::get<1>(knot_indices), std::get<1>(coeff_indices)),
          std::get<1>(spline_)
              .template eval<deriv::dx + deriv::dz, memory_optimized>(
                  xi, std::get<1>(knot_indices), std::get<1>(coeff_indices)),
          std::get<1>(spline_)
              .template eval<deriv::dy + deriv::dx, memory_optimized>(
                  xi, std::get<1>(knot_indices), std::get<1>(coeff_indices)),
          std::get<1>(spline_).template eval<deriv::dy ^ 2, memory_optimized>(
              xi, std::get<1>(knot_indices), std::get<1>(coeff_indices)),
          std::get<1>(spline_)
              .template eval<deriv::dy + deriv::dz, memory_optimized>(
                  xi, std::get<1>(knot_indices), std::get<1>(coeff_indices)),
          std::get<1>(spline_)
              .template eval<deriv::dz + deriv::dx, memory_optimized>(
                  xi, std::get<1>(knot_indices), std::get<1>(coeff_indices)),
          std::get<1>(spline_)
              .template eval<deriv::dz + deriv::dy, memory_optimized>(
                  xi, std::get<1>(knot_indices), std::get<1>(coeff_indices)),
          std::get<1>(spline_).template eval<deriv::dz ^ 2, memory_optimized>(
              xi, std::get<1>(knot_indices), std::get<1>(coeff_indices)),

          std::get<2>(spline_).template eval<deriv::dx ^ 2, memory_optimized>(
              xi, std::get<2>(knot_indices), std::get<2>(coeff_indices)),
          std::get<2>(spline_)
              .template eval<deriv::dx + deriv::dy, memory_optimized>(
                  xi, std::get<2>(knot_indices), std::get<2>(coeff_indices)),
          std::get<2>(spline_)
              .template eval<deriv::dx + deriv::dz, memory_optimized>(
                  xi, std::get<2>(knot_indices), std::get<2>(coeff_indices)),
          std::get<2>(spline_)
              .template eval<deriv::dy + deriv::dx, memory_optimized>(
                  xi, std::get<2>(knot_indices), std::get<2>(coeff_indices)),
          std::get<2>(spline_).template eval<deriv::dy ^ 2, memory_optimized>(
              xi, std::get<2>(knot_indices), std::get<2>(coeff_indices)),
          std::get<2>(spline_)
              .template eval<deriv::dy + deriv::dz, memory_optimized>(
                  xi, std::get<2>(knot_indices), std::get<2>(coeff_indices)),
          std::get<2>(spline_)
              .template eval<deriv::dz + deriv::dx, memory_optimized>(
                  xi, std::get<2>(knot_indices), std::get<2>(coeff_indices)),
          std::get<2>(spline_)
              .template eval<deriv::dz + deriv::dy, memory_optimized>(
                  xi, std::get<2>(knot_indices), std::get<2>(coeff_indices)),
          std::get<2>(spline_).template eval<deriv::dz ^ 2, memory_optimized>(
              xi, std::get<2>(knot_indices), std::get<2>(coeff_indices)));
    }
  }

  template <functionspace comp = functionspace::interior,
            bool memory_optimized = false>
  inline auto
  hess(const utils::TensorArray4 &xi,
       const std::tuple<utils::TensorArray4, utils::TensorArray4,
                        utils::TensorArray4, utils::TensorArray4> &knot_indices,
       const std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
                        torch::Tensor> &coeff_indices) const {

    assert(xi[0].sizes() == std::get<0>(knot_indices)[0].sizes() &&
           xi[1].sizes() == std::get<1>(knot_indices)[0].sizes() &&
           xi[2].sizes() == std::get<2>(knot_indices)[0].sizes() &&
           xi[3].sizes() == std::get<3>(knot_indices)[0].sizes() &&
           xi[0].sizes() == xi[1].sizes() && xi[1].sizes() == xi[2].sizes() &&
           xi[2].sizes() == xi[3].sizes());

    if constexpr (comp == functionspace::interior) {
      static_assert(std::tuple_element_t<0, spline_type>::geoDim() == 1 &&
                        std::tuple_element_t<1, spline_type>::geoDim() == 1 &&
                        std::tuple_element_t<2, spline_type>::geoDim() == 1 &&
                        std::tuple_element_t<3, spline_type>::geoDim() == 1,
                    "hess(.) for vector-valued spaces requires 1D variables");

      return utils::BlockTensor<torch::Tensor, 4, 4, 4>(
          std::get<0>(spline_).template eval<deriv::dx ^ 2, memory_optimized>(
              xi, std::get<0>(knot_indices), std::get<0>(coeff_indices)),
          std::get<0>(spline_)
              .template eval<deriv::dx + deriv::dy, memory_optimized>(
                  xi, std::get<0>(knot_indices), std::get<0>(coeff_indices)),
          std::get<0>(spline_)
              .template eval<deriv::dx + deriv::dz, memory_optimized>(
                  xi, std::get<0>(knot_indices), std::get<0>(coeff_indices)),
          std::get<0>(spline_)
              .template eval<deriv::dx + deriv::dt, memory_optimized>(
                  xi, std::get<0>(knot_indices), std::get<0>(coeff_indices)),
          std::get<0>(spline_)
              .template eval<deriv::dy + deriv::dx, memory_optimized>(
                  xi, std::get<0>(knot_indices), std::get<0>(coeff_indices)),
          std::get<0>(spline_).template eval<deriv::dy ^ 2, memory_optimized>(
              xi, std::get<0>(knot_indices), std::get<0>(coeff_indices)),
          std::get<0>(spline_)
              .template eval<deriv::dy + deriv::dz, memory_optimized>(
                  xi, std::get<0>(knot_indices), std::get<0>(coeff_indices)),
          std::get<0>(spline_)
              .template eval<deriv::dy + deriv::dt, memory_optimized>(
                  xi, std::get<0>(knot_indices), std::get<0>(coeff_indices)),
          std::get<0>(spline_)
              .template eval<deriv::dz + deriv::dx, memory_optimized>(
                  xi, std::get<0>(knot_indices), std::get<0>(coeff_indices)),
          std::get<0>(spline_)
              .template eval<deriv::dz + deriv::dy, memory_optimized>(
                  xi, std::get<0>(knot_indices), std::get<0>(coeff_indices)),
          std::get<0>(spline_).template eval<deriv::dz ^ 2, memory_optimized>(
              xi, std::get<0>(knot_indices), std::get<0>(coeff_indices)),
          std::get<0>(spline_)
              .template eval<deriv::dz + deriv::dt, memory_optimized>(
                  xi, std::get<0>(knot_indices), std::get<0>(coeff_indices)),
          std::get<0>(spline_)
              .template eval<deriv::dt + deriv::dx, memory_optimized>(
                  xi, std::get<0>(knot_indices), std::get<0>(coeff_indices)),
          std::get<0>(spline_)
              .template eval<deriv::dt + deriv::dy, memory_optimized>(
                  xi, std::get<0>(knot_indices), std::get<0>(coeff_indices)),
          std::get<0>(spline_)
              .template eval<deriv::dt + deriv::dz, memory_optimized>(
                  xi, std::get<0>(knot_indices), std::get<0>(coeff_indices)),
          std::get<0>(spline_).template eval<deriv::dt ^ 2, memory_optimized>(
              xi, std::get<0>(knot_indices), std::get<0>(coeff_indices)),

          std::get<1>(spline_).template eval<deriv::dx ^ 2, memory_optimized>(
              xi, std::get<1>(knot_indices), std::get<1>(coeff_indices)),
          std::get<1>(spline_)
              .template eval<deriv::dx + deriv::dy, memory_optimized>(
                  xi, std::get<1>(knot_indices), std::get<1>(coeff_indices)),
          std::get<1>(spline_)
              .template eval<deriv::dx + deriv::dz, memory_optimized>(
                  xi, std::get<1>(knot_indices), std::get<1>(coeff_indices)),
          std::get<1>(spline_)
              .template eval<deriv::dx + deriv::dt, memory_optimized>(
                  xi, std::get<1>(knot_indices), std::get<1>(coeff_indices)),
          std::get<1>(spline_)
              .template eval<deriv::dy + deriv::dx, memory_optimized>(
                  xi, std::get<1>(knot_indices), std::get<1>(coeff_indices)),
          std::get<1>(spline_).template eval<deriv::dy ^ 2, memory_optimized>(
              xi, std::get<1>(knot_indices), std::get<1>(coeff_indices)),
          std::get<1>(spline_)
              .template eval<deriv::dy + deriv::dz, memory_optimized>(
                  xi, std::get<1>(knot_indices), std::get<1>(coeff_indices)),
          std::get<1>(spline_)
              .template eval<deriv::dy + deriv::dt, memory_optimized>(
                  xi, std::get<1>(knot_indices), std::get<1>(coeff_indices)),
          std::get<1>(spline_)
              .template eval<deriv::dz + deriv::dx, memory_optimized>(
                  xi, std::get<1>(knot_indices), std::get<1>(coeff_indices)),
          std::get<1>(spline_)
              .template eval<deriv::dz + deriv::dy, memory_optimized>(
                  xi, std::get<1>(knot_indices), std::get<1>(coeff_indices)),
          std::get<1>(spline_).template eval<deriv::dz ^ 2, memory_optimized>(
              xi, std::get<1>(knot_indices), std::get<1>(coeff_indices)),
          std::get<1>(spline_)
              .template eval<deriv::dz + deriv::dt, memory_optimized>(
                  xi, std::get<1>(knot_indices), std::get<1>(coeff_indices)),
          std::get<1>(spline_)
              .template eval<deriv::dt + deriv::dx, memory_optimized>(
                  xi, std::get<1>(knot_indices), std::get<1>(coeff_indices)),
          std::get<1>(spline_)
              .template eval<deriv::dt + deriv::dy, memory_optimized>(
                  xi, std::get<1>(knot_indices), std::get<1>(coeff_indices)),
          std::get<1>(spline_)
              .template eval<deriv::dt + deriv::dz, memory_optimized>(
                  xi, std::get<1>(knot_indices), std::get<1>(coeff_indices)),
          std::get<1>(spline_).template eval<deriv::dt ^ 2, memory_optimized>(
              xi, std::get<1>(knot_indices), std::get<1>(coeff_indices)),

          std::get<2>(spline_).template eval<deriv::dx ^ 2, memory_optimized>(
              xi, std::get<2>(knot_indices), std::get<2>(coeff_indices)),
          std::get<2>(spline_)
              .template eval<deriv::dx + deriv::dy, memory_optimized>(
                  xi, std::get<2>(knot_indices), std::get<2>(coeff_indices)),
          std::get<2>(spline_)
              .template eval<deriv::dx + deriv::dz, memory_optimized>(
                  xi, std::get<2>(knot_indices), std::get<2>(coeff_indices)),
          std::get<2>(spline_)
              .template eval<deriv::dx + deriv::dt, memory_optimized>(
                  xi, std::get<2>(knot_indices), std::get<2>(coeff_indices)),
          std::get<2>(spline_)
              .template eval<deriv::dy + deriv::dx, memory_optimized>(
                  xi, std::get<2>(knot_indices), std::get<2>(coeff_indices)),
          std::get<2>(spline_).template eval<deriv::dy ^ 2, memory_optimized>(
              xi, std::get<2>(knot_indices), std::get<2>(coeff_indices)),
          std::get<2>(spline_)
              .template eval<deriv::dy + deriv::dz, memory_optimized>(
                  xi, std::get<2>(knot_indices), std::get<2>(coeff_indices)),
          std::get<2>(spline_)
              .template eval<deriv::dy + deriv::dt, memory_optimized>(
                  xi, std::get<2>(knot_indices), std::get<2>(coeff_indices)),
          std::get<2>(spline_)
              .template eval<deriv::dz + deriv::dx, memory_optimized>(
                  xi, std::get<2>(knot_indices), std::get<2>(coeff_indices)),
          std::get<2>(spline_)
              .template eval<deriv::dz + deriv::dy, memory_optimized>(
                  xi, std::get<2>(knot_indices), std::get<2>(coeff_indices)),
          std::get<2>(spline_).template eval<deriv::dz ^ 2, memory_optimized>(
              xi, std::get<2>(knot_indices), std::get<2>(coeff_indices)),
          std::get<2>(spline_)
              .template eval<deriv::dz + deriv::dt, memory_optimized>(
                  xi, std::get<2>(knot_indices), std::get<2>(coeff_indices)),
          std::get<2>(spline_)
              .template eval<deriv::dt + deriv::dx, memory_optimized>(
                  xi, std::get<2>(knot_indices), std::get<2>(coeff_indices)),
          std::get<2>(spline_)
              .template eval<deriv::dt + deriv::dy, memory_optimized>(
                  xi, std::get<2>(knot_indices), std::get<2>(coeff_indices)),
          std::get<2>(spline_)
              .template eval<deriv::dt + deriv::dz, memory_optimized>(
                  xi, std::get<2>(knot_indices), std::get<2>(coeff_indices)),
          std::get<2>(spline_).template eval<deriv::dt ^ 2, memory_optimized>(
              xi, std::get<2>(knot_indices), std::get<2>(coeff_indices)),

          std::get<3>(spline_).template eval<deriv::dx ^ 2, memory_optimized>(
              xi, std::get<3>(knot_indices), std::get<3>(coeff_indices)),
          std::get<3>(spline_)
              .template eval<deriv::dx + deriv::dy, memory_optimized>(
                  xi, std::get<3>(knot_indices), std::get<3>(coeff_indices)),
          std::get<3>(spline_)
              .template eval<deriv::dx + deriv::dz, memory_optimized>(
                  xi, std::get<3>(knot_indices), std::get<3>(coeff_indices)),
          std::get<3>(spline_)
              .template eval<deriv::dx + deriv::dt, memory_optimized>(
                  xi, std::get<3>(knot_indices), std::get<3>(coeff_indices)),
          std::get<3>(spline_)
              .template eval<deriv::dy + deriv::dx, memory_optimized>(
                  xi, std::get<3>(knot_indices), std::get<3>(coeff_indices)),
          std::get<3>(spline_).template eval<deriv::dy ^ 2, memory_optimized>(
              xi, std::get<3>(knot_indices), std::get<3>(coeff_indices)),
          std::get<3>(spline_)
              .template eval<deriv::dy + deriv::dz, memory_optimized>(
                  xi, std::get<3>(knot_indices), std::get<3>(coeff_indices)),
          std::get<3>(spline_)
              .template eval<deriv::dy + deriv::dt, memory_optimized>(
                  xi, std::get<3>(knot_indices), std::get<3>(coeff_indices)),
          std::get<3>(spline_)
              .template eval<deriv::dz + deriv::dx, memory_optimized>(
                  xi, std::get<3>(knot_indices), std::get<3>(coeff_indices)),
          std::get<3>(spline_)
              .template eval<deriv::dz + deriv::dy, memory_optimized>(
                  xi, std::get<3>(knot_indices), std::get<3>(coeff_indices)),
          std::get<3>(spline_).template eval<deriv::dz ^ 2, memory_optimized>(
              xi, std::get<3>(knot_indices), std::get<3>(coeff_indices)),
          std::get<3>(spline_)
              .template eval<deriv::dz + deriv::dt, memory_optimized>(
                  xi, std::get<3>(knot_indices), std::get<3>(coeff_indices)),
          std::get<3>(spline_)
              .template eval<deriv::dt + deriv::dx, memory_optimized>(
                  xi, std::get<3>(knot_indices), std::get<3>(coeff_indices)),
          std::get<3>(spline_)
              .template eval<deriv::dt + deriv::dy, memory_optimized>(
                  xi, std::get<3>(knot_indices), std::get<3>(coeff_indices)),
          std::get<3>(spline_)
              .template eval<deriv::dt + deriv::dz, memory_optimized>(
                  xi, std::get<3>(knot_indices), std::get<3>(coeff_indices)),
          std::get<3>(spline_).template eval<deriv::dt ^ 2, memory_optimized>(
              xi, std::get<3>(knot_indices), std::get<3>(coeff_indices)));
    }
  }
  /// @}

  //  clang-format off
  /// @brief Returns a block-tensor with the Jacobian of the function
  /// space object in the points `xi` with respect to the parametric
  /// variables
  ///
  /// @param[in] xi Point(s) where to evaluate the Jacobian
  ///
  /// @param[in] knot_indices Knot indices where to evaluate the Jacobian
  ///
  /// @param[in] coeff_indices Coefficient indices where to evaluate the
  /// Jacobian
  ///
  /// @result Block-tensor with the Jacobian with respect to the
  /// parametric variables
  /// \f[
  ///     J_{\boldsymbol{\xi}}(u)
  ///        =
  ///     \begin{bmatrix}
  ///           \frac{\partial u_0}{\partial \xi_0}&
  ///           \frac{\partial u_0}{\partial \xi_1}&
  ///           \dots&
  ///           \frac{\partial u_0}{\partial \xi_{d_\text{par}}} \\
  ///           \frac{\partial u_1}{\partial \xi_0}&
  ///           \frac{\partial u_1}{\partial \xi_1}&
  ///           \dots&
  ///           \frac{\partial u_1}{\partial \xi_{d_\text{par}}} \\
  ///           \vdots& \vdots & \ddots & \vdots \\
  ///           \frac{\partial u_{d_\text{geo}}}{\partial \xi_0}&
  ///           \frac{\partial u_{d_\text{geo}}}{\partial \xi_1}&
  ///           \dots&
  ///           \frac{\partial u_{d_\text{geo}}}{\partial \xi_{d_\text{par}}}
  ///     \end{bmatrix}
  /// \f]
  ///
  //  clang-format on
  /// @{
  template <functionspace comp = functionspace::interior,
            bool memory_optimized = false>
  inline auto jac(const utils::TensorArray1 &xi,
                  const std::tuple<utils::TensorArray1> &knot_indices,
                  const std::tuple<torch::Tensor> &coeff_indices) const {

    assert(xi[0].sizes() == std::get<0>(knot_indices)[0].sizes());

    if constexpr (comp == functionspace::interior) {
      static_assert(std::tuple_element_t<0, spline_type>::geoDim() == 1,
                    "jac(.) for vector-valued spaces requires 1D variables");

      return utils::BlockTensor<torch::Tensor, 1, 1>(
          std::get<0>(spline_).template eval<deriv::dx, memory_optimized>(
              xi, std::get<0>(knot_indices), std::get<0>(coeff_indices))[0]);
    }
  }

  template <functionspace comp = functionspace::interior,
            bool memory_optimized = false>
  inline auto
  jac(const utils::TensorArray2 &xi,
      const std::tuple<utils::TensorArray2, utils::TensorArray2> &knot_indices,
      const std::tuple<torch::Tensor, torch::Tensor> &coeff_indices) const {

    assert(xi[0].sizes() == std::get<0>(knot_indices)[0].sizes() &&
           xi[1].sizes() == std::get<1>(knot_indices)[0].sizes() &&
           xi[0].sizes() == xi[1].sizes());

    if constexpr (comp == functionspace::interior) {
      static_assert(std::tuple_element_t<0, spline_type>::geoDim() == 1 &&
                        std::tuple_element_t<1, spline_type>::geoDim() == 1,
                    "jac(.) for vector-valued spaces requires 1D variables");

      return utils::BlockTensor<torch::Tensor, 2, 2>(
          std::get<0>(spline_).template eval<deriv::dx, memory_optimized>(
              xi, std::get<0>(knot_indices), std::get<0>(coeff_indices))[0],
          std::get<0>(spline_).template eval<deriv::dy, memory_optimized>(
              xi, std::get<0>(knot_indices), std::get<0>(coeff_indices))[0],

          std::get<1>(spline_).template eval<deriv::dx, memory_optimized>(
              xi, std::get<1>(knot_indices), std::get<1>(coeff_indices))[0],
          std::get<1>(spline_).template eval<deriv::dy, memory_optimized>(
              xi, std::get<1>(knot_indices), std::get<1>(coeff_indices))[0]);
    }
  }

  template <functionspace comp = functionspace::interior,
            bool memory_optimized = false>
  inline auto jac(const utils::TensorArray3 &xi,
                  const std::tuple<utils::TensorArray3, utils::TensorArray3,
                                   utils::TensorArray3> &knot_indices,
                  const std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
                      &coeff_indices) const {

    assert(xi[0].sizes() == std::get<0>(knot_indices)[0].sizes() &&
           xi[1].sizes() == std::get<1>(knot_indices)[0].sizes() &&
           xi[2].sizes() == std::get<2>(knot_indices)[0].sizes() &&
           xi[0].sizes() == xi[1].sizes() && xi[1].sizes() == xi[2].sizes());

    if constexpr (comp == functionspace::interior) {
      static_assert(std::tuple_element_t<0, spline_type>::geoDim() == 1 &&
                        std::tuple_element_t<1, spline_type>::geoDim() == 1 &&
                        std::tuple_element_t<2, spline_type>::geoDim() == 1,
                    "jac(.) for vector-valued spaces requires 1D variables");

      return utils::BlockTensor<torch::Tensor, 3, 3>(
          std::get<0>(spline_).template eval<deriv::dx, memory_optimized>(
              xi, std::get<0>(knot_indices), std::get<0>(coeff_indices))[0],
          std::get<0>(spline_).template eval<deriv::dy, memory_optimized>(
              xi, std::get<0>(knot_indices), std::get<0>(coeff_indices))[0],
          std::get<0>(spline_).template eval<deriv::dz, memory_optimized>(
              xi, std::get<0>(knot_indices), std::get<0>(coeff_indices))[0],

          std::get<1>(spline_).template eval<deriv::dx, memory_optimized>(
              xi, std::get<1>(knot_indices), std::get<1>(coeff_indices))[0],
          std::get<1>(spline_).template eval<deriv::dy, memory_optimized>(
              xi, std::get<1>(knot_indices), std::get<1>(coeff_indices))[0],
          std::get<1>(spline_).template eval<deriv::dz, memory_optimized>(
              xi, std::get<1>(knot_indices), std::get<1>(coeff_indices))[0],

          std::get<2>(spline_).template eval<deriv::dx, memory_optimized>(
              xi, std::get<2>(knot_indices), std::get<2>(coeff_indices))[0],
          std::get<2>(spline_).template eval<deriv::dy, memory_optimized>(
              xi, std::get<2>(knot_indices), std::get<2>(coeff_indices))[0],
          std::get<2>(spline_).template eval<deriv::dz, memory_optimized>(
              xi, std::get<2>(knot_indices), std::get<2>(coeff_indices))[0]);
    }
  }

  template <functionspace comp = functionspace::interior,
            bool memory_optimized = false>
  inline auto
  jac(const utils::TensorArray4 &xi,
      const std::tuple<utils::TensorArray4, utils::TensorArray4,
                       utils::TensorArray4, utils::TensorArray4> &knot_indices,
      const std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
                       torch::Tensor> &coeff_indices) const {

    assert(xi[0].sizes() == std::get<0>(knot_indices)[0].sizes() &&
           xi[1].sizes() == std::get<1>(knot_indices)[0].sizes() &&
           xi[2].sizes() == std::get<2>(knot_indices)[0].sizes() &&
           xi[3].sizes() == std::get<3>(knot_indices)[0].sizes() &&
           xi[0].sizes() == xi[1].sizes() && xi[1].sizes() == xi[2].sizes() &&
           xi[2].sizes() == xi[3].sizes());

    if constexpr (comp == functionspace::interior) {
      static_assert(std::tuple_element_t<0, spline_type>::geoDim() == 1 &&
                        std::tuple_element_t<1, spline_type>::geoDim() == 1 &&
                        std::tuple_element_t<2, spline_type>::geoDim() == 1,
                    "jac(.) for vector-valued spaces requires 1D variables");

      return utils::BlockTensor<torch::Tensor, 4, 4>(
          std::get<0>(spline_).template eval<deriv::dx, memory_optimized>(
              xi, std::get<0>(knot_indices), std::get<0>(coeff_indices))[0],
          std::get<0>(spline_).template eval<deriv::dy, memory_optimized>(
              xi, std::get<0>(knot_indices), std::get<0>(coeff_indices))[0],
          std::get<0>(spline_).template eval<deriv::dz, memory_optimized>(
              xi, std::get<0>(knot_indices), std::get<0>(coeff_indices))[0],
          std::get<0>(spline_).template eval<deriv::dt, memory_optimized>(
              xi, std::get<0>(knot_indices), std::get<0>(coeff_indices))[0],

          std::get<1>(spline_).template eval<deriv::dx, memory_optimized>(
              xi, std::get<1>(knot_indices), std::get<1>(coeff_indices))[0],
          std::get<1>(spline_).template eval<deriv::dy, memory_optimized>(
              xi, std::get<1>(knot_indices), std::get<1>(coeff_indices))[0],
          std::get<1>(spline_).template eval<deriv::dz, memory_optimized>(
              xi, std::get<1>(knot_indices), std::get<1>(coeff_indices))[0],
          std::get<1>(spline_).template eval<deriv::dt, memory_optimized>(
              xi, std::get<1>(knot_indices), std::get<1>(coeff_indices))[0],

          std::get<2>(spline_).template eval<deriv::dx, memory_optimized>(
              xi, std::get<2>(knot_indices), std::get<2>(coeff_indices))[0],
          std::get<2>(spline_).template eval<deriv::dy, memory_optimized>(
              xi, std::get<2>(knot_indices), std::get<2>(coeff_indices))[0],
          std::get<2>(spline_).template eval<deriv::dz, memory_optimized>(
              xi, std::get<2>(knot_indices), std::get<2>(coeff_indices))[0],
          std::get<2>(spline_).template eval<deriv::dt, memory_optimized>(
              xi, std::get<2>(knot_indices), std::get<2>(coeff_indices))[0],

          std::get<3>(spline_).template eval<deriv::dx, memory_optimized>(
              xi, std::get<3>(knot_indices), std::get<3>(coeff_indices))[0],
          std::get<3>(spline_).template eval<deriv::dy, memory_optimized>(
              xi, std::get<3>(knot_indices), std::get<3>(coeff_indices))[0],
          std::get<3>(spline_).template eval<deriv::dz, memory_optimized>(
              xi, std::get<3>(knot_indices), std::get<3>(coeff_indices))[0],
          std::get<3>(spline_).template eval<deriv::dt, memory_optimized>(
              xi, std::get<3>(knot_indices), std::get<3>(coeff_indices))[0]);
    }
  }
  /// @}

  //  clang-format off
  /// @brief Returns a block-tensor with the Laplacian of the function space
  /// object in the points `xi` with respect to the parametric variables
  ///
  /// @param[in] xi Point(s) where to evaluate the Laplacian
  ///
  /// @result Block-tensor with the Laplacian with respect to the
  /// parametric variables `xi`
  /// \f[
  ///     L_{\boldsymbol{\xi}}(u)
  ///        =
  ///     \sum_{i,j=0\atop|i+j|=2}^2
  ///     \frac{\partial^2 u}{\partial \xi_i\partial \xi_{j}}
  /// \f]
  ///
  /// @note If the function space object has geometric dimension larger
  /// then one then all Laplacians are returned as a vector.
  //  clang-format on
  ///
  /// @{
  template <functionspace comp = functionspace::interior,
            bool memory_optimized = false>
  inline auto lapl(const utils::TensorArray1 &xi,
                   const std::tuple<utils::TensorArray1> &knot_indices,
                   const std::tuple<torch::Tensor> &coeff_indices) const {

    assert(xi[0].sizes() == std::get<0>(knot_indices)[0].sizes());

    if constexpr (comp == functionspace::interior) {
      static_assert(std::tuple_element_t<0, spline_type>::geoDim() == 1,
                    "lapl(.) for vector-valued spaces requires 1D variables");

      return utils::BlockTensor<torch::Tensor, 1, 1>(
          std::get<0>(spline_).template eval<deriv::dx ^ 2, memory_optimized>(
              xi, std::get<0>(knot_indices), std::get<0>(coeff_indices))[0]);
    }
  }

  template <functionspace comp = functionspace::interior,
            bool memory_optimized = false>
  inline auto
  lapl(const utils::TensorArray2 &xi,
       const std::tuple<utils::TensorArray2, utils::TensorArray2> &knot_indices,
       const std::tuple<torch::Tensor, torch::Tensor> &coeff_indices) const {

    assert(xi[0].sizes() == std::get<0>(knot_indices)[0].sizes() &&
           xi[1].sizes() == std::get<1>(knot_indices)[0].sizes() &&
           xi[0].sizes() == xi[1].sizes());

    if constexpr (comp == functionspace::interior) {
      static_assert(std::tuple_element_t<0, spline_type>::geoDim() == 1 &&
                        std::tuple_element_t<1, spline_type>::geoDim() == 1,
                    "lapl(.) for vector-valued spaces requires 1D variables");

      return utils::BlockTensor<torch::Tensor, 1, 1>(
          *std::get<0>(spline_).template eval<deriv::dx ^ 2, memory_optimized>(
              xi, std::get<0>(knot_indices), std::get<0>(coeff_indices))[0] +
          *std::get<1>(spline_).template eval<deriv::dy ^ 2, memory_optimized>(
              xi, std::get<1>(knot_indices), std::get<1>(coeff_indices))[0]);
    }
  }

  template <functionspace comp = functionspace::interior,
            bool memory_optimized = false>
  inline auto lapl(const utils::TensorArray3 &xi,
                   const std::tuple<utils::TensorArray3, utils::TensorArray3,
                                    utils::TensorArray3> &knot_indices,
                   const std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
                       &coeff_indices) const {

    assert(xi[0].sizes() == std::get<0>(knot_indices)[0].sizes() &&
           xi[1].sizes() == std::get<1>(knot_indices)[0].sizes() &&
           xi[2].sizes() == std::get<2>(knot_indices)[0].sizes() &&
           xi[0].sizes() == xi[1].sizes() && xi[1].sizes() == xi[2].sizes());

    if constexpr (comp == functionspace::interior) {
      static_assert(std::tuple_element_t<0, spline_type>::geoDim() == 1 &&
                        std::tuple_element_t<1, spline_type>::geoDim() == 1 &&
                        std::tuple_element_t<2, spline_type>::geoDim() == 1,
                    "div(.) for vector-valued spaces requires 1D variables");

      return utils::BlockTensor<torch::Tensor, 1, 1>(
          *std::get<0>(spline_).template eval<deriv::dx ^ 2, memory_optimized>(
              xi, std::get<0>(knot_indices), std::get<0>(coeff_indices))[0] +
          *std::get<1>(spline_).template eval<deriv::dy ^ 2, memory_optimized>(
              xi, std::get<1>(knot_indices), std::get<1>(coeff_indices))[0] +
          *std::get<2>(spline_).template eval<deriv::dz ^ 2, memory_optimized>(
              xi, std::get<2>(knot_indices), std::get<2>(coeff_indices))[0]);
    }
  }

  template <functionspace comp = functionspace::interior,
            bool memory_optimized = false>
  inline auto
  lapl(const utils::TensorArray4 &xi,
       const std::tuple<utils::TensorArray4, utils::TensorArray4,
                        utils::TensorArray4, utils::TensorArray4> &knot_indices,
       const std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
                        torch::Tensor> &coeff_indices) const {

    assert(xi[0].sizes() == std::get<0>(knot_indices)[0].sizes() &&
           xi[1].sizes() == std::get<1>(knot_indices)[0].sizes() &&
           xi[2].sizes() == std::get<2>(knot_indices)[0].sizes() &&
           xi[3].sizes() == std::get<3>(knot_indices)[0].sizes() &&
           xi[0].sizes() == xi[1].sizes() && xi[1].sizes() == xi[2].sizes() &&
           xi[2].sizes() == xi[3].sizes());

    if constexpr (comp == functionspace::interior) {
      static_assert(std::tuple_element_t<0, spline_type>::geoDim() == 1 &&
                        std::tuple_element_t<1, spline_type>::geoDim() == 1 &&
                        std::tuple_element_t<2, spline_type>::geoDim() == 1 &&
                        std::tuple_element_t<3, spline_type>::geoDim() == 1,
                    "div(.) for vector-valued spaces requires 1D variables");

      return utils::BlockTensor<torch::Tensor, 1, 1>(
          *std::get<0>(spline_).template eval<deriv::dx ^ 2, memory_optimized>(
              xi, std::get<0>(knot_indices), std::get<0>(coeff_indices))[0] +
          *std::get<1>(spline_).template eval<deriv::dy ^ 2, memory_optimized>(
              xi, std::get<1>(knot_indices), std::get<1>(coeff_indices))[0] +
          *std::get<2>(spline_).template eval<deriv::dz ^ 2, memory_optimized>(
              xi, std::get<2>(knot_indices), std::get<2>(coeff_indices))[0] +
          *std::get<3>(spline_).template eval<deriv::dt ^ 2, memory_optimized>(
              xi, std::get<3>(knot_indices), std::get<3>(coeff_indices))[0]);
    }
  }
  /// @}

#define GENERATE_EXPR_MACRO(r, data, name)                                     \
private:                                                                       \
  template <functionspace comp = functionspace::interior,                      \
            bool memory_optimized = false, std::size_t... Is, typename... Xi>  \
  inline auto BOOST_PP_CAT(name, _all_)(std::index_sequence<Is...>,            \
                                        const std::tuple<Xi...> &xi) const {   \
    if constexpr (comp == functionspace::interior)                             \
      return std::tuple(std::get<Is>(spline_).template name<memory_optimized>( \
          std::get<Is>(xi))...);                                               \
    else if constexpr (comp == functionspace::boundary)                        \
      return std::tuple(                                                       \
          std::get<Is>(boundary_).template name<memory_optimized>(             \
              std::get<Is>(xi))...);                                           \
  }                                                                            \
                                                                               \
  template <functionspace comp = functionspace::interior,                      \
            bool memory_optimized = false, std::size_t... Is, typename... Xi,  \
            typename... Knot_Indices>                                          \
  inline auto BOOST_PP_CAT(name, _all_)(                                       \
      std::index_sequence<Is...>, const std::tuple<Xi...> &xi,                 \
      const std::tuple<Knot_Indices...> &knot_indices) const {                 \
    if constexpr (comp == functionspace::interior)                             \
      return std::tuple(std::get<Is>(spline_).template name<memory_optimized>( \
          std::get<Is>(xi), std::get<Is>(knot_indices))...);                   \
    else if constexpr (comp == functionspace::boundary)                        \
      return std::tuple(                                                       \
          std::get<Is>(boundary_).template name<memory_optimized>(             \
              std::get<Is>(xi), std::get<Is>(knot_indices))...);               \
  }                                                                            \
                                                                               \
  template <functionspace comp = functionspace::interior,                      \
            bool memory_optimized = false, std::size_t... Is, typename... Xi,  \
            typename... Knot_Indices, typename... Coeff_Indices>               \
  inline auto BOOST_PP_CAT(name, _all_)(                                       \
      std::index_sequence<Is...>, const std::tuple<Xi...> &xi,                 \
      const std::tuple<Knot_Indices...> &knot_indices,                         \
      const std::tuple<Coeff_Indices...> &coeff_indices) const {               \
    if constexpr (comp == functionspace::interior)                             \
      return std::tuple(std::get<Is>(spline_).template name<memory_optimized>( \
          std::get<Is>(xi), std::get<Is>(knot_indices),                        \
          std::get<Is>(coeff_indices))...);                                    \
    else if constexpr (comp == functionspace::boundary)                        \
      return std::tuple(                                                       \
          std::get<Is>(boundary_).template name<memory_optimized>(             \
              std::get<Is>(xi), std::get<Is>(knot_indices),                    \
              std::get<Is>(coeff_indices))...);                                \
  }                                                                            \
                                                                               \
  template <functionspace comp = functionspace::interior,                      \
            bool memory_optimized = false, std::size_t... Is, std::size_t N>   \
  inline auto BOOST_PP_CAT(name, _)(std::index_sequence<Is...>,                \
                                    const utils::TensorArray<N> &xi) const {   \
    if constexpr (comp == functionspace::interior)                             \
      return name<comp, memory_optimized>(                                     \
          xi, std::tuple(std::get<Is>(spline_).find_knot_indices(xi)...));     \
    else if constexpr (comp == functionspace::boundary)                        \
      return name<comp, memory_optimized>(                                     \
          xi, std::tuple(std::get<Is>(boundary_).find_knot_indices(xi)...));   \
  }                                                                            \
                                                                               \
  template <functionspace comp = functionspace::interior,                      \
            bool memory_optimized = false, std::size_t... Is, std::size_t N,   \
            typename... Knot_Indices>                                          \
  inline auto BOOST_PP_CAT(name, _)(                                           \
      std::index_sequence<Is...>, const utils::TensorArray<N> &xi,             \
      const std::tuple<Knot_Indices...> &knot_indices) const {                 \
    if constexpr (comp == functionspace::interior)                             \
      return name<comp, memory_optimized>(                                     \
          xi, knot_indices,                                                    \
          std::tuple(std::get<Is>(spline_).find_coeff_indices(                 \
              std::get<Is>(knot_indices))...));                                \
    else if constexpr (comp == functionspace::boundary)                        \
      return name<comp, memory_optimized>(                                     \
          xi, knot_indices,                                                    \
          std::tuple(std::get<Is>(boundary_).find_coeff_indices(               \
              std::get<Is>(knot_indices))...));                                \
  }                                                                            \
                                                                               \
public:                                                                        \
  template <functionspace comp = functionspace::interior,                      \
            bool memory_optimized = false, typename... Args>                   \
  inline auto BOOST_PP_CAT(name, _all)(const Args &...args) const {            \
    return BOOST_PP_CAT(name, _all_)<comp, memory_optimized>(                  \
        std::make_index_sequence<FunctionSpace::nspaces()>{}, args...);        \
  }                                                                            \
                                                                               \
  template <functionspace comp = functionspace::interior,                      \
            bool memory_optimized = false>                                     \
  inline auto name(const torch::Tensor &xi) const {                            \
    return name<comp, memory_optimized>(utils::TensorArray1({xi}));            \
  }                                                                            \
                                                                               \
  template <functionspace comp = functionspace::interior,                      \
            bool memory_optimized = false, std::size_t N>                      \
  inline auto name(const utils::TensorArray<N> &xi) const {                    \
    return BOOST_PP_CAT(name, _)<comp, memory_optimized>(                      \
        std::make_index_sequence<FunctionSpace::nspaces()>{}, xi);             \
  }                                                                            \
                                                                               \
  template <functionspace comp = functionspace::interior,                      \
            bool memory_optimized = false, std::size_t N,                      \
            typename... Knot_Indices>                                          \
  inline auto name(const utils::TensorArray<N> &xi,                            \
                   const std::tuple<Knot_Indices...> &knot_indices) const {    \
    return BOOST_PP_CAT(name, _)<comp, memory_optimized>(                      \
        std::make_index_sequence<FunctionSpace::nspaces()>{}, xi,              \
        knot_indices);                                                         \
  }

  /// @brief Auto-generated functions
  /// @{
  BOOST_PP_SEQ_FOR_EACH(GENERATE_EXPR_MACRO, _, GENERATE_EXPR_SEQ)
/// @}
#undef GENERATE_EXPR_MACRO

#define GENERATE_IEXPR_MACRO(r, data, name)                                    \
private:                                                                       \
  template <functionspace comp = functionspace::interior,                      \
            bool memory_optimized = false, std::size_t... Is,                  \
            typename Geometry, typename... Xi>                                 \
  inline auto BOOST_PP_CAT(name, _all_)(std::index_sequence<Is...>,            \
                                        const Geometry &G,                     \
                                        const std::tuple<Xi...> &xi) const {   \
    if constexpr (comp == functionspace::interior) {                           \
      if constexpr (Geometry::nspaces() == 1)                                  \
        return std::tuple(                                                     \
            std::get<Is>(spline_).template name<memory_optimized>(             \
                G.space(), std::get<Is>(xi))...);                              \
      else if constexpr (Geometry::nspaces() == nspaces())                     \
        return std::tuple(                                                     \
            std::get<Is>(spline_).template name<memory_optimized>(             \
                G.template space<Is>(), std::get<Is>(xi))...);                 \
    } else if constexpr (comp == functionspace::boundary) {                    \
      if constexpr (Geometry::nboundaries() == 1)                              \
        return std::tuple(                                                     \
            std::get<Is>(boundary_).template name<memory_optimized>(           \
                static_cast<typename Geometry::boundary_type::boundary_type>(  \
                    G.boundary().coeffs()),                                    \
                std::get<Is>(xi))...);                                         \
      else if constexpr (Geometry::nboundaries() == nboundaries())             \
        return std::tuple(                                                     \
            std::get<Is>(boundary_).template name<memory_optimized>(           \
                G.template boundary<Is>().coeffs(), std::get<Is>(xi))...);     \
    }                                                                          \
  }                                                                            \
                                                                               \
  template <functionspace comp = functionspace::interior,                      \
            bool memory_optimized = false, std::size_t... Is,                  \
            typename Geometry, typename... Xi, typename... Knot_Indices,       \
            typename... Knot_Indices_G>                                        \
  inline auto BOOST_PP_CAT(name, _all_)(                                       \
      std::index_sequence<Is...>, const Geometry &G,                           \
      const std::tuple<Xi...> &xi,                                             \
      const std::tuple<Knot_Indices...> &knot_indices,                         \
      const std::tuple<Knot_Indices_G...> &knot_indices_G) const {             \
    if constexpr (comp == functionspace::interior) {                           \
      if constexpr (Geometry::nspaces() == 1)                                  \
        return std::tuple(                                                     \
            std::get<Is>(spline_).template name<memory_optimized>(             \
                G.space(), std::get<Is>(xi), std::get<Is>(knot_indices),       \
                std::get<Is>(knot_indices_G))...);                             \
      else                                                                     \
        return std::tuple(                                                     \
            std::get<Is>(spline_).template name<memory_optimized>(             \
                std::get<Is>(G), std::get<Is>(xi), std::get<Is>(knot_indices), \
                std::get<Is>(knot_indices_G))...);                             \
    } else if constexpr (comp == functionspace::boundary) {                    \
      if constexpr (Geometry::nspaces() == 1)                                  \
        return std::tuple(                                                     \
            std::get<Is>(boundary_).template name<memory_optimized>(           \
                static_cast<typename Geometry::boundary_type::boundary_type>(  \
                    G.boundary().coeffs()),                                    \
                std::get<Is>(xi), std::get<Is>(knot_indices),                  \
                std::get<Is>(knot_indices_G))...);                             \
      else                                                                     \
        return std::tuple(                                                     \
            std::get<Is>(boundary_).template name<memory_optimized>(           \
                std::get<Is>(G).boundary().coeffs(), std::get<Is>(xi),         \
                std::get<Is>(knot_indices), std::get<Is>(knot_indices_G))...); \
    }                                                                          \
  }                                                                            \
                                                                               \
  template <functionspace comp = functionspace::interior,                      \
            bool memory_optimized = false, std::size_t... Is,                  \
            typename Geometry, typename... Xi, typename... Knot_Indices,       \
            typename... Coeff_Indices, typename... Knot_Indices_G,             \
            typename... Coeff_Indices_G>                                       \
  inline auto BOOST_PP_CAT(name, _all_)(                                       \
      std::index_sequence<Is...>, const Geometry &G,                           \
      const std::tuple<Xi...> &xi,                                             \
      const std::tuple<Knot_Indices...> &knot_indices,                         \
      const std::tuple<Coeff_Indices...> &coeff_indices,                       \
      const std::tuple<Knot_Indices_G...> &knot_indices_G,                     \
      const std::tuple<Coeff_Indices_G...> &coeff_indices_G) const {           \
    if constexpr (comp == functionspace::interior) {                           \
      if constexpr (Geometry::nspaces() == 1)                                  \
        return std::tuple(                                                     \
            std::get<Is>(spline_).template name<memory_optimized>(             \
                G.space(), std::get<Is>(xi), std::get<Is>(knot_indices),       \
                std::get<Is>(coeff_indices), std::get<Is>(knot_indices_G),     \
                std::get<Is>(coeff_indices_G))...);                            \
      else                                                                     \
        return std::tuple(                                                     \
            std::get<Is>(spline_).template name<memory_optimized>(             \
                std::get<Is>(G), std::get<Is>(xi), std::get<Is>(knot_indices), \
                std::get<Is>(coeff_indices), std::get<Is>(knot_indices_G),     \
                std::get<Is>(coeff_indices_G))...);                            \
    } else if constexpr (comp == functionspace::boundary) {                    \
      if constexpr (Geometry::nspaces() == 1)                                  \
        return std::tuple(                                                     \
            std::get<Is>(boundary_).template name<memory_optimized>(           \
                static_cast<typename Geometry::boundary_type::boundary_type>(  \
                    G.boundary().coeffs()),                                    \
                std::get<Is>(xi), std::get<Is>(knot_indices),                  \
                std::get<Is>(coeff_indices), std::get<Is>(knot_indices_G),     \
                std::get<Is>(coeff_indices_G))...);                            \
      else                                                                     \
        return std::tuple(                                                     \
            std::get<Is>(boundary_).template name<memory_optimized>(           \
                std::get<Is>(G).boundary().coeffs(), std::get<Is>(xi),         \
                std::get<Is>(knot_indices), std::get<Is>(coeff_indices),       \
                std::get<Is>(knot_indices_G),                                  \
                std::get<Is>(coeff_indices_G))...);                            \
    }                                                                          \
  }                                                                            \
                                                                               \
public:                                                                        \
  template <functionspace comp = functionspace::interior,                      \
            bool memory_optimized = false, typename... Args>                   \
  inline auto BOOST_PP_CAT(name, _all)(const Args &...args) const {            \
    return BOOST_PP_CAT(name, _all_)<comp, memory_optimized>(                  \
        std::make_index_sequence<FunctionSpace::nspaces()>{}, args...);        \
  }

  /// @brief Auto-generated functions
  /// @{
  BOOST_PP_SEQ_FOR_EACH(GENERATE_IEXPR_MACRO, _, GENERATE_IEXPR_SEQ)
/// @}
#undef GENERATE_IEXPR_MACRO
};

/// @brief Print (as string) a function space object
template <typename... Splines>
inline std::ostream &operator<<(std::ostream &os,
                                const FunctionSpace<Splines...> &obj) {
  obj.pretty_print(os);
  return os;
}

/// @brief Function space
///
/// @note This class is not meant for direct use in
/// applications. Instead use S, TH, NE, or RT.
template <typename Spline, typename Boundary>
class FunctionSpace : public utils::Serializable,
                      private utils::FullQualifiedName {

  static_assert(is_SplineType_v<Spline>, "Spline must be a valid SplineType");
  static_assert(is_BoundaryType_v<Boundary>,
                "Boundary must be a valid BoundaryType");

public:
  /// @brief Value type
  using value_type = typename Spline::value_type;

  /// @brief Spline type
  using spline_type = Spline;

  /// @brief Spline evaluation type
  using eval_type = utils::TensorArray<Spline::parDim()>;

  /// @brief Boundary type
  using boundary_type = Boundary;

  /// @brief Boundary evaluation type
  using boundary_eval_type = typename Boundary::eval_type;

protected:
  /// @brief Spline
  spline_type spline_;

  /// @brief Boundary
  boundary_type boundary_;

public:
  /// @brief Default constructor
  FunctionSpace() = default;

  /// @brief Copy constructor
  FunctionSpace(const FunctionSpace &) = default;

  /// @brief Move constructor
  FunctionSpace(FunctionSpace &&) = default;

  /// @brief Constructor
  /// @{
  FunctionSpace(const std::array<int64_t, Spline::parDim()> &ncoeffs,
                enum init init = init::greville,
                Options<value_type> options = iganet::Options<value_type>{})
      : spline_(ncoeffs, init, options),
        boundary_(ncoeffs, init::none, options) {
    boundary_.from_full_tensor(spline_.as_tensor());
  }

  FunctionSpace(std::array<std::vector<value_type>, Spline::parDim()> kv,
                enum init init = init::greville,
                Options<value_type> options = iganet::Options<value_type>{})
      : spline_(kv, init, options), boundary_(kv, init::none, options) {
    static_assert(Spline::is_nonuniform(),
                  "Constructor is only available for non-uniform splines");
    boundary_.from_full_tensor(spline_.as_tensor());
  }

  explicit FunctionSpace(const Spline &spline)
      : spline_(spline),
        boundary_(spline.ncoeffs(), init::none, spline.options()) {
    boundary_.from_full_tensor(spline_.as_tensor());
  }

  explicit FunctionSpace(Spline &&spline)
      : spline_(spline),
        boundary_(spline.ncoeffs(), init::none, spline.options()) {
    boundary_.from_full_tensor(spline_.as_tensor());
  }
  /// @}

  /// @brief Returns the number of function spaces
  inline static constexpr short_t nspaces() noexcept { return 1; }

  /// @brief Returns the number of boundaries
  inline static constexpr short_t nboundaries() noexcept { return 1; }

  /// @brief Returns a constant reference to the \f$s\f$-th function space
  template <short_t s = 0>
  inline constexpr const spline_type &space() const noexcept {
    static_assert(s >= 0 && s < nspaces());
    return spline_;
  }

  /// @brief Returns a non-constant reference to the \f$s\f$-th function space
  template <short_t s = 0> inline constexpr spline_type &space() noexcept {
    static_assert(s >= 0 && s < nspaces());
    return spline_;
  }

  /// @brief Returns a constant reference to the \f$s\f$-th boundary object
  template <short_t s = 0>
  inline constexpr const boundary_type &boundary() const noexcept {
    static_assert(s >= 0 && s < nboundaries());
    return boundary_;
  }

  /// @brief Returns a non-constant reference to the \f$s\f$-th boundary object
  /// object
  template <short_t s = 0> inline constexpr boundary_type &boundary() noexcept {
    static_assert(s >= 0 && s < nboundaries());
    return boundary_;
  }

  /// @brief Returns a clone of the function space
  inline constexpr FunctionSpace clone() const noexcept {
    return FunctionSpace(*this);
  }

  /// @brief Returns a subset of the tuple of function spaces
  template <short_t... s> inline constexpr auto clone() const noexcept {

    static_assert(((s >= 0 && s < nspaces()) && ... && true));

    if constexpr (sizeof...(s) == 1)
      return FunctionSpace(*this);
    else
      return FunctionSpace<
          std::tuple<std::tuple_element_t<s, std::tuple<spline_type>>...>,
          std::tuple<std::tuple_element_t<s, std::tuple<boundary_type>>...>>(
          std::get<s>(std::make_tuple(spline_))...,
          std::get<s>(std::make_tuple(boundary_))...);
  }

  /// @brief Returns a single-tensor representation of the space
  virtual inline torch::Tensor spaces_as_tensor() const noexcept {
    return spline_.as_tensor();
  }

  /// @brief Returns a single-tensor representation of the boundary
  virtual inline torch::Tensor boundary_as_tensor() const noexcept {
    return boundary_.as_tensor();
  }

  /// @brief Returns a single-tensor representation of the
  /// function space object
  ///
  /// @note The default implementation behaves identical to
  /// spaces_as_tensor() but can be overridden in a derived class
  virtual inline torch::Tensor as_tensor() const noexcept {
    return spaces_as_tensor();
  }

  /// @brief Returns the size of the single-tensor representation of
  /// the space
  virtual inline int64_t spaces_as_tensor_size() const noexcept {
    return spline_.as_tensor_size();
  }

  /// @brief Returns the size of the single-tensor representation of
  /// the boundary
  virtual inline int64_t boundary_as_tensor_size() const noexcept {
    return boundary_.as_tensor_size();
  }

  /// @brief Returns the size of the single-tensor representation of
  /// the function space object
  ///
  /// @note The default implementation behaves identical to
  /// spaces_as_tensor_size() but can be overridden in a derived class
  virtual inline int64_t as_tensor_size() const noexcept {
    return spaces_as_tensor_size();
  }

  /// @brief Sets the space from a single-tensor representation
  virtual inline FunctionSpace &
  spaces_from_tensor(const torch::Tensor &coeffs) noexcept {
    spline_.from_tensor(coeffs);
    return *this;
  }

  /// @brief Sets the boundary from a single-tensor representation of the
  /// boundary only
  virtual inline FunctionSpace &
  boundary_from_tensor(const torch::Tensor &coeffs) noexcept {
    boundary_.from_tensor(coeffs);
    return *this;
  }

  /// @brief Sets the boundary from a single-tensor representation
  virtual inline FunctionSpace &
  boundary_from_full_tensor(const torch::Tensor &coeffs) noexcept {
    boundary_.from_full_tensor(coeffs);
    return *this;
  }

  /// @brief Sets the function space object from a single-tensor representation
  inline FunctionSpace &from_tensor(const torch::Tensor &coeffs) noexcept {
    spline_.from_tensor(coeffs);
    boundary_.from_full_tensor(coeffs);
    return *this;
  }

  /// @brief Returns the function space object as XML object
  inline pugi::xml_document to_xml(int id = 0, std::string label = "") const {
    pugi::xml_document doc;
    pugi::xml_node root = doc.append_child("xml");
    to_xml(root, id, label);

    return doc;
  }

  /// @brief Returns the function space object as XML node
  inline pugi::xml_node &to_xml(pugi::xml_node &root, int id = 0,
                                std::string label = "") const {
    return spline_.to_xml(root, id, label);
  }

  /// @brief Updates the function space object from XML object
  inline FunctionSpace &from_xml(const pugi::xml_document &doc, int id = 0,
                                 std::string label = "") {
    return from_xml(doc.child("xml"), id, label);
  }

  /// @brief Updates the function space object from XML node
  inline FunctionSpace &from_xml(const pugi::xml_node &root, int id = 0,
                                 std::string label = "") {
    spline_.from_xml(root, id, label);
    return *this;
  }

  /// @brief Serialization to JSON
  nlohmann::json to_json() const override {
    auto json = nlohmann::json::array();
    json.push_back(spline_.to_json());
    json.push_back(boundary_.to_json());
    return json;
  }

  /// @brief Transforms the coefficients based on the given mapping
  inline FunctionSpace &transform(
      const std::function<std::array<typename Spline::value_type,
                                     Spline::geoDim()>(
          const std::array<typename Spline::value_type, Spline::parDim()> &)>
          transformation) {
    spline_.transform(transformation);
    return *this;
  }

private:
  /// @brief Returns the values of the spline object in the points `xi`
  /// @{
  template <functionspace comp = functionspace::interior,
            deriv deriv = deriv::func, bool memory_optimized = false,
            std::size_t... Is, typename... Xi>
  inline auto eval_(std::index_sequence<Is...>,
                    const std::tuple<Xi...> &xi) const {
    if constexpr (comp == functionspace::interior)
      return std::tuple(
          spline_.template eval<deriv, memory_optimized>(std::get<Is>(xi))...);
    else if constexpr (comp == functionspace::boundary)
      return std::tuple(boundary_.template eval<deriv, memory_optimized>(
          std::get<Is>(xi))...);
  }

  template <functionspace comp = functionspace::interior,
            deriv deriv = deriv::func, bool memory_optimized = false,
            std::size_t... Is, typename... Xi, typename... Knot_Indices>
  inline auto eval_(std::index_sequence<Is...>, const std::tuple<Xi...> &xi,
                    const std::tuple<Knot_Indices...> &knot_indices) const {
    if constexpr (comp == functionspace::interior)
      return std::tuple(spline_.template eval<deriv, memory_optimized>(
          std::get<Is>(xi), std::get<Is>(knot_indices))...);
    else if constexpr (comp == functionspace::boundary)
      return std::tuple(boundary_.template eval<deriv, memory_optimized>(
          std::get<Is>(xi), std::get<Is>(knot_indices))...);
  }

  template <functionspace comp = functionspace::interior,
            deriv deriv = deriv::func, bool memory_optimized = false,
            std::size_t... Is, typename... Xi, typename... Knot_Indices,
            typename... Coeff_Indices>
  inline auto eval_(std::index_sequence<Is...>, const std::tuple<Xi...> &xi,
                    const std::tuple<Knot_Indices...> &knot_indices,
                    const std::tuple<Coeff_Indices...> &coeff_indices) const {
    if constexpr (comp == functionspace::interior)
      return std::tuple(spline_.template eval<deriv, memory_optimized>(
          std::get<Is>(xi), std::get<Is>(knot_indices),
          std::get<Is>(coeff_indices))...);
    else if constexpr (comp == functionspace::boundary)
      return std::tuple(boundary_.template eval<deriv, memory_optimized>(
          std::get<Is>(xi), std::get<Is>(knot_indices),
          std::get<Is>(coeff_indices))...);
  }
  /// @}

public:
  /// @brief Returns the values of the spline object in the points `xi`
  template <functionspace comp = functionspace::interior,
            deriv deriv = deriv::func, bool memory_optimized = false,
            typename Arg, typename... Args>
  inline auto eval(const Arg &arg, const Args &...args) const {
    if constexpr (comp == functionspace::interior)
      if constexpr (utils::is_tuple_v<Arg>)
        return eval_<comp, deriv, memory_optimized>(
            std::make_index_sequence<std::tuple_size_v<Arg>>{}, arg, args...);
      else
        return spline_.template eval<deriv, memory_optimized>(arg, args...);
    else if constexpr (comp == functionspace::boundary) {
      if constexpr (utils::is_tuple_of_tuples_v<Arg>)
        return eval_<comp, deriv, memory_optimized>(
            std::make_index_sequence<std::tuple_size_v<Arg>>{}, arg, args...);
      else
        return boundary_.template eval<deriv, memory_optimized>(arg, args...);
    }
  }

  /// @brief Returns the value of the spline object from
  /// precomputed basis function
  template <functionspace comp = functionspace::interior, typename... Args>
  inline auto eval_from_precomputed(const Args &...args) const {
    if constexpr (comp == functionspace::interior)
      return spline_.eval_from_precomputed(args...);
    else if constexpr (comp == functionspace::boundary)
      return boundary_.eval_from_precomputed(args...);
  }

private:
  /// @brief Returns the knot indicies of knot spans containing `xi`
  template <functionspace comp = functionspace::interior, std::size_t... Is,
            typename Xi>
  inline auto find_knot_indices_(std::index_sequence<Is...>,
                                 const Xi &xi) const {
    if constexpr (comp == functionspace::interior)
      return std::tuple(spline_.find_knot_indices(std::get<Is>(xi))...);
    else
      return std::tuple(boundary_.find_knot_indices(std::get<Is>(xi))...);
  }

public:
  /// @brief Returns the knot indicies of knot spans containing `xi`
  template <functionspace comp = functionspace::interior, typename Xi>
  inline auto find_knot_indices(const Xi &xi) const {
    if constexpr (comp == functionspace::interior)
      if constexpr (utils::is_tuple_v<Xi>)
        return find_knot_indices_<comp>(
            std::make_index_sequence<std::tuple_size_v<Xi>>{}, xi);
      else
        return spline_.find_knot_indices(xi);
    else if constexpr (comp == functionspace::boundary) {
      if constexpr (utils::is_tuple_of_tuples_v<Xi>)
        return find_knot_indices_<comp>(
            std::make_index_sequence<std::tuple_size_v<Xi>>{}, xi);
      else
        return boundary_.find_knot_indices(xi);
    }
  }

  /// @brief Returns the values of the spline objects' basis
  /// functions in the points `xi`
  template <functionspace comp = functionspace::interior,
            deriv deriv = deriv::func, bool memory_optimized = false,
            typename... Args>
  inline auto eval_basfunc(const Args &...args) const {
    if constexpr (comp == functionspace::interior)
      return spline_.template eval_basfunc<deriv, memory_optimized>(args...);
    else if constexpr (comp == functionspace::boundary)
      return boundary_.template eval_basfunc<deriv, memory_optimized>(args...);
  }

private:
  /// @brief Returns the indices of the spline objects'
  /// coefficients corresponding to the knot indices `indices`
  template <functionspace comp = functionspace::interior,
            bool memory_optimized = false, std::size_t... Is,
            typename Knot_Indices>
  inline auto find_coeff_indices_(std::index_sequence<Is...>,
                                  const Knot_Indices &knot_indices) const {
    if constexpr (comp == functionspace::interior)
      return std::tuple(spline_.template find_coeff_indices<memory_optimized>(
          std::get<Is>(knot_indices))...);
    else
      return std::tuple(boundary_.template find_coeff_indices<memory_optimized>(
          std::get<Is>(knot_indices))...);
  }

public:
  /// @brief Returns the indices of the spline objects'
  /// coefficients corresponding to the knot indices `indices`
  template <functionspace comp = functionspace::interior,
            bool memory_optimized = false, typename Knot_Indices>
  inline auto find_coeff_indices(const Knot_Indices &knot_indices) const {
    if constexpr (comp == functionspace::interior)
      if constexpr (utils::is_tuple_v<Knot_Indices>)
        return find_coeff_indices_<comp, memory_optimized>(
            std::make_index_sequence<std::tuple_size_v<Knot_Indices>>{},
            knot_indices);
      else
        return spline_.template find_coeff_indices<memory_optimized>(
            knot_indices);
    else if constexpr (comp == functionspace::boundary) {
      if constexpr (utils::is_tuple_of_tuples_v<Knot_Indices>)
        return find_coeff_indices_<comp, memory_optimized>(
            std::make_index_sequence<std::tuple_size_v<Knot_Indices>>{},
            knot_indices);
      else
        return boundary_.template find_coeff_indices<memory_optimized>(
            knot_indices);
    }
  }

  /// @brief Returns the spline objects with uniformly refined
  /// knot and coefficient vectors
  inline auto &uniform_refine(int numRefine = 1, int dimRefine = -1) {
    spline_.uniform_refine(numRefine, dimRefine);
    boundary_.uniform_refine(numRefine, dimRefine);
    return *this;
  }

  /// @brief Returns a copy of the function space object with settings from
  /// options
  template <typename real_t> inline auto to(Options<real_t> options) const {
    return FunctionSpace<
        typename spline_type::template real_derived_self_type<real_t>,
        typename boundary_type::template real_derived_self_type<real_t>>(
        spline_.to(options), boundary_.to(options));
  }

  /// @brief Returns a copy of the function space object with settings from
  /// device
  inline auto to(torch::Device device) const {
    return FunctionSpace(spline_.to(device), boundary_.to(device));
  }

  /// @brief Returns a copy of the function space object with real_t type
  template <typename real_t> inline auto to() const {
    return FunctionSpace<
        typename spline_type::template real_derived_self_type<real_t>,
        typename boundary_type::template real_derived_self_type<real_t>>(
        spline_.template to<real_t>(), boundary_.template to<real_t>());
  }

  /// @brief Scales the function space object by a scalar
  inline auto scale(value_type s, int dim = -1) {
    spline_.scale(s, dim);
    boundary_.from_full_tensor(spline_.as_tensor());
    return *this;
  }

  /// @brief Scales the function space object by a vector
  template <size_t N> inline auto scale(std::array<value_type, N> v) {
    spline_.scale(v);
    boundary_.from_full_tensor(spline_.as_tensor());
    return *this;
  }

  /// @brief Translates the function space object by a vector
  template <size_t N> inline auto translate(std::array<value_type, N> v) {
    spline_.translate(v);
    boundary_.from_full_tensor(spline_.as_tensor());
    return *this;
  }

  /// @brief Rotates the function space object by an angle in 2d
  inline auto rotate(value_type angle) {
    spline_.rotate(angle);
    boundary_.from_full_tensor(spline_.as_tensor());
    return *this;
  }

  /// @brief Rotates the function space object by three angles in 3d
  inline auto rotate(std::array<value_type, 3> angle) {
    spline_.rotate(angle);
    boundary_.from_full_tensor(spline_.as_tensor());
    return *this;
  }

  /// @brief Writes the function space object into a
  /// torch::serialize::OutputArchive object
  inline torch::serialize::OutputArchive &
  write(torch::serialize::OutputArchive &archive,
        const std::string &key = "functionspace") const {
    spline_.write(archive, key);
    boundary_.write(archive, key);
    return archive;
  }

  /// @brief Loads the function space object from a
  /// torch::serialize::InputArchive object
  inline torch::serialize::InputArchive &
  read(torch::serialize::InputArchive &archive,
       const std::string &key = "functionspace") {
    spline_.read(archive, key);
    boundary_.read(archive, key);
    return archive;
  }

  /// @brief Returns a string representation of the function space object
  inline virtual void
  pretty_print(std::ostream &os = Log(log::info)) const noexcept override {
    os << name() << "(\nspline = ";
    spline_.pretty_print(os);
    os << "\nboundary = ";
    boundary_.pretty_print(os);
    os << "\n)";
  }

#define GENERATE_EXPR_MACRO(r, data, name)                                     \
private:                                                                       \
  template <functionspace comp = functionspace::interior,                      \
            bool memory_optimized = false, std::size_t... Is, typename... Xi>  \
  inline auto BOOST_PP_CAT(name, _all_)(std::index_sequence<Is...>,            \
                                        const std::tuple<Xi...> &xi) const {   \
    if constexpr (comp == functionspace::interior)                             \
      return std::tuple(                                                       \
          spline_.template name<memory_optimized>(std::get<Is>(xi))...);       \
    else if constexpr (comp == functionspace::boundary)                        \
      return std::tuple(                                                       \
          boundary_.template name<memory_optimized>(std::get<Is>(xi))...);     \
  }                                                                            \
                                                                               \
  template <functionspace comp = functionspace::interior,                      \
            bool memory_optimized = false, std::size_t... Is, typename... Xi,  \
            typename... Knot_Indices>                                          \
  inline auto BOOST_PP_CAT(name, _all_)(                                       \
      std::index_sequence<Is...>, const std::tuple<Xi...> &xi,                 \
      const std::tuple<Knot_Indices...> &knot_indices) const {                 \
    if constexpr (comp == functionspace::interior)                             \
      return std::tuple(spline_.template name<memory_optimized>(               \
          std::get<Is>(xi), std::get<Is>(knot_indices))...);                   \
    else if constexpr (comp == functionspace::boundary)                        \
      return std::tuple(boundary_.template name<memory_optimized>(             \
          std::get<Is>(xi), std::get<Is>(knot_indices))...);                   \
  }                                                                            \
                                                                               \
  template <functionspace comp = functionspace::interior,                      \
            bool memory_optimized = false, std::size_t... Is, typename... Xi,  \
            typename... Knot_Indices, typename... Coeff_Indices>               \
  inline auto BOOST_PP_CAT(name, _all_)(                                       \
      std::index_sequence<Is...>, const std::tuple<Xi...> &xi,                 \
      const std::tuple<Knot_Indices...> &knot_indices,                         \
      const std::tuple<Coeff_Indices...> &coeff_indices) const {               \
    if constexpr (comp == functionspace::interior)                             \
      return std::tuple(spline_.template name<memory_optimized>(               \
          std::get<Is>(xi), std::get<Is>(knot_indices),                        \
          std::get<Is>(coeff_indices))...);                                    \
    else if constexpr (comp == functionspace::boundary)                        \
      return std::tuple(boundary_.template name<memory_optimized>(             \
          std::get<Is>(xi), std::get<Is>(knot_indices),                        \
          std::get<Is>(coeff_indices))...);                                    \
  }                                                                            \
                                                                               \
public:                                                                        \
  template <functionspace comp = functionspace::interior,                      \
            bool memory_optimized = false, typename Arg, typename... Args>     \
  inline auto name(const Arg &arg, const Args &...args) const {                \
    if constexpr (comp == functionspace::interior)                             \
      if constexpr (utils::is_tuple_v<Arg>)                                    \
        return BOOST_PP_CAT(name, _all_)<comp, memory_optimized>(              \
            std::make_index_sequence<std::tuple_size_v<Arg>>{}, arg, args...); \
      else                                                                     \
        return spline_.template name<memory_optimized>(arg, args...);          \
    else if constexpr (comp == functionspace::boundary) {                      \
      if constexpr (utils::is_tuple_of_tuples_v<Arg>)                          \
        return BOOST_PP_CAT(name, _all_)<comp, memory_optimized>(              \
            std::make_index_sequence<std::tuple_size_v<Arg>>{}, arg, args...); \
      else                                                                     \
        return boundary_.template name<memory_optimized>(arg, args...);        \
    }                                                                          \
  }

  /// @brief Auto-generated functions
  /// @{
  BOOST_PP_SEQ_FOR_EACH(GENERATE_EXPR_MACRO, _, GENERATE_EXPR_SEQ)
  /// @}
#undef GENERATE_EXPR_MACRO

#define GENERATE_IEXPR_MACRO(r, data, name)                                    \
private:                                                                       \
  template <functionspace comp = functionspace::interior,                      \
            bool memory_optimized = false, std::size_t... Is,                  \
            typename Geometry, typename... Xi>                                 \
  inline auto BOOST_PP_CAT(name, _all_)(std::index_sequence<Is...>,            \
                                        const Geometry &G,                     \
                                        const std::tuple<Xi...> &xi) const {   \
    if constexpr (comp == functionspace::interior)                             \
      return std::tuple(spline_.template name<memory_optimized>(               \
          G.space(), std::get<Is>(xi))...);                                    \
    else if constexpr (comp == functionspace::boundary)                        \
      return std::tuple(boundary_.template name<memory_optimized>(             \
          static_cast<typename Geometry::boundary_type::boundary_type>(        \
              G.boundary().coeffs()),                                          \
          std::get<Is>(xi))...);                                               \
  }                                                                            \
                                                                               \
  template <functionspace comp = functionspace::interior,                      \
            bool memory_optimized = false, std::size_t... Is,                  \
            typename Geometry, typename... Xi, typename... Knot_Indices,       \
            typename... Knot_Indices_G>                                        \
  inline auto BOOST_PP_CAT(name, _all_)(                                       \
      std::index_sequence<Is...>, const Geometry &G,                           \
      const std::tuple<Xi...> &xi,                                             \
      const std::tuple<Knot_Indices...> &knot_indices,                         \
      const std::tuple<Knot_Indices_G...> &knot_indices_G) const {             \
    if constexpr (comp == functionspace::interior)                             \
      return std::tuple(spline_.template name<memory_optimized>(               \
          G.space(), std::get<Is>(xi), std::get<Is>(knot_indices),             \
          std::get<Is>(knot_indices_G))...);                                   \
    else if constexpr (comp == functionspace::boundary)                        \
      return std::tuple(boundary_.template name<memory_optimized>(             \
          static_cast<typename Geometry::boundary_type::boundary_type>(        \
              G.boundary().coeffs()),                                          \
          std::get<Is>(xi), std::get<Is>(knot_indices),                        \
          std::get<Is>(knot_indices_G))...);                                   \
  }                                                                            \
                                                                               \
  template <functionspace comp = functionspace::interior,                      \
            bool memory_optimized = false, std::size_t... Is,                  \
            typename Geometry, typename... Xi, typename... Knot_Indices,       \
            typename... Coeff_Indices, typename... Knot_Indices_G,             \
            typename... Coeff_Indices_G>                                       \
  inline auto BOOST_PP_CAT(name, _all_)(                                       \
      std::index_sequence<Is...>, const Geometry &G,                           \
      const std::tuple<Xi...> &xi,                                             \
      const std::tuple<Knot_Indices...> &knot_indices,                         \
      const std::tuple<Coeff_Indices...> &coeff_indices,                       \
      const std::tuple<Knot_Indices_G...> &knot_indices_G,                     \
      const std::tuple<Coeff_Indices_G...> &coeff_indices_G) const {           \
    if constexpr (comp == functionspace::interior)                             \
      return std::tuple(spline_.template name<memory_optimized>(               \
          G.space(), std::get<Is>(xi), std::get<Is>(knot_indices),             \
          std::get<Is>(coeff_indices), std::get<Is>(knot_indices_G),           \
          std::get<Is>(coeff_indices_G))...);                                  \
    else if constexpr (comp == functionspace::boundary)                        \
      return std::tuple(boundary_.template name<memory_optimized>(             \
          static_cast<typename Geometry::boundary_type::boundary_type>(        \
              G.boundary().coeffs()),                                          \
          std::get<Is>(xi), std::get<Is>(knot_indices),                        \
          std::get<Is>(coeff_indices), std::get<Is>(knot_indices_G),           \
          std::get<Is>(coeff_indices_G))...);                                  \
  }                                                                            \
                                                                               \
public:                                                                        \
  template <functionspace comp = functionspace::interior,                      \
            bool memory_optimized = false, typename Geometry, typename Arg,    \
            typename... Args>                                                  \
  inline auto name(const Geometry &G, const Arg &arg, const Args &...args)     \
      const {                                                                  \
    if constexpr (comp == functionspace::interior) {                           \
      if constexpr (utils::is_tuple_v<Arg>)                                    \
        return BOOST_PP_CAT(name, _all_)<comp, memory_optimized>(              \
            std::make_index_sequence<std::tuple_size_v<Arg>>{}, G, arg,        \
            args...);                                                          \
      else                                                                     \
        return spline_.template name<memory_optimized>(G.space(), arg,         \
                                                       args...);               \
    } else if constexpr (comp == functionspace::boundary) {                    \
      if constexpr (utils::is_tuple_of_tuples_v<Arg>)                          \
        return BOOST_PP_CAT(name, _all_)<comp, memory_optimized>(              \
            std::make_index_sequence<std::tuple_size_v<Arg>>{}, G, arg,        \
            args...);                                                          \
      else                                                                     \
        return boundary_.template name<memory_optimized>(                      \
            static_cast<typename Geometry::boundary_type::boundary_type>(      \
                G.boundary().coeffs()),                                        \
            arg, args...);                                                     \
    }                                                                          \
  }

  /// @brief Auto-generated functions
  /// @{
  BOOST_PP_SEQ_FOR_EACH(GENERATE_IEXPR_MACRO, _, GENERATE_IEXPR_SEQ)
/// @}
#undef GENERATE_IEXPR_MACRO
};

/// Forward declaration
template <typename... Args> struct FunctionSpace_trait;

/// Function space with default boundary
template <typename Spline> struct FunctionSpace_trait<Spline> {
  using type = FunctionSpace<Spline, Boundary<Spline>>;
};

/// Function space with non-default boundary
template <typename Spline, typename Boundary>
struct FunctionSpace_trait<Spline, Boundary> {
  using type = FunctionSpace<Spline, Boundary>;
};

/// Tensor-product function space with default boundary
template <typename... Splines>
struct FunctionSpace_trait<std::tuple<Splines...>> {
  using type = FunctionSpace<utils::tuple_cat_t<Splines...>,
                             utils::tuple_cat_t<Boundary<Splines>...>>;
};

/// Tensor-product function space with non-default boundary
template <typename... Splines, typename... Boundaries>
struct FunctionSpace_trait<std::tuple<Splines...>, std::tuple<Boundaries...>> {
  using type = FunctionSpace<utils::tuple_cat_t<Splines...>,
                             utils::tuple_cat_t<Boundaries...>>;
};

/// Function space
template <typename Spline, typename Boundary>
struct FunctionSpace_trait<FunctionSpace<Spline, Boundary>> {
  using type = typename FunctionSpace_trait<Spline, Boundary>::type;
};

/// Tensor-product function space with default boundary
template <typename... Splines, typename... Boundaries>
struct FunctionSpace_trait<std::tuple<FunctionSpace<Splines, Boundaries>...>> {
  using type =
      typename FunctionSpace_trait<utils::tuple_cat_t<Splines...>,
                                   utils::tuple_cat_t<Boundaries...>>::type;
};

} // namespace detail

/// @brief Function space alias
template <typename... Args>
using FunctionSpace = typename detail::FunctionSpace_trait<Args...>::type;

/// @brief Print (as string) a function space object
template <typename Splines, typename Boundaries>
inline std::ostream &operator<<(std::ostream &os,
                                const FunctionSpace<Splines, Boundaries> &obj) {
  obj.pretty_print(os);
  return os;
}

/// @brief Spline function space \f$ S^{\mathbf{p}}_{\mathbf{p}-1}
/// \f$
///
/// This class implements the function space
///
/// \f[
/// S^{\mathbf{p}}_{\mathbf{p}-1}
/// =
/// S^{p_1,\dots,p_{n_{d_\text{par}}}}_{p_1-1,\dots,p_{n_{d_\text{par}}}-1}
/// \f]
///
/// where the superscript \f$ \mathbf{p} \f$ denotes the degrees of
/// the B-spline basis functions and the subscript \f$ \mathbf{p-1}
/// \f$ the regularity assuming that the knot vector does not contain
/// any repeated knots.
///
/// @tparam Spline Type of the spline objects
template <typename Spline> using S = FunctionSpace<Spline>;

/// @brief Taylor-Hood like function space
template <typename Spline, short_t = Spline::parDim()> class TH;

/// @brief Taylor-Hood like function space
///
/// This class implements the Taylor-Hood like function space
///
/// \f[
/// S^{p+1}_{p-1} \otimes S^{p}_{p-1}
/// \f]
///
/// in one spatial dimension \cite Buffa:2011.
template <typename Spline>
class TH<Spline, 1>
    : public FunctionSpace<
          std::tuple<typename Spline::template derived_self_type<
                         typename Spline::value_type, Spline::geoDim(),
                         Spline::degree(0) + 1>,
                     typename Spline::template derived_self_type<
                         typename Spline::value_type, Spline::geoDim(),
                         Spline::degree(0)>>> {
public:
  /// @brief Base type
  using Base = FunctionSpace<std::tuple<
      typename Spline::template derived_self_type<
          typename Spline::value_type, Spline::geoDim(), Spline::degree(0) + 1>,
      typename Spline::template derived_self_type<
          typename Spline::value_type, Spline::geoDim(), Spline::degree(0)>>>;

  /// @brief Constructor
  /// @{
  TH(const std::array<int64_t, 1> &ncoeffs, enum init init = init::greville,
     Options<typename Spline::value_type> options =
         iganet::Options<typename Spline::value_type>{})
      : Base(ncoeffs + utils::to_array(1_i64), ncoeffs, init, options) {
    static_assert(Spline::is_nonuniform(),
                  "TH function space requires non-uniform splines");
    Base::template space<0>().reduce_continuity();
  }

  TH(const std::array<std::vector<typename Spline::value_type>, 1> &kv,
     enum init init = init::greville,
     Options<typename Spline::value_type> options =
         iganet::Options<typename Spline::value_type>{})
      : Base({{kv[0].front + kv[0] + kv[0].back(), kv[1]}}, kv, init, options) {
    static_assert(Spline::is_nonuniform(),
                  "TH function space requires non-uniform splines");
    Base::template space<0>().reduce_continuity();
  }
  /// @}

  IGANET_FUNCTIONSPACE_DEFAULT_OPS(TH);
};

/// @brief Taylor-Hood like function space
///
/// This class implements the Taylor-Hood like function space
///
/// \f[
/// S^{p_1+1,p_2+1}_{p_1-1,p_2-1} \otimes
/// S^{p_1+1,p_2+1}_{p_1-1,p_2-1} \otimes
/// S^{p_1,p_2}_{p_1-1,p_2-1}
/// \f]
///
/// in two spatial dimensions \cite Buffa:2011.
template <typename Spline>
class TH<Spline, 2>
    : public FunctionSpace<
          std::tuple<typename Spline::template derived_self_type<
                         typename Spline::value_type, Spline::geoDim(),
                         Spline::degree(0) + 1, Spline::degree(1) + 1>,
                     typename Spline::template derived_self_type<
                         typename Spline::value_type, Spline::geoDim(),
                         Spline::degree(0) + 1, Spline::degree(1) + 1>,
                     typename Spline::template derived_self_type<
                         typename Spline::value_type, Spline::geoDim(),
                         Spline::degree(0), Spline::degree(1)>>> {
public:
  /// @brief Base type
  using Base = FunctionSpace<
      std::tuple<typename Spline::template derived_self_type<
                     typename Spline::value_type, Spline::geoDim(),
                     Spline::degree(0) + 1, Spline::degree(1) + 1>,
                 typename Spline::template derived_self_type<
                     typename Spline::value_type, Spline::geoDim(),
                     Spline::degree(0) + 1, Spline::degree(1) + 1>,
                 typename Spline::template derived_self_type<
                     typename Spline::value_type, Spline::geoDim(),
                     Spline::degree(0), Spline::degree(1)>>>;

  /// @brief Constructor
  /// @{
  TH(const std::array<int64_t, 2> &ncoeffs, enum init init = init::greville,
     Options<typename Spline::value_type> options =
         iganet::Options<typename Spline::value_type>{})
      : Base(ncoeffs + utils::to_array(1_i64, 1_i64),
             ncoeffs + utils::to_array(1_i64, 1_i64), ncoeffs, init, options) {
    static_assert(Spline::is_nonuniform(),
                  "TH function space requires non-uniform splines");
    Base::template space<0>().reduce_continuity();
    Base::template space<1>().reduce_continuity();
  }

  TH(const std::array<std::vector<typename Spline::value_type>, 2> &kv,
     enum init init = init::greville,
     Options<typename Spline::value_type> options =
         iganet::Options<typename Spline::value_type>{})
      : Base({{kv[0].front() + kv[0] + kv[0].back(),
               kv[1].front() + kv[1] + kv[1].back()}},
             {{kv[0].front() + kv[0] + kv[0].back(),
               kv[1].front() + kv[1] + kv[1].back()}},
             kv, init, options) {
    static_assert(Spline::is_nonuniform(),
                  "TH function space requires non-uniform splines");
    Base::template space<0>().reduce_continuity();
    Base::template space<1>().reduce_continuity();
  }
  /// @}

  IGANET_FUNCTIONSPACE_DEFAULT_OPS(TH);
};

/// @brief Taylor-Hood like function space
///
/// This class implements the Taylor-Hood like function space
///
/// \f[
/// S^{p+1,p+1,p+1}_{p-1,p-1,p-1} \otimes
/// S^{p+1,p+1,p+1}_{p-1,p-1,p-1} \otimes
/// S^{p+1,p+1,p+1}_{p-1,p-1,p-1} \otimes
/// S^{p,p,p}_{p-1,p-1,p-1}
/// \f]
///
/// in three spatial dimensions \cite Buffa:2011.
template <typename Spline>
class TH<Spline, 3>
    : public FunctionSpace<std::tuple<
          typename Spline::template derived_self_type<
              typename Spline::value_type, Spline::geoDim(),
              Spline::degree(0) + 1, Spline::degree(1) + 1,
              Spline::degree(2) + 1>,
          typename Spline::template derived_self_type<
              typename Spline::value_type, Spline::geoDim(),
              Spline::degree(0) + 1, Spline::degree(1) + 1,
              Spline::degree(2) + 1>,
          typename Spline::template derived_self_type<
              typename Spline::value_type, Spline::geoDim(),
              Spline::degree(0) + 1, Spline::degree(1) + 1,
              Spline::degree(2) + 1>,
          typename Spline::template derived_self_type<
              typename Spline::value_type, Spline::geoDim(), Spline::degree(0),
              Spline::degree(1), Spline::degree(2)>>> {
public:
  /// @brief Base type
  using Base = FunctionSpace<std::tuple<
      typename Spline::template derived_self_type<
          typename Spline::value_type, Spline::geoDim(), Spline::degree(0) + 1,
          Spline::degree(1) + 1, Spline::degree(2) + 1>,
      typename Spline::template derived_self_type<
          typename Spline::value_type, Spline::geoDim(), Spline::degree(0) + 1,
          Spline::degree(1) + 1, Spline::degree(2) + 1>,
      typename Spline::template derived_self_type<
          typename Spline::value_type, Spline::geoDim(), Spline::degree(0) + 1,
          Spline::degree(1) + 1, Spline::degree(2) + 1>,
      typename Spline::template derived_self_type<
          typename Spline::value_type, Spline::geoDim(), Spline::degree(0),
          Spline::degree(1), Spline::degree(2)>>>;

  /// @brief Constructor
  /// @{
  TH(const std::array<int64_t, 3> &ncoeffs, enum init init = init::greville,
     Options<typename Spline::value_type> options =
         iganet::Options<typename Spline::value_type>{})
      : Base(ncoeffs + utils::to_array(1_i64, 1_i64, 1_i64),
             ncoeffs + utils::to_array(1_i64, 1_i64, 1_i64),
             ncoeffs + utils::to_array(1_i64, 1_i64, 1_i64), ncoeffs, init,
             options) {
    static_assert(Spline::is_nonuniform(),
                  "TH function space requires non-uniform splines");
    Base::template space<0>().reduce_continuity();
    Base::template space<1>().reduce_continuity();
    Base::template space<2>().reduce_continuity();
  }

  TH(const std::array<std::vector<typename Spline::value_type>, 3> &kv,
     enum init init = init::greville,
     Options<typename Spline::value_type> options =
         iganet::Options<typename Spline::value_type>{})
      : Base({{kv[0].front() + kv[0] + kv[0].back(),
               kv[1].front() + kv[1] + kv[1].back(),
               kv[2].front() + kv[2] + kv[2].back()}},
             {{kv[0].front() + kv[0] + kv[0].back(),
               kv[1].front() + kv[1] + kv[1].back(),
               kv[2].front() + kv[2] + kv[2].back()}},
             {{kv[0].front() + kv[0] + kv[0].back(),
               kv[1].front() + kv[1] + kv[1].back(),
               kv[2].front() + kv[2] + kv[2].back()}},
             kv, init, options) {
    static_assert(Spline::is_nonuniform(),
                  "TH function space requires non-uniform splines");
    Base::template space<0>().reduce_continuity();
    Base::template space<1>().reduce_continuity();
    Base::template space<2>().reduce_continuity();
  }
  /// @}

  IGANET_FUNCTIONSPACE_DEFAULT_OPS(TH);
};

/// @brief Taylor-Hood like function space
///
/// This class implements the Taylor-Hood like function space
///
/// \f[
/// S^{p+1,p+1,p+1,p+1}_{p-1,p-1,p-1,p-1} \otimes
/// S^{p+1,p+1,p+1,p+1}_{p-1,p-1,p-1,p-1} \otimes
/// S^{p+1,p+1,p+1,p+1}_{p-1,p-1,p-1,p-1} \otimes
/// S^{p+1,p+1,p+1,p+1}_{p-1,p-1,p-1,p-1} \otimes
/// S^{p,p,p,p}_{p-1,p-1,p-1,p-1}
/// \f]
///
/// in four spatial dimensions \cite Buffa:2011.
template <typename Spline>
class TH<Spline, 4>
    : public FunctionSpace<std::tuple<
          typename Spline::template derived_self_type<
              typename Spline::value_type, Spline::geoDim(),
              Spline::degree(0) + 1, Spline::degree(1) + 1,
              Spline::degree(2) + 1, Spline::degree(3) + 1>,
          typename Spline::template derived_self_type<
              typename Spline::value_type, Spline::geoDim(),
              Spline::degree(0) + 1, Spline::degree(1) + 1,
              Spline::degree(2) + 1, Spline::degree(3) + 1>,
          typename Spline::template derived_self_type<
              typename Spline::value_type, Spline::geoDim(),
              Spline::degree(0) + 1, Spline::degree(1) + 1,
              Spline::degree(2) + 1, Spline::degree(3) + 1>,
          typename Spline::template derived_self_type<
              typename Spline::value_type, Spline::geoDim(),
              Spline::degree(0) + 1, Spline::degree(1) + 1,
              Spline::degree(2) + 1, Spline::degree(3) + 1>,
          typename Spline::template derived_self_type<
              typename Spline::value_type, Spline::geoDim(), Spline::degree(0),
              Spline::degree(1), Spline::degree(2), Spline::degree(3)>>> {
public:
  /// @brief Base type
  using Base = FunctionSpace<std::tuple<
      typename Spline::template derived_self_type<
          typename Spline::value_type, Spline::geoDim(), Spline::degree(0) + 1,
          Spline::degree(1) + 1, Spline::degree(2) + 1, Spline::degree(3) + 1>,
      typename Spline::template derived_self_type<
          typename Spline::value_type, Spline::geoDim(), Spline::degree(0) + 1,
          Spline::degree(1) + 1, Spline::degree(2) + 1, Spline::degree(3) + 1>,
      typename Spline::template derived_self_type<
          typename Spline::value_type, Spline::geoDim(), Spline::degree(0) + 1,
          Spline::degree(1) + 1, Spline::degree(2) + 1, Spline::degree(3) + 1>,
      typename Spline::template derived_self_type<
          typename Spline::value_type, Spline::geoDim(), Spline::degree(0) + 1,
          Spline::degree(1) + 1, Spline::degree(2) + 1, Spline::degree(3) + 1>,
      typename Spline::template derived_self_type<
          typename Spline::value_type, Spline::geoDim(), Spline::degree(0),
          Spline::degree(1), Spline::degree(2), Spline::degree(3)>>>;

  /// @brief Constructor
  /// @{
  TH(const std::array<int64_t, 4> &ncoeffs, enum init init = init::greville,
     Options<typename Spline::value_type> options =
         iganet::Options<typename Spline::value_type>{})
      : Base(ncoeffs + utils::to_array(1_i64, 1_i64, 1_i64, 1_i64),
             ncoeffs + utils::to_array(1_i64, 1_i64, 1_i64, 1_i64),
             ncoeffs + utils::to_array(1_i64, 1_i64, 1_i64, 1_i64),
             ncoeffs + utils::to_array(1_i64, 1_i64, 1_i64, 1_i64), ncoeffs,
             init, options) {
    static_assert(Spline::is_nonuniform(),
                  "TH function space requires non-uniform splines");
    Base::template space<0>().reduce_continuity();
    Base::template space<1>().reduce_continuity();
    Base::template space<2>().reduce_continuity();
    Base::template space<3>().reduce_continuity();
  }

  TH(const std::array<std::vector<typename Spline::value_type>, 4> &kv,
     enum init init = init::greville,
     Options<typename Spline::value_type> options =
         iganet::Options<typename Spline::value_type>{})
      : Base({{kv[0].front() + kv[0] + kv[0].back(),
               kv[1].front() + kv[1] + kv[1].back(),
               kv[2].front() + kv[2] + kv[2].back(),
               kv[3].front() + kv[3] + kv[3].back()}},
             {{kv[0].front() + kv[0] + kv[0].back(),
               kv[1].front() + kv[1] + kv[1].back(),
               kv[2].front() + kv[2] + kv[2].back(),
               kv[3].front() + kv[3] + kv[3].back()}},
             {{kv[0].front() + kv[0] + kv[0].back(),
               kv[1].front() + kv[1] + kv[1].back(),
               kv[2].front() + kv[2] + kv[2].back(),
               kv[3].front() + kv[3] + kv[3].back()}},
             {{kv[0].front() + kv[0] + kv[0].back(),
               kv[1].front() + kv[1] + kv[1].back(),
               kv[2].front() + kv[2] + kv[2].back(),
               kv[3].front() + kv[3] + kv[3].back()}},
             kv, init, options) {
    static_assert(Spline::is_nonuniform(),
                  "TH function space requires non-uniform splines");
    Base::template space<0>().reduce_continuity();
    Base::template space<1>().reduce_continuity();
    Base::template space<2>().reduce_continuity();
    Base::template space<3>().reduce_continuity();
  }
  /// @}

  IGANET_FUNCTIONSPACE_DEFAULT_OPS(TH);
};

/// IGANET_FUNCTIONSPACE_TUPLE_WRAPPER(TH);

/// @brief Nedelec like function space
template <typename Spline, short_t = Spline::parDim()> class NE;

/// @brief Nedelec like function space
///
/// This class implements the Nedelec like function space
///
/// \f[
/// S^{p+1}_{p} \otimes S^{p}_{p-1}
/// \f]
///
/// in one spatial dimension \cite Buffa:2011.
template <typename Spline>
class NE<Spline, 1>
    : public FunctionSpace<
          std::tuple<typename Spline::template derived_self_type<
                         typename Spline::value_type, Spline::geoDim(),
                         Spline::degree(0) + 1>,
                     typename Spline::template derived_self_type<
                         typename Spline::value_type, Spline::geoDim(),
                         Spline::degree(0)>>> {
public:
  /// @brief Base type
  using Base = FunctionSpace<std::tuple<
      typename Spline::template derived_self_type<
          typename Spline::value_type, Spline::geoDim(), Spline::degree(0) + 1>,
      typename Spline::template derived_self_type<
          typename Spline::value_type, Spline::geoDim(), Spline::degree(0)>>>;

  /// @brief Constructor
  /// @{
  NE(const std::array<int64_t, 1> &ncoeffs, enum init init = init::greville,
     Options<typename Spline::value_type> options =
         iganet::Options<typename Spline::value_type>{})
      : Base(ncoeffs + utils::to_array(1_i64), ncoeffs, init, options) {}

  NE(const std::array<std::vector<typename Spline::value_type>, 1> &kv,
     enum init init = init::greville,
     Options<typename Spline::value_type> options =
         iganet::Options<typename Spline::value_type>{})
      : Base(kv, kv, init, options) {
    static_assert(Spline::is_nonuniform(),
                  "Constructor only available for non-uniform splines");
  }
  /// @}

  IGANET_FUNCTIONSPACE_DEFAULT_OPS(NE);
};

/// @brief Nedelec like function space
///
/// This class implements the Nedelec like function space
/// \f[
/// S^{p+1,p+1}_{p,p-1} \otimes
/// S^{p+1,p+1}_{p-1,p} \otimes
/// S^{p,p}_{p-1,p-1}
/// \f]
///
/// in two spatial dimensions \cite Buffa:2011.
template <typename Spline>
class NE<Spline, 2>
    : public FunctionSpace<
          std::tuple<typename Spline::template derived_self_type<
                         typename Spline::value_type, Spline::geoDim(),
                         Spline::degree(0) + 1, Spline::degree(1) + 1>,
                     typename Spline::template derived_self_type<
                         typename Spline::value_type, Spline::geoDim(),
                         Spline::degree(0) + 1, Spline::degree(1) + 1>,
                     typename Spline::template derived_self_type<
                         typename Spline::value_type, Spline::geoDim(),
                         Spline::degree(0), Spline::degree(1)>>> {
public:
  /// @brief Base type
  using Base = FunctionSpace<
      std::tuple<typename Spline::template derived_self_type<
                     typename Spline::value_type, Spline::geoDim(),
                     Spline::degree(0) + 1, Spline::degree(1) + 1>,
                 typename Spline::template derived_self_type<
                     typename Spline::value_type, Spline::geoDim(),
                     Spline::degree(0) + 1, Spline::degree(1) + 1>,
                 typename Spline::template derived_self_type<
                     typename Spline::value_type, Spline::geoDim(),
                     Spline::degree(0), Spline::degree(1)>>>;

  /// @brief Constructor
  /// @{
  NE(const std::array<int64_t, 2> &ncoeffs, enum init init = init::greville,
     Options<typename Spline::value_type> options =
         iganet::Options<typename Spline::value_type>{})
      : Base(ncoeffs + utils::to_array(1_i64, 1_i64),
             ncoeffs + utils::to_array(1_i64, 1_i64), ncoeffs, init, options) {
    static_assert(Spline::is_nonuniform(),
                  "NE function space requires non-uniform splines");
    Base::template space<0>().reduce_continuity(1, 1);
    Base::template space<1>().reduce_continuity(1, 0);
  }

  NE(const std::array<std::vector<typename Spline::value_type>, 2> &kv,
     enum init init = init::greville,
     Options<typename Spline::value_type> options =
         iganet::Options<typename Spline::value_type>{})
      : Base(kv, kv, kv, init, options) {
    static_assert(Spline::is_nonuniform(),
                  "NE function space requires non-uniform splines");
    Base::template space<0>().reduce_continuity(1, 1);
    Base::template space<1>().reduce_continuity(1, 0);
  }
  /// @}

  IGANET_FUNCTIONSPACE_DEFAULT_OPS(NE);
};

/// @brief Nedelec like function space
///
/// This class implements the Nedelec like function space
///
/// \f[
/// S^{p+1,p+1,p+1}_{p,p-1,p-1} \otimes
/// S^{p+1,p+1,p+1}_{p-1,p,p-1} \otimes
/// S^{p+1,p+1,p+1}_{p-1,p-1,p} \otimes
/// S^{p,p,p}_{p-1,p-1,p-1}
/// \f]
///
/// in three spatial dimensions \cite Buffa:2011.
template <typename Spline>
class NE<Spline, 3>
    : public FunctionSpace<std::tuple<
          typename Spline::template derived_self_type<
              typename Spline::value_type, Spline::geoDim(),
              Spline::degree(0) + 1, Spline::degree(1) + 1,
              Spline::degree(2) + 1>,
          typename Spline::template derived_self_type<
              typename Spline::value_type, Spline::geoDim(),
              Spline::degree(0) + 1, Spline::degree(1) + 1,
              Spline::degree(2) + 1>,
          typename Spline::template derived_self_type<
              typename Spline::value_type, Spline::geoDim(),
              Spline::degree(0) + 1, Spline::degree(1) + 1,
              Spline::degree(2) + 1>,
          typename Spline::template derived_self_type<
              typename Spline::value_type, Spline::geoDim(), Spline::degree(0),
              Spline::degree(1), Spline::degree(2)>>> {
public:
  /// @brief Base type
  using Base = FunctionSpace<std::tuple<
      typename Spline::template derived_self_type<
          typename Spline::value_type, Spline::geoDim(), Spline::degree(0) + 1,
          Spline::degree(1) + 1, Spline::degree(2) + 1>,
      typename Spline::template derived_self_type<
          typename Spline::value_type, Spline::geoDim(), Spline::degree(0) + 1,
          Spline::degree(1) + 1, Spline::degree(2) + 1>,
      typename Spline::template derived_self_type<
          typename Spline::value_type, Spline::geoDim(), Spline::degree(0) + 1,
          Spline::degree(1) + 1, Spline::degree(2) + 1>,
      typename Spline::template derived_self_type<
          typename Spline::value_type, Spline::geoDim(), Spline::degree(0),
          Spline::degree(1), Spline::degree(2)>>>;

  /// @brief Constructor
  /// @{
  NE(const std::array<int64_t, 3> &ncoeffs, enum init init = init::greville,
     Options<typename Spline::value_type> options =
         iganet::Options<typename Spline::value_type>{})
      : Base(ncoeffs + utils::to_array(1_i64, 1_i64, 1_i64),
             ncoeffs + utils::to_array(1_i64, 1_i64, 1_i64),
             ncoeffs + utils::to_array(1_i64, 1_i64, 1_i64), ncoeffs, init,
             options) {
    static_assert(Spline::is_nonuniform(),
                  "NE function space requires non-uniform splines");
    Base::template space<0>().reduce_continuity(1, 1).reduce_continuity(1, 2);
    Base::template space<1>().reduce_continuity(1, 0).reduce_continuity(1, 2);
    Base::template space<2>().reduce_continuity(1, 0).reduce_continuity(1, 1);
  }

  NE(const std::array<std::vector<typename Spline::value_type>, 3> &kv,
     enum init init = init::greville,
     Options<typename Spline::value_type> options =
         iganet::Options<typename Spline::value_type>{})
      : Base(kv, kv, kv, kv, init, options) {
    static_assert(Spline::is_nonuniform(),
                  "NE function space requires non-uniform splines");
    Base::template space<0>().reduce_continuity(1, 1).reduce_continuity(1, 2);
    Base::template space<1>().reduce_continuity(1, 0).reduce_continuity(1, 2);
    Base::template space<2>().reduce_continuity(1, 0).reduce_continuity(1, 1);
  }
  /// @}

  IGANET_FUNCTIONSPACE_DEFAULT_OPS(NE);
};

/// @brief Nedelec like function space
///
/// This class implements the Nedelec like function space
///
/// \f[
/// S^{p+1,p+1,p+1,p+1}_{p,p-1,p-1,p-1} \otimes
/// S^{p+1,p+1,p+1,p+1}_{p-1,p,p-1,p-1} \otimes
/// S^{p+1,p+1,p+1,p+1}_{p-1,p-1,p,p-1} \otimes
/// S^{p+1,p+1,p+1,p+1}_{p-1,p-1,p-1,p} \otimes
/// S^{p,p,p,p}_{p-1,p-1,p-1,p-1}
/// \f]
///
/// in four spatial dimensions \cite Buffa:2011.
template <typename Spline>
class NE<Spline, 4>
    : public FunctionSpace<std::tuple<
          typename Spline::template derived_self_type<
              typename Spline::value_type, Spline::geoDim(),
              Spline::degree(0) + 1, Spline::degree(1) + 1,
              Spline::degree(2) + 1, Spline::degree(3) + 1>,
          typename Spline::template derived_self_type<
              typename Spline::value_type, Spline::geoDim(),
              Spline::degree(0) + 1, Spline::degree(1) + 1,
              Spline::degree(2) + 1, Spline::degree(3) + 1>,
          typename Spline::template derived_self_type<
              typename Spline::value_type, Spline::geoDim(),
              Spline::degree(0) + 1, Spline::degree(1) + 1,
              Spline::degree(2) + 1, Spline::degree(3) + 1>,
          typename Spline::template derived_self_type<
              typename Spline::value_type, Spline::geoDim(),
              Spline::degree(0) + 1, Spline::degree(1) + 1,
              Spline::degree(2) + 1, Spline::degree(3) + 1>,
          typename Spline::template derived_self_type<
              typename Spline::value_type, Spline::geoDim(), Spline::degree(0),
              Spline::degree(1), Spline::degree(2), Spline::degree(3)>>> {
public:
  /// @brief Base type
  using Base = FunctionSpace<std::tuple<
      typename Spline::template derived_self_type<
          typename Spline::value_type, Spline::geoDim(), Spline::degree(0) + 1,
          Spline::degree(1) + 1, Spline::degree(2) + 1, Spline::degree(3) + 1>,
      typename Spline::template derived_self_type<
          typename Spline::value_type, Spline::geoDim(), Spline::degree(0) + 1,
          Spline::degree(1) + 1, Spline::degree(2) + 1, Spline::degree(3) + 1>,
      typename Spline::template derived_self_type<
          typename Spline::value_type, Spline::geoDim(), Spline::degree(0) + 1,
          Spline::degree(1) + 1, Spline::degree(2) + 1, Spline::degree(3) + 1>,
      typename Spline::template derived_self_type<
          typename Spline::value_type, Spline::geoDim(), Spline::degree(0) + 1,
          Spline::degree(1) + 1, Spline::degree(2) + 1, Spline::degree(3) + 1>,
      typename Spline::template derived_self_type<
          typename Spline::value_type, Spline::geoDim(), Spline::degree(0),
          Spline::degree(1), Spline::degree(2), Spline::degree(3)>>>;

  /// @brief Constructor
  /// @{
  NE(const std::array<int64_t, 4> &ncoeffs, enum init init = init::greville,
     Options<typename Spline::value_type> options =
         iganet::Options<typename Spline::value_type>{})
      : Base(ncoeffs + utils::to_array(1_i64, 1_i64, 1_i64, 1_i64),
             ncoeffs + utils::to_array(1_i64, 1_i64, 1_i64, 1_i64),
             ncoeffs + utils::to_array(1_i64, 1_i64, 1_i64, 1_i64),
             ncoeffs + utils::to_array(1_i64, 1_i64, 1_i64, 1_i64), ncoeffs,
             init, options) {
    static_assert(Spline::is_nonuniform(),
                  "NE function space requires non-uniform splines");
    Base::template space<0>()
        .reduce_continuity(1, 1)
        .reduce_continuity(1, 2)
        .reduce_continuity(1, 3);
    Base::template space<1>()
        .reduce_continuity(1, 0)
        .reduce_continuity(1, 2)
        .reduce_continuity(1, 3);
    Base::template space<2>()
        .reduce_continuity(1, 0)
        .reduce_continuity(1, 1)
        .reduce_continuity(1, 3);
    Base::template space<3>()
        .reduce_continuity(1, 0)
        .reduce_continuity(1, 1)
        .reduce_continuity(1, 2);
  }

  NE(const std::array<std::vector<typename Spline::value_type>,
                      Spline::parDim()> &kv,
     enum init init = init::greville,
     Options<typename Spline::value_type> options =
         iganet::Options<typename Spline::value_type>{})
      : Base(kv, kv, kv, kv, kv, init, options) {
    static_assert(Spline::is_nonuniform(),
                  "NE function space requires non-uniform splines");
    Base::template space<0>()
        .reduce_continuity(1, 1)
        .reduce_continuity(1, 2)
        .reduce_continuity(1, 3);
    Base::template space<1>()
        .reduce_continuity(1, 0)
        .reduce_continuity(1, 2)
        .reduce_continuity(1, 3);
    Base::template space<2>()
        .reduce_continuity(1, 0)
        .reduce_continuity(1, 1)
        .reduce_continuity(1, 3);
    Base::template space<3>()
        .reduce_continuity(1, 0)
        .reduce_continuity(1, 1)
        .reduce_continuity(1, 2);
  }
  /// @}

  IGANET_FUNCTIONSPACE_DEFAULT_OPS(NE);
};

/// IGANET_FUNCTIONSPACE_TUPLE_WRAPPER(NE);

/// @brief Raviart-Thomas like function space
template <typename Spline, short_t = Spline::parDim()> class RT;

/// @brief Raviart-Thomas like function space
///
/// This class implements the Raviart-Thomas like function space
///
/// \f[
/// S^{p+1}_{p} \otimes S^{p}_{p-1}
/// \f]
///
/// in one spatial dimension \cite Buffa:2011.
template <typename Spline>
class RT<Spline, 1>
    : public FunctionSpace<
          std::tuple<typename Spline::template derived_self_type<
                         typename Spline::value_type, Spline::geoDim(),
                         Spline::degree(0) + 1>,
                     typename Spline::template derived_self_type<
                         typename Spline::value_type, Spline::geoDim(),
                         Spline::degree(0)>>> {
public:
  /// @brief Base type
  using Base = FunctionSpace<std::tuple<
      typename Spline::template derived_self_type<
          typename Spline::value_type, Spline::geoDim(), Spline::degree(0) + 1>,
      typename Spline::template derived_self_type<
          typename Spline::value_type, Spline::geoDim(), Spline::degree(0)>>>;

  /// @brief Constructor
  /// @{
  RT(const std::array<int64_t, 1> &ncoeffs, enum init init = init::greville,
     Options<typename Spline::value_type> options =
         iganet::Options<typename Spline::value_type>{})
      : Base(ncoeffs + utils::to_array(1_i64), ncoeffs, init, options) {}

  RT(const std::array<std::vector<typename Spline::value_type>, 1> &kv,
     enum init init = init::greville,
     Options<typename Spline::value_type> options =
         iganet::Options<typename Spline::value_type>{})
      : Base({{kv[0].front() + kv[0] + kv[0].back(), kv[1]}}, kv, init,
             options) {
    static_assert(Spline::is_nonuniform(),
                  "Constructor only available for non-uniform splines");
  }
  /// @}

  IGANET_FUNCTIONSPACE_DEFAULT_OPS(RT);
};

/// @brief Raviart-Thomas like function space
///
/// This class implements the Raviart-Thomas like function space
///
/// \f[
/// S^{p+1,p}_{p,p-1} \otimes
/// S^{p,p+1}_{p-1,p} \otimes
/// S^{p,p}_{p-1,p-1} \f$
/// \f]
///
/// in two spatial dimensions \cite Buffa:2011.
template <typename Spline>
class RT<Spline, 2>
    : public FunctionSpace<
          std::tuple<typename Spline::template derived_self_type<
                         typename Spline::value_type, Spline::geoDim(),
                         Spline::degree(0) + 1, Spline::degree(1)>,
                     typename Spline::template derived_self_type<
                         typename Spline::value_type, Spline::geoDim(),
                         Spline::degree(0), Spline::degree(1) + 1>,
                     typename Spline::template derived_self_type<
                         typename Spline::value_type, Spline::geoDim(),
                         Spline::degree(0), Spline::degree(1)>>> {
public:
  using Base = FunctionSpace<
      std::tuple<typename Spline::template derived_self_type<
                     typename Spline::value_type, Spline::geoDim(),
                     Spline::degree(0) + 1, Spline::degree(1)>,
                 typename Spline::template derived_self_type<
                     typename Spline::value_type, Spline::geoDim(),
                     Spline::degree(0), Spline::degree(1) + 1>,
                 typename Spline::template derived_self_type<
                     typename Spline::value_type, Spline::geoDim(),
                     Spline::degree(0), Spline::degree(1)>>>;

  /// @brief Constructor
  /// @{
  RT(const std::array<int64_t, 2> &ncoeffs, enum init init = init::greville,
     Options<typename Spline::value_type> options =
         iganet::Options<typename Spline::value_type>{})
      : Base(ncoeffs + utils::to_array(1_i64, 0_i64),
             ncoeffs + utils::to_array(0_i64, 1_i64), ncoeffs, init, options) {}

  RT(const std::array<std::vector<typename Spline::value_type>, 2> &kv,
     enum init init = init::greville,
     Options<typename Spline::value_type> options =
         iganet::Options<typename Spline::value_type>{})
      : Base({{kv[0].front() + kv[0] + kv[0].back(), kv[1]}},
             {{kv[0], kv[1].front() + kv[1] + kv[1].back()}}, kv, init,
             options) {
    static_assert(Spline::is_nonuniform(),
                  "Constructor only available for non-uniform splines");
  }
  /// @}

  IGANET_FUNCTIONSPACE_DEFAULT_OPS(RT);
};

/// @brief Raviart-Thomas like function space
///
/// This class implements the Raviart-Thomas like function space
///
/// \f[
/// S^{p+1,p,p}_{p,p-1,p-1} \otimes
/// S^{p,p+1,p}_{p-1,p,p-1} \otimes
/// S^{p,p,p+1}_{p-1,p-1,p} \otimes
/// S^{p,p,p}_{p-1,p-1,p-1}
/// \f]
/// in three spatial dimensions \cite Buffa:2011.
template <typename Spline>
class RT<Spline, 3>
    : public FunctionSpace<std::tuple<
          typename Spline::template derived_self_type<
              typename Spline::value_type, Spline::geoDim(),
              Spline::degree(0) + 1, Spline::degree(1), Spline::degree(2)>,
          typename Spline::template derived_self_type<
              typename Spline::value_type, Spline::geoDim(), Spline::degree(0),
              Spline::degree(1) + 1, Spline::degree(2)>,
          typename Spline::template derived_self_type<
              typename Spline::value_type, Spline::geoDim(), Spline::degree(0),
              Spline::degree(1), Spline::degree(2) + 1>,
          typename Spline::template derived_self_type<
              typename Spline::value_type, Spline::geoDim(), Spline::degree(0),
              Spline::degree(1), Spline::degree(2)>>> {
public:
  /// @brief Base type
  using Base = FunctionSpace<std::tuple<
      typename Spline::template derived_self_type<
          typename Spline::value_type, Spline::geoDim(), Spline::degree(0) + 1,
          Spline::degree(1), Spline::degree(2)>,
      typename Spline::template derived_self_type<
          typename Spline::value_type, Spline::geoDim(), Spline::degree(0),
          Spline::degree(1) + 1, Spline::degree(2)>,
      typename Spline::template derived_self_type<
          typename Spline::value_type, Spline::geoDim(), Spline::degree(0),
          Spline::degree(1), Spline::degree(2) + 1>,
      typename Spline::template derived_self_type<
          typename Spline::value_type, Spline::geoDim(), Spline::degree(0),
          Spline::degree(1), Spline::degree(2)>>>;

  /// @brief Constructor
  /// @{
  RT(const std::array<int64_t, 3> &ncoeffs, enum init init = init::greville,
     Options<typename Spline::value_type> options =
         iganet::Options<typename Spline::value_type>{})
      : Base(ncoeffs + utils::to_array(1_i64, 0_i64, 0_i64),
             ncoeffs + utils::to_array(0_i64, 1_i64, 0_i64),
             ncoeffs + utils::to_array(0_i64, 0_i64, 1_i64), ncoeffs, init,
             options) {}

  RT(const std::array<std::vector<typename Spline::value_type>, 3> &kv,
     enum init init = init::greville,
     Options<typename Spline::value_type> options =
         iganet::Options<typename Spline::value_type>{})
      : Base({{kv[0].front() + kv[0] + kv[0].back(), kv[1], kv[2]}},
             {{kv[0], kv[1].front() + kv[1] + kv[1].back(), kv[2]}},
             {{kv[0], kv[1], kv[2].front() + kv[2] + kv[2].back()}}, kv, init,
             options) {
    static_assert(Spline::is_nonuniform(),
                  "Constructor only available for non-uniform splines");
  }
  /// @}

  IGANET_FUNCTIONSPACE_DEFAULT_OPS(RT);
};

/// @brief Raviart-Thomas like function space
///
/// This class implements the Raviart-Thomas like function space
///
/// \f[
/// S^{p+1,p,p,p}_{p,p-1,p-1,p-1} \otimes
/// S^{p,p+1,p,p}_{p-1,p,p-1,p-1} \otimes
/// S^{p,p,p+1,p}_{p-1,p-1,p,p-1} \otimes
/// S^{p,p,p,p+1}_{p-1,p-1,p-1,p} \otimes
/// S^{p,p,p,p}_{p-1,p-1,p-1,p-1}
/// \f]
///
/// in four spatial dimensions \cite Buffa:2011.
template <typename Spline>
class RT<Spline, 4>
    : public FunctionSpace<std::tuple<
          typename Spline::template derived_self_type<
              typename Spline::value_type, Spline::geoDim(),
              Spline::degree(0) + 1, Spline::degree(1), Spline::degree(2),
              Spline::degree(3)>,
          typename Spline::template derived_self_type<
              typename Spline::value_type, Spline::geoDim(), Spline::degree(0),
              Spline::degree(1) + 1, Spline::degree(2), Spline::degree(3)>,
          typename Spline::template derived_self_type<
              typename Spline::value_type, Spline::geoDim(), Spline::degree(0),
              Spline::degree(1), Spline::degree(2) + 1, Spline::degree(3)>,
          typename Spline::template derived_self_type<
              typename Spline::value_type, Spline::geoDim(), Spline::degree(0),
              Spline::degree(1), Spline::degree(2), Spline::degree(3) + 1>,
          typename Spline::template derived_self_type<
              typename Spline::value_type, Spline::geoDim(), Spline::degree(0),
              Spline::degree(1), Spline::degree(2), Spline::degree(3)>>> {
public:
  /// @brief Base type
  using Base = FunctionSpace<std::tuple<
      typename Spline::template derived_self_type<
          typename Spline::value_type, Spline::geoDim(), Spline::degree(0) + 1,
          Spline::degree(1), Spline::degree(2), Spline::degree(3)>,
      typename Spline::template derived_self_type<
          typename Spline::value_type, Spline::geoDim(), Spline::degree(0),
          Spline::degree(1) + 1, Spline::degree(2), Spline::degree(3)>,
      typename Spline::template derived_self_type<
          typename Spline::value_type, Spline::geoDim(), Spline::degree(0),
          Spline::degree(1), Spline::degree(2) + 1, Spline::degree(3)>,
      typename Spline::template derived_self_type<
          typename Spline::value_type, Spline::geoDim(), Spline::degree(0),
          Spline::degree(1), Spline::degree(2), Spline::degree(3) + 1>,
      typename Spline::template derived_self_type<
          typename Spline::value_type, Spline::geoDim(), Spline::degree(0),
          Spline::degree(1), Spline::degree(2), Spline::degree(3)>>>;

  /// @brief Constructor
  /// @{
  RT(const std::array<int64_t, 4> &ncoeffs, enum init init = init::greville,
     Options<typename Spline::value_type> options =
         iganet::Options<typename Spline::value_type>{})
      : Base(ncoeffs + utils::to_array(1_i64, 0_i64, 0_i64, 0_i64),
             ncoeffs + utils::to_array(0_i64, 1_i64, 0_i64, 0_i64),
             ncoeffs + utils::to_array(0_i64, 0_i64, 1_i64, 0_i64),
             ncoeffs + utils::to_array(0_i64, 0_i64, 0_i64, 1_i64), ncoeffs,
             init, options) {}

  RT(const std::array<std::vector<typename Spline::value_type>, 4> &kv,
     enum init init = init::greville,
     Options<typename Spline::value_type> options =
         iganet::Options<typename Spline::value_type>{})
      : Base({{kv[0].front() + kv[0] + kv[0].back(), kv[1], kv[2], kv[3]}},
             {{kv[0], kv[1].front() + kv[1] + kv[1].back(), kv[2], kv[3]}},
             {{kv[0], kv[1], kv[2].front() + kv[2] + kv[2].back(), kv[3]}},
             {{kv[0], kv[1], kv[2], kv[3].front() + kv[3] + kv[3].back()}}, kv,
             init, options) {
    static_assert(Spline::is_nonuniform(),
                  "Constructor only available for non-uniform splines");
  }
  /// @}

  IGANET_FUNCTIONSPACE_DEFAULT_OPS(RT);
};

/// IGANET_FUNCTIONSPACE_TUPLE_WRAPPER(RT);

/// @brief H(curl) function space
template <typename Spline, short_t = Spline::parDim()> class Hcurl;

/// @brief H(curl) function space
///
/// This class implements the H(curl) function space
///
/// \f[
/// S_{p, p+1, p+1} \otimes
/// S_{p+1, p, p+1} \otimes
/// S_{p+1, p+1, p}
/// \f]
///
/// in three spatial dimensions
template <typename Spline>
class Hcurl<Spline, 3>
    : public FunctionSpace<std::tuple<
          typename Spline::template derived_self_type<
              typename Spline::value_type, Spline::geoDim(), Spline::degree(0),
              Spline::degree(1) + 1, Spline::degree(2) + 1>,
          typename Spline::template derived_self_type<
              typename Spline::value_type, Spline::geoDim(),
              Spline::degree(0) + 1, Spline::degree(1), Spline::degree(2) + 1>,
          typename Spline::template derived_self_type<
              typename Spline::value_type, Spline::geoDim(),
              Spline::degree(0) + 1, Spline::degree(1) + 1,
              Spline::degree(2)>>> {

public:
  /// @brief Base type
  using Base = FunctionSpace<std::tuple<
      typename Spline::template derived_self_type<
          typename Spline::value_type, Spline::geoDim(), Spline::degree(0),
          Spline::degree(1) + 1, Spline::degree(2) + 1>,
      typename Spline::template derived_self_type<
          typename Spline::value_type, Spline::geoDim(), Spline::degree(0) + 1,
          Spline::degree(1), Spline::degree(2) + 1>,
      typename Spline::template derived_self_type<
          typename Spline::value_type, Spline::geoDim(), Spline::degree(0) + 1,
          Spline::degree(1) + 1, Spline::degree(2)>>>;

  /// @brief Constructor
  /// @{
  Hcurl(const std::array<int64_t, 3> &ncoeffs, enum init init = init::greville,
        Options<typename Spline::value_type> options =
            iganet::Options<typename Spline::value_type>{})
      : Base(ncoeffs + utils::to_array(1_i64, 0_i64, 0_i64),
             ncoeffs + utils::to_array(0_i64, 1_i64, 0_i64),
             ncoeffs + utils::to_array(0_i64, 0_i64, 1_i64), init, options) {}

  Hcurl(const std::array<std::vector<typename Spline::value_type>, 3> &kv,
        enum init init = init::greville,
        Options<typename Spline::value_type> options =
            iganet::Options<typename Spline::value_type>{})
      : Base({{kv[0].front() + kv[0] + kv[0].back(), kv[1], kv[2]}},
             {{kv[0], kv[1].front() + kv[1] + kv[1].back(), kv[2]}},
             {{kv[0], kv[1], kv[2].front() + kv[2] + kv[2].back()}}, init,
             options) {
    static_assert(Spline::is_nonuniform(),
                  "Constructor only available for non-uniform splines");
  }
  /// @}
  IGANET_FUNCTIONSPACE_DEFAULT_OPS(Hcurl);
};

/// IGANET_FUNCTIONSPACE_TUPLE_WRAPPER(Hcurl);

/// #undef IGANET_FUNCTIONSPACE_TUPLE_WRAPPER
#undef IGANET_FUNCTIONSPACE_DEFAULT_OPS

} // namespace iganet
