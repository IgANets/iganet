/**
   @file include/igabase.hpp

   @brief Isogeometric analysis base class

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <filesystem>

namespace iganet {

/// @brief Enumerator for the collocation point specifier
enum class collPts : short_t {
  greville = 0,          /*!< Greville points */
  greville_interior = 1, /*!< Greville points in the interior */
  greville_ref1 = 2,     /*!< Greville points, once refined */
  greville_interior_ref1 =
      3,             /*!< Greville points in the interior, once refined */
  greville_ref2 = 4, /*!< Greville points, twice refined */
  greville_interior_ref2 =
      5, /*!< Greville points in the interior, twice refined */
};

/// @brief IgA base class (no reference data)
///
/// This class implements the base functionality of IgA solvers and nets for the
/// case that no reference solution is required
template <typename GeometryMap, typename Variable> class IgABaseNoRefData {
public:
  /// @brief Value type
  using value_type =
      typename std::common_type<typename GeometryMap::value_type,
                                typename Variable::value_type>::type;

  /// @brief Type of the geometry map function space(s)
  using geometryMap_type = GeometryMap;

  /// @brief Type of the variable function space(s)
  using variable_type = Variable;

  /// @brief Type of the geometry map collocation points
  using geometryMap_collPts_type =
      std::pair<typename GeometryMap::eval_type,
                typename GeometryMap::boundary_eval_type>;

  /// @brief Type of the variable collocation points
  using variable_collPts_type =
      std::pair<typename Variable::eval_type,
                typename Variable::boundary_eval_type>;

  /// @brief Indicates whether this class provides a geometry map
  bool static constexpr has_GeometryMap = true;

  /// @brief Indicates whether this class provides reference data
  bool static constexpr has_RefData = false;

  /// @brief Indicates whether this class provides a solution
  bool static constexpr has_Solution = true;

protected:
  /// @brief Spline representation of the geometry map
  GeometryMap G_;

  /// @brief Spline representation of the solution
  Variable u_;

private:
  /// @brief Constructor: number of spline coefficients (different for Geometry
  /// and Variable types)
  template <typename... GeometryMapSplines, size_t... Is,
            typename... VariableSplines, size_t... Js>
  IgABaseNoRefData(
      std::tuple<GeometryMapSplines...> geometryMap_splines,
      std::index_sequence<Is...>,
      std::tuple<VariableSplines...> variable_splines,
      std::index_sequence<Js...>,
      iganet::Options<value_type> options = iganet::Options<value_type>{})
      : // Construct the different spline objects individually
        G_(std::get<Is>(geometryMap_splines)..., init::greville, options),
        u_(std::get<Js>(variable_splines)..., init::random, options) {}

public:
  /// @brief Default constructor
  explicit IgABaseNoRefData(
      iganet::Options<value_type> options = iganet::Options<value_type>{})
      : G_(), u_() {}

  /// @brief Constructor: number of spline coefficients (same for geometry map
  /// and variables)
  template <typename... Splines>
  IgABaseNoRefData(
      std::tuple<Splines...> splines,
      iganet::Options<value_type> options = iganet::Options<value_type>{})
      : IgABaseNoRefData(splines, splines, options) {}

  /// @brief Constructor: number of spline coefficients (different for
  /// geometry map and variables)
  template <typename... GeometryMapSplines, typename... VariableSplines>
  IgABaseNoRefData(
      std::tuple<GeometryMapSplines...> geometryMap_splines,
      std::tuple<VariableSplines...> variable_splines,
      iganet::Options<value_type> options = iganet::Options<value_type>{})
      : IgABaseNoRefData(
            geometryMap_splines,
            std::make_index_sequence<sizeof...(GeometryMapSplines)>{},
            variable_splines,
            std::make_index_sequence<sizeof...(VariableSplines)>{}, options) {}

  /// @brief Returns a constant reference to the spline
  /// representation of the geometry map
  inline const GeometryMap &G() const { return G_; }

  /// @brief Returns a non-constant reference to the spline
  /// representation of the geometry map
  inline GeometryMap &G() { return G_; }

  /// @brief Returns a constant reference to the spline
  /// representation of the solution
  inline const Variable &u() const { return u_; }

  /// @brief Returns a non-constant reference to the spline
  /// representation of the solution
  inline Variable &u() { return u_; }

private:
  /// @brief Returns the geometry map collocation points
  ///
  /// In the default implementation the collocation points are the Greville
  /// abscissae in the interior of the domain and on the boundary
  /// faces. This behavior can be changed by overriding this virtual
  /// function in a derived class.
  template <size_t... Is>
  geometryMap_collPts_type
  geometryMap_collPts(enum collPts collPts, std::index_sequence<Is...>) const {
    geometryMap_collPts_type collPts_;

    switch (collPts) {

    case collPts::greville:
      // Get Greville abscissae inside the domain and at the boundary
      ((std::get<Is>(collPts_.first) =
            std::get<Is>(G_).greville(/* interior */ false)),
       ...);

      // Get Greville abscissae at the domain
      ((std::get<Is>(collPts_.second) = std::get<Is>(G_.boundary()).greville()),
       ...);
      break;

    case collPts::greville_interior:
      // Get Greville abscissae inside the domain
      ((std::get<Is>(collPts_.first) =
            std::get<Is>(G_).greville(/* interior */ true)),
       ...);

      // Get Greville abscissae at the domain
      ((std::get<Is>(collPts_.second) = std::get<Is>(G_.boundary()).greville()),
       ...);
      break;

    case collPts::greville_ref1:
      // Get Greville abscissae inside the domain and at the boundary
      ((std::get<Is>(collPts_.first) =
            std::get<Is>(G_).clone().uniform_refine().greville(
                /* interior */ false)),
       ...);

      // Get Greville abscissae at the domain
      ((std::get<Is>(collPts_.second) =
            std::get<Is>(G_.boundary()).clone().uniform_refine().greville()),
       ...);
      break;

    case collPts::greville_interior_ref1:
      // Get Greville abscissae inside the domain
      ((std::get<Is>(collPts_.first) =
            std::get<Is>(G_).clone().uniform_refine().greville(
                /* interior */ true)),
       ...);

      // Get Greville abscissae at the domain
      ((std::get<Is>(collPts_.second) =
            std::get<Is>(G_.boundary()).clone().uniform_refine().greville()),
       ...);
      break;

    case collPts::greville_ref2:
      // Get Greville abscissae inside the domain and at the boundary
      ((std::get<Is>(collPts_.first) =
            std::get<Is>(G_).clone().uniform_refine(2, -1).greville(
                /* interior */ false)),
       ...);

      // Get Greville abscissae at the domain
      ((std::get<Is>(collPts_.second) = std::get<Is>(G_.boundary())
                                            .clone()
                                            .uniform_refine(2, -1)
                                            .greville()),
       ...);
      break;

    case collPts::greville_interior_ref2:
      // Get Greville abscissae inside the domain
      ((std::get<Is>(collPts_.first) =
            std::get<Is>(G_).clone().uniform_refine(2, -1).greville(
                /* interior */ true)),
       ...);

      // Get Greville abscissae at the domain
      ((std::get<Is>(collPts_.second) = std::get<Is>(G_.boundary())
                                            .clone()
                                            .uniform_refine(2, -1)
                                            .greville()),
       ...);
      break;

    default:
      throw std::runtime_error("Invalid collocation point specifier");
    }

    return collPts_;
  }

  /// @brief Returns the variable collocation points
  ///
  /// In the default implementation the collocation points are the Greville
  /// abscissae in the interior of the domain and on the boundary
  /// faces. This behavior can be changed by overriding this virtual
  /// function in a derived class.
  template <size_t... Is>
  variable_collPts_type variable_collPts(enum collPts collPts,
                                         std::index_sequence<Is...>) const {
    variable_collPts_type collPts_;

    switch (collPts) {

    case collPts::greville:
      // Get Greville abscissae inside the domain and at the boundary
      ((std::get<Is>(collPts_.first) =
            std::get<Is>(u_).greville(/* interior */ false)),
       ...);

      // Get Greville abscissae at the domain
      ((std::get<Is>(collPts_.second) = std::get<Is>(u_.boundary()).greville()),
       ...);
      break;

    case collPts::greville_interior:
      // Get Greville abscissae inside the domain and at the boundary
      ((std::get<Is>(collPts_.first) =
            std::get<Is>(u_).greville(/* interior */ true)),
       ...);

      // Get Greville abscissae at the domain
      ((std::get<Is>(collPts_.second) = std::get<Is>(u_.boundary()).greville()),
       ...);
      break;

    case collPts::greville_ref1:
      // Get Greville abscissae inside the domain and at the boundary
      ((std::get<Is>(collPts_.first) =
            std::get<Is>(u_).clone().uniform_refine().greville(
                /* interior */ false)),
       ...);

      // Get Greville abscissae at the domain
      ((std::get<Is>(collPts_.second) =
            std::get<Is>(u_.boundary()).clone().uniform_refine().greville()),
       ...);
      break;

    case collPts::greville_interior_ref1:
      // Get Greville abscissae inside the domain and at the boundary
      ((std::get<Is>(collPts_.first) =
            std::get<Is>(u_).clone().uniform_refine().greville(
                /* interior */ true)),
       ...);

      // Get Greville abscissae at the domain
      ((std::get<Is>(collPts_.second) =
            std::get<Is>(u_.boundary()).clone().uniform_refine().greville()),
       ...);
      break;

    case collPts::greville_ref2:
      // Get Greville abscissae inside the domain and at the boundary
      ((std::get<Is>(collPts_.first) =
            std::get<Is>(u_).clone().uniform_refine(2, -1).greville(
                /* interior */ false)),
       ...);

      // Get Greville abscissae at the domain
      ((std::get<Is>(collPts_.second) = std::get<Is>(u_.boundary())
                                            .clone()
                                            .uniform_refine(2, -1)
                                            .greville()),
       ...);
      break;

    case collPts::greville_interior_ref2:
      // Get Greville abscissae inside the domain and at the boundary
      ((std::get<Is>(collPts_.first) =
            std::get<Is>(u_).clone().uniform_refine(2, -1).greville(
                /* interior */ true)),
       ...);

      // Get Greville abscissae at the domain
      ((std::get<Is>(collPts_.second) = std::get<Is>(u_.boundary())
                                            .clone()
                                            .uniform_refine(2, -1)
                                            .greville()),
       ...);
      break;

    default:
      throw std::runtime_error("Invalid collocation point specifier");
    }

    return collPts_;
  }

public:
  /// @brief Returns the geometry map collocation points
  ///
  /// In the default implementation the collocation points are the Greville
  /// abscissae in the interior of the domain and on the boundary
  /// faces. This behavior can be changed by overriding this virtual
  /// function in a derived class.
  virtual geometryMap_collPts_type
  geometryMap_collPts(enum collPts collPts) const {
    if constexpr (GeometryMap::nspaces() == 1)

      switch (collPts) {

      case collPts::greville:
        return {G_.greville(/* interior */ false), G_.boundary().greville()};

      case collPts::greville_interior:
        return {G_.greville(/* interior */ true), G_.boundary().greville()};

      case collPts::greville_ref1:
        return {G_.clone().uniform_refine().greville(/* interior */ false),
                G_.clone().uniform_refine().boundary().greville()};

      case collPts::greville_interior_ref1:
        return {G_.clone().uniform_refine().greville(/* interior */ true),
                G_.clone().uniform_refine().boundary().greville()};

      case collPts::greville_ref2:
        return {G_.clone().uniform_refine(2, -1).greville(/* interior */ false),
                G_.clone().uniform_refine(2, -1).boundary().greville()};

      case collPts::greville_interior_ref2:
        return {G_.clone().uniform_refine(2, -1).greville(/* interior */ true),
                G_.clone().uniform_refine(2, -1).boundary().greville()};

      default:
        throw std::runtime_error("Invalid collocation point specifier");
      }

    else
      return geometryMap_collPts(
          collPts, std::make_index_sequence<GeometryMap::nspaces()>{});
  }

  /// @brief Returns the variable collocation points
  ///
  /// In the default implementation the collocation points are the Greville
  /// abscissae in the interior of the domain and on the boundary
  /// faces. This behavior can be changed by overriding this virtual
  /// function in a derived class.
  virtual variable_collPts_type variable_collPts(enum collPts collPts) const {
    if constexpr (Variable::nspaces() == 1)

      switch (collPts) {

      case collPts::greville:
        return {u_.greville(/* interior */ false), u_.boundary().greville()};

      case collPts::greville_interior:
        return {u_.greville(/* interior */ true), u_.boundary().greville()};

      case collPts::greville_ref1:
        return {u_.clone().uniform_refine().greville(/* interior */ false),
                u_.boundary().clone().uniform_refine().greville()};

      case collPts::greville_interior_ref1:
        return {u_.clone().uniform_refine().greville(/* interior */ true),
                u_.clone().uniform_refine().boundary().greville()};

      case collPts::greville_ref2:
        return {u_.clone().uniform_refine(2, -1).greville(/* interior */ false),
                u_.boundary().clone().uniform_refine(2, -1).greville()};

      case collPts::greville_interior_ref2:
        return {u_.clone().uniform_refine(2, -1).greville(/* interior */ true),
                u_.clone().uniform_refine(2, -1).boundary().greville()};

      default:
        throw std::runtime_error("Invalid collocation point specifier");
      }

    else
      return variable_collPts(collPts,
                              std::make_index_sequence<Variable::nspaces()>{});
  }
};

/// @brief IgA base class
///
/// This class implements the base functionality of IgA solvers and nets
template <typename GeometryMap, typename Variable>
class IgABase : public IgABaseNoRefData<GeometryMap, Variable> {
public:
  /// @brief Base type
  using Base = IgABaseNoRefData<GeometryMap, Variable>;

  /// @brief Value type
  using value_type = Base::value_type;

  /// @brief Type of the geometry map function space(s)
  using geometryMap_type = GeometryMap;

  /// @brief Type of the variable function space(s)
  using variable_type = Variable;

  /// @brief Type of the geometry map collocation points
  using geometryMap_collPts_type = Base::geometryMap_collPts_type;

  /// @brief Type of the variable collocation points
  using variable_collPts_type = Base::variable_collPts_type;

  /// @brief Indicates whether this class provides a geometry map
  bool static constexpr has_GeometryMap = true;

  /// @brief Indicates whether this class provides a reference solution
  bool static constexpr has_RefData = true;

  /// @brief Indicates whether this class provides a solution
  bool static constexpr has_Solution = true;

protected:
  /// @brief Spline representation of the reference data
  Variable f_;

private:
  /// @brief Constructor: number of spline coefficients (different for Geometry
  /// and Variable types)
  template <typename... GeometryMapSplines, size_t... Is,
            typename... VariableSplines, size_t... Js>
  IgABase(std::tuple<GeometryMapSplines...> geometryMap_splines,
          std::index_sequence<Is...>,
          std::tuple<VariableSplines...> variable_splines,
          std::index_sequence<Js...>,
          iganet::Options<value_type> options = iganet::Options<value_type>{})
      : // Construct the different spline objects individually
        Base(geometryMap_splines, variable_splines, options),
        f_(std::get<Js>(variable_splines)..., init::zeros, options) {}

public:
  /// @brief Default constructor
  explicit IgABase(
      iganet::Options<value_type> options = iganet::Options<value_type>{})
      : Base::G_(), f_(), Base::u_() {}

  /// @brief Constructor: number of spline coefficients (same for geometry map
  /// and variables)
  template <typename... Splines>
  IgABase(std::tuple<Splines...> splines,
          iganet::Options<value_type> options = iganet::Options<value_type>{})
      : IgABase(splines, splines, options) {}

  /// @brief Constructor: number of spline coefficients (different for
  /// geometry map and variables)
  template <typename... GeometryMapSplines, typename... VariableSplines>
  IgABase(std::tuple<GeometryMapSplines...> geometryMap_splines,
          std::tuple<VariableSplines...> variable_splines,
          iganet::Options<value_type> options = iganet::Options<value_type>{})
      : IgABase(geometryMap_splines,
                std::make_index_sequence<sizeof...(GeometryMapSplines)>{},
                variable_splines,
                std::make_index_sequence<sizeof...(VariableSplines)>{},
                options) {}

  /// @brief Returns a constant reference to the spline
  /// representation of the reference data
  inline const Variable &f() const { return f_; }

  /// @brief Returns a non-constant reference to the spline
  /// representation of the reference data
  inline Variable &f() { return f_; }
};

/// @brief IgA dataset base class
///
/// This class implements the specialization of the torch dataset
/// class for IgA solvers and nets
class IgADatasetBase {
protected:
  /// @brief Reads a function space from file
  template <typename T>
  inline void read_from_xml(std::string location, T &obj,
                            std::vector<torch::Tensor> &v) {

    std::filesystem::path path(location);

    if (std::filesystem::exists(path)) {
      if (std::filesystem::is_regular_file(path)) {
        try {
          pugi::xml_document doc;
          doc.load_file(path.c_str());
          v.emplace_back(obj.from_xml(doc).as_tensor());
        } catch (...) {
        }
      } else if (std::filesystem::is_directory(path)) {
        for (const auto &file : std::filesystem::directory_iterator(path)) {
          if (file.is_regular_file() && file.path().extension() == ".xml") {
            try {
              pugi::xml_document doc;
              doc.load_file(file.path().c_str());
              v.emplace_back(obj.from_xml(doc).as_tensor());
            } catch (...) {
            }
          }
        }
      } else
        throw std::runtime_error(
            "The path refers to neither a file nor a directory");
    } else
      throw std::runtime_error("The path does not exist");
  }
};

/// @brief IgA dataset class
///
/// This class implements the specialization of the torch dataset
/// class for IgA solvers and nets
/// @{
template <bool solution = false> class IgADataset;

template <>
class IgADataset<false>
    : public IgADatasetBase,
      public torch::data::Dataset<
          IgADataset<false>,
          torch::data::Example<torch::Tensor, torch::data::example::NoTarget>> {
private:
  /// @brief Vector of tensors representing the geometry maps
  std::vector<torch::Tensor> G_;

  /// @brief Vector of tensors representing the reference data
  std::vector<torch::Tensor> f_;

public:
  /// @brief Example type
  using example_type =
      torch::data::Example<torch::Tensor, torch::data::example::NoTarget>;

  /// @brief Adds a geometry map from file
  /// @{
  template <typename T> void add_geometryMap(T &obj, std::string location) {
    read_from_xml(location, obj, G_);
  }

  template <typename T> void add_geometryMap(T &&obj, std::string location) {
    read_from_xml(location, obj, G_);
  }
  /// @}

  /// @brief Adds a geometry map from XML object
  /// @{
  template <typename T>
  void add_geometryMap(T &obj, const pugi::xml_document &doc, int id = 0,
                       std::string label = "") {
    G_.emplace_back(obj.from_xml(doc.child("xml"), id, label).as_tensor());
  }

  template <typename T>
  void add_geometryMap(T &&obj, const pugi::xml_document &doc, int id = 0,
                       std::string label = "") {
    G_.emplace_back(obj.from_xml(doc.child("xml"), id, label).as_tensor());
  }
  /// @}

  /// @brief Adds a geometry map from XML node
  /// @{
  template <typename T>
  void add_geometryMap(T &obj, const pugi::xml_node &root, int id = 0,
                       std::string label = "") {
    G_.emplace_back(obj.from_xml(root, id, label).as_tensor());
  }

  template <typename T>
  void add_geometryMap(T &&obj, const pugi::xml_node &root, int id = 0,
                       std::string label = "") {
    G_.emplace_back(obj.from_xml(root, id, label).as_tensor());
  }
  /// @}

  /// @brief Adds a reference data set from file
  /// @{
  template <typename T> void add_referenceData(T &obj, std::string location) {
    read_from_xml(location, obj, f_);
  }

  template <typename T> void add_referenceData(T &&obj, std::string location) {
    read_from_xml(location, obj, f_);
  }
  /// @}

  /// @brief Adds a reference data set from XML object
  /// @{
  template <typename T>
  void add_referenceData(T &obj, const pugi::xml_document &doc, int id = 0,
                         std::string label = "") {
    f_.emplace_back(obj.from_xml(doc.child("xml"), id, label).as_tensor());
  }

  template <typename T>
  void add_referenceData(T &&obj, const pugi::xml_document &doc, int id = 0,
                         std::string label = "") {
    f_.emplace_back(obj.from_xml(doc.child("xml"), id, label).as_tensor());
  }
  ///@}

  /// @brief Adds a reference data set from XML node
  /// @{
  template <typename T>
  void add_referenceData(T &obj, const pugi::xml_node &root, int id = 0,
                         std::string label = "") {
    f_.emplace_back(obj.from_xml(root, id, label).as_tensor());
  }

  template <typename T>
  void add_referenceData(T &&obj, const pugi::xml_node &root, int id = 0,
                         std::string label = "") {
    f_.emplace_back(obj.from_xml(root, id, label).as_tensor());
  }
  /// @}

  /// @brief Adds a reference data set from XML node
  /// @{
  template <typename T, typename Func>
  void add_referenceData(T &obj, Func func) {
    f_.emplace_back(obj.transform(func).as_tensor());
  }

  template <typename T, typename Func>
  void add_referenceData(T &&obj, Func func) {
    f_.emplace_back(obj.transform(func).as_tensor());
  }
  /// @}

  /// @brief Returns the data set at location index
  inline example_type get(std::size_t index) override {

    std::size_t geo_index = index / (f_.empty() ? 1 : f_.size());
    std::size_t ref_index = index - geo_index * f_.size();

    if (!G_.empty()) {
      if (!f_.empty())
        return torch::cat({G_.at(geo_index), f_.at(ref_index)});
      else
        return G_.at(geo_index);
    } else {
      if (!f_.empty())
        return f_.at(ref_index);
      else
        throw std::runtime_error("No geometry maps and reference data");
    }
  };

  // @brief Return the total size of the data set
  inline torch::optional<std::size_t> size() const override {
    return (G_.empty() ? 1 : G_.size()) * (f_.empty() ? 1 : f_.size());
  }
};

template <>
class IgADataset<true>
    : public IgADatasetBase,
      public torch::data::Dataset<IgADataset<true>, torch::data::Example<>> {
private:
  /// @brief Vector of tensors representing the geometry maps
  std::vector<torch::Tensor> G_;

  /// @brief Vector of tensors representing the reference data
  std::vector<torch::Tensor> f_;

  /// @brief Vector of tensors representing the solution data
  std::vector<torch::Tensor> u_;

public:
  /// @brief Adds a geometry map from file
  /// @{
  template <typename T> void add_geometryMap(T &obj, std::string location) {
    read_from_xml(location, obj, G_);
  }

  template <typename T> void add_geometryMap(T &&obj, std::string location) {
    read_from_xml(location, obj, G_);
  }
  /// @}

  /// @brief Adds a geometry map from XML object
  /// @{
  template <typename T>
  void add_geometryMap(T &obj, const pugi::xml_document &doc, int id = 0,
                       std::string label = "") {
    G_.emplace_back(obj.from_xml(doc.child("xml"), id, label).as_tensor());
  }

  template <typename T>
  void add_geometryMap(T &&obj, const pugi::xml_document &doc, int id = 0,
                       std::string label = "") {
    G_.emplace_back(obj.from_xml(doc.child("xml"), id, label).as_tensor());
  }
  /// @}

  /// @brief Adds a geometry map from XML node
  /// @{
  template <typename T>
  void add_geometryMap(T &obj, const pugi::xml_node &root, int id = 0,
                       std::string label = "") {
    G_.emplace_back(obj.from_xml(root, id, label).as_tensor());
  }

  template <typename T>
  void add_geometryMap(T &&obj, const pugi::xml_node &root, int id = 0,
                       std::string label = "") {
    G_.emplace_back(obj.from_xml(root, id, label).as_tensor());
  }
  /// @}

  /// @brief Adds a reference data set from file
  /// @{
  template <typename T> void add_referenceData(T &obj, std::string location) {
    read_from_xml(location, obj, f_);
  }

  template <typename T> void add_referenceData(T &&obj, std::string location) {
    read_from_xml(location, obj, f_);
  }
  /// @}

  /// @brief Adds a reference data set from XML object
  /// @{
  template <typename T>
  void add_referenceData(T &obj, const pugi::xml_document &doc, int id = 0,
                         std::string label = "") {
    f_.emplace_back(obj.from_xml(doc.child("xml"), id, label).as_tensor());
  }

  template <typename T>
  void add_referenceData(T &&obj, const pugi::xml_document &doc, int id = 0,
                         std::string label = "") {
    f_.emplace_back(obj.from_xml(doc.child("xml"), id, label).as_tensor());
  }
  ///@}

  /// @brief Adds a reference data set from XML node
  /// @{
  template <typename T>
  void add_referenceData(T &obj, const pugi::xml_node &root, int id = 0,
                         std::string label = "") {
    f_.emplace_back(obj.from_xml(root, id, label).as_tensor());
  }

  template <typename T>
  void add_referenceData(T &&obj, const pugi::xml_node &root, int id = 0,
                         std::string label = "") {
    f_.emplace_back(obj.from_xml(root, id, label).as_tensor());
  }
  /// @}

  /// @brief Adds a reference data set from XML node
  /// @{
  template <typename T, typename Func>
  void add_referenceData(T &obj, Func func) {
    f_.emplace_back(obj.transform(func).as_tensor());
  }

  template <typename T, typename Func>
  void add_referenceData(T &&obj, Func func) {
    f_.emplace_back(obj.transform(func).as_tensor());
  }
  /// @}

  /// @brief Adds a solution from file
  /// @{
  template <typename T> void add_solution(T &obj, std::string location) {
    read_from_xml(location, obj, u_);
  }

  template <typename T> void add_solution(T &&obj, std::string location) {
    read_from_xml(location, obj, u_);
  }
  /// @}

  /// @brief Adds a solution from XML object
  /// @{
  template <typename T>
  void add_solution(T &obj, const pugi::xml_document &doc, int id = 0,
                    std::string label = "") {
    u_.emplace_back(obj.from_xml(doc.child("xml"), id, label).as_tensor());
  }

  template <typename T>
  void add_solution(T &&obj, const pugi::xml_document &doc, int id = 0,
                    std::string label = "") {
    u_.emplace_back(obj.from_xml(doc.child("xml"), id, label).as_tensor());
  }
  /// @}

  /// @brief Adds a solution from XML node
  /// @{
  template <typename T>
  void add_solution(T &obj, const pugi::xml_node &root, int id = 0,
                    std::string label = "") {
    u_.emplace_back(obj.from_xml(root, id, label).as_tensor());
  }

  template <typename T>
  void add_solution(T &&obj, const pugi::xml_node &root, int id = 0,
                    std::string label = "") {
    u_.emplace_back(obj.from_xml(root, id, label).as_tensor());
  }
  /// @}

  /// @brief Returns the data set at location index
  inline torch::data::Example<> get(std::size_t index) override {

    std::size_t geo_index = index / (f_.empty() ? 1 : f_.size());
    std::size_t ref_index = index - geo_index * f_.size();

    if (!G_.empty()) {
      if (!f_.empty())
        return {torch::cat({G_.at(geo_index), f_.at(ref_index)}), u_.at(index)};
      else
        return {G_.at(geo_index), u_.at(index)};
    } else {
      if (!f_.empty())
        return {f_.at(ref_index), u_.at(index)};
      else
        throw std::runtime_error("No geometry maps and reference data");
    }
  };

  // @brief Return the total size of the data set
  inline torch::optional<std::size_t> size() const override {
    return (G_.empty() ? 1 : G_.size()) * (f_.empty() ? 1 : f_.size());
  }
};
/// @}

} // namespace iganet
