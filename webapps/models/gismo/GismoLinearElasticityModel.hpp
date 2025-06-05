/**
   @file webapps/models/gismo/GismoLinearElasticityModel.hpp

   @brief G+Smo Linear elasticity model

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <GismoPdeModel.hpp>

using namespace std::string_literals;

namespace iganet {

namespace webapp {

/// @brief G+Smo Linear elasticity model
template <short_t d, typename T>
class GismoLinearElasticityModel : public GismoPdeModel<d, T> {

  static_assert(d >= 1 && d <= 3, "Spatial dimension must be between 1 and 3");

private:
  /// @brief Base class
  using Base = GismoPdeModel<d, T>;

  /// @brief Multi-patch basis
  gismo::gsMultiBasis<T> basis_;

  /// @brief Boundary conditions
  gismo::gsBoundaryConditions<T> bc_;

  /// @brief Right-hand side function
  gismo::gsFunctionExpr<T> rhsFunc_;
  gismo::gsFunctionExpr<T> loadFunc_;

  /// @brief Boundary condition values
  std::array<gismo::gsFunctionExpr<T>, 2 * d> bcFunc_;

  /// @brief Boundary condition type
  std::array<gismo::condition_type::type, 2 * d> bcType_;

  /// @brief Young's modulus
  T YoungsModulus_;

  /// @brief Poisson's ratio
  T PoissonsRatio_;

  /// @brief Solve the Linear elasticity problem
  void solve() {

    // Setup assembler
    gismo::gsElasticityAssembler<T> assembler(Base::geo_, basis_, bc_, rhsFunc_);
    assembler.options().setReal("YoungsModulus", YoungsModulus_);
    assembler.options().setReal("PoissonsRatio", PoissonsRatio_);
    assembler.options().setInt("MaterialLaw", gismo::material_law::hooke);
    assembler.options().setInt("DirichletStrategy", gismo::dirichlet::elimination);

    // Initialize assembler
    assembler.assemble();

    // Solve system
    typename gismo::gsSparseSolver<T>::CGDiagonal solver(assembler.matrix());
    gismo::gsMatrix<T> solution(solver.solve(assembler.rhs()));

    // Extract solution
    assembler.constructSolution(solution, assembler.allFixedDofs(),
                                Base::solution_);
  }

public:
  /// @brief Default constructor
  GismoLinearElasticityModel() = delete;

  /// @brief Constructor for equidistant knot vectors
  GismoLinearElasticityModel(const std::array<short_t, d> degrees,
                             const std::array<int64_t, d> ncoeffs,
                             const std::array<int64_t, d> npatches,
                             const std::array<T, d> dimensions)
      : Base(degrees, ncoeffs, npatches, dimensions), basis_(Base::geo_, true),
        YoungsModulus_(210e9), PoissonsRatio_(0.3), rhsFunc_("0", "0", "0", 3),
        loadFunc_("0", "0", "-1e5", 3) {

    // Set boundary conditions type and expression
    for (const auto &side : GismoBoundarySides<d>) {
      bcType_[side - 1] = gismo::condition_type::unknownType;
      bcFunc_[side - 1] = gismo::give(gismo::gsFunctionExpr<T>("0", "0", "0", 3));
      // bc_.addCondition(0, side, bcType_[side - 1], &bcFunc_[side - 1]);
    }

    bc_.addCondition(0, gismo::boundary::west, gismo::condition_type::dirichlet,
                     nullptr, 0);
    bc_.addCondition(0, gismo::boundary::west, gismo::condition_type::dirichlet,
                     nullptr, 1);
    bc_.addCondition(0, gismo::boundary::west, gismo::condition_type::dirichlet,
                     nullptr, 2);

    bc_.addCondition(0, gismo::boundary::east, gismo::condition_type::neumann,
                     &loadFunc_);

    // Set geometry
    bc_.setGeoMap(Base::geo_);

    // Regenerate solution
    solve();
  }

  /// @brief Destructor
  ~GismoLinearElasticityModel() {}

  /// @brief Returns the model's name
  std::string getName() const override {
    return "GismoLinearElasticity" + std::to_string(d) + "d";
  }

  /// @brief Returns the model's description
  std::string getDescription() const override {
    return "G+Smo linear elasticity model in " + std::to_string(d) +
           " dimensions";
  };

  /// @brief Returns the model's options
  nlohmann::json getOptions() const override {

    nlohmann::json json = Base::getOptions();

    return json;
  }

  /// @brief Returns the model's outputs
  nlohmann::json getOutputs() const override {
    auto json = R"([{
           "name" : "Displacement",
           "description" : "Displament magnitude",
           "type" : 1},{
           "name" : "Displacement_x",
           "description" : "Displacement x-component",
           "type" : 1},{
           "name" : "Displacement_y",
           "description" : "Displacement x-component",
           "type" : 1},{
           "name" : "Displacement_z",
           "description" : "Displacement z-component",
           "type" : 1}])"_json;

    for (auto const &output : Base::getOutputs())
      json.push_back(output);

    return json;
  }

  /// @brief Returns the model's parameters
  nlohmann::json getParameters() const override {

    auto json = nlohmann::json::array();
    int uiid = 0;

    // Lambda expression to add a JSON entry
    auto add_json = [&json, &uiid]<typename Type, typename Value>(
                        const std::string &name, const std::string &label,
                        const std::string &group,
                        const std::string &description, const Type &type,
                        const Value &value) {
      nlohmann::json item;
      item["name"] = name;
      item["label"] = label;
      item["description"] = description;
      item["group"] = group;
      item["type"] = type;
      item["value"] = value;
      item["default"] = value;
      item["uuid"] = uiid++;
      json.push_back(item);
    };

    // Lambda expression to add a JSON entry with different default type
    auto add_json_default =
        [&json, &uiid]<typename Type, typename Value, typename DefaultValue>(
            const std::string &name, const std::string &label,
            const std::string &group, const std::string &description,
            const Type &type, const Value &value,
            const DefaultValue &defaultValue) {
          nlohmann::json item;
          item["name"] = name;
          item["label"] = label;
          item["description"] = description;
          item["group"] = group;
          item["type"] = type;
          item["value"] = value;
          item["default"] = defaultValue;
          item["uuid"] = uiid++;
          json.push_back(item);
        };

    add_json("YoungModulus", "Young", "", "Young's modulus", "float",
             YoungsModulus_);
    add_json("PoissonRatio", "Poisson", "", "Poisson's ratio", "float",
             PoissonsRatio_);

    //    add_json("rhs", "Right-hand side function",
    //             std::vector<std::string>{"text", "text", "text"},
    //             std::vector<std::string>{rhsFunc_.expression(0),
    //                                      rhsFunc_.expression(1),
    //                                      rhsFunc_.expression(2)});

    return json;
  }

  /// @brief Updates the attributes of the model
  nlohmann::json updateAttribute(const std::string &patch,
                                 const std::string &component,
                                 const std::string &attribute,
                                 const nlohmann::json &json) override {

    nlohmann::json result = R"({})"_json;

    if (attribute == "YoungModulus") {
      if (!json.contains("data"))
        throw InvalidModelAttributeException();
      if (!json["data"].contains("YoungModulus"))
        throw InvalidModelAttributeException();

      YoungsModulus_ = json["data"]["YoungModulus"].get<T>();
    }

    else if (attribute == "PoissonRatio") {
      if (!json.contains("data"))
        throw InvalidModelAttributeException();
      if (!json["data"].contains("PoissonRatio"))
        throw InvalidModelAttributeException();

      PoissonsRatio_ = json["data"]["PoissonRatio"].get<T>();
    }

    else
      result = Base::updateAttribute(patch, component, attribute, json);

    // Solve updated problem
    solve();

    return result;
  }

  /// @brief Evaluates the model
  nlohmann::json eval(const std::string &patch, const std::string &component,
                      const nlohmann::json &json) const override {

    int patchIndex(-1);

    try {
      patchIndex = stoi(patch);
    } catch (...) {
      // Invalid patchIndex
      return R"({ INVALID REQUEST })"_json;
    }
    
    // Create uniform grid
    gismo::gsMatrix<T> ab = Base::geo_.patch(patchIndex).support();
    gismo::gsVector<T> a = ab.col(0);
    gismo::gsVector<T> b = ab.col(1);

    gismo::gsVector<unsigned> np(Base::geo_.parDim());
    np.setConstant(25);

    if (json.contains("data"))
      if (json["data"].contains("resolution")) {
        auto res = json["data"]["resolution"].get<std::array<int64_t, d>>();

        for (std::size_t i = 0; i < d; ++i)
          np(i) = res[i];
      }

    // Uniform parameters for evaluation
    gismo::gsMatrix<T> pts = gismo::gsPointGrid(a, b, np);
    gismo::gsMatrix<T> eval = Base::solution_.patch(patchIndex).eval(pts);

    if (component == "Displacement") {
      gismo::gsMatrix<T> result = eval.colwise().norm();
      return utils::to_json(result, true, false);
    } else if (component == "Displacement_x") {
      gismo::gsMatrix<T> result = eval.row(0);
      return utils::to_json(result, true, false);
    } else if (component == "Displacement_y") {
      gismo::gsMatrix<T> result = eval.row(1);
      return utils::to_json(result, true, false);
    } else if (component == "Displacement_z") {
      gismo::gsMatrix<T> result = eval.row(2);
      return utils::to_json(result, true, false);
    }

    else
      return Base::eval(patch, component, json);
  }

  /// @brief Elevates the model's degrees, preserves smoothness
  void elevate(const nlohmann::json &json = NULL) override {

    bool geometry = true;

    if (json.contains("data"))
      if (json["data"].contains("num"))
        geometry = json["data"]["geometry"].get<bool>();

    if (geometry) {
      // Elevate geometry
      Base::elevate(json);

      // Set geometry
      bc_.setGeoMap(Base::geo_);
    }

    int num(1), dim(-1), patchIndex(-1);

    if (json.contains("data")) {
      if (json["data"].contains("num"))
        num = json["data"]["num"].get<int>();

      if (json["data"].contains("dim"))
        dim = json["data"]["dim"].get<int>();

      if (json["data"].contains("patch"))
        patchIndex = json["data"]["patch"].get<int>();
    }

    // Degree elevate basis of solution space
    if (patchIndex == -1)
      basis_.degreeElevate(num, dim);
    else
      basis_.basis(patchIndex).degreeElevate(num, dim);

    // Regenerate solution
    solve();
  }

  /// @brief Increases the model's degrees, preserves multiplicity
  void increase(const nlohmann::json &json = NULL) override {

    bool geometry = true;

    if (json.contains("data"))
      if (json["data"].contains("num"))
        geometry = json["data"]["geometry"].get<bool>();

    if (geometry) {
      // Increase geometry
      Base::increase(json);

      // Set geometry
      bc_.setGeoMap(Base::geo_);
    }

    int num(1), dim(-1), patchIndex(-1);

    if (json.contains("data")) {
      if (json["data"].contains("num"))
        num = json["data"]["num"].get<int>();

      if (json["data"].contains("dim"))
        dim = json["data"]["dim"].get<int>();

      if (json["data"].contains("patch"))
        patchIndex = json["data"]["patch"].get<int>();
    }

    // Degree increase basis of solution space
    if (patchIndex == -1)
      basis_.degreeIncrease(num, dim);
    else
      basis_.basis(patchIndex).degreeIncrease(num, dim);

    // Regenerate solution
    solve();
  }

  /// @brief Refines the model
  void refine(const nlohmann::json &json = NULL) override {

    bool geometry = true;

    if (json.contains("data"))
      if (json["data"].contains("num"))
        geometry = json["data"]["geometry"].get<bool>();

    if (geometry) {
      // Refine geometry
      Base::refine(json);

      // Set geometry
      bc_.setGeoMap(Base::geo_);
    }

    int num(1), dim(-1), patchIndex(-1);

    if (json.contains("data")) {
      if (json["data"].contains("num"))
        num = json["data"]["num"].get<int>();

      if (json["data"].contains("dim"))
        dim = json["data"]["dim"].get<int>();

      if (json["data"].contains("patch"))
        patchIndex = json["data"]["patch"].get<int>();
    }

    // Refine basis of solution space
    if (patchIndex == -1)
      basis_.uniformRefine(num, 1, dim);
    else
      basis_.basis(patchIndex).uniformRefine(num, 1, dim);

    // Regenerate solution
    solve();
  }

  /// @brief Add new patch to the model
  void addPatch(const nlohmann::json &json = NULL) override {

    // Add patch from geometry
    Base::addPatch(json);

    // Set geometry
    bc_.setGeoMap(Base::geo_);
    
    throw std::runtime_error("Adding patches is not yet implemented in G+Smo");

    // Regenerate solution
    solve();
  }
  
  /// @brief Remove existing patch from the model
  void removePatch(const nlohmann::json &json = NULL) override {

    // Remove patch from geometry
    Base::removePatch(json);

    // Set geometry
    bc_.setGeoMap(Base::geo_);
    
    int patchIndex(-1);

    if (json.contains("data")) {      
      if (json["data"].contains("patch"))
        patchIndex = json["data"]["patch"].get<int>();
    }

    // if (patchIndex == -1)
    //   throw std::runtime_error("Invalid patch index");

    // throw std::runtime_error("Patch removal is not yet implemented in G+Smo");

    // Regenerate solution
    solve();
  }
};

} // namespace webapp
} // namespace iganet
