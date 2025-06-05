/**
   @file webapps/models/gismo/GismoPoissonModel.hpp

   @brief G+Smo Poisson model

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

/// @brief G+Smo Poisson model
template <short_t d, typename T>
class GismoPoissonModel : public GismoPdeModel<d, T> {

  static_assert(d >= 1 && d <= 3, "Spatial dimension must be between 1 and 3");

private:
  /// @brief Base class
  using Base = GismoPdeModel<d, T>;

  /// @brief Type of the geometry mapping
  using geometryMap_type = typename gismo::gsExprAssembler<T>::geometryMap;

  /// @brief Type of the variable
  using variable_type = typename gismo::gsExprAssembler<T>::variable;

  /// @brief Type of the function space
  using space_type = typename gismo::gsExprAssembler<T>::space;

  /// @brief Type of the solution
  using solution_type = typename gismo::gsExprAssembler<T>::solution;

  /// @brief Multi-patch basis
  gismo::gsMultiBasis<T> basis_;

  /// @brief Boundary conditions
  gismo::gsBoundaryConditions<T> bc_;

  /// @brief Right-hand side function
  gismo::gsFunctionExpr<T> rhsFunc_;

  /// @brief Right-hand side function defined on parametric domain (default
  /// false)
  bool rhsFuncParametric_;

  /// @brief Boundary condition look-up table
  GismoBoundaryConditionMap<T> bcMap_;
  
  /// @brief Expression assembler
  gismo::gsExprAssembler<T> assembler_;

  /// @brief Solve the Poisson problem
  void solve() {

    // Set up expression assembler
    auto G = assembler_.getMap(Base::geo_);
    auto u = assembler_.getSpace(basis_);

    // Impose boundary conditions
    u.setup(bc_, gismo::dirichlet::l2Projection, 0);

    // Set up system
    assembler_.initSystem();
    if (rhsFuncParametric_) {
      auto f = assembler_.getCoeff(rhsFunc_);
      assembler_.assemble(igrad(u, G) * igrad(u, G).tr() * meas(G) // matrix
                          ,
                          u * f * meas(G) // rhs vector
      );
    } else {
      auto f = assembler_.getCoeff(rhsFunc_, G);
      assembler_.assemble(igrad(u, G) * igrad(u, G).tr() * meas(G) // matrix
                          ,
                          u * f * meas(G) // rhs vector
      );
    }

    // Compute the Neumann terms defined on physical space
    auto bcNeumann = bc_.get("Neumann");
    if (!bcNeumann.empty()) {
      auto g = assembler_.getBdrFunction(G);
      assembler_.assembleBdr(bcNeumann, u * g * meas(G));
    }

    // Compute the Neumann terms defined on parametric space
    auto bcNeumannParametric = bc_.get("NeumannParametric");
    if (!bcNeumannParametric.empty()) {
      auto g = assembler_.getBdrFunction();
      assembler_.assembleBdr(bcNeumannParametric, u * g * meas(G));
    }

    // Solve system
    typename gismo::gsSparseSolver<T>::CGDiagonal solver;
    solver.compute(assembler_.matrix());

    gismo::gsMatrix<T> solutionVector;
    solution_type solution = assembler_.getSolution(u, solutionVector);
    solutionVector = solver.solve(assembler_.rhs());

    // Extract solution
    solution.extract(Base::solution_);
  }

public:
  /// @brief Default constructor
  GismoPoissonModel() = delete;

  /// @brief Constructor for equidistant knot vectors
  GismoPoissonModel(const std::array<short_t, d> degrees,
                    const std::array<int64_t, d> ncoeffs,
                    const std::array<int64_t, d> npatches,
                    const std::array<T, d> dimensions)
    : Base(degrees, ncoeffs, npatches, dimensions), basis_(Base::geo_, true),
      rhsFuncParametric_(true),
      rhsFunc_(d == 1   ? "2*pi^2*sin(pi*x)"
               : d == 2 ? "2*pi^2*sin(pi*x)*sin(pi*y)"
               : "2*pi^2*sin(pi*x)*sin(pi*y)*sin(pi*z)",
                 /* rhsFuncParametric_ == false */ d),
        assembler_(1, 1) {
    
    // Specify assembler options
    gismo::gsOptionList Aopt = gismo::gsExprAssembler<>::defaultOptions();

    // Set assembler options
    assembler_.setOptions(Aopt);

    // Set assembler basis
    assembler_.setIntegrationElements(basis_);

    // Initialize boundary conditions
    for (auto const &bdr : Base::geo_.boundaries()) {
      auto patch = bdr.patch;
      auto side = bdr.side();
      auto bc = bcMap_[patch][side] = { gismo::gsFunctionExpr<T>("0", d),
                                        gismo::condition_type::dirichlet,
                                        true };
    }

    // Set boundary conditions
    for (auto const & p : bcMap_) {
      std::size_t patch = p.first;
      
      for (auto const & bc : p.second) {
        auto side = static_cast<gismo::boundary::side>(bc.first);

        bc_.addCondition(patch, side, bc.second.type, bc.second.function, 0, bc.second.isParametric);  
      }
    }
    
    // Set geometry
    bc_.setGeoMap(Base::geo_);

    // Regenerate solution
    solve();
  }

  /// @brief Destructor
  ~GismoPoissonModel() {}

  /// @brief Returns the model's name
  std::string getName() const override {
    return "GismoPoisson" + std::to_string(d) + "d";
  }

  /// @brief Returns the model's description
  std::string getDescription() const override {
    return "G+Smo Poisson model in " + std::to_string(d) + " dimensions";
  };

  /// @brief Returns the model's outputs
  nlohmann::json getOutputs() const override {
    auto json = R"([{
           "name" : "Solution",
           "description" : "Solution of the Poisson equation",
           "type" : 1},{
           "name" : "Rhs",
           "description" : "Right-hand side function",
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
    auto add_json = [&json, &uiid]<typename Type, typename Value>(int patch,
                                                                  const std::string &name,
                                                                  const std::string &label,
                                                                  const std::string &group,
                                                                  const std::string &description,
                                                                  const Type &type,
                                                                  const Value &value) {
      nlohmann::json item;
      item["patch"] = patch;
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
      [&json, &uiid]<typename Type, typename Value, typename DefaultValue>(int patch,
            const std::string &name, const std::string &label,
            const std::string &group, const std::string &description,
            const Type &type, const Value &value,
            const DefaultValue &defaultValue) {
      nlohmann::json item;
      item["patch"] = patch;
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

    add_json(0, "rhs", "Rhs function", "rhs", "Right-hand side function", "text",
             rhsFunc_.expression(0));
    add_json(0, "rhs_parametric", "Parametric", "rhs",
             "Right-hand side function defined in parametric domain", "bool",
             rhsFuncParametric_);

    for (auto const & p : bcMap_) {
      std::size_t patch = p.first;
      
      for (auto const & bc : p.second) {
        auto side = static_cast<gismo::boundary::side>(bc.first);
        std::string str = *(GismoBoundarySideStrings<d>.begin()+side-1);

        add_json(patch, "bc["s + std::to_string(patch) + ":" + str + "]"s, "Value", str,
                 "Boundary value at the "s + str + " boundary of patch "s + std::to_string(patch), "text",
                 bc.second.function.expression(0));
        add_json(patch, "bc_parametric["s + std::to_string(patch) + ":" + str + "]", "Parametric", str,
                 "Boundary value at the "s + str +
                 " boundary of patch "s + std::to_string(patch) + "defined in parametric domain"s,
                 "bool", bc.second.isParametric);
        add_json_default(patch, "bc_type["s + std::to_string(patch) + ":" + str + "]", "Type", str,
                         "Type of boundary condition at the "s + str + " boundary of patch "s + std::to_string(patch), "select",
                         R"([ "Dirichlet", "Neumann" ])"_json,
                         bc.second.type == gismo::condition_type::dirichlet ? "Dirichlet" : "Neumann");        
      }
    }       

    return json;
  }

  /// @brief Updates the attributes of the model
  nlohmann::json updateAttribute(const std::string &patch,
                                 const std::string &component,
                                 const std::string &attribute,
                                 const nlohmann::json &json) override {

    bool updateBC(false);
    nlohmann::json result = R"({})"_json;

    for (auto & p : bcMap_) {
      std::size_t patch = p.first;
      
      for (auto & bc : p.second) {
        auto side = static_cast<gismo::boundary::side>(bc.first);
        std::string str = *(GismoBoundarySideStrings<d>.begin()+side-1);
    
        // bc_parametric[*]
        if (attribute == "bc_parametric["s + std::to_string(patch) + ":" + str + "]"s) {
          if (!json.contains("data"))
            throw InvalidModelAttributeException();
          if (!json["data"].contains("bc_parametric["s + std::to_string(patch) + ":" + str + "]"s))
            throw InvalidModelAttributeException();

          bc.second.isParametric =
            json["data"]["bc_parametric["s + std::to_string(patch) + ":" + str + "]"s].template get<bool>();

          bc.second.function =
            gismo::give(gismo::gsFunctionExpr<T>(bc.second.function.expression(0),
                                          bc.second.isParametric ? d : 3));

          updateBC = true;
          break;
        }

        // bc_type[*]
        else if (attribute == "bc_type["s + std::to_string(patch) + ":" + str + "]"s) {
          if (!json.contains("data"))
            throw InvalidModelAttributeException();
          if (!json["data"].contains("bc_type["s + std::to_string(patch) + ":" + str + "]"s))
            throw InvalidModelAttributeException();

          std::string bc_type =
            json["data"]["bc_type["s + std::to_string(patch) + ":" + str + "]"s].template get<std::string>();

          if (bc_type == "Dirichlet")
            bc.second.type = gismo::condition_type::type::dirichlet;
          else if (bc_type == "Neumann")
            bc.second.type = gismo::condition_type::type::neumann;
          else
            throw InvalidModelAttributeException();

          updateBC = true;
          break;
        }

        // bc[*]
        else if (attribute == "bc["s + std::to_string(patch) + ":" + str + "]"s) {
          if (!json.contains("data"))
            throw InvalidModelAttributeException();
          if (!json["data"].contains("bc["s + std::to_string(patch) + ":" + str + "]"s))
            throw InvalidModelAttributeException();

          bc.second.function =gismo::give(gismo::gsFunctionExpr<T>(
            json["data"]["bc["s + std::to_string(patch) + ":" + str + "]"s].template get<std::string>(),
            bc.second.isParametric ? d : 3));

          updateBC = true;
          break;
        }
      }
    }

    // rhs_parametric
    if (attribute == "rhs_parametric") {
      if (!json.contains("data"))
        throw InvalidModelAttributeException();
      if (!json["data"].contains("rhs_parametric"))
        throw InvalidModelAttributeException();
      rhsFuncParametric_ = json["data"]["rhs_parametric"].get<bool>();
      rhsFunc_ = gismo::give(gismo::gsFunctionExpr<T>(rhsFunc_.expression(0),
                                               rhsFuncParametric_ ? d : 3));
    }

    // rhs
    else if (attribute == "rhs") {
      if (!json.contains("data"))
        throw InvalidModelAttributeException();
      if (!json["data"].contains("rhs"))
        throw InvalidModelAttributeException();
      rhsFunc_ = gismo::give(gismo::gsFunctionExpr<T>(
          json["data"]["rhs"].get<std::string>(), rhsFuncParametric_ ? d : 3));
    }

    else if (!updateBC)
      result = Base::updateAttribute(patch, component, attribute, json);

    if (updateBC) {
      bc_.clear();

      // Set boundary conditions
      for (auto & p : bcMap_) {
        std::size_t patch = p.first;
      
        for (auto & bc : p.second) {
          auto side = static_cast<gismo::boundary::side>(bc.first);

          bc_.addCondition(patch, side, bc.second.type, bc.second.function, 0, bc.second.isParametric);           
        }
      }
    }

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
    
    if (component == "Solution" || component == "Rhs") {

      // Get grid resolution
      gismo::gsVector<unsigned> npts(Base::geo_.parDim());
      npts.setConstant(25);

      if (json.contains("data"))
        if (json["data"].contains("resolution")) {
          auto res = json["data"]["resolution"].get<std::array<int64_t, d>>();

          for (std::size_t i = 0; i < d; ++i)
            npts(i) = res[i];
        }

      if (component == "Solution" ||
          component == "Rhs" && !rhsFuncParametric_) {

        // Create uniform grid in physical domain
        gismo::gsMatrix<T> ab = Base::geo_.patch(patchIndex).support();
        gismo::gsVector<T> a = ab.col(0);
        gismo::gsVector<T> b = ab.col(1);
        gismo::gsMatrix<T> pts = gismo::gsPointGrid(a, b, npts);

        if (component == "Solution") {
          gismo::gsMatrix<T> eval = Base::solution_.patch(patchIndex).eval(pts);
          return utils::to_json(eval, true, false);
        } else {
          gismo::gsMatrix<T> eval = rhsFunc_.eval(Base::geo_.patch(patchIndex).eval(pts));
          return utils::to_json(eval, true, false);
        }

      } else {

        // Create uniform grid in parametric domain
        gismo::gsMatrix<T> ab = Base::geo_.patch(patchIndex).parameterRange();
        gismo::gsVector<T> a = ab.col(0);
        gismo::gsVector<T> b = ab.col(1);
        gismo::gsMatrix<T> pts = gismo::gsPointGrid(a, b, npts);

        gismo::gsMatrix<T> eval = rhsFunc_.eval(pts);
        return utils::to_json(eval, true, false);
      }
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

    // Set assembler basis
    assembler_.setIntegrationElements(basis_);

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

    // Set assembler basis
    assembler_.setIntegrationElements(basis_);

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
    
    // Set assembler basis
    assembler_.setIntegrationElements(basis_);
    
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

    // throw std::runtime_error("Removing patches is not yet implemented in G+Smo");

    // Regenerate solution
    solve();
  }
};

} // namespace webapp
} // namespace iganet
