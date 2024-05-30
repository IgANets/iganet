/**
   @file webapps/models/gismo/GismoPoissonModel.hpp

   @brief G+Smo Poisson model

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <GismoPdeModel.hpp>

namespace iganet {

namespace webapp {

/// @brief G+Smo Poisson model
template <short_t d, typename T>
class GismoPoissonModel : public GismoPdeModel<d, T>,
                          public ModelEval,
                          public ModelParameters {

private:
  /// @brief Base class
  using Base = GismoPdeModel<d, T>;

  /// @brief Type of the geometry mapping
  using geometryMap_type = typename gsExprAssembler<T>::geometryMap;

  /// @brief Type of the variable
  using variable_type = typename gsExprAssembler<T>::variable;

  /// @brief Type of the function space
  using space_type = typename gsExprAssembler<T>::space;

  /// @brief Type of the solution
  using solution_type = typename gsExprAssembler<T>::solution;

  /// @brief Multi-patch basis
  gsMultiBasis<T> basis_;

  /// @brief Boundary conditions
  gsBoundaryConditions<T> bc_;

  /// @brief Right-hand side values
  gsFunctionExpr<T> rhsFunc_;

  /// @brief Boundary values
  std::array<gsFunctionExpr<T>, 2 * d> bcFunc_;

  /// @brief Expression assembler
  gsExprAssembler<T> assembler_;

  /// @brief Solution
  gsMultiPatch<T> solution_;

  /// @brief Solve the Poisson problem
  void solve() {

    // Set up expression assembler
    auto G = assembler_.getMap(Base::geo_);
    auto u = assembler_.getSpace(basis_);
    auto f = assembler_.getCoeff(rhsFunc_, G);

    // Impose boundary conditions
    u.setup(bc_, gismo::dirichlet::l2Projection, 0);

    // Set up system
    assembler_.initSystem();
    assembler_.assemble(igrad(u, G) * igrad(u, G).tr() * meas(G) // matrix
                        ,
                        u * f * meas(G) // rhs vector
    );

    // Solve system
    typename gismo::gsSparseSolver<T>::CGDiagonal solver;
    solver.compute(assembler_.matrix());

    gsMatrix<T> solutionVector;
    solution_type solution = assembler_.getSolution(u, solutionVector);
    solutionVector = solver.solve(assembler_.rhs());

    // Extract solution
    solution.extract(solution_);
  }

public:
  /// @brief Default constructor
  GismoPoissonModel() = delete;

  /// @brief Constructor for equidistant knot vectors
  GismoPoissonModel(const std::array<short_t, d> degrees,
                    const std::array<int64_t, d> ncoeffs,
                    const std::array<int64_t, d> npatches)
      : Base(degrees, ncoeffs, npatches), basis_(Base::geo_, true),
        rhsFunc_("2*pi^2*sin(pi*x)*sin(pi*y)", 3), assembler_(1, 1) {
    // Specify assembler options
    gsOptionList Aopt;

    Aopt.addInt("DirichletStrategy",
                "Method for enforcement of Dirichlet BCs [11..14]", 11);
    Aopt.addInt("DirichletValues",
                "Method for computation of Dirichlet DoF values [100..103]",
                101);
    Aopt.addInt("InterfaceStrategy",
                "Method of treatment of patch interfaces [0..3]", 1);
    Aopt.addReal(
        "bdA", "Estimated nonzeros per column of the matrix: bdA*deg + bdB", 2);
    Aopt.addInt(
        "bdB", "Estimated nonzeros per column of the matrix: bdA*deg + bdB", 1);
    Aopt.addReal(
        "bdO",
        "Overhead of sparse mem. allocation: (1+bdO)(bdA*deg + bdB) [0..1]",
        0.333);
    Aopt.addReal("quA", "Number of quadrature points: quA*deg + quB", 1);
    Aopt.addInt("quB", "Number of quadrature points: quA*deg + quB", 1);
    Aopt.addInt("quRule", "Quadrature rule [1:GaussLegendre, 2:GaussLobatto]",
                1);

    // Set assembler options
    assembler_.setOptions(Aopt);

    // Set assembler basis
    assembler_.setIntegrationElements(basis_);

    // Set boundary conditions
    for (short_t i = 0; i < 2 * d; ++i) {
      if constexpr (d == 1)
        bcFunc_[i] = gismo::give(gsFunctionExpr<T>("sin(pi*x)", 3));
      else if constexpr (d == 2)
        bcFunc_[i] = gismo::give(gsFunctionExpr<T>("sin(pi*x)*sin(pi*y)", 3));
      else if constexpr (d == 3)
        bcFunc_[i] =
            gismo::give(gsFunctionExpr<T>("sin(pi*x)*sin(pi*y)*sin(pi*z)", 3));
      else if constexpr (d == 4)
        bcFunc_[i] = gismo::give(
            gsFunctionExpr<T>("sin(pi*x)*sin(pi*y)*sin(pi*z)*sin(pi*t)", 4));

      bc_.addCondition(i + 1, gismo::condition_type::dirichlet, &bcFunc_[i]);
    }

    // Set geometry
    bc_.setGeoMap(Base::geo_);

    // Generate solution
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
    return R"([{
           "name" : "Solution",
           "description" : "Solution of the Poisson equation",
           "type" : 1},{
           "name" : "RHS",
           "description" : "Right-hand side function",
           "type" : 1}])"_json;
  }

  /// @brief Returns the model's parameters
  nlohmann::json getParameters() const override {

    if constexpr (d == 1)
      return R"([{
         "name" : "bc_east",
         "description" : "Boundary condition at the east boundary",
             "type" : "text",
             "value" : "0",
             "default" : "0",
             "uiid" : 0},{
         "name" : "bc_west",
         "description" : "Boundary condition at the west boundary",
             "type" : "text",
             "value" : "0",
             "default" : "0",
             "uiid" : 1},{
         "name" : "rhs",
         "description" : "Right-hand side function",
             "type" : "text",
             "value" : "0",
             "default" : "0",
             "uiid" : 2}])"_json;
    else if constexpr (d == 2)
      return R"([{
         "name" : "bc_north",
         "description" : "Boundary condition at the north boundary",
             "type" : "text",
             "value" : "0",
             "default" : "0",
             "uiid" : 0},{
         "name" : "bc_east",
         "description" : "Boundary condition at the east boundary",
             "type" : "text",
             "value" : "0",
             "default" : "0",
             "uiid" : 1},{
         "name" : "bc_south",
         "description" : "Boundary condition at the south boundary",
             "type" : "text",
             "value" : "0",
             "default" : "0",
             "uiid" : 2},{
         "name" : "bc_west",
         "description" : "Boundary condition at the west boundary",
             "type" : "text",
             "value" : "0",
             "default" : "0",
             "uiid" : 3},{
         "name" : "rhs",
         "description" : "Right-hand side function",
             "type" : "text",
             "value" : "0",
             "default" : "0",
             "uiid" : 4}])"_json;
    else if constexpr (d == 3)
      return R"([{
         "name" : "bc_north",
         "description" : "Boundary condition at the north boundary",
             "type" : "text",
             "value" : "0",
             "default" : "0",
             "uiid" : 0},{
         "name" : "bc_east",
         "description" : "Boundary condition at the east boundary",
             "type" : "text",
             "value" : "0",
             "default" : "0",
             "uiid" : 1},{
         "name" : "bc_south",
         "description" : "Boundary condition at the south boundary",
             "type" : "text",
             "value" : "0",
             "default" : "0",
             "uiid" : 2},{
         "name" : "bc_west",
         "description" : "Boundary condition at the west boundary",
             "type" : "text",
             "value" : "0",
             "default" : "0",
             "uiid" : 3},{
         "name" : "bc_front",
         "description" : "Boundary condition at the front boundary",
             "type" : "text",
             "value" : "0",
             "default" : "0",
             "uiid" : 4},{
         "name" : "bc_back",
         "description" : "Boundary condition at the back boundary",
             "type" : "text",
             "value" : "0",
             "default" : "0",
             "uiid" : 5},{
         "name" : "rhs",
         "description" : "Right-hand side function",
             "type" : "text",
             "value" : "0",
             "default" : "0",
             "uiid" : 6}])"_json;
    else if constexpr (d == 4)
      return R"([{
         "name" : "bc_north",
         "description" : "Boundary condition at the north boundary",
             "type" : "text",
             "value" : "0",
             "default" : "0",
             "uiid" : 0},{
         "name" : "bc_east",
         "description" : "Boundary condition at the east boundary",
             "type" : "text",
             "value" : "0",
             "default" : "0",
             "uiid" : 1},{
         "name" : "bc_south",
         "description" : "Boundary condition at the south boundary",
             "type" : "text",
             "value" : "0",
             "default" : "0",
             "uiid" : 2},{
         "name" : "bc_west",
         "description" : "Boundary condition at the west boundary",
             "type" : "text",
             "value" : "0",
             "default" : "0",
             "uiid" : 3},{
         "name" : "bc_front",
         "description" : "Boundary condition at the front boundary",
             "type" : "text",
             "value" : "0",
             "default" : "0",
             "uiid" : 4},{
         "name" : "bc_back",
         "description" : "Boundary condition at the back boundary",
             "type" : "text",
             "value" : "0",
             "default" : "0",
             "uiid" : 5},{
         "name" : "bc_stime",
         "description" : "Boundary condition at the start-time boundary",
             "type" : "text",
             "value" : "0",
             "default" : "0",
             "uiid" : 6},{
         "name" : "bc_etime",
         "description" : "Boundary condition at the end-time boundary",
             "type" : "text",
             "value" : "0",
             "default" : "0",
             "uiid" : 7},{
         "name" : "rhs",
         "description" : "Right-hand side function",
             "type" : "text",
             "value" : "0",
             "default" : "0",
             "uiid" : 8}])"_json;
    else
      return R"({ INVALID REQUEST })"_json;
  }

  /// @brief Updates the attributes of the model
  nlohmann::json updateAttribute(const std::string &component,
                                 const std::string &attribute,
                                 const nlohmann::json &json) override {

    nlohmann::json result = R"({})"_json;

    if (attribute == "bc_north") {
      if (!json.contains("data"))
        throw InvalidModelAttributeException();
      if (!json["data"].contains("bc_north"))
        throw InvalidModelAttributeException();

      bcFunc_[gismo::boundary::north] =
          gsFunctionExpr<T>(json["data"]["bc_north"].get<std::string>(), d);
    }

    else if (attribute == "bc_east") {
      if (!json.contains("data"))
        throw InvalidModelAttributeException();
      if (!json["data"].contains("bc_east"))
        throw InvalidModelAttributeException();

      bcFunc_[gismo::boundary::east] =
          gsFunctionExpr<T>(json["data"]["bc_east"].get<std::string>(), d);
    }

    else if (attribute == "bc_south") {
      if (!json.contains("data"))
        throw InvalidModelAttributeException();
      if (!json["data"].contains("bc_south"))
        throw InvalidModelAttributeException();

      bcFunc_[gismo::boundary::south] =
          gsFunctionExpr<T>(json["data"]["bc_south"].get<std::string>(), d);
    }

    else if (attribute == "bc_west") {
      if (!json.contains("data"))
        throw InvalidModelAttributeException();
      if (!json["data"].contains("bc_west"))
        throw InvalidModelAttributeException();

      bcFunc_[gismo::boundary::west] =
          gsFunctionExpr<T>(json["data"]["bc_west"].get<std::string>(), d);
    }

    else if (attribute == "bc_front") {
      if (!json.contains("data"))
        throw InvalidModelAttributeException();
      if (!json["data"].contains("bc_front"))
        throw InvalidModelAttributeException();

      bcFunc_[gismo::boundary::front] =
          gsFunctionExpr<T>(json["data"]["bc_front"].get<std::string>(), d);
    }

    else if (attribute == "bc_back") {
      if (!json.contains("data"))
        throw InvalidModelAttributeException();
      if (!json["data"].contains("bc_back"))
        throw InvalidModelAttributeException();

      bcFunc_[gismo::boundary::back] =
          gsFunctionExpr<T>(json["data"]["bc_back"].get<std::string>(), d);
    }

    else if (attribute == "bc_stime") {
      if (!json.contains("data"))
        throw InvalidModelAttributeException();
      if (!json["data"].contains("bc_stime"))
        throw InvalidModelAttributeException();

      bcFunc_[gismo::boundary::stime] =
          gsFunctionExpr<T>(json["data"]["bc_stime"].get<std::string>(), d);
    }

    else if (attribute == "bc_etime") {
      if (!json.contains("data"))
        throw InvalidModelAttributeException();
      if (!json["data"].contains("bc_etime"))
        throw InvalidModelAttributeException();

      bcFunc_[gismo::boundary::etime] =
          gsFunctionExpr<T>(json["data"]["bc_etime"].get<std::string>(), d);
    }

    else if (attribute == "rhs") {
      if (!json.contains("data"))
        throw InvalidModelAttributeException();
      if (!json["data"].contains("rhs"))
        throw InvalidModelAttributeException();

      rhsFunc_ = gsFunctionExpr<T>(json["data"]["rhs"].get<std::string>(), d);
    }

    else
      result = Base::updateAttribute(component, attribute, json);

    // Solve updated problem
    solve();

    return result;
  }

  /// @brief Evaluates the model
  nlohmann::json eval(const std::string &component,
                      const nlohmann::json &json) const override {

    // Create uniform grid
    gsMatrix<T> ab = Base::geo_.patch(0).support();
    gsVector<T> a = ab.col(0);
    gsVector<T> b = ab.col(1);

    gsVector<unsigned> np(Base::geo_.parDim());
    np.setConstant(25);

    if (json.contains("data"))
      if (json["data"].contains("resolution")) {
        auto res = json["data"]["resolution"].get<std::array<int64_t, d>>();

        for (std::size_t i = 0; i < d; ++i)
          np(i) = res[i];
      }

    // Uniform parameters for evaluation
    gsMatrix<T> pts = gsPointGrid(a, b, np);

    if (component == "Solution") {
      gsMatrix<T> eval = solution_.patch(0).eval(pts);
      return utils::to_json(eval, true);
    } else if (component == "RHS") {
      gsMatrix<T> eval = rhsFunc_.eval(pts);
      return utils::to_json(eval, true);
    } else
      return R"({ INVALID REQUEST })"_json;
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

    int num = 1, dim = -1;

    if (json.contains("data")) {
      if (json["data"].contains("num"))
        num = json["data"]["num"].get<int>();

      if (json["data"].contains("dim"))
        dim = json["data"]["dim"].get<int>();
    }

    // Degree elevate basis of solution space
    basis_.basis(0).degreeElevate(num, dim);

    // Set assembler basis
    assembler_.setIntegrationElements(basis_);

    // Generate solution
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

    int num = 1, dim = -1;

    if (json.contains("data")) {
      if (json["data"].contains("num"))
        num = json["data"]["num"].get<int>();

      if (json["data"].contains("dim"))
        dim = json["data"]["dim"].get<int>();
    }

    // Degree increase basis of solution space
    basis_.basis(0).degreeIncrease(num, dim);

    // Set assembler basis
    assembler_.setIntegrationElements(basis_);

    // Generate solution
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

    int num = 1, dim = -1;

    if (json.contains("data")) {
      if (json["data"].contains("num"))
        num = json["data"]["num"].get<int>();

      if (json["data"].contains("dim"))
        dim = json["data"]["dim"].get<int>();
    }

    // Refine basis of solution space
    basis_.basis(0).uniformRefine(num, 1, dim);

    // Set assembler basis
    assembler_.setIntegrationElements(basis_);

    // Generate solution
    solve();
  }
};

} // namespace webapp
} // namespace iganet
