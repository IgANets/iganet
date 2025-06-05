/**
   @file webapps/models/gismo/GismoPdeModel.hpp

   @brief G+Smo PDE model

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <GismoGeometryModel.hpp>

namespace iganet {

namespace webapp {

/// @brief G+Smo boundary condition POD
template<typename T>
struct GismoBoundaryCondition {
  /// @brief Boundary function expression
  gismo::gsFunctionExpr<T> function;

  /// @brief Boundary type
  gismo::condition_type::type type;

  /// @brief Flag that indicates whether the boundary conditions is
  /// imposed on the parametric of physical domain
  bool isParametric;
};

/// @brief G+Smo boundary condition look-up table
template<typename T>
using GismoBoundaryConditionMap = std::map<int, std::map<int, GismoBoundaryCondition<T>>>;
  
/// @brief G+Smo function POD
template<typename T>
struct GismoFunction {
  /// @brief Function expression
  gismo::gsFunctionExpr<T> function;

  /// @brief Flag that indicates whether the function expression is
  /// imposed on the parametric of physical domain
  bool isParametric;
};
  
/// @brief G+Smo function look-up table
template<typename T>
using GismoFunctionMap = std::map<int, std::map<int, GismoFunction<T>>>;
  
/// @brief G+Smo PDE model
template <short_t d, class T>
class GismoPdeModel : public GismoGeometryModel<d, T> {

public:
  /// @brief Constructors
  using GismoGeometryModel<d, T>::GismoGeometryModel;

  /// @brief Serializes the model to JSON
  nlohmann::json to_json(const std::string &patch, const std::string &component,
                         const std::string &attribute) const override {

    if (component == "solution") {
      
      if (patch == "" && attribute == "") {

        // Return solution as multipatch structure
        return utils::to_json(solution_);
      }

      else if (patch != "") {

        // Return individual patch of the solution
        std::size_t patchIndex(0);

        try {
          patchIndex = stoi(patch);
        } catch (...) {
          // Invalid patchIndex          
          return R"({ INVALID REQUEST })"_json;
        }

        if (patchIndex >= solution_.nPatches())
          return R"({ INVALID REQUEST })"_json;

        if (attribute == "") {

          // Return all attributes
          return utils::to_json(solution_.patch(patchIndex));

        } else {
          
          nlohmann::json json;

          // Return an individual attribute
          if (attribute == "degrees") {
            json["degrees"] = nlohmann::json::array();

            for (std::size_t i = 0; i < solution_.patch(patchIndex).parDim(); ++i)
              json["degrees"].push_back(solution_.patch(patchIndex).degree(i));
          }

          else if (attribute == "geoDim")
            json["geoDim"] = solution_.patch(patchIndex).geoDim();

          else if (attribute == "parDim")
            json["parDim"] = solution_.patch(patchIndex).parDim();

          else if (attribute == "ncoeffs") {
            json["ncoeffs"] = nlohmann::json::array();

            if (auto bspline =
                dynamic_cast<const gismo::gsBSpline<T> *>(&solution_.patch(patchIndex)))
              for (std::size_t i = 0; i < bspline->parDim(); ++i)
                json["ncoeffs"].push_back(bspline->basis().size(i));
            else if (auto bspline = dynamic_cast<const gismo::gsTensorBSpline<d, T> *>(
                                                                                &solution_.patch(patchIndex)))
              for (std::size_t i = 0; i < bspline->parDim(); ++i)
                json["ncoeffs"].push_back(bspline->basis().size(i));
            else
              return R"({ INVALID REQUEST })"_json;
          }

          else if (attribute == "nknots") {
            json["nknots"] = nlohmann::json::array();

            if (auto bspline =
                dynamic_cast<const gismo::gsBSpline<T> *>(&solution_.patch(patchIndex)))
              for (std::size_t i = 0; i < bspline->parDim(); ++i)
                json["nknots"].push_back(bspline->knots(i).size());
            else if (auto bspline = dynamic_cast<const gismo::gsTensorBSpline<d, T> *>(
                                                                                &solution_.patch(patchIndex)))
              for (std::size_t i = 0; i < bspline->parDim(); ++i)
                json["nknots"].push_back(bspline->knots(i).size());
            else
              return R"({ INVALID REQUEST })"_json;
          }

          else if (attribute == "coeffs") {

            if (auto bspline =
                dynamic_cast<const gismo::gsBSpline<T> *>(&solution_.patch(patchIndex)))
              json["coeffs"] = utils::to_json(bspline->coefs());
            else if (auto bspline = dynamic_cast<const gismo::gsTensorBSpline<d, T> *>(
                                                                                &solution_.patch(patchIndex)))
              json["coeffs"] = utils::to_json(bspline->coefs());
            else
              return R"({ INVALID REQUEST })"_json;

          }

          else if (attribute == "knots") {
            json["knots"] = nlohmann::json::array();

            if (auto bspline =
                dynamic_cast<const gismo::gsBSpline<T> *>(&solution_.patch(patchIndex)))
              for (std::size_t i = 0; i < bspline->parDim(); ++i)
                json["knots"].push_back(bspline->knots(i));
            else if (auto bspline = dynamic_cast<const gismo::gsTensorBSpline<d, T> *>(
                                                                                &solution_.patch(patchIndex)))
              for (std::size_t i = 0; i < bspline->parDim(); ++i)
                json["knots"].push_back(bspline->knots(i));
            else
              return R"({ INVALID REQUEST })"_json;
          }

          else
            // Invalid attribute
            return R"({ INVALID REQUEST })"_json;

          return json;
        }
      }

      else
        return R"({ INVALID REQUEST })"_json;
    }

    // Handle component != "solution"
    return GismoGeometryModel<d, T>::to_json(patch, component, attribute);
  }

protected:
  /// @brief Solution
  gismo::gsMultiPatch<T> solution_;
};

} // namespace webapp
} // namespace iganet
