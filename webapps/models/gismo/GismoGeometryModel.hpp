/**
   @file webapps/models/gismo/GismoGeometryModel.hpp

   @brief G+Smo geometry model

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <GismoModel.hpp>

namespace iganet {

namespace webapp {

/// @brief G+Smo geometry model
template <short_t d, class T>
class GismoGeometryModel : public GismoModel<T>,
                           public ModelAddPatch,
                           public ModelElevate,
                           public ModelEval,
                           public ModelIncrease,
                           public ModelRefine,
                           public ModelReparameterize,
                           public ModelRemovePatch,
                           public ModelXML {

  static_assert(d >= 1 && d <= 4, "Spatial dimension must be between 1 and 4");

protected:
  /// @brief Multi-patch geometry
  gismo::gsMultiPatch<T> geo_;

public:
  /// @brief Default constructor
  GismoGeometryModel() = default;

  /// @brief Constructor for equidistant knot vectors
  GismoGeometryModel(const std::array<short_t, d> degrees,
                     const std::array<int64_t, d> ncoeffs,
                     const std::array<int64_t, d> npatches,
                     const std::array<T, d> dimensions)
      : GismoModel<T>() {

    if constexpr (d == 1) {
      gismo::gsKnotVector<T> KV0(0, 1, ncoeffs[0] - degrees[0] - 1, degrees[0] + 1);

      gismo::gsMatrix<T> C(ncoeffs[0], 3), P;

      for (int64_t i = 0; i < ncoeffs[0]; ++i) {
        C(i, 0) = ((T)i) / (T)(ncoeffs[0] - 1);
        C(i, 1) = (T)0;
        C(i, 2) = (T)0;
      }

      for (int64_t i = 0; i < npatches[0]; ++i) {
        P = C;

        P.col(0) *= dimensions[0] / npatches[0];

        P.col(0).array() += (T)(i) / (T)npatches[0];

        geo_.addPatch(gismo::gsBSpline<T>(KV0, give(P)));
      }
      geo_.computeTopology();

    } else if constexpr (d == 2) {
      gismo::gsKnotVector<T> KV0(0, 1, ncoeffs[0] - degrees[0] - 1, degrees[0] + 1);
      gismo::gsKnotVector<T> KV1(0, 1, ncoeffs[1] - degrees[1] - 1, degrees[1] + 1);

      gismo::gsMatrix<T> C(ncoeffs[0] * ncoeffs[1], 3), P;

      int64_t r = 0;
      for (int64_t j = 0; j < ncoeffs[1]; ++j)
        for (int64_t i = 0; i < ncoeffs[0]; ++i) {
          C(r, 0) = ((T)i) / (T)(ncoeffs[0] - 1);
          C(r, 1) = ((T)j) / (T)(ncoeffs[1] - 1);
          C(r, 2) = (T)0;
          ++r;
        }

      for (int64_t j = 0; j < npatches[1]; ++j)
        for (int64_t i = 0; i < npatches[0]; ++i) {
          P = C;

          P.col(0) *= dimensions[0] / (T)npatches[0];
          P.col(1) *= dimensions[1] / (T)npatches[1];

          P.col(0).array() += (T)(i) / (T)npatches[0];
          P.col(1).array() += (T)(j) / (T)npatches[1];

          geo_.addPatch(gismo::gsTensorBSpline<2, T>(KV0, KV1, give(P)));
        }
      geo_.computeTopology();

    } else if constexpr (d == 3) {
      gismo::gsKnotVector<T> KV0(0, 1, ncoeffs[0] - degrees[0] - 1, degrees[0] + 1);
      gismo::gsKnotVector<T> KV1(0, 1, ncoeffs[1] - degrees[1] - 1, degrees[1] + 1);
      gismo::gsKnotVector<T> KV2(0, 1, ncoeffs[2] - degrees[2] - 1, degrees[2] + 1);

      gismo::gsMatrix<T> C(ncoeffs[0] * ncoeffs[1] * ncoeffs[2], 3), P;

      int64_t r = 0;
      for (int64_t k = 0; k < ncoeffs[2]; ++k)
        for (int64_t j = 0; j < ncoeffs[1]; ++j)
          for (int64_t i = 0; i < ncoeffs[0]; ++i) {
            C(r, 0) = ((T)i) / (T)(ncoeffs[0] - 1);
            C(r, 1) = ((T)j) / (T)(ncoeffs[1] - 1);
            C(r, 2) = ((T)k) / (T)(ncoeffs[2] - 1);
            ++r;
          }

      for (int64_t k = 0; k < npatches[2]; ++k)
        for (int64_t j = 0; j < npatches[1]; ++j)
          for (int64_t i = 0; i < npatches[0]; ++i) {
            P = C;

            P.col(0) *= dimensions[0] / npatches[0];
            P.col(1) *= dimensions[1] / npatches[1];
            P.col(2) *= dimensions[2] / npatches[2];

            P.col(0).array() += (T)(i) / (T)npatches[0];
            P.col(1).array() += (T)(j) / (T)npatches[1];
            P.col(2).array() += (T)(k) / (T)npatches[2];

            geo_.addPatch(gismo::gsTensorBSpline<3, T>(KV0, KV1, KV2, give(P)));
          }
      geo_.computeTopology();

    } else if constexpr (d == 4) {
      gismo::gsKnotVector<T> KV0(0, 1, ncoeffs[0] - degrees[0] - 1, degrees[0] + 1);
      gismo::gsKnotVector<T> KV1(0, 1, ncoeffs[1] - degrees[1] - 1, degrees[1] + 1);
      gismo::gsKnotVector<T> KV2(0, 1, ncoeffs[2] - degrees[2] - 1, degrees[2] + 1);
      gismo::gsKnotVector<T> KV3(0, 1, ncoeffs[3] - degrees[3] - 1, degrees[3] + 1);

      gismo::gsMatrix<T> C(ncoeffs[0] * ncoeffs[1] * ncoeffs[2] * ncoeffs[3], 4), P;

      int64_t r = 0;
      for (int64_t l = 0; l < ncoeffs[3]; ++l)
        for (int64_t k = 0; k < ncoeffs[2]; ++k)
          for (int64_t j = 0; j < ncoeffs[1]; ++j)
            for (int64_t i = 0; i < ncoeffs[0]; ++i) {
              C(r, 0) = ((T)i) / (T)(ncoeffs[0] - 1);
              C(r, 1) = ((T)j) / (T)(ncoeffs[1] - 1);
              C(r, 2) = ((T)k) / (T)(ncoeffs[2] - 1);
              C(r, 3) = ((T)l) / (T)(ncoeffs[3] - 1);
              ++r;
            }

      for (int64_t l = 0; l < npatches[3]; ++l)
        for (int64_t k = 0; k < npatches[2]; ++k)
          for (int64_t j = 0; j < npatches[1]; ++j)
            for (int64_t i = 0; i < npatches[0]; ++i) {
              P = C;

              P.col(0) *= dimensions[0] / npatches[0];
              P.col(1) *= dimensions[1] / npatches[1];
              P.col(2) *= dimensions[2] / npatches[2];
              P.col(3) *= dimensions[3] / npatches[3];

              P.col(0).array() += (T)(i) / (T)npatches[0];
              P.col(1).array() += (T)(j) / (T)npatches[1];
              P.col(2).array() += (T)(k) / (T)npatches[2];
              P.col(3).array() += (T)(l) / (T)npatches[3];

              geo_.addPatch(gismo::gsTensorBSpline<4, T>(KV0, KV1, KV2, KV3, give(P)));
            }
      geo_.computeTopology();
    }
  }

  /// @brief Constructor from XML
  GismoGeometryModel(const pugi::xml_node root)
    : GismoModel<T>() {
  }  

  /// @brief Returns the model's options
  nlohmann::json getOptions() const override {
    if constexpr (d == 1)
      return R"([{
             "name" : "npatches",
             "label" : "Number of patches",
             "description" : "Number of patches per spatial dimension",
             "type" : ["int"],
             "value" : [1],
             "default" : [1],
             "uiid" : 0},{
             "name" : "degree",
             "label" : "Spline degree",
             "description" : "Spline degree",
             "type" : ["int"],
             "value" : [2],
             "default" : [2],
             "uiid" : 1},{
             "name" : "ncoeffs",
             "label" : "Number of coefficients",
             "description" : "Number of coefficients per parametric dimension",
             "type" : ["int"],
             "value" : [3],
             "default" : [3],
             "uiid" : 2},{
             "name" : "dimension",
             "label" : "Dimension [width]",
             "description" : "Dimension in physical space",
             "type" : ["float"],
             "value" : [1.0],
             "default" : [1.0],
             "uiid" : 3}])"_json;

    else if constexpr (d == 2)
      return R"([{
             "name" : "npatches",
             "label" : "Number of patches",
             "description" : "Number of patches per spatial dimension",
             "type" : ["int","int"],
             "value" : [1,1],
             "default" : [1,1],
             "uiid" : 0},{
             "name" : "degrees",
             "label" : "Spline degrees",
             "description" : "Spline degrees per parametric dimension",
             "type" : ["int","int"],
             "value" : [2,2],
             "default" : [2,2],
             "uiid" : 1},{
             "name" : "ncoeffs",
             "label" : "Number of coefficients",
             "description" : "Number of coefficients per parametric dimension",
             "type" : ["int","int"],
             "value" : [3,3],
             "default" : [3,3],
             "uiid" : 2},{
             "name" : "dimensions",
             "label" : "Dimensions [width, height]",
             "description" : "Dimensions in physical space",
             "type" : ["float", "float"],
             "value" : [1.0, 1.0],
             "default" : [1.0, 1.0],
             "uiid" : 3}])"_json;

    else if constexpr (d == 3)
      return R"([{
             "name" : "npatches",
             "label" : "Number of patches",
             "description" : "Number of patches per spatial dimension",
             "type" : ["int","int","int"],
             "value" : [1,1,1],
             "default" : [1,1,1],
             "uiid" : 0},{
             "name" : "degrees",
             "label" : "Spline degrees",
             "description" : "Spline degrees per parametric dimension",
             "type" : ["int","int","int"],
             "value" : [2,2,2],
             "default" : [2,2,2],
             "uiid" : 1},{
             "name" : "ncoeffs",
             "label" : "Number of coefficients",
             "description" : "Number of coefficients per parametric dimension",
             "type" : ["int","int","int"],
             "value" : [3,3,3],
             "default" : [3,3,3],
             "uiid" : 2},{
             "name" : "dimensions",
             "label" : "Dimensions [width, height, depth]",
             "description" : "Dimensions in physical space",
             "type" : ["float", "float", "float"],
             "value" : [1.0, 1.0, 1.0],
             "default" : [1.0, 1.0, 1.0],
             "uiid" : 3}])"_json;

    else if constexpr (d == 4)
      return R"([{
             "name" : "npatches",
             "label" : "Number of patches",
             "description" : "Number of patches per spatial dimension",
             "type" : ["int","int","int","int"],
             "value" : [1,1,1,1],
             "default" : [1,1,1,1],
             "uiid" : 0},{
             "name" : "degrees",
             "label" : "Spline degrees",
             "description" : "Spline degrees per parametric dimension",
             "type" : ["int","int","int","int"],
             "value" : [2,2,2,2],
             "default" : [2,2,2,2],
             "uiid" : 1},{
             "name" : "ncoeffs",
             "label" : "Number of coefficients",
             "description" : "Number of coefficients per parametric dimension",
             "type" : ["int","int","int","int"],
             "value" : [3,3,3,3],
             "default" : [3,3,3,3],
             "uiid" : 2},{
             "name" : "dimensions",
             "label" : "Dimensions [width, height, depth, time]",
             "description" : "Dimensions in physical space",
             "type" : ["float", "float", "float", "float"],
             "value" : [1.0, 1.0, 1.0, 1.0],
             "default" : [1.0, 1.0, 1.0, 1.0],
             "uiid" : 3}])"_json;

    else
      return R"({ INVALID REQUEST })"_json;
  }

  /// @brief Returns the model's inputs
  nlohmann::json getInputs() const override {
    return R"([{
              "name" : "geometry",
              "description" : "Geometry",
              "type" : 2}])"_json;
  }

  /// @brief Returns the model's outputs
  nlohmann::json getOutputs() const override {
    return R"([{
           "name" : "ScaledJacobian",
           "description" : "Scaled Jacobian of the geometry mapping as quality measure for orthogonality",
           "type" : 1},{
           "name" : "UniformityMetric",
           "description" : "Uniformity metric quality measure for area distortion of the geometry map",
           "type" : 1}])"_json;
  }

  /// @brief Serializes the model to JSON
  nlohmann::json to_json(const std::string &patch, const std::string &component,
                         const std::string &attribute) const override {
    
    if (component == "geometry" || component == "") {

      if (patch == "" && attribute == "") {

        // Return solution as multipatch structure
        return utils::to_json(geo_);
      }

      else if (patch != "") {

        // Return individual patch of the solution
        int patchIndex(-1);

        try {
          patchIndex = stoi(patch);
        } catch (...) {
          // Invalid patchIndex
          return R"({ INVALID REQUEST })"_json;
        }

        if (patchIndex >= geo_.nPatches())
          return R"({ INVALID REQUEST })"_json;

        if (attribute == "") {

          // Return all attributes                  
          return utils::to_json(geo_.patch(patchIndex));

        } else {

          nlohmann::json json;

          // Return an individual attribute
          if (attribute == "degrees") {
            json["degrees"] = nlohmann::json::array();

            for (std::size_t i = 0; i < geo_.patch(patchIndex).parDim(); ++i)
              json["degrees"].push_back(geo_.patch(patchIndex).degree(i));
          }

          else if (attribute == "geoDim")
            json["geoDim"] = geo_.patch(patchIndex).geoDim();

          else if (attribute == "parDim")
            json["parDim"] = geo_.patch(patchIndex).parDim();

          else if (attribute == "ncoeffs") {
            json["ncoeffs"] = nlohmann::json::array();

            if (auto bspline =
                dynamic_cast<const gismo::gsBSpline<T> *>(&geo_.patch(patchIndex)))
              for (std::size_t i = 0; i < bspline->parDim(); ++i)
                json["ncoeffs"].push_back(bspline->basis().size(i));
            else if (auto bspline = dynamic_cast<const gismo::gsTensorBSpline<d, T> *>(
                                                                                &geo_.patch(patchIndex)))
              for (std::size_t i = 0; i < bspline->parDim(); ++i)
                json["ncoeffs"].push_back(bspline->basis().size(i));
            else
              return R"({ INVALID REQUEST })"_json;

          }

          else if (attribute == "nknots") {
            json["nknots"] = nlohmann::json::array();

            if (auto bspline =
                dynamic_cast<const gismo::gsBSpline<T> *>(&geo_.patch(patchIndex)))
              for (std::size_t i = 0; i < bspline->parDim(); ++i)
                json["nknots"].push_back(bspline->knots(i).size());
            else if (auto bspline = dynamic_cast<const gismo::gsTensorBSpline<d, T> *>(
                                                                                &geo_.patch(patchIndex)))
              for (std::size_t i = 0; i < bspline->parDim(); ++i)
                json["nknots"].push_back(bspline->knots(i).size());
            else
              return R"({ INVALID REQUEST })"_json;
          }

          else if (attribute == "coeffs") {

            if (auto bspline =
                dynamic_cast<const gismo::gsBSpline<T> *>(&geo_.patch(patchIndex)))
              json["coeffs"] = utils::to_json(bspline->coefs());
            else if (auto bspline = dynamic_cast<const gismo::gsTensorBSpline<d, T> *>(
                                                                                &geo_.patch(patchIndex)))
              json["coeffs"] = utils::to_json(bspline->coefs());
            else
              return R"({ INVALID REQUEST })"_json;

          }

          else if (attribute == "knots") {
            json["knots"] = nlohmann::json::array();

            if (auto bspline =
                dynamic_cast<const gismo::gsBSpline<T> *>(&geo_.patch(patchIndex)))
              for (std::size_t i = 0; i < bspline->parDim(); ++i)
                json["knots"].push_back(bspline->knots(i));
            else if (auto bspline = dynamic_cast<const gismo::gsTensorBSpline<d, T> *>(
                                                                                &geo_.patch(patchIndex)))
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

    // Handle component != "geometry"
    return GismoModel<T>::to_json(patch, component, attribute);
  }

  /// @brief Updates the attributes of the model
  nlohmann::json updateAttribute(const std::string &patch,
                                 const std::string &component,
                                 const std::string &attribute,
                                 const nlohmann::json &json) override {

    int patchIndex(-1);

    try {
      patchIndex = stoi(patch);
    } catch (...) {
      return R"({ INVALID REQUEST })"_json;
    }

    if (attribute == "coeffs") {
      if (!json.contains("data"))
        throw InvalidModelAttributeException();
      if (!json["data"].contains("indices") || !json["data"].contains("coeffs"))
        throw InvalidModelAttributeException();

      auto indices = json["data"]["indices"].get<std::vector<int64_t>>();
      auto ncoeffs = geo_.patch(patchIndex).coefs().rows();

      switch (geo_.geoDim()) {
      case (1): {
        auto coords = json["data"]["coeffs"].get<std::vector<std::tuple<T>>>();

        for (const auto &[index, coord] : iganet::utils::zip(indices, coords)) {
          if (index < 0 || index >= ncoeffs)
            throw IndexOutOfBoundsException();

          geo_.patch(patchIndex).coef(index, 0) = std::get<0>(coord);
        }
        break;
      }
      case (2): {
        auto coords =
            json["data"]["coeffs"].get<std::vector<std::tuple<T, T>>>();

        for (const auto &[index, coord] : iganet::utils::zip(indices, coords)) {
          if (index < 0 || index >= ncoeffs)
            throw IndexOutOfBoundsException();

          geo_.patch(patchIndex).coef(index, 0) = std::get<0>(coord);
          geo_.patch(patchIndex).coef(index, 1) = std::get<1>(coord);
        }
        break;
      }
      case (3): {
        auto coords =
            json["data"]["coeffs"].get<std::vector<std::tuple<T, T, T>>>();

        for (const auto &[index, coord] : iganet::utils::zip(indices, coords)) {
          if (index < 0 || index >= ncoeffs)
            throw IndexOutOfBoundsException();

          geo_.patch(patchIndex).coef(index, 0) = std::get<0>(coord);
          geo_.patch(patchIndex).coef(index, 1) = std::get<1>(coord);
          geo_.patch(patchIndex).coef(index, 2) = std::get<2>(coord);
        }
        break;
      }
      case (4): {
        auto coords =
            json["data"]["coeffs"].get<std::vector<std::tuple<T, T, T, T>>>();

        for (const auto &[index, coord] : iganet::utils::zip(indices, coords)) {
          if (index < 0 || index >= ncoeffs)
            throw IndexOutOfBoundsException();

          geo_.patch(patchIndex).coef(index, 0) = std::get<0>(coord);
          geo_.patch(patchIndex).coef(index, 1) = std::get<1>(coord);
          geo_.patch(patchIndex).coef(index, 2) = std::get<2>(coord);
          geo_.patch(patchIndex).coef(index, 3) = std::get<3>(coord);
        }
        break;
      }
      default:
        throw InvalidModelAttributeException();
      }

      return R"({})"_json;
    } else
      return GismoModel<T>::updateAttribute(patch, component, attribute, json);
  }

  /// @brief Evaluates the model
  nlohmann::json eval(const std::string &patch, const std::string &component,
                      const nlohmann::json &json) const override {

    int patchIndex(-1);

    try {
      patchIndex = stoi(patch);
    } catch (...) {
      return R"({ INVALID REQUEST })"_json;
    }

    if (component == "ScaledJacobian" || component == "UniformityMetric") {

      // Get grid resolution
      gismo::gsVector<unsigned> np(geo_.parDim());
      np.setConstant(25);

      if (json.contains("data"))
        if (json["data"].contains("resolution")) {
          auto res = json["data"]["resolution"].get<std::array<int64_t, d>>();

          for (std::size_t i = 0; i < d; ++i)
            np(i) = res[i];
        }

      // Create uniform grid in physical space
      gismo::gsMatrix<T> ab = geo_.patch(patchIndex).support();
      gismo::gsVector<T> a = ab.col(0);
      gismo::gsVector<T> b = ab.col(1);
      gismo::gsMatrix<T> pts = gismo::gsPointGrid(a, b, np);
      gismo::gsMatrix<T> eval(1, pts.cols());

      gismo::gsExprEvaluator<T> ev;
      gismo::gsMultiBasis<T> basis(geo_);
      ev.setIntegrationElements(basis);
      typename gismo::gsExprAssembler<T>::geometryMap G = ev.getMap(geo_);

      if (component == "ScaledJacobian") {

        int parDim = geo_.parDim();

        if (parDim == 2 && geo_.geoDim() == 3) {
          for (std::size_t i = 0; i < pts.cols(); i++) {
            auto jac = ev.eval(gismo::expr::jac(G), pts.col(i));
            eval(0, i) = jac.col(0).dot(jac.col(1));
            for (std::size_t j = 0; j < parDim; j++)
              eval(0, i) /= (jac.col(j).norm());
          }
        } else {
          for (std::size_t i = 0; i < pts.cols(); i++) {
            auto jac = ev.eval(gismo::expr::jac(G), pts.col(i));
            eval(0, i) = jac.determinant();
            for (std::size_t j = 0; j < parDim; j++)
              eval(0, i) /= (jac.col(j).norm());
          }
        }

        return utils::to_json(eval, true, true);
      }

      else if (component == "UniformityMetric") {

        T areaTotal = ev.integral(gismo::expr::meas(G));
        gismo::gsConstantFunction<T> areaConstFunc(areaTotal, geo_.parDim());
        auto area = ev.getVariable(areaConstFunc);
        auto expr = gismo::expr::pow((gismo::expr::meas(G) - area.val()) / area.val(), 2);

        for (std::size_t i = 0; i < pts.cols(); i++)
          eval(0, i) = ev.eval(expr, pts.col(i))(0);

        return utils::to_json(eval, true, true);
      } else
        return R"({ INVALID REQUEST })"_json;
    } else
      return R"({ INVALID REQUEST })"_json;
  }

  /// @brief Elevates the model's degrees, preserves smoothness
  void elevate(const nlohmann::json &json = NULL) override {
    int num(1), dim(-1), patchIndex(-1);

    if (json.contains("data")) {
      if (json["data"].contains("num"))
        num = json["data"]["num"].get<int>();

      if (json["data"].contains("dim"))
        dim = json["data"]["dim"].get<int>();

      if (json["data"].contains("patch"))
        patchIndex = json["data"]["patch"].get<int>();
    }

    if (patchIndex == -1)
      geo_.degreeElevate(num, dim);
    else
      geo_.patch(patchIndex).degreeElevate(num, dim);
  }

  /// @brief Increases the model's degrees, preserves multiplicity
  void increase(const nlohmann::json &json = NULL) override {
    int num(1), dim(-1), patchIndex(-1);

    if (json.contains("data")) {
      if (json["data"].contains("num"))
        num = json["data"]["num"].get<int>();

      if (json["data"].contains("dim"))
        dim = json["data"]["dim"].get<int>();

      if (json["data"].contains("patch"))
        patchIndex = json["data"]["patch"].get<int>();
    }

    if (patchIndex == -1)
      geo_.degreeIncrease(num, dim);
    else
      geo_.patch(patchIndex).degreeIncrease(num, dim);
  }

  /// @brief Refines the model
  void refine(const nlohmann::json &json = NULL) override {
    int num(1), dim(-1), patchIndex(-1);

    if (json.contains("data")) {
      if (json["data"].contains("num"))
        num = json["data"]["num"].get<int>();

      if (json["data"].contains("dim"))
        dim = json["data"]["dim"].get<int>();

      if (json["data"].contains("patch"))
        patchIndex = json["data"]["patch"].get<int>();
    }
    
    if (patchIndex == -1)
      geo_.uniformRefine(num, 1, dim);
    else
      geo_.patch(patchIndex).uniformRefine(num, 1, dim);
  }

  /// @brief Reparameterize the model
  void reparameterize(const nlohmann::json &json = NULL) override {
    std::string type("volume");
    int maxiter(200);
    T tol(1e-3);

    if (json.contains("data")) {
      if (json["data"].contains("type"))
        type = json["data"]["type"].get<std::string>();

      if (json["data"].contains("maxiter"))
        maxiter = json["data"]["maxiter"].get<int>();

      if (json["data"].contains("tol"))
        tol = json["data"]["tol"].get<T>();
    }

    if (type == "surface") {

      // bivariate surface
      if (geo_.parDim() == 2) {
        gismo::gsHLBFGS<real_t> optimizer;
        optimizer.options().setReal("MinGradLen", 1e-6);
        optimizer.options().setReal("MinStepLen", 1e-6);
        optimizer.options().setInt("MaxIterations", 200);
        optimizer.options().setInt("Verbose", 0);

        for (auto& p : geo_) {
          gismo::gsMultiPatch<T> mp; mp.addPatch(*p);
          gismo::SurfaceReparameterization<T> reparam(mp, optimizer);
          *p = reparam.solve().patch(0);
        }        
        // gismo::gsMultiPatch<T> tempGeo_;
        // for (int i=0; i< geo_.nPatches(); ++i) {
        //   auto mp = gismo::gsMultiPatch(geo_.patch(i));
        //   gismo::SurfaceReparameterization<T> reparam(mp, optimizer);
        //   //geo_.setPatch(i, std::make_unique<gismo::gsPatch<T>>(reparam.solve().patch(0)));
        //   tempGeo_.addPatch(reparam.solve().patch(0));
        // }
        // geo_ = tempGeo_;
      }

      else if (geo_.parDim() == 3) {

        gismo::gsHLBFGS<real_t> optimizer;
        optimizer.options().setReal("MinGradLen", 1e-6);
        optimizer.options().setReal("MinStepLen", 1e-6);
        optimizer.options().setInt("MaxIterations", 200);
        optimizer.options().setInt("Verbose", 0);

        // trivariate surface
        //for (int i=1; i<=6; ++i) {
        //  gismo::SurfaceReparameterization<T> reparam(*geo_.boundary(i), optimizer);
        //  *geo_.boundary(i) = reparam.solve();
        // }

      }

    } else if (type == "volume") {

      if (geo_.parDim() == 2 && geo_.geoDim() == 3) {
        // bivariate surface
        geo_.embed(2);
        gismo::gsBarrierPatch<2, T> opt(geo_, false);
        opt.options().setInt("ParamMethod", 1); // penalty
        opt.options().setInt("Verbose", 0);
        opt.compute();
        geo_ = opt.result();
        geo_.embed(3);
      }

      else if (geo_.parDim() == 3 && geo_.geoDim() == 3) {
        // trivariate volume
        gismo::gsBarrierPatch<d, T> opt(geo_, true);
        opt.options().setInt("ParamMethod", 2); // penalty
        opt.options().setInt("Verbose", 0);
        opt.compute();
        geo_ = opt.result();
      }
    }
  }

  /// @brief Add new patch to the model
  void addPatch(const nlohmann::json &json = NULL) override {
    
    throw std::runtime_error("Adding patches is not yet implemented in G+Smo");
  }
  
  /// @brief Remove existing patch from the model
  void removePatch(const nlohmann::json &json = NULL) override {
    int patchIndex(-1);

    if (json.contains("data")) {      
      if (json["data"].contains("patch"))
        patchIndex = json["data"]["patch"].get<int>();
    }

    // if (patchIndex == -1)
    //   throw std::runtime_error("Invalid patch index");

    // throw std::runtime_error("Removing patches is not yet implemented in G+Smo");
  }
  
  /// @brief Imports the model from XML (as JSON object)
  void importXML(const std::string &patch,
                 const std::string &component,
                 const nlohmann::json &json, 
                 int id) override {

    if (json.contains("data")) {
      if (json["data"].contains("xml")) {

        std::string xml_str = json["data"]["xml"].get<std::string>();
        
        gismo::internal::gsXmlTree xml;
        xml.parse<0>(const_cast<char*>(xml_str.c_str()));

        importXML(patch, component, xml, id);

        return;
      }
    }

    throw std::runtime_error("No XML node in JSON object");
  }

  /// @brief Imports the model from XML (as XML object)
  void importXML(const std::string &patch,
                 const std::string &component,
                 const pugi::xml_node &xml,
                 int id) override {

    gsWarn << "Using generic importXML implementation\n";
    
    if (component == "geometry" || component == "") {

      if (patch == "") {
        
      } else {
        
      }
    }
    else
      throw std::runtime_error("Unsupported component");
  }

  /// @brief Imports the model from XML (as XML object) optimized for G+Smo
  void importXML(const std::string &patch,
                 const std::string &component,
                 const gismo::internal::gsXmlTree &xml,
                 int id) {

    if (component == "geometry" || component == "") {

      if (patch == "") {
        
        auto * geo = gismo::internal::gsXml<gismo::gsMultiPatch<T>>::getFirst(xml.getRoot());
        geo_ = give(*geo);
        delete geo;

      } else {

        auto * p = gismo::internal::gsXml<gismo::gsGeometry<T>>::getFirst(xml.getRoot());
        geo_.patch(stoi(patch)) = give(*p);
        delete p;
      }

      geo_.topology();
    }
    else
      throw std::runtime_error("Unsupported component");
  }

  /// @brief Exports the model to XML (as JSON object)
  nlohmann::json exportXML(const std::string &patch, const std::string &component, int id) override {
    gismo::internal::gsXmlTree xml;
    xml.makeRoot();

    exportXML(patch, component, xml, id);

    std::string xml_str;
    rapidxml::print(std::back_inserter(xml_str), xml, 0);

    return xml_str;

    // pugi::xml_document doc;
    // pugi::xml_node xml = doc.append_child("xml");
    // xml = exportXML(patch, component, xml, id);

    // // serialize to JSON
    // std::ostringstream oss;
    // doc.save(oss);

    // return oss.str();
  }

  /// @brief Exports the model to XML (as XML object)
  pugi::xml_node &exportXML(const std::string &patch,
                            const std::string &component,
                            pugi::xml_node &xml, 
                            int id) override {

    gsWarn << "Using generic exportXML implementation\n";
    
    if (component == "geometry" || component == "") {

      gismo::internal::gsXmlTree data;     
      data.makeRoot();

      gismo::internal::gsXmlNode *node = (patch == ""
                                   ?
                                          gismo::internal::gsXml<gismo::gsMultiPatch<T>>::put(geo_, data)
                                   :
                                          gismo::internal::gsXml<gismo::gsGeometry<T>>::put(geo_.patch(stoi(patch)), data));
      
      if (node)
          data.appendToRoot(node, -1);

      std::string xml_str;
      rapidxml::print(std::back_inserter(xml_str), data, 1);
      
      pugi::xml_document doc;
      pugi::xml_parse_result result = doc.load_string(xml_str.c_str());
      
      if (result)
        for (auto node : doc.first_child())
          xml.append_copy(node);      
    }
    else
      throw std::runtime_error("Unsupported component");

    return xml;
  }

  /// @brief Exports the model to XML (as XML object) optimized for G+Smo
  gismo::internal::gsXmlTree &exportXML(const std::string &patch,
                                 const std::string &component,
                                        gismo::internal::gsXmlTree &xml, 
                                 int id) {
    
    if (component == "geometry" || component == "") {

      gismo::internal::gsXmlNode *node = (patch == ""
                                   ?
                                          gismo::internal::gsXml<gismo::gsMultiPatch<T>>::put(geo_, xml)
                                   :
                                          gismo::internal::gsXml<gismo::gsGeometry<T>>::put(geo_.patch(stoi(patch)), xml));
      
      if (node)
        xml.appendToRoot(node, -1);
    }
    else
      throw std::runtime_error("Unsupported component");

    return xml;
  }

  
};

} // namespace webapp
} // namespace iganet
