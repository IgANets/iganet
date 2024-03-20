/**
   @file webapps/models/gismo/GismoPdeModel.hpp

   @brief G+Smo PDE model

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <GismoModel.hpp>

namespace iganet {

namespace webapp {

/// @brief G+Smo PDE model
template <short_t d, class T>
class GismoPdeModel : public GismoModel<T>,
                      public ModelElevate,
                      public ModelIncrease,
                      public ModelRefine,
                      public ModelReparameterize {

protected:
  /// @brief Multi-patch geometry
  gsMultiPatch<T> geo_;

public:
  /// @brief Default constructor
  GismoPdeModel() = default;

  /// @brief Constructor for equidistant knot vectors
  GismoPdeModel(const std::array<short_t, d> degrees,
                const std::array<int64_t, d> ncoeffs,
                const std::array<int64_t, d> npatches)
      : GismoModel<T>() {
    if constexpr (d == 1) {
      gsKnotVector<T> KV0(0, 1, ncoeffs[0] - degrees[0] - 1, degrees[0] + 1);

      gsMatrix<T> C(ncoeffs[0], 3);
      for (int64_t i = 0; i < ncoeffs[0]; ++i) {
        C(i, 0) = ((T)i) / (T)(ncoeffs[0] - 1);
        C(i, 1) = (T)0;
        C(i, 2) = (T)0;
      }

      geo_.addPatch(gsBSpline<T>(give(KV0), give(C)));

    } else if constexpr (d == 2) {
      gsKnotVector<T> KV0(0, 1, ncoeffs[0] - degrees[0] - 1, degrees[0] + 1);
      gsKnotVector<T> KV1(0, 1, ncoeffs[1] - degrees[1] - 1, degrees[1] + 1);

      gsMatrix<T> C(ncoeffs[0] * ncoeffs[1], 3);

      int64_t r = 0;
      for (int64_t j = 0; j < ncoeffs[1]; ++j)
        for (int64_t i = 0; i < ncoeffs[0]; ++i) {
          C(r, 0) = ((T)i) / (T)(ncoeffs[0] - 1);
          C(r, 1) = ((T)j) / (T)(ncoeffs[1] - 1);
          C(r, 2) = (T)0;
          ++r;
        }

      geo_.addPatch(gsTensorBSpline<2, T>(give(KV0), give(KV1), give(C)));

    } else if constexpr (d == 3) {
      gsKnotVector<T> KV0(0, 1, ncoeffs[0] - degrees[0] - 1, degrees[0] + 1);
      gsKnotVector<T> KV1(0, 1, ncoeffs[1] - degrees[1] - 1, degrees[1] + 1);
      gsKnotVector<T> KV2(0, 1, ncoeffs[2] - degrees[2] - 1, degrees[2] + 1);

      gsMatrix<T> C(ncoeffs[0] * ncoeffs[1] * ncoeffs[2], 3);

      int64_t r = 0;
      for (int64_t k = 0; k < ncoeffs[2]; ++k)
        for (int64_t j = 0; j < ncoeffs[1]; ++j)
          for (int64_t i = 0; i < ncoeffs[0]; ++i) {
            C(r, 0) = ((T)i) / (T)(ncoeffs[0] - 1);
            C(r, 1) = ((T)j) / (T)(ncoeffs[1] - 1);
            C(r, 2) = ((T)k) / (T)(ncoeffs[2] - 1);
            ++r;
          }

      geo_.addPatch(
          gsTensorBSpline<3, T>(give(KV0), give(KV1), give(KV2), give(C)));

    } else if constexpr (d == 4) {
      gsKnotVector<T> KV0(0, 1, ncoeffs[0] - degrees[0] - 1, degrees[0] + 1);
      gsKnotVector<T> KV1(0, 1, ncoeffs[1] - degrees[1] - 1, degrees[1] + 1);
      gsKnotVector<T> KV2(0, 1, ncoeffs[2] - degrees[2] - 1, degrees[2] + 1);
      gsKnotVector<T> KV3(0, 1, ncoeffs[3] - degrees[3] - 1, degrees[3] + 1);

      gsMatrix<T> C(ncoeffs[0] * ncoeffs[1] * ncoeffs[2] * ncoeffs[3], 4);

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

      geo_.addPatch(gsTensorBSpline<4, T>(give(KV0), give(KV1), give(KV2),
                                          give(KV3), give(C)));
    }
  }

  /// @brief Returns the model's options
  nlohmann::json getOptions() const override {
    if constexpr (d == 1)
      return R"([{
             "name" : "npatches",
             "description" : "Number of patches per spatial dimension",
             "type" : ["int"],
             "value" : [1],
             "default" : [1],
             "uiid" : 0},{
             "name" : "degree",
             "description" : "Polynomial degree of the B-spline",
             "type" : ["int"],
             "value" : [2],
             "default" : [2],
             "uiid" : 1},{
             "name" : "ncoeffs",
             "description" : "Number of coefficients per parametric dimension",
             "type" : ["int"],
             "value" : [3],
             "default" : [3],
             "uiid" : 2}])"_json;

    else if constexpr (d == 2)
      return R"([{
             "name" : "npatches",
             "description" : "Number of patches per spatial dimension",
             "type" : ["int","int"],
             "value" : [1,1],
             "default" : [1,1],
             "uiid" : 0},{
             "name" : "degrees",
             "description" : "Polynomial degrees of the B-spline",
             "type" : ["int","int"],
             "value" : [2,2],
             "default" : [2,2],
             "uiid" : 1},{
             "name" : "ncoeffs",
             "description" : "Number of coefficients per parametric dimension",
             "type" : ["int","int"],
             "value" : [3,3],
             "default" : [3,3],
             "uiid" : 2}])"_json;

    else if constexpr (d == 3)
      return R"([{
             "name" : "npatches",
             "description" : "Number of patches per spatial dimension",
             "type" : ["int","int","int"],
             "value" : [1,1,1],
             "default" : [1,1,1],
             "uiid" : 0},{
             "name" : "degrees",
             "description" : "Polynomial degrees of the B-spline",
             "type" : ["int","int","int"],
             "value" : [2,2,2],
             "default" : [2,2,2],
             "uiid" : 1},{
             "name" : "ncoeffs",
             "description" : "Number of coefficients per parametric dimension",
             "type" : ["int","int","int"],
             "value" : [3,3,3],
             "default" : [3,3,3],
             "uiid" : 2}])"_json;

    else if constexpr (d == 4)
      return R"([{
             "name" : "npatches",
             "description" : "Number of patches per spatial dimension",
             "type" : ["int","int","int","int"],
             "value" : [1,1,1,1],
             "default" : [1,1,1,1],
             "uiid" : 0},{
             "name" : "degrees",
             "description" : "Polynomial degrees of the B-spline",
             "type" : ["int","int","int","int"],
             "value" : [2,2,2,2],
             "default" : [2,2,2,2],
             "uiid" : 1},{
             "name" : "ncoeffs",
             "description" : "Number of coefficients per parametric dimension",
             "type" : ["int","int","int","int"],
             "value" : [3,3,3,3],
             "default" : [3,3,3,3],
             "uiid" : 2}])"_json;

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

  /// @brief Serializes the model to JSON
  nlohmann::json to_json(const std::string &component,
                         const std::string &attribute) const override {

    if (attribute != "") {
      nlohmann::json json;

      if (attribute == "degrees") {
        json["degrees"] = nlohmann::json::array();

        for (std::size_t i = 0; i < geo_.patch(0).parDim(); ++i)
          json["degrees"].push_back(geo_.patch(0).degree(i));
      }

      else if (attribute == "geoDim")
        json["geoDim"] = geo_.patch(0).geoDim();

      else if (attribute == "parDim")
        json["parDim"] = geo_.patch(0).parDim();

      else if (attribute == "ncoeffs") {
        json["ncoeffs"] = nlohmann::json::array();

        if (auto bspline = dynamic_cast<const gsBSpline<T> *>(&geo_.patch(0)))
          for (std::size_t i = 0; i < bspline->parDim(); ++i)
            json["ncoeffs"].push_back(bspline->basis().size(i));
        else if (auto bspline = dynamic_cast<const gsTensorBSpline<d, T> *>(
                     &geo_.patch(0)))
          for (std::size_t i = 0; i < bspline->parDim(); ++i)
            json["ncoeffs"].push_back(bspline->basis().size(i));
        else
          return R"({ INVALID REQUEST })"_json;

      }

      else if (attribute == "nknots") {
        json["nknots"] = nlohmann::json::array();

        if (auto bspline = dynamic_cast<const gsBSpline<T> *>(&geo_.patch(0)))
          for (std::size_t i = 0; i < bspline->parDim(); ++i)
            json["nknots"].push_back(bspline->knots(i).size());
        else if (auto bspline = dynamic_cast<const gsTensorBSpline<d, T> *>(
                     &geo_.patch(0)))
          for (std::size_t i = 0; i < bspline->parDim(); ++i)
            json["nknots"].push_back(bspline->knots(i).size());
        else
          return R"({ INVALID REQUEST })"_json;
      }

      else if (attribute == "coeffs") {

        if (auto bspline = dynamic_cast<const gsBSpline<T> *>(&geo_.patch(0)))
          json["coeffs"] = utils::to_json(bspline->coefs());
        else if (auto bspline = dynamic_cast<const gsTensorBSpline<d, T> *>(
                     &geo_.patch(0)))
          json["coeffs"] = utils::to_json(bspline->coefs());
        else
          return R"({ INVALID REQUEST })"_json;

      }

      else if (attribute == "knots") {
        json["knots"] = nlohmann::json::array();

        if (auto bspline = dynamic_cast<const gsBSpline<T> *>(&geo_.patch(0)))
          for (std::size_t i = 0; i < bspline->parDim(); ++i)
            json["knots"].push_back(bspline->knots(i));
        else if (auto bspline = dynamic_cast<const gsTensorBSpline<d, T> *>(
                     &geo_.patch(0)))
          for (std::size_t i = 0; i < bspline->parDim(); ++i)
            json["knots"].push_back(bspline->knots(i));
        else
          return R"({ INVALID REQUEST })"_json;
      }

      return json;

    } else {
      auto json = utils::to_json(geo_);
      json.update(GismoModel<T>::to_json("transform", ""), true);

      return json;
    }

    return R"({ INVALID REQUEST })"_json;
  }

  /// @brief Updates the attributes of the model
  nlohmann::json updateAttribute(const std::string &component,
                                 const std::string &attribute,
                                 const nlohmann::json &json) override {

    if (attribute == "coeffs") {
      if (!json.contains("data"))
        throw InvalidModelAttributeException();
      if (!json["data"].contains("indices") || !json["data"].contains("coeffs"))
        throw InvalidModelAttributeException();

      auto indices = json["data"]["indices"].get<std::vector<int64_t>>();
      auto ncoeffs = geo_.patch(0).coefs().rows();

      switch (geo_.geoDim()) {
      case (1): {
        auto coords = json["data"]["coeffs"].get<std::vector<std::tuple<T>>>();

        for (const auto &[index, coord] : iganet::utils::zip(indices, coords)) {
          if (index < 0 || index >= ncoeffs)
            throw IndexOutOfBoundsException();

          geo_.patch(0).coef(index, 0) = std::get<0>(coord);
        }
        break;
      }
      case (2): {
        auto coords =
            json["data"]["coeffs"].get<std::vector<std::tuple<T, T>>>();

        for (const auto &[index, coord] : iganet::utils::zip(indices, coords)) {
          if (index < 0 || index >= ncoeffs)
            throw IndexOutOfBoundsException();

          geo_.patch(0).coef(index, 0) = std::get<0>(coord);
          geo_.patch(0).coef(index, 1) = std::get<1>(coord);
        }
        break;
      }
      case (3): {
        auto coords =
            json["data"]["coeffs"].get<std::vector<std::tuple<T, T, T>>>();

        for (const auto &[index, coord] : iganet::utils::zip(indices, coords)) {
          if (index < 0 || index >= ncoeffs)
            throw IndexOutOfBoundsException();

          geo_.patch(0).coef(index, 0) = std::get<0>(coord);
          geo_.patch(0).coef(index, 1) = std::get<1>(coord);
          geo_.patch(0).coef(index, 2) = std::get<2>(coord);
        }
        break;
      }
      case (4): {
        auto coords =
            json["data"]["coeffs"].get<std::vector<std::tuple<T, T, T, T>>>();

        for (const auto &[index, coord] : iganet::utils::zip(indices, coords)) {
          if (index < 0 || index >= ncoeffs)
            throw IndexOutOfBoundsException();

          geo_.patch(0).coef(index, 0) = std::get<0>(coord);
          geo_.patch(0).coef(index, 1) = std::get<1>(coord);
          geo_.patch(0).coef(index, 2) = std::get<2>(coord);
          geo_.patch(0).coef(index, 3) = std::get<3>(coord);
        }
        break;
      }
      default:
        throw InvalidModelAttributeException();
      }

      return R"({})"_json;
    } else
      return GismoModel<T>::updateAttribute(component, attribute, json);
  }

  /// @brief Elevates the model's degrees, preserves smoothness
  void elevate(const nlohmann::json &json = NULL) override {
    int num = 1, dim = -1;

    if (json.contains("data")) {
      if (json["data"].contains("num"))
        num = json["data"]["num"].get<int>();

      if (json["data"].contains("dim"))
        dim = json["data"]["dim"].get<int>();
    }

    geo_.patch(0).degreeElevate(num, dim);
  }

  /// @brief Increases the model's degrees, preserves multiplicity
  void increase(const nlohmann::json &json = NULL) override {
    int num = 1, dim = -1;

    if (json.contains("data")) {
      if (json["data"].contains("num"))
        num = json["data"]["num"].get<int>();

      if (json["data"].contains("dim"))
        dim = json["data"]["dim"].get<int>();
    }

    geo_.patch(0).degreeIncrease(num, dim);
  }

  /// @brief Refines the model
  void refine(const nlohmann::json &json = NULL) override {
    int num = 1, dim = -1;

    if (json.contains("data")) {
      if (json["data"].contains("num"))
        num = json["data"]["num"].get<int>();

      if (json["data"].contains("dim"))
        dim = json["data"]["dim"].get<int>();
    }

    geo_.patch(0).uniformRefine(num, 1, dim);
  }

  /// @brief Reparameterized the model
  void reparameterize(const nlohmann::json &json = NULL) override {

    gismo::gsBarrierPatch<2, T> opt(geo_, false);
    opt.options().setInt("ParamMethod", 1);
    opt.compute();

    geo_ = opt.result();
  }
};

} // namespace webapp
} // namespace iganet
