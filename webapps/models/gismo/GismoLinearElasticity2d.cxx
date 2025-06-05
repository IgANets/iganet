/**
   @file webapps/models/gismo/GismoLinearElasticity2d.cxx

   @brief G+Smo Linear elasticity solver in 2d

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <modelmanager.hpp>

#include <GismoLinearElasticityModel.hpp>

#ifdef _WIN32
extern "C" __declspec(dllexport)
#else
extern "C"
#endif
{
  // @brief List of JIT-compiled model handlers
  static std::map<std::string, std::shared_ptr<iganet::ModelHandler>> models;

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"

  /// @brief Create a G+Smo Linear elasticity solver
  std::shared_ptr<iganet::Model<iganet::real_t>> create(
      const nlohmann::json &json) {
    std::array<iganet::short_t, 2> degrees = {1, 1};
    std::array<int64_t, 2> ncoeffs = {4, 4};
    std::array<int64_t, 2> npatches = {1, 1};
    std::array<iganet::real_t, 2> dimensions = {1.0, 1.0};

    if (json.contains("data")) {

      if (json["data"].contains("degrees"))
        degrees = json["data"]["degrees"].get<std::array<iganet::short_t, 2>>();

      if (json["data"].contains("ncoeffs"))
        ncoeffs = json["data"]["ncoeffs"].get<std::array<int64_t, 2>>();

      if (json["data"].contains("npatches"))
        npatches = json["data"]["npatches"].get<std::array<int64_t, 2>>();

      if (json["data"].contains("dimensions"))
        dimensions =
            json["data"]["dimensions"].get<std::array<iganet::real_t, 2>>();
    }

    return std::make_shared<
        iganet::webapp::GismoLinearElasticityModel<2, iganet::real_t>>(
        degrees, ncoeffs, npatches, dimensions);
  }

#pragma clang diagnostic pop
}
