/**
   @file webapps/models/gismo/GismoLinearElasticity3d.cxx

   @brief G+Smo Linear elasticity solver in 3d

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <jit.hpp>
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

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wreturn-type-c-linkage"

  /// @brief Create a G+Smo Linear elasticity solver
  std::shared_ptr<iganet::Model> create(const nlohmann::json &json) {
    std::array<iganet::short_t, 3> degrees = {1, 1, 1};
    std::array<int64_t, 3> ncoeffs = {4, 4, 4};
    std::array<int64_t, 3> npatches = {1, 1, 1};

    if (json.contains("data")) {

      if (json["data"].contains("degrees"))
        degrees = json["data"]["degrees"].get<std::array<iganet::short_t, 3>>();

      if (json["data"].contains("ncoeffs"))
        ncoeffs = json["data"]["ncoeffs"].get<std::array<int64_t, 3>>();

      if (json["data"].contains("npatches"))
        npatches = json["data"]["npatches"].get<std::array<int64_t, 3>>();
    }

    return std::make_shared<
        iganet::webapp::GismoLinearElasticityModel<3, iganet::real_t>>(
        degrees, ncoeffs, npatches);
  }

#pragma GCC diagnostic pop
}
