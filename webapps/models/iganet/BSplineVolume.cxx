/**
   @file webapps/models/iganet/BSplineVolume.cxx

   @brief B-Spline volume

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <BSplineModel.hpp>
#include <jit.hpp>
#include <modelmanager.hpp>

#ifdef _WIN32
extern "C" __declspec(dllexport)
#else
extern "C"
#endif
{
  /// @brief List of JIT-compiled model handlers
  static std::map<std::string, std::shared_ptr<iganet::ModelHandler>> models;

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"

  /// @brief Create a B-spline volume
  std::shared_ptr<iganet::Model<iganet::real_t>> create(
      const nlohmann::json &json) {
    enum iganet::webapp::degree degree = iganet::webapp::degree::linear;
    enum iganet::init init = iganet::init::linear;
    std::array<int64_t, 3> ncoeffs = {4, 4, 4};
    bool nonuniform = false;

    if (json.contains("data")) {

      if (json["data"].contains("degree"))
        degree = json["data"]["degree"].get<enum iganet::webapp::degree>();

      if (json["data"].contains("init"))
        init = json["data"]["init"].get<enum iganet::init>();

      if (json["data"].contains("ncoeffs"))
        ncoeffs = json["data"]["ncoeffs"].get<std::array<int64_t, 3>>();

      if (json["data"].contains("nonuniform"))
        nonuniform = json["data"]["nonuniform"].get<bool>();

      try {
        // generate list of include files
        std::string includes =
            "#include <BSplineModel.hpp>\n"
            "#pragma clang diagnostic push\n"
            "#pragma clang diagnostic ignored \"-Wreturn-type-c-linkage\"\n";

        // generate source code
        std::string src = "std::shared_ptr<iganet::Model<iganet::real_t>> "
                          "create(const std::array<int64_t, "
                          "3>& ncoeffs, enum iganet::init init)\n{\n";

        if (nonuniform)
          src.append("return "
                     "std::make_shared<iganet::webapp::BSplineModel<iganet::"
                     "NonUniformBSpline<iganet::real_t, ");
        else
          src.append("return "
                     "std::make_shared<iganet::webapp::BSplineModel<iganet::"
                     "UniformBSpline<iganet::real_t, ");

        src.append("3, ");
        src.append(std::to_string((int)degree) + ", " +
                   std::to_string((int)degree) + ", " +
                   std::to_string((int)degree) +
                   ">>>(ncoeffs, init);\n}\n#pragma clang diagnostic pop\n");

        // compile dynamic library
        auto libname = iganet::jit{}.compile(includes, src, "BSplineVolume");

        // Search for library name
        auto model = models.find(libname);
        if (model == models.end()) {
          models[libname] =
              std::make_shared<iganet::ModelHandler>(libname.c_str());
          model = models.find(libname);
        }

        // create model instance
        std::shared_ptr<iganet::Model<iganet::real_t>> (*create)(
            const std::array<int64_t, 3> &, enum iganet::init);
        create =
            reinterpret_cast<std::shared_ptr<iganet::Model<iganet::real_t>> (*)(
                const std::array<int64_t, 3> &, enum iganet::init)>(
                model->second->getSymbol("create"));
        return create(ncoeffs, init);
      } catch (...) {
        throw iganet::InvalidModelException();
      }
    } else if (nonuniform)
      return std::make_shared<iganet::webapp::BSplineModel<
          iganet::NonUniformBSpline<iganet::real_t, 3, 1, 1, 1>>>(ncoeffs,
                                                                  init);
    else
      return std::make_shared<iganet::webapp::BSplineModel<
          iganet::UniformBSpline<iganet::real_t, 3, 1, 1, 1>>>(ncoeffs, init);
  }

  /// @brief Load a B-spline volume
  std::shared_ptr<iganet::Model<iganet::real_t>> load(
      const nlohmann::json &json) {

    if (json.contains("data")) {
      if (json["data"].contains("binary")) {

        // get binary vector from JSON object
        auto binary = json["data"]["binary"].get<std::vector<std::uint8_t>>();

        // recover input archive from binary vector
        torch::serialize::InputArchive archive;
        archive.load_from(reinterpret_cast<const char *>(binary.data()),
                          binary.size());

        try {
          // get model hash
          c10::IValue model;
          archive.read("model", model);

          // check if model can be processed
          if (model.toInt() ==
              static_cast<int64_t>(std::hash<std::string>{}("BSplineVolume"))) {

            // get (non)uniform attribute
            archive.read("nonuniform", model);
            bool nonuniform = model.toBool();

            torch::Tensor tensor;

            // get parametric dimension
            archive.read("geometry.parDim", tensor);
            iganet::short_t parDim = tensor.item<int64_t>();

            // get geometric dimension
            archive.read("geometry.geoDim", tensor);
            iganet::short_t geoDim = tensor.item<int64_t>();

            if (parDim != 3)
              throw iganet::InvalidModelException();

            // get degrees
            std::array<iganet::short_t, 3> degrees;
            for (iganet::short_t i = 0; i < parDim; ++i) {
              archive.read("geometry.degree[" + std::to_string(i) + "]",
                           tensor);
              degrees[i] = tensor.item<int64_t>();
            }

            // get ncoeffs
            std::array<int64_t, 3> ncoeffs;
            for (iganet::short_t i = 0; i < parDim; ++i) {
              archive.read("geometry.ncoeffs[" + std::to_string(i) + "]",
                           tensor);
              ncoeffs[i] = tensor.item<int64_t>();
            }

            // generate list of include files
            std::string includes = "#include <BSplineModel.hpp>\n"
                                   "#pragma clang diagnostic push\n"
                                   "#pragma clang diagnostic ignored "
                                   "\"-Wreturn-type-c-linkage\"\n";

            // generate source code
            std::string src =
                "std::shared_ptr<iganet::Model<iganet::real_t>> create(const "
                "std::array<int64_t, 3>& ncoeffs, enum iganet::init init)\n{\n";

            if (nonuniform)
              src.append("return "
                         "std::make_shared<iganet::webapp::BSplineModel<iganet:"
                         ":NonUniformBSpline<iganet::real_t, ");
            else
              src.append("return "
                         "std::make_shared<iganet::webapp::BSplineModel<iganet:"
                         ":UniformBSpline<iganet::real_t, ");

            src.append("3, ");
            src.append(
                std::to_string((int)degrees[0]) + ", " +
                std::to_string((int)degrees[1]) + ", " +
                std::to_string((int)degrees[2]) +
                ">>>(ncoeffs, init);\n}\n#pragma clang diagnostic pop\n");

            // compile dynamic library
            auto libname =
                iganet::jit{}.compile(includes, src, "BSplineVolume");

            // Search for library name
            auto model = models.find(libname);
            if (model == models.end()) {
              models[libname] =
                  std::make_shared<iganet::ModelHandler>(libname.c_str());
              model = models.find(libname);
            }

            // create model instance and load data
            std::shared_ptr<iganet::Model<iganet::real_t>> (*create)(
                const std::array<int64_t, 3> &, enum iganet::init);
            create = reinterpret_cast<
                std::shared_ptr<iganet::Model<iganet::real_t>> (*)(
                    const std::array<int64_t, 3> &, enum iganet::init)>(
                model->second->getSymbol("create"));

            auto m = create(ncoeffs, iganet::init::greville);
            if (auto m_ = std::dynamic_pointer_cast<iganet::ModelSerialize>(m))
              m_->load(json);
            else
              throw iganet::InvalidModelException();

            return m;
          } else {
            throw iganet::InvalidModelException();
          }
        } catch (...) {
          throw iganet::InvalidModelException();
        }
      }
    }

    throw iganet::InvalidModelException();
  }

#pragma clang diagnostic pop
}
