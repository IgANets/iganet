/**
   @file webapps/models/Poisson2d.cxx

   @brief Poisson equation in 2d model

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <iganet.hpp>
#include <model.hpp>

namespace iganet {

/// @brief Poisson equation in 2d model
class Poisson2dModel : public Model,
                       public ModelEval,
                       public ModelRefine,
                       public ModelSerialize,
                       public ModelXML {

private:
  /// @brief Global offset vector
  torch::Tensor offset;

  /// @brief Global rotation vector
  torch::Tensor rotation;

public:
  /// @brief Default constructor
  Poisson2dModel() = default;

  /// @brief Destructor
  ~Poisson2dModel() {}

  /// @brief Returns the model's name
  std::string getName() const override { return "Poisson2d"; }

  /// @brief Returns the model's description
  std::string getDescription() const override { return "Poisson 2d model"; }

  /// @brief Returns the model's options
  nlohmann::json getOptions() const override { return R"([])"_json; }

  /// @brief Returns the model's inputs
  nlohmann::json getInputs() const override { return R"([])"_json; }

  /// @brief Returns the model's outputs
  nlohmann::json getOutputs() const override { return R"([])"_json; }

  /// @brief Serializes the model to JSON
  nlohmann::json to_json(const std::string &component,
                         const std::string &attribute) const override {
    return R"({ "reason" : "Not implemented yet" })"_json;
  }

  /// @brief Updates the attrbutes of the model
  nlohmann::json updateAttribute(const std::string &component,
                                 const std::string &attribute,
                                 const nlohmann::json &json) override {
    return R"({ "reason" : "Not implemented yet" })"_json;
  }

  /// @brief Evaluates the model
  nlohmann::json eval(const std::string &component,
                      const nlohmann::json &json) const override {
    return R"({ "reason" : "Not implemented yet" })"_json;
  }

  /// @brief Refines the model
  void refine(const nlohmann::json &json = NULL) override {}

  /// @brief Loads model from LibTorch file
  void load(const nlohmann::json &json) override {}

  /// @brief Saves model to LibTorch file
  nlohmann::json save() const override {
    return R"({ "reason" : "Not implemented yet" })"_json;
  }

  /// @brief Imports the model from XML (as JSON object)
  void importXML(const nlohmann::json &json, const std::string &component,
                 int id = 0) override {}

  /// @brief Imports the model from XML (as XML object)
  void importXML(const pugi::xml_node &root, const std::string &component,
                 int id = 0) override {}

  /// @brief Exports the model to XML (as JSON object)
  nlohmann::json exportXML(const std::string &component, int id) override {
    return R"({ "reason" : "Not implemented yet" })"_json;
  }

  /// @brief Exports the model to XML (as XML object)
  pugi::xml_node &exportXML(pugi::xml_node &root, const std::string &component,
                            int id) override {
    return root;
  }
};
} // namespace iganet

#ifdef _WIN32
extern "C" __declspec(dllexport)
#else
extern "C"
#endif
{

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"

  std::shared_ptr<iganet::Model> create(const nlohmann::json &json) {
    return std::make_shared<iganet::Poisson2dModel>();
  }

#pragma clang diagnostic pop
}
