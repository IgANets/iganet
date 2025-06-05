/**
   @file webapps/model.hpp

   @brief Model capabilities

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <iganet.h>
#include <webapps_config.hpp>

namespace iganet {

/// @brief Enumerator for specifying the output type
enum class io {
  scalar = 0,               /*!< scalar value */
  scalarfield = 1,          /*!< scalar field */
  vectorfield = 2,          /*!< vector field */
  scalarfield_boundary = 3, /*!< scalar field at the boundary */
  vectorfield_boundary = 4  /*!< vector field at the boundary */
};

/// @brief IndexOutOfBounds exception
struct IndexOutOfBoundsException : public std::exception {
  const char *what() const throw() { return "Index is out of bounds"; }
};

/// @brief InvalidModel exception
struct InvalidModelException : public std::exception {
  const char *what() const throw() { return "Invalid model name"; }
};

/// @brief InvalidModelAttribute exception
struct InvalidModelAttributeException : public std::exception {
  const char *what() const throw() { return "Invalid model attribute"; }
};

/// @brief Model add patch
class ModelAddPatch {
public:
  /// @brief Adds a patch to a model
  virtual void addPatch(const nlohmann::json &json) = 0;

  // @brief Returns model capabilities
  std::vector<std::string> getCapabilities() const {
    return std::vector{std::string("addpatch")};
  }
};

/// @brief Model remove patch
class ModelRemovePatch {
public:
  /// @brief Adds a patch to a model
  virtual void removePatch(const nlohmann::json &json) = 0;

  // @brief Returns model capabilities
  std::vector<std::string> getCapabilities() const {
    return std::vector{std::string("removepatch")};
  }
};  
  
/// @brief Model error computation
class ModelComputeError {
public:
  /// @brief Computes the model's error
  virtual nlohmann::json computeError(const nlohmann::json &json) const = 0;

  // @brief Returns model capabilities
  std::vector<std::string> getCapabilities() const {
    return std::vector{std::string("computeL1error"),
                       std::string("computeL2error"),
                       std::string("computeH1error")};
  }
};

/// @brief Model degree elevation
class ModelElevate {
public:
  /// @brief Elevates the model's degrees, preserves smoothness
  virtual void elevate(const nlohmann::json &json) = 0;

  // @brief Returns model capabilities
  std::vector<std::string> getCapabilities() const {
    return std::vector{std::string("elevate")};
  }
};

/// @brief Model evaluator
class ModelEval {
public:
  /// @brief Evaluates model
  virtual nlohmann::json eval(const std::string &patch,
                              const std::string &component,
                              const nlohmann::json &json) const = 0;

  // @brief Returns model capabilities
  std::vector<std::string> getCapabilities() const {
    return std::vector{std::string("eval")};
  }
};

/// @brief Model degree increase
class ModelIncrease {
public:
  /// @brief Increases the model's degrees, preserves multiplicity
  virtual void increase(const nlohmann::json &json) = 0;

  // @brief Returns model capabilities
  std::vector<std::string> getCapabilities() const {
    return std::vector{std::string("increase")};
  }
};

/// @brief Model refinement
class ModelRefine {
public:
  /// @brief Refines model
  virtual void refine(const nlohmann::json &json) = 0;

  // @brief Returns model capabilities
  std::vector<std::string> getCapabilities() const {
    return std::vector{std::string("refine")};
  }
};

/// @brief Model reparameterization
class ModelReparameterize {
public:
  /// @brief Reparameterizes the model
  virtual void reparameterize(const nlohmann::json &json) = 0;

  // @brief Returns model capabilities
  std::vector<std::string> getCapabilities() const {
    return std::vector{std::string("reparameterize")};
  }
};

/// @brief Model serialization
class ModelSerialize {
public:
  /// @brief Loads model from LibTorch file
  virtual void load(const nlohmann::json &json) = 0;

  /// @brief Saves model to LibTorch file
  virtual nlohmann::json save() const = 0;

  // @brief Returns model capabilities
  std::vector<std::string> getCapabilities() const {
    return std::vector{std::string("load"), std::string("save")};
  }
};

/// @brief Model XML serialization
class ModelXML {
public:
  /// @brief Imports model from XML (as JSON object)
  virtual void importXML(const std::string &patch,
                         const std::string &component,
                         const nlohmann::json &json,
                         int id) = 0;

  /// @brief Imports model from XML (as XML object)
  virtual void importXML(const std::string &patch,
                         const std::string &component,
                         const pugi::xml_node &xml,
                         int id) = 0;

  /// @brief Exports model to XML (as JSON object)
  virtual nlohmann::json exportXML(const std::string &patch,
                                   const std::string &component,
                                   int id) = 0;

  /// @brief Exports model to XML (as XML object)
  virtual pugi::xml_node &exportXML(const std::string &patch,
                                    const std::string &component,
                                    pugi::xml_node &root,
                                    int id) = 0;

  // @brief Returns model capabilities
  std::vector<std::string> getCapabilities() const {
    return std::vector{std::string("exportxml"), std::string("importxml")};
  }
};

/// @brief Model interface
template <typename T> class Model {
public:
  /// @brief Constructor
  Model(){};

  /// @brief Destructor
  virtual ~Model(){};

  /// @brief Returns the model's name
  virtual std::string getName() const = 0;

  /// @brief Returns the model's description
  virtual std::string getDescription() const = 0;

  /// @brief Returns the model's options
  virtual nlohmann::json getOptions() const = 0;

  /// @brief Returns the model's inputs
  virtual nlohmann::json getInputs() const = 0;

  /// @brief Returns the model's outputs
  virtual nlohmann::json getOutputs() const = 0;

  /// @brief Return's the model's parameters
  virtual nlohmann::json getParameters() const = 0;

  /// @brief Returns the model's JSON serialization
  virtual nlohmann::json getModel() const {

    nlohmann::json json;
    json["name"] = getName();
    json["description"] = getDescription();

    json["options"] = getOptions();
    json["capabilities"] = getCapabilities();
    json["inputs"] = getInputs();
    json["outputs"] = getOutputs();
    json["parameters"] = getParameters();

    return json;
  }

  /// @brief Returns the model's capabilities
  virtual nlohmann::json getCapabilities() const {

    auto json = nlohmann::json::array();

    json.push_back("create");
    json.push_back("remove");

    if (auto m = dynamic_cast<const ModelAddPatch *>(this))
      for (auto const &capability : m->getCapabilities())
        json.push_back(capability);
    
    if (auto m = dynamic_cast<const ModelComputeError *>(this))
      for (auto const &capability : m->getCapabilities())
        json.push_back(capability);

    if (auto m = dynamic_cast<const ModelElevate *>(this))
      for (auto const &capability : m->getCapabilities())
        json.push_back(capability);

    if (auto m = dynamic_cast<const ModelEval *>(this))
      for (auto const &capability : m->getCapabilities())
        json.push_back(capability);

    if (auto m = dynamic_cast<const ModelIncrease *>(this))
      for (auto const &capability : m->getCapabilities())
        json.push_back(capability);

    if (auto m = dynamic_cast<const ModelRefine *>(this))
      for (auto const &capability : m->getCapabilities())
        json.push_back(capability);

    if (auto m = dynamic_cast<const ModelRemovePatch *>(this))
      for (auto const &capability : m->getCapabilities())
        json.push_back(capability);
    
    if (auto m = dynamic_cast<const ModelReparameterize *>(this))
      for (auto const &capability : m->getCapabilities())
        json.push_back(capability);

    if (auto m = dynamic_cast<const ModelSerialize *>(this))
      for (auto const &capability : m->getCapabilities())
        json.push_back(capability);

    if (auto m = dynamic_cast<const ModelXML *>(this))
      for (auto const &capability : m->getCapabilities())
        json.push_back(capability);

    return json;
  }

  /// @brief Serializes the model to JSON
  virtual nlohmann::json to_json(const std::string &patch,
                                 const std::string &component,
                                 const std::string &attribute) const {

    return "{ INVALID REQUEST }";
  }

  /// @brief Updates the attributes of the model
  virtual nlohmann::json updateAttribute(const std::string &patch,
                                         const std::string &component,
                                         const std::string &attribute,
                                         const nlohmann::json &json) {

    return "{ INVALID REQUEST }";
  }
};

} // namespace iganet
