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

/// @brief Enumerator for specifying the capabilities
enum class capability {
  /*! Model create/remove */
  create = 0, /*!< create object */
  remove = 1, /*!< remove object */

  /*! Model parameters */
  parameters = 2, /*!< model has extra parameters */

  /*!< Model evaluation and adaption */
  eval = 3,     /*!< evaluates object */
  refine = 4,   /*!< h-refines object */
  elevate = 5,  /*!< p-refines object */
  increase = 6, /*!< p-refines object */

  /*!< Model reparameterization */
  reparameterize = 7, /*!< reparameterizes the model's geometry */

  /*!< Model loading/saving */
  load = 8, /*!< loads model from PyTorch file */
  save = 9, /*!< saves model to PyTorch file */

  /*!< Model import/export */
  importXML = 10, /*!< imports object from G+Smo XML file */
  exportXML = 11, /*!< exports object to G+Smo XML file */

  /*!< Error computation */
  computeL1error = 12, /*!< computes model's L1-error */
  computeL2error = 13, /*!< computes model's L2-error */
  computeH1error = 14  /*!< computes model's H1-error */
};

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
  virtual nlohmann::json eval(const std::string &component,
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

/// @brief Model parameters
class ModelParameters {
public:
  /// @brief Return's the model's parameters
  virtual nlohmann::json getParameters() const = 0;

  // @brief Returns model capabilities
  std::vector<std::string> getCapabilities() const {
    return std::vector{std::string("parameters")};
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
  virtual void importXML(const nlohmann::json &json,
                         const std::string &component, int id) = 0;

  /// @brief Imports model from XML (as XML object)
  virtual void importXML(const pugi::xml_node &xml,
                         const std::string &component, int id) = 0;

  /// @brief Exports model to XML (as JSON object)
  virtual nlohmann::json exportXML(const std::string &component, int id) = 0;

  /// @brief Exports model to XML (as XML object)
  virtual pugi::xml_node &exportXML(pugi::xml_node &root,
                                    const std::string &component, int id) = 0;

  // @brief Returns model capabilities
  std::vector<std::string> getCapabilities() const {
    return std::vector{std::string("exportxml"), std::string("importxml")};
  }
};

/// @brief Model interface
class Model {
public:
  /// @brief Constructor
  Model() : transform_(torch::eye(4, Options<iganet::real_t>{})){};

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

  /// @brief Returns the model's JSON serialization
  virtual nlohmann::json getModel() const {

    nlohmann::json json;
    json["name"] = getName();
    json["description"] = getDescription();

    json["options"] = getOptions();
    json["capabilities"] = getCapabilities();
    json["inputs"] = getInputs();
    json["outputs"] = getOutputs();

    return json;
  }

  /// @brief Returns the model's capabilities
  virtual nlohmann::json getCapabilities() const {

    auto json = nlohmann::json::array();

    json.push_back("create");
    json.push_back("remove");

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

    if (auto m = dynamic_cast<const ModelParameters *>(this))
      for (auto const &capability : m->getCapabilities())
        json.push_back(capability);

    if (auto m = dynamic_cast<const ModelRefine *>(this))
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
  virtual nlohmann::json to_json(const std::string &component,
                                 const std::string &attribute) const {
    if (component == "transform") {

      nlohmann::json json;
      json["matrix"] = utils::to_json<iganet::real_t, 1>(transform_.flatten());

      return json;
    }

    else
      return "{ INVALID REQUEST }";
  }

  /// @brief Updates the attributes of the model
  virtual nlohmann::json updateAttribute(const std::string &component,
                                         const std::string &attribute,
                                         const nlohmann::json &json) {
    if (attribute == "transform") {
      if (!json.contains("data"))
        throw InvalidModelAttributeException();
      if (!json["data"].contains("matrix"))
        throw InvalidModelAttributeException();

      auto matrix = json["data"]["matrix"].get<std::vector<iganet::real_t>>();

      if (matrix.size() != 16)
        throw IndexOutOfBoundsException();

      auto transform_cpu =
          utils::to_tensorAccessor<iganet::real_t, 2>(transform_, torch::kCPU);
      auto transformAccessor = std::get<1>(transform_cpu);

      std::size_t index(0);
      for (const auto &entry : matrix) {
        transformAccessor[index / 4][index % 4] = entry;
        index++;
      }

      return "{}";
    }

    else
      return "{ INVALID REQUEST }";
  }

protected:
  /// @brief Global transformation matrix
  torch::Tensor transform_;
};

} // namespace iganet
