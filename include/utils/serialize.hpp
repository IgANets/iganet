/**
   @file include/utils/serialize.hpp

   @brief Srialization utility functions

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <utils/tensorarray.hpp>

#include <nlohmann/json.hpp>
#include <pugixml.hpp>

#include <torch/torch.h>

namespace iganet {
namespace utils {

/// @brief Serialization prototype
///
/// This abstract class defines the functions that must be
/// implemented to serialize an object
struct Serializable {
  /// @brief Returns the object as JSON object
  virtual nlohmann::json to_json() const = 0;

  /// @brief Returns a string representation of the object
  virtual void pretty_print(std::ostream &os = std::cout) const = 0;
};

/// @brief Converts a torch::TensorAccessor object to a JSON object
template <typename T, std::size_t N>
inline auto to_json(const torch::TensorAccessor<T, N> &accessor) {
  auto json = nlohmann::json::array();

  if constexpr (N == 1) {
    for (int64_t i = 0; i < accessor.size(0); ++i)
      json.push_back(accessor[i]);
  } else if constexpr (N == 2) {
    for (int64_t i = 0; i < accessor.size(0); ++i)
      for (int64_t j = 0; j < accessor.size(1); ++j)
        json.push_back(accessor[i][j]);
  } else if constexpr (N == 3) {
    for (int64_t i = 0; i < accessor.size(0); ++i)
      for (int64_t j = 0; j < accessor.size(1); ++j)
        for (int64_t k = 0; k < accessor.size(2); ++k)
          json.push_back(accessor[i][j][k]);
  } else if constexpr (N == 4) {
    for (int64_t i = 0; i < accessor.size(0); ++i)
      for (int64_t j = 0; j < accessor.size(1); ++j)
        for (int64_t k = 0; k < accessor.size(2); ++k)
          for (int64_t l = 0; l < accessor.size(3); ++l)
            json.push_back(accessor[i][j][k][l]);
  }

  return json;
}

/// @brief Converts a torch::Tensor object to a JSON object
template <typename T, std::size_t N> inline auto to_json(const torch::Tensor &tensor) {
  if (tensor.is_cuda()) {
    auto [tensor_cpu, accessor] = to_tensorAccessor<T, N>(tensor, torch::kCPU);
    return to_json(accessor);
  } else {
    auto accessor = to_tensorAccessor<T, N>(tensor);
    return to_json(accessor);
  }
}

/// @brief Converts an std::array of torch::Tensor objects to a JSON
/// object
template <typename T, std::size_t N, std::size_t M>
inline auto to_json(const std::array<torch::Tensor, M> &tensors) {
  auto json = nlohmann::json::array();

  for (std::size_t i = 0; i < M; ++i) {
    if (tensors[i].is_cuda()) {
      auto [tensor_cpu, accessor] =
          to_tensorAccessor<T, N>(tensors[i], torch::kCPU);
      json.push_back(to_json<T, N>(accessor));
    } else {
      auto accessor = to_tensorAccessor<T, N>(tensors[i]);
      json.push_back(to_json<T, N>(accessor));
    }
  }

  return json;
}

/// @brief Converts a torch::TensorAccessor object to an XML document object
template <typename T, std::size_t N>
inline pugi::xml_document to_xml(const torch::TensorAccessor<T, N> &accessor, torch::IntArrayRef sizes, std::string tag = "Matrix", int id = 0, std::string label = "", int index = -1) {
  pugi::xml_document doc;
  pugi::xml_node root = doc.append_child("xml");
  to_xml(accessor, sizes, root, id, label, index);
  
  return doc;
}
  
/// @brief Converts a torch::TensorAccessor object to an XML object
template <typename T, std::size_t N>
inline pugi::xml_node &to_xml(const torch::TensorAccessor<T, N> &accessor, torch::IntArrayRef sizes, pugi::xml_node &root, std::string tag = "Matrix", int id = 0, std::string label = "", int index = -1) {

  // add node
  pugi::xml_node node = root.append_child(tag.c_str());

  if (id >= 0)
    node.append_attribute("id") = id;

  if (index >= 0)
    node.append_attribute("index") = index;
  
  if (!label.empty())
    node.append_attribute("label") = label.c_str();

  // add rows/cols or dimensions
  if (tag == "Matrix") {
    if constexpr (N == 1) {
      node.append_attribute("rows") = sizes[0];
      node.append_attribute("cols") = 1;

      std::stringstream ss;
      for (std::size_t i = 0; i < sizes[0]; ++i) 
        ss << std::to_string(accessor[i])
           << (i < sizes[0] - 1 ? " " : "");      
      node.append_child(pugi::node_pcdata).set_value(ss.str().c_str());
    }
    else if constexpr (N == 2) {
      node.append_attribute("rows") = sizes[0];
      node.append_attribute("cols") = sizes[1];

      std::stringstream ss;
      for (std::size_t i = 0; i < sizes[0]; ++i)
        for (std::size_t j = 0; j < sizes[1]; ++j) 
          ss << std::to_string(accessor[i][j])
             << (j < sizes[1] - 1 ? " " : (i < sizes[0] - 1 ? " " : ""));
      node.append_child(pugi::node_pcdata).set_value(ss.str().c_str());
    }
    else
      throw std::runtime_error("Tag \"Matrix\" only supports 1- and 2-dimensional tensors");
  } else {
    std::stringstream ss;
    for (const auto& size : sizes)
      ss << std::to_string(size) << " ";
    
    pugi::xml_node dims = node.append_child("Dimensions");
    dims.append_child(pugi::node_pcdata).set_value(ss.str().c_str());

    ss.str("");
    if constexpr (N == 1) {
      for (std::size_t i = 0; i < sizes[0]; ++i) 
        ss << std::to_string(accessor[i]) << " ";
    }
    else if constexpr (N == 2) {
      for (std::size_t i = 0; i < sizes[0]; ++i)
        for (std::size_t j = 0; j < sizes[1]; ++j) 
          ss << std::to_string(accessor[i][j]) << " ";
    }
    else if constexpr (N == 3) {
      for (std::size_t i = 0; i < sizes[0]; ++i)
        for (std::size_t j = 0; j < sizes[1]; ++j)
          for (std::size_t k = 0; j < sizes[2]; ++k) 
            ss << std::to_string(accessor[i][j][k]) << " ";
    }
    else if constexpr (N == 4) {
      for (std::size_t i = 0; i < sizes[0]; ++i)
        for (std::size_t j = 0; j < sizes[1]; ++j)
          for (std::size_t k = 0; k < sizes[2]; ++k)
            for (std::size_t l = 0; l < sizes[3]; ++l) 
              ss << std::to_string(accessor[i][j][k][l]) << " ";

    }
    else if constexpr (N == 5) {
      for (std::size_t i = 0; i < sizes[0]; ++i)
        for (std::size_t j = 0; j < sizes[1]; ++j)
          for (std::size_t k = 0; k < sizes[2]; ++k)
            for (std::size_t l = 0; l < sizes[3]; ++l)
              for (std::size_t m = 0; m < sizes[4]; ++m) 
                ss << std::to_string(accessor[i][j][k][l][m]) << " ";
    }
    else if constexpr (N == 6) {
      for (std::size_t i = 0; i < sizes[0]; ++i)
        for (std::size_t j = 0; j < sizes[1]; ++j)
          for (std::size_t k = 0; k < sizes[2]; ++k)
            for (std::size_t l = 0; l < sizes[3]; ++l)
              for (std::size_t m = 0; m < sizes[4]; ++m)
                for (std::size_t n = 0; n < sizes[5]; ++n) 
                  ss << std::to_string(accessor[i][j][k][l][m][n]) << " ";
    }
    else
      throw std::runtime_error("Dimensions higher than 4 are not implemented yet");
    
    pugi::xml_node data = node.append_child("Data");
    data.append_child(pugi::node_pcdata).set_value(ss.str().c_str());    
  }
    
  return root;
}

  /// @brief Converts a torch::Tensor object to an XML document object
  template <typename T, std::size_t N> inline pugi::xml_document to_xml(const torch::Tensor &tensor, std::string tag = "Matrix", int id = 0, std::string label = "", int index = -1) {
    pugi::xml_document doc;
    pugi::xml_node root = doc.append_child("xml");
    to_xml<T, N>(tensor, root, id, label, index);
    
    return doc;
  }
  
  /// @brief Converts a torch::Tensor object to an XML object
  template <typename T, std::size_t N> inline pugi::xml_node &to_xml(const torch::Tensor &tensor, pugi::xml_node &root, std::string tag = "Matrix", int id = 0, std::string label = "", int index = -1) {

    if (tensor.is_cuda()) {
      auto [tensor_cpu, accessor] = to_tensorAccessor<T, N>(tensor, torch::kCPU);
      return to_xml(accessor, tensor.sizes(), root, tag, id, label, index);
    } else {
      auto accessor = to_tensorAccessor<T, N>(tensor);
      return to_xml(accessor, tensor.sizes(), root, tag, id, label, index);
    }
  }

  /// @brief Converts an std::array of torch::Tensor objects to an XML
  /// object
  template <typename T, std::size_t N, std::size_t M>
  inline pugi::xml_document to_xml(const std::array<torch::Tensor, M> &tensors, std::string tag = "Matrix", int id = 0, std::string label = "") {
    pugi::xml_document doc;
    pugi::xml_node root = doc.append_child("xml");
    to_xml<T, N>(tensors, root, id, label, index);
    
    return doc;
  }
  
  /// @brief Converts an std::array of torch::Tensor objects to an XML
  /// object
  template <typename T, std::size_t N, std::size_t M>
  inline pugi::xml_node &to_xml(const std::array<torch::Tensor, M> &tensors, pugi::xml_node &root, std::string tag = "Matrix", int id = 0, std::string label = "") {
    
    for (std::size_t i = 0; i < M; ++i) {
      if (tensors[i].is_cuda()) {
        auto [tensor_cpu, accessor] =
          to_tensorAccessor<T, N>(tensors[i], torch::kCPU);
        to_xml(accessor, tensors[i].sizes(), root, tag, id, label, i);
      } else {
        auto accessor = to_tensorAccessor<T, N>(tensors[i]);
        to_xml(accessor, tensors[i].sizes(), root, tag, id, label, i);
      }
    }
    
    return root;
  }

  /// @brief Converts an XML documentobject to a torch::TensorAccessor object
  template <typename T, std::size_t N>
  inline torch::TensorAccessor<T, N> &from_xml(const pugi::xml_document &doc, torch::TensorAccessor<T, N> &accessor, torch::IntArrayRef sizes, std::string tag = "Matrix", int id = 0, std::string label = "", int index = -1) {
    return from_xml(doc.child("xml"), accessor, sizes, tag, id, label, index);
  }
  
  /// @brief Converts an XML object to a torch::TensorAccessor object
  template <typename T, std::size_t N>
  inline torch::TensorAccessor<T, N> &from_xml(const pugi::xml_node &root, torch::TensorAccessor<T, N> &accessor, torch::IntArrayRef sizes, std::string tag = "Matrix", int id = 0, std::string label = "", int index = -1) {

    return accessor;
  }

  /// @brief Converts an XML document object to a torch::Tensor object
  template <typename T, std::size_t N> inline torch::Tensor &from_xml(const pugi::xml_document &doc, torch::Tensor &tensor, std::string tag = "Matrix", int id = 0, std::string label = "", bool alloc = true, int index = -1) {
    return from_xml<T, N>(doc.child("xml"), tensor, tag, id, label, index);
  }
  
  /// @brief Converts an XML object to a torch::Tensor object
  template <typename T, std::size_t N> inline torch::Tensor &from_xml(const pugi::xml_node &root, torch::Tensor &tensor, std::string tag = "Matrix", int id = 0, std::string label = "", bool alloc = true, int index = -1) {

    // Loop through all nodes
    for (pugi::xml_node node : root.children(tag.c_str())) {
      
      if ((id >= 0 ? node.attribute("id").as_int() == id : true) &&
          (index >= 0 ? node.attribute("index").as_int() == index : true) &&
          (!label.empty() ? node.attribute("label").value() == label : true)) {
        
        if (tag == "Matrix") {

          int64_t rows = node.attribute("rows").as_int();
          int64_t cols = node.attribute("cols").as_int();
          
          if (!alloc && (tensor.size(0) != rows || tensor.size(1) != cols))
            throw std::runtime_error("Invalid matrix dimensions");

          else if (alloc && (tensor.size(0) != rows || tensor.size(1) != cols))            
            tensor = torch::zeros({rows, cols}, tensor.options());
          
          std::string values = std::regex_replace(node.text().get(), std::regex("[\t\r\n\a]+| +"), " ");
          
          auto [tensor_cpu, accessor] = to_tensorAccessor<T, N>(tensor, torch::kCPU);
          auto value = strtok(&values[0], " ");

          for (int64_t i = 0; i < rows; ++i)
            for (int64_t j = 0; j < cols; ++j) {
              if (value == NULL)
                throw std::runtime_error(
                                         "XML object does not provide enough coefficients");
              
              accessor[i][j] = static_cast<T>(std::stod(value));
              value = strtok(NULL, " ");
            }

          if (value != NULL)
            throw std::runtime_error(
                                     "XML object provides too many coefficients");

          if (tensor.device().type() != torch::kCPU)
            tensor = std::move(tensor_cpu);

          return tensor;
          
        } else {

          std::vector<int64_t> sizes;
          
          // Check for "Dimensions"
          if (pugi::xml_node dims = node.child("Dimensions")) {
            
            std::string values = std::regex_replace(
                                                    dims.text().get(), std::regex("[\t\r\n\a]+| +"), " ");
            for (auto value = strtok(&values[0], " "); value != NULL;
                 value = strtok(NULL, " "))
              sizes.push_back(static_cast<std::size_t>(std::stoi(value)));

            if (!alloc && (tensor.sizes() != sizes))
              throw std::runtime_error("Invalid tensor dimensions");

            else if (alloc && (tensor.sizes() != sizes))            
              tensor = torch::zeros(torch::IntArrayRef{sizes}, tensor.options());

            if (sizes.size() != N)
              throw std::runtime_error("Invalid tensor dimensions");
            
            // Check for "Data"
            if (pugi::xml_node data = node.child("Data")) {
              std::string values = std::regex_replace(data.text().get(), std::regex("[\t\r\n\a]+| +"), " ");
              
              auto [tensor_cpu, accessor] = to_tensorAccessor<T, N>(tensor, torch::kCPU);
              auto value = strtok(&values[0], " ");
              
              if constexpr (N == 1) {
                for (int64_t i = 0; i < sizes[0]; ++i) {                
                  if (value == NULL)
                    throw std::runtime_error(
                                             "XML object does not provide enough coefficients");
                
                  accessor[i] = static_cast<T>(std::stod(value));
                  value = strtok(NULL, " ");
                }
              }
              else if constexpr (N == 2) {                
                for (int64_t i = 0; i < sizes[0]; ++i)
                  for (int64_t j = 0; j < sizes[1]; ++j) {                
                    if (value == NULL)
                      throw std::runtime_error(
                                               "XML object does not provide enough coefficients");
                  
                    accessor[i][j] = static_cast<T>(std::stod(value));
                    value = strtok(NULL, " ");
                  }              
              }
              else if constexpr (N == 3) {
                for (int64_t i = 0; i < sizes[0]; ++i)
                  for (int64_t j = 0; j < sizes[1]; ++j)
                    for (int64_t k = 0; k < sizes[2]; ++k) {                
                      if (value == NULL)
                        throw std::runtime_error(
                                                 "XML object does not provide enough coefficients");
                    
                      accessor[i][j][k] = static_cast<T>(std::stod(value));
                      value = strtok(NULL, " ");
                    }              
              }
              else if constexpr (N == 4) {
                for (int64_t i = 0; i < sizes[0]; ++i)
                  for (int64_t j = 0; j < sizes[1]; ++j)
                    for (int64_t k = 0; k < sizes[2]; ++k)
                      for (int64_t l = 0; l < sizes[3]; ++l) {                
                        if (value == NULL)
                          throw std::runtime_error(
                                                   "XML object does not provide enough coefficients");
                      
                        accessor[i][j][k][l] = static_cast<T>(std::stod(value));
                        value = strtok(NULL, " ");
                      }              
              }
              else if constexpr (N == 5) {
                for (int64_t i = 0; i < sizes[0]; ++i)
                  for (int64_t j = 0; j < sizes[1]; ++j)
                    for (int64_t k = 0; k < sizes[2]; ++k)
                      for (int64_t l = 0; l < sizes[3]; ++l)
                        for (int64_t m = 0; m < sizes[4]; ++m) {                
                          if (value == NULL)
                            throw std::runtime_error(
                                                     "XML object does not provide enough coefficients");
                        
                          accessor[i][j][k][l][m] = static_cast<T>(std::stod(value));
                          value = strtok(NULL, " ");
                        }              
              }
              else if constexpr (N == 6) {
                for (int64_t i = 0; i < sizes[0]; ++i)
                  for (int64_t j = 0; j < sizes[1]; ++j)
                    for (int64_t k = 0; k < sizes[2]; ++k)
                      for (int64_t l = 0; l < sizes[3]; ++l)
                        for (int64_t m = 0; m < sizes[4]; ++m)
                          for (int64_t n = 0; n < sizes[5]; ++n) {                
                            if (value == NULL)
                              throw std::runtime_error(
                                                       "XML object does not provide enough coefficients");
                          
                            accessor[i][j][k][l][m][n] = static_cast<T>(std::stod(value));
                            value = strtok(NULL, " ");
                          }              
              }
            
              if (value != NULL)
                throw std::runtime_error(
                                         "XML object provides too many coefficients");

              if (tensor.device().type() != torch::kCPU)
                tensor = std::move(tensor_cpu);
            
              return tensor;
            } // "Data"
          } // "Dimenions"

          throw std::runtime_error(
                                   "XML object does not provide a \"Dimensions\" tag");
          
          return tensor;
        }

      } // try next node      
    } // "tag"
    
    throw std::runtime_error(
                             "XML object does not provide tag with given id, index, and/or label");
    return tensor;
  }

  /// @brief Converts an XML document object to an std::array of torch::Tensor objects
  template <typename T, std::size_t N, std::size_t M>
  inline std::array<torch::Tensor, M> &from_xml(const pugi::xml_document &doc, std::array<torch::Tensor, M> &tensors, std::string tag = "Matrix", int id = 0, bool alloc = true, std::string label = "") {

    return from_xml<T, N>(doc.child("xml"), tensors, tag, id, label, alloc);
  }
  
  /// @brief Converts an XML object to an std::array of torch::Tensor objects
  template <typename T, std::size_t N, std::size_t M>
  inline std::array<torch::Tensor, M> &from_xml(const pugi::xml_node &root, std::array<torch::Tensor, M> &tensors, std::string tag = "Matrix", int id = 0, bool alloc = true, std::string label = "") {

    for (std::size_t i = 0; i < M; ++i) {
      from_xml<T, N>(root, tensors[i], tag, id, label, alloc, i);
    }

    return tensors;
  }
  
} // namespace utils
} // namespace iganet
