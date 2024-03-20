/**
   @file include/memory.hpp

   @brief Memory debugger

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <chrono>

#include <core.hpp>

namespace iganet {

/// @brief Memory debugger
template <std::size_t id = 0> class MemoryDebugger {
private:
  /// @brief Memory object
  struct MemoryObject {
    std::string name_;
    int64_t bytes_;

    MemoryObject(const std::string &name, int64_t bytes)
        : name_(name), bytes_(bytes) {}
  };

  /// @brief Map holding the list of registered objects
  std::map<std::chrono::high_resolution_clock::time_point, MemoryObject>
      objects_;

  /// @brief Counter holding the number of registered objects
  int64_t counter_;

  /// @brief Counter holding the memory of registered objects in bytes
  int64_t bytes_;

  /// @brief Reference time point
  std::chrono::high_resolution_clock::time_point init_;

  /// @brief Converts bytes into best human-readable unit
  std::string convert_bytes(int64_t bytes) const {
    if (bytes < 1024ull)
      return std::to_string(bytes) + "b";
    else if (bytes < 1024ull * 1024ull)
      return std::to_string(bytes / double(1024)) + "kb";
    else if (bytes < 1024ull * 1024ull * 1024ull)
      return std::to_string(bytes / double(1024 * 1024)) + "mb";
    else if (bytes < 1024ull * 1024ull * 1024ull * 1024ull)
      return std::to_string(bytes / double(1024 * 1024 * 1024)) + "gb";
    else
      return std::to_string(bytes / double(1024) / double(1024) / double(1024) /
                            double(1024)) +
             "tb";
  }

public:
  /// @brief Default constructor
  MemoryDebugger()
      : counter_(0), bytes_(0),
        init_(std::chrono::high_resolution_clock::now()) {}

  /// @brief Clears memory debugger
  void clear() {
    counter_ = 0;
    bytes_ = 0;
    objects_.clear();
  }

  /// @brief Returns a string representation of the memory debugger
  inline void pretty_print(std::ostream &os = Log(log::info)) const {
    using namespace std::literals;

    os << "Memory debugger (ID=" << std::to_string(id) << ")\n";
    for (const auto &obj : objects_)
      os << "[" << std::right << std::setw(10) << (obj.first - init_) / 1ns
         << "ns] " << std::right << std::setw(10) << obj.second.name_ << " "
         << std::right << std::setw(10) << convert_bytes(obj.second.bytes_)
         << "\n";
    os << "[     Total  ] " << std::right << std::setw(10) << counter_ << " "
       << std::right << std::setw(10) << convert_bytes(bytes_) << "\n";
  }

  /// @brief Registers generic type to memory debugger
  template <typename T> void add(const std::string &name, const T &obj) {
    counter_++;
    bytes_ += sizeof(obj);
    objects_.insert(
        std::pair<std::chrono::high_resolution_clock::time_point, MemoryObject>(
            std::chrono::high_resolution_clock::now(),
            MemoryObject(name, sizeof(obj))));
  }

  /// @brief Registers torch::Tensor to memory debugger
  void add(const std::string &name, const torch::Tensor &tensor) {
    counter_++;
    bytes_ += tensor.element_size() * tensor.numel();
    objects_.insert(
        std::pair<std::chrono::high_resolution_clock::time_point, MemoryObject>(
            std::chrono::high_resolution_clock::now(),
            MemoryObject(name, tensor.element_size() * tensor.numel())));
  }

  /// @brief Registers std::array to memory debugger
  template <typename T, std::size_t N>
  void add(const std::string &name, const std::array<T, N> &array) {
    for (std::size_t i = 0; i < N; ++i)
      add(name + std::to_string(i), array[i]);
  }
};

/// @brief Print (as string) a memory debugger object
template <std::size_t id>
inline std::ostream &operator<<(std::ostream &os,
                                const MemoryDebugger<id> &obj) {
  obj.pretty_print(os);
  return os;
}

/// @brief System-wide memory debugger
static MemoryDebugger<std::numeric_limits<std::size_t>::max()>
    global_memory_debugger;

#define register_memory(obj) ::iganet::global_memory_debugger.add(#obj, obj)

} // namespace iganet
