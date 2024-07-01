/**
   @file include/multipatch.hpp

   @brief Multi-patch container class

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

namespace iganet {

/// @brief Multi-patch container class
///
/// This class implements a container for a set of patches and their
/// topology, that is, the interface connections and outer boundary
/// faces.
template <typename Patch> class MultiPatch {

public:
  /// @brief Default constructor
  MultiPatch() {}

  /// @brief Copy constructor
  MultiPatch(const MultiPatch &other) : patches_(other.patches_) {}

  /// @brief Move constructor
  MultiPatch(MultiPatch &&other) { patches_.swap(other.patches_); }

public:
  /// @brief Returns an iterator to the patches
  auto begin() { return patches_.begin(); }

  /// @brief Returns a const-iterator to the patches
  /// @{
  auto begin() const { return patches_.begin(); }
  auto cbegin() const noexcept { return patches_.cbegin(); }
  /// @}

  /// @brief Returns an iterator to the end of the patches
  auto end() { return patches_.end(); }

  /// @brief Returns a const-iterator to the end of the patches
  /// @{
  auto end() const { return patches_.end(); }
  auto cend() const noexcept { return patches_.cend(); }
  /// @}

  /// @brief Returns a reverse iterator to the patches
  auto rbegin() { return patches_.rbegin(); }

  /// @brief Returns a reverse const-iterator to the patches
  /// @{
  auto rbegin() const { return patches_.rbegin(); }
  auto crbegin() const noexcept { return patches_.crbegin(); }
  /// @}

  /// @brief Returns a reverse iterator to the end of the patches
  auto rend() { return patches_.rend(); }

  /// @brief Returns a reverse const-iterator to the end of the patches
  /// @{
  auto rend() const { return patches_.rend(); }
  auto crend() const noexcept { return patches_.crend(); }
  /// @}

public:
  /// @brief Returns the number of patches
  std::size_t npatches() const { return patches_.size(); }

  /// @brief Returns the number of interfaces
  std::size_t ninterfaces() const { return patches_.size(); }

  /// @brief Returns the number of outer boundaries
  std::size_t nboundaries() const { return patches_.size(); }

public:
  /// @brief Adds a single patch
  /// @{
  std::size_t addPatch(std::shared_ptr<Patch> patch) {
    std::size_t index = patches_.size();
    patches_.push_back(patch);
    return index;
  }

  std::size_t addPatch(std::unique_ptr<Patch> patch) {
    std::size_t index = patches_.size();
    patches_.push_back(patch.release());
    return index;
  }
  /// @}

  /// @brief Removes all patches
  void clear() { patches_.clear(); }

  /// @brief Returns a non-constant reference to a single patch
  Patch &patch(std::size_t index) {
    assert(index >= 0 && index < patches_.size());
    return *patches_[index];
  }

  /// @brief Returns a constant reference to a single patch
  const Patch &patch(std::size_t index) const {
    assert(index >= 0 && index < patches_.size());
    return *patches_[index];
  }

  /// @brief Returns a constant reference to the vector of patches
  std::vector<std::shared_ptr<Patch>> &patches() const { return patches_; }

  /// @brief Returns the index of a given single patch
  /// @{
  std::size_t findPatchIndex(const Patch &patch) const {
    return findPatchIndex(&patch);
  }

  std::size_t findPatchIndex(const Patch *patch) const {
    auto it = std::find(patches_.begin(), patches_.end(), patch);
    if (it != patches_.end())
      throw std::runtime_error("Did not find the patch index");

    return it - patches_.begin();
  }
  /// @}

private:
  /// @brief Vector of single-patch objects
  std::vector<std::shared_ptr<Patch>> patches_;
};

} // namespace iganet
