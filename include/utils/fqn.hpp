/**
   @file include/utils/fqn.hpp

   @brief Full qualified name utility functions

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <core.hpp>

#include <torch/csrc/api/include/torch/types.h>
#include <torch/torch.h>

namespace iganet {
namespace utils {

/// @brief Full qualified name descriptor
class FullQualifiedName {
public:
  /// @brief Returns the full qualified name of the object
  ///
  /// @result Full qualified name of the object as string
  inline const virtual std::string &name() const noexcept {
    // If the name optional is empty at this point, we grab the name of the
    // dynamic type via RTTI. Note that we cannot do this in the constructor,
    // because in the constructor of a base class `this` always refers to the
    // base type. Inheritance effectively does not work in constructors. Also
    // this note from http://en.cppreference.com/w/cpp/language/typeid: If
    // typeid is used on an object under construction or destruction (in a
    // destructor or in a constructor, including constructor's initializer list
    // or default member initializers), then the std::type_info object referred
    // to by this typeid represents the class that is being constructed or
    // destroyed even if it is not the most-derived class.
    if (!name_.has_value()) {
      name_ = c10::demangle(typeid(*this).name());
#if defined(_WIN32)
      // Windows adds "struct" or "class" as a prefix.
      if (name_->find("struct ") == 0) {
        name_->erase(name_->begin(), name_->begin() + 7);
      } else if (name_->find("class ") == 0) {
        name_->erase(name_->begin(), name_->begin() + 6);
      }
#endif // defined(_WIN32)
    }
    return *name_;
  }

  /// @brief Returns a string representation
  inline virtual void
  pretty_print(std::ostream &os = Log(log::info)) const noexcept = 0;

protected:
  /// @brief String storing the full qualified name of the object
  mutable at::optional<std::string> name_;
};

} // namespace utils
} // namespace iganet
