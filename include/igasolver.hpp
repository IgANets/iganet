/**
   @file include/igasolver.hpp

   @brief Isogeometric analysis solver

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <boundary.hpp>
#include <functionspace.hpp>
#include <igabase.hpp>

namespace iganet {

/// @brief IgA solver
///
/// This class implements the core functionality of IgA solvers
template <typename GeometryMap, typename Variable>
class IgASolver : public IgABase<GeometryMap, Variable>,
                  private utils::FullQualifiedName {
public:
  /// @brief Base type
  using Base = IgABase<GeometryMap, Variable>;

  /// @brief Base class constructord
  using Base::IgABase;

  /// @brief Returns a string representation of the IgANet object
  inline virtual void
  pretty_print(std::ostream &os = Log(log::info)) const noexcept override {
    os << name() << "(\n"
       << "G = " << Base::G_ << "\n"
       << "f = " << Base::f_ << "\n"
       << "u = " << Base::u_ << "\n)";
  }
};

} // namespace iganet
