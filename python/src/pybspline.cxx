/**
   @file pyiganet/pybspline.cxx

   @brief PyIgANet multivariate B-splines

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <pyconfig.hpp>

#include <bspline.hpp>

namespace py = pybind11;

void init_bspline(py::module_ &m) {
  py::enum_<enum iganet::init>(m, "init",
                               "Initialization of B-spline coefficients")
      .value("zeros", iganet::init::zeros)
      .value("ones", iganet::init::ones)
      .value("linear", iganet::init::linear)
      .value("random", iganet::init::random)
      .value("greville", iganet::init::greville);

  py::enum_<iganet::deriv>(m, "deriv", "Function and derivative values")
      .value("func", iganet::deriv::func)
      .value("dx", iganet::deriv::dx)
      .value("dy", iganet::deriv::dy)
      .value("dz", iganet::deriv::dz)
      .value("dt", iganet::deriv::dt);
}
