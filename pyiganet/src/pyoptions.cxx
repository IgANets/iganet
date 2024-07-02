/**
   @file pyiganet/pycore.cxx

   @brief PyIgANet core components

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11_json/pybind11_json.hpp>

#include <torch/extension.h>

#include <pyconfig.hpp>

#include <options.hpp>

namespace py = pybind11;

void init_options(py::module_ &m) {

  py::enum_<enum c10::MemoryFormat>(m, "MemoryFormat", "Memory format types")
      .value("Contiguous", c10::MemoryFormat::Contiguous)
      .value("Preserve", c10::MemoryFormat::Preserve)
      .value("ChannelsLast", c10::MemoryFormat::ChannelsLast)
      .value("ChannelsLast3d", c10::MemoryFormat::ChannelsLast3d);

  py::enum_<enum c10::Layout>(m, "Layout", "Memory layout types")
      .value("Strided", c10::Layout::Strided)
      .value("Sparse", c10::Layout::Sparse)
      .value("SparseCsr", c10::Layout::SparseCsr)
      .value("Mkldnn", c10::Layout::Mkldnn)
      .value("SparseCsc", c10::Layout::SparseCsc)
      .value("SparseBsr", c10::Layout::SparseBsr)
      .value("SparseBsc", c10::Layout::SparseBsc);

  py::class_<iganet::Options<pyiganet::real_t>>(m, "Options")

      // Constructors
      .def(py::init<>())

      // Member functions
      .def("device",
           py::overload_cast<>(&iganet::Options<pyiganet::real_t>::device,
                               py::const_),
           "Returns the `device` property")

      .def("device",
           py::overload_cast<torch::Device>(
               &iganet::Options<pyiganet::real_t>::device, py::const_),
           "Returns a new Options object with the `device` property as given")

      .def("device_index",
           py::overload_cast<>(&iganet::Options<pyiganet::real_t>::device_index,
                               py::const_),
           "Returns the `device_index` property")

      .def("device_index",
           py::overload_cast<int16_t>(
               &iganet::Options<pyiganet::real_t>::device_index, py::const_),
           "Returns a new Options object with the `device_index` property as "
           "given")

      .def(
          "dtype",
          [](const iganet::Options<pyiganet::real_t> &self) {
            return self.dtype();
          },
          "Returns the `dtype` property")

      .def("layout",
           py::overload_cast<>(&iganet::Options<pyiganet::real_t>::layout,
                               py::const_),
           "Returns the `layout` property")

      .def("layout",
           py::overload_cast<torch::Layout>(
               &iganet::Options<pyiganet::real_t>::layout, py::const_),
           "Returns a new Options object with the `layout` property as given")

      .def("requires_grad",
           py::overload_cast<>(
               &iganet::Options<pyiganet::real_t>::requires_grad, py::const_),
           "Returns the `requires_grad` property")

      .def("requires_grad",
           py::overload_cast<bool>(
               &iganet::Options<pyiganet::real_t>::requires_grad, py::const_),
           "Returns a new Options object with the `requires_grad` property as "
           "given")

      .def("pinned_memory",
           py::overload_cast<>(
               &iganet::Options<pyiganet::real_t>::pinned_memory, py::const_),
           "Returns the `pinned_memory` property")

      .def("pinned_memory",
           py::overload_cast<bool>(
               &iganet::Options<pyiganet::real_t>::pinned_memory, py::const_),
           "Returns a new Options object with the `pinned_memory` property as "
           "given")

      .def("is_sparse",
           py::overload_cast<>(&iganet::Options<pyiganet::real_t>::is_sparse,
                               py::const_),
           "Returns if the layout is sparse")

      // Print
      .def("__repr__", [](const iganet::Options<pyiganet::real_t> &obj) {
        std::stringstream ss;
        obj.pretty_print(ss);
        return ss.str();
      });
}
