/**
   @file examples/uniform_bspline_surface_gismo.cxx

   @brief Demonstration of B-spline surface G+Smo functionality

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <iganet.h>
#include <iostream>

int main() {
  iganet::verbose(std::cout);
  using real_t = double;
  iganet::init();

  // Bivariate uniform B-spline of degree 2 in both directions
  iganet::UniformBSpline<real_t, 2, 2, 2> bspline;

  // Load XML file
  pugi::xml_document xml;
  xml.load_file(IGANET_DATA_DIR "surfaces/2d/geo03.xml");

  // Load B-spline from XML object
  bspline.from_xml(xml);

#ifdef IGANET_WITH_GISMO
  // Create gsTensorBSpline object
  auto bspline_gismo = bspline.to_gismo();

  // Export as ParaView file
  std::string out = "Geometry";
  gsWriteParaview(bspline_gismo, out);

  out = "Basis";
  gsWriteParaview(bspline_gismo.basis(), out);

  gsMesh<real_t> mesh;
  bspline_gismo.controlNet(mesh);

  out = "ContolNet";
  gsWriteParaview(mesh, out);

  // Optimize parametrization with PDE-based approach
  gsBarrierPatch<2, real_t> opt(bspline_gismo, false);
  opt.options().setInt("ParamMethod", 1);
  opt.compute();

  // Convert result back into B-spline object
  gsMultiPatch<real_t> mp = opt.result();
  bspline.from_gismo(
      dynamic_cast<const gsTensorBSpline<2, real_t> &>(mp.patch(0)));

  // Export as ParaView file
  out = "Geometry_opt";
  gsWriteParaview(mp.patch(0), out);

  out = "Basis_opt";
  gsWriteParaview(mp.patch(0).basis(), out);

  mp.patch(0).controlNet(mesh);

  out = "ContolNet_opt";
  gsWriteParaview(mesh, out);
#endif

  return 0;
}
