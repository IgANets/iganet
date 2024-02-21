/**
   @file examples/uniform_bspline_surface_io.cxx

   @brief Demonstration of B-spline surface input/output functionality

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <iganet.h>
#include <iostream>

int main() {
  std::cout << iganet::verbose;
  using real_t = double;
  iganet::init();

  // Bivariate uniform B-spline of degree 2 in both directions
  iganet::UniformBSpline<real_t, 2, 2, 2> bspline;

  // Load XML file
  pugi::xml_document xml;
  xml.load_file(IGANET_DATA_DIR "surfaces/2d/geo03.xml");

  // Load B-spline from XML object
  bspline.from_xml(xml);

#ifdef IGANET_WITH_MATPLOT
  // Plot B-spline
  bspline.plot(50, 50);
#endif

  // Refine B-Spline
  bspline.uniform_refine(2);

#ifdef IGANET_WITH_MATPLOT
  // Plot B-spline
  bspline.plot(50, 50);
#endif

  // Export B-spline to XML
  std::cout << bspline.to_xml() << std::endl;

  return 0;
}
