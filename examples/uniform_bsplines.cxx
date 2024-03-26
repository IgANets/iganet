/**
   @file examples/uniform_bsplines.cxx

   @brief Demonstration of the uniform B-spline class

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <iganet.h>
#include <iostream>

int main() {
  iganet::init();
  iganet::verbose(std::cout);
  using real_t = double;

  nlohmann::json json;
  json["res0"] = 50;
  json["res1"] = 50;
  json["cnet"] = true;

  {
    // Univariate uniform B-spline of degree 2 with 6 control points in R^1
    iganet::UniformBSpline<real_t, 1, 2> bspline({6}), color({6});

    // Print information
    iganet::Log(iganet::log::info) << bspline << std::endl;

    // Map control points to phyiscal coordinates
    bspline.transform([](const std::array<real_t, 1> xi) {
      return std::array<real_t, 1>{xi[0] * xi[0]};
    });

    // Map colors
    color.transform([](const std::array<real_t, 1> xi) {
      return std::array<real_t, 1>{xi[0]};
    });

    // Evaluate B-spline at xi=0, xi=0.5, and xi=1
    iganet::Log(iganet::log::info)
        << bspline.eval(iganet::utils::to_tensorArray<real_t>({0.0, 0.5, 1.0}))
        << std::endl;

#ifdef IGANET_WITH_MATPLOT
    // Plot B-spline
    bspline.plot(json)->show();
    bspline.plot(color, json)->show();
#endif

    // Export B-spline to XML
    bspline.to_xml().print(iganet::Log(iganet::log::info));
  }

  {
    // Univariate uniform B-spline of degree 2 with 6 control points in R^2
    iganet::UniformBSpline<real_t, 2, 2> bspline({6});
    iganet::UniformBSpline<real_t, 1, 2> color({6});

    // Print information
    iganet::Log(iganet::log::info) << bspline << std::endl;

    // Map control points to phyiscal coordinates
    bspline.transform([](const std::array<real_t, 1> xi) {
      return std::array<real_t, 2>{xi[0] * xi[0],
                                   sin(static_cast<real_t>(M_PI) * xi[0])};
    });

    // Map colors
    color.transform([](const std::array<real_t, 1> xi) {
      return std::array<real_t, 1>{xi[0]};
    });

    // Evaluate B-spline at xi=0, xi=0.5, and xi=1
    iganet::Log(iganet::log::info)
        << bspline.eval(iganet::utils::to_tensorArray<real_t>({0.0, 0.5, 1.0}))
        << std::endl;

#ifdef IGANET_WITH_MATPLOT
    // Plot B-spline
    bspline.plot(json)->show();
    bspline.plot(color, json)->show();
#endif

    // Export B-spline to XML
    bspline.to_xml().print(iganet::Log(iganet::log::info));
  }

  {
    // Univariate uniform B-spline of degree 2 with 6 control points in R^3
    iganet::UniformBSpline<real_t, 3, 2> bspline({6});
    iganet::UniformBSpline<real_t, 1, 2> color({6});

    // Print information
    iganet::Log(iganet::log::info) << bspline << std::endl;

    // Map control points to phyiscal coordinates
    bspline.transform([](const std::array<real_t, 1> xi) {
      return std::array<real_t, 3>{
          xi[0] * xi[0], sin(static_cast<real_t>(M_PI) * xi[0]), xi[0]};
    });

    // Map colors
    color.transform([](const std::array<real_t, 1> xi) {
      return std::array<real_t, 1>{xi[0]};
    });

    // Evaluate B-spline at xi=0, xi=0.5, and xi=1
    iganet::Log(iganet::log::info)
        << bspline.eval(iganet::utils::to_tensorArray<real_t>({0.0, 0.5, 1.0}))
        << std::endl;

#ifdef IGANET_WITH_MATPLOT
    // Plot B-spline
    bspline.plot(json)->show();
    bspline.plot(color, json)->show();
#endif

    // Export B-spline to XML
    bspline.to_xml().print(iganet::Log(iganet::log::info));
  }

  {
    // Bivariate uniform B-spline of degree 3 in xi-direction and 4
    // in eta-direction with 5 x 6 control points in R^2
    iganet::UniformBSpline<real_t, 2, 3, 4> bspline({5, 6});
    iganet::UniformBSpline<real_t, 1, 3, 4> color({5, 6});

    // Print information
    iganet::Log(iganet::log::info) << bspline << std::endl;

    // Map control points to phyiscal coordinates
    bspline.transform([](const std::array<real_t, 2> xi) {
      return std::array<real_t, 2>{
          (xi[0] + 1) * cos(static_cast<real_t>(M_PI) * xi[1]),
          (xi[0] + 1) * sin(static_cast<real_t>(M_PI) * xi[1])};
    });

    // Map colors
    color.transform([](const std::array<real_t, 2> xi) {
      return std::array<real_t, 1>{xi[0] * xi[1]};
    });

    // Evaluate B-spline at xi=0, xi=0.5, and xi=1
    iganet::Log(iganet::log::info)
        << bspline.eval(iganet::utils::to_tensorArray<real_t>({0.0, 0.5, 1.0},
                                                              {0.0, 0.5, 0.5}))
        << std::endl;

#ifdef IGANET_WITH_MATPLOT
    // Plot B-spline
    bspline.plot(json)->show();
    bspline.plot(color, json)->show();
#endif

    // Export B-spline to XML
    bspline.to_xml().print(iganet::Log(iganet::log::info));
  }

  {
    // Bivariate uniform B-spline of degree 3 in xi-direction and 4
    // in eta-direction with 5 x 6 control points in R^3
    iganet::UniformBSpline<real_t, 3, 3, 4> bspline({5, 6});
    iganet::UniformBSpline<real_t, 1, 3, 4> color({5, 6});

    // Print information
    iganet::Log(iganet::log::info) << bspline << std::endl;

    // Map control points to phyiscal coordinates
    bspline.transform([](const std::array<real_t, 2> xi) {
      return std::array<real_t, 3>{
          (xi[0] + 1) * cos(static_cast<real_t>(M_PI) * xi[1]),
          (xi[0] + 1) * sin(static_cast<real_t>(M_PI) * xi[1]), xi[0]};
    });

    // Map colors
    color.transform([](const std::array<real_t, 2> xi) {
      return std::array<real_t, 1>{xi[0] * xi[1]};
    });

    // Evaluate B-spline at (xi=0,eta=0), (xi=0.5,eta=0.5), and (xi=1,eta=0.5)
    iganet::Log(iganet::log::info)
        << bspline.eval(iganet::utils::to_tensorArray<real_t>({0.0, 0.5, 1.0},
                                                              {0.0, 0.5, 0.5}))
        << std::endl;

#ifdef IGANET_WITH_MATPLOT
    // Plot B-spline
    bspline.plot(json)->show();
    bspline.plot(color, json)->show();
#endif

    // Export B-spline to XML
    bspline.to_xml().print(iganet::Log(iganet::log::info));

    auto xi =
        iganet::utils::to_tensorArray<real_t>({0.0, 0.5, 1.0}, {0.0, 0.5, 0.5});
  }

  return 0;
}
