/**
   @file unittests/unittest_nonuniform_bspline_eval.cxx

   @brief B-Spline unittests

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <filesystem>
#include <iganet.h>
#include <iostream>

#include <gtest/gtest.h>
#include <unittest_bsplinelib.hpp>
#include <unittest_config.hpp>

using namespace iganet::unittests::literals;

class BSplineTest : public ::testing::Test {
protected:
  using real_t = iganet::unittests::real_t;
  iganet::Options<real_t> options;

  static constexpr auto trafo_parDim1_geoDim1 =
      [](const std::array<real_t, 1> xi) {
        return std::array<real_t, 1>{xi[0] * xi[0]};
      };
  static constexpr auto trafo_parDim1_geoDim2 =
      [](const std::array<real_t, 1> xi) {
        return std::array<real_t, 2>{xi[0] * xi[0],
                                     sin(static_cast<real_t>(M_PI) * xi[0])};
      };
  static constexpr auto trafo_parDim1_geoDim3 =
      [](const std::array<real_t, 1> xi) {
        return std::array<real_t, 3>{
            xi[0] * xi[0], sin(static_cast<real_t>(M_PI) * xi[0]), xi[0]};
      };
  static constexpr auto trafo_parDim1_geoDim4 =
      [](const std::array<real_t, 1> xi) {
        return std::array<real_t, 4>{
            xi[0] * xi[0], sin(static_cast<real_t>(M_PI) * xi[0]), xi[0],
            cos(static_cast<real_t>(M_PI) * xi[0])};
      };

  static constexpr auto trafo_parDim2_geoDim1 =
      [](const std::array<real_t, 2> xi) {
        return std::array<real_t, 1>{xi[0] * xi[1]};
      };
  static constexpr auto trafo_parDim2_geoDim2 =
      [](const std::array<real_t, 2> xi) {
        return std::array<real_t, 2>{xi[0] * xi[1],
                                     sin(static_cast<real_t>(M_PI) * xi[0])};
      };
  static constexpr auto trafo_parDim2_geoDim3 =
      [](const std::array<real_t, 2> xi) {
        return std::array<real_t, 3>{
            xi[0] * xi[1], sin(static_cast<real_t>(M_PI) * xi[0]), xi[1]};
      };
  static constexpr auto trafo_parDim2_geoDim4 =
      [](const std::array<real_t, 2> xi) {
        return std::array<real_t, 4>{
            xi[0] * xi[1], sin(static_cast<real_t>(M_PI) * xi[0]), xi[1],
            cos(static_cast<real_t>(M_PI) * xi[1])};
      };

  static constexpr auto trafo_parDim3_geoDim1 =
      [](const std::array<real_t, 3> xi) {
        return std::array<real_t, 1>{xi[0] * xi[1] * xi[2]};
      };
  static constexpr auto trafo_parDim3_geoDim2 =
      [](const std::array<real_t, 3> xi) {
        return std::array<real_t, 2>{xi[0] * xi[1] * xi[2],
                                     sin(static_cast<real_t>(M_PI) * xi[0])};
      };
  static constexpr auto trafo_parDim3_geoDim3 =
      [](const std::array<real_t, 3> xi) {
        return std::array<real_t, 3>{xi[0] * xi[1] * xi[2],
                                     sin(static_cast<real_t>(M_PI) * xi[0]),
                                     xi[1] * xi[2]};
      };
  static constexpr auto trafo_parDim3_geoDim4 =
      [](const std::array<real_t, 3> xi) {
        return std::array<real_t, 4>{
            xi[0] * xi[1] * xi[2], sin(static_cast<real_t>(M_PI) * xi[0]),
            xi[1] * xi[2], cos(static_cast<real_t>(M_PI) * xi[1])};
      };

  static constexpr auto trafo_parDim4_geoDim1 =
      [](const std::array<real_t, 4> xi) {
        return std::array<real_t, 1>{xi[0] * xi[1] * xi[2] * xi[3]};
      };
  static constexpr auto trafo_parDim4_geoDim2 =
      [](const std::array<real_t, 4> xi) {
        return std::array<real_t, 2>{xi[0] * xi[1] * xi[2] * xi[3],
                                     sin(static_cast<real_t>(M_PI) * xi[0])};
      };
  static constexpr auto trafo_parDim4_geoDim3 =
      [](const std::array<real_t, 4> xi) {
        return std::array<real_t, 3>{xi[0] * xi[1] * xi[2] * xi[3],
                                     sin(static_cast<real_t>(M_PI) * xi[0]),
                                     xi[1] * xi[2] * xi[3]};
      };
  static constexpr auto trafo_parDim4_geoDim4 =
      [](const std::array<real_t, 4> xi) {
        return std::array<real_t, 4>{xi[0] * xi[1] * xi[2] * xi[3],
                                     sin(static_cast<real_t>(M_PI) * xi[0]),
                                     xi[1] * xi[2] * xi[3],
                                     cos(static_cast<real_t>(M_PI) * xi[1])};
      };
};

TEST_F(BSplineTest, NonUniformBSpline_eval_parDim1_geoDim1_degrees1) {
  iganet::NonUniformBSpline<real_t, 1, 1> geo(
      {{{0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r}}}, iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 1, 1> bspline(
      {{{0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r}}}, iganet::init::zeros, options);
  bspline.transform(trafo_parDim1_geoDim1);
  auto xi = iganet::utils::to_tensorArray(
      options, {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r});
  test_bspline_eval(geo, bspline, xi, 1e-12);
}

TEST_F(BSplineTest, NonUniformBSpline_eval_parDim1_geoDim1_degrees2) {
  iganet::NonUniformBSpline<real_t, 1, 2> geo(
      {{{0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 1, 2> bspline(
      {{{0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::zeros, options);
  bspline.transform(trafo_parDim1_geoDim1);
  auto xi = iganet::utils::to_tensorArray(
      options, {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r});
  test_bspline_eval(geo, bspline, xi, 1e-12);
}

TEST_F(BSplineTest, NonUniformBSpline_eval_parDim1_geoDim1_degrees3) {
  iganet::NonUniformBSpline<real_t, 1, 3> geo(
      {{{0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 1, 3> bspline(
      {{{0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::zeros, options);
  bspline.transform(trafo_parDim1_geoDim1);
  auto xi = iganet::utils::to_tensorArray(
      options, {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r});
  test_bspline_eval(geo, bspline, xi, 1e-12);
}

TEST_F(BSplineTest, NonUniformBSpline_eval_parDim1_geoDim1_degrees4) {
  iganet::NonUniformBSpline<real_t, 1, 4> geo(
      {{{0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r,
         1.0_r}}},
      iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 1, 4> bspline(
      {{{0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r,
         1.0_r}}},
      iganet::init::zeros, options);
  bspline.transform(trafo_parDim1_geoDim1);
  auto xi = iganet::utils::to_tensorArray(
      options, {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r});
  test_bspline_eval(geo, bspline, xi, 1e-12);
}

TEST_F(BSplineTest, NonUniformBSpline_eval_parDim1_geoDim1_degrees5) {
  iganet::NonUniformBSpline<real_t, 1, 5> geo(
      {{{0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r,
         1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 1, 5> bspline(
      {{{0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r,
         1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::zeros, options);
  bspline.transform(trafo_parDim1_geoDim1);
  auto xi = iganet::utils::to_tensorArray(
      options, {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r});
  test_bspline_eval(geo, bspline, xi, 1e-12);
}

TEST_F(BSplineTest, NonUniformBSpline_eval_parDim1_geoDim1_degrees6) {
  iganet::NonUniformBSpline<real_t, 1, 6> geo(
      {{{0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r,
         1.0_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 1, 6> bspline(
      {{{0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r,
         1.0_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::zeros, options);
  bspline.transform(trafo_parDim1_geoDim1);
  auto xi = iganet::utils::to_tensorArray(
      options, {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r});
  test_bspline_eval(geo, bspline, xi, 1e-12);
}

TEST_F(BSplineTest, NonUniformBSpline_eval_parDim1_geoDim2_degrees1) {
  iganet::NonUniformBSpline<real_t, 1, 1> geo(
      {{{0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r}}}, iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 2, 1> bspline(
      {{{0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r}}}, iganet::init::zeros, options);
  bspline.transform(trafo_parDim1_geoDim2);
  auto xi = iganet::utils::to_tensorArray(
      options, {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r});
  test_bspline_eval(geo, bspline, xi, 1e-12);
}

TEST_F(BSplineTest, NonUniformBSpline_eval_parDim1_geoDim2_degrees2) {
  iganet::NonUniformBSpline<real_t, 1, 2> geo(
      {{{0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 2, 2> bspline(
      {{{0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::zeros, options);
  bspline.transform(trafo_parDim1_geoDim2);
  auto xi = iganet::utils::to_tensorArray(
      options, {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r});
  test_bspline_eval(geo, bspline, xi, 1e-12);
}

TEST_F(BSplineTest, NonUniformBSpline_eval_parDim1_geoDim2_degrees3) {
  iganet::NonUniformBSpline<real_t, 1, 3> geo(
      {{{0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 2, 3> bspline(
      {{{0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::zeros, options);
  bspline.transform(trafo_parDim1_geoDim2);
  auto xi = iganet::utils::to_tensorArray(
      options, {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r});
  test_bspline_eval(geo, bspline, xi, 1e-12);
}

TEST_F(BSplineTest, NonUniformBSpline_eval_parDim1_geoDim2_degrees4) {
  iganet::NonUniformBSpline<real_t, 1, 4> geo(
      {{{0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r,
         1.0_r}}},
      iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 2, 4> bspline(
      {{{0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r,
         1.0_r}}},
      iganet::init::zeros, options);
  bspline.transform(trafo_parDim1_geoDim2);
  auto xi = iganet::utils::to_tensorArray(
      options, {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r});
  test_bspline_eval(geo, bspline, xi, 1e-12);
}

TEST_F(BSplineTest, NonUniformBSpline_eval_parDim1_geoDim2_degrees5) {
  iganet::NonUniformBSpline<real_t, 1, 5> geo(
      {{{0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r,
         1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 2, 5> bspline(
      {{{0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r,
         1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::zeros, options);
  bspline.transform(trafo_parDim1_geoDim2);
  auto xi = iganet::utils::to_tensorArray(
      options, {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r});
  test_bspline_eval(geo, bspline, xi, 1e-12);
}

TEST_F(BSplineTest, NonUniformBSpline_eval_parDim1_geoDim2_degrees6) {
  iganet::NonUniformBSpline<real_t, 1, 6> geo(
      {{{0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r,
         1.0_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 2, 6> bspline(
      {{{0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r,
         1.0_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::zeros, options);
  bspline.transform(trafo_parDim1_geoDim2);
  auto xi = iganet::utils::to_tensorArray(
      options, {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r});
  test_bspline_eval(geo, bspline, xi, 1e-11);
}

TEST_F(BSplineTest, NonUniformBSpline_eval_parDim1_geoDim3_degrees1) {
  iganet::NonUniformBSpline<real_t, 1, 1> geo(
      {{{0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r}}}, iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 3, 1> bspline(
      {{{0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r}}}, iganet::init::zeros, options);
  bspline.transform(trafo_parDim1_geoDim3);
  auto xi = iganet::utils::to_tensorArray(
      options, {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r});
  test_bspline_eval(geo, bspline, xi, 1e-12);
}

TEST_F(BSplineTest, NonUniformBSpline_eval_parDim1_geoDim3_degrees2) {
  iganet::NonUniformBSpline<real_t, 1, 2> geo(
      {{{0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 3, 2> bspline(
      {{{0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::zeros, options);
  bspline.transform(trafo_parDim1_geoDim3);
  auto xi = iganet::utils::to_tensorArray(
      options, {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r});
  test_bspline_eval(geo, bspline, xi, 1e-12);
}

TEST_F(BSplineTest, NonUniformBSpline_eval_parDim1_geoDim3_degrees3) {
  iganet::NonUniformBSpline<real_t, 1, 3> geo(
      {{{0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 3, 3> bspline(
      {{{0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::zeros, options);
  bspline.transform(trafo_parDim1_geoDim3);
  auto xi = iganet::utils::to_tensorArray(
      options, {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r});
  test_bspline_eval(geo, bspline, xi, 1e-12);
}

TEST_F(BSplineTest, NonUniformBSpline_eval_parDim1_geoDim3_degrees4) {
  iganet::NonUniformBSpline<real_t, 1, 4> geo(
      {{{0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r,
         1.0_r}}},
      iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 3, 4> bspline(
      {{{0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r,
         1.0_r}}},
      iganet::init::zeros, options);
  bspline.transform(trafo_parDim1_geoDim3);
  auto xi = iganet::utils::to_tensorArray(
      options, {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r});
  test_bspline_eval(geo, bspline, xi, 1e-12);
}

TEST_F(BSplineTest, NonUniformBSpline_eval_parDim1_geoDim3_degrees5) {
  iganet::NonUniformBSpline<real_t, 1, 5> geo(
      {{{0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r,
         1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 3, 5> bspline(
      {{{0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r,
         1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::zeros, options);
  bspline.transform(trafo_parDim1_geoDim3);
  auto xi = iganet::utils::to_tensorArray(
      options, {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r});
  test_bspline_eval(geo, bspline, xi, 1e-12);
}

TEST_F(BSplineTest, NonUniformBSpline_eval_parDim1_geoDim3_degrees6) {
  iganet::NonUniformBSpline<real_t, 1, 6> geo(
      {{{0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r,
         1.0_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 3, 6> bspline(
      {{{0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r,
         1.0_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::zeros, options);
  bspline.transform(trafo_parDim1_geoDim3);
  auto xi = iganet::utils::to_tensorArray(
      options, {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r});
  test_bspline_eval(geo, bspline, xi, 1e-11);
}

TEST_F(BSplineTest, NonUniformBSpline_eval_parDim1_geoDim4_degrees1) {
  iganet::NonUniformBSpline<real_t, 1, 1> geo(
      {{{0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r}}}, iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 4, 1> bspline(
      {{{0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r}}}, iganet::init::zeros, options);
  bspline.transform(trafo_parDim1_geoDim4);
  auto xi = iganet::utils::to_tensorArray(
      options, {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r});
  test_bspline_eval(geo, bspline, xi, 1e-12);
}

TEST_F(BSplineTest, NonUniformBSpline_eval_parDim1_geoDim4_degrees2) {
  iganet::NonUniformBSpline<real_t, 1, 2> geo(
      {{{0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 4, 2> bspline(
      {{{0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::zeros, options);
  bspline.transform(trafo_parDim1_geoDim4);
  auto xi = iganet::utils::to_tensorArray(
      options, {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r});
  test_bspline_eval(geo, bspline, xi, 1e-12);
}

TEST_F(BSplineTest, NonUniformBSpline_eval_parDim1_geoDim4_degrees3) {
  iganet::NonUniformBSpline<real_t, 1, 3> geo(
      {{{0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 4, 3> bspline(
      {{{0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::zeros, options);
  bspline.transform(trafo_parDim1_geoDim4);
  auto xi = iganet::utils::to_tensorArray(
      options, {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r});
  test_bspline_eval(geo, bspline, xi, 1e-12);
}

TEST_F(BSplineTest, NonUniformBSpline_eval_parDim1_geoDim4_degrees4) {
  iganet::NonUniformBSpline<real_t, 1, 4> geo(
      {{{0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r,
         1.0_r}}},
      iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 4, 4> bspline(
      {{{0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r,
         1.0_r}}},
      iganet::init::zeros, options);
  bspline.transform(trafo_parDim1_geoDim4);
  auto xi = iganet::utils::to_tensorArray(
      options, {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r});
  test_bspline_eval(geo, bspline, xi, 1e-12);
}

TEST_F(BSplineTest, NonUniformBSpline_eval_parDim1_geoDim4_degrees5) {
  iganet::NonUniformBSpline<real_t, 1, 5> geo(
      {{{0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r,
         1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 4, 5> bspline(
      {{{0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r,
         1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::zeros, options);
  bspline.transform(trafo_parDim1_geoDim4);
  auto xi = iganet::utils::to_tensorArray(
      options, {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r});
  test_bspline_eval(geo, bspline, xi, 1e-12);
}

TEST_F(BSplineTest, NonUniformBSpline_eval_parDim1_geoDim4_degrees6) {
  iganet::NonUniformBSpline<real_t, 1, 6> geo(
      {{{0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r,
         1.0_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 4, 6> bspline(
      {{{0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r,
         1.0_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::zeros, options);
  bspline.transform(trafo_parDim1_geoDim4);
  auto xi = iganet::utils::to_tensorArray(
      options, {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r});
  test_bspline_eval(geo, bspline, xi, 1e-10);
}

TEST_F(BSplineTest, NonUniformBSpline_eval_parDim2_geoDim1_degrees22) {
  iganet::NonUniformBSpline<real_t, 2, 2, 2> geo(
      {{{0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 1, 2, 2> bspline(
      {{{0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::zeros, options);
  bspline.transform(trafo_parDim2_geoDim1);
  auto xi = iganet::utils::to_tensorArray(
      options, {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r});
  test_bspline_eval(geo, bspline, xi, 1e-12);
}

TEST_F(BSplineTest, NonUniformBSpline_eval_parDim2_geoDim1_degrees46) {
  iganet::NonUniformBSpline<real_t, 2, 4, 6> geo(
      {{{0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r,
         1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r,
         1.0_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 1, 4, 6> bspline(
      {{{0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r,
         1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r,
         1.0_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::zeros, options);
  bspline.transform(trafo_parDim2_geoDim1);
  auto xi = iganet::utils::to_tensorArray(
      options, {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r});
  test_bspline_eval(geo, bspline, xi, 1e-12);
}

TEST_F(BSplineTest, NonUniformBSpline_eval_parDim2_geoDim1_degrees64) {
  iganet::NonUniformBSpline<real_t, 2, 6, 4> geo(
      {{{0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r,
         1.0_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r,
         1.0_r}}},
      iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 1, 6, 4> bspline(
      {{{0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r,
         1.0_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r,
         1.0_r}}},
      iganet::init::zeros, options);
  bspline.transform(trafo_parDim2_geoDim1);
  auto xi = iganet::utils::to_tensorArray(
      options, {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r});
  test_bspline_eval(geo, bspline, xi, 1e-12);
}

TEST_F(BSplineTest, NonUniformBSpline_eval_parDim2_geoDim2_degrees22) {
  iganet::NonUniformBSpline<real_t, 2, 2, 2> geo(
      {{{0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 2, 2, 2> bspline(
      {{{0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::zeros, options);
  bspline.transform(trafo_parDim2_geoDim2);
  auto xi = iganet::utils::to_tensorArray(
      options, {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r});
  test_bspline_eval(geo, bspline, xi, 1e-12);
}

TEST_F(BSplineTest, NonUniformBSpline_eval_parDim2_geoDim2_degrees46) {
  iganet::NonUniformBSpline<real_t, 2, 4, 6> geo(
      {{{0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r,
         1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r,
         1.0_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 2, 4, 6> bspline(
      {{{0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r,
         1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r,
         1.0_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::zeros, options);
  bspline.transform(trafo_parDim2_geoDim2);
  auto xi = iganet::utils::to_tensorArray(
      options, {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r});
  test_bspline_eval(geo, bspline, xi, 1e-12);
}

TEST_F(BSplineTest, NonUniformBSpline_eval_parDim2_geoDim2_degrees64) {
  iganet::NonUniformBSpline<real_t, 2, 6, 4> geo(
      {{{0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r,
         1.0_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r,
         1.0_r}}},
      iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 2, 6, 4> bspline(
      {{{0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r,
         1.0_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r,
         1.0_r}}},
      iganet::init::zeros, options);
  bspline.transform(trafo_parDim2_geoDim2);
  auto xi = iganet::utils::to_tensorArray(
      options, {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r});
  test_bspline_eval(geo, bspline, xi, 1e-12);
}

TEST_F(BSplineTest, NonUniformBSpline_eval_parDim2_geoDim3_degrees22) {
  iganet::NonUniformBSpline<real_t, 2, 2, 2> geo(
      {{{0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 3, 2, 2> bspline(
      {{{0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::zeros, options);
  bspline.transform(trafo_parDim2_geoDim3);
  auto xi = iganet::utils::to_tensorArray(
      options, {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r});
  test_bspline_eval(geo, bspline, xi, 1e-12);
}

TEST_F(BSplineTest, NonUniformBSpline_eval_parDim2_geoDim3_degrees46) {
  iganet::NonUniformBSpline<real_t, 2, 4, 6> geo(
      {{{0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r,
         1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r,
         1.0_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 3, 4, 6> bspline(
      {{{0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r,
         1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r,
         1.0_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::zeros, options);
  bspline.transform(trafo_parDim2_geoDim3);
  auto xi = iganet::utils::to_tensorArray(
      options, {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r});
  test_bspline_eval(geo, bspline, xi, 1e-11);
}

TEST_F(BSplineTest, NonUniformBSpline_eval_parDim2_geoDim3_degrees64) {
  iganet::NonUniformBSpline<real_t, 2, 6, 4> geo(
      {{{0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r,
         1.0_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r,
         1.0_r}}},
      iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 3, 6, 4> bspline(
      {{{0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r,
         1.0_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r,
         1.0_r}}},
      iganet::init::zeros, options);
  bspline.transform(trafo_parDim2_geoDim3);
  auto xi = iganet::utils::to_tensorArray(
      options, {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r});
  test_bspline_eval(geo, bspline, xi, 1e-12);
}

TEST_F(BSplineTest, NonUniformBSpline_eval_parDim2_geoDim4_degrees22) {
  iganet::NonUniformBSpline<real_t, 2, 2, 2> geo(
      {{{0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 4, 2, 2> bspline(
      {{{0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::zeros, options);
  bspline.transform(trafo_parDim2_geoDim4);
  auto xi = iganet::utils::to_tensorArray(
      options, {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r});
  test_bspline_eval(geo, bspline, xi, 1e-12);
}

TEST_F(BSplineTest, NonUniformBSpline_eval_parDim2_geoDim4_degrees46) {
  iganet::NonUniformBSpline<real_t, 2, 4, 6> geo(
      {{{0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r,
         1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r,
         1.0_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 4, 4, 6> bspline(
      {{{0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r,
         1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r,
         1.0_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::zeros, options);
  bspline.transform(trafo_parDim2_geoDim4);
  auto xi = iganet::utils::to_tensorArray(
      options, {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r});
  test_bspline_eval(geo, bspline, xi, 1e-11);
}

TEST_F(BSplineTest, NonUniformBSpline_eval_parDim2_geoDim4_degrees64) {
  iganet::NonUniformBSpline<real_t, 2, 6, 4> geo(
      {{{0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r,
         1.0_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r,
         1.0_r}}},
      iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 4, 6, 4> bspline(
      {{{0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r,
         1.0_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r,
         1.0_r}}},
      iganet::init::zeros, options);
  bspline.transform(trafo_parDim2_geoDim4);
  auto xi = iganet::utils::to_tensorArray(
      options, {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r});
  test_bspline_eval(geo, bspline, xi, 1e-10);
}

TEST_F(BSplineTest, NonUniformBSpline_eval_parDim3_geoDim1_degrees222) {
  iganet::NonUniformBSpline<real_t, 3, 2, 2, 2> geo(
      {{{0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 1, 2, 2, 2> bspline(
      {{{0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::zeros, options);
  bspline.transform(trafo_parDim3_geoDim1);
  auto xi = iganet::utils::to_tensorArray(
      options, {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r});
  test_bspline_eval(geo, bspline, xi, 1e-12);
}

TEST_F(BSplineTest, NonUniformBSpline_eval_parDim3_geoDim1_degrees462) {
  iganet::NonUniformBSpline<real_t, 3, 4, 6, 2> geo(
      {{{0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r,
         1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r,
         1.0_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 1, 4, 6, 2> bspline(
      {{{0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r,
         1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r,
         1.0_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::zeros, options);
  bspline.transform(trafo_parDim3_geoDim1);
  auto xi = iganet::utils::to_tensorArray(
      options, {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r});
  test_bspline_eval(geo, bspline, xi, 1e-12);
}

TEST_F(BSplineTest, NonUniformBSpline_eval_parDim3_geoDim1_degrees642) {
  iganet::NonUniformBSpline<real_t, 3, 6, 4, 2> geo(
      {{{0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r,
         1.0_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r,
         1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 1, 6, 4, 2> bspline(
      {{{0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r,
         1.0_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r,
         1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::zeros, options);
  bspline.transform(trafo_parDim3_geoDim1);
  auto xi = iganet::utils::to_tensorArray(
      options, {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r});
  test_bspline_eval(geo, bspline, xi, 1e-12);
}

TEST_F(BSplineTest, NonUniformBSpline_eval_parDim3_geoDim2_degrees222) {
  iganet::NonUniformBSpline<real_t, 3, 2, 2, 2> geo(
      {{{0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 2, 2, 2, 2> bspline(
      {{{0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::zeros, options);
  bspline.transform(trafo_parDim3_geoDim2);
  auto xi = iganet::utils::to_tensorArray(
      options, {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r});
  test_bspline_eval(geo, bspline, xi, 1e-12);
}

TEST_F(BSplineTest, NonUniformBSpline_eval_parDim3_geoDim2_degrees462) {
  iganet::NonUniformBSpline<real_t, 3, 4, 6, 2> geo(
      {{{0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r,
         1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r,
         1.0_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 2, 4, 6, 2> bspline(
      {{{0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r,
         1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r,
         1.0_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::zeros, options);
  bspline.transform(trafo_parDim3_geoDim2);
  auto xi = iganet::utils::to_tensorArray(
      options, {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r});
  test_bspline_eval(geo, bspline, xi, 1e-12);
}

TEST_F(BSplineTest, NonUniformBSpline_eval_parDim3_geoDim2_degrees642) {
  iganet::NonUniformBSpline<real_t, 3, 6, 4, 2> geo(
      {{{0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r,
         1.0_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r,
         1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 2, 6, 4, 2> bspline(
      {{{0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r,
         1.0_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r,
         1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::zeros, options);
  bspline.transform(trafo_parDim3_geoDim2);
  auto xi = iganet::utils::to_tensorArray(
      options, {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r});
  test_bspline_eval(geo, bspline, xi, 1e-12);
}

TEST_F(BSplineTest, NonUniformBSpline_eval_parDim3_geoDim3_degrees222) {
  iganet::NonUniformBSpline<real_t, 3, 2, 2, 2> geo(
      {{{0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 3, 2, 2, 2> bspline(
      {{{0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::zeros, options);
  bspline.transform(trafo_parDim3_geoDim3);
  auto xi = iganet::utils::to_tensorArray(
      options, {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r});
  test_bspline_eval(geo, bspline, xi, 1e-12);
}

TEST_F(BSplineTest, NonUniformBSpline_eval_parDim3_geoDim3_degrees462) {
  iganet::NonUniformBSpline<real_t, 3, 4, 6, 2> geo(
      {{{0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r,
         1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r,
         1.0_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 3, 4, 6, 2> bspline(
      {{{0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r,
         1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r,
         1.0_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::zeros, options);
  bspline.transform(trafo_parDim3_geoDim3);
  auto xi = iganet::utils::to_tensorArray(
      options, {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r});
  test_bspline_eval(geo, bspline, xi, 1e-12);
}

TEST_F(BSplineTest, NonUniformBSpline_eval_parDim3_geoDim3_degrees642) {
  iganet::NonUniformBSpline<real_t, 3, 6, 4, 2> geo(
      {{{0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r,
         1.0_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r,
         1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 3, 6, 4, 2> bspline(
      {{{0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r,
         1.0_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r,
         1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::zeros, options);
  bspline.transform(trafo_parDim3_geoDim3);
  auto xi = iganet::utils::to_tensorArray(
      options, {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r});
  test_bspline_eval(geo, bspline, xi, 1e-12);
}

TEST_F(BSplineTest, NonUniformBSpline_eval_parDim3_geoDim4_degrees222) {
  iganet::NonUniformBSpline<real_t, 3, 2, 2, 2> geo(
      {{{0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 4, 2, 2, 2> bspline(
      {{{0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::zeros, options);
  bspline.transform(trafo_parDim3_geoDim4);
  auto xi = iganet::utils::to_tensorArray(
      options, {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r});
  test_bspline_eval(geo, bspline, xi, 1e-12);
}

TEST_F(BSplineTest, NonUniformBSpline_eval_parDim3_geoDim4_degrees462) {
  iganet::NonUniformBSpline<real_t, 3, 4, 6, 2> geo(
      {{{0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r,
         1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r,
         1.0_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 4, 4, 6, 2> bspline(
      {{{0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r,
         1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r,
         1.0_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::zeros, options);
  bspline.transform(trafo_parDim3_geoDim4);
  auto xi = iganet::utils::to_tensorArray(
      options, {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r});
  test_bspline_eval(geo, bspline, xi, 1e-12);
}

TEST_F(BSplineTest, NonUniformBSpline_eval_parDim3_geoDim4_degrees642) {
  iganet::NonUniformBSpline<real_t, 3, 6, 4, 2> geo(
      {{{0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r,
         1.0_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r,
         1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 4, 6, 4, 2> bspline(
      {{{0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r,
         1.0_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r,
         1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::zeros, options);
  bspline.transform(trafo_parDim3_geoDim4);
  auto xi = iganet::utils::to_tensorArray(
      options, {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r});
  test_bspline_eval(geo, bspline, xi, 1e-10);
}

TEST_F(BSplineTest, NonUniformBSpline_eval_parDim4_geoDim1_degrees2222) {
  iganet::NonUniformBSpline<real_t, 4, 2, 2, 2, 2> geo(
      {{{0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 1, 2, 2, 2, 2> bspline(
      {{{0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::zeros, options);
  bspline.transform(trafo_parDim4_geoDim1);
  auto xi = iganet::utils::to_tensorArray(
      options, {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r});
  test_bspline_eval(geo, bspline, xi, 1e-12);
}

TEST_F(BSplineTest, NonUniformBSpline_eval_parDim4_geoDim1_degrees2463) {
  iganet::NonUniformBSpline<real_t, 4, 2, 4, 6, 3> geo(
      {{{0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r,
         1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r,
         1.0_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 1, 2, 4, 6, 3> bspline(
      {{{0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r,
         1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r,
         1.0_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::zeros, options);
  bspline.transform(trafo_parDim4_geoDim1);
  auto xi = iganet::utils::to_tensorArray(
      options, {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r});
  test_bspline_eval(geo, bspline, xi, 1e-12);
}

TEST_F(BSplineTest, NonUniformBSpline_eval_parDim4_geoDim2_degrees2222) {
  iganet::NonUniformBSpline<real_t, 4, 2, 2, 2, 2> geo(
      {{{0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 2, 2, 2, 2, 2> bspline(
      {{{0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::zeros, options);
  bspline.transform(trafo_parDim4_geoDim2);
  auto xi = iganet::utils::to_tensorArray(
      options, {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r});
  test_bspline_eval(geo, bspline, xi, 1e-12);
}

TEST_F(BSplineTest, NonUniformBSpline_eval_parDim4_geoDim2_degrees2463) {
  iganet::NonUniformBSpline<real_t, 4, 2, 4, 6, 3> geo(
      {{{0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r,
         1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r,
         1.0_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 2, 2, 4, 6, 3> bspline(
      {{{0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r,
         1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r,
         1.0_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::zeros, options);
  bspline.transform(trafo_parDim4_geoDim2);
  auto xi = iganet::utils::to_tensorArray(
      options, {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r});
  test_bspline_eval(geo, bspline, xi, 1e-12);
}

TEST_F(BSplineTest, NonUniformBSpline_eval_parDim4_geoDim3_degrees2222) {
  iganet::NonUniformBSpline<real_t, 4, 2, 2, 2, 2> geo(
      {{{0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 3, 2, 2, 2, 2> bspline(
      {{{0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::zeros, options);
  bspline.transform(trafo_parDim4_geoDim3);
  auto xi = iganet::utils::to_tensorArray(
      options, {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r});
  test_bspline_eval(geo, bspline, xi, 1e-12);
}

TEST_F(BSplineTest, NonUniformBSpline_eval_parDim4_geoDim3_degrees2463) {
  iganet::NonUniformBSpline<real_t, 4, 2, 4, 6, 3> geo(
      {{{0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r,
         1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r,
         1.0_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 3, 2, 4, 6, 3> bspline(
      {{{0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r,
         1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r,
         1.0_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::zeros, options);
  bspline.transform(trafo_parDim4_geoDim3);
  auto xi = iganet::utils::to_tensorArray(
      options, {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r});
  test_bspline_eval(geo, bspline, xi, 1e-12);
}

TEST_F(BSplineTest, NonUniformBSpline_eval_parDim4_geoDim4_degrees2222) {
  iganet::NonUniformBSpline<real_t, 4, 2, 2, 2, 2> geo(
      {{{0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 4, 2, 2, 2, 2> bspline(
      {{{0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::zeros, options);
  bspline.transform(trafo_parDim4_geoDim4);
  auto xi = iganet::utils::to_tensorArray(
      options, {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r});
  test_bspline_eval(geo, bspline, xi, 1e-12);
}

TEST_F(BSplineTest, NonUniformBSpline_eval_parDim4_geoDim4_degrees2463) {
  iganet::NonUniformBSpline<real_t, 4, 2, 4, 6, 3> geo(
      {{{0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r,
         1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r,
         1.0_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 4, 2, 4, 6, 3> bspline(
      {{{0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r,
         1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r,
         1.0_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::zeros, options);
  bspline.transform(trafo_parDim4_geoDim4);
  auto xi = iganet::utils::to_tensorArray(
      options, {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r});
  test_bspline_eval(geo, bspline, xi, 1e-12);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  iganet::init();
  return RUN_ALL_TESTS();
}
