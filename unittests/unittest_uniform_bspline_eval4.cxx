/**
   @file unittests/unittest_uniform_bspline_eval4.cxx

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

#pragma nv_diag_suppress 20208

using namespace iganet::unittests::literals;

class BSplineTest : public ::testing::Test {
protected:
  using real_t = iganet::unittests::real_t;
  iganet::Options<real_t> options;

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

TEST_F(BSplineTest, UniformBSpline_eval_parDim4_geoDim1_degrees2222) {
  iganet::UniformBSpline<real_t, 4, 2, 2, 2, 2> geo(
      {11, 5, 3, 8}, iganet::init::greville, options);
  iganet::UniformBSpline<real_t, 1, 2, 2, 2, 2> bspline(
      {11, 5, 3, 8}, iganet::init::zeros, options);
  bspline.transform(trafo_parDim4_geoDim1);
  auto xi = iganet::utils::to_tensorArray(
      options, {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r});
  test_bspline_eval(geo, bspline, xi, 1e-12);
}

TEST_F(BSplineTest, UniformBSpline_eval_parDim4_geoDim1_degrees2643) {
  iganet::UniformBSpline<real_t, 4, 2, 6, 4, 3> geo(
      {3, 11, 5, 8}, iganet::init::greville, options);
  iganet::UniformBSpline<real_t, 1, 2, 6, 4, 3> bspline(
      {3, 11, 5, 8}, iganet::init::zeros, options);
  bspline.transform(trafo_parDim4_geoDim1);
  auto xi = iganet::utils::to_tensorArray(
      options, {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r});
  test_bspline_eval(geo, bspline, xi, 1e-12);
}

TEST_F(BSplineTest, UniformBSpline_eval_parDim4_geoDim2_degrees2222) {
  iganet::UniformBSpline<real_t, 4, 2, 2, 2, 2> geo(
      {11, 5, 3, 8}, iganet::init::greville, options);
  iganet::UniformBSpline<real_t, 2, 2, 2, 2, 2> bspline(
      {11, 5, 3, 8}, iganet::init::zeros, options);
  bspline.transform(trafo_parDim4_geoDim2);
  auto xi = iganet::utils::to_tensorArray(
      options, {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r});
  test_bspline_eval(geo, bspline, xi, 1e-12);
}

TEST_F(BSplineTest, UniformBSpline_eval_parDim4_geoDim2_degrees2643) {
  iganet::UniformBSpline<real_t, 4, 2, 6, 4, 3> geo(
      {3, 11, 5, 8}, iganet::init::greville, options);
  iganet::UniformBSpline<real_t, 2, 2, 6, 4, 3> bspline(
      {3, 11, 5, 8}, iganet::init::zeros, options);
  bspline.transform(trafo_parDim4_geoDim2);
  auto xi = iganet::utils::to_tensorArray(
      options, {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r});
  test_bspline_eval(geo, bspline, xi, 1e-12);
}

TEST_F(BSplineTest, UniformBSpline_eval_parDim4_geoDim3_degrees2222) {
  iganet::UniformBSpline<real_t, 4, 2, 2, 2, 2> geo(
      {11, 5, 3, 8}, iganet::init::greville, options);
  iganet::UniformBSpline<real_t, 3, 2, 2, 2, 2> bspline(
      {11, 5, 3, 8}, iganet::init::zeros, options);
  bspline.transform(trafo_parDim4_geoDim3);
  auto xi = iganet::utils::to_tensorArray(
      options, {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r});
  test_bspline_eval(geo, bspline, xi, 1e-12);
}

TEST_F(BSplineTest, UniformBSpline_eval_parDim4_geoDim3_degrees2643) {
  iganet::UniformBSpline<real_t, 4, 2, 6, 4, 3> geo(
      {3, 11, 5, 8}, iganet::init::greville, options);
  iganet::UniformBSpline<real_t, 3, 2, 6, 4, 3> bspline(
      {3, 11, 5, 8}, iganet::init::zeros, options);
  bspline.transform(trafo_parDim4_geoDim3);
  auto xi = iganet::utils::to_tensorArray(
      options, {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r});
  test_bspline_eval(geo, bspline, xi, 1e-12);
}

TEST_F(BSplineTest, UniformBSpline_eval_parDim4_geoDim4_degrees2222) {
  iganet::UniformBSpline<real_t, 4, 2, 2, 2, 2> geo(
      {11, 5, 3, 8}, iganet::init::greville, options);
  iganet::UniformBSpline<real_t, 4, 2, 2, 2, 2> bspline(
      {11, 5, 3, 8}, iganet::init::zeros, options);
  bspline.transform(trafo_parDim4_geoDim4);
  auto xi = iganet::utils::to_tensorArray(
      options, {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r},
      {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r});
  test_bspline_eval(geo, bspline, xi, 1e-12);
}

TEST_F(BSplineTest, UniformBSpline_eval_parDim4_geoDim4_degrees2643) {
  iganet::UniformBSpline<real_t, 4, 2, 6, 4, 3> geo(
      {3, 11, 5, 8}, iganet::init::greville, options);
  iganet::UniformBSpline<real_t, 4, 2, 6, 4, 3> bspline(
      {3, 11, 5, 8}, iganet::init::zeros, options);
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
  int result = RUN_ALL_TESTS();
  iganet::finalize();
  return result;
}

#pragma nv_diag_default 20208
