/**
   @file unittests/unittest_boundary.cxx

   @brief Boundary unittests

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
#include <unittest_config.hpp>

using namespace iganet::unittests::literals;

class BoundaryTest : public ::testing::Test {
public:
  BoundaryTest() { std::srand(std::time(nullptr)); }

protected:
  using real_t = iganet::unittests::real_t;
  iganet::Options<real_t> options;
};

TEST_F(BoundaryTest, Boundary_parDim1_geoDim1_degrees2) {
  using iganet::deriv;
  using iganet::side;
  using BSpline = iganet::UniformBSpline<real_t, 1, 2>;
  iganet::Boundary<BSpline> boundary({0}, iganet::init::greville, options);

  auto xi = std::tuple{std::array<torch::Tensor, 0>{},
                       std::array<torch::Tensor, 0>{}};

  // Evaluation
  auto values = boundary.eval<deriv::func, false>(xi);

  EXPECT_TRUE(torch::equal(*std::get<side::left - 1>(values)[0],
                           torch::ones(1, options)));
  EXPECT_TRUE(torch::equal(*std::get<side::right - 1>(values)[0],
                           torch::ones(1, options)));

  values = boundary.eval<deriv::dx, false>(xi);

  EXPECT_TRUE(torch::equal(*std::get<side::left - 1>(values)[0],
                           torch::zeros(1, options)));
  EXPECT_TRUE(torch::equal(*std::get<side::right - 1>(values)[0],
                           torch::zeros(1, options)));

  values = boundary.eval<deriv::dx ^ 2, false>(xi);

  EXPECT_TRUE(torch::equal(*std::get<side::left - 1>(values)[0],
                           torch::zeros(1, options)));
  EXPECT_TRUE(torch::equal(*std::get<side::right - 1>(values)[0],
                           torch::zeros(1, options)));

  // Evaluation from precomputed coefficients and basis functions
  auto knot_indices = boundary.find_knot_indices(xi);
  auto coeff_indices = boundary.find_coeff_indices(knot_indices);

  auto numel = [](const auto &xi) {
    return std::tuple<int64_t, int64_t>{1, 1};
  };
  auto sizes = [](const auto &xi) {
    return std::tuple{torch::IntArrayRef{}, torch::IntArrayRef{}};
  };

  auto basfunc =
      boundary.template eval_basfunc<deriv::func, false>(xi, knot_indices);
  values = boundary.eval_from_precomputed(basfunc, coeff_indices, numel(xi),
                                          sizes(xi));

  EXPECT_TRUE(torch::equal(*std::get<side::left - 1>(values)[0],
                           torch::ones({}, options)));
  EXPECT_TRUE(torch::equal(*std::get<side::right - 1>(values)[0],
                           torch::ones({}, options)));

  basfunc = boundary.template eval_basfunc<deriv::dx, false>(xi, knot_indices);
  values = boundary.eval_from_precomputed(basfunc, coeff_indices, numel(xi),
                                          sizes(xi));

  EXPECT_TRUE(torch::equal(*std::get<side::left - 1>(values)[0],
                           torch::zeros({}, options)));
  EXPECT_TRUE(torch::equal(*std::get<side::right - 1>(values)[0],
                           torch::zeros({}, options)));

  basfunc =
      boundary.template eval_basfunc<deriv::dx ^ 2, false>(xi, knot_indices);
  values = boundary.eval_from_precomputed(basfunc, coeff_indices, numel(xi),
                                          sizes(xi));

  EXPECT_TRUE(torch::equal(*std::get<side::left - 1>(values)[0],
                           torch::zeros({}, options)));
  EXPECT_TRUE(torch::equal(*std::get<side::right - 1>(values)[0],
                           torch::zeros({}, options)));
}

TEST_F(BoundaryTest, Boundary_parDim2_geoDim1_degrees23) {
  using iganet::deriv;
  using iganet::side;
  using BSpline = iganet::UniformBSpline<real_t, 1, 2, 3>;
  iganet::Boundary<BSpline> boundary({5, 4}, iganet::init::greville, options);

  auto xi = std::tuple{
      iganet::utils::to_tensorArray(options, {1.0_r, 0.2_r, 0.1_r, 0.5_r, 0.9_r,
                                              0.75_r, 0.0_r}) /* west  */,
      iganet::utils::to_tensorArray(options, {1.0_r, 0.2_r, 0.1_r, 0.5_r, 0.9_r,
                                              0.75_r, 0.0_r}) /* east  */,
      iganet::utils::to_tensorArray(
          options,
          {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r}) /* south */,
      iganet::utils::to_tensorArray(
          options,
          {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r}) /* north */};

  // Evaluation
  auto values = boundary.eval<deriv::func, false>(xi);

  iganet::UniformBSpline<real_t, 1, 2> bspline_bdrNS(
      {5}, iganet::init::greville, options);
  iganet::UniformBSpline<real_t, 1, 3> bspline_bdrEW(
      {4}, iganet::init::greville, options);

  EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(values)[0]),
                           *(bspline_bdrNS.eval<deriv::func, false>(
                               std::get<side::north - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(values)[0]),
                           *(bspline_bdrNS.eval<deriv::func, false>(
                               std::get<side::south - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(values)[0]),
                           *(bspline_bdrEW.eval<deriv::func, false>(
                               std::get<side::east - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(values)[0]),
                           *(bspline_bdrEW.eval<deriv::func, false>(
                               std::get<side::west - 1>(xi))[0])));

  values = boundary.eval<deriv::dx, false>(xi);

  EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(values)[0]),
                           *(bspline_bdrNS.eval<deriv::dx, false>(
                               std::get<side::north - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(values)[0]),
                           *(bspline_bdrNS.eval<deriv::dx, false>(
                               std::get<side::south - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(values)[0]),
                           *(bspline_bdrEW.eval<deriv::dx, false>(
                               std::get<side::east - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(values)[0]),
                           *(bspline_bdrEW.eval<deriv::dx, false>(
                               std::get<side::west - 1>(xi))[0])));

  values = boundary.eval<deriv::dx ^ 2, false>(xi);

  EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(values)[0]),
                           *(bspline_bdrNS.eval<deriv::dx ^ 2, false>(
                               std::get<side::north - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(values)[0]),
                           *(bspline_bdrNS.eval<deriv::dx ^ 2, false>(
                               std::get<side::south - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(values)[0]),
                           *(bspline_bdrEW.eval<deriv::dx ^ 2, false>(
                               std::get<side::east - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(values)[0]),
                           *(bspline_bdrEW.eval<deriv::dx ^ 2, false>(
                               std::get<side::west - 1>(xi))[0])));

  values = boundary.eval<deriv::dy, false>(xi);

  EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(values)[0]),
                           *(bspline_bdrNS.eval<deriv::dy, false>(
                               std::get<side::north - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(values)[0]),
                           *(bspline_bdrNS.eval<deriv::dy, false>(
                               std::get<side::south - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(values)[0]),
                           *(bspline_bdrEW.eval<deriv::dy, false>(
                               std::get<side::east - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(values)[0]),
                           *(bspline_bdrEW.eval<deriv::dy, false>(
                               std::get<side::west - 1>(xi))[0])));

  values = boundary.eval<deriv::dy ^ 2, false>(xi);

  EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(values)[0]),
                           *(bspline_bdrNS.eval<deriv::dy ^ 2, false>(
                               std::get<side::north - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(values)[0]),
                           *(bspline_bdrNS.eval<deriv::dy ^ 2, false>(
                               std::get<side::south - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(values)[0]),
                           *(bspline_bdrEW.eval<deriv::dy ^ 2, false>(
                               std::get<side::east - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(values)[0]),
                           *(bspline_bdrEW.eval<deriv::dy ^ 2, false>(
                               std::get<side::west - 1>(xi))[0])));

  values = boundary.eval<deriv::dx + deriv::dy, false>(xi);

  EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(values)[0]),
                           *(bspline_bdrNS.eval<deriv::dx + deriv::dy, false>(
                               std::get<side::north - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(values)[0]),
                           *(bspline_bdrNS.eval<deriv::dx + deriv::dy, false>(
                               std::get<side::south - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(values)[0]),
                           *(bspline_bdrEW.eval<deriv::dx + deriv::dy, false>(
                               std::get<side::east - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(values)[0]),
                           *(bspline_bdrEW.eval<deriv::dx + deriv::dy, false>(
                               std::get<side::west - 1>(xi))[0])));

  // Evaluation from precomputed coefficients and basis functions
  auto knot_indices = boundary.find_knot_indices(xi);
  auto coeff_indices = boundary.find_coeff_indices(knot_indices);

  auto numel = [](const auto &xi) {
    return std::tuple{std::get<0>(xi)[0].numel(), std::get<1>(xi)[0].numel(),
                      std::get<2>(xi)[0].numel(), std::get<3>(xi)[0].numel()};
  };
  auto sizes = [](const auto &xi) {
    return std::tuple{std::get<0>(xi)[0].sizes(), std::get<1>(xi)[0].sizes(),
                      std::get<2>(xi)[0].sizes(), std::get<3>(xi)[0].sizes()};
  };

  auto basfunc =
      boundary.template eval_basfunc<deriv::func, false>(xi, knot_indices);
  values = boundary.eval_from_precomputed(basfunc, coeff_indices, numel(xi),
                                          sizes(xi));

  EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(values)[0]),
                           *(bspline_bdrNS.eval<deriv::func, false>(
                               std::get<side::north - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(values)[0]),
                           *(bspline_bdrNS.eval<deriv::func, false>(
                               std::get<side::south - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(values)[0]),
                           *(bspline_bdrEW.eval<deriv::func, false>(
                               std::get<side::east - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(values)[0]),
                           *(bspline_bdrEW.eval<deriv::func, false>(
                               std::get<side::west - 1>(xi))[0])));

  basfunc = boundary.template eval_basfunc<deriv::dx, false>(xi, knot_indices);
  values = boundary.eval_from_precomputed(basfunc, coeff_indices, numel(xi),
                                          sizes(xi));

  EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(values)[0]),
                           *(bspline_bdrNS.eval<deriv::dx, false>(
                               std::get<side::north - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(values)[0]),
                           *(bspline_bdrNS.eval<deriv::dx, false>(
                               std::get<side::south - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(values)[0]),
                           *(bspline_bdrEW.eval<deriv::dx, false>(
                               std::get<side::east - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(values)[0]),
                           *(bspline_bdrEW.eval<deriv::dx, false>(
                               std::get<side::west - 1>(xi))[0])));

  basfunc =
      boundary.template eval_basfunc<deriv::dx ^ 2, false>(xi, knot_indices);
  values = boundary.eval_from_precomputed(basfunc, coeff_indices, numel(xi),
                                          sizes(xi));

  EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(values)[0]),
                           *(bspline_bdrNS.eval<deriv::dx ^ 2, false>(
                               std::get<side::north - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(values)[0]),
                           *(bspline_bdrNS.eval<deriv::dx ^ 2, false>(
                               std::get<side::south - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(values)[0]),
                           *(bspline_bdrEW.eval<deriv::dx ^ 2, false>(
                               std::get<side::east - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(values)[0]),
                           *(bspline_bdrEW.eval<deriv::dx ^ 2, false>(
                               std::get<side::west - 1>(xi))[0])));

  basfunc = boundary.template eval_basfunc<deriv::dy, false>(xi, knot_indices);
  values = boundary.eval_from_precomputed(basfunc, coeff_indices, numel(xi),
                                          sizes(xi));

  EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(values)[0]),
                           *(bspline_bdrNS.eval<deriv::dy, false>(
                               std::get<side::north - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(values)[0]),
                           *(bspline_bdrNS.eval<deriv::dy, false>(
                               std::get<side::south - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(values)[0]),
                           *(bspline_bdrEW.eval<deriv::dy, false>(
                               std::get<side::east - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(values)[0]),
                           *(bspline_bdrEW.eval<deriv::dy, false>(
                               std::get<side::west - 1>(xi))[0])));

  basfunc =
      boundary.template eval_basfunc<deriv::dy ^ 2, false>(xi, knot_indices);
  values = boundary.eval_from_precomputed(basfunc, coeff_indices, numel(xi),
                                          sizes(xi));

  EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(values)[0]),
                           *(bspline_bdrNS.eval<deriv::dy ^ 2, false>(
                               std::get<side::north - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(values)[0]),
                           *(bspline_bdrNS.eval<deriv::dy ^ 2, false>(
                               std::get<side::south - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(values)[0]),
                           *(bspline_bdrEW.eval<deriv::dy ^ 2, false>(
                               std::get<side::east - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(values)[0]),
                           *(bspline_bdrEW.eval<deriv::dy ^ 2, false>(
                               std::get<side::west - 1>(xi))[0])));

  basfunc = boundary.template eval_basfunc<deriv::dx + deriv::dy, false>(
      xi, knot_indices);
  values = boundary.eval_from_precomputed(basfunc, coeff_indices, numel(xi),
                                          sizes(xi));

  EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(values)[0]),
                           *(bspline_bdrNS.eval<deriv::dx + deriv::dy, false>(
                               std::get<side::north - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(values)[0]),
                           *(bspline_bdrNS.eval<deriv::dx + deriv::dy, false>(
                               std::get<side::south - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(values)[0]),
                           *(bspline_bdrEW.eval<deriv::dx + deriv::dy, false>(
                               std::get<side::east - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(values)[0]),
                           *(bspline_bdrEW.eval<deriv::dx + deriv::dy, false>(
                               std::get<side::west - 1>(xi))[0])));
}

TEST_F(BoundaryTest, Boundary_parDim3_geoDim1_degrees234) {
  using iganet::deriv;
  using iganet::side;
  using BSpline = iganet::UniformBSpline<real_t, 1, 2, 3, 4>;
  iganet::Boundary<BSpline> boundary({5, 4, 7}, iganet::init::greville,
                                     options);

  auto xi = std::tuple{
      iganet::utils::to_tensorArray(
          options, {1.0_r, 0.2_r, 0.1_r, 0.5_r, 0.9_r, 0.75_r, 0.0_r} /* v */,
          {0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r, 0.0_r,
           0.1_r} /* w */) /* west  */,
      iganet::utils::to_tensorArray(
          options, {1.0_r, 0.2_r, 0.1_r, 0.5_r, 0.9_r, 0.75_r, 0.0_r} /* v */,
          {0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r, 0.0_r,
           0.1_r} /* w */) /* east  */,
      iganet::utils::to_tensorArray(
          options, {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r} /* u */,
          {0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r, 0.0_r,
           0.1_r} /* w */) /* south */,
      iganet::utils::to_tensorArray(
          options, {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r} /* u */,
          {0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r, 0.0_r,
           0.1_r} /* w */) /* north */,
      iganet::utils::to_tensorArray(
          options, {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r} /* u */,
          {1.0_r, 0.2_r, 0.1_r, 0.5_r, 0.9_r, 0.75_r,
           0.0_r} /* v */) /* front */,
      iganet::utils::to_tensorArray(
          options, {0.0_r, 0.1_r, 0.2_r, 0.5_r, 0.75_r, 0.9_r, 1.0_r} /* u */,
          {1.0_r, 0.2_r, 0.1_r, 0.5_r, 0.9_r, 0.75_r,
           0.0_r} /* v */) /* back  */};

  // Evaluation
  auto values = boundary.eval<deriv::func, false>(xi);

  iganet::UniformBSpline<real_t, 1, 2, 4> bspline_bdrNS(
      {5, 7}, iganet::init::greville, options);
  iganet::UniformBSpline<real_t, 1, 3, 4> bspline_bdrEW(
      {4, 7}, iganet::init::greville, options);
  iganet::UniformBSpline<real_t, 1, 2, 3> bspline_bdrFB(
      {5, 4}, iganet::init::greville, options);

  EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(values)[0]),
                           *(bspline_bdrNS.eval<deriv::func, false>(
                               std::get<side::north - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(values)[0]),
                           *(bspline_bdrNS.eval<deriv::func, false>(
                               std::get<side::south - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(values)[0]),
                           *(bspline_bdrEW.eval<deriv::func, false>(
                               std::get<side::east - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(values)[0]),
                           *(bspline_bdrEW.eval<deriv::func, false>(
                               std::get<side::west - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::front - 1>(values)[0]),
                           *(bspline_bdrFB.eval<deriv::func, false>(
                               std::get<side::front - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::back - 1>(values)[0]),
                           *(bspline_bdrFB.eval<deriv::func, false>(
                               std::get<side::back - 1>(xi))[0])));

  values = boundary.eval<deriv::dx, false>(xi);

  EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(values)[0]),
                           *(bspline_bdrNS.eval<deriv::dx, false>(
                               std::get<side::north - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(values)[0]),
                           *(bspline_bdrNS.eval<deriv::dx, false>(
                               std::get<side::south - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(values)[0]),
                           *(bspline_bdrEW.eval<deriv::dx, false>(
                               std::get<side::east - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(values)[0]),
                           *(bspline_bdrEW.eval<deriv::dx, false>(
                               std::get<side::west - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::front - 1>(values)[0]),
                           *(bspline_bdrFB.eval<deriv::dx, false>(
                               std::get<side::front - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::back - 1>(values)[0]),
                           *(bspline_bdrFB.eval<deriv::dx, false>(
                               std::get<side::back - 1>(xi))[0])));

  values = boundary.eval<deriv::dx ^ 2, false>(xi);

  EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(values)[0]),
                           *(bspline_bdrNS.eval<deriv::dx ^ 2, false>(
                               std::get<side::north - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(values)[0]),
                           *(bspline_bdrNS.eval<deriv::dx ^ 2, false>(
                               std::get<side::south - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(values)[0]),
                           *(bspline_bdrEW.eval<deriv::dx ^ 2, false>(
                               std::get<side::east - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(values)[0]),
                           *(bspline_bdrEW.eval<deriv::dx ^ 2, false>(
                               std::get<side::west - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::front - 1>(values)[0]),
                           *(bspline_bdrFB.eval<deriv::dx ^ 2, false>(
                               std::get<side::front - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::back - 1>(values)[0]),
                           *(bspline_bdrFB.eval<deriv::dx ^ 2, false>(
                               std::get<side::back - 1>(xi))[0])));

  values = boundary.eval<deriv::dy, false>(xi);

  EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(values)[0]),
                           *(bspline_bdrNS.eval<deriv::dy, false>(
                               std::get<side::north - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(values)[0]),
                           *(bspline_bdrNS.eval<deriv::dy, false>(
                               std::get<side::south - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(values)[0]),
                           *(bspline_bdrEW.eval<deriv::dy, false>(
                               std::get<side::east - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(values)[0]),
                           *(bspline_bdrEW.eval<deriv::dy, false>(
                               std::get<side::west - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::front - 1>(values)[0]),
                           *(bspline_bdrFB.eval<deriv::dy, false>(
                               std::get<side::front - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::back - 1>(values)[0]),
                           *(bspline_bdrFB.eval<deriv::dy, false>(
                               std::get<side::back - 1>(xi))[0])));

  values = boundary.eval<deriv::dy ^ 2, false>(xi);

  EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(values)[0]),
                           *(bspline_bdrNS.eval<deriv::dy ^ 2, false>(
                               std::get<side::north - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(values)[0]),
                           *(bspline_bdrNS.eval<deriv::dy ^ 2, false>(
                               std::get<side::south - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(values)[0]),
                           *(bspline_bdrEW.eval<deriv::dy ^ 2, false>(
                               std::get<side::east - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(values)[0]),
                           *(bspline_bdrEW.eval<deriv::dy ^ 2, false>(
                               std::get<side::west - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::front - 1>(values)[0]),
                           *(bspline_bdrFB.eval<deriv::dy ^ 2, false>(
                               std::get<side::front - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::back - 1>(values)[0]),
                           *(bspline_bdrFB.eval<deriv::dy ^ 2, false>(
                               std::get<side::back - 1>(xi))[0])));

  values = boundary.eval<deriv::dz, false>(xi);

  EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(values)[0]),
                           *(bspline_bdrNS.eval<deriv::dz, false>(
                               std::get<side::north - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(values)[0]),
                           *(bspline_bdrNS.eval<deriv::dz, false>(
                               std::get<side::south - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(values)[0]),
                           *(bspline_bdrEW.eval<deriv::dz, false>(
                               std::get<side::east - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(values)[0]),
                           *(bspline_bdrEW.eval<deriv::dz, false>(
                               std::get<side::west - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::front - 1>(values)[0]),
                           *(bspline_bdrFB.eval<deriv::dz, false>(
                               std::get<side::front - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::back - 1>(values)[0]),
                           *(bspline_bdrFB.eval<deriv::dz, false>(
                               std::get<side::back - 1>(xi))[0])));

  values = boundary.eval<deriv::dz ^ 2, false>(xi);

  EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(values)[0]),
                           *(bspline_bdrNS.eval<deriv::dz ^ 2, false>(
                               std::get<side::north - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(values)[0]),
                           *(bspline_bdrNS.eval<deriv::dz ^ 2, false>(
                               std::get<side::south - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(values)[0]),
                           *(bspline_bdrEW.eval<deriv::dz ^ 2, false>(
                               std::get<side::east - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(values)[0]),
                           *(bspline_bdrEW.eval<deriv::dz ^ 2, false>(
                               std::get<side::west - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::front - 1>(values)[0]),
                           *(bspline_bdrFB.eval<deriv::dz ^ 2, false>(
                               std::get<side::front - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::back - 1>(values)[0]),
                           *(bspline_bdrFB.eval<deriv::dz ^ 2, false>(
                               std::get<side::back - 1>(xi))[0])));

  values = boundary.eval<deriv::dx + deriv::dy, false>(xi);

  EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(values)[0]),
                           *(bspline_bdrNS.eval<deriv::dx + deriv::dy, false>(
                               std::get<side::north - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(values)[0]),
                           *(bspline_bdrNS.eval<deriv::dx + deriv::dy, false>(
                               std::get<side::south - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(values)[0]),
                           *(bspline_bdrEW.eval<deriv::dx + deriv::dy, false>(
                               std::get<side::east - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(values)[0]),
                           *(bspline_bdrEW.eval<deriv::dx + deriv::dy, false>(
                               std::get<side::west - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::front - 1>(values)[0]),
                           *(bspline_bdrFB.eval<deriv::dx + deriv::dy, false>(
                               std::get<side::front - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::back - 1>(values)[0]),
                           *(bspline_bdrFB.eval<deriv::dx + deriv::dy, false>(
                               std::get<side::back - 1>(xi))[0])));

  values = boundary.eval<deriv::dx + deriv::dz, false>(xi);

  EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(values)[0]),
                           *(bspline_bdrNS.eval<deriv::dx + deriv::dz, false>(
                               std::get<side::north - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(values)[0]),
                           *(bspline_bdrNS.eval<deriv::dx + deriv::dz, false>(
                               std::get<side::south - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(values)[0]),
                           *(bspline_bdrEW.eval<deriv::dx + deriv::dz, false>(
                               std::get<side::east - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(values)[0]),
                           *(bspline_bdrEW.eval<deriv::dx + deriv::dz, false>(
                               std::get<side::west - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::front - 1>(values)[0]),
                           *(bspline_bdrFB.eval<deriv::dx + deriv::dz, false>(
                               std::get<side::front - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::back - 1>(values)[0]),
                           *(bspline_bdrFB.eval<deriv::dx + deriv::dz, false>(
                               std::get<side::back - 1>(xi))[0])));

  values = boundary.eval<deriv::dy + deriv::dz, false>(xi);

  EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(values)[0]),
                           *(bspline_bdrNS.eval<deriv::dy + deriv::dz, false>(
                               std::get<side::north - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(values)[0]),
                           *(bspline_bdrNS.eval<deriv::dy + deriv::dz, false>(
                               std::get<side::south - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(values)[0]),
                           *(bspline_bdrEW.eval<deriv::dy + deriv::dz, false>(
                               std::get<side::east - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(values)[0]),
                           *(bspline_bdrEW.eval<deriv::dy + deriv::dz, false>(
                               std::get<side::west - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::front - 1>(values)[0]),
                           *(bspline_bdrFB.eval<deriv::dy + deriv::dz, false>(
                               std::get<side::front - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::back - 1>(values)[0]),
                           *(bspline_bdrFB.eval<deriv::dy + deriv::dz, false>(
                               std::get<side::back - 1>(xi))[0])));

  // Evaluation from precomputed coefficients and basis functions
  auto knot_indices = boundary.find_knot_indices(xi);
  auto coeff_indices = boundary.find_coeff_indices(knot_indices);

  auto numel = [](const auto &xi) {
    return std::tuple{std::get<0>(xi)[0].numel(), std::get<1>(xi)[0].numel(),
                      std::get<2>(xi)[0].numel(), std::get<3>(xi)[0].numel(),
                      std::get<4>(xi)[0].numel(), std::get<5>(xi)[0].numel()};
  };
  auto sizes = [](const auto &xi) {
    return std::tuple{std::get<0>(xi)[0].sizes(), std::get<1>(xi)[0].sizes(),
                      std::get<2>(xi)[0].sizes(), std::get<3>(xi)[0].sizes(),
                      std::get<4>(xi)[0].sizes(), std::get<5>(xi)[0].sizes()};
  };

  auto basfunc =
      boundary.template eval_basfunc<deriv::func, false>(xi, knot_indices);
  values = boundary.eval_from_precomputed(basfunc, coeff_indices, numel(xi),
                                          sizes(xi));

  EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(values)[0]),
                           *(bspline_bdrNS.eval<deriv::func, false>(
                               std::get<side::north - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(values)[0]),
                           *(bspline_bdrNS.eval<deriv::func, false>(
                               std::get<side::south - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(values)[0]),
                           *(bspline_bdrEW.eval<deriv::func, false>(
                               std::get<side::east - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(values)[0]),
                           *(bspline_bdrEW.eval<deriv::func, false>(
                               std::get<side::west - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::front - 1>(values)[0]),
                           *(bspline_bdrFB.eval<deriv::func, false>(
                               std::get<side::front - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::back - 1>(values)[0]),
                           *(bspline_bdrFB.eval<deriv::func, false>(
                               std::get<side::back - 1>(xi))[0])));

  basfunc = boundary.template eval_basfunc<deriv::dx, false>(xi, knot_indices);
  values = boundary.eval_from_precomputed(basfunc, coeff_indices, numel(xi),
                                          sizes(xi));

  EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(values)[0]),
                           *(bspline_bdrNS.eval<deriv::dx, false>(
                               std::get<side::north - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(values)[0]),
                           *(bspline_bdrNS.eval<deriv::dx, false>(
                               std::get<side::south - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(values)[0]),
                           *(bspline_bdrEW.eval<deriv::dx, false>(
                               std::get<side::east - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(values)[0]),
                           *(bspline_bdrEW.eval<deriv::dx, false>(
                               std::get<side::west - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::front - 1>(values)[0]),
                           *(bspline_bdrFB.eval<deriv::dx, false>(
                               std::get<side::front - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::back - 1>(values)[0]),
                           *(bspline_bdrFB.eval<deriv::dx, false>(
                               std::get<side::back - 1>(xi))[0])));

  basfunc =
      boundary.template eval_basfunc<deriv::dx ^ 2, false>(xi, knot_indices);
  values = boundary.eval_from_precomputed(basfunc, coeff_indices, numel(xi),
                                          sizes(xi));

  EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(values)[0]),
                           *(bspline_bdrNS.eval<deriv::dx ^ 2, false>(
                               std::get<side::north - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(values)[0]),
                           *(bspline_bdrNS.eval<deriv::dx ^ 2, false>(
                               std::get<side::south - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(values)[0]),
                           *(bspline_bdrEW.eval<deriv::dx ^ 2, false>(
                               std::get<side::east - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(values)[0]),
                           *(bspline_bdrEW.eval<deriv::dx ^ 2, false>(
                               std::get<side::west - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::front - 1>(values)[0]),
                           *(bspline_bdrFB.eval<deriv::dx ^ 2, false>(
                               std::get<side::front - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::back - 1>(values)[0]),
                           *(bspline_bdrFB.eval<deriv::dx ^ 2, false>(
                               std::get<side::back - 1>(xi))[0])));

  basfunc = boundary.template eval_basfunc<deriv::dy, false>(xi, knot_indices);
  values = boundary.eval_from_precomputed(basfunc, coeff_indices, numel(xi),
                                          sizes(xi));

  EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(values)[0]),
                           *(bspline_bdrNS.eval<deriv::dy, false>(
                               std::get<side::north - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(values)[0]),
                           *(bspline_bdrNS.eval<deriv::dy, false>(
                               std::get<side::south - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(values)[0]),
                           *(bspline_bdrEW.eval<deriv::dy, false>(
                               std::get<side::east - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(values)[0]),
                           *(bspline_bdrEW.eval<deriv::dy, false>(
                               std::get<side::west - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::front - 1>(values)[0]),
                           *(bspline_bdrFB.eval<deriv::dy, false>(
                               std::get<side::front - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::back - 1>(values)[0]),
                           *(bspline_bdrFB.eval<deriv::dy, false>(
                               std::get<side::back - 1>(xi))[0])));

  basfunc =
      boundary.template eval_basfunc<deriv::dy ^ 2, false>(xi, knot_indices);
  values = boundary.eval_from_precomputed(basfunc, coeff_indices, numel(xi),
                                          sizes(xi));

  EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(values)[0]),
                           *(bspline_bdrNS.eval<deriv::dy ^ 2, false>(
                               std::get<side::north - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(values)[0]),
                           *(bspline_bdrNS.eval<deriv::dy ^ 2, false>(
                               std::get<side::south - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(values)[0]),
                           *(bspline_bdrEW.eval<deriv::dy ^ 2, false>(
                               std::get<side::east - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(values)[0]),
                           *(bspline_bdrEW.eval<deriv::dy ^ 2, false>(
                               std::get<side::west - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::front - 1>(values)[0]),
                           *(bspline_bdrFB.eval<deriv::dy ^ 2, false>(
                               std::get<side::front - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::back - 1>(values)[0]),
                           *(bspline_bdrFB.eval<deriv::dy ^ 2, false>(
                               std::get<side::back - 1>(xi))[0])));

  basfunc = boundary.template eval_basfunc<deriv::dz, false>(xi, knot_indices);
  values = boundary.eval_from_precomputed(basfunc, coeff_indices, numel(xi),
                                          sizes(xi));

  EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(values)[0]),
                           *(bspline_bdrNS.eval<deriv::dz, false>(
                               std::get<side::north - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(values)[0]),
                           *(bspline_bdrNS.eval<deriv::dz, false>(
                               std::get<side::south - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(values)[0]),
                           *(bspline_bdrEW.eval<deriv::dz, false>(
                               std::get<side::east - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(values)[0]),
                           *(bspline_bdrEW.eval<deriv::dz, false>(
                               std::get<side::west - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::front - 1>(values)[0]),
                           *(bspline_bdrFB.eval<deriv::dz, false>(
                               std::get<side::front - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::back - 1>(values)[0]),
                           *(bspline_bdrFB.eval<deriv::dz, false>(
                               std::get<side::back - 1>(xi))[0])));

  basfunc =
      boundary.template eval_basfunc<deriv::dz ^ 2, false>(xi, knot_indices);
  values = boundary.eval_from_precomputed(basfunc, coeff_indices, numel(xi),
                                          sizes(xi));

  EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(values)[0]),
                           *(bspline_bdrNS.eval<deriv::dz ^ 2, false>(
                               std::get<side::north - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(values)[0]),
                           *(bspline_bdrNS.eval<deriv::dz ^ 2, false>(
                               std::get<side::south - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(values)[0]),
                           *(bspline_bdrEW.eval<deriv::dz ^ 2, false>(
                               std::get<side::east - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(values)[0]),
                           *(bspline_bdrEW.eval<deriv::dz ^ 2, false>(
                               std::get<side::west - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::front - 1>(values)[0]),
                           *(bspline_bdrFB.eval<deriv::dz ^ 2, false>(
                               std::get<side::front - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::back - 1>(values)[0]),
                           *(bspline_bdrFB.eval<deriv::dz ^ 2, false>(
                               std::get<side::back - 1>(xi))[0])));

  basfunc = boundary.template eval_basfunc<deriv::dx + deriv::dy, false>(
      xi, knot_indices);
  values = boundary.eval_from_precomputed(basfunc, coeff_indices, numel(xi),
                                          sizes(xi));

  EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(values)[0]),
                           *(bspline_bdrNS.eval<deriv::dx + deriv::dy, false>(
                               std::get<side::north - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(values)[0]),
                           *(bspline_bdrNS.eval<deriv::dx + deriv::dy, false>(
                               std::get<side::south - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(values)[0]),
                           *(bspline_bdrEW.eval<deriv::dx + deriv::dy, false>(
                               std::get<side::east - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(values)[0]),
                           *(bspline_bdrEW.eval<deriv::dx + deriv::dy, false>(
                               std::get<side::west - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::front - 1>(values)[0]),
                           *(bspline_bdrFB.eval<deriv::dx + deriv::dy, false>(
                               std::get<side::front - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::back - 1>(values)[0]),
                           *(bspline_bdrFB.eval<deriv::dx + deriv::dy, false>(
                               std::get<side::back - 1>(xi))[0])));

  basfunc = boundary.template eval_basfunc<deriv::dx + deriv::dz, false>(
      xi, knot_indices);
  values = boundary.eval_from_precomputed(basfunc, coeff_indices, numel(xi),
                                          sizes(xi));

  EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(values)[0]),
                           *(bspline_bdrNS.eval<deriv::dx + deriv::dz, false>(
                               std::get<side::north - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(values)[0]),
                           *(bspline_bdrNS.eval<deriv::dx + deriv::dz, false>(
                               std::get<side::south - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(values)[0]),
                           *(bspline_bdrEW.eval<deriv::dx + deriv::dz, false>(
                               std::get<side::east - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(values)[0]),
                           *(bspline_bdrEW.eval<deriv::dx + deriv::dz, false>(
                               std::get<side::west - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::front - 1>(values)[0]),
                           *(bspline_bdrFB.eval<deriv::dx + deriv::dz, false>(
                               std::get<side::front - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::back - 1>(values)[0]),
                           *(bspline_bdrFB.eval<deriv::dx + deriv::dz, false>(
                               std::get<side::back - 1>(xi))[0])));

  basfunc = boundary.template eval_basfunc<deriv::dy + deriv::dz, false>(
      xi, knot_indices);
  values = boundary.eval_from_precomputed(basfunc, coeff_indices, numel(xi),
                                          sizes(xi));

  EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(values)[0]),
                           *(bspline_bdrNS.eval<deriv::dy + deriv::dz, false>(
                               std::get<side::north - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(values)[0]),
                           *(bspline_bdrNS.eval<deriv::dy + deriv::dz, false>(
                               std::get<side::south - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(values)[0]),
                           *(bspline_bdrEW.eval<deriv::dy + deriv::dz, false>(
                               std::get<side::east - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(values)[0]),
                           *(bspline_bdrEW.eval<deriv::dy + deriv::dz, false>(
                               std::get<side::west - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::front - 1>(values)[0]),
                           *(bspline_bdrFB.eval<deriv::dy + deriv::dz, false>(
                               std::get<side::front - 1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::back - 1>(values)[0]),
                           *(bspline_bdrFB.eval<deriv::dy + deriv::dz, false>(
                               std::get<side::back - 1>(xi))[0])));
}

TEST_F(BoundaryTest, Boundary_init) {
  {
    using iganet::side;
    using BSpline = iganet::UniformBSpline<real_t, 1, 2, 3>;
    iganet::Boundary<BSpline> boundary({5, 4}, iganet::init::zeros, options);

    EXPECT_TRUE(torch::equal(boundary.side<side::west>().coeffs(0),
                             torch::zeros(4, options)));
    EXPECT_TRUE(torch::equal(boundary.side<side::east>().coeffs(0),
                             torch::zeros(4, options)));

    EXPECT_TRUE(torch::equal(boundary.side<side::north>().coeffs(0),
                             torch::zeros(5, options)));
    EXPECT_TRUE(torch::equal(boundary.side<side::south>().coeffs(0),
                             torch::zeros(5, options)));
  }

  {
    using iganet::side;
    using BSpline = iganet::UniformBSpline<real_t, 1, 2, 3>;
    iganet::Boundary<BSpline> boundary({5, 4}, iganet::init::ones, options);

    EXPECT_TRUE(torch::equal(boundary.side<side::west>().coeffs(0),
                             torch::ones(4, options)));
    EXPECT_TRUE(torch::equal(boundary.side<side::east>().coeffs(0),
                             torch::ones(4, options)));

    EXPECT_TRUE(torch::equal(boundary.side<side::north>().coeffs(0),
                             torch::ones(5, options)));
    EXPECT_TRUE(torch::equal(boundary.side<side::south>().coeffs(0),
                             torch::ones(5, options)));
  }

  {
    using iganet::side;
    using BSpline = iganet::UniformBSpline<real_t, 1, 2, 3>;
    iganet::Boundary<BSpline> boundary({5, 4}, iganet::init::linear, options);

    EXPECT_TRUE(torch::equal(boundary.side<side::west>().coeffs(0),
                             torch::linspace(0, 1, 4, options)));
    EXPECT_TRUE(torch::equal(boundary.side<side::east>().coeffs(0),
                             torch::linspace(0, 1, 4, options)));

    EXPECT_TRUE(torch::equal(boundary.side<side::north>().coeffs(0),
                             torch::linspace(0, 1, 5, options)));
    EXPECT_TRUE(torch::equal(boundary.side<side::south>().coeffs(0),
                             torch::linspace(0, 1, 5, options)));
  }

  {
    using iganet::side;
    using BSpline = iganet::UniformBSpline<real_t, 1, 1, 1>;
    iganet::Boundary<BSpline> boundary({5, 4}, iganet::init::greville, options);

    EXPECT_TRUE(torch::allclose(boundary.side<side::west>().coeffs(0),
                                torch::linspace(0, 1, 4, options)));
    EXPECT_TRUE(torch::allclose(boundary.side<side::east>().coeffs(0),
                                torch::linspace(0, 1, 4, options)));

    EXPECT_TRUE(torch::allclose(boundary.side<side::north>().coeffs(0),
                                torch::linspace(0, 1, 5, options)));
    EXPECT_TRUE(torch::allclose(boundary.side<side::south>().coeffs(0),
                                torch::linspace(0, 1, 5, options)));
  }
}

TEST_F(BoundaryTest, Boundary_refine) {

  // parDim == 2

  {
    using BSpline = iganet::UniformBSpline<real_t, 1, 2, 3>;
    iganet::Boundary<BSpline> boundary({5, 4}, iganet::init::greville, options);
    iganet::Boundary<BSpline> boundary_ref({8, 5}, iganet::init::greville,
                                           options);
    boundary.uniform_refine();
    EXPECT_TRUE(boundary.isclose(boundary_ref));
  }

  {
    using BSpline = iganet::UniformBSpline<real_t, 1, 2, 3>;
    iganet::Boundary<BSpline> boundary({5, 4}, iganet::init::greville, options);
    iganet::Boundary<BSpline> boundary_ref({14, 7}, iganet::init::greville,
                                           options);
    boundary.uniform_refine(2);
    EXPECT_TRUE(boundary.isclose(boundary_ref));
  }

  {
    using BSpline = iganet::UniformBSpline<real_t, 1, 2, 3>;
    iganet::Boundary<BSpline> boundary({5, 4}, iganet::init::greville, options);
    iganet::Boundary<BSpline> boundary_ref({8, 4}, iganet::init::greville,
                                           options);
    boundary.uniform_refine(1, 0);
    EXPECT_TRUE(boundary.isclose(boundary_ref));
  }

  {
    using BSpline = iganet::UniformBSpline<real_t, 1, 2, 3>;
    iganet::Boundary<BSpline> boundary({5, 4}, iganet::init::greville, options);
    iganet::Boundary<BSpline> boundary_ref({5, 5}, iganet::init::greville,
                                           options);
    boundary.uniform_refine(1, 1);
    EXPECT_TRUE(boundary.isclose(boundary_ref));
  }

  {
    using BSpline = iganet::UniformBSpline<real_t, 1, 2, 3>;
    iganet::Boundary<BSpline> boundary({5, 4}, iganet::init::greville, options);
    EXPECT_THROW((boundary.uniform_refine(1, 2)), std::runtime_error);
    EXPECT_THROW((boundary.uniform_refine(1, 3)), std::runtime_error);
    EXPECT_THROW((boundary.uniform_refine(1, 4)), std::runtime_error);
  }

  {
    using BSpline = iganet::UniformBSpline<real_t, 1, 2, 3>;
    iganet::Boundary<BSpline> boundary({5, 4}, iganet::init::greville, options);
    iganet::Boundary<BSpline> boundary_ref({14, 5}, iganet::init::greville,
                                           options);
    boundary.uniform_refine(2, 0).uniform_refine(1, 1);
    EXPECT_TRUE(boundary.isclose(boundary_ref));
  }

  // parDim == 3

  {
    using BSpline = iganet::UniformBSpline<real_t, 1, 2, 3, 4>;
    iganet::Boundary<BSpline> boundary({5, 4, 7}, iganet::init::greville,
                                       options);
    iganet::Boundary<BSpline> boundary_ref({8, 5, 10}, iganet::init::greville,
                                           options);
    boundary.uniform_refine();
    EXPECT_TRUE(boundary.isclose(boundary_ref));
  }

  {
    using BSpline = iganet::UniformBSpline<real_t, 1, 2, 3, 4>;
    iganet::Boundary<BSpline> boundary({5, 4, 7}, iganet::init::greville,
                                       options);
    iganet::Boundary<BSpline> boundary_ref({14, 7, 16}, iganet::init::greville,
                                           options);
    boundary.uniform_refine(2);
    EXPECT_TRUE(boundary.isclose(boundary_ref));
  }

  {
    using BSpline = iganet::UniformBSpline<real_t, 1, 2, 3, 4>;
    iganet::Boundary<BSpline> boundary({5, 4, 7}, iganet::init::greville,
                                       options);
    iganet::Boundary<BSpline> boundary_ref({8, 4, 7}, iganet::init::greville,
                                           options);
    boundary.uniform_refine(1, 0);
    EXPECT_TRUE(boundary.isclose(boundary_ref));
  }

  {
    using BSpline = iganet::UniformBSpline<real_t, 1, 2, 3, 4>;
    iganet::Boundary<BSpline> boundary({5, 4, 7}, iganet::init::greville,
                                       options);
    iganet::Boundary<BSpline> boundary_ref({5, 5, 7}, iganet::init::greville,
                                           options);
    boundary.uniform_refine(1, 1);
    EXPECT_TRUE(boundary.isclose(boundary_ref));
  }

  {
    using BSpline = iganet::UniformBSpline<real_t, 1, 2, 3, 4>;
    iganet::Boundary<BSpline> boundary({5, 4, 7}, iganet::init::greville,
                                       options);
    iganet::Boundary<BSpline> boundary_ref({5, 4, 10}, iganet::init::greville,
                                           options);
    boundary.uniform_refine(1, 2);
    EXPECT_TRUE(boundary.isclose(boundary_ref));
  }

  {
    using BSpline = iganet::UniformBSpline<real_t, 1, 2, 3, 4>;
    iganet::Boundary<BSpline> boundary({5, 4, 7}, iganet::init::greville,
                                       options);
    iganet::Boundary<BSpline> boundary_ref({5, 4, 10}, iganet::init::greville,
                                           options);
    EXPECT_THROW((boundary.uniform_refine(1, 3)), std::runtime_error);
    EXPECT_THROW((boundary.uniform_refine(1, 4)), std::runtime_error);
  }

  {
    using BSpline = iganet::UniformBSpline<real_t, 1, 2, 3, 4>;
    iganet::Boundary<BSpline> boundary({5, 4, 7}, iganet::init::greville,
                                       options);
    iganet::Boundary<BSpline> boundary_ref({8, 7, 10}, iganet::init::greville,
                                           options);
    boundary.uniform_refine(1, 0).uniform_refine(2, 1).uniform_refine(1, 2);
    EXPECT_TRUE(boundary.isclose(boundary_ref));
  }

  // parDim == 4

  {
    using BSpline = iganet::UniformBSpline<real_t, 1, 2, 3, 4, 5>;
    iganet::Boundary<BSpline> boundary({5, 4, 7, 8}, iganet::init::greville,
                                       options);
    iganet::Boundary<BSpline> boundary_ref({8, 5, 10, 11},
                                           iganet::init::greville, options);
    boundary.uniform_refine();
    EXPECT_TRUE(boundary.isclose(boundary_ref));
  }

  {
    using BSpline = iganet::UniformBSpline<real_t, 1, 2, 3, 4, 5>;
    iganet::Boundary<BSpline> boundary({5, 4, 7, 8}, iganet::init::greville,
                                       options);
    iganet::Boundary<BSpline> boundary_ref({14, 7, 16, 17},
                                           iganet::init::greville, options);
    boundary.uniform_refine(2);
    EXPECT_TRUE(boundary.isclose(boundary_ref));
  }

  {
    using BSpline = iganet::UniformBSpline<real_t, 1, 2, 3, 4, 5>;
    iganet::Boundary<BSpline> boundary({5, 4, 7, 8}, iganet::init::greville,
                                       options);
    iganet::Boundary<BSpline> boundary_ref({8, 4, 7, 8}, iganet::init::greville,
                                           options);
    boundary.uniform_refine(1, 0);
    EXPECT_TRUE(boundary.isclose(boundary_ref));
  }

  {
    using BSpline = iganet::UniformBSpline<real_t, 1, 2, 3, 4, 5>;
    iganet::Boundary<BSpline> boundary({5, 4, 7, 8}, iganet::init::greville,
                                       options);
    iganet::Boundary<BSpline> boundary_ref({5, 5, 7, 8}, iganet::init::greville,
                                           options);
    boundary.uniform_refine(1, 1);
    EXPECT_TRUE(boundary.isclose(boundary_ref));
  }

  {
    using BSpline = iganet::UniformBSpline<real_t, 1, 2, 3, 4, 5>;
    iganet::Boundary<BSpline> boundary({5, 4, 7, 8}, iganet::init::greville,
                                       options);
    iganet::Boundary<BSpline> boundary_ref({5, 4, 10, 8},
                                           iganet::init::greville, options);
    boundary.uniform_refine(1, 2);
    EXPECT_TRUE(boundary.isclose(boundary_ref));
  }

  {
    using BSpline = iganet::UniformBSpline<real_t, 1, 2, 3, 4, 5>;
    iganet::Boundary<BSpline> boundary({5, 4, 7, 8}, iganet::init::greville,
                                       options);
    iganet::Boundary<BSpline> boundary_ref({5, 4, 7, 11},
                                           iganet::init::greville, options);
    boundary.uniform_refine(1, 3);
    EXPECT_TRUE(boundary.isclose(boundary_ref));
  }

  {
    using BSpline = iganet::UniformBSpline<real_t, 1, 2, 3, 4, 5>;
    iganet::Boundary<BSpline> boundary({5, 4, 7, 8}, iganet::init::greville,
                                       options);
    EXPECT_THROW((boundary.uniform_refine(1, 4)), std::runtime_error);
  }

  {
    using BSpline = iganet::UniformBSpline<real_t, 1, 2, 3, 4, 5>;
    iganet::Boundary<BSpline> boundary({5, 4, 7, 8}, iganet::init::greville,
                                       options);
    iganet::Boundary<BSpline> boundary_ref({8, 7, 10, 17},
                                           iganet::init::greville, options);
    boundary.uniform_refine(1, 0)
        .uniform_refine(2, 1)
        .uniform_refine(1, 2)
        .uniform_refine(2, 3);
    EXPECT_TRUE(boundary.isclose(boundary_ref));
  }
}

TEST_F(BoundaryTest, Boundary_copy_constructor) {
  using BSpline = iganet::UniformBSpline<real_t, 1, 2, 3>;
  iganet::Boundary<BSpline> boundary_orig({5, 4}, iganet::init::greville,
                                          options);
  iganet::Boundary<BSpline> boundary_copy(boundary_orig);

  boundary_orig.side<iganet::side::north>().transform(
      [](const std::array<real_t, 1> xi) {
        return std::array<real_t, 1>{0.0_r};
      });

  EXPECT_TRUE(boundary_orig == boundary_copy);
}

TEST_F(BoundaryTest, Boundary_clone_constructor) {
  using BSpline = iganet::UniformBSpline<real_t, 1, 2, 3>;
  iganet::Boundary<BSpline> boundary_ref({5, 4}, iganet::init::greville,
                                         options);
  iganet::Boundary<BSpline> boundary_orig({5, 4}, iganet::init::greville,
                                          options);
  iganet::Boundary<BSpline> boundary_clone(boundary_orig, true);

  boundary_orig.side<iganet::side::north>().transform(
      [](const std::array<real_t, 1> xi) {
        return std::array<real_t, 1>{0.0_r};
      });

  EXPECT_TRUE(boundary_ref == boundary_clone);
}

TEST_F(BoundaryTest, Boundary_move_constructor) {
  using BSpline = iganet::UniformBSpline<real_t, 1, 2, 3>;
  iganet::Boundary<BSpline> boundary_ref({14, 7}, iganet::init::greville,
                                         options);
  auto boundary(
      iganet::Boundary<BSpline>({5, 4}, iganet::init::greville, options)
          .uniform_refine(2));

  EXPECT_TRUE(boundary.isclose(boundary_ref));
}

TEST_F(BoundaryTest, Boundary_read_write) {
  std::filesystem::path filename =
      std::filesystem::temp_directory_path() / std::to_string(rand());
  using BSpline = iganet::UniformBSpline<real_t, 1, 2, 3>;
  iganet::Boundary<BSpline> boundary_out({5, 4}, iganet::init::greville,
                                         options);
  boundary_out.save(filename.c_str());

  iganet::Boundary<BSpline> boundary_in(options);
  boundary_in.load(filename.c_str());
  std::filesystem::remove(filename);

  EXPECT_TRUE(boundary_in == boundary_out);
  EXPECT_FALSE(boundary_in != boundary_out);
}

TEST_F(BoundaryTest, Boundary_to_from_xml) {
  {
    using BSpline = iganet::UniformBSpline<real_t, 4, 2>;
    iganet::Boundary<BSpline> boundary_out({5}, iganet::init::greville,
                                           options);

    boundary_out.side<iganet::side::east>().transform(
        [](const std::array<real_t, 0>) {
          return std::array<real_t, 4>{static_cast<real_t>(std::rand()),
                                       static_cast<real_t>(std::rand()),
                                       static_cast<real_t>(std::rand()),
                                       static_cast<real_t>(std::rand())};
        });

    boundary_out.side<iganet::side::west>().transform(
        [](const std::array<real_t, 0>) {
          return std::array<real_t, 4>{static_cast<real_t>(std::rand()),
                                       static_cast<real_t>(std::rand()),
                                       static_cast<real_t>(std::rand()),
                                       static_cast<real_t>(std::rand())};
        });

    pugi::xml_document doc = boundary_out.to_xml();

    iganet::Boundary<BSpline> boundary_in(options);
    boundary_in.from_xml(doc);

    EXPECT_TRUE(boundary_in.isclose(boundary_out));

    // non-matching degree
    EXPECT_THROW(
        (iganet::Boundary<iganet::UniformBSpline<real_t, 1, 1>>{}.from_xml(doc,
                                                                           0)),
        std::runtime_error); // XML object provides too many coefficients
    EXPECT_THROW(
        (iganet::Boundary<iganet::UniformBSpline<real_t, 1, 3>>{}.from_xml(doc,
                                                                           0)),
        std::runtime_error); // XML object provides too many coefficients
    EXPECT_THROW(
        (iganet::Boundary<iganet::UniformBSpline<real_t, 1, 4>>{}.from_xml(doc,
                                                                           0)),
        std::runtime_error); // XML object provides too few coefficients

    // non-matching parametric dimension
    EXPECT_THROW(
        (iganet::Boundary<iganet::UniformBSpline<real_t, 1, 2, 1>>{}.from_xml(
            doc, 0)),
        std::runtime_error);
    EXPECT_THROW((iganet::Boundary<iganet::UniformBSpline<real_t, 1, 2, 1, 1>>{}
                      .from_xml(doc, 0)),
                 std::runtime_error);
    EXPECT_THROW(
        (iganet::Boundary<iganet::UniformBSpline<real_t, 1, 2, 1, 1, 1>>{}
             .from_xml(doc, 0)),
        std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW(
        (iganet::Boundary<iganet::UniformBSpline<real_t, 1, 2>>{}.from_xml(doc,
                                                                           0)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::Boundary<iganet::UniformBSpline<real_t, 2, 2>>{}.from_xml(doc,
                                                                           0)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::Boundary<iganet::UniformBSpline<real_t, 3, 2>>{}.from_xml(doc,
                                                                           0)),
        std::runtime_error);

    // non-matching id
    EXPECT_THROW((iganet::Boundary<BSpline>{}.from_xml(doc, 1)),
                 std::runtime_error);
  }

  {
    using BSpline = iganet::UniformBSpline<real_t, 1, 2, 3>;
    iganet::Boundary<BSpline> boundary_out({5, 4}, iganet::init::greville,
                                           options);

    boundary_out.side<iganet::side::east>().transform(
        [](const std::array<real_t, 1> xi) {
          return std::array<real_t, 1>{static_cast<real_t>(std::rand())};
        });

    boundary_out.side<iganet::side::west>().transform(
        [](const std::array<real_t, 1> xi) {
          return std::array<real_t, 1>{static_cast<real_t>(std::rand())};
        });

    boundary_out.side<iganet::side::north>().transform(
        [](const std::array<real_t, 1> xi) {
          return std::array<real_t, 1>{static_cast<real_t>(std::rand())};
        });

    boundary_out.side<iganet::side::south>().transform(
        [](const std::array<real_t, 1> xi) {
          return std::array<real_t, 1>{static_cast<real_t>(std::rand())};
        });

    pugi::xml_document doc = boundary_out.to_xml();

    iganet::Boundary<BSpline> boundary_in(options);
    boundary_in.from_xml(doc);

    EXPECT_TRUE(boundary_in.isclose(boundary_out));

    // non-matching degree
    EXPECT_THROW(
        (iganet::Boundary<iganet::UniformBSpline<real_t, 1, 1, 3>>{}.from_xml(
            doc, 0)),
        std::runtime_error); // XML object provides too many coefficients
    EXPECT_THROW(
        (iganet::Boundary<iganet::UniformBSpline<real_t, 1, 3, 3>>{}.from_xml(
            doc, 0)),
        std::runtime_error); // XML object provides too many coefficients
    EXPECT_THROW(
        (iganet::Boundary<iganet::UniformBSpline<real_t, 1, 2, 2>>{}.from_xml(
            doc, 0)),
        std::runtime_error); // XML object provides too few coefficients
    EXPECT_THROW(
        (iganet::Boundary<iganet::UniformBSpline<real_t, 1, 2, 4>>{}.from_xml(
            doc, 0)),
        std::runtime_error); // XML object provides too few coefficients

    // non-matching parametric dimension
    EXPECT_THROW(
        (iganet::Boundary<iganet::UniformBSpline<real_t, 1, 2>>{}.from_xml(doc,
                                                                           0)),
        std::runtime_error);
    EXPECT_THROW((iganet::Boundary<iganet::UniformBSpline<real_t, 1, 2, 3, 1>>{}
                      .from_xml(doc, 0)),
                 std::runtime_error);
    EXPECT_THROW(
        (iganet::Boundary<iganet::UniformBSpline<real_t, 1, 2, 3, 1, 1>>{}
             .from_xml(doc, 0)),
        std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW(
        (iganet::Boundary<iganet::UniformBSpline<real_t, 2, 2, 3>>{}.from_xml(
            doc, 0)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::Boundary<iganet::UniformBSpline<real_t, 3, 2, 3>>{}.from_xml(
            doc, 0)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::Boundary<iganet::UniformBSpline<real_t, 4, 2, 3>>{}.from_xml(
            doc, 0)),
        std::runtime_error);

    // non-matching id
    EXPECT_THROW((iganet::Boundary<BSpline>{}.from_xml(doc, 1)),
                 std::runtime_error);
  }

  {
    using BSpline = iganet::UniformBSpline<real_t, 3, 2, 3, 2>;
    iganet::Boundary<BSpline> boundary_out({5, 4, 5}, iganet::init::greville,
                                           options);

    boundary_out.side<iganet::side::east>().transform(
        [](const std::array<real_t, 2> xi) {
          return std::array<real_t, 3>{static_cast<real_t>(std::rand()),
                                       static_cast<real_t>(std::rand()),
                                       static_cast<real_t>(std::rand())};
        });

    boundary_out.side<iganet::side::west>().transform(
        [](const std::array<real_t, 2> xi) {
          return std::array<real_t, 3>{static_cast<real_t>(std::rand()),
                                       static_cast<real_t>(std::rand()),
                                       static_cast<real_t>(std::rand())};
        });

    boundary_out.side<iganet::side::north>().transform(
        [](const std::array<real_t, 2> xi) {
          return std::array<real_t, 3>{static_cast<real_t>(std::rand()),
                                       static_cast<real_t>(std::rand()),
                                       static_cast<real_t>(std::rand())};
        });

    boundary_out.side<iganet::side::south>().transform(
        [](const std::array<real_t, 2> xi) {
          return std::array<real_t, 3>{static_cast<real_t>(std::rand()),
                                       static_cast<real_t>(std::rand()),
                                       static_cast<real_t>(std::rand())};
        });

    boundary_out.side<iganet::side::front>().transform(
        [](const std::array<real_t, 2> xi) {
          return std::array<real_t, 3>{static_cast<real_t>(std::rand()),
                                       static_cast<real_t>(std::rand()),
                                       static_cast<real_t>(std::rand())};
        });

    boundary_out.side<iganet::side::back>().transform(
        [](const std::array<real_t, 2> xi) {
          return std::array<real_t, 3>{static_cast<real_t>(std::rand()),
                                       static_cast<real_t>(std::rand()),
                                       static_cast<real_t>(std::rand())};
        });

    pugi::xml_document doc = boundary_out.to_xml();

    iganet::Boundary<BSpline> boundary_in(options);
    boundary_in.from_xml(doc);

    EXPECT_TRUE(boundary_in.isclose(boundary_out));

    // non-matching degree
    EXPECT_THROW((iganet::Boundary<iganet::UniformBSpline<real_t, 3, 2, 4, 2>>{}
                      .from_xml(doc, 0)),
                 std::runtime_error);

    // non-matching parametric dimension
    EXPECT_THROW(
        (iganet::Boundary<iganet::UniformBSpline<real_t, 3, 2>>{}.from_xml(doc,
                                                                           0)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::Boundary<iganet::UniformBSpline<real_t, 3, 2, 3>>{}.from_xml(
            doc, 0)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::Boundary<iganet::UniformBSpline<real_t, 3, 2, 3, 2, 1>>{}
             .from_xml(doc, 0)),
        std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW((iganet::Boundary<iganet::UniformBSpline<real_t, 1, 2, 3, 2>>{}
                      .from_xml(doc, 0)),
                 std::runtime_error);
    EXPECT_THROW((iganet::Boundary<iganet::UniformBSpline<real_t, 2, 2, 3, 2>>{}
                      .from_xml(doc, 0)),
                 std::runtime_error);
    EXPECT_THROW((iganet::Boundary<iganet::UniformBSpline<real_t, 4, 2, 3, 2>>{}
                      .from_xml(doc, 0)),
                 std::runtime_error);

    // non-matching id
    EXPECT_THROW((iganet::Boundary<BSpline>{}.from_xml(doc, 1)),
                 std::runtime_error);
  }

  {
    using BSpline = iganet::UniformBSpline<real_t, 2, 2, 3, 2, 3>;
    iganet::Boundary<BSpline> boundary_out({5, 4, 5, 6}, iganet::init::greville,
                                           options);

    boundary_out.side<iganet::side::east>().transform(
        [](const std::array<real_t, 3> xi) {
          return std::array<real_t, 2>{static_cast<real_t>(std::rand()),
                                       static_cast<real_t>(std::rand())};
        });

    boundary_out.side<iganet::side::west>().transform(
        [](const std::array<real_t, 3> xi) {
          return std::array<real_t, 2>{static_cast<real_t>(std::rand()),
                                       static_cast<real_t>(std::rand())};
        });

    boundary_out.side<iganet::side::north>().transform(
        [](const std::array<real_t, 3> xi) {
          return std::array<real_t, 2>{static_cast<real_t>(std::rand()),
                                       static_cast<real_t>(std::rand())};
        });

    boundary_out.side<iganet::side::south>().transform(
        [](const std::array<real_t, 3> xi) {
          return std::array<real_t, 2>{static_cast<real_t>(std::rand()),
                                       static_cast<real_t>(std::rand())};
        });

    boundary_out.side<iganet::side::front>().transform(
        [](const std::array<real_t, 3> xi) {
          return std::array<real_t, 2>{static_cast<real_t>(std::rand()),
                                       static_cast<real_t>(std::rand())};
        });

    boundary_out.side<iganet::side::back>().transform(
        [](const std::array<real_t, 3> xi) {
          return std::array<real_t, 2>{static_cast<real_t>(std::rand()),
                                       static_cast<real_t>(std::rand())};
        });

    boundary_out.side<iganet::side::stime>().transform(
        [](const std::array<real_t, 3> xi) {
          return std::array<real_t, 2>{static_cast<real_t>(std::rand()),
                                       static_cast<real_t>(std::rand())};
        });

    boundary_out.side<iganet::side::etime>().transform(
        [](const std::array<real_t, 3> xi) {
          return std::array<real_t, 2>{static_cast<real_t>(std::rand()),
                                       static_cast<real_t>(std::rand())};
        });

    pugi::xml_document doc = boundary_out.to_xml();

    iganet::Boundary<BSpline> boundary_in(options);
    boundary_in.from_xml(doc);

    EXPECT_TRUE(boundary_in.isclose(boundary_out));

    // non-matching degree
    EXPECT_THROW(
        (iganet::Boundary<iganet::UniformBSpline<real_t, 2, 2, 4, 2, 3>>{}
             .from_xml(doc, 0)),
        std::runtime_error);

    // non-matching parametric dimension
    EXPECT_THROW(
        (iganet::Boundary<iganet::UniformBSpline<real_t, 2, 2>>{}.from_xml(doc,
                                                                           0)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::Boundary<iganet::UniformBSpline<real_t, 2, 2, 3>>{}.from_xml(
            doc, 0)),
        std::runtime_error);
    EXPECT_THROW((iganet::Boundary<iganet::UniformBSpline<real_t, 2, 2, 3, 2>>{}
                      .from_xml(doc, 0)),
                 std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW(
        (iganet::Boundary<iganet::UniformBSpline<real_t, 1, 2, 3, 2, 3>>{}
             .from_xml(doc, 0)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::Boundary<iganet::UniformBSpline<real_t, 3, 2, 3, 2, 3>>{}
             .from_xml(doc, 0)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::Boundary<iganet::UniformBSpline<real_t, 4, 2, 3, 2, 3>>{}
             .from_xml(doc, 0)),
        std::runtime_error);

    // non-matching id
    EXPECT_THROW((iganet::Boundary<BSpline>{}.from_xml(doc, 1)),
                 std::runtime_error);
  }
}

TEST_F(BoundaryTest, Boundary_load_from_xml) {
  {
    pugi::xml_document doc;
    pugi::xml_parse_result result =
        doc.load_file(IGANET_DATA_DIR "domain1d/line_boundary.xml");

    using BSpline = iganet::UniformBSpline<real_t, 4, 2>;
    iganet::Boundary<BSpline> boundary_in(options);
    boundary_in.from_xml(doc);

    iganet::Boundary<BSpline> boundary_ref({5}, iganet::init::greville,
                                           options);

    boundary_ref.side<iganet::side::east>().transform(
        [](const std::array<real_t, 0>) {
          return std::array<real_t, 4>{
              static_cast<real_t>(1.0), static_cast<real_t>(2.0),
              static_cast<real_t>(3.0), static_cast<real_t>(4.0)};
        });

    boundary_ref.side<iganet::side::west>().transform(
        [](const std::array<real_t, 0>) {
          return std::array<real_t, 4>{
              static_cast<real_t>(-1.0), static_cast<real_t>(-2.0),
              static_cast<real_t>(-3.0), static_cast<real_t>(-4.0)};
        });

    EXPECT_TRUE(boundary_in.isclose(boundary_ref));
  }

  {
    pugi::xml_document doc;
    pugi::xml_parse_result result =
        doc.load_file(IGANET_DATA_DIR "domain2d/square_boundary.xml");

    using BSpline = iganet::UniformBSpline<real_t, 1, 2, 3>;
    iganet::Boundary<BSpline> boundary_in(options);
    boundary_in.from_xml(doc);

    iganet::Boundary<BSpline> boundary_ref({5, 4}, iganet::init::greville,
                                           options);

    boundary_ref.side<iganet::side::east>().transform(
        [](const std::array<real_t, 1> xi) {
          return std::array<real_t, 1>{xi[0]};
        });

    boundary_ref.side<iganet::side::west>().transform(
        [](const std::array<real_t, 1> xi) {
          return std::array<real_t, 1>{xi[0] + 10};
        });

    boundary_ref.side<iganet::side::north>().transform(
        [](const std::array<real_t, 1> xi) {
          return std::array<real_t, 1>{xi[0] + 20};
        });

    boundary_ref.side<iganet::side::south>().transform(
        [](const std::array<real_t, 1> xi) {
          return std::array<real_t, 1>{xi[0] + 30};
        });

    EXPECT_TRUE(boundary_in.isclose(boundary_ref));
  }

  {
    pugi::xml_document doc;
    pugi::xml_parse_result result =
        doc.load_file(IGANET_DATA_DIR "domain3d/cube_boundary.xml");

    using BSpline = iganet::UniformBSpline<real_t, 3, 2, 3, 2>;
    iganet::Boundary<BSpline> boundary_in(options);
    boundary_in.from_xml(doc);

    iganet::Boundary<BSpline> boundary_ref({5, 4, 5}, iganet::init::greville,
                                           options);

    boundary_ref.side<iganet::side::east>().transform(
        [](const std::array<real_t, 2> xi) {
          return std::array<real_t, 3>{xi[0] + xi[1] + 1, xi[0] + xi[1] + 2,
                                       xi[0] + xi[1] + 3};
        });

    boundary_ref.side<iganet::side::west>().transform(
        [](const std::array<real_t, 2> xi) {
          return std::array<real_t, 3>{xi[0] + xi[1] + 11, xi[0] + xi[1] + 12,
                                       xi[0] + xi[1] + 13};
        });

    boundary_ref.side<iganet::side::north>().transform(
        [](const std::array<real_t, 2> xi) {
          return std::array<real_t, 3>{xi[0] + xi[1] + 21, xi[0] + xi[1] + 22,
                                       xi[0] + xi[1] + 23};
        });

    boundary_ref.side<iganet::side::south>().transform(
        [](const std::array<real_t, 2> xi) {
          return std::array<real_t, 3>{xi[0] + xi[1] + 31, xi[0] + xi[1] + 32,
                                       xi[0] + xi[1] + 33};
        });

    boundary_ref.side<iganet::side::front>().transform(
        [](const std::array<real_t, 2> xi) {
          return std::array<real_t, 3>{xi[0] + xi[1] + 41, xi[0] + xi[1] + 42,
                                       xi[0] + xi[1] + 43};
        });

    boundary_ref.side<iganet::side::back>().transform(
        [](const std::array<real_t, 2> xi) {
          return std::array<real_t, 3>{xi[0] + xi[1] + 51, xi[0] + xi[1] + 52,
                                       xi[0] + xi[1] + 53};
        });

    EXPECT_TRUE(boundary_in.isclose(boundary_ref));
  }

  {
    pugi::xml_document doc;
    pugi::xml_parse_result result =
        doc.load_file(IGANET_DATA_DIR "domain4d/hypercube_boundary.xml");

    using BSpline = iganet::UniformBSpline<real_t, 2, 2, 3, 2, 3>;
    iganet::Boundary<BSpline> boundary_in(options);
    boundary_in.from_xml(doc);

    iganet::Boundary<BSpline> boundary_ref({5, 4, 5, 6}, iganet::init::greville,
                                           options);

    boundary_ref.side<iganet::side::east>().transform(
        [](const std::array<real_t, 3> xi) {
          return std::array<real_t, 2>{xi[0] + xi[1] + xi[2] + 1,
                                       xi[0] + xi[1] + xi[2] + 2};
        });

    boundary_ref.side<iganet::side::west>().transform(
        [](const std::array<real_t, 3> xi) {
          return std::array<real_t, 2>{xi[0] + xi[1] + xi[2] + 11,
                                       xi[0] + xi[1] + xi[2] + 12};
        });

    boundary_ref.side<iganet::side::north>().transform(
        [](const std::array<real_t, 3> xi) {
          return std::array<real_t, 2>{xi[0] + xi[1] + xi[2] + 21,
                                       xi[0] + xi[1] + xi[2] + 22};
        });

    boundary_ref.side<iganet::side::south>().transform(
        [](const std::array<real_t, 3> xi) {
          return std::array<real_t, 2>{xi[0] + xi[1] + xi[2] + 31,
                                       xi[0] + xi[1] + xi[2] + 32};
        });

    boundary_ref.side<iganet::side::front>().transform(
        [](const std::array<real_t, 3> xi) {
          return std::array<real_t, 2>{xi[0] + xi[1] + xi[2] + 41,
                                       xi[0] + xi[1] + xi[2] + 42};
        });

    boundary_ref.side<iganet::side::back>().transform(
        [](const std::array<real_t, 3> xi) {
          return std::array<real_t, 2>{xi[0] + xi[1] + xi[2] + 51,
                                       xi[0] + xi[1] + xi[2] + 52};
        });

    boundary_ref.side<iganet::side::stime>().transform(
        [](const std::array<real_t, 3> xi) {
          return std::array<real_t, 2>{xi[0] + xi[1] + xi[2] + 61,
                                       xi[0] + xi[1] + xi[2] + 62};
        });

    boundary_ref.side<iganet::side::etime>().transform(
        [](const std::array<real_t, 3> xi) {
          return std::array<real_t, 2>{xi[0] + xi[1] + xi[2] + 71,
                                       xi[0] + xi[1] + xi[2] + 72};
        });

    EXPECT_TRUE(boundary_in.isclose(boundary_ref));
  }
}

TEST_F(BoundaryTest, Boundary_to_from_json) {
  {
    using BSpline = iganet::UniformBSpline<real_t, 4, 2>;
    iganet::Boundary<BSpline> boundary_out({5}, iganet::init::greville,
                                           options);

    boundary_out.side<iganet::side::east>().transform(
        [](const std::array<real_t, 0>) {
          return std::array<real_t, 4>{static_cast<real_t>(std::rand()),
                                       static_cast<real_t>(std::rand()),
                                       static_cast<real_t>(std::rand()),
                                       static_cast<real_t>(std::rand())};
        });

    boundary_out.side<iganet::side::west>().transform(
        [](const std::array<real_t, 0>) {
          return std::array<real_t, 4>{static_cast<real_t>(std::rand()),
                                       static_cast<real_t>(std::rand()),
                                       static_cast<real_t>(std::rand()),
                                       static_cast<real_t>(std::rand())};
        });

    nlohmann::json json = boundary_out.to_json();

    iganet::Boundary<BSpline> boundary_in(options);
    boundary_in.from_json(json);

    EXPECT_TRUE(boundary_in.isclose(boundary_out));
  }

  {
    using BSpline = iganet::UniformBSpline<real_t, 1, 2, 3>;
    iganet::Boundary<BSpline> boundary_out({5, 4}, iganet::init::greville,
                                           options);

    boundary_out.side<iganet::side::east>().transform(
        [](const std::array<real_t, 1> xi) {
          return std::array<real_t, 1>{static_cast<real_t>(std::rand())};
        });

    boundary_out.side<iganet::side::west>().transform(
        [](const std::array<real_t, 1> xi) {
          return std::array<real_t, 1>{static_cast<real_t>(std::rand())};
        });

    boundary_out.side<iganet::side::north>().transform(
        [](const std::array<real_t, 1> xi) {
          return std::array<real_t, 1>{static_cast<real_t>(std::rand())};
        });

    boundary_out.side<iganet::side::south>().transform(
        [](const std::array<real_t, 1> xi) {
          return std::array<real_t, 1>{static_cast<real_t>(std::rand())};
        });

    nlohmann::json json = boundary_out.to_json();

    iganet::Boundary<BSpline> boundary_in(options);
    boundary_in.from_json(json);

    EXPECT_TRUE(boundary_in.isclose(boundary_out));
  }

  {
    using BSpline = iganet::UniformBSpline<real_t, 3, 2, 3, 2>;
    iganet::Boundary<BSpline> boundary_out({5, 4, 5}, iganet::init::greville,
                                           options);

    boundary_out.side<iganet::side::east>().transform(
        [](const std::array<real_t, 2> xi) {
          return std::array<real_t, 3>{static_cast<real_t>(std::rand()),
                                       static_cast<real_t>(std::rand()),
                                       static_cast<real_t>(std::rand())};
        });

    boundary_out.side<iganet::side::west>().transform(
        [](const std::array<real_t, 2> xi) {
          return std::array<real_t, 3>{static_cast<real_t>(std::rand()),
                                       static_cast<real_t>(std::rand()),
                                       static_cast<real_t>(std::rand())};
        });

    boundary_out.side<iganet::side::north>().transform(
        [](const std::array<real_t, 2> xi) {
          return std::array<real_t, 3>{static_cast<real_t>(std::rand()),
                                       static_cast<real_t>(std::rand()),
                                       static_cast<real_t>(std::rand())};
        });

    boundary_out.side<iganet::side::south>().transform(
        [](const std::array<real_t, 2> xi) {
          return std::array<real_t, 3>{static_cast<real_t>(std::rand()),
                                       static_cast<real_t>(std::rand()),
                                       static_cast<real_t>(std::rand())};
        });

    boundary_out.side<iganet::side::front>().transform(
        [](const std::array<real_t, 2> xi) {
          return std::array<real_t, 3>{static_cast<real_t>(std::rand()),
                                       static_cast<real_t>(std::rand()),
                                       static_cast<real_t>(std::rand())};
        });

    boundary_out.side<iganet::side::back>().transform(
        [](const std::array<real_t, 2> xi) {
          return std::array<real_t, 3>{static_cast<real_t>(std::rand()),
                                       static_cast<real_t>(std::rand()),
                                       static_cast<real_t>(std::rand())};
        });

    nlohmann::json json = boundary_out.to_json();

    iganet::Boundary<BSpline> boundary_in(options);
    boundary_in.from_json(json);

    EXPECT_TRUE(boundary_in.isclose(boundary_out));
  }

  {
    using BSpline = iganet::UniformBSpline<real_t, 2, 2, 3, 2, 3>;
    iganet::Boundary<BSpline> boundary_out({5, 4, 5, 6}, iganet::init::greville,
                                           options);

    boundary_out.side<iganet::side::east>().transform(
        [](const std::array<real_t, 3> xi) {
          return std::array<real_t, 2>{static_cast<real_t>(std::rand()),
                                       static_cast<real_t>(std::rand())};
        });

    boundary_out.side<iganet::side::west>().transform(
        [](const std::array<real_t, 3> xi) {
          return std::array<real_t, 2>{static_cast<real_t>(std::rand()),
                                       static_cast<real_t>(std::rand())};
        });

    boundary_out.side<iganet::side::north>().transform(
        [](const std::array<real_t, 3> xi) {
          return std::array<real_t, 2>{static_cast<real_t>(std::rand()),
                                       static_cast<real_t>(std::rand())};
        });

    boundary_out.side<iganet::side::south>().transform(
        [](const std::array<real_t, 3> xi) {
          return std::array<real_t, 2>{static_cast<real_t>(std::rand()),
                                       static_cast<real_t>(std::rand())};
        });

    boundary_out.side<iganet::side::front>().transform(
        [](const std::array<real_t, 3> xi) {
          return std::array<real_t, 2>{static_cast<real_t>(std::rand()),
                                       static_cast<real_t>(std::rand())};
        });

    boundary_out.side<iganet::side::back>().transform(
        [](const std::array<real_t, 3> xi) {
          return std::array<real_t, 2>{static_cast<real_t>(std::rand()),
                                       static_cast<real_t>(std::rand())};
        });

    boundary_out.side<iganet::side::stime>().transform(
        [](const std::array<real_t, 3> xi) {
          return std::array<real_t, 2>{static_cast<real_t>(std::rand()),
                                       static_cast<real_t>(std::rand())};
        });

    boundary_out.side<iganet::side::etime>().transform(
        [](const std::array<real_t, 3> xi) {
          return std::array<real_t, 2>{static_cast<real_t>(std::rand()),
                                       static_cast<real_t>(std::rand())};
        });

    nlohmann::json json = boundary_out.to_json();

    iganet::Boundary<BSpline> boundary_in(options);
    boundary_in.from_json(json);

    EXPECT_TRUE(boundary_in.isclose(boundary_out));
  }
}

TEST_F(BoundaryTest, Boundary_query_property) {
  using BSpline = iganet::UniformBSpline<real_t, 1, 2, 3, 4>;
  iganet::Boundary<BSpline> boundary({5, 4, 7}, iganet::init::greville,
                                     options);

  EXPECT_TRUE(
      std::apply([this](auto... is_uniform) { return (is_uniform && ...); },
                 boundary.is_uniform()));
  EXPECT_FALSE(std::apply(
      [this](auto... is_nonuniform) { return (is_nonuniform || ...); },
      boundary.is_nonuniform()));

  EXPECT_TRUE(std::apply(
      [this](auto... device) { return ((device == options.device()) && ...); },
      boundary.device()));
  EXPECT_TRUE(std::apply(
      [this](auto... device_index) {
        return ((device_index == options.device_index()) && ...);
      },
      boundary.device_index()));
  EXPECT_TRUE(std::apply(
      [this](auto... dtype) { return ((dtype == options.dtype()) && ...); },
      boundary.dtype()));
  EXPECT_TRUE(std::apply(
      [this](auto... is_sparse) {
        return ((is_sparse == options.is_sparse()) && ...);
      },
      boundary.is_sparse()));
  EXPECT_TRUE(std::apply(
      [this](auto... layout) { return ((layout == options.layout()) && ...); },
      boundary.layout()));
  EXPECT_TRUE(std::apply(
      [this](auto... pinned_memory) {
        return ((pinned_memory == options.pinned_memory()) && ...);
      },
      boundary.pinned_memory()));
}

template <typename Values, typename Xi, std::size_t... Is>
inline void
check_requires_grad(std::index_sequence<Is...>, const Values &values,
                    const Xi &xi,
                    const iganet::Options<iganet::unittests::real_t> &options) {
  auto check = [&options](const auto &values, const auto &xi) {
    values[0]->operator[](0).backward();
    EXPECT_TRUE(torch::allclose(xi[0].grad(),
                                iganet::utils::to_tensor({1.0_r}, options)));
  };

  (check(std::get<Is>(values), std::get<Is>(xi)), ...);
}

template <typename Values, typename Xi, std::size_t... Is>
inline void check_requires_grad_throw(
    std::index_sequence<Is...>, const Values &values, const Xi &xi,
    const iganet::Options<iganet::unittests::real_t> &options) {
  auto check = [&options](const auto &values, const auto &xi) {
    values[0]->operator[](0).backward(
        {}, true); // otherwise we cannot run backward() a second time
    EXPECT_THROW(torch::allclose(xi[0].grad(), torch::empty({})), c10::Error);
  };

  (check(std::get<Is>(values), std::get<Is>(xi)), ...);
}

TEST_F(BoundaryTest, Boundary_requires_grad) {
  {
    using iganet::side;
    using BSpline = iganet::UniformBSpline<real_t, 1, 2, 3, 4>;
    iganet::Boundary<BSpline> boundary({5, 4, 7}, iganet::init::greville,
                                       options);

    EXPECT_FALSE(
        std::apply([](auto... requires_grad) { return (requires_grad || ...); },
                   boundary.requires_grad()));

    auto xi = std::tuple{
        iganet::utils::to_tensorArray(options, {0.5_r} /* v */, {0.5_r} /* w */
                                      ) /* west  */,
        iganet::utils::to_tensorArray(options, {0.5_r} /* v */, {0.5_r} /* w */
                                      ) /* east  */,
        iganet::utils::to_tensorArray(options, {0.5_r} /* u */, {0.5_r} /* w */
                                      ) /* south */,
        iganet::utils::to_tensorArray(options, {0.5_r} /* u */, {0.5_r} /* w */
                                      ) /* north */,
        iganet::utils::to_tensorArray(options, {0.5_r} /* u */, {0.5_r} /* v */
                                      ) /* front */,
        iganet::utils::to_tensorArray(options, {0.5_r} /* u */, {0.5_r} /* v */
                                      ) /* back  */};
    auto values = boundary.eval(xi);

    // We expect an error when calling backward() because no tensor
    // has requires_grad = true
    std::apply(
        [](const auto &...values) {
          auto check = [](const auto &values) {
            EXPECT_THROW(values[0]->backward(), c10::Error);
          };
          (check(values), ...);
        },
        values);

    xi = std::tuple{iganet::utils::to_tensorArray(options.requires_grad(true),
                                                  {0.5_r} /* v */,
                                                  {0.5_r} /* w */) /* west  */,
                    iganet::utils::to_tensorArray(options.requires_grad(true),
                                                  {0.5_r} /* v */,
                                                  {0.5_r} /* w */) /* east  */,
                    iganet::utils::to_tensorArray(options.requires_grad(true),
                                                  {0.5_r} /* u */,
                                                  {0.5_r} /* w */) /* south */,
                    iganet::utils::to_tensorArray(options.requires_grad(true),
                                                  {0.5_r} /* u */,
                                                  {0.5_r} /* w */) /* north */,
                    iganet::utils::to_tensorArray(options.requires_grad(true),
                                                  {0.5_r} /* u */,
                                                  {0.5_r} /* v */) /* front */,
                    iganet::utils::to_tensorArray(options.requires_grad(true),
                                                  {0.5_r} /* u */,
                                                  {0.5_r} /* v */) /* back  */};
    values = boundary.eval(xi);

    // Note that this check cannot be implemented using std::apply as
    // this functions only accepts one tuple as argument whereas here
    // both values and xi need to be passed
    check_requires_grad(
        std::make_index_sequence<iganet::Boundary<BSpline>::nsides()>{}, values,
        xi, options);
  }

  {
    using iganet::side;
    using BSpline = iganet::UniformBSpline<real_t, 1, 2, 3, 4>;
    iganet::Boundary<BSpline> boundary({5, 4, 7}, iganet::init::greville,
                                       options.requires_grad(true));

    EXPECT_TRUE(
        std::apply([](auto... requires_grad) { return (requires_grad && ...); },
                   boundary.requires_grad()));

    auto xi =
        std::tuple{iganet::utils::to_tensorArray(options, {0.5_r} /* v */,
                                                 {0.5_r} /* w */) /* west  */,
                   iganet::utils::to_tensorArray(options, {0.5_r} /* v */,
                                                 {0.5_r} /* w */) /* east  */,
                   iganet::utils::to_tensorArray(options, {0.5_r} /* u */,
                                                 {0.5_r} /* w */) /* south */,
                   iganet::utils::to_tensorArray(options, {0.5_r} /* u */,
                                                 {0.5_r} /* w */) /* north */,
                   iganet::utils::to_tensorArray(options, {0.5_r} /* u */,
                                                 {0.5_r} /* v */) /* front */,
                   iganet::utils::to_tensorArray(options, {0.5_r} /* u */,
                                                 {0.5_r} /* v */) /* back  */};
    auto values = boundary.eval(xi);

    // We expect an error because xi[0].grad() is an undefined tensor
    //
    // Note that this check cannot be implemented using std::apply as
    // this functions only accepts one tuple as argument whereas here
    // both values and xi need to be passed
    check_requires_grad_throw(
        std::make_index_sequence<iganet::Boundary<BSpline>::nsides()>{}, values,
        xi, options);

    xi = std::tuple{iganet::utils::to_tensorArray(options.requires_grad(true),
                                                  {0.5_r} /* v */,
                                                  {0.5_r} /* w */) /* west  */,
                    iganet::utils::to_tensorArray(options.requires_grad(true),
                                                  {0.5_r} /* v */,
                                                  {0.5_r} /* w */) /* east  */,
                    iganet::utils::to_tensorArray(options.requires_grad(true),
                                                  {0.5_r} /* u */,
                                                  {0.5_r} /* w */) /* south */,
                    iganet::utils::to_tensorArray(options.requires_grad(true),
                                                  {0.5_r} /* u */,
                                                  {0.5_r} /* w */) /* north */,
                    iganet::utils::to_tensorArray(options.requires_grad(true),
                                                  {0.5_r} /* u */,
                                                  {0.5_r} /* v */) /* front */,
                    iganet::utils::to_tensorArray(options.requires_grad(true),
                                                  {0.5_r} /* u */,
                                                  {0.5_r} /* v */) /* back  */};
    values = boundary.eval(xi);

    // Note that this check cannot be implemented using std::apply as
    // this functions only accepts one tuple as argument whereas here
    // both values and xi need to be passed
    check_requires_grad(
        std::make_index_sequence<iganet::Boundary<BSpline>::nsides()>{}, values,
        xi, options);
  }
}

TEST_F(BoundaryTest, Boundary_to_dtype) {
  {
    using iganet::side;
    using BSpline = iganet::UniformBSpline<real_t, 1, 2, 3, 4>;
    iganet::Boundary<BSpline> boundary({5, 4, 7}, iganet::init::greville,
                                       options);

    auto boundary_double = boundary.to<double>();
    auto boundary_float = boundary.to<float>();

    if constexpr (std::is_same<real_t, double>::value)
      EXPECT_TRUE(boundary == boundary_double);
    else
      EXPECT_TRUE(boundary != boundary_double);

    if constexpr (std::is_same<real_t, float>::value)
      EXPECT_TRUE(boundary == boundary_float);
    else
      EXPECT_TRUE(boundary != boundary_float);
  }

  {
    using iganet::side;
    using BSpline = iganet::UniformBSpline<real_t, 1, 2, 3, 4>;
    iganet::Boundary<BSpline> boundary({5, 4, 7}, iganet::init::greville,
                                       options);

    auto boundary_double = boundary.to(iganet::Options<double>{});
    auto boundary_float = boundary.to(iganet::Options<float>{});

    if constexpr (std::is_same<real_t, double>::value)
      EXPECT_TRUE(boundary == boundary_double);
    else
      EXPECT_TRUE(boundary != boundary_double);

    if constexpr (std::is_same<real_t, float>::value)
      EXPECT_TRUE(boundary == boundary_float);
    else
      EXPECT_TRUE(boundary != boundary_float);
  }
}

TEST_F(BoundaryTest, Boundary_to_device) {
  {
    using iganet::side;
    using BSpline = iganet::UniformBSpline<real_t, 1, 2, 3, 4>;
    iganet::Options<real_t> options =
        iganet::Options<real_t>{}.device(torch::kCPU);
    iganet::Boundary<BSpline> boundary({5, 4, 7}, iganet::init::greville,
                                       options);

    auto boundary_cpu = boundary.to(torch::kCPU);
    EXPECT_TRUE(boundary == boundary_cpu);

    if (torch::cuda::is_available()) {
      auto boundary_cuda = boundary.to(torch::kCUDA);
      EXPECT_THROW((void)(boundary == boundary_cuda), c10::Error);
    } else
      EXPECT_THROW(boundary.to(torch::kCUDA), c10::Error);

    if (at::hasHIP()) {
      auto boundary_hip = boundary.to(torch::kHIP);
      EXPECT_THROW((void)(boundary == boundary_hip), c10::Error);
    } else
      EXPECT_THROW(boundary.to(torch::kHIP), c10::Error);

    if (at::hasMPS() && // will become torch::mps::is_available()
        (options.dtype() != iganet::dtype<double>())) {
      auto boundary_mps = boundary.to(torch::kMPS);
      EXPECT_THROW((void)(boundary == boundary_mps), c10::Error);
    } else
      EXPECT_THROW(boundary.to(torch::kMPS), c10::Error);
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  iganet::init();
  return RUN_ALL_TESTS();
}
