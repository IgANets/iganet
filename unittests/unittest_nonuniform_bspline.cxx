/**
   @file unittests/unittest_nonuniform_bspline.cxx

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
#include <unittest_config.hpp>

#pragma nv_diag_suppress 20208

using namespace iganet::unittests::literals;

class BSplineTest : public ::testing::Test {
public:
  BSplineTest() { std::srand(std::time(nullptr)); }

protected:
  using real_t = iganet::unittests::real_t;
  iganet::Options<real_t> options;
};

TEST_F(BSplineTest, NonUniformBSpline_parDim1_geoDim1_degrees1) {
  EXPECT_THROW(
      (iganet::NonUniformBSpline<real_t, 1, 1>({{{0.0_r, 0.0_r, 1.0_r}}})),
      std::runtime_error);
  iganet::NonUniformBSpline<real_t, 1, 1> bspline(
      {{{0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r}}});
  EXPECT_EQ(bspline.parDim(), 1);
  EXPECT_EQ(bspline.geoDim(), 1);
  EXPECT_EQ(bspline.degree(0), 1);
  EXPECT_EQ(bspline.nknots(0), 5);
  EXPECT_EQ(bspline.ncoeffs(0), 3);
  EXPECT_EQ(bspline.ncumcoeffs(), 3);
  EXPECT_TRUE(bspline.is_nonuniform());
  EXPECT_FALSE(bspline.is_uniform());
}

TEST_F(BSplineTest, NonUniformBSpline_parDim1_geoDim2_degrees1) {
  EXPECT_THROW(
      (iganet::NonUniformBSpline<real_t, 2, 1>({{{0.0_r, 0.0_r, 1.0_r}}})),
      std::runtime_error);
  iganet::NonUniformBSpline<real_t, 2, 1> bspline(
      {{{0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r}}});
  EXPECT_EQ(bspline.parDim(), 1);
  EXPECT_EQ(bspline.geoDim(), 2);
  EXPECT_EQ(bspline.degree(0), 1);
  EXPECT_EQ(bspline.nknots(0), 5);
  EXPECT_EQ(bspline.ncoeffs(0), 3);
  EXPECT_EQ(bspline.ncumcoeffs(), 3);
}

TEST_F(BSplineTest, NonUniformBSpline_parDim1_geoDim3_degrees1) {
  EXPECT_THROW(
      (iganet::NonUniformBSpline<real_t, 3, 1>({{{0.0_r, 0.0_r, 1.0_r}}})),
      std::runtime_error);
  iganet::NonUniformBSpline<real_t, 3, 1> bspline(
      {{{0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r}}});
  EXPECT_EQ(bspline.parDim(), 1);
  EXPECT_EQ(bspline.geoDim(), 3);
  EXPECT_EQ(bspline.degree(0), 1);
  EXPECT_EQ(bspline.nknots(0), 5);
  EXPECT_EQ(bspline.ncoeffs(0), 3);
  EXPECT_EQ(bspline.ncumcoeffs(), 3);
}

TEST_F(BSplineTest, NonUniformBSpline_parDim1_geoDim4_degrees1) {
  EXPECT_THROW(
      (iganet::NonUniformBSpline<real_t, 4, 1>({{{0.0_r, 0.0_r, 1.0_r}}})),
      std::runtime_error);
  iganet::NonUniformBSpline<real_t, 4, 1> bspline(
      {{{0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r}}});
  EXPECT_EQ(bspline.parDim(), 1);
  EXPECT_EQ(bspline.geoDim(), 4);
  EXPECT_EQ(bspline.degree(0), 1);
  EXPECT_EQ(bspline.nknots(0), 5);
  EXPECT_EQ(bspline.ncoeffs(0), 3);
  EXPECT_EQ(bspline.ncumcoeffs(), 3);
}

TEST_F(BSplineTest, NonUniformBSpline_parDim2_geoDim1_degrees12) {
  iganet::NonUniformBSpline<real_t, 1, 1, 2> bspline(
      {{{0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 1.0_r, 1.0_r, 1.0_r}}});
  EXPECT_EQ(bspline.parDim(), 2);
  EXPECT_EQ(bspline.geoDim(), 1);
  EXPECT_EQ(bspline.degree(0), 1);
  EXPECT_EQ(bspline.degree(1), 2);
  EXPECT_EQ(bspline.nknots(0), 5);
  EXPECT_EQ(bspline.nknots(1), 6);
  EXPECT_EQ(bspline.ncoeffs(0), 3);
  EXPECT_EQ(bspline.ncoeffs(1), 3);
  EXPECT_EQ(bspline.ncumcoeffs(), 9);
}

TEST_F(BSplineTest, NonUniformBSpline_parDim2_geoDim2_degrees12) {
  iganet::NonUniformBSpline<real_t, 2, 1, 2> bspline(
      {{{0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 1.0_r, 1.0_r, 1.0_r}}});
  EXPECT_EQ(bspline.parDim(), 2);
  EXPECT_EQ(bspline.geoDim(), 2);
  EXPECT_EQ(bspline.degree(0), 1);
  EXPECT_EQ(bspline.degree(1), 2);
  EXPECT_EQ(bspline.nknots(0), 5);
  EXPECT_EQ(bspline.nknots(1), 6);
  EXPECT_EQ(bspline.ncoeffs(0), 3);
  EXPECT_EQ(bspline.ncoeffs(1), 3);
  EXPECT_EQ(bspline.ncumcoeffs(), 9);
}

TEST_F(BSplineTest, NonUniformBSpline_parDim2_geoDim3_degrees12) {
  iganet::NonUniformBSpline<real_t, 3, 1, 2> bspline(
      {{{0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 1.0_r, 1.0_r, 1.0_r}}});
  EXPECT_EQ(bspline.parDim(), 2);
  EXPECT_EQ(bspline.geoDim(), 3);
  EXPECT_EQ(bspline.degree(0), 1);
  EXPECT_EQ(bspline.degree(1), 2);
  EXPECT_EQ(bspline.nknots(0), 5);
  EXPECT_EQ(bspline.nknots(1), 6);
  EXPECT_EQ(bspline.ncoeffs(0), 3);
  EXPECT_EQ(bspline.ncoeffs(1), 3);
  EXPECT_EQ(bspline.ncumcoeffs(), 9);
}

TEST_F(BSplineTest, NonUniformBSpline_parDim2_geoDim4_degrees12) {
  iganet::NonUniformBSpline<real_t, 4, 1, 2> bspline(
      {{{0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 1.0_r, 1.0_r, 1.0_r}}});
  EXPECT_EQ(bspline.parDim(), 2);
  EXPECT_EQ(bspline.geoDim(), 4);
  EXPECT_EQ(bspline.degree(0), 1);
  EXPECT_EQ(bspline.degree(1), 2);
  EXPECT_EQ(bspline.nknots(0), 5);
  EXPECT_EQ(bspline.nknots(1), 6);
  EXPECT_EQ(bspline.ncoeffs(0), 3);
  EXPECT_EQ(bspline.ncoeffs(1), 3);
  EXPECT_EQ(bspline.ncumcoeffs(), 9);
}

TEST_F(BSplineTest, NonUniformBSpline_parDim3_geoDim1_degrees123) {
  iganet::NonUniformBSpline<real_t, 1, 1, 2, 3> bspline(
      {{{0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r}}});
  EXPECT_EQ(bspline.parDim(), 3);
  EXPECT_EQ(bspline.geoDim(), 1);
  EXPECT_EQ(bspline.degree(0), 1);
  EXPECT_EQ(bspline.degree(1), 2);
  EXPECT_EQ(bspline.degree(2), 3);
  EXPECT_EQ(bspline.nknots(0), 5);
  EXPECT_EQ(bspline.nknots(1), 6);
  EXPECT_EQ(bspline.nknots(2), 9);
  EXPECT_EQ(bspline.ncoeffs(0), 3);
  EXPECT_EQ(bspline.ncoeffs(1), 3);
  EXPECT_EQ(bspline.ncoeffs(2), 5);
  EXPECT_EQ(bspline.ncumcoeffs(), 45);
}

TEST_F(BSplineTest, NonUniformBSpline_parDim3_geoDim2_degrees123) {
  iganet::NonUniformBSpline<real_t, 2, 1, 2, 3> bspline(
      {{{0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r}}});
  EXPECT_EQ(bspline.parDim(), 3);
  EXPECT_EQ(bspline.geoDim(), 2);
  EXPECT_EQ(bspline.degree(0), 1);
  EXPECT_EQ(bspline.degree(1), 2);
  EXPECT_EQ(bspline.degree(2), 3);
  EXPECT_EQ(bspline.nknots(0), 5);
  EXPECT_EQ(bspline.nknots(1), 6);
  EXPECT_EQ(bspline.nknots(2), 9);
  EXPECT_EQ(bspline.ncoeffs(0), 3);
  EXPECT_EQ(bspline.ncoeffs(1), 3);
  EXPECT_EQ(bspline.ncoeffs(2), 5);
  EXPECT_EQ(bspline.ncumcoeffs(), 45);
}

TEST_F(BSplineTest, NonUniformBSpline_parDim3_geoDim3_degrees123) {
  iganet::NonUniformBSpline<real_t, 3, 1, 2, 3> bspline(
      {{{0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r}}});
  EXPECT_EQ(bspline.parDim(), 3);
  EXPECT_EQ(bspline.geoDim(), 3);
  EXPECT_EQ(bspline.degree(0), 1);
  EXPECT_EQ(bspline.degree(1), 2);
  EXPECT_EQ(bspline.degree(2), 3);
  EXPECT_EQ(bspline.nknots(0), 5);
  EXPECT_EQ(bspline.nknots(1), 6);
  EXPECT_EQ(bspline.nknots(2), 9);
  EXPECT_EQ(bspline.ncoeffs(0), 3);
  EXPECT_EQ(bspline.ncoeffs(1), 3);
  EXPECT_EQ(bspline.ncoeffs(2), 5);
  EXPECT_EQ(bspline.ncumcoeffs(), 45);
}

TEST_F(BSplineTest, NonUniformBSpline_parDim3_geoDim4_degrees123) {
  iganet::NonUniformBSpline<real_t, 4, 1, 2, 3> bspline(
      {{{0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r}}});
  EXPECT_EQ(bspline.parDim(), 3);
  EXPECT_EQ(bspline.geoDim(), 4);
  EXPECT_EQ(bspline.degree(0), 1);
  EXPECT_EQ(bspline.degree(1), 2);
  EXPECT_EQ(bspline.degree(2), 3);
  EXPECT_EQ(bspline.nknots(0), 5);
  EXPECT_EQ(bspline.nknots(1), 6);
  EXPECT_EQ(bspline.nknots(2), 9);
  EXPECT_EQ(bspline.ncoeffs(0), 3);
  EXPECT_EQ(bspline.ncoeffs(1), 3);
  EXPECT_EQ(bspline.ncoeffs(2), 5);
  EXPECT_EQ(bspline.ncumcoeffs(), 45);
}

TEST_F(BSplineTest, NonUniformBSpline_parDim4_geoDim1_degrees1234) {
  iganet::NonUniformBSpline<real_t, 1, 1, 2, 3, 4> bspline(
      {{{0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r,
         1.0_r}}});
  EXPECT_EQ(bspline.parDim(), 4);
  EXPECT_EQ(bspline.geoDim(), 1);
  EXPECT_EQ(bspline.degree(0), 1);
  EXPECT_EQ(bspline.degree(1), 2);
  EXPECT_EQ(bspline.degree(2), 3);
  EXPECT_EQ(bspline.degree(3), 4);
  EXPECT_EQ(bspline.nknots(0), 5);
  EXPECT_EQ(bspline.nknots(1), 6);
  EXPECT_EQ(bspline.nknots(2), 9);
  EXPECT_EQ(bspline.nknots(3), 11);
  EXPECT_EQ(bspline.ncoeffs(0), 3);
  EXPECT_EQ(bspline.ncoeffs(1), 3);
  EXPECT_EQ(bspline.ncoeffs(2), 5);
  EXPECT_EQ(bspline.ncoeffs(3), 6);
  EXPECT_EQ(bspline.ncumcoeffs(), 270);
}

TEST_F(BSplineTest, NonUniformBSpline_parDim4_geoDim2_degrees1234) {
  iganet::NonUniformBSpline<real_t, 2, 1, 2, 3, 4> bspline(
      {{{0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r,
         1.0_r}}});
  EXPECT_EQ(bspline.parDim(), 4);
  EXPECT_EQ(bspline.geoDim(), 2);
  EXPECT_EQ(bspline.degree(0), 1);
  EXPECT_EQ(bspline.degree(1), 2);
  EXPECT_EQ(bspline.degree(2), 3);
  EXPECT_EQ(bspline.degree(3), 4);
  EXPECT_EQ(bspline.nknots(0), 5);
  EXPECT_EQ(bspline.nknots(1), 6);
  EXPECT_EQ(bspline.nknots(2), 9);
  EXPECT_EQ(bspline.nknots(3), 11);
  EXPECT_EQ(bspline.ncoeffs(0), 3);
  EXPECT_EQ(bspline.ncoeffs(1), 3);
  EXPECT_EQ(bspline.ncoeffs(2), 5);
  EXPECT_EQ(bspline.ncoeffs(3), 6);
  EXPECT_EQ(bspline.ncumcoeffs(), 270);
}

TEST_F(BSplineTest, NonUniformBSpline_parDim4_geoDim3_degrees1234) {
  iganet::NonUniformBSpline<real_t, 3, 1, 2, 3, 4> bspline(
      {{{0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r,
         1.0_r}}});
  EXPECT_EQ(bspline.parDim(), 4);
  EXPECT_EQ(bspline.geoDim(), 3);
  EXPECT_EQ(bspline.degree(0), 1);
  EXPECT_EQ(bspline.degree(1), 2);
  EXPECT_EQ(bspline.degree(2), 3);
  EXPECT_EQ(bspline.degree(3), 4);
  EXPECT_EQ(bspline.nknots(0), 5);
  EXPECT_EQ(bspline.nknots(1), 6);
  EXPECT_EQ(bspline.nknots(2), 9);
  EXPECT_EQ(bspline.nknots(3), 11);
  EXPECT_EQ(bspline.ncoeffs(0), 3);
  EXPECT_EQ(bspline.ncoeffs(1), 3);
  EXPECT_EQ(bspline.ncoeffs(2), 5);
  EXPECT_EQ(bspline.ncoeffs(3), 6);
  EXPECT_EQ(bspline.ncumcoeffs(), 270);
}

TEST_F(BSplineTest, NonUniformBSpline_parDim4_geoDim4_degrees1234) {
  iganet::NonUniformBSpline<real_t, 4, 1, 2, 3, 4> bspline(
      {{{0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r,
         1.0_r}}});
  EXPECT_EQ(bspline.parDim(), 4);
  EXPECT_EQ(bspline.geoDim(), 4);
  EXPECT_EQ(bspline.degree(0), 1);
  EXPECT_EQ(bspline.degree(1), 2);
  EXPECT_EQ(bspline.degree(2), 3);
  EXPECT_EQ(bspline.degree(3), 4);
  EXPECT_EQ(bspline.nknots(0), 5);
  EXPECT_EQ(bspline.nknots(1), 6);
  EXPECT_EQ(bspline.nknots(2), 9);
  EXPECT_EQ(bspline.nknots(3), 11);
  EXPECT_EQ(bspline.ncoeffs(0), 3);
  EXPECT_EQ(bspline.ncoeffs(1), 3);
  EXPECT_EQ(bspline.ncoeffs(2), 5);
  EXPECT_EQ(bspline.ncoeffs(3), 6);
  EXPECT_EQ(bspline.ncumcoeffs(), 270);
}

TEST_F(BSplineTest, NonUniformBSpline_init) {
  {
    iganet::NonUniformBSpline<real_t, 1, 1> bspline(
        {{{0.0_r, 0.0_r, 0.25_r, 0.5_r, 0.75_r, 1.0_r, 1.0_r}}},
        iganet::init::zeros, options);
    EXPECT_TRUE(torch::equal(bspline.coeffs(0), torch::zeros(5, options)));
  }

  {
    iganet::NonUniformBSpline<real_t, 1, 1> bspline(
        {{{0.0_r, 0.0_r, 0.25_r, 0.5_r, 0.75_r, 1.0_r, 1.0_r}}},
        iganet::init::ones, options);
    EXPECT_TRUE(torch::equal(bspline.coeffs(0), torch::ones(5, options)));
  }

  {
    iganet::NonUniformBSpline<real_t, 1, 1> bspline(
        {{{0.0_r, 0.0_r, 0.25_r, 0.5_r, 0.75_r, 1.0_r, 1.0_r}}},
        iganet::init::linear, options);
    EXPECT_TRUE(
        torch::equal(bspline.coeffs(0), torch::linspace(0, 1, 5, options)));
  }

  {
    iganet::NonUniformBSpline<real_t, 1, 1> bspline(
        {{{0.0_r, 0.0_r, 0.25_r, 0.5_r, 0.75_r, 1.0_r, 1.0_r}}},
        iganet::init::greville, options);
    EXPECT_TRUE(
        torch::equal(bspline.coeffs(0), torch::linspace(0, 1, 5, options)));
  }

  {
    iganet::NonUniformBSpline<real_t, 2, 1> bspline(
        {{{0.0_r, 0.0_r, 0.25_r, 0.5_r, 0.75_r, 1.0_r, 1.0_r}}},
        iganet::init::zeros, options);
    EXPECT_TRUE(torch::equal(bspline.coeffs(0), torch::zeros(5, options)));
    EXPECT_TRUE(torch::equal(bspline.coeffs(1), torch::zeros(5, options)));
  }

  {
    iganet::NonUniformBSpline<real_t, 2, 1> bspline(
        {{{0.0_r, 0.0_r, 0.25_r, 0.5_r, 0.75_r, 1.0_r, 1.0_r}}},
        iganet::init::ones, options);
    EXPECT_TRUE(torch::equal(bspline.coeffs(0), torch::ones(5, options)));
    EXPECT_TRUE(torch::equal(bspline.coeffs(1), torch::ones(5, options)));
  }

  {
    iganet::NonUniformBSpline<real_t, 2, 1> bspline(
        {{{0.0_r, 0.0_r, 0.25_r, 0.5_r, 0.75_r, 1.0_r, 1.0_r}}},
        iganet::init::linear, options);
    EXPECT_TRUE(
        torch::equal(bspline.coeffs(0), torch::linspace(0, 1, 5, options)));
    EXPECT_TRUE(torch::equal(bspline.coeffs(1), torch::ones(5, options)));
  }

  {
    iganet::NonUniformBSpline<real_t, 2, 1> bspline(
        {{{0.0_r, 0.0_r, 0.25_r, 0.5_r, 0.75_r, 1.0_r, 1.0_r}}},
        iganet::init::greville, options);
    EXPECT_TRUE(
        torch::equal(bspline.coeffs(0), torch::linspace(0, 1, 5, options)));
    EXPECT_TRUE(torch::equal(bspline.coeffs(1), torch::ones(5, options)));
  }

  {
    iganet::NonUniformBSpline<real_t, 2, 2, 2> bspline(
        {{{0.0_r, 0.0_r, 0.0_r, 0.2_r, 0.4_r, 0.6_r, 0.8_r, 1.0_r, 1.0_r,
           1.0_r},
          {0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r}}},
        iganet::init::zeros, options);
    EXPECT_TRUE(torch::equal(bspline.coeffs(0), torch::zeros(28, options)));
    EXPECT_TRUE(torch::equal(bspline.coeffs(1), torch::zeros(28, options)));
  }

  {
    iganet::NonUniformBSpline<real_t, 2, 2, 2> bspline(
        {{{0.0_r, 0.0_r, 0.0_r, 0.2_r, 0.4_r, 0.6_r, 0.8_r, 1.0_r, 1.0_r,
           1.0_r},
          {0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r}}},
        iganet::init::ones, options);
    EXPECT_TRUE(torch::equal(bspline.coeffs(0), torch::ones(28, options)));
    EXPECT_TRUE(torch::equal(bspline.coeffs(1), torch::ones(28, options)));
  }

  {
    iganet::NonUniformBSpline<real_t, 2, 2, 2> bspline(
        {{{0.0_r, 0.0_r, 0.0_r, 0.2_r, 0.4_r, 0.6_r, 0.8_r, 1.0_r, 1.0_r,
           1.0_r},
          {0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r}}},
        iganet::init::linear, options);
    EXPECT_TRUE(torch::equal(bspline.coeffs(0),
                             torch::linspace(0, 1, 7, options).repeat(4)));
    EXPECT_TRUE(
        torch::equal(bspline.coeffs(1),
                     torch::linspace(0, 1, 4, options).repeat_interleave(7)));
  }

  {
    iganet::NonUniformBSpline<real_t, 2, 1, 1> bspline(
        {{{0.0_r, 0.0_r, 0.25_r, 0.5_r, 0.75_r, 1.0_r, 1.0_r},
          {0.0_r, 0.0_r, 0.2_r, 0.4_r, 0.6_r, 0.8_r, 1.0_r, 1.0_r}}},
        iganet::init::greville, options);
    EXPECT_TRUE(torch::equal(bspline.coeffs(0),
                             torch::linspace(0, 1, 5, options).repeat(6)));
    EXPECT_TRUE(
        torch::equal(bspline.coeffs(1),
                     torch::linspace(0, 1, 6, options).repeat_interleave(5)));
  }

  {
    iganet::NonUniformBSpline<real_t, 3, 2, 2> bspline(
        {{{0.0_r, 0.0_r, 0.0_r, 0.2_r, 0.4_r, 0.6_r, 0.8_r, 1.0_r, 1.0_r,
           1.0_r},
          {0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r}}},
        iganet::init::zeros, options);
    EXPECT_TRUE(torch::equal(bspline.coeffs(0), torch::zeros(28, options)));
    EXPECT_TRUE(torch::equal(bspline.coeffs(1), torch::zeros(28, options)));
    EXPECT_TRUE(torch::equal(bspline.coeffs(2), torch::zeros(28, options)));
  }

  {
    iganet::NonUniformBSpline<real_t, 3, 2, 2> bspline(
        {{{0.0_r, 0.0_r, 0.0_r, 0.2_r, 0.4_r, 0.6_r, 0.8_r, 1.0_r, 1.0_r,
           1.0_r},
          {0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r}}},
        iganet::init::ones, options);
    EXPECT_TRUE(torch::equal(bspline.coeffs(0), torch::ones(28, options)));
    EXPECT_TRUE(torch::equal(bspline.coeffs(1), torch::ones(28, options)));
    EXPECT_TRUE(torch::equal(bspline.coeffs(2), torch::ones(28, options)));
  }

  {
    iganet::NonUniformBSpline<real_t, 3, 2, 2> bspline(
        {{{0.0_r, 0.0_r, 0.0_r, 0.2_r, 0.4_r, 0.6_r, 0.8_r, 1.0_r, 1.0_r,
           1.0_r},
          {0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r}}},
        iganet::init::linear, options);
    EXPECT_TRUE(torch::equal(bspline.coeffs(0),
                             torch::linspace(0, 1, 7, options).repeat(4)));
    EXPECT_TRUE(
        torch::equal(bspline.coeffs(1),
                     torch::linspace(0, 1, 4, options).repeat_interleave(7)));
    EXPECT_TRUE(torch::equal(bspline.coeffs(2), torch::ones(28, options)));
  }

  {
    iganet::NonUniformBSpline<real_t, 3, 1, 1> bspline(
        {{{0.0_r, 0.0_r, 0.25_r, 0.5_r, 0.75_r, 1.0_r, 1.0_r},
          {0.0_r, 0.0_r, 0.2_r, 0.4_r, 0.6_r, 0.8_r, 1.0_r, 1.0_r}}},
        iganet::init::greville, options);
    EXPECT_TRUE(torch::equal(bspline.coeffs(0),
                             torch::linspace(0, 1, 5, options).repeat(6)));
    EXPECT_TRUE(
        torch::equal(bspline.coeffs(1),
                     torch::linspace(0, 1, 6, options).repeat_interleave(5)));
    EXPECT_TRUE(torch::equal(bspline.coeffs(2), torch::ones(30, options)));
  }

  {
    iganet::NonUniformBSpline<real_t, 4, 2, 2> bspline(
        {{{0.0_r, 0.0_r, 0.0_r, 0.2_r, 0.4_r, 0.6_r, 0.8_r, 1.0_r, 1.0_r,
           1.0_r},
          {0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r}}},
        iganet::init::zeros, options);
    EXPECT_TRUE(torch::equal(bspline.coeffs(0), torch::zeros(28, options)));
    EXPECT_TRUE(torch::equal(bspline.coeffs(1), torch::zeros(28, options)));
    EXPECT_TRUE(torch::equal(bspline.coeffs(2), torch::zeros(28, options)));
    EXPECT_TRUE(torch::equal(bspline.coeffs(3), torch::zeros(28, options)));
  }

  {
    iganet::NonUniformBSpline<real_t, 4, 2, 2> bspline(
        {{{0.0_r, 0.0_r, 0.0_r, 0.2_r, 0.4_r, 0.6_r, 0.8_r, 1.0_r, 1.0_r,
           1.0_r},
          {0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r}}},
        iganet::init::ones, options);
    EXPECT_TRUE(torch::equal(bspline.coeffs(0), torch::ones(28, options)));
    EXPECT_TRUE(torch::equal(bspline.coeffs(1), torch::ones(28, options)));
    EXPECT_TRUE(torch::equal(bspline.coeffs(2), torch::ones(28, options)));
    EXPECT_TRUE(torch::equal(bspline.coeffs(3), torch::ones(28, options)));
  }

  {
    iganet::NonUniformBSpline<real_t, 4, 2, 2> bspline(
        {{{0.0_r, 0.0_r, 0.0_r, 0.2_r, 0.4_r, 0.6_r, 0.8_r, 1.0_r, 1.0_r,
           1.0_r},
          {0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r}}},
        iganet::init::linear, options);
    EXPECT_TRUE(torch::equal(bspline.coeffs(0),
                             torch::linspace(0, 1, 7, options).repeat(4)));
    EXPECT_TRUE(
        torch::equal(bspline.coeffs(1),
                     torch::linspace(0, 1, 4, options).repeat_interleave(7)));
    EXPECT_TRUE(torch::equal(bspline.coeffs(2), torch::ones(28, options)));
    EXPECT_TRUE(torch::equal(bspline.coeffs(3), torch::ones(28, options)));
  }

  {
    iganet::NonUniformBSpline<real_t, 4, 1, 1> bspline(
        {{{0.0_r, 0.0_r, 0.25_r, 0.5_r, 0.75_r, 1.0_r, 1.0_r},
          {0.0_r, 0.0_r, 0.2_r, 0.4_r, 0.6_r, 0.8_r, 1.0_r, 1.0_r}}},
        iganet::init::greville, options);
    EXPECT_TRUE(torch::equal(bspline.coeffs(0),
                             torch::linspace(0, 1, 5, options).repeat(6)));
    EXPECT_TRUE(
        torch::equal(bspline.coeffs(1),
                     torch::linspace(0, 1, 6, options).repeat_interleave(5)));
    EXPECT_TRUE(torch::equal(bspline.coeffs(2), torch::ones(30, options)));
    EXPECT_TRUE(torch::equal(bspline.coeffs(3), torch::ones(30, options)));
  }
}

TEST_F(BSplineTest, NonUniformBSpline_uniform_refine) {
  {
    iganet::NonUniformBSpline<real_t, 3, 3, 4> bspline({4, 5});
    iganet::NonUniformBSpline<real_t, 3, 3, 4> bspline_ref({5, 6});
    bspline.uniform_refine();

    EXPECT_TRUE(bspline.isclose(bspline_ref));
  }

  {
    iganet::NonUniformBSpline<real_t, 3, 3, 4> bspline({4, 5});
    iganet::NonUniformBSpline<real_t, 3, 3, 4> bspline_ref({7, 8});
    bspline.uniform_refine(2);

    EXPECT_TRUE(bspline.isclose(bspline_ref));
  }

  {
    iganet::NonUniformBSpline<real_t, 3, 3, 4> bspline({4, 5});
    iganet::NonUniformBSpline<real_t, 3, 3, 4> bspline_ref({5, 5});
    bspline.uniform_refine(1, 0);

    EXPECT_TRUE(bspline.isclose(bspline_ref));
  }

  {
    iganet::NonUniformBSpline<real_t, 3, 3, 4> bspline({4, 5});
    iganet::NonUniformBSpline<real_t, 3, 3, 4> bspline_ref({5, 8});
    bspline.uniform_refine(1, 0).uniform_refine(2, 1);

    EXPECT_TRUE(bspline.isclose(bspline_ref));
  }
}

TEST_F(BSplineTest, NonUniformBSpline_copy_constructor) {
  iganet::NonUniformBSpline<real_t, 3, 3, 4> bspline_orig(
      {4, 5}, iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 3, 3, 4> bspline_copy(bspline_orig);

  bspline_orig.transform([](const std::array<real_t, 2> xi) {
    return std::array<real_t, 3>{0.0_r, 1.0_r, 2.0_r};
  });

  EXPECT_TRUE(bspline_orig == bspline_copy);
}

TEST_F(BSplineTest, NonUniformBSpline_clone_constructor) {
  iganet::NonUniformBSpline<real_t, 3, 3, 4> bspline_ref(
      {4, 5}, iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 3, 3, 4> bspline_orig(
      {4, 5}, iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 3, 3, 4> bspline_clone(bspline_orig, true);

  bspline_orig.transform([](const std::array<real_t, 2> xi) {
    return std::array<real_t, 3>{0.0_r, 1.0_r, 2.0_r};
  });

  EXPECT_TRUE(bspline_ref == bspline_clone);
}

TEST_F(BSplineTest, NonUniformBSpline_move_constructor) {
  iganet::NonUniformBSpline<real_t, 3, 3, 4> bspline_ref(
      {7, 8}, iganet::init::greville, options);
  auto bspline(iganet::NonUniformBSpline<real_t, 3, 3, 4>(
                   {4, 5}, iganet::init::greville, options)
                   .uniform_refine(2));

  EXPECT_TRUE(bspline.isclose(bspline_ref));
}

TEST_F(BSplineTest, NonUniformBSpline_copy_coeffs_constructor) {
  iganet::NonUniformBSpline<real_t, 3, 3, 4> bspline_orig(
      {4, 5}, iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 3, 3, 4> bspline_copy(
      bspline_orig, bspline_orig.coeffs());

  bspline_orig.transform([](const std::array<real_t, 2> xi) {
    return std::array<real_t, 3>{0.0_r, 1.0_r, 2.0_r};
  });

  EXPECT_TRUE(bspline_orig == bspline_copy);
}

TEST_F(BSplineTest, NonUniformBSpline_clone_coeffs_constructor) {
  iganet::NonUniformBSpline<real_t, 3, 3, 4> bspline_ref(
      {4, 5}, iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 3, 3, 4> bspline_orig(
      {4, 5}, iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 3, 3, 4> bspline_clone(
      bspline_orig, bspline_orig.coeffs(), true);

  bspline_orig.transform([](const std::array<real_t, 2> xi) {
    return std::array<real_t, 3>{0.0_r, 1.0_r, 2.0_r};
  });

  EXPECT_TRUE(bspline_ref == bspline_clone);
}

TEST_F(BSplineTest, NonUniformBSpline_read_write) {
  std::filesystem::path filename =
      std::filesystem::temp_directory_path() / std::to_string(rand());
  iganet::NonUniformBSpline<real_t, 3, 1, 2, 3> bspline_out(
      {{{0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 1.0_r, 1.0_r, 1.0_r},
        {0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r}}},
      iganet::init::greville, options);
  bspline_out.save(filename.c_str());
  iganet::NonUniformBSpline<real_t, 3, 1, 2, 3> bspline_in(options);
  bspline_in.load(filename.c_str());
  std::filesystem::remove(filename);

  EXPECT_TRUE(bspline_in == bspline_out);
  EXPECT_FALSE(bspline_in != bspline_out);
}

TEST_F(BSplineTest, NonUniformBSpline_to_from_xml) {
  {
    iganet::NonUniformBSpline<real_t, 1, 3> bspline_out(
        {4}, iganet::init::zeros, options);

    bspline_out.transform([](const std::array<real_t, 1> xi) {
      return std::array<real_t, 1>{static_cast<real_t>(std::rand())};
    });

    pugi::xml_document doc = bspline_out.to_xml();

    iganet::NonUniformBSpline<real_t, 1, 3> bspline_in(options);
    bspline_in.from_xml(doc);

    EXPECT_TRUE(bspline_in == bspline_out);

    // non-matching degree
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 1, 2>{}.from_xml(doc, 0)),
                 std::runtime_error);

    // non-matching parametric dimension
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 1, 3, 3>{}.from_xml(doc, 0)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 1, 3, 3, 3>{}.from_xml(doc, 0)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 1, 3, 3, 3, 3>{}.from_xml(doc, 0)),
        std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 2, 3>{}.from_xml(doc, 0)),
                 std::runtime_error);
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 3, 3>{}.from_xml(doc, 0)),
                 std::runtime_error);
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 4, 3>{}.from_xml(doc, 0)),
                 std::runtime_error);

    // non-matching id
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 1, 3>{}.from_xml(doc, 1)),
                 std::runtime_error);
  }

  {
    iganet::NonUniformBSpline<real_t, 2, 3> bspline_out(
        {4}, iganet::init::zeros, options);

    bspline_out.transform([](const std::array<real_t, 1> xi) {
      return std::array<real_t, 2>{static_cast<real_t>(std::rand()),
                                   static_cast<real_t>(std::rand())};
    });

    pugi::xml_document doc = bspline_out.to_xml();

    iganet::NonUniformBSpline<real_t, 2, 3> bspline_in(options);
    bspline_in.from_xml(doc);

    EXPECT_TRUE(bspline_in == bspline_out);

    // non-matching degree
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 2, 2>{}.from_xml(doc, 0)),
                 std::runtime_error);

    // non-matching parametric dimension
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 2, 3, 3>{}.from_xml(doc, 0)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 2, 3, 3, 3>{}.from_xml(doc, 0)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 2, 3, 3, 3, 3>{}.from_xml(doc, 0)),
        std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 1, 3>{}.from_xml(doc, 0)),
                 std::runtime_error);
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 3, 3>{}.from_xml(doc, 0)),
                 std::runtime_error);
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 4, 3>{}.from_xml(doc, 0)),
                 std::runtime_error);

    // non-matching id
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 1, 3>{}.from_xml(doc, 1)),
                 std::runtime_error);
  }

  {
    iganet::NonUniformBSpline<real_t, 3, 3> bspline_out(
        {4}, iganet::init::zeros, options);

    bspline_out.transform([](const std::array<real_t, 1> xi) {
      return std::array<real_t, 3>{static_cast<real_t>(std::rand()),
                                   static_cast<real_t>(std::rand()),
                                   static_cast<real_t>(std::rand())};
    });

    pugi::xml_document doc = bspline_out.to_xml();

    iganet::NonUniformBSpline<real_t, 3, 3> bspline_in(options);
    bspline_in.from_xml(doc);

    EXPECT_TRUE(bspline_in == bspline_out);

    // non-matching degree
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 3, 2>{}.from_xml(doc, 0)),
                 std::runtime_error);

    // non-matching parametric dimension
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 3, 3, 3>{}.from_xml(doc, 0)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 3, 3, 3, 3>{}.from_xml(doc, 0)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 3, 3, 3, 3, 3>{}.from_xml(doc, 0)),
        std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 1, 3>{}.from_xml(doc, 0)),
                 std::runtime_error);
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 2, 3>{}.from_xml(doc, 0)),
                 std::runtime_error);
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 4, 3>{}.from_xml(doc, 0)),
                 std::runtime_error);

    // non-matching id
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 3, 3>{}.from_xml(doc, 1)),
                 std::runtime_error);
  }

  {
    iganet::NonUniformBSpline<real_t, 4, 3> bspline_out(
        {4}, iganet::init::zeros, options);

    bspline_out.transform([](const std::array<real_t, 1> xi) {
      return std::array<real_t, 4>{
          static_cast<real_t>(std::rand()), static_cast<real_t>(std::rand()),
          static_cast<real_t>(std::rand()), static_cast<real_t>(std::rand())};
    });

    pugi::xml_document doc = bspline_out.to_xml();

    iganet::NonUniformBSpline<real_t, 4, 3> bspline_in(options);
    bspline_in.from_xml(doc);

    EXPECT_TRUE(bspline_in == bspline_out);

    // non-matching degree
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 4, 2>{}.from_xml(doc, 0)),
                 std::runtime_error);

    // non-matching parametric dimension
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 4, 3, 3>{}.from_xml(doc, 0)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 4, 3, 3, 3>{}.from_xml(doc, 0)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 4, 3, 3, 3, 3>{}.from_xml(doc, 0)),
        std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 1, 3>{}.from_xml(doc, 0)),
                 std::runtime_error);
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 2, 3>{}.from_xml(doc, 0)),
                 std::runtime_error);
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 3, 3>{}.from_xml(doc, 0)),
                 std::runtime_error);

    // non-matching id
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 4, 3>{}.from_xml(doc, 1)),
                 std::runtime_error);
  }

  {
    iganet::NonUniformBSpline<real_t, 1, 3, 4> bspline_out(
        {4, 5}, iganet::init::zeros, options);

    bspline_out.transform([](const std::array<real_t, 2> xi) {
      return std::array<real_t, 1>{static_cast<real_t>(std::rand())};
    });

    pugi::xml_document doc = bspline_out.to_xml();

    iganet::NonUniformBSpline<real_t, 1, 3, 4> bspline_in(options);
    bspline_in.from_xml(doc);

    EXPECT_TRUE(bspline_in == bspline_out);

    // non-matching degree
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 1, 3, 3>{}.from_xml(doc, 0)),
        std::runtime_error);

    // non-matching parametric dimension
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 1, 3>{}.from_xml(doc, 0)),
                 std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 1, 3, 4, 3>{}.from_xml(doc, 0)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 1, 3, 4, 3, 3>{}.from_xml(doc, 0)),
        std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 2, 3, 4>{}.from_xml(doc, 0)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 3, 3, 4>{}.from_xml(doc, 0)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 4, 3, 4>{}.from_xml(doc, 0)),
        std::runtime_error);

    // non-matching id
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 1, 3, 4>{}.from_xml(doc, 1)),
        std::runtime_error);
  }

  {
    iganet::NonUniformBSpline<real_t, 2, 3, 4> bspline_out(
        {4, 5}, iganet::init::zeros, options);

    bspline_out.transform([](const std::array<real_t, 2> xi) {
      return std::array<real_t, 2>{static_cast<real_t>(std::rand()),
                                   static_cast<real_t>(std::rand())};
    });

    pugi::xml_document doc = bspline_out.to_xml();

    iganet::NonUniformBSpline<real_t, 2, 3, 4> bspline_in(options);
    bspline_in.from_xml(doc);

    EXPECT_TRUE(bspline_in == bspline_out);

    // non-matching degree
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 2, 3, 3>{}.from_xml(doc, 0)),
        std::runtime_error);

    // non-matching parametric dimension
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 2, 3>{}.from_xml(doc, 0)),
                 std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 2, 3, 4, 3>{}.from_xml(doc, 0)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 2, 3, 4, 3, 3>{}.from_xml(doc, 0)),
        std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 1, 3, 4>{}.from_xml(doc, 0)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 3, 3, 4>{}.from_xml(doc, 0)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 4, 3, 4>{}.from_xml(doc, 0)),
        std::runtime_error);

    // non-matching id
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 2, 3, 4>{}.from_xml(doc, 1)),
        std::runtime_error);
  }

  {
    iganet::NonUniformBSpline<real_t, 3, 3, 4> bspline_out(
        {4, 5}, iganet::init::zeros, options);

    bspline_out.transform([](const std::array<real_t, 2> xi) {
      return std::array<real_t, 3>{static_cast<real_t>(std::rand()),
                                   static_cast<real_t>(std::rand()),
                                   static_cast<real_t>(std::rand())};
    });

    pugi::xml_document doc = bspline_out.to_xml();

    iganet::NonUniformBSpline<real_t, 3, 3, 4> bspline_in(options);
    bspline_in.from_xml(doc);

    EXPECT_TRUE(bspline_in == bspline_out);

    // non-matching degree
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 3, 3, 3>{}.from_xml(doc, 0)),
        std::runtime_error);

    // non-matching parametric dimension
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 3, 3>{}.from_xml(doc, 0)),
                 std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 3, 3, 4, 3>{}.from_xml(doc, 0)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 3, 3, 4, 3, 3>{}.from_xml(doc, 0)),
        std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 1, 3, 4>{}.from_xml(doc, 0)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 2, 3, 4>{}.from_xml(doc, 0)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 4, 3, 4>{}.from_xml(doc, 0)),
        std::runtime_error);

    // non-matching id
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 3, 3, 4>{}.from_xml(doc, 1)),
        std::runtime_error);
  }

  {
    iganet::NonUniformBSpline<real_t, 4, 3, 4> bspline_out(
        {4, 5}, iganet::init::zeros, options);

    bspline_out.transform([](const std::array<real_t, 2> xi) {
      return std::array<real_t, 4>{
          static_cast<real_t>(std::rand()), static_cast<real_t>(std::rand()),
          static_cast<real_t>(std::rand()), static_cast<real_t>(std::rand())};
    });

    pugi::xml_document doc = bspline_out.to_xml();

    iganet::NonUniformBSpline<real_t, 4, 3, 4> bspline_in(options);
    bspline_in.from_xml(doc);

    EXPECT_TRUE(bspline_in == bspline_out);

    // non-matching degree
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 4, 3, 3>{}.from_xml(doc, 0)),
        std::runtime_error);

    // non-matching parametric dimension
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 4, 3>{}.from_xml(doc, 0)),
                 std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 4, 3, 4, 3>{}.from_xml(doc, 0)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 4, 3, 4, 3, 3>{}.from_xml(doc, 0)),
        std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 1, 3, 4>{}.from_xml(doc, 0)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 2, 3, 4>{}.from_xml(doc, 0)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 3, 3, 4>{}.from_xml(doc, 0)),
        std::runtime_error);

    // non-matching id
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 4, 3, 4>{}.from_xml(doc, 1)),
        std::runtime_error);
  }

  {
    iganet::NonUniformBSpline<real_t, 1, 3, 4, 5> bspline_out(
        {4, 5, 6}, iganet::init::zeros, options);

    bspline_out.transform([](const std::array<real_t, 3> xi) {
      return std::array<real_t, 1>{static_cast<real_t>(std::rand())};
    });

    pugi::xml_document doc = bspline_out.to_xml();

    iganet::NonUniformBSpline<real_t, 1, 3, 4, 5> bspline_in(options);
    bspline_in.from_xml(doc);

    EXPECT_TRUE(bspline_in == bspline_out);

    // non-matching degree
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 1, 3, 3, 3>{}.from_xml(doc, 0)),
        std::runtime_error);

    // non-matching parametric dimension
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 1, 3>{}.from_xml(doc, 0)),
                 std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 1, 3, 4>{}.from_xml(doc, 0)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 1, 3, 4, 5, 3>{}.from_xml(doc, 0)),
        std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 2, 3, 4, 5>{}.from_xml(doc, 0)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 3, 3, 4, 5>{}.from_xml(doc, 0)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 4, 3, 4, 5>{}.from_xml(doc, 0)),
        std::runtime_error);

    // non-matching id
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 1, 3, 4, 5>{}.from_xml(doc, 1)),
        std::runtime_error);
  }

  {
    iganet::NonUniformBSpline<real_t, 2, 3, 4, 5> bspline_out(
        {4, 5, 6}, iganet::init::zeros, options);

    bspline_out.transform([](const std::array<real_t, 3> xi) {
      return std::array<real_t, 2>{static_cast<real_t>(std::rand()),
                                   static_cast<real_t>(std::rand())};
    });

    pugi::xml_document doc = bspline_out.to_xml();

    iganet::NonUniformBSpline<real_t, 2, 3, 4, 5> bspline_in(options);
    bspline_in.from_xml(doc);

    EXPECT_TRUE(bspline_in == bspline_out);

    // non-matching degree
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 2, 3, 3, 3>{}.from_xml(doc, 0)),
        std::runtime_error);

    // non-matching parametric dimension
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 2, 3>{}.from_xml(doc, 0)),
                 std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 2, 3, 4>{}.from_xml(doc, 0)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 2, 3, 4, 5, 3>{}.from_xml(doc, 0)),
        std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 1, 3, 4, 5>{}.from_xml(doc, 0)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 3, 3, 4, 5>{}.from_xml(doc, 0)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 4, 3, 4, 5>{}.from_xml(doc, 0)),
        std::runtime_error);

    // non-matching id
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 2, 3, 4, 5>{}.from_xml(doc, 1)),
        std::runtime_error);
  }

  {
    iganet::NonUniformBSpline<real_t, 3, 3, 4, 5> bspline_out(
        {4, 5, 6}, iganet::init::zeros, options);

    bspline_out.transform([](const std::array<real_t, 3> xi) {
      return std::array<real_t, 3>{static_cast<real_t>(std::rand()),
                                   static_cast<real_t>(std::rand()),
                                   static_cast<real_t>(std::rand())};
    });

    pugi::xml_document doc = bspline_out.to_xml();

    iganet::NonUniformBSpline<real_t, 3, 3, 4, 5> bspline_in(options);
    bspline_in.from_xml(doc);

    EXPECT_TRUE(bspline_in == bspline_out);

    // non-matching degree
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 3, 3, 3, 3>{}.from_xml(doc, 0)),
        std::runtime_error);

    // non-matching parametric dimension
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 3, 3>{}.from_xml(doc, 0)),
                 std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 3, 3, 4>{}.from_xml(doc, 0)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 3, 3, 4, 5, 3>{}.from_xml(doc, 0)),
        std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 1, 3, 4, 5>{}.from_xml(doc, 0)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 2, 3, 4, 5>{}.from_xml(doc, 0)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 4, 3, 4, 5>{}.from_xml(doc, 0)),
        std::runtime_error);

    // non-matching id
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 3, 3, 4, 5>{}.from_xml(doc, 1)),
        std::runtime_error);
  }

  {
    iganet::NonUniformBSpline<real_t, 4, 3, 4, 5> bspline_out(
        {4, 5, 6}, iganet::init::zeros, options);

    bspline_out.transform([](const std::array<real_t, 3> xi) {
      return std::array<real_t, 4>{
          static_cast<real_t>(std::rand()), static_cast<real_t>(std::rand()),
          static_cast<real_t>(std::rand()), static_cast<real_t>(std::rand())};
    });

    pugi::xml_document doc = bspline_out.to_xml();

    iganet::NonUniformBSpline<real_t, 4, 3, 4, 5> bspline_in(options);
    bspline_in.from_xml(doc);

    EXPECT_TRUE(bspline_in == bspline_out);

    // non-matching degree
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 4, 3, 3, 3>{}.from_xml(doc, 0)),
        std::runtime_error);

    // non-matching parametric dimension
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 4, 3>{}.from_xml(doc, 0)),
                 std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 4, 3, 4>{}.from_xml(doc, 0)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 4, 3, 4, 5, 3>{}.from_xml(doc, 0)),
        std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 1, 3, 4, 5>{}.from_xml(doc, 0)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 2, 3, 4, 5>{}.from_xml(doc, 0)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 3, 3, 4, 5>{}.from_xml(doc, 0)),
        std::runtime_error);

    // non-matching id
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 4, 3, 4, 5>{}.from_xml(doc, 1)),
        std::runtime_error);
  }

  {
    iganet::NonUniformBSpline<real_t, 1, 3, 4, 5, 1> bspline_out(
        {4, 5, 6, 2}, iganet::init::zeros, options);

    bspline_out.transform([](const std::array<real_t, 4> xi) {
      return std::array<real_t, 1>{static_cast<real_t>(std::rand())};
    });

    pugi::xml_document doc = bspline_out.to_xml();

    iganet::NonUniformBSpline<real_t, 1, 3, 4, 5, 1> bspline_in(options);
    bspline_in.from_xml(doc);

    EXPECT_TRUE(bspline_in == bspline_out);

    // non-matching degree
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 1, 3, 4, 5, 2>{}.from_xml(doc, 0)),
        std::runtime_error);

    // non-matching parametric dimension
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 1, 3>{}.from_xml(doc, 0)),
                 std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 1, 3, 4>{}.from_xml(doc, 0)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 1, 3, 4, 5>{}.from_xml(doc, 0)),
        std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 2, 3, 4, 5, 1>{}.from_xml(doc, 0)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 3, 3, 4, 5, 1>{}.from_xml(doc, 0)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 4, 3, 4, 5, 1>{}.from_xml(doc, 0)),
        std::runtime_error);

    // non-matching id
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 1, 3, 4, 5, 1>{}.from_xml(doc, 1)),
        std::runtime_error);
  }

  {
    iganet::NonUniformBSpline<real_t, 2, 3, 4, 5, 1> bspline_out(
        {4, 5, 6, 2}, iganet::init::zeros, options);

    bspline_out.transform([](const std::array<real_t, 4> xi) {
      return std::array<real_t, 2>{static_cast<real_t>(std::rand()),
                                   static_cast<real_t>(std::rand())};
    });

    pugi::xml_document doc = bspline_out.to_xml();

    iganet::NonUniformBSpline<real_t, 2, 3, 4, 5, 1> bspline_in(options);
    bspline_in.from_xml(doc);

    EXPECT_TRUE(bspline_in == bspline_out);

    // non-matching degree
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 2, 3, 4, 5, 2>{}.from_xml(doc, 0)),
        std::runtime_error);

    // non-matching parametric dimension
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 2, 3>{}.from_xml(doc, 0)),
                 std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 2, 3, 4>{}.from_xml(doc, 0)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 2, 3, 4, 5>{}.from_xml(doc, 0)),
        std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 1, 3, 4, 5, 1>{}.from_xml(doc, 0)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 3, 3, 4, 5, 1>{}.from_xml(doc, 0)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 4, 3, 4, 5, 1>{}.from_xml(doc, 0)),
        std::runtime_error);

    // non-matching id
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 2, 3, 4, 5, 1>{}.from_xml(doc, 1)),
        std::runtime_error);
  }

  {
    iganet::NonUniformBSpline<real_t, 3, 3, 4, 5, 1> bspline_out(
        {4, 5, 6, 2}, iganet::init::zeros, options);

    bspline_out.transform([](const std::array<real_t, 4> xi) {
      return std::array<real_t, 3>{static_cast<real_t>(std::rand()),
                                   static_cast<real_t>(std::rand()),
                                   static_cast<real_t>(std::rand())};
    });

    pugi::xml_document doc = bspline_out.to_xml();

    iganet::NonUniformBSpline<real_t, 3, 3, 4, 5, 1> bspline_in(options);
    bspline_in.from_xml(doc);

    EXPECT_TRUE(bspline_in == bspline_out);

    // non-matching degree
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 3, 3, 4, 5, 2>{}.from_xml(doc, 0)),
        std::runtime_error);

    // non-matching parametric dimension
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 3, 3>{}.from_xml(doc, 0)),
                 std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 3, 3, 4>{}.from_xml(doc, 0)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 3, 3, 4, 5>{}.from_xml(doc, 0)),
        std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 1, 3, 4, 5, 1>{}.from_xml(doc, 0)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 2, 3, 4, 5, 1>{}.from_xml(doc, 0)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 4, 3, 4, 5, 1>{}.from_xml(doc, 0)),
        std::runtime_error);

    // non-matching id
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 3, 3, 4, 5, 1>{}.from_xml(doc, 1)),
        std::runtime_error);
  }

  {
    iganet::NonUniformBSpline<real_t, 4, 3, 4, 5, 1> bspline_out(
        {4, 5, 6, 2}, iganet::init::zeros, options);

    bspline_out.transform([](const std::array<real_t, 4> xi) {
      return std::array<real_t, 4>{
          static_cast<real_t>(std::rand()), static_cast<real_t>(std::rand()),
          static_cast<real_t>(std::rand()), static_cast<real_t>(std::rand())};
    });

    pugi::xml_document doc = bspline_out.to_xml();

    iganet::NonUniformBSpline<real_t, 4, 3, 4, 5, 1> bspline_in(options);
    bspline_in.from_xml(doc);

    EXPECT_TRUE(bspline_in == bspline_out);

    // non-matching degree
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 4, 3, 4, 5, 2>{}.from_xml(doc, 0)),
        std::runtime_error);

    // non-matching parametric dimension
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 4, 3>{}.from_xml(doc, 0)),
                 std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 4, 3, 4>{}.from_xml(doc, 0)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 4, 3, 4, 5>{}.from_xml(doc, 0)),
        std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 1, 3, 4, 5, 1>{}.from_xml(doc, 0)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 2, 3, 4, 5, 1>{}.from_xml(doc, 0)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 3, 3, 4, 5, 1>{}.from_xml(doc, 0)),
        std::runtime_error);

    // non-matching id
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 4, 3, 4, 5, 1>{}.from_xml(doc, 1)),
        std::runtime_error);
  }
}

TEST_F(BSplineTest, NonUniformBSpline_load_from_xml) {
  {
    pugi::xml_document doc;
    pugi::xml_parse_result result =
        doc.load_file(IGANET_DATA_DIR "domain1d/line.xml");

    iganet::NonUniformBSpline<real_t, 3, 2> bspline_in(options);
    bspline_in.from_xml(doc);

    iganet::NonUniformBSpline<real_t, 3, 2> bspline_ref(
        {3}, iganet::init::zeros, options);

    bspline_ref.transform([](const std::array<real_t, 1> xi) {
      return std::array<real_t, 3>{xi[0], 0.0_r, 0.0_r};
    });

    EXPECT_TRUE(bspline_in == bspline_ref);
  }

  {
    pugi::xml_document doc;
    pugi::xml_parse_result result =
        doc.load_file(IGANET_DATA_DIR "domain2d/square.xml");

    iganet::NonUniformBSpline<real_t, 2, 1, 1> bspline_in(options);
    bspline_in.from_xml(doc, 1);

    iganet::NonUniformBSpline<real_t, 2, 1, 1> bspline_ref(
        {2, 2}, iganet::init::greville, options);

    EXPECT_TRUE(bspline_in == bspline_ref);
  }

  {
    pugi::xml_document doc;
    pugi::xml_parse_result result =
        doc.load_file(IGANET_DATA_DIR "domain3d/GshapedVolume.xml");

    iganet::NonUniformBSpline<real_t, 3, 2, 2, 2> bspline_in(options);
    bspline_in.from_xml(doc);
  }

  {
    pugi::xml_document doc;
    pugi::xml_parse_result result =
        doc.load_file(IGANET_DATA_DIR "surfaces/g_plus_s_surf.xml");

    iganet::NonUniformBSpline<real_t, 3, 3, 3> bspline_in0(options);
    iganet::NonUniformBSpline<real_t, 3, 3, 1> bspline_in1(options);

    for (int i = 0; i < 126; ++i) {
      try {
        bspline_in0.from_xml(doc, i);
      } catch (...) {
        bspline_in1.from_xml(doc, i);
      }
    }
  }
}

TEST_F(BSplineTest, NonUniformBSpline_to_from_json) {
  {
    iganet::NonUniformBSpline<real_t, 1, 3> bspline_out(
        {4}, iganet::init::zeros, options);

    bspline_out.transform([](const std::array<real_t, 1> xi) {
      return std::array<real_t, 1>{static_cast<real_t>(std::rand())};
    });

    nlohmann::json json = bspline_out.to_json();

    iganet::NonUniformBSpline<real_t, 1, 3> bspline_in(options);
    bspline_in.from_json(json);

    EXPECT_TRUE(bspline_in == bspline_out);

    // non-matching degree
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 1, 2>{}.from_json(json)),
                 std::runtime_error);

    // non-matching parametric dimension
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 1, 3, 3>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 1, 3, 3, 3>{}.from_json(json)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 1, 3, 3, 3, 3>{}.from_json(json)),
        std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 2, 3>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 3, 3>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 4, 3>{}.from_json(json)),
                 std::runtime_error);
  }

  {
    iganet::NonUniformBSpline<real_t, 2, 3> bspline_out(
        {4}, iganet::init::zeros, options);

    bspline_out.transform([](const std::array<real_t, 1> xi) {
      return std::array<real_t, 2>{static_cast<real_t>(std::rand()),
                                   static_cast<real_t>(std::rand())};
    });

    nlohmann::json json = bspline_out.to_json();

    iganet::NonUniformBSpline<real_t, 2, 3> bspline_in(options);
    bspline_in.from_json(json);

    EXPECT_TRUE(bspline_in == bspline_out);

    // non-matching degree
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 2, 2>{}.from_json(json)),
                 std::runtime_error);

    // non-matching parametric dimension
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 2, 3, 3>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 2, 3, 3, 3>{}.from_json(json)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 2, 3, 3, 3, 3>{}.from_json(json)),
        std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 1, 3>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 3, 3>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 4, 3>{}.from_json(json)),
                 std::runtime_error);
  }

  {
    iganet::NonUniformBSpline<real_t, 3, 3> bspline_out(
        {4}, iganet::init::zeros, options);

    bspline_out.transform([](const std::array<real_t, 1> xi) {
      return std::array<real_t, 3>{static_cast<real_t>(std::rand()),
                                   static_cast<real_t>(std::rand()),
                                   static_cast<real_t>(std::rand())};
    });

    nlohmann::json json = bspline_out.to_json();

    iganet::NonUniformBSpline<real_t, 3, 3> bspline_in(options);
    bspline_in.from_json(json);

    EXPECT_TRUE(bspline_in == bspline_out);

    // non-matching degree
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 3, 2>{}.from_json(json)),
                 std::runtime_error);

    // non-matching parametric dimension
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 3, 3, 3>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 3, 3, 3, 3>{}.from_json(json)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 3, 3, 3, 3, 3>{}.from_json(json)),
        std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 1, 3>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 2, 3>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 4, 3>{}.from_json(json)),
                 std::runtime_error);
  }

  {
    iganet::NonUniformBSpline<real_t, 4, 3> bspline_out(
        {4}, iganet::init::zeros, options);

    bspline_out.transform([](const std::array<real_t, 1> xi) {
      return std::array<real_t, 4>{
          static_cast<real_t>(std::rand()), static_cast<real_t>(std::rand()),
          static_cast<real_t>(std::rand()), static_cast<real_t>(std::rand())};
    });

    nlohmann::json json = bspline_out.to_json();

    iganet::NonUniformBSpline<real_t, 4, 3> bspline_in(options);
    bspline_in.from_json(json);

    EXPECT_TRUE(bspline_in == bspline_out);

    // non-matching degree
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 4, 2>{}.from_json(json)),
                 std::runtime_error);

    // non-matching parametric dimension
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 4, 3, 3>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 4, 3, 3, 3>{}.from_json(json)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 4, 3, 3, 3, 3>{}.from_json(json)),
        std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 1, 3>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 2, 3>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 3, 3>{}.from_json(json)),
                 std::runtime_error);
  }

  {
    iganet::NonUniformBSpline<real_t, 1, 3, 4> bspline_out(
        {4, 5}, iganet::init::zeros, options);

    bspline_out.transform([](const std::array<real_t, 2> xi) {
      return std::array<real_t, 1>{static_cast<real_t>(std::rand())};
    });

    nlohmann::json json = bspline_out.to_json();

    iganet::NonUniformBSpline<real_t, 1, 3, 4> bspline_in(options);
    bspline_in.from_json(json);

    EXPECT_TRUE(bspline_in == bspline_out);

    // non-matching degree
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 1, 3, 3>{}.from_json(json)),
                 std::runtime_error);

    // non-matching parametric dimension
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 1, 3>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 1, 3, 4, 3>{}.from_json(json)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 1, 3, 4, 3, 3>{}.from_json(json)),
        std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 2, 3, 4>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 3, 3, 4>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 4, 3, 4>{}.from_json(json)),
                 std::runtime_error);
  }

  {
    iganet::NonUniformBSpline<real_t, 2, 3, 4> bspline_out(
        {4, 5}, iganet::init::zeros, options);

    bspline_out.transform([](const std::array<real_t, 2> xi) {
      return std::array<real_t, 2>{static_cast<real_t>(std::rand()),
                                   static_cast<real_t>(std::rand())};
    });

    nlohmann::json json = bspline_out.to_json();

    iganet::NonUniformBSpline<real_t, 2, 3, 4> bspline_in(options);
    bspline_in.from_json(json);

    EXPECT_TRUE(bspline_in == bspline_out);

    // non-matching degree
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 2, 3, 3>{}.from_json(json)),
                 std::runtime_error);

    // non-matching parametric dimension
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 2, 3>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 2, 3, 4, 3>{}.from_json(json)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 2, 3, 4, 3, 3>{}.from_json(json)),
        std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 1, 3, 4>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 3, 3, 4>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 4, 3, 4>{}.from_json(json)),
                 std::runtime_error);
  }

  {
    iganet::NonUniformBSpline<real_t, 3, 3, 4> bspline_out(
        {4, 5}, iganet::init::zeros, options);

    bspline_out.transform([](const std::array<real_t, 2> xi) {
      return std::array<real_t, 3>{static_cast<real_t>(std::rand()),
                                   static_cast<real_t>(std::rand()),
                                   static_cast<real_t>(std::rand())};
    });

    nlohmann::json json = bspline_out.to_json();

    iganet::NonUniformBSpline<real_t, 3, 3, 4> bspline_in(options);
    bspline_in.from_json(json);

    EXPECT_TRUE(bspline_in == bspline_out);

    // non-matching degree
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 3, 3, 3>{}.from_json(json)),
                 std::runtime_error);

    // non-matching parametric dimension
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 3, 3>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 3, 3, 4, 3>{}.from_json(json)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 3, 3, 4, 3, 3>{}.from_json(json)),
        std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 1, 3, 4>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 2, 3, 4>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 4, 3, 4>{}.from_json(json)),
                 std::runtime_error);
  }

  {
    iganet::NonUniformBSpline<real_t, 4, 3, 4> bspline_out(
        {4, 5}, iganet::init::zeros, options);

    bspline_out.transform([](const std::array<real_t, 2> xi) {
      return std::array<real_t, 4>{
          static_cast<real_t>(std::rand()), static_cast<real_t>(std::rand()),
          static_cast<real_t>(std::rand()), static_cast<real_t>(std::rand())};
    });

    nlohmann::json json = bspline_out.to_json();

    iganet::NonUniformBSpline<real_t, 4, 3, 4> bspline_in(options);
    bspline_in.from_json(json);

    EXPECT_TRUE(bspline_in == bspline_out);

    // non-matching degree
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 4, 3, 3>{}.from_json(json)),
                 std::runtime_error);

    // non-matching parametric dimension
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 4, 3>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 4, 3, 4, 3>{}.from_json(json)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 4, 3, 4, 3, 3>{}.from_json(json)),
        std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 1, 3, 4>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 2, 3, 4>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 3, 3, 4>{}.from_json(json)),
                 std::runtime_error);
  }

  {
    iganet::NonUniformBSpline<real_t, 1, 3, 4, 5> bspline_out(
        {4, 5, 6}, iganet::init::zeros, options);

    bspline_out.transform([](const std::array<real_t, 3> xi) {
      return std::array<real_t, 1>{static_cast<real_t>(std::rand())};
    });

    nlohmann::json json = bspline_out.to_json();

    iganet::NonUniformBSpline<real_t, 1, 3, 4, 5> bspline_in(options);
    bspline_in.from_json(json);

    EXPECT_TRUE(bspline_in == bspline_out);

    // non-matching degree
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 1, 3, 3, 3>{}.from_json(json)),
        std::runtime_error);

    // non-matching parametric dimension
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 1, 3>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 1, 3, 4>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 1, 3, 4, 5, 3>{}.from_json(json)),
        std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 2, 3, 4, 5>{}.from_json(json)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 3, 3, 4, 5>{}.from_json(json)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 4, 3, 4, 5>{}.from_json(json)),
        std::runtime_error);
  }

  {
    iganet::NonUniformBSpline<real_t, 2, 3, 4, 5> bspline_out(
        {4, 5, 6}, iganet::init::zeros, options);

    bspline_out.transform([](const std::array<real_t, 3> xi) {
      return std::array<real_t, 2>{static_cast<real_t>(std::rand()),
                                   static_cast<real_t>(std::rand())};
    });

    nlohmann::json json = bspline_out.to_json();

    iganet::NonUniformBSpline<real_t, 2, 3, 4, 5> bspline_in(options);
    bspline_in.from_json(json);

    EXPECT_TRUE(bspline_in == bspline_out);

    // non-matching degree
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 2, 3, 3, 3>{}.from_json(json)),
        std::runtime_error);

    // non-matching parametric dimension
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 2, 3>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 2, 3, 4>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 2, 3, 4, 5, 3>{}.from_json(json)),
        std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 1, 3, 4, 5>{}.from_json(json)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 3, 3, 4, 5>{}.from_json(json)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 4, 3, 4, 5>{}.from_json(json)),
        std::runtime_error);
  }

  {
    iganet::NonUniformBSpline<real_t, 3, 3, 4, 5> bspline_out(
        {4, 5, 6}, iganet::init::zeros, options);

    bspline_out.transform([](const std::array<real_t, 3> xi) {
      return std::array<real_t, 3>{static_cast<real_t>(std::rand()),
                                   static_cast<real_t>(std::rand()),
                                   static_cast<real_t>(std::rand())};
    });

    nlohmann::json json = bspline_out.to_json();

    iganet::NonUniformBSpline<real_t, 3, 3, 4, 5> bspline_in(options);
    bspline_in.from_json(json);

    EXPECT_TRUE(bspline_in == bspline_out);

    // non-matching degree
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 3, 3, 3, 3>{}.from_json(json)),
        std::runtime_error);

    // non-matching parametric dimension
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 3, 3>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 3, 3, 4>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 3, 3, 4, 5, 3>{}.from_json(json)),
        std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 1, 3, 4, 5>{}.from_json(json)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 2, 3, 4, 5>{}.from_json(json)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 4, 3, 4, 5>{}.from_json(json)),
        std::runtime_error);
  }

  {
    iganet::NonUniformBSpline<real_t, 4, 3, 4, 5> bspline_out(
        {4, 5, 6}, iganet::init::zeros, options);

    bspline_out.transform([](const std::array<real_t, 3> xi) {
      return std::array<real_t, 4>{
          static_cast<real_t>(std::rand()), static_cast<real_t>(std::rand()),
          static_cast<real_t>(std::rand()), static_cast<real_t>(std::rand())};
    });

    nlohmann::json json = bspline_out.to_json();

    iganet::NonUniformBSpline<real_t, 4, 3, 4, 5> bspline_in(options);
    bspline_in.from_json(json);

    EXPECT_TRUE(bspline_in == bspline_out);

    // non-matching degree
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 4, 3, 3, 3>{}.from_json(json)),
        std::runtime_error);

    // non-matching parametric dimension
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 4, 3>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 4, 3, 4>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 4, 3, 4, 5, 3>{}.from_json(json)),
        std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 1, 3, 4, 5>{}.from_json(json)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 2, 3, 4, 5>{}.from_json(json)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 3, 3, 4, 5>{}.from_json(json)),
        std::runtime_error);
  }

  {
    iganet::NonUniformBSpline<real_t, 1, 3, 4, 5, 1> bspline_out(
        {4, 5, 6, 2}, iganet::init::zeros, options);

    bspline_out.transform([](const std::array<real_t, 4> xi) {
      return std::array<real_t, 1>{static_cast<real_t>(std::rand())};
    });

    nlohmann::json json = bspline_out.to_json();

    iganet::NonUniformBSpline<real_t, 1, 3, 4, 5, 1> bspline_in(options);
    bspline_in.from_json(json);

    EXPECT_TRUE(bspline_in == bspline_out);

    // non-matching degree
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 1, 3, 4, 5, 2>{}.from_json(json)),
        std::runtime_error);

    // non-matching parametric dimension
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 1, 3>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 1, 3, 4>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 1, 3, 4, 5>{}.from_json(json)),
        std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 2, 3, 4, 5, 1>{}.from_json(json)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 3, 3, 4, 5, 1>{}.from_json(json)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 4, 3, 4, 5, 1>{}.from_json(json)),
        std::runtime_error);
  }

  {
    iganet::NonUniformBSpline<real_t, 2, 3, 4, 5, 1> bspline_out(
        {4, 5, 6, 2}, iganet::init::zeros, options);

    bspline_out.transform([](const std::array<real_t, 4> xi) {
      return std::array<real_t, 2>{static_cast<real_t>(std::rand()),
                                   static_cast<real_t>(std::rand())};
    });

    nlohmann::json json = bspline_out.to_json();

    iganet::NonUniformBSpline<real_t, 2, 3, 4, 5, 1> bspline_in(options);
    bspline_in.from_json(json);

    EXPECT_TRUE(bspline_in == bspline_out);

    // non-matching degree
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 2, 3, 4, 5, 2>{}.from_json(json)),
        std::runtime_error);

    // non-matching parametric dimension
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 2, 3>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 2, 3, 4>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 2, 3, 4, 5>{}.from_json(json)),
        std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 1, 3, 4, 5, 1>{}.from_json(json)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 3, 3, 4, 5, 1>{}.from_json(json)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 4, 3, 4, 5, 1>{}.from_json(json)),
        std::runtime_error);
  }

  {
    iganet::NonUniformBSpline<real_t, 3, 3, 4, 5, 1> bspline_out(
        {4, 5, 6, 2}, iganet::init::zeros, options);

    bspline_out.transform([](const std::array<real_t, 4> xi) {
      return std::array<real_t, 3>{static_cast<real_t>(std::rand()),
                                   static_cast<real_t>(std::rand()),
                                   static_cast<real_t>(std::rand())};
    });

    nlohmann::json json = bspline_out.to_json();

    iganet::NonUniformBSpline<real_t, 3, 3, 4, 5, 1> bspline_in(options);
    bspline_in.from_json(json);

    EXPECT_TRUE(bspline_in == bspline_out);

    // non-matching degree
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 3, 3, 4, 5, 2>{}.from_json(json)),
        std::runtime_error);

    // non-matching parametric dimension
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 3, 3>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 3, 3, 4>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 3, 3, 4, 5>{}.from_json(json)),
        std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 1, 3, 4, 5, 1>{}.from_json(json)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 2, 3, 4, 5, 1>{}.from_json(json)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 4, 3, 4, 5, 1>{}.from_json(json)),
        std::runtime_error);
  }

  {
    iganet::NonUniformBSpline<real_t, 4, 3, 4, 5, 1> bspline_out(
        {4, 5, 6, 2}, iganet::init::zeros, options);

    bspline_out.transform([](const std::array<real_t, 4> xi) {
      return std::array<real_t, 4>{
          static_cast<real_t>(std::rand()), static_cast<real_t>(std::rand()),
          static_cast<real_t>(std::rand()), static_cast<real_t>(std::rand())};
    });

    nlohmann::json json = bspline_out.to_json();

    iganet::NonUniformBSpline<real_t, 4, 3, 4, 5, 1> bspline_in(options);
    bspline_in.from_json(json);

    EXPECT_TRUE(bspline_in == bspline_out);

    // non-matching degree
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 4, 3, 4, 5, 2>{}.from_json(json)),
        std::runtime_error);

    // non-matching parametric dimension
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 4, 3>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW((iganet::NonUniformBSpline<real_t, 4, 3, 4>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 4, 3, 4, 5>{}.from_json(json)),
        std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 1, 3, 4, 5, 1>{}.from_json(json)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 2, 3, 4, 5, 1>{}.from_json(json)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::NonUniformBSpline<real_t, 3, 3, 4, 5, 1>{}.from_json(json)),
        std::runtime_error);
  }
}

TEST_F(BSplineTest, NonUniformBSpline_query_property) {
  iganet::NonUniformBSpline<real_t, 2, 3, 4> bspline(
      {4, 5}, iganet::init::greville, options);

  EXPECT_FALSE(bspline.is_uniform());
  EXPECT_TRUE(bspline.is_nonuniform());

  EXPECT_EQ(bspline.device(), options.device());
  EXPECT_EQ(bspline.device_index(), options.device_index());
  EXPECT_EQ(bspline.dtype(), options.dtype());
  EXPECT_EQ(bspline.is_sparse(), options.is_sparse());
  EXPECT_EQ(bspline.layout(), options.layout());
  EXPECT_EQ(bspline.pinned_memory(), options.pinned_memory());
}

TEST_F(BSplineTest, NonUniformBSpline_requires_grad) {
  {
    iganet::NonUniformBSpline<real_t, 2, 3, 4> bspline(
        {4, 5}, iganet::init::greville, options);

    EXPECT_EQ(bspline.requires_grad(), false);

    for (iganet::short_t i = 0; i < bspline.parDim(); ++i)
      EXPECT_EQ(bspline.knots(i).requires_grad(), false);

    for (iganet::short_t i = 0; i < bspline.geoDim(); ++i)
      EXPECT_EQ(bspline.coeffs(i).requires_grad(), false);

    auto xi = iganet::utils::to_tensorArray(options, {0.5_r}, {0.5_r});
    auto values = bspline.eval(xi);

    // We expect an error when calling backward() because no tensor
    // has requires_grad = true
    EXPECT_THROW(values[0]->backward(), c10::Error);

    xi = iganet::utils::to_tensorArray(options.requires_grad(true), {0.5_r},
                                       {0.5_r});
    values = bspline.eval(xi);
    values[0]->backward();
    EXPECT_TRUE(torch::allclose(xi[0].grad(),
                                iganet::utils::to_tensor({1.0_r}, options)));
  }

  {
    iganet::NonUniformBSpline<real_t, 2, 3, 4> bspline(
        {4, 5}, iganet::init::linear, options.requires_grad(true));

    EXPECT_TRUE(bspline.requires_grad());

    for (iganet::short_t i = 0; i < bspline.parDim(); ++i)
      EXPECT_TRUE(bspline.knots(i).requires_grad());

    for (iganet::short_t i = 0; i < bspline.geoDim(); ++i)
      EXPECT_TRUE(bspline.coeffs(i).requires_grad());

    auto xi = iganet::utils::to_tensorArray(options, {0.5_r}, {0.5_r});
    auto values = bspline.eval(xi);
    values[0]->backward(
        {}, true); // otherwise we cannot run backward() a second time

    // We expect an error because xi[0].grad() is an undefined tensor
    EXPECT_THROW(torch::allclose(xi[0].grad(), torch::empty({})), c10::Error);

    xi = iganet::utils::to_tensorArray(options.requires_grad(true), {0.5_r},
                                       {0.5_r});
    values = bspline.eval(xi);
    values[0]->backward();
    EXPECT_TRUE(torch::allclose(xi[0].grad(),
                                iganet::utils::to_tensor({1.0_r}, options)));

    EXPECT_TRUE(torch::allclose(
        bspline.coeffs(0).grad(),
        iganet::utils::to_tensor(
            {0.015625_r, 0.046875_r, 0.046875_r, 0.015625_r, 0.0625_r,
             0.1875_r,   0.1875_r,   0.0625_r,   0.09375_r,  0.28125_r,
             0.28125_r,  0.09375_r,  0.0625_r,   0.1875_r,   0.1875_r,
             0.0625_r,   0.015625_r, 0.046875_r, 0.046875_r, 0.015625_r},
            options)));
  }
}

TEST_F(BSplineTest, NonUniformBSpline_to_dtype) {
  {
    iganet::NonUniformBSpline<real_t, 2, 3, 4> bspline(
        {4, 5}, iganet::init::greville, options);

    auto bspline_double = bspline.to<double>();
    auto bspline_float = bspline.to<float>();

    if constexpr (std::is_same<real_t, double>::value)
      EXPECT_TRUE(bspline == bspline_double);
    else
      EXPECT_TRUE(bspline != bspline_double);

    if constexpr (std::is_same<real_t, float>::value)
      EXPECT_TRUE(bspline == bspline_float);
    else
      EXPECT_TRUE(bspline != bspline_float);
  }

  {
    iganet::NonUniformBSpline<real_t, 2, 3, 4> bspline(
        {4, 5}, iganet::init::greville, options);

    auto bspline_double = bspline.to(iganet::Options<double>{});
    auto bspline_float = bspline.to(iganet::Options<float>{});

    if constexpr (std::is_same<real_t, double>::value)
      EXPECT_TRUE(bspline == bspline_double);
    else
      EXPECT_TRUE(bspline != bspline_double);

    if constexpr (std::is_same<real_t, float>::value)
      EXPECT_TRUE(bspline == bspline_float);
    else
      EXPECT_TRUE(bspline != bspline_float);
  }
}

TEST_F(BSplineTest, NonUniformBSpline_to_device) {
  {
    iganet::Options<real_t> options =
        iganet::Options<real_t>{}.device(torch::kCPU);
    iganet::NonUniformBSpline<real_t, 2, 3, 4> bspline(
        {4, 5}, iganet::init::greville, options);

    auto bspline_cpu = bspline.to(torch::kCPU);
    EXPECT_TRUE(bspline == bspline_cpu);

    if (torch::cuda::is_available()) {
      auto bspline_cuda = bspline.to(torch::kCUDA);
      EXPECT_THROW((void)(bspline == bspline_cuda), c10::Error);
    } else
      EXPECT_THROW(bspline.to(torch::kCUDA), c10::Error);

    if (at::hasHIP()) {
      auto bspline_hip = bspline.to(torch::kHIP);
      EXPECT_THROW((void)(bspline == bspline_hip), c10::Error);
    } else
      EXPECT_THROW(bspline.to(torch::kHIP), c10::Error);

    if (at::hasMPS() && // will become torch::mps::is_available()
        (options.dtype() != iganet::dtype<double>())) {
      auto bspline_mps = bspline.to(torch::kMPS);
      EXPECT_THROW((void)(bspline == bspline_mps), c10::Error);
    } else
      EXPECT_THROW(bspline.to(torch::kMPS), c10::Error);
  }
}

TEST_F(BSplineTest, NonUniformBSpline_reduce_continuity) {
  {
    iganet::NonUniformBSpline<real_t, 3, 3, 4> bspline({5, 6});
    iganet::NonUniformBSpline<real_t, 3, 3, 4> bspline_ref(
        iganet::utils::to_array(
            iganet::utils::to_vector(0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 0.5_r,
                                     1.0_r, 1.0_r, 1.0_r, 1.0_r),
            iganet::utils::to_vector(0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r,
                                     0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r,
                                     1.0_r)));
    bspline.reduce_continuity();

    EXPECT_TRUE(bspline.isclose(bspline_ref));
  }

  {
    iganet::NonUniformBSpline<real_t, 3, 3, 4> bspline({5, 6});
    iganet::NonUniformBSpline<real_t, 3, 3, 4> bspline_ref(
        iganet::utils::to_array(
            iganet::utils::to_vector(0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 0.5_r,
                                     0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r),
            iganet::utils::to_vector(0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r,
                                     0.5_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r,
                                     1.0_r)));
    bspline.reduce_continuity(2);

    EXPECT_TRUE(bspline.isclose(bspline_ref));
  }

  {
    iganet::NonUniformBSpline<real_t, 3, 3, 4> bspline({5, 6});
    iganet::NonUniformBSpline<real_t, 3, 3, 4> bspline_ref(
        iganet::utils::to_array(
            iganet::utils::to_vector(0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r, 0.5_r,
                                     1.0_r, 1.0_r, 1.0_r, 1.0_r),
            iganet::utils::to_vector(0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.5_r,
                                     0.5_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r,
                                     1.0_r)));
    bspline.reduce_continuity(1, 0).reduce_continuity(2, 1);

    EXPECT_TRUE(bspline.isclose(bspline_ref));
  }
}

TEST_F(BSplineTest, NonUniformBSpline_insert_knots) {
  {
    iganet::NonUniformBSpline<real_t, 3, 3, 4> bspline({5, 6});
    iganet::NonUniformBSpline<real_t, 3, 3, 4> bspline_ref(
        iganet::utils::to_array(
            iganet::utils::to_vector(0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.1_r, 0.3_r,
                                     0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r),
            iganet::utils::to_vector(0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r, 0.2_r,
                                     0.4_r, 0.5_r, 1.0_r, 1.0_r, 1.0_r, 1.0_r,
                                     1.0_r)));
    bspline.insert_knots(
        iganet::utils::to_tensorArray({0.1_r, 0.3_r}, {0.2_r, 0.4_r}));

    EXPECT_TRUE(bspline.isclose(bspline_ref));
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  iganet::init();
  int result = RUN_ALL_TESTS();
  iganet::finalize();
  return result;
}

#pragma nv_diag_default 20208
