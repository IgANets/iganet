/**
   @file unittests/unittest_uniform_bspline.cxx

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

using namespace iganet::unittests::literals;

class BSplineTest : public ::testing::Test {
public:
  BSplineTest() { std::srand(std::time(nullptr)); }

protected:
  using real_t = iganet::unittests::real_t;
  iganet::Options<real_t> options;
};

TEST_F(BSplineTest, UniformBSpline_parDim1_geoDim1_degrees1) {
  for (iganet::short_t n0 = 0; n0 < 2; n0++)
    EXPECT_THROW((iganet::UniformBSpline<real_t, 1, 1>({n0})),
                 std::runtime_error);

  iganet::UniformBSpline<real_t, 1, 1> bspline({2});
  EXPECT_EQ(bspline.parDim(), 1);
  EXPECT_EQ(bspline.geoDim(), 1);
  EXPECT_EQ(bspline.degree(0), 1);
  EXPECT_EQ(bspline.nknots(0), 4);
  EXPECT_EQ(bspline.ncoeffs(0), 2);
  EXPECT_EQ(bspline.ncumcoeffs(), 2);
}

TEST_F(BSplineTest, UniformBSpline_parDim1_geoDim1_degrees2) {
  for (iganet::short_t n0 = 0; n0 < 3; n0++)
    EXPECT_THROW((iganet::UniformBSpline<real_t, 1, 2>({n0})),
                 std::runtime_error);

  iganet::UniformBSpline<real_t, 1, 2> bspline({3});
  EXPECT_EQ(bspline.parDim(), 1);
  EXPECT_EQ(bspline.geoDim(), 1);
  EXPECT_EQ(bspline.degree(0), 2);
  EXPECT_EQ(bspline.nknots(0), 6);
  EXPECT_EQ(bspline.ncoeffs(0), 3);
  EXPECT_EQ(bspline.ncumcoeffs(), 3);
}

TEST_F(BSplineTest, UniformBSpline_parDim1_geoDim1_degrees3) {
  for (iganet::short_t n0 = 0; n0 < 4; n0++)
    EXPECT_THROW((iganet::UniformBSpline<real_t, 1, 3>({n0})),
                 std::runtime_error);

  iganet::UniformBSpline<real_t, 1, 3> bspline({4});
  EXPECT_EQ(bspline.parDim(), 1);
  EXPECT_EQ(bspline.geoDim(), 1);
  EXPECT_EQ(bspline.degree(0), 3);
  EXPECT_EQ(bspline.nknots(0), 8);
  EXPECT_EQ(bspline.ncoeffs(0), 4);
  EXPECT_EQ(bspline.ncumcoeffs(), 4);
}

TEST_F(BSplineTest, UniformBSpline_parDim1_geoDim2_degrees4) {
  for (iganet::short_t n0 = 0; n0 < 5; n0++)
    EXPECT_THROW((iganet::UniformBSpline<real_t, 2, 4>({n0})),
                 std::runtime_error);

  iganet::UniformBSpline<real_t, 2, 4> bspline({5});
  EXPECT_EQ(bspline.parDim(), 1);
  EXPECT_EQ(bspline.geoDim(), 2);
  EXPECT_EQ(bspline.degree(0), 4);
  EXPECT_EQ(bspline.nknots(0), 10);
  EXPECT_EQ(bspline.ncoeffs(0), 5);
  EXPECT_EQ(bspline.ncumcoeffs(), 5);
}

TEST_F(BSplineTest, UniformBSpline_parDim1_geoDim3_degrees5) {
  for (iganet::short_t n0 = 0; n0 < 6; n0++)
    EXPECT_THROW((iganet::UniformBSpline<real_t, 3, 5>({n0})),
                 std::runtime_error);

  iganet::UniformBSpline<real_t, 3, 5> bspline({6});
  EXPECT_EQ(bspline.parDim(), 1);
  EXPECT_EQ(bspline.geoDim(), 3);
  EXPECT_EQ(bspline.degree(0), 5);
  EXPECT_EQ(bspline.nknots(0), 12);
  EXPECT_EQ(bspline.ncoeffs(0), 6);
  EXPECT_EQ(bspline.ncumcoeffs(), 6);
}

TEST_F(BSplineTest, UniformBSpline_parDim1_geoDim4_degrees6) {
  for (iganet::short_t n0 = 0; n0 < 7; n0++)
    EXPECT_THROW((iganet::UniformBSpline<real_t, 4, 6>({n0})),
                 std::runtime_error);

  iganet::UniformBSpline<real_t, 4, 6> bspline({7});
  EXPECT_EQ(bspline.parDim(), 1);
  EXPECT_EQ(bspline.geoDim(), 4);
  EXPECT_EQ(bspline.degree(0), 6);
  EXPECT_EQ(bspline.nknots(0), 14);
  EXPECT_EQ(bspline.ncoeffs(0), 7);
  EXPECT_EQ(bspline.ncumcoeffs(), 7);
}

TEST_F(BSplineTest, UniformBSpline_parDim2_geoDim1_degrees34) {
  for (iganet::short_t n0 = 0; n0 < 4; n0++)
    for (iganet::short_t n1 = 0; n1 < 5; n1++)
      EXPECT_THROW((iganet::UniformBSpline<real_t, 1, 3, 4>({n0, n1})),
                   std::runtime_error);

  iganet::UniformBSpline<real_t, 1, 3, 4> bspline({4, 5});
  EXPECT_EQ(bspline.parDim(), 2);
  EXPECT_EQ(bspline.geoDim(), 1);
  EXPECT_EQ(bspline.degree(0), 3);
  EXPECT_EQ(bspline.degree(1), 4);
  EXPECT_EQ(bspline.nknots(0), 8);
  EXPECT_EQ(bspline.nknots(1), 10);
  EXPECT_EQ(bspline.ncoeffs(0), 4);
  EXPECT_EQ(bspline.ncoeffs(1), 5);
  EXPECT_EQ(bspline.ncumcoeffs(), 20);
}

TEST_F(BSplineTest, UniformBSpline_parDim2_geoDim2_degrees34) {
  for (iganet::short_t n0 = 0; n0 < 4; n0++)
    for (iganet::short_t n1 = 0; n1 < 5; n1++)
      EXPECT_THROW((iganet::UniformBSpline<real_t, 2, 3, 4>({n0, n1})),
                   std::runtime_error);

  iganet::UniformBSpline<real_t, 2, 3, 4> bspline({4, 5});
  EXPECT_EQ(bspline.parDim(), 2);
  EXPECT_EQ(bspline.geoDim(), 2);
  EXPECT_EQ(bspline.degree(0), 3);
  EXPECT_EQ(bspline.degree(1), 4);
  EXPECT_EQ(bspline.nknots(0), 8);
  EXPECT_EQ(bspline.nknots(1), 10);
  EXPECT_EQ(bspline.ncoeffs(0), 4);
  EXPECT_EQ(bspline.ncoeffs(1), 5);
  EXPECT_EQ(bspline.ncumcoeffs(), 20);
}

TEST_F(BSplineTest, UniformBSpline_parDim2_geoDim3_degrees34) {
  for (iganet::short_t n0 = 0; n0 < 4; n0++)
    for (iganet::short_t n1 = 0; n1 < 5; n1++)
      EXPECT_THROW((iganet::UniformBSpline<real_t, 3, 3, 4>({n0, n1})),
                   std::runtime_error);

  iganet::UniformBSpline<real_t, 3, 3, 4> bspline({4, 5});
  EXPECT_EQ(bspline.parDim(), 2);
  EXPECT_EQ(bspline.geoDim(), 3);
  EXPECT_EQ(bspline.degree(0), 3);
  EXPECT_EQ(bspline.degree(1), 4);
  EXPECT_EQ(bspline.nknots(0), 8);
  EXPECT_EQ(bspline.nknots(1), 10);
  EXPECT_EQ(bspline.ncoeffs(0), 4);
  EXPECT_EQ(bspline.ncoeffs(1), 5);
  EXPECT_EQ(bspline.ncumcoeffs(), 20);
}

TEST_F(BSplineTest, UniformBSpline_parDim2_geoDim4_degrees34) {
  for (iganet::short_t n0 = 0; n0 < 4; n0++)
    for (iganet::short_t n1 = 0; n1 < 5; n1++)
      EXPECT_THROW((iganet::UniformBSpline<real_t, 4, 3, 4>({n0, n1})),
                   std::runtime_error);

  iganet::UniformBSpline<real_t, 4, 3, 4> bspline({4, 5});
  EXPECT_EQ(bspline.parDim(), 2);
  EXPECT_EQ(bspline.geoDim(), 4);
  EXPECT_EQ(bspline.degree(0), 3);
  EXPECT_EQ(bspline.degree(1), 4);
  EXPECT_EQ(bspline.nknots(0), 8);
  EXPECT_EQ(bspline.nknots(1), 10);
  EXPECT_EQ(bspline.ncoeffs(0), 4);
  EXPECT_EQ(bspline.ncoeffs(1), 5);
  EXPECT_EQ(bspline.ncumcoeffs(), 20);
}

TEST_F(BSplineTest, UniformBSpline_parDim3_geoDim1_degrees342) {
  for (iganet::short_t n0 = 0; n0 < 4; n0++)
    for (iganet::short_t n1 = 0; n1 < 5; n1++)
      for (iganet::short_t n2 = 0; n2 < 3; n2++)
        EXPECT_THROW((iganet::UniformBSpline<real_t, 1, 3, 4, 2>({n0, n1, n2})),
                     std::runtime_error);

  iganet::UniformBSpline<real_t, 1, 3, 4, 2> bspline({4, 5, 3});
  EXPECT_EQ(bspline.parDim(), 3);
  EXPECT_EQ(bspline.geoDim(), 1);
  EXPECT_EQ(bspline.degree(0), 3);
  EXPECT_EQ(bspline.degree(1), 4);
  EXPECT_EQ(bspline.degree(2), 2);
  EXPECT_EQ(bspline.nknots(0), 8);
  EXPECT_EQ(bspline.nknots(1), 10);
  EXPECT_EQ(bspline.nknots(2), 6);
  EXPECT_EQ(bspline.ncoeffs(0), 4);
  EXPECT_EQ(bspline.ncoeffs(1), 5);
  EXPECT_EQ(bspline.ncoeffs(2), 3);
  EXPECT_EQ(bspline.ncumcoeffs(), 60);
}

TEST_F(BSplineTest, UniformBSpline_parDim2_geoDim3_degrees342) {
  for (iganet::short_t n0 = 0; n0 < 4; n0++)
    for (iganet::short_t n1 = 0; n1 < 5; n1++)
      for (iganet::short_t n2 = 0; n2 < 3; n2++)
        EXPECT_THROW((iganet::UniformBSpline<real_t, 2, 3, 4, 2>({n0, n1, n2})),
                     std::runtime_error);

  iganet::UniformBSpline<real_t, 2, 3, 4, 2> bspline({4, 5, 3});
  EXPECT_EQ(bspline.parDim(), 3);
  EXPECT_EQ(bspline.geoDim(), 2);
  EXPECT_EQ(bspline.degree(0), 3);
  EXPECT_EQ(bspline.degree(1), 4);
  EXPECT_EQ(bspline.degree(2), 2);
  EXPECT_EQ(bspline.nknots(0), 8);
  EXPECT_EQ(bspline.nknots(1), 10);
  EXPECT_EQ(bspline.nknots(2), 6);
  EXPECT_EQ(bspline.ncoeffs(0), 4);
  EXPECT_EQ(bspline.ncoeffs(1), 5);
  EXPECT_EQ(bspline.ncoeffs(2), 3);
  EXPECT_EQ(bspline.ncumcoeffs(), 60);
}

TEST_F(BSplineTest, UniformBSpline_parDim3_geoDim3_degrees342) {
  for (iganet::short_t n0 = 0; n0 < 4; n0++)
    for (iganet::short_t n1 = 0; n1 < 5; n1++)
      for (iganet::short_t n2 = 0; n2 < 3; n2++)
        EXPECT_THROW((iganet::UniformBSpline<real_t, 3, 3, 4, 2>({n0, n1, n2})),
                     std::runtime_error);

  iganet::UniformBSpline<real_t, 3, 3, 4, 2> bspline({4, 5, 3});
  EXPECT_EQ(bspline.parDim(), 3);
  EXPECT_EQ(bspline.geoDim(), 3);
  EXPECT_EQ(bspline.degree(0), 3);
  EXPECT_EQ(bspline.degree(1), 4);
  EXPECT_EQ(bspline.degree(2), 2);
  EXPECT_EQ(bspline.nknots(0), 8);
  EXPECT_EQ(bspline.nknots(1), 10);
  EXPECT_EQ(bspline.nknots(2), 6);
  EXPECT_EQ(bspline.ncoeffs(0), 4);
  EXPECT_EQ(bspline.ncoeffs(1), 5);
  EXPECT_EQ(bspline.ncoeffs(2), 3);
  EXPECT_EQ(bspline.ncumcoeffs(), 60);
}

TEST_F(BSplineTest, UniformBSpline_parDim3_geoDim4_degrees342) {
  for (iganet::short_t n0 = 0; n0 < 4; n0++)
    for (iganet::short_t n1 = 0; n1 < 5; n1++)
      for (iganet::short_t n2 = 0; n2 < 3; n2++)
        EXPECT_THROW((iganet::UniformBSpline<real_t, 4, 3, 4, 2>({n0, n1, n2})),
                     std::runtime_error);

  iganet::UniformBSpline<real_t, 4, 3, 4, 2> bspline({4, 5, 3});
  EXPECT_EQ(bspline.parDim(), 3);
  EXPECT_EQ(bspline.geoDim(), 4);
  EXPECT_EQ(bspline.degree(0), 3);
  EXPECT_EQ(bspline.degree(1), 4);
  EXPECT_EQ(bspline.degree(2), 2);
  EXPECT_EQ(bspline.nknots(0), 8);
  EXPECT_EQ(bspline.nknots(1), 10);
  EXPECT_EQ(bspline.nknots(2), 6);
  EXPECT_EQ(bspline.ncoeffs(0), 4);
  EXPECT_EQ(bspline.ncoeffs(1), 5);
  EXPECT_EQ(bspline.ncoeffs(2), 3);
  EXPECT_EQ(bspline.ncumcoeffs(), 60);
}

TEST_F(BSplineTest, UniformBSpline_parDim4_geoDim1_degrees3421) {
  for (iganet::short_t n0 = 0; n0 < 4; n0++)
    for (iganet::short_t n1 = 0; n1 < 5; n1++)
      for (iganet::short_t n2 = 0; n2 < 3; n2++)
        for (iganet::short_t n3 = 0; n3 < 2; n3++)
          EXPECT_THROW(
              (iganet::UniformBSpline<real_t, 1, 3, 4, 2, 1>({n0, n1, n2, n3})),
              std::runtime_error);

  iganet::UniformBSpline<real_t, 1, 3, 4, 2, 1> bspline({4, 5, 3, 2});
  EXPECT_EQ(bspline.parDim(), 4);
  EXPECT_EQ(bspline.geoDim(), 1);
  EXPECT_EQ(bspline.degree(0), 3);
  EXPECT_EQ(bspline.degree(1), 4);
  EXPECT_EQ(bspline.degree(2), 2);
  EXPECT_EQ(bspline.degree(3), 1);
  EXPECT_EQ(bspline.nknots(0), 8);
  EXPECT_EQ(bspline.nknots(1), 10);
  EXPECT_EQ(bspline.nknots(2), 6);
  EXPECT_EQ(bspline.nknots(3), 4);
  EXPECT_EQ(bspline.ncoeffs(0), 4);
  EXPECT_EQ(bspline.ncoeffs(1), 5);
  EXPECT_EQ(bspline.ncoeffs(2), 3);
  EXPECT_EQ(bspline.ncoeffs(3), 2);
  EXPECT_EQ(bspline.ncumcoeffs(), 120);
}

TEST_F(BSplineTest, UniformBSpline_parDim4_geoDim2_degrees3421) {
  for (iganet::short_t n0 = 0; n0 < 4; n0++)
    for (iganet::short_t n1 = 0; n1 < 5; n1++)
      for (iganet::short_t n2 = 0; n2 < 3; n2++)
        for (iganet::short_t n3 = 0; n3 < 2; n3++)
          EXPECT_THROW(
              (iganet::UniformBSpline<real_t, 2, 3, 4, 2, 1>({n0, n1, n2, n3})),
              std::runtime_error);

  iganet::UniformBSpline<real_t, 2, 3, 4, 2, 1> bspline({4, 5, 3, 2});
  EXPECT_EQ(bspline.parDim(), 4);
  EXPECT_EQ(bspline.geoDim(), 2);
  EXPECT_EQ(bspline.degree(0), 3);
  EXPECT_EQ(bspline.degree(1), 4);
  EXPECT_EQ(bspline.degree(2), 2);
  EXPECT_EQ(bspline.degree(3), 1);
  EXPECT_EQ(bspline.nknots(0), 8);
  EXPECT_EQ(bspline.nknots(1), 10);
  EXPECT_EQ(bspline.nknots(2), 6);
  EXPECT_EQ(bspline.nknots(3), 4);
  EXPECT_EQ(bspline.ncoeffs(0), 4);
  EXPECT_EQ(bspline.ncoeffs(1), 5);
  EXPECT_EQ(bspline.ncoeffs(2), 3);
  EXPECT_EQ(bspline.ncoeffs(3), 2);
  EXPECT_EQ(bspline.ncumcoeffs(), 120);
}

TEST_F(BSplineTest, UniformBSpline_parDim4_geoDim3_degrees3421) {
  for (iganet::short_t n0 = 0; n0 < 4; n0++)
    for (iganet::short_t n1 = 0; n1 < 5; n1++)
      for (iganet::short_t n2 = 0; n2 < 3; n2++)
        for (iganet::short_t n3 = 0; n3 < 2; n3++)
          EXPECT_THROW(
              (iganet::UniformBSpline<real_t, 3, 3, 4, 2, 1>({n0, n1, n2, n3})),
              std::runtime_error);

  iganet::UniformBSpline<real_t, 3, 3, 4, 2, 1> bspline({4, 5, 3, 2});
  EXPECT_EQ(bspline.parDim(), 4);
  EXPECT_EQ(bspline.geoDim(), 3);
  EXPECT_EQ(bspline.degree(0), 3);
  EXPECT_EQ(bspline.degree(1), 4);
  EXPECT_EQ(bspline.degree(2), 2);
  EXPECT_EQ(bspline.degree(3), 1);
  EXPECT_EQ(bspline.nknots(0), 8);
  EXPECT_EQ(bspline.nknots(1), 10);
  EXPECT_EQ(bspline.nknots(2), 6);
  EXPECT_EQ(bspline.nknots(3), 4);
  EXPECT_EQ(bspline.ncoeffs(0), 4);
  EXPECT_EQ(bspline.ncoeffs(1), 5);
  EXPECT_EQ(bspline.ncoeffs(2), 3);
  EXPECT_EQ(bspline.ncoeffs(3), 2);
  EXPECT_EQ(bspline.ncumcoeffs(), 120);
}

TEST_F(BSplineTest, UniformBSpline_parDim4_geoDim4_degrees3421) {
  for (iganet::short_t n0 = 0; n0 < 4; n0++)
    for (iganet::short_t n1 = 0; n1 < 5; n1++)
      for (iganet::short_t n2 = 0; n2 < 3; n2++)
        for (iganet::short_t n3 = 0; n3 < 2; n3++)
          EXPECT_THROW(
              (iganet::UniformBSpline<real_t, 4, 3, 4, 2, 1>({n0, n1, n2, n3})),
              std::runtime_error);

  iganet::UniformBSpline<real_t, 4, 3, 4, 2, 1> bspline({4, 5, 3, 2});
  EXPECT_EQ(bspline.parDim(), 4);
  EXPECT_EQ(bspline.geoDim(), 4);
  EXPECT_EQ(bspline.degree(0), 3);
  EXPECT_EQ(bspline.degree(1), 4);
  EXPECT_EQ(bspline.degree(2), 2);
  EXPECT_EQ(bspline.degree(3), 1);
  EXPECT_EQ(bspline.nknots(0), 8);
  EXPECT_EQ(bspline.nknots(1), 10);
  EXPECT_EQ(bspline.nknots(2), 6);
  EXPECT_EQ(bspline.nknots(3), 4);
  EXPECT_EQ(bspline.ncoeffs(0), 4);
  EXPECT_EQ(bspline.ncoeffs(1), 5);
  EXPECT_EQ(bspline.ncoeffs(2), 3);
  EXPECT_EQ(bspline.ncoeffs(3), 2);
  EXPECT_EQ(bspline.ncumcoeffs(), 120);
}

TEST_F(BSplineTest, UniformBSpline_init) {
  {
    iganet::UniformBSpline<real_t, 1, 1> bspline({5}, iganet::init::zeros,
                                                 options);
    EXPECT_TRUE(torch::equal(bspline.coeffs(0), torch::zeros(5, options)));
  }

  {
    iganet::UniformBSpline<real_t, 1, 1> bspline({5}, iganet::init::ones,
                                                 options);
    EXPECT_TRUE(torch::equal(bspline.coeffs(0), torch::ones(5, options)));
  }

  {
    iganet::UniformBSpline<real_t, 1, 1> bspline({5}, iganet::init::linear,
                                                 options);
    EXPECT_TRUE(
        torch::equal(bspline.coeffs(0), torch::linspace(0, 1, 5, options)));
  }

  {
    iganet::UniformBSpline<real_t, 1, 1> bspline({5}, iganet::init::greville,
                                                 options);
    EXPECT_TRUE(
        torch::equal(bspline.coeffs(0), torch::linspace(0, 1, 5, options)));
  }

  {
    iganet::UniformBSpline<real_t, 2, 1> bspline({5}, iganet::init::zeros,
                                                 options);
    EXPECT_TRUE(torch::equal(bspline.coeffs(0), torch::zeros(5, options)));
    EXPECT_TRUE(torch::equal(bspline.coeffs(1), torch::zeros(5, options)));
  }

  {
    iganet::UniformBSpline<real_t, 2, 1> bspline({5}, iganet::init::ones,
                                                 options);
    EXPECT_TRUE(torch::equal(bspline.coeffs(0), torch::ones(5, options)));
    EXPECT_TRUE(torch::equal(bspline.coeffs(1), torch::ones(5, options)));
  }

  {
    iganet::UniformBSpline<real_t, 2, 1> bspline({5}, iganet::init::linear,
                                                 options);
    EXPECT_TRUE(
        torch::equal(bspline.coeffs(0), torch::linspace(0, 1, 5, options)));
    EXPECT_TRUE(torch::equal(bspline.coeffs(1), torch::ones(5, options)));
  }

  {
    iganet::UniformBSpline<real_t, 2, 1> bspline({5}, iganet::init::greville,
                                                 options);
    EXPECT_TRUE(
        torch::equal(bspline.coeffs(0), torch::linspace(0, 1, 5, options)));
    EXPECT_TRUE(torch::equal(bspline.coeffs(1), torch::ones(5, options)));
  }

  {
    iganet::UniformBSpline<real_t, 2, 2, 2> bspline({5, 8}, iganet::init::zeros,
                                                    options);
    EXPECT_TRUE(torch::equal(bspline.coeffs(0), torch::zeros(40, options)));
    EXPECT_TRUE(torch::equal(bspline.coeffs(1), torch::zeros(40, options)));
  }

  {
    iganet::UniformBSpline<real_t, 2, 2, 2> bspline({5, 8}, iganet::init::ones,
                                                    options);
    EXPECT_TRUE(torch::equal(bspline.coeffs(0), torch::ones(40, options)));
    EXPECT_TRUE(torch::equal(bspline.coeffs(1), torch::ones(40, options)));
  }

  {
    iganet::UniformBSpline<real_t, 2, 2, 2> bspline(
        {5, 8}, iganet::init::linear, options);
    EXPECT_TRUE(torch::equal(bspline.coeffs(0),
                             torch::linspace(0, 1, 5, options).repeat(8)));
    EXPECT_TRUE(
        torch::equal(bspline.coeffs(1),
                     torch::linspace(0, 1, 8, options).repeat_interleave(5)));
  }

  {
    iganet::UniformBSpline<real_t, 2, 1, 1> bspline(
        {5, 8}, iganet::init::greville, options);
    EXPECT_TRUE(torch::equal(bspline.coeffs(0),
                             torch::linspace(0, 1, 5, options).repeat(8)));
    EXPECT_TRUE(torch::allclose(
        bspline.coeffs(1),
        torch::linspace(0, 1, 8, options).repeat_interleave(5)));
  }

  {
    iganet::UniformBSpline<real_t, 3, 2, 2> bspline({5, 8}, iganet::init::zeros,
                                                    options);
    EXPECT_TRUE(torch::equal(bspline.coeffs(0), torch::zeros(40, options)));
    EXPECT_TRUE(torch::equal(bspline.coeffs(1), torch::zeros(40, options)));
    EXPECT_TRUE(torch::equal(bspline.coeffs(2), torch::zeros(40, options)));
  }

  {
    iganet::UniformBSpline<real_t, 3, 2, 2> bspline({5, 8}, iganet::init::ones,
                                                    options);
    EXPECT_TRUE(torch::equal(bspline.coeffs(0), torch::ones(40, options)));
    EXPECT_TRUE(torch::equal(bspline.coeffs(1), torch::ones(40, options)));
    EXPECT_TRUE(torch::equal(bspline.coeffs(2), torch::ones(40, options)));
  }

  {
    iganet::UniformBSpline<real_t, 3, 2, 2> bspline(
        {5, 8}, iganet::init::linear, options);
    EXPECT_TRUE(torch::equal(bspline.coeffs(0),
                             torch::linspace(0, 1, 5, options).repeat(8)));
    EXPECT_TRUE(
        torch::equal(bspline.coeffs(1),
                     torch::linspace(0, 1, 8, options).repeat_interleave(5)));
    EXPECT_TRUE(torch::equal(bspline.coeffs(2), torch::ones(40, options)));
  }

  {
    iganet::UniformBSpline<real_t, 3, 1, 1> bspline(
        {5, 8}, iganet::init::greville, options);
    EXPECT_TRUE(torch::equal(bspline.coeffs(0),
                             torch::linspace(0, 1, 5, options).repeat(8)));
    EXPECT_TRUE(torch::allclose(
        bspline.coeffs(1),
        torch::linspace(0, 1, 8, options).repeat_interleave(5)));
    EXPECT_TRUE(torch::equal(bspline.coeffs(2), torch::ones(40, options)));
  }

  {
    iganet::UniformBSpline<real_t, 4, 2, 2> bspline({5, 8}, iganet::init::zeros,
                                                    options);
    EXPECT_TRUE(torch::equal(bspline.coeffs(0), torch::zeros(40, options)));
    EXPECT_TRUE(torch::equal(bspline.coeffs(1), torch::zeros(40, options)));
    EXPECT_TRUE(torch::equal(bspline.coeffs(2), torch::zeros(40, options)));
    EXPECT_TRUE(torch::equal(bspline.coeffs(3), torch::zeros(40, options)));
  }

  {
    iganet::UniformBSpline<real_t, 4, 2, 2> bspline({5, 8}, iganet::init::ones,
                                                    options);
    EXPECT_TRUE(torch::equal(bspline.coeffs(0), torch::ones(40, options)));
    EXPECT_TRUE(torch::equal(bspline.coeffs(1), torch::ones(40, options)));
    EXPECT_TRUE(torch::equal(bspline.coeffs(2), torch::ones(40, options)));
    EXPECT_TRUE(torch::equal(bspline.coeffs(3), torch::ones(40, options)));
  }

  {
    iganet::UniformBSpline<real_t, 4, 2, 2> bspline(
        {5, 8}, iganet::init::linear, options);
    EXPECT_TRUE(torch::equal(bspline.coeffs(0),
                             torch::linspace(0, 1, 5, options).repeat(8)));
    EXPECT_TRUE(
        torch::equal(bspline.coeffs(1),
                     torch::linspace(0, 1, 8, options).repeat_interleave(5)));
    EXPECT_TRUE(torch::equal(bspline.coeffs(2), torch::ones(40, options)));
    EXPECT_TRUE(torch::equal(bspline.coeffs(3), torch::ones(40, options)));
  }

  {
    iganet::UniformBSpline<real_t, 4, 1, 1> bspline(
        {5, 8}, iganet::init::greville, options);
    EXPECT_TRUE(torch::equal(bspline.coeffs(0),
                             torch::linspace(0, 1, 5, options).repeat(8)));
    EXPECT_TRUE(torch::allclose(
        bspline.coeffs(1),
        torch::linspace(0, 1, 8, options).repeat_interleave(5)));
    EXPECT_TRUE(torch::equal(bspline.coeffs(2), torch::ones(40, options)));
    EXPECT_TRUE(torch::equal(bspline.coeffs(3), torch::ones(40, options)));
  }
}

TEST_F(BSplineTest, UniformBSpline_uniform_refine) {
  {
    iganet::UniformBSpline<real_t, 3, 3, 4> bspline({4, 5});
    iganet::UniformBSpline<real_t, 3, 3, 4> bspline_ref({5, 6});
    bspline.uniform_refine();

    EXPECT_TRUE(bspline.isclose(bspline_ref));
  }

  {
    iganet::UniformBSpline<real_t, 3, 3, 4> bspline({4, 5});
    iganet::UniformBSpline<real_t, 3, 3, 4> bspline_ref({7, 8});
    bspline.uniform_refine(2);

    EXPECT_TRUE(bspline.isclose(bspline_ref));
  }

  {
    iganet::UniformBSpline<real_t, 3, 3, 4> bspline({4, 5});
    iganet::UniformBSpline<real_t, 3, 3, 4> bspline_ref({5, 5});
    bspline.uniform_refine(1, 0);

    EXPECT_TRUE(bspline.isclose(bspline_ref));
  }

  {
    iganet::UniformBSpline<real_t, 3, 3, 4> bspline({4, 5});
    iganet::UniformBSpline<real_t, 3, 3, 4> bspline_ref({5, 8});
    bspline.uniform_refine(1, 0).uniform_refine(2, 1);

    EXPECT_TRUE(bspline.isclose(bspline_ref));
  }
}

TEST_F(BSplineTest, UniformBSpline_copy_constructor) {
  iganet::UniformBSpline<real_t, 3, 3, 4> bspline_orig(
      {4, 5}, iganet::init::greville, options);
  iganet::UniformBSpline<real_t, 3, 3, 4> bspline_copy(bspline_orig);

  bspline_orig.transform([](const std::array<real_t, 2> xi) {
    return std::array<real_t, 3>{0.0_r, 1.0_r, 2.0_r};
  });

  EXPECT_TRUE(bspline_orig == bspline_copy);
}

TEST_F(BSplineTest, UniformBSpline_clone_constructor) {
  iganet::UniformBSpline<real_t, 3, 3, 4> bspline_ref(
      {4, 5}, iganet::init::greville, options);
  iganet::UniformBSpline<real_t, 3, 3, 4> bspline_orig(
      {4, 5}, iganet::init::greville, options);
  iganet::UniformBSpline<real_t, 3, 3, 4> bspline_clone(bspline_orig, true);

  bspline_orig.transform([](const std::array<real_t, 2> xi) {
    return std::array<real_t, 3>{0.0_r, 1.0_r, 2.0_r};
  });

  EXPECT_TRUE(bspline_ref == bspline_clone);
}

TEST_F(BSplineTest, UniformBSpline_move_constructor) {
  iganet::UniformBSpline<real_t, 3, 3, 4> bspline_ref(
      {7, 8}, iganet::init::greville, options);
  auto bspline(iganet::UniformBSpline<real_t, 3, 3, 4>(
                   {4, 5}, iganet::init::greville, options)
                   .uniform_refine(2));

  EXPECT_TRUE(bspline.isclose(bspline_ref));
}

TEST_F(BSplineTest, UniformBSpline_copy_coeffs_constructor) {
  iganet::UniformBSpline<real_t, 3, 3, 4> bspline_orig(
      {4, 5}, iganet::init::greville, options);
  iganet::UniformBSpline<real_t, 3, 3, 4> bspline_copy(bspline_orig,
                                                       bspline_orig.coeffs());

  bspline_orig.transform([](const std::array<real_t, 2> xi) {
    return std::array<real_t, 3>{0.0_r, 1.0_r, 2.0_r};
  });

  EXPECT_TRUE(bspline_orig == bspline_copy);
}

TEST_F(BSplineTest, UniformBSpline_clone_coeffs_constructor) {
  iganet::UniformBSpline<real_t, 3, 3, 4> bspline_ref(
      {4, 5}, iganet::init::greville, options);
  iganet::UniformBSpline<real_t, 3, 3, 4> bspline_orig(
      {4, 5}, iganet::init::greville, options);
  iganet::UniformBSpline<real_t, 3, 3, 4> bspline_clone(
      bspline_orig, bspline_orig.coeffs(), true);

  bspline_orig.transform([](const std::array<real_t, 2> xi) {
    return std::array<real_t, 3>{0.0_r, 1.0_r, 2.0_r};
  });

  EXPECT_TRUE(bspline_ref == bspline_clone);
}

TEST_F(BSplineTest, UniformBSpline_read_write) {
  std::filesystem::path filename =
      std::filesystem::temp_directory_path() / std::to_string(rand());
  iganet::UniformBSpline<real_t, 3, 3, 4> bspline_out(
      {4, 5}, iganet::init::greville, options);
  bspline_out.save(filename.c_str());

  iganet::UniformBSpline<real_t, 3, 3, 4> bspline_in(options);
  bspline_in.load(filename.c_str());
  std::filesystem::remove(filename);

  EXPECT_TRUE(bspline_in == bspline_out);
  EXPECT_FALSE(bspline_in != bspline_out);
}

TEST_F(BSplineTest, UniformBSpline_to_from_xml) {
  {
    iganet::UniformBSpline<real_t, 1, 3> bspline_out({4}, iganet::init::zeros,
                                                     options);

    bspline_out.transform([](const std::array<real_t, 1> xi) {
      return std::array<real_t, 1>{static_cast<real_t>(std::rand())};
    });

    pugi::xml_document doc = bspline_out.to_xml();

    iganet::UniformBSpline<real_t, 1, 3> bspline_in(options);
    bspline_in.from_xml(doc);

    EXPECT_TRUE(bspline_in == bspline_out);

    // non-matching degree
    EXPECT_THROW((iganet::UniformBSpline<real_t, 1, 2>{}.from_xml(doc, 0)),
                 std::runtime_error);

    // non-matching parametric dimension
    EXPECT_THROW((iganet::UniformBSpline<real_t, 1, 3, 3>{}.from_xml(doc, 0)),
                 std::runtime_error);
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 1, 3, 3, 3>{}.from_xml(doc, 0)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 1, 3, 3, 3, 3>{}.from_xml(doc, 0)),
        std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW((iganet::UniformBSpline<real_t, 2, 3>{}.from_xml(doc, 0)),
                 std::runtime_error);
    EXPECT_THROW((iganet::UniformBSpline<real_t, 3, 3>{}.from_xml(doc, 0)),
                 std::runtime_error);
    EXPECT_THROW((iganet::UniformBSpline<real_t, 4, 3>{}.from_xml(doc, 0)),
                 std::runtime_error);

    // non-matching id
    EXPECT_THROW((iganet::UniformBSpline<real_t, 1, 3>{}.from_xml(doc, 1)),
                 std::runtime_error);
  }

  {
    iganet::UniformBSpline<real_t, 2, 3> bspline_out({4}, iganet::init::zeros,
                                                     options);

    bspline_out.transform([](const std::array<real_t, 1> xi) {
      return std::array<real_t, 2>{static_cast<real_t>(std::rand()),
                                   static_cast<real_t>(std::rand())};
    });

    pugi::xml_document doc = bspline_out.to_xml();

    iganet::UniformBSpline<real_t, 2, 3> bspline_in(options);
    bspline_in.from_xml(doc);

    EXPECT_TRUE(bspline_in == bspline_out);

    // non-matching degree
    EXPECT_THROW((iganet::UniformBSpline<real_t, 2, 2>{}.from_xml(doc, 0)),
                 std::runtime_error);

    // non-matching parametric dimension
    EXPECT_THROW((iganet::UniformBSpline<real_t, 2, 3, 3>{}.from_xml(doc, 0)),
                 std::runtime_error);
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 2, 3, 3, 3>{}.from_xml(doc, 0)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 2, 3, 3, 3, 3>{}.from_xml(doc, 0)),
        std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW((iganet::UniformBSpline<real_t, 1, 3>{}.from_xml(doc, 0)),
                 std::runtime_error);
    EXPECT_THROW((iganet::UniformBSpline<real_t, 3, 3>{}.from_xml(doc, 0)),
                 std::runtime_error);
    EXPECT_THROW((iganet::UniformBSpline<real_t, 4, 3>{}.from_xml(doc, 0)),
                 std::runtime_error);

    // non-matching id
    EXPECT_THROW((iganet::UniformBSpline<real_t, 1, 3>{}.from_xml(doc, 1)),
                 std::runtime_error);
  }

  {
    iganet::UniformBSpline<real_t, 3, 3> bspline_out({4}, iganet::init::zeros,
                                                     options);

    bspline_out.transform([](const std::array<real_t, 1> xi) {
      return std::array<real_t, 3>{static_cast<real_t>(std::rand()),
                                   static_cast<real_t>(std::rand()),
                                   static_cast<real_t>(std::rand())};
    });

    pugi::xml_document doc = bspline_out.to_xml();

    iganet::UniformBSpline<real_t, 3, 3> bspline_in(options);
    bspline_in.from_xml(doc);

    EXPECT_TRUE(bspline_in == bspline_out);

    // non-matching degree
    EXPECT_THROW((iganet::UniformBSpline<real_t, 3, 2>{}.from_xml(doc, 0)),
                 std::runtime_error);

    // non-matching parametric dimension
    EXPECT_THROW((iganet::UniformBSpline<real_t, 3, 3, 3>{}.from_xml(doc, 0)),
                 std::runtime_error);
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 3, 3, 3, 3>{}.from_xml(doc, 0)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 3, 3, 3, 3, 3>{}.from_xml(doc, 0)),
        std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW((iganet::UniformBSpline<real_t, 1, 3>{}.from_xml(doc, 0)),
                 std::runtime_error);
    EXPECT_THROW((iganet::UniformBSpline<real_t, 2, 3>{}.from_xml(doc, 0)),
                 std::runtime_error);
    EXPECT_THROW((iganet::UniformBSpline<real_t, 4, 3>{}.from_xml(doc, 0)),
                 std::runtime_error);

    // non-matching id
    EXPECT_THROW((iganet::UniformBSpline<real_t, 3, 3>{}.from_xml(doc, 1)),
                 std::runtime_error);
  }

  {
    iganet::UniformBSpline<real_t, 4, 3> bspline_out({4}, iganet::init::zeros,
                                                     options);

    bspline_out.transform([](const std::array<real_t, 1> xi) {
      return std::array<real_t, 4>{
          static_cast<real_t>(std::rand()), static_cast<real_t>(std::rand()),
          static_cast<real_t>(std::rand()), static_cast<real_t>(std::rand())};
    });

    pugi::xml_document doc = bspline_out.to_xml();

    iganet::UniformBSpline<real_t, 4, 3> bspline_in(options);
    bspline_in.from_xml(doc);

    EXPECT_TRUE(bspline_in == bspline_out);

    // non-matching degree
    EXPECT_THROW((iganet::UniformBSpline<real_t, 4, 2>{}.from_xml(doc, 0)),
                 std::runtime_error);

    // non-matching parametric dimension
    EXPECT_THROW((iganet::UniformBSpline<real_t, 4, 3, 3>{}.from_xml(doc, 0)),
                 std::runtime_error);
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 4, 3, 3, 3>{}.from_xml(doc, 0)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 4, 3, 3, 3, 3>{}.from_xml(doc, 0)),
        std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW((iganet::UniformBSpline<real_t, 1, 3>{}.from_xml(doc, 0)),
                 std::runtime_error);
    EXPECT_THROW((iganet::UniformBSpline<real_t, 2, 3>{}.from_xml(doc, 0)),
                 std::runtime_error);
    EXPECT_THROW((iganet::UniformBSpline<real_t, 3, 3>{}.from_xml(doc, 0)),
                 std::runtime_error);

    // non-matching id
    EXPECT_THROW((iganet::UniformBSpline<real_t, 4, 3>{}.from_xml(doc, 1)),
                 std::runtime_error);
  }

  {
    iganet::UniformBSpline<real_t, 1, 3, 4> bspline_out(
        {4, 5}, iganet::init::zeros, options);

    bspline_out.transform([](const std::array<real_t, 2> xi) {
      return std::array<real_t, 1>{static_cast<real_t>(std::rand())};
    });

    pugi::xml_document doc = bspline_out.to_xml();

    iganet::UniformBSpline<real_t, 1, 3, 4> bspline_in(options);
    bspline_in.from_xml(doc);

    EXPECT_TRUE(bspline_in == bspline_out);

    // non-matching degree
    EXPECT_THROW((iganet::UniformBSpline<real_t, 1, 3, 3>{}.from_xml(doc, 0)),
                 std::runtime_error);

    // non-matching parametric dimension
    EXPECT_THROW((iganet::UniformBSpline<real_t, 1, 3>{}.from_xml(doc, 0)),
                 std::runtime_error);
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 1, 3, 4, 3>{}.from_xml(doc, 0)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 1, 3, 4, 3, 3>{}.from_xml(doc, 0)),
        std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW((iganet::UniformBSpline<real_t, 2, 3, 4>{}.from_xml(doc, 0)),
                 std::runtime_error);
    EXPECT_THROW((iganet::UniformBSpline<real_t, 3, 3, 4>{}.from_xml(doc, 0)),
                 std::runtime_error);
    EXPECT_THROW((iganet::UniformBSpline<real_t, 4, 3, 4>{}.from_xml(doc, 0)),
                 std::runtime_error);

    // non-matching id
    EXPECT_THROW((iganet::UniformBSpline<real_t, 1, 3, 4>{}.from_xml(doc, 1)),
                 std::runtime_error);
  }

  {
    iganet::UniformBSpline<real_t, 2, 3, 4> bspline_out(
        {4, 5}, iganet::init::zeros, options);

    bspline_out.transform([](const std::array<real_t, 2> xi) {
      return std::array<real_t, 2>{static_cast<real_t>(std::rand()),
                                   static_cast<real_t>(std::rand())};
    });

    pugi::xml_document doc = bspline_out.to_xml();

    iganet::UniformBSpline<real_t, 2, 3, 4> bspline_in(options);
    bspline_in.from_xml(doc);

    EXPECT_TRUE(bspline_in == bspline_out);

    // non-matching degree
    EXPECT_THROW((iganet::UniformBSpline<real_t, 2, 3, 3>{}.from_xml(doc, 0)),
                 std::runtime_error);

    // non-matching parametric dimension
    EXPECT_THROW((iganet::UniformBSpline<real_t, 2, 3>{}.from_xml(doc, 0)),
                 std::runtime_error);
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 2, 3, 4, 3>{}.from_xml(doc, 0)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 2, 3, 4, 3, 3>{}.from_xml(doc, 0)),
        std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW((iganet::UniformBSpline<real_t, 1, 3, 4>{}.from_xml(doc, 0)),
                 std::runtime_error);
    EXPECT_THROW((iganet::UniformBSpline<real_t, 3, 3, 4>{}.from_xml(doc, 0)),
                 std::runtime_error);
    EXPECT_THROW((iganet::UniformBSpline<real_t, 4, 3, 4>{}.from_xml(doc, 0)),
                 std::runtime_error);

    // non-matching id
    EXPECT_THROW((iganet::UniformBSpline<real_t, 2, 3, 4>{}.from_xml(doc, 1)),
                 std::runtime_error);
  }

  {
    iganet::UniformBSpline<real_t, 3, 3, 4> bspline_out(
        {4, 5}, iganet::init::zeros, options);

    bspline_out.transform([](const std::array<real_t, 2> xi) {
      return std::array<real_t, 3>{static_cast<real_t>(std::rand()),
                                   static_cast<real_t>(std::rand()),
                                   static_cast<real_t>(std::rand())};
    });

    pugi::xml_document doc = bspline_out.to_xml();

    iganet::UniformBSpline<real_t, 3, 3, 4> bspline_in(options);
    bspline_in.from_xml(doc);

    EXPECT_TRUE(bspline_in == bspline_out);

    // non-matching degree
    EXPECT_THROW((iganet::UniformBSpline<real_t, 3, 3, 3>{}.from_xml(doc, 0)),
                 std::runtime_error);

    // non-matching parametric dimension
    EXPECT_THROW((iganet::UniformBSpline<real_t, 3, 3>{}.from_xml(doc, 0)),
                 std::runtime_error);
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 3, 3, 4, 3>{}.from_xml(doc, 0)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 3, 3, 4, 3, 3>{}.from_xml(doc, 0)),
        std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW((iganet::UniformBSpline<real_t, 1, 3, 4>{}.from_xml(doc, 0)),
                 std::runtime_error);
    EXPECT_THROW((iganet::UniformBSpline<real_t, 2, 3, 4>{}.from_xml(doc, 0)),
                 std::runtime_error);
    EXPECT_THROW((iganet::UniformBSpline<real_t, 4, 3, 4>{}.from_xml(doc, 0)),
                 std::runtime_error);

    // non-matching id
    EXPECT_THROW((iganet::UniformBSpline<real_t, 3, 3, 4>{}.from_xml(doc, 1)),
                 std::runtime_error);
  }

  {
    iganet::UniformBSpline<real_t, 4, 3, 4> bspline_out(
        {4, 5}, iganet::init::zeros, options);

    bspline_out.transform([](const std::array<real_t, 2> xi) {
      return std::array<real_t, 4>{
          static_cast<real_t>(std::rand()), static_cast<real_t>(std::rand()),
          static_cast<real_t>(std::rand()), static_cast<real_t>(std::rand())};
    });

    pugi::xml_document doc = bspline_out.to_xml();

    iganet::UniformBSpline<real_t, 4, 3, 4> bspline_in(options);
    bspline_in.from_xml(doc);

    EXPECT_TRUE(bspline_in == bspline_out);

    // non-matching degree
    EXPECT_THROW((iganet::UniformBSpline<real_t, 4, 3, 3>{}.from_xml(doc, 0)),
                 std::runtime_error);

    // non-matching parametric dimension
    EXPECT_THROW((iganet::UniformBSpline<real_t, 4, 3>{}.from_xml(doc, 0)),
                 std::runtime_error);
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 4, 3, 4, 3>{}.from_xml(doc, 0)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 4, 3, 4, 3, 3>{}.from_xml(doc, 0)),
        std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW((iganet::UniformBSpline<real_t, 1, 3, 4>{}.from_xml(doc, 0)),
                 std::runtime_error);
    EXPECT_THROW((iganet::UniformBSpline<real_t, 2, 3, 4>{}.from_xml(doc, 0)),
                 std::runtime_error);
    EXPECT_THROW((iganet::UniformBSpline<real_t, 3, 3, 4>{}.from_xml(doc, 0)),
                 std::runtime_error);

    // non-matching id
    EXPECT_THROW((iganet::UniformBSpline<real_t, 4, 3, 4>{}.from_xml(doc, 1)),
                 std::runtime_error);
  }

  {
    iganet::UniformBSpline<real_t, 1, 3, 4, 5> bspline_out(
        {4, 5, 6}, iganet::init::zeros, options);

    bspline_out.transform([](const std::array<real_t, 3> xi) {
      return std::array<real_t, 1>{static_cast<real_t>(std::rand())};
    });

    pugi::xml_document doc = bspline_out.to_xml();

    iganet::UniformBSpline<real_t, 1, 3, 4, 5> bspline_in(options);
    bspline_in.from_xml(doc);

    EXPECT_TRUE(bspline_in == bspline_out);

    // non-matching degree
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 1, 3, 3, 3>{}.from_xml(doc, 0)),
        std::runtime_error);

    // non-matching parametric dimension
    EXPECT_THROW((iganet::UniformBSpline<real_t, 1, 3>{}.from_xml(doc, 0)),
                 std::runtime_error);
    EXPECT_THROW((iganet::UniformBSpline<real_t, 1, 3, 4>{}.from_xml(doc, 0)),
                 std::runtime_error);
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 1, 3, 4, 5, 3>{}.from_xml(doc, 0)),
        std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 2, 3, 4, 5>{}.from_xml(doc, 0)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 3, 3, 4, 5>{}.from_xml(doc, 0)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 4, 3, 4, 5>{}.from_xml(doc, 0)),
        std::runtime_error);

    // non-matching id
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 1, 3, 4, 5>{}.from_xml(doc, 1)),
        std::runtime_error);
  }

  {
    iganet::UniformBSpline<real_t, 2, 3, 4, 5> bspline_out(
        {4, 5, 6}, iganet::init::zeros, options);

    bspline_out.transform([](const std::array<real_t, 3> xi) {
      return std::array<real_t, 2>{static_cast<real_t>(std::rand()),
                                   static_cast<real_t>(std::rand())};
    });

    pugi::xml_document doc = bspline_out.to_xml();

    iganet::UniformBSpline<real_t, 2, 3, 4, 5> bspline_in(options);
    bspline_in.from_xml(doc);

    EXPECT_TRUE(bspline_in == bspline_out);

    // non-matching degree
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 2, 3, 3, 3>{}.from_xml(doc, 0)),
        std::runtime_error);

    // non-matching parametric dimension
    EXPECT_THROW((iganet::UniformBSpline<real_t, 2, 3>{}.from_xml(doc, 0)),
                 std::runtime_error);
    EXPECT_THROW((iganet::UniformBSpline<real_t, 2, 3, 4>{}.from_xml(doc, 0)),
                 std::runtime_error);
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 2, 3, 4, 5, 3>{}.from_xml(doc, 0)),
        std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 1, 3, 4, 5>{}.from_xml(doc, 0)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 3, 3, 4, 5>{}.from_xml(doc, 0)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 4, 3, 4, 5>{}.from_xml(doc, 0)),
        std::runtime_error);

    // non-matching id
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 2, 3, 4, 5>{}.from_xml(doc, 1)),
        std::runtime_error);
  }

  {
    iganet::UniformBSpline<real_t, 3, 3, 4, 5> bspline_out(
        {4, 5, 6}, iganet::init::zeros, options);

    bspline_out.transform([](const std::array<real_t, 3> xi) {
      return std::array<real_t, 3>{static_cast<real_t>(std::rand()),
                                   static_cast<real_t>(std::rand()),
                                   static_cast<real_t>(std::rand())};
    });

    pugi::xml_document doc = bspline_out.to_xml();

    iganet::UniformBSpline<real_t, 3, 3, 4, 5> bspline_in(options);
    bspline_in.from_xml(doc);

    EXPECT_TRUE(bspline_in == bspline_out);

    // non-matching degree
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 3, 3, 3, 3>{}.from_xml(doc, 0)),
        std::runtime_error);

    // non-matching parametric dimension
    EXPECT_THROW((iganet::UniformBSpline<real_t, 3, 3>{}.from_xml(doc, 0)),
                 std::runtime_error);
    EXPECT_THROW((iganet::UniformBSpline<real_t, 3, 3, 4>{}.from_xml(doc, 0)),
                 std::runtime_error);
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 3, 3, 4, 5, 3>{}.from_xml(doc, 0)),
        std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 1, 3, 4, 5>{}.from_xml(doc, 0)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 2, 3, 4, 5>{}.from_xml(doc, 0)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 4, 3, 4, 5>{}.from_xml(doc, 0)),
        std::runtime_error);

    // non-matching id
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 3, 3, 4, 5>{}.from_xml(doc, 1)),
        std::runtime_error);
  }

  {
    iganet::UniformBSpline<real_t, 4, 3, 4, 5> bspline_out(
        {4, 5, 6}, iganet::init::zeros, options);

    bspline_out.transform([](const std::array<real_t, 3> xi) {
      return std::array<real_t, 4>{
          static_cast<real_t>(std::rand()), static_cast<real_t>(std::rand()),
          static_cast<real_t>(std::rand()), static_cast<real_t>(std::rand())};
    });

    pugi::xml_document doc = bspline_out.to_xml();

    iganet::UniformBSpline<real_t, 4, 3, 4, 5> bspline_in(options);
    bspline_in.from_xml(doc);

    EXPECT_TRUE(bspline_in == bspline_out);

    // non-matching degree
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 4, 3, 3, 3>{}.from_xml(doc, 0)),
        std::runtime_error);

    // non-matching parametric dimension
    EXPECT_THROW((iganet::UniformBSpline<real_t, 4, 3>{}.from_xml(doc, 0)),
                 std::runtime_error);
    EXPECT_THROW((iganet::UniformBSpline<real_t, 4, 3, 4>{}.from_xml(doc, 0)),
                 std::runtime_error);
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 4, 3, 4, 5, 3>{}.from_xml(doc, 0)),
        std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 1, 3, 4, 5>{}.from_xml(doc, 0)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 2, 3, 4, 5>{}.from_xml(doc, 0)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 3, 3, 4, 5>{}.from_xml(doc, 0)),
        std::runtime_error);

    // non-matching id
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 4, 3, 4, 5>{}.from_xml(doc, 1)),
        std::runtime_error);
  }

  {
    iganet::UniformBSpline<real_t, 1, 3, 4, 5, 1> bspline_out(
        {4, 5, 6, 2}, iganet::init::zeros, options);

    bspline_out.transform([](const std::array<real_t, 4> xi) {
      return std::array<real_t, 1>{static_cast<real_t>(std::rand())};
    });

    pugi::xml_document doc = bspline_out.to_xml();

    iganet::UniformBSpline<real_t, 1, 3, 4, 5, 1> bspline_in(options);
    bspline_in.from_xml(doc);

    EXPECT_TRUE(bspline_in == bspline_out);

    // non-matching degree
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 1, 3, 4, 5, 2>{}.from_xml(doc, 0)),
        std::runtime_error);

    // non-matching parametric dimension
    EXPECT_THROW((iganet::UniformBSpline<real_t, 1, 3>{}.from_xml(doc, 0)),
                 std::runtime_error);
    EXPECT_THROW((iganet::UniformBSpline<real_t, 1, 3, 4>{}.from_xml(doc, 0)),
                 std::runtime_error);
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 1, 3, 4, 5>{}.from_xml(doc, 0)),
        std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 2, 3, 4, 5, 1>{}.from_xml(doc, 0)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 3, 3, 4, 5, 1>{}.from_xml(doc, 0)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 4, 3, 4, 5, 1>{}.from_xml(doc, 0)),
        std::runtime_error);

    // non-matching id
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 1, 3, 4, 5, 1>{}.from_xml(doc, 1)),
        std::runtime_error);
  }

  {
    iganet::UniformBSpline<real_t, 2, 3, 4, 5, 1> bspline_out(
        {4, 5, 6, 2}, iganet::init::zeros, options);

    bspline_out.transform([](const std::array<real_t, 4> xi) {
      return std::array<real_t, 2>{static_cast<real_t>(std::rand()),
                                   static_cast<real_t>(std::rand())};
    });

    pugi::xml_document doc = bspline_out.to_xml();

    iganet::UniformBSpline<real_t, 2, 3, 4, 5, 1> bspline_in(options);
    bspline_in.from_xml(doc);

    EXPECT_TRUE(bspline_in == bspline_out);

    // non-matching degree
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 2, 3, 4, 5, 2>{}.from_xml(doc, 0)),
        std::runtime_error);

    // non-matching parametric dimension
    EXPECT_THROW((iganet::UniformBSpline<real_t, 2, 3>{}.from_xml(doc, 0)),
                 std::runtime_error);
    EXPECT_THROW((iganet::UniformBSpline<real_t, 2, 3, 4>{}.from_xml(doc, 0)),
                 std::runtime_error);
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 2, 3, 4, 5>{}.from_xml(doc, 0)),
        std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 1, 3, 4, 5, 1>{}.from_xml(doc, 0)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 3, 3, 4, 5, 1>{}.from_xml(doc, 0)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 4, 3, 4, 5, 1>{}.from_xml(doc, 0)),
        std::runtime_error);

    // non-matching id
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 2, 3, 4, 5, 1>{}.from_xml(doc, 1)),
        std::runtime_error);
  }

  {
    iganet::UniformBSpline<real_t, 3, 3, 4, 5, 1> bspline_out(
        {4, 5, 6, 2}, iganet::init::zeros, options);

    bspline_out.transform([](const std::array<real_t, 4> xi) {
      return std::array<real_t, 3>{static_cast<real_t>(std::rand()),
                                   static_cast<real_t>(std::rand()),
                                   static_cast<real_t>(std::rand())};
    });

    pugi::xml_document doc = bspline_out.to_xml();

    iganet::UniformBSpline<real_t, 3, 3, 4, 5, 1> bspline_in(options);
    bspline_in.from_xml(doc);

    EXPECT_TRUE(bspline_in == bspline_out);

    // non-matching degree
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 3, 3, 4, 5, 2>{}.from_xml(doc, 0)),
        std::runtime_error);

    // non-matching parametric dimension
    EXPECT_THROW((iganet::UniformBSpline<real_t, 3, 3>{}.from_xml(doc, 0)),
                 std::runtime_error);
    EXPECT_THROW((iganet::UniformBSpline<real_t, 3, 3, 4>{}.from_xml(doc, 0)),
                 std::runtime_error);
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 3, 3, 4, 5>{}.from_xml(doc, 0)),
        std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 1, 3, 4, 5, 1>{}.from_xml(doc, 0)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 2, 3, 4, 5, 1>{}.from_xml(doc, 0)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 4, 3, 4, 5, 1>{}.from_xml(doc, 0)),
        std::runtime_error);

    // non-matching id
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 3, 3, 4, 5, 1>{}.from_xml(doc, 1)),
        std::runtime_error);
  }

  {
    iganet::UniformBSpline<real_t, 4, 3, 4, 5, 1> bspline_out(
        {4, 5, 6, 2}, iganet::init::zeros, options);

    bspline_out.transform([](const std::array<real_t, 4> xi) {
      return std::array<real_t, 4>{
          static_cast<real_t>(std::rand()), static_cast<real_t>(std::rand()),
          static_cast<real_t>(std::rand()), static_cast<real_t>(std::rand())};
    });

    pugi::xml_document doc = bspline_out.to_xml();

    iganet::UniformBSpline<real_t, 4, 3, 4, 5, 1> bspline_in(options);
    bspline_in.from_xml(doc);

    EXPECT_TRUE(bspline_in == bspline_out);

    // non-matching degree
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 4, 3, 4, 5, 2>{}.from_xml(doc, 0)),
        std::runtime_error);

    // non-matching parametric dimension
    EXPECT_THROW((iganet::UniformBSpline<real_t, 4, 3>{}.from_xml(doc, 0)),
                 std::runtime_error);
    EXPECT_THROW((iganet::UniformBSpline<real_t, 4, 3, 4>{}.from_xml(doc, 0)),
                 std::runtime_error);
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 4, 3, 4, 5>{}.from_xml(doc, 0)),
        std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 1, 3, 4, 5, 1>{}.from_xml(doc, 0)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 2, 3, 4, 5, 1>{}.from_xml(doc, 0)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 3, 3, 4, 5, 1>{}.from_xml(doc, 0)),
        std::runtime_error);

    // non-matching id
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 4, 3, 4, 5, 1>{}.from_xml(doc, 1)),
        std::runtime_error);
  }
}

TEST_F(BSplineTest, UniformBSpline_load_from_xml) {
  {
    pugi::xml_document doc;
    pugi::xml_parse_result result =
        doc.load_file(IGANET_DATA_DIR "domain1d/line.xml");

    iganet::UniformBSpline<real_t, 3, 2> bspline_in(options);
    bspline_in.from_xml(doc);

    iganet::UniformBSpline<real_t, 3, 2> bspline_ref({3}, iganet::init::zeros,
                                                     options);

    bspline_ref.transform([](const std::array<real_t, 1> xi) {
      return std::array<real_t, 3>{xi[0], 0.0_r, 0.0_r};
    });

    EXPECT_TRUE(bspline_in == bspline_ref);
  }

  {
    pugi::xml_document doc;
    pugi::xml_parse_result result =
        doc.load_file(IGANET_DATA_DIR "domain2d/square.xml");

    iganet::UniformBSpline<real_t, 2, 1, 1> bspline_in(options);
    bspline_in.from_xml(doc, 1);

    iganet::UniformBSpline<real_t, 2, 1, 1> bspline_ref(
        {2, 2}, iganet::init::greville, options);

    EXPECT_TRUE(bspline_in == bspline_ref);
  }

  {
    pugi::xml_document doc;
    pugi::xml_parse_result result =
        doc.load_file(IGANET_DATA_DIR "domain3d/GshapedVolume.xml");

    iganet::UniformBSpline<real_t, 3, 2, 2, 2> bspline_in(options);
    bspline_in.from_xml(doc);
  }

  {
    pugi::xml_document doc;
    pugi::xml_parse_result result =
        doc.load_file(IGANET_DATA_DIR "surfaces/g_plus_s_surf.xml");

    iganet::UniformBSpline<real_t, 3, 3, 3> bspline_in0(options);
    iganet::UniformBSpline<real_t, 3, 3, 1> bspline_in1(options);

    for (int i = 0; i < 126; ++i) {
      try {
        bspline_in0.from_xml(doc, i);
      } catch (...) {
        bspline_in1.from_xml(doc, i);
      }
    }
  }
}

TEST_F(BSplineTest, UniformBSpline_to_from_json) {
  {
    iganet::UniformBSpline<real_t, 1, 3> bspline_out({4}, iganet::init::zeros,
                                                     options);

    bspline_out.transform([](const std::array<real_t, 1> xi) {
      return std::array<real_t, 1>{static_cast<real_t>(std::rand())};
    });

    nlohmann::json json = bspline_out.to_json();

    iganet::UniformBSpline<real_t, 1, 3> bspline_in(options);
    bspline_in.from_json(json);

    EXPECT_TRUE(bspline_in == bspline_out);

    // non-matching degree
    EXPECT_THROW((iganet::UniformBSpline<real_t, 1, 2>{}.from_json(json)),
                 std::runtime_error);

    // non-matching parametric dimension
    EXPECT_THROW((iganet::UniformBSpline<real_t, 1, 3, 3>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW((iganet::UniformBSpline<real_t, 1, 3, 3, 3>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 1, 3, 3, 3, 3>{}.from_json(json)),
        std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW((iganet::UniformBSpline<real_t, 2, 3>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW((iganet::UniformBSpline<real_t, 3, 3>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW((iganet::UniformBSpline<real_t, 4, 3>{}.from_json(json)),
                 std::runtime_error);
  }

  {
    iganet::UniformBSpline<real_t, 2, 3> bspline_out({4}, iganet::init::zeros,
                                                     options);

    bspline_out.transform([](const std::array<real_t, 1> xi) {
      return std::array<real_t, 2>{static_cast<real_t>(std::rand()),
                                   static_cast<real_t>(std::rand())};
    });

    nlohmann::json json = bspline_out.to_json();

    iganet::UniformBSpline<real_t, 2, 3> bspline_in(options);
    bspline_in.from_json(json);

    EXPECT_TRUE(bspline_in == bspline_out);

    // non-matching degree
    EXPECT_THROW((iganet::UniformBSpline<real_t, 2, 2>{}.from_json(json)),
                 std::runtime_error);

    // non-matching parametric dimension
    EXPECT_THROW((iganet::UniformBSpline<real_t, 2, 3, 3>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW((iganet::UniformBSpline<real_t, 2, 3, 3, 3>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 2, 3, 3, 3, 3>{}.from_json(json)),
        std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW((iganet::UniformBSpline<real_t, 1, 3>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW((iganet::UniformBSpline<real_t, 3, 3>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW((iganet::UniformBSpline<real_t, 4, 3>{}.from_json(json)),
                 std::runtime_error);
  }

  {
    iganet::UniformBSpline<real_t, 3, 3> bspline_out({4}, iganet::init::zeros,
                                                     options);

    bspline_out.transform([](const std::array<real_t, 1> xi) {
      return std::array<real_t, 3>{static_cast<real_t>(std::rand()),
                                   static_cast<real_t>(std::rand()),
                                   static_cast<real_t>(std::rand())};
    });

    nlohmann::json json = bspline_out.to_json();

    iganet::UniformBSpline<real_t, 3, 3> bspline_in(options);
    bspline_in.from_json(json);

    EXPECT_TRUE(bspline_in == bspline_out);

    // non-matching degree
    EXPECT_THROW((iganet::UniformBSpline<real_t, 3, 2>{}.from_json(json)),
                 std::runtime_error);

    // non-matching parametric dimension
    EXPECT_THROW((iganet::UniformBSpline<real_t, 3, 3, 3>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW((iganet::UniformBSpline<real_t, 3, 3, 3, 3>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 3, 3, 3, 3, 3>{}.from_json(json)),
        std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW((iganet::UniformBSpline<real_t, 1, 3>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW((iganet::UniformBSpline<real_t, 2, 3>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW((iganet::UniformBSpline<real_t, 4, 3>{}.from_json(json)),
                 std::runtime_error);
  }

  {
    iganet::UniformBSpline<real_t, 4, 3> bspline_out({4}, iganet::init::zeros,
                                                     options);

    bspline_out.transform([](const std::array<real_t, 1> xi) {
      return std::array<real_t, 4>{
          static_cast<real_t>(std::rand()), static_cast<real_t>(std::rand()),
          static_cast<real_t>(std::rand()), static_cast<real_t>(std::rand())};
    });

    nlohmann::json json = bspline_out.to_json();

    iganet::UniformBSpline<real_t, 4, 3> bspline_in(options);
    bspline_in.from_json(json);

    EXPECT_TRUE(bspline_in == bspline_out);

    // non-matching degree
    EXPECT_THROW((iganet::UniformBSpline<real_t, 4, 2>{}.from_json(json)),
                 std::runtime_error);

    // non-matching parametric dimension
    EXPECT_THROW((iganet::UniformBSpline<real_t, 4, 3, 3>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW((iganet::UniformBSpline<real_t, 4, 3, 3, 3>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 4, 3, 3, 3, 3>{}.from_json(json)),
        std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW((iganet::UniformBSpline<real_t, 1, 3>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW((iganet::UniformBSpline<real_t, 2, 3>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW((iganet::UniformBSpline<real_t, 3, 3>{}.from_json(json)),
                 std::runtime_error);
  }

  {
    iganet::UniformBSpline<real_t, 1, 3, 4> bspline_out(
        {4, 5}, iganet::init::zeros, options);

    bspline_out.transform([](const std::array<real_t, 2> xi) {
      return std::array<real_t, 1>{static_cast<real_t>(std::rand())};
    });

    nlohmann::json json = bspline_out.to_json();

    iganet::UniformBSpline<real_t, 1, 3, 4> bspline_in(options);
    bspline_in.from_json(json);

    EXPECT_TRUE(bspline_in == bspline_out);

    // non-matching degree
    EXPECT_THROW((iganet::UniformBSpline<real_t, 1, 3, 3>{}.from_json(json)),
                 std::runtime_error);

    // non-matching parametric dimension
    EXPECT_THROW((iganet::UniformBSpline<real_t, 1, 3>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW((iganet::UniformBSpline<real_t, 1, 3, 4, 3>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 1, 3, 4, 3, 3>{}.from_json(json)),
        std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW((iganet::UniformBSpline<real_t, 2, 3, 4>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW((iganet::UniformBSpline<real_t, 3, 3, 4>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW((iganet::UniformBSpline<real_t, 4, 3, 4>{}.from_json(json)),
                 std::runtime_error);
  }

  {
    iganet::UniformBSpline<real_t, 2, 3, 4> bspline_out(
        {4, 5}, iganet::init::zeros, options);

    bspline_out.transform([](const std::array<real_t, 2> xi) {
      return std::array<real_t, 2>{static_cast<real_t>(std::rand()),
                                   static_cast<real_t>(std::rand())};
    });

    nlohmann::json json = bspline_out.to_json();

    iganet::UniformBSpline<real_t, 2, 3, 4> bspline_in(options);
    bspline_in.from_json(json);

    EXPECT_TRUE(bspline_in == bspline_out);

    // non-matching degree
    EXPECT_THROW((iganet::UniformBSpline<real_t, 2, 3, 3>{}.from_json(json)),
                 std::runtime_error);

    // non-matching parametric dimension
    EXPECT_THROW((iganet::UniformBSpline<real_t, 2, 3>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW((iganet::UniformBSpline<real_t, 2, 3, 4, 3>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 2, 3, 4, 3, 3>{}.from_json(json)),
        std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW((iganet::UniformBSpline<real_t, 1, 3, 4>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW((iganet::UniformBSpline<real_t, 3, 3, 4>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW((iganet::UniformBSpline<real_t, 4, 3, 4>{}.from_json(json)),
                 std::runtime_error);
  }

  {
    iganet::UniformBSpline<real_t, 3, 3, 4> bspline_out(
        {4, 5}, iganet::init::zeros, options);

    bspline_out.transform([](const std::array<real_t, 2> xi) {
      return std::array<real_t, 3>{static_cast<real_t>(std::rand()),
                                   static_cast<real_t>(std::rand()),
                                   static_cast<real_t>(std::rand())};
    });

    nlohmann::json json = bspline_out.to_json();

    iganet::UniformBSpline<real_t, 3, 3, 4> bspline_in(options);
    bspline_in.from_json(json);

    EXPECT_TRUE(bspline_in == bspline_out);

    // non-matching degree
    EXPECT_THROW((iganet::UniformBSpline<real_t, 3, 3, 3>{}.from_json(json)),
                 std::runtime_error);

    // non-matching parametric dimension
    EXPECT_THROW((iganet::UniformBSpline<real_t, 3, 3>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW((iganet::UniformBSpline<real_t, 3, 3, 4, 3>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 3, 3, 4, 3, 3>{}.from_json(json)),
        std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW((iganet::UniformBSpline<real_t, 1, 3, 4>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW((iganet::UniformBSpline<real_t, 2, 3, 4>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW((iganet::UniformBSpline<real_t, 4, 3, 4>{}.from_json(json)),
                 std::runtime_error);
  }

  {
    iganet::UniformBSpline<real_t, 4, 3, 4> bspline_out(
        {4, 5}, iganet::init::zeros, options);

    bspline_out.transform([](const std::array<real_t, 2> xi) {
      return std::array<real_t, 4>{
          static_cast<real_t>(std::rand()), static_cast<real_t>(std::rand()),
          static_cast<real_t>(std::rand()), static_cast<real_t>(std::rand())};
    });

    nlohmann::json json = bspline_out.to_json();

    iganet::UniformBSpline<real_t, 4, 3, 4> bspline_in(options);
    bspline_in.from_json(json);

    EXPECT_TRUE(bspline_in == bspline_out);

    // non-matching degree
    EXPECT_THROW((iganet::UniformBSpline<real_t, 4, 3, 3>{}.from_json(json)),
                 std::runtime_error);

    // non-matching parametric dimension
    EXPECT_THROW((iganet::UniformBSpline<real_t, 4, 3>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW((iganet::UniformBSpline<real_t, 4, 3, 4, 3>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 4, 3, 4, 3, 3>{}.from_json(json)),
        std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW((iganet::UniformBSpline<real_t, 1, 3, 4>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW((iganet::UniformBSpline<real_t, 2, 3, 4>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW((iganet::UniformBSpline<real_t, 3, 3, 4>{}.from_json(json)),
                 std::runtime_error);
  }

  {
    iganet::UniformBSpline<real_t, 1, 3, 4, 5> bspline_out(
        {4, 5, 6}, iganet::init::zeros, options);

    bspline_out.transform([](const std::array<real_t, 3> xi) {
      return std::array<real_t, 1>{static_cast<real_t>(std::rand())};
    });

    nlohmann::json json = bspline_out.to_json();

    iganet::UniformBSpline<real_t, 1, 3, 4, 5> bspline_in(options);
    bspline_in.from_json(json);

    EXPECT_TRUE(bspline_in == bspline_out);

    // non-matching degree
    EXPECT_THROW((iganet::UniformBSpline<real_t, 1, 3, 3, 3>{}.from_json(json)),
                 std::runtime_error);

    // non-matching parametric dimension
    EXPECT_THROW((iganet::UniformBSpline<real_t, 1, 3>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW((iganet::UniformBSpline<real_t, 1, 3, 4>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 1, 3, 4, 5, 3>{}.from_json(json)),
        std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW((iganet::UniformBSpline<real_t, 2, 3, 4, 5>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW((iganet::UniformBSpline<real_t, 3, 3, 4, 5>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW((iganet::UniformBSpline<real_t, 4, 3, 4, 5>{}.from_json(json)),
                 std::runtime_error);
  }

  {
    iganet::UniformBSpline<real_t, 2, 3, 4, 5> bspline_out(
        {4, 5, 6}, iganet::init::zeros, options);

    bspline_out.transform([](const std::array<real_t, 3> xi) {
      return std::array<real_t, 2>{static_cast<real_t>(std::rand()),
                                   static_cast<real_t>(std::rand())};
    });

    nlohmann::json json = bspline_out.to_json();

    iganet::UniformBSpline<real_t, 2, 3, 4, 5> bspline_in(options);
    bspline_in.from_json(json);

    EXPECT_TRUE(bspline_in == bspline_out);

    // non-matching degree
    EXPECT_THROW((iganet::UniformBSpline<real_t, 2, 3, 3, 3>{}.from_json(json)),
                 std::runtime_error);

    // non-matching parametric dimension
    EXPECT_THROW((iganet::UniformBSpline<real_t, 2, 3>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW((iganet::UniformBSpline<real_t, 2, 3, 4>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 2, 3, 4, 5, 3>{}.from_json(json)),
        std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW((iganet::UniformBSpline<real_t, 1, 3, 4, 5>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW((iganet::UniformBSpline<real_t, 3, 3, 4, 5>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW((iganet::UniformBSpline<real_t, 4, 3, 4, 5>{}.from_json(json)),
                 std::runtime_error);
  }

  {
    iganet::UniformBSpline<real_t, 3, 3, 4, 5> bspline_out(
        {4, 5, 6}, iganet::init::zeros, options);

    bspline_out.transform([](const std::array<real_t, 3> xi) {
      return std::array<real_t, 3>{static_cast<real_t>(std::rand()),
                                   static_cast<real_t>(std::rand()),
                                   static_cast<real_t>(std::rand())};
    });

    nlohmann::json json = bspline_out.to_json();

    iganet::UniformBSpline<real_t, 3, 3, 4, 5> bspline_in(options);
    bspline_in.from_json(json);

    EXPECT_TRUE(bspline_in == bspline_out);

    // non-matching degree
    EXPECT_THROW((iganet::UniformBSpline<real_t, 3, 3, 3, 3>{}.from_json(json)),
                 std::runtime_error);

    // non-matching parametric dimension
    EXPECT_THROW((iganet::UniformBSpline<real_t, 3, 3>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW((iganet::UniformBSpline<real_t, 3, 3, 4>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 3, 3, 4, 5, 3>{}.from_json(json)),
        std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW((iganet::UniformBSpline<real_t, 1, 3, 4, 5>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW((iganet::UniformBSpline<real_t, 2, 3, 4, 5>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW((iganet::UniformBSpline<real_t, 4, 3, 4, 5>{}.from_json(json)),
                 std::runtime_error);
  }

  {
    iganet::UniformBSpline<real_t, 4, 3, 4, 5> bspline_out(
        {4, 5, 6}, iganet::init::zeros, options);

    bspline_out.transform([](const std::array<real_t, 3> xi) {
      return std::array<real_t, 4>{
          static_cast<real_t>(std::rand()), static_cast<real_t>(std::rand()),
          static_cast<real_t>(std::rand()), static_cast<real_t>(std::rand())};
    });

    nlohmann::json json = bspline_out.to_json();

    iganet::UniformBSpline<real_t, 4, 3, 4, 5> bspline_in(options);
    bspline_in.from_json(json);

    EXPECT_TRUE(bspline_in == bspline_out);

    // non-matching degree
    EXPECT_THROW((iganet::UniformBSpline<real_t, 4, 3, 3, 3>{}.from_json(json)),
                 std::runtime_error);

    // non-matching parametric dimension
    EXPECT_THROW((iganet::UniformBSpline<real_t, 4, 3>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW((iganet::UniformBSpline<real_t, 4, 3, 4>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 4, 3, 4, 5, 3>{}.from_json(json)),
        std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW((iganet::UniformBSpline<real_t, 1, 3, 4, 5>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW((iganet::UniformBSpline<real_t, 2, 3, 4, 5>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW((iganet::UniformBSpline<real_t, 3, 3, 4, 5>{}.from_json(json)),
                 std::runtime_error);
  }

  {
    iganet::UniformBSpline<real_t, 1, 3, 4, 5, 1> bspline_out(
        {4, 5, 6, 2}, iganet::init::zeros, options);

    bspline_out.transform([](const std::array<real_t, 4> xi) {
      return std::array<real_t, 1>{static_cast<real_t>(std::rand())};
    });

    nlohmann::json json = bspline_out.to_json();

    iganet::UniformBSpline<real_t, 1, 3, 4, 5, 1> bspline_in(options);
    bspline_in.from_json(json);

    EXPECT_TRUE(bspline_in == bspline_out);

    // non-matching degree
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 1, 3, 4, 5, 2>{}.from_json(json)),
        std::runtime_error);

    // non-matching parametric dimension
    EXPECT_THROW((iganet::UniformBSpline<real_t, 1, 3>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW((iganet::UniformBSpline<real_t, 1, 3, 4>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW((iganet::UniformBSpline<real_t, 1, 3, 4, 5>{}.from_json(json)),
                 std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 2, 3, 4, 5, 1>{}.from_json(json)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 3, 3, 4, 5, 1>{}.from_json(json)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 4, 3, 4, 5, 1>{}.from_json(json)),
        std::runtime_error);
  }

  {
    iganet::UniformBSpline<real_t, 2, 3, 4, 5, 1> bspline_out(
        {4, 5, 6, 2}, iganet::init::zeros, options);

    bspline_out.transform([](const std::array<real_t, 4> xi) {
      return std::array<real_t, 2>{static_cast<real_t>(std::rand()),
                                   static_cast<real_t>(std::rand())};
    });

    nlohmann::json json = bspline_out.to_json();

    iganet::UniformBSpline<real_t, 2, 3, 4, 5, 1> bspline_in(options);
    bspline_in.from_json(json);

    EXPECT_TRUE(bspline_in == bspline_out);

    // non-matching degree
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 2, 3, 4, 5, 2>{}.from_json(json)),
        std::runtime_error);

    // non-matching parametric dimension
    EXPECT_THROW((iganet::UniformBSpline<real_t, 2, 3>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW((iganet::UniformBSpline<real_t, 2, 3, 4>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW((iganet::UniformBSpline<real_t, 2, 3, 4, 5>{}.from_json(json)),
                 std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 1, 3, 4, 5, 1>{}.from_json(json)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 3, 3, 4, 5, 1>{}.from_json(json)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 4, 3, 4, 5, 1>{}.from_json(json)),
        std::runtime_error);
  }

  {
    iganet::UniformBSpline<real_t, 3, 3, 4, 5, 1> bspline_out(
        {4, 5, 6, 2}, iganet::init::zeros, options);

    bspline_out.transform([](const std::array<real_t, 4> xi) {
      return std::array<real_t, 3>{static_cast<real_t>(std::rand()),
                                   static_cast<real_t>(std::rand()),
                                   static_cast<real_t>(std::rand())};
    });

    nlohmann::json json = bspline_out.to_json();

    iganet::UniformBSpline<real_t, 3, 3, 4, 5, 1> bspline_in(options);
    bspline_in.from_json(json);

    EXPECT_TRUE(bspline_in == bspline_out);

    // non-matching degree
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 3, 3, 4, 5, 2>{}.from_json(json)),
        std::runtime_error);

    // non-matching parametric dimension
    EXPECT_THROW((iganet::UniformBSpline<real_t, 3, 3>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW((iganet::UniformBSpline<real_t, 3, 3, 4>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW((iganet::UniformBSpline<real_t, 3, 3, 4, 5>{}.from_json(json)),
                 std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 1, 3, 4, 5, 1>{}.from_json(json)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 2, 3, 4, 5, 1>{}.from_json(json)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 4, 3, 4, 5, 1>{}.from_json(json)),
        std::runtime_error);
  }

  {
    iganet::UniformBSpline<real_t, 4, 3, 4, 5, 1> bspline_out(
        {4, 5, 6, 2}, iganet::init::zeros, options);

    bspline_out.transform([](const std::array<real_t, 4> xi) {
      return std::array<real_t, 4>{
          static_cast<real_t>(std::rand()), static_cast<real_t>(std::rand()),
          static_cast<real_t>(std::rand()), static_cast<real_t>(std::rand())};
    });

    nlohmann::json json = bspline_out.to_json();

    iganet::UniformBSpline<real_t, 4, 3, 4, 5, 1> bspline_in(options);
    bspline_in.from_json(json);

    EXPECT_TRUE(bspline_in == bspline_out);

    // non-matching degree
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 4, 3, 4, 5, 2>{}.from_json(json)),
        std::runtime_error);

    // non-matching parametric dimension
    EXPECT_THROW((iganet::UniformBSpline<real_t, 4, 3>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW((iganet::UniformBSpline<real_t, 4, 3, 4>{}.from_json(json)),
                 std::runtime_error);
    EXPECT_THROW((iganet::UniformBSpline<real_t, 4, 3, 4, 5>{}.from_json(json)),
                 std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 1, 3, 4, 5, 1>{}.from_json(json)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 2, 3, 4, 5, 1>{}.from_json(json)),
        std::runtime_error);
    EXPECT_THROW(
        (iganet::UniformBSpline<real_t, 3, 3, 4, 5, 1>{}.from_json(json)),
        std::runtime_error);
  }
}

TEST_F(BSplineTest, UniformBSpline_query_property) {
  iganet::UniformBSpline<real_t, 2, 3, 4> bspline(
      {4, 5}, iganet::init::greville, options);

  EXPECT_TRUE(bspline.is_uniform());
  EXPECT_FALSE(bspline.is_nonuniform());

  EXPECT_EQ(bspline.device(), options.device());
  EXPECT_EQ(bspline.device_index(), options.device_index());
  EXPECT_EQ(bspline.dtype(), options.dtype());
  EXPECT_EQ(bspline.is_sparse(), options.is_sparse());
  EXPECT_EQ(bspline.layout(), options.layout());
  EXPECT_EQ(bspline.pinned_memory(), options.pinned_memory());
}

TEST_F(BSplineTest, UniformBSpline_requires_grad) {
  {
    iganet::UniformBSpline<real_t, 2, 3, 4> bspline(
        {4, 5}, iganet::init::greville, options);

    EXPECT_FALSE(bspline.requires_grad());

    for (iganet::short_t i = 0; i < bspline.parDim(); ++i)
      EXPECT_FALSE(bspline.knots(i).requires_grad());

    for (iganet::short_t i = 0; i < bspline.geoDim(); ++i)
      EXPECT_FALSE(bspline.coeffs(i).requires_grad());

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
    iganet::UniformBSpline<real_t, 2, 3, 4> bspline(
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

TEST_F(BSplineTest, UniformBSpline_to_dtype) {
  {
    iganet::UniformBSpline<real_t, 2, 3, 4> bspline(
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
    iganet::UniformBSpline<real_t, 2, 3, 4> bspline(
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

TEST_F(BSplineTest, UniformBSpline_to_device) {
  {
    iganet::Options<real_t> options =
        iganet::Options<real_t>{}.device(torch::kCPU);
    iganet::UniformBSpline<real_t, 2, 3, 4> bspline(
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

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  iganet::init();
  return RUN_ALL_TESTS();
}
