/**
   @file unittests/unittest_options.cxx

   @brief Options unittests

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <complex>
#include <iganet.h>
#include <iostream>

#include <gtest/gtest.h>

TEST(Options, Options_default) {
  iganet::Options<double> options;

  EXPECT_EQ(options.dtype(), torch::kDouble);
  EXPECT_EQ(options.device(),
            torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
  EXPECT_EQ(options.layout(), torch::kStrided);
  EXPECT_FALSE(options.requires_grad());
  EXPECT_FALSE(options.pinned_memory());
  EXPECT_FALSE(options.is_sparse());
}

TEST(Options, Options_nondefault) {
  auto options = iganet::Options<float>{}
                     .device(torch::kCPU)
                     .layout(torch::kSparse)
                     .requires_grad(true);

  EXPECT_EQ(options.dtype(), torch::kFloat);
  EXPECT_EQ(options.device(), torch::kCPU);
  EXPECT_EQ(options.layout(), torch::kSparse);
  EXPECT_TRUE(options.requires_grad());
  EXPECT_FALSE(options.pinned_memory());
  EXPECT_TRUE(options.is_sparse());
}

TEST(Options, Options_dtype) {
  EXPECT_EQ(iganet::Options<double>{}.dtype(), torch::kDouble);
  EXPECT_EQ(iganet::Options<float>{}.dtype(), torch::kFloat);
  EXPECT_EQ(iganet::Options<iganet::half>{}.dtype(), torch::kHalf);
  EXPECT_EQ(iganet::Options<long>{}.dtype(), torch::kLong);
  EXPECT_EQ(iganet::Options<int>{}.dtype(), torch::kInt);
  EXPECT_EQ(iganet::Options<short>{}.dtype(), torch::kShort);
  EXPECT_EQ(iganet::Options<char>{}.dtype(), torch::kChar);
  EXPECT_EQ(iganet::Options<bool>{}.dtype(), torch::kBool);

  EXPECT_EQ(iganet::Options<std::complex<double>>{}.dtype(),
            torch::kComplexDouble);
  EXPECT_EQ(iganet::Options<std::complex<float>>{}.dtype(),
            torch::kComplexFloat);
  EXPECT_EQ(iganet::Options<std::complex<iganet::half>>{}.dtype(),
            torch::kComplexHalf);
}

TEST(Options, Options_clone) {
  auto options = iganet::Options<float>{}
                     .device(torch::kCPU)
                     .layout(torch::kSparse)
                     .requires_grad(false);

  auto options_clone(options);

  EXPECT_EQ(options_clone.dtype(), torch::kFloat);
  EXPECT_EQ(options_clone.device(), torch::kCPU);
  EXPECT_EQ(options_clone.layout(), torch::kSparse);
  EXPECT_FALSE(options_clone.requires_grad());
  EXPECT_FALSE(options_clone.pinned_memory());
  EXPECT_TRUE(options_clone.is_sparse());
}

TEST(Options, Options_conversion) {
  auto options = iganet::Options<float>{}
                     .device(torch::kCPU)
                     .layout(torch::kSparse)
                     .requires_grad(true);

  torch::TensorOptions tensorOptions(options);

  EXPECT_EQ(tensorOptions.dtype(), torch::kFloat);
  EXPECT_EQ(tensorOptions.device(), torch::kCPU);
  EXPECT_EQ(tensorOptions.layout(), torch::kSparse);
  EXPECT_TRUE(tensorOptions.requires_grad());
  EXPECT_FALSE(tensorOptions.pinned_memory());
  EXPECT_TRUE(tensorOptions.is_sparse());
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  iganet::init();
  return RUN_ALL_TESTS();
}
