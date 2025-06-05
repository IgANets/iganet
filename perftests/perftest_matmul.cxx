/**
   @file perftests/perftest_matmul.cxx

   @brief Matrix-matrix multiplication performance tests

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <iganet.h>

#include <chrono>
#include <iostream>

#include <gtest/gtest.h>
#include <perftest_config.hpp>

TEST(Performance, MatmulTensorLayout) {
  iganet::Options<iganet::perftests::real_t> options =
      iganet::Options<iganet::perftests::real_t>{}.requires_grad(false);

  for (iganet::short_t n : {2, 3, 4, 5}) {
    for (int64_t m : {100, 500, 1000, 5000, 10000, 50000, 100000}) {

      { // (n,m) data format
        torch::Tensor a = torch::ones({n, m}, options);
        torch::Tensor b = torch::ones({n, m}, options);
        torch::Tensor c;

        auto t1 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 100; i++)
          c = torch::sum(torch::mul(a, b), 0);
        auto t2 = std::chrono::high_resolution_clock::now();

        std::cout << "(" << std::right << std::setw(8) << n << "," << std::right
                  << std::setw(8) << m << ") " << std::right << std::setw(12)
                  << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 -
                                                                          t1)
                             .count() /
                         double(n * m * 100)
                  << " (ns/entry)";

        EXPECT_EQ(c.sizes(), c10::IntArrayRef({m}));
      }

      { // (m,n) data format
        torch::Tensor a = torch::ones({m, n}, options);
        torch::Tensor b = torch::ones({m, n}, options);
        torch::Tensor c;

        auto t1 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 100; i++)
          c = torch::sum(torch::mul(a, b), 1);
        auto t2 = std::chrono::high_resolution_clock::now();

        std::cout << "   (" << std::right << std::setw(8) << m << ","
                  << std::right << std::setw(8) << n << ") " << std::right
                  << std::setw(12)
                  << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 -
                                                                          t1)
                             .count() /
                         double(n * m * 100)
                  << " (ns/entry)" << std::endl;

        EXPECT_EQ(c.sizes(), c10::IntArrayRef({m}));
      }
    }
  }
}

int main(int argc, char **argv) {
  ::testing::GTEST_FLAG(filter) = ":-:*";
  ::testing::InitGoogleTest(&argc, argv);
  iganet::init();
  int result = RUN_ALL_TESTS();
  iganet::finalize();
  return result;
}
