/**
   @file perftests/perftest_bspline_eval.cxx

   @brief B-spline evaluation performance unittests

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <iganet.h>

#include <chrono>
#include <iomanip>
#include <iostream>

#include "../unittests/unittest_bsplinelib.hpp"
#include <gtest/gtest.h>
#include <perftest_config.hpp>

/// @brief Fixture for B-spline performance test
class BSplinePerformanceTest : public ::testing::Test {
public:
  /// @brief Evaluation functor
  ///
  /// @note GoogleTest does not support fixtures with multiple
  /// non-type template parameters. This functor is a work-around
  template <typename BSpline, iganet::deriv deriv, bool memory_optimized,
            bool precompute, bool requires_grad, bool bsplinelib>
  struct eval {

    /// @brief Call operator
    void operator()(int64_t ncoeffs, int64_t nsamples) {

      iganet::Options<iganet::perftests::real_t> options =
          iganet::Options<iganet::perftests::real_t>{}.requires_grad(
              requires_grad);

      // 1D
      if constexpr (BSpline::parDim() == 1) {

        BSpline bspline({ncoeffs}, iganet::init::linear, options);
        iganet::utils::TensorArray1 xi = {torch::rand(nsamples, options)};

        auto t1 = std::chrono::high_resolution_clock::now();

        if constexpr (precompute) {
          auto knot_idx = bspline.find_knot_indices(xi);
          auto basfunc = bspline.template eval_basfunc<deriv, memory_optimized>(
              xi, knot_idx);
          auto coeff_idx =
              bspline.template find_coeff_indices<memory_optimized>(knot_idx);
          for (int i = 0; i < 10; i++)
            auto bspline_val = bspline.eval_from_precomputed(
                basfunc, coeff_idx, xi[0].numel(), xi[0].sizes());
        } else
          for (int i = 0; i < 10; i++)
            bspline.template eval<deriv, memory_optimized>(xi);

        auto t2 = std::chrono::high_resolution_clock::now();

        std::cout << std::right << std::setw(10)
                  << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 -
                                                                          t1)
                             .count() /
                         double(nsamples * 10)
                  << "\t";

        if constexpr (bsplinelib) {
          if (nsamples == 1) {
            auto splinelib_bspline = to_bsplinelib_bspline(bspline);

            // B-spline evaluation (parametric dim = 1)
            using ParametricCoordinate =
                typename decltype(splinelib_bspline)::ParametricCoordinate_;
            using Derivative = typename decltype(splinelib_bspline)::
                ParameterSpace_::Derivative_;
            using Coordinate =
                typename decltype(splinelib_bspline)::Coordinate_;

            ParametricCoordinate query{0.5};
            Derivative der_query{(iganet::short_t)deriv % 10};
            Coordinate result(BSpline::geoDim());

            auto t1 = std::chrono::high_resolution_clock::now();

            for (int i = 0; i < 1000; i++)
              splinelib_bspline.EvaluateDerivative(
                  query.data(), der_query.data(), result.data());

            auto t2 = std::chrono::high_resolution_clock::now();

            std::cout << std::right << std::setw(10)
                      << std::chrono::duration_cast<std::chrono::nanoseconds>(
                             t2 - t1)
                                 .count() /
                             double(1000)
                      << "\t";
          } else {
            std::cout << std::right << std::setw(10) << "--"
                      << "\t";
          }
        }
      }

      // 2D
      else if constexpr (BSpline::parDim() == 2) {

        BSpline bspline({ncoeffs, ncoeffs}, iganet::init::linear, options);
        iganet::utils::TensorArray2 xi = {torch::rand(nsamples, options),
                                          torch::rand(nsamples, options)};

        auto t1 = std::chrono::high_resolution_clock::now();

        if constexpr (precompute) {
          auto knot_idx = bspline.find_knot_indices(xi);
          auto basfunc = bspline.template eval_basfunc<deriv, memory_optimized>(
              xi, knot_idx);
          auto coeff_idx =
              bspline.template find_coeff_indices<memory_optimized>(knot_idx);
          for (int i = 0; i < 10; i++)
            auto bspline_val = bspline.eval_from_precomputed(
                basfunc, coeff_idx, xi[0].numel(), xi[0].sizes());
        } else
          for (int i = 0; i < 10; i++)
            bspline.template eval<deriv, memory_optimized>(xi);

        auto t2 = std::chrono::high_resolution_clock::now();

        std::cout << std::right << std::setw(10)
                  << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 -
                                                                          t1)
                             .count() /
                         double(nsamples * 10)
                  << "\t";

        if constexpr (bsplinelib) {
          if (nsamples == 1) {
            auto splinelib_bspline = to_bsplinelib_bspline(bspline);

            // B-spline evaluation
            using ParametricCoordinate =
                typename decltype(splinelib_bspline)::ParametricCoordinate_;
            using Derivative = typename decltype(splinelib_bspline)::
                ParameterSpace_::Derivative_;
            using Coordinate =
                typename decltype(splinelib_bspline)::Coordinate_;

            ParametricCoordinate query{0.5, 0.5};
            Derivative der_query{(iganet::short_t)deriv % 10,
                                 ((iganet::short_t)deriv / 10) % 10};
            Coordinate result(BSpline::geoDim());

            auto t1 = std::chrono::high_resolution_clock::now();

            for (int i = 0; i < 1000; i++)
              splinelib_bspline.EvaluateDerivative(
                  query.data(), der_query.data(), result.data());

            auto t2 = std::chrono::high_resolution_clock::now();

            std::cout << std::right << std::setw(10)
                      << std::chrono::duration_cast<std::chrono::nanoseconds>(
                             t2 - t1)
                                 .count() /
                             double(1000)
                      << "\t";
          } else {
            std::cout << std::right << std::setw(10) << "--"
                      << "\t";
          }
        }
      }

      // 3D
      else if constexpr (BSpline::parDim() == 3) {

        BSpline bspline({ncoeffs, ncoeffs, ncoeffs}, iganet::init::linear,
                        options);
        iganet::utils::TensorArray3 xi = {torch::rand(nsamples, options),
                                          torch::rand(nsamples, options),
                                          torch::rand(nsamples, options)};

        auto t1 = std::chrono::high_resolution_clock::now();

        if constexpr (precompute) {
          auto knot_idx = bspline.find_knot_indices(xi);
          auto basfunc = bspline.template eval_basfunc<deriv, memory_optimized>(
              xi, knot_idx);
          auto coeff_idx =
              bspline.template find_coeff_indices<memory_optimized>(knot_idx);
          for (int i = 0; i < 10; i++)
            auto bspline_val = bspline.eval_from_precomputed(
                basfunc, coeff_idx, xi[0].numel(), xi[0].sizes());
        } else
          for (int i = 0; i < 10; i++)
            bspline.template eval<deriv, memory_optimized>(xi);

        auto t2 = std::chrono::high_resolution_clock::now();

        std::cout << std::right << std::setw(10)
                  << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 -
                                                                          t1)
                             .count() /
                         double(nsamples * 10)
                  << "\t";

        if constexpr (bsplinelib) {
          if (nsamples == 1) {
            auto splinelib_bspline = to_bsplinelib_bspline(bspline);

            // B-spline evaluation
            using ParametricCoordinate =
                typename decltype(splinelib_bspline)::ParametricCoordinate_;
            using Derivative = typename decltype(splinelib_bspline)::
                ParameterSpace_::Derivative_;
            using Coordinate =
                typename decltype(splinelib_bspline)::Coordinate_;

            ParametricCoordinate query{0.5, 0.5, 0.5};
            Derivative der_query{(iganet::short_t)deriv % 10,
                                 ((iganet::short_t)deriv / 10) % 10,
                                 ((iganet::short_t)deriv / 100) % 10};
            Coordinate result(BSpline::geoDim());

            auto t1 = std::chrono::high_resolution_clock::now();

            for (int i = 0; i < 1000; i++)
              splinelib_bspline.EvaluateDerivative(
                  query.data(), der_query.data(), result.data());

            auto t2 = std::chrono::high_resolution_clock::now();

            std::cout << std::right << std::setw(10)
                      << std::chrono::duration_cast<std::chrono::nanoseconds>(
                             t2 - t1)
                                 .count() /
                             double(1000)
                      << "\t";
          } else {
            std::cout << std::right << std::setw(10) << "--"
                      << "\t";
          }
        }
      }

      // 4D
      else if constexpr (BSpline::parDim() == 4) {

        BSpline bspline({ncoeffs, ncoeffs, ncoeffs, ncoeffs},
                        iganet::init::linear, options);
        iganet::utils::TensorArray4 xi = {
            torch::rand(nsamples, options), torch::rand(nsamples, options),
            torch::rand(nsamples, options), torch::rand(nsamples, options)};

        auto t1 = std::chrono::high_resolution_clock::now();

        if constexpr (precompute) {
          auto knot_idx = bspline.find_knot_indices(xi);
          auto basfunc = bspline.template eval_basfunc<deriv, memory_optimized>(
              xi, knot_idx);
          auto coeff_idx =
              bspline.template find_coeff_indices<memory_optimized>(knot_idx);
          for (int i = 0; i < 10; i++)
            auto bspline_val = bspline.eval_from_precomputed(
                basfunc, coeff_idx, xi[0].numel(), xi[0].sizes());
        } else
          for (int i = 0; i < 10; i++)
            bspline.template eval<deriv, memory_optimized>(xi);
        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << std::right << std::setw(10)
                  << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 -
                                                                          t1)
                             .count() /
                         double(nsamples * 10)
                  << "\t";

        if constexpr (bsplinelib) {
          if (nsamples == 1) {
            auto splinelib_bspline = to_bsplinelib_bspline(bspline);

            // B-spline evaluation
            using ParametricCoordinate =
                typename decltype(splinelib_bspline)::ParametricCoordinate_;
            using Derivative = typename decltype(splinelib_bspline)::
                ParameterSpace_::Derivative_;
            using Coordinate =
                typename decltype(splinelib_bspline)::Coordinate_;

            ParametricCoordinate query{0.5, 0.5, 0.5, 0.5};
            Derivative der_query{(iganet::short_t)deriv % 10,
                                 ((iganet::short_t)deriv / 10) % 10,
                                 ((iganet::short_t)deriv / 100) % 10,
                                 ((iganet::short_t)deriv / 1000) % 10};
            Coordinate result(BSpline::geoDim());

            auto t1 = std::chrono::high_resolution_clock::now();

            for (int i = 0; i < 1000; i++)
              splinelib_bspline.EvaluateDerivative(
                  query.data(), der_query.data(), result.data());

            auto t2 = std::chrono::high_resolution_clock::now();

            std::cout << std::right << std::setw(10)
                      << std::chrono::duration_cast<std::chrono::nanoseconds>(
                             t2 - t1)
                                 .count() /
                             double(1000)
                      << "\t";
          } else {
            std::cout << std::right << std::setw(10) << "--"
                      << "\t";
          }
        }
      }

      else
        throw std::runtime_error("Unsupported parametric dimension");
    }
  };

  template <typename BSpline, iganet::deriv deriv, bool memory_optimized,
            bool precompute, bool requires_grad, bool bsplinelib>
  static auto eval1(int64_t ncoeffs, int64_t nsamples) {

    iganet::Options<iganet::perftests::real_t> options =
        iganet::Options<iganet::perftests::real_t>{}.requires_grad(
            requires_grad);

    // 1D
    if constexpr (BSpline::parDim() == 1) {

      BSpline bspline({ncoeffs}, iganet::init::linear, options);
      iganet::utils::TensorArray1 xi = {torch::rand(nsamples, options)};

      auto t1 = std::chrono::high_resolution_clock::now();

      if constexpr (precompute) {
        auto knot_idx = bspline.find_knot_indices(xi);
        auto basfunc = bspline.template eval_basfunc<deriv, memory_optimized>(
            xi, knot_idx);
        auto coeff_idx =
            bspline.template find_coeff_indices<memory_optimized>(knot_idx);
        for (int i = 0; i < 10; i++)
          auto bspline_val = bspline.eval_from_precomputed(
              basfunc, coeff_idx, xi[0].numel(), xi[0].sizes());
      } else
        for (int i = 0; i < 10; i++)
          bspline.template eval<deriv, memory_optimized>(xi);

      auto t2 = std::chrono::high_resolution_clock::now();

      std::cout << std::right << std::setw(10)
                << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1)
                           .count() /
                       double(nsamples * 10)
                << "\t";

      if constexpr (bsplinelib) {
        if (nsamples == 1) {
          auto splinelib_bspline = to_bsplinelib_bspline(bspline);

          // B-spline evaluation (parametric dim = 1)
          using ParametricCoordinate =
              typename decltype(splinelib_bspline)::ParametricCoordinate_;
          using Derivative = typename decltype(splinelib_bspline)::
              ParameterSpace_::Derivative_;
          using Coordinate = typename decltype(splinelib_bspline)::Coordinate_;

          ParametricCoordinate query{0.5};
          Derivative der_query{(iganet::short_t)deriv % 10};
          Coordinate result(BSpline::geoDim());

          auto t1 = std::chrono::high_resolution_clock::now();

          for (int i = 0; i < 1000; i++)
            splinelib_bspline.EvaluateDerivative(query.data(), der_query.data(),
                                                 result.data());

          auto t2 = std::chrono::high_resolution_clock::now();

          std::cout << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 -
                                                                            t1)
                               .count() /
                           double(1000)
                    << "\t";
        } else {
          std::cout << std::right << std::setw(10) << "--"
                    << "\t";
        }
      }

      // 2D
      else if constexpr (BSpline::parDim() == 2) {

        BSpline bspline({ncoeffs, ncoeffs}, iganet::init::linear, options);
        iganet::utils::TensorArray2 xi = {torch::rand(nsamples, options),
                                          torch::rand(nsamples, options)};

        auto t1 = std::chrono::high_resolution_clock::now();

        if constexpr (precompute) {
          auto knot_idx = bspline.find_knot_indices(xi);
          auto basfunc = bspline.template eval_basfunc<deriv, memory_optimized>(
              xi, knot_idx);
          auto coeff_idx =
              bspline.template find_coeff_indices<memory_optimized>(knot_idx);
          for (int i = 0; i < 10; i++)
            auto bspline_val = bspline.eval_from_precomputed(
                basfunc, coeff_idx, xi[0].numel(), xi[0].sizes());
        } else
          for (int i = 0; i < 10; i++)
            bspline.template eval<deriv, memory_optimized>(xi);

        auto t2 = std::chrono::high_resolution_clock::now();

        std::cout << std::right << std::setw(10)
                  << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 -
                                                                          t1)
                             .count() /
                         double(nsamples * 10)
                  << "\t";

        if constexpr (bsplinelib) {
          if (nsamples == 1) {
            auto splinelib_bspline = to_bsplinelib_bspline(bspline);

            // B-spline evaluation
            using ParametricCoordinate =
                typename decltype(splinelib_bspline)::ParametricCoordinate_;
            using Derivative = typename decltype(splinelib_bspline)::
                ParameterSpace_::Derivative_;
            using Coordinate =
                typename decltype(splinelib_bspline)::Coordinate_;

            ParametricCoordinate query{0.5, 0.5};
            Derivative der_query{(iganet::short_t)deriv % 10,
                                 ((iganet::short_t)deriv / 10) % 10};
            Coordinate result(BSpline::geoDim());

            auto t1 = std::chrono::high_resolution_clock::now();

            for (int i = 0; i < 1000; i++)
              splinelib_bspline.EvaluateDerivative(
                  query.data(), der_query.data(), result.data());

            auto t2 = std::chrono::high_resolution_clock::now();

            std::cout << std::right << std::setw(10)
                      << std::chrono::duration_cast<std::chrono::nanoseconds>(
                             t2 - t1)
                                 .count() /
                             double(1000)
                      << "\t";
          } else {
            std::cout << std::right << std::setw(10) << "--"
                      << "\t";
          }
        }
      }

      // 3D
      else if constexpr (BSpline::parDim() == 3) {

        BSpline bspline({ncoeffs, ncoeffs, ncoeffs}, iganet::init::linear,
                        options);
        iganet::utils::TensorArray3 xi = {torch::rand(nsamples, options),
                                          torch::rand(nsamples, options),
                                          torch::rand(nsamples, options)};

        auto t1 = std::chrono::high_resolution_clock::now();

        if constexpr (precompute) {
          auto knot_idx = bspline.find_knot_indices(xi);
          auto basfunc = bspline.template eval_basfunc<deriv, memory_optimized>(
              xi, knot_idx);
          auto coeff_idx =
              bspline.template find_coeff_indices<memory_optimized>(knot_idx);
          for (int i = 0; i < 10; i++)
            auto bspline_val = bspline.eval_from_precomputed(
                basfunc, coeff_idx, xi[0].numel(), xi[0].sizes());
        } else
          for (int i = 0; i < 10; i++)
            bspline.template eval<deriv, memory_optimized>(xi);

        auto t2 = std::chrono::high_resolution_clock::now();

        std::cout << std::right << std::setw(10)
                  << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 -
                                                                          t1)
                             .count() /
                         double(nsamples * 10)
                  << "\t";

        if constexpr (bsplinelib) {
          if (nsamples == 1) {
            auto splinelib_bspline = to_bsplinelib_bspline(bspline);

            // B-spline evaluation
            using ParametricCoordinate =
                typename decltype(splinelib_bspline)::ParametricCoordinate_;
            using Derivative = typename decltype(splinelib_bspline)::
                ParameterSpace_::Derivative_;
            using Coordinate =
                typename decltype(splinelib_bspline)::Coordinate_;

            ParametricCoordinate query{0.5, 0.5, 0.5};
            Derivative der_query{(iganet::short_t)deriv % 10,
                                 ((iganet::short_t)deriv / 10) % 10,
                                 ((iganet::short_t)deriv / 100) % 10};
            Coordinate result(BSpline::geoDim());

            auto t1 = std::chrono::high_resolution_clock::now();

            for (int i = 0; i < 1000; i++)
              splinelib_bspline.EvaluateDerivative(
                  query.data(), der_query.data(), result.data());

            auto t2 = std::chrono::high_resolution_clock::now();

            std::cout << std::right << std::setw(10)
                      << std::chrono::duration_cast<std::chrono::nanoseconds>(
                             t2 - t1)
                                 .count() /
                             double(1000)
                      << "\t";
          } else {
            std::cout << std::right << std::setw(10) << "--"
                      << "\t";
          }
        }
      }

      // 4D
      else if constexpr (BSpline::parDim() == 4) {

        BSpline bspline({ncoeffs, ncoeffs, ncoeffs, ncoeffs},
                        iganet::init::linear, options);
        iganet::utils::TensorArray4 xi = {
            torch::rand(nsamples, options), torch::rand(nsamples, options),
            torch::rand(nsamples, options), torch::rand(nsamples, options)};

        auto t1 = std::chrono::high_resolution_clock::now();

        if constexpr (precompute) {
          auto knot_idx = bspline.find_knot_indices(xi);
          auto basfunc = bspline.template eval_basfunc<deriv, memory_optimized>(
              xi, knot_idx);
          auto coeff_idx =
              bspline.template find_coeff_indices<memory_optimized>(knot_idx);
          for (int i = 0; i < 10; i++)
            auto bspline_val = bspline.eval_from_precomputed(
                basfunc, coeff_idx, xi[0].numel(), xi[0].sizes());
        } else
          for (int i = 0; i < 10; i++)
            bspline.template eval<deriv, memory_optimized>(xi);
        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << std::right << std::setw(10)
                  << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 -
                                                                          t1)
                             .count() /
                         double(nsamples * 10)
                  << "\t";

        if constexpr (bsplinelib) {
          if (nsamples == 1) {
            auto splinelib_bspline = to_bsplinelib_bspline(bspline);

            // B-spline evaluation
            using ParametricCoordinate =
                typename decltype(splinelib_bspline)::ParametricCoordinate_;
            using Derivative = typename decltype(splinelib_bspline)::
                ParameterSpace_::Derivative_;
            using Coordinate =
                typename decltype(splinelib_bspline)::Coordinate_;

            ParametricCoordinate query{0.5, 0.5, 0.5, 0.5};
            Derivative der_query{(iganet::short_t)deriv % 10,
                                 ((iganet::short_t)deriv / 10) % 10,
                                 ((iganet::short_t)deriv / 100) % 10,
                                 ((iganet::short_t)deriv / 1000) % 10};
            Coordinate result(BSpline::geoDim());

            auto t1 = std::chrono::high_resolution_clock::now();

            for (int i = 0; i < 1000; i++)
              splinelib_bspline.EvaluateDerivative(
                  query.data(), der_query.data(), result.data());

            auto t2 = std::chrono::high_resolution_clock::now();

            std::cout << std::right << std::setw(10)
                      << std::chrono::duration_cast<std::chrono::nanoseconds>(
                             t2 - t1)
                                 .count() /
                             double(1000)
                      << "\t";
          } else {
            std::cout << std::right << std::setw(10) << "--"
                      << "\t";
          }
        }
      }

      else
        throw std::runtime_error("Unsupported parametric dimension");
    }
  }
};

template <bool memory_optimized, bool precompute, bool requires_grad>
void make_test_UniformBSpline_parDim1() {

  std::cout << "memory_optimized : " << memory_optimized
            << ", precompute : " << precompute
            << ", requires_grad : " << requires_grad << std::endl;

  std::cout << std::scientific << std::setprecision(3);

  for (int64_t ncoeffs :
       iganet::utils::getenv("IGANET_NCOEFFS", {10, 100, 1000})) {
    for (int64_t nsamples : iganet::utils::getenv(
             "IGANET_NSAMPLES", {1, 10, 100, 1000, 10000, 25000, 50000, 100000,
                                 250000, 500000, 1000000})) {

      std::cout << std::right << std::setw(8) << ncoeffs << "\t" << std::right
                << std::setw(8) << nsamples << "\t";

      std::apply(
          [&](auto &&...args) { ((args(ncoeffs, nsamples)), ...); },
          std::make_tuple(
              BSplinePerformanceTest::eval<
                  iganet::UniformBSpline<iganet::perftests::real_t, 1, 1>,
                  iganet::deriv::func, memory_optimized, precompute,
                  requires_grad, true>{},
              BSplinePerformanceTest::eval<
                  iganet::UniformBSpline<iganet::perftests::real_t, 1, 2>,
                  iganet::deriv::func, memory_optimized, precompute,
                  requires_grad, true>{},
              BSplinePerformanceTest::eval<
                  iganet::UniformBSpline<iganet::perftests::real_t, 1, 3>,
                  iganet::deriv::func, memory_optimized, precompute,
                  requires_grad, true>{},
              BSplinePerformanceTest::eval<
                  iganet::UniformBSpline<iganet::perftests::real_t, 1, 4>,
                  iganet::deriv::func, memory_optimized, precompute,
                  requires_grad, true>{},
              BSplinePerformanceTest::eval<
                  iganet::UniformBSpline<iganet::perftests::real_t, 1, 5>,
                  iganet::deriv::func, memory_optimized, precompute,
                  requires_grad, true>{}));

      std::cout << std::endl;
    }
  }
}

template <bool memory_optimized, bool precompute, bool requires_grad>
void make_test_NonUniformBSpline_parDim1() {

  std::cout << "memory_optimized : " << memory_optimized
            << ", precompute : " << precompute
            << ", requires_grad : " << requires_grad << std::endl;

  std::cout << std::scientific << std::setprecision(3);

  for (int64_t ncoeffs :
       iganet::utils::getenv("IGANET_NCOEFFS", {10, 100, 1000})) {
    for (int64_t nsamples : iganet::utils::getenv(
             "IGANET_NSAMPLES", {1, 10, 100, 1000, 10000, 25000, 50000, 100000,
                                 250000, 500000, 1000000})) {

      std::cout << std::right << std::setw(8) << ncoeffs << "\t" << std::right
                << std::setw(8) << nsamples << "\t";

      std::apply(
          [&](auto &&...args) { ((args(ncoeffs, nsamples)), ...); },
          std::make_tuple(
              BSplinePerformanceTest::eval<
                  iganet::NonUniformBSpline<iganet::perftests::real_t, 1, 1>,
                  iganet::deriv::func, memory_optimized, precompute,
                  requires_grad, true>{},
              BSplinePerformanceTest::eval<
                  iganet::NonUniformBSpline<iganet::perftests::real_t, 1, 2>,
                  iganet::deriv::func, memory_optimized, precompute,
                  requires_grad, true>{},
              BSplinePerformanceTest::eval<
                  iganet::NonUniformBSpline<iganet::perftests::real_t, 1, 3>,
                  iganet::deriv::func, memory_optimized, precompute,
                  requires_grad, true>{},
              BSplinePerformanceTest::eval<
                  iganet::NonUniformBSpline<iganet::perftests::real_t, 1, 4>,
                  iganet::deriv::func, memory_optimized, precompute,
                  requires_grad, true>{},
              BSplinePerformanceTest::eval<
                  iganet::NonUniformBSpline<iganet::perftests::real_t, 1, 5>,
                  iganet::deriv::func, memory_optimized, precompute,
                  requires_grad, true>{}));

      std::cout << std::endl;
    }
  }
}

template <bool memory_optimized, bool precompute, bool requires_grad>
void make_test_UniformBSpline_parDim2() {

  std::cout << "memory_optimized : " << memory_optimized
            << ", precompute : " << precompute
            << ", requires_grad : " << requires_grad << std::endl;

  std::cout << std::scientific << std::setprecision(3);

  for (int64_t ncoeffs :
       iganet::utils::getenv("IGANET_NCOEFFS", {10, 100, 1000})) {
    for (int64_t nsamples : iganet::utils::getenv(
             "IGANET_NSAMPLES", {1, 10, 100, 1000, 10000, 25000, 50000, 100000,
                                 250000, 500000, 1000000})) {

      std::cout << std::right << std::setw(8) << ncoeffs << "\t" << std::right
                << std::setw(8) << nsamples << "\t";

      std::apply(
          [&](auto &&...args) { ((args(ncoeffs, nsamples)), ...); },
          std::make_tuple(
              BSplinePerformanceTest::eval<
                  iganet::UniformBSpline<iganet::perftests::real_t, 1, 1, 1>,
                  iganet::deriv::func, memory_optimized, precompute,
                  requires_grad, true>{},
              BSplinePerformanceTest::eval<
                  iganet::UniformBSpline<iganet::perftests::real_t, 1, 2, 2>,
                  iganet::deriv::func, memory_optimized, precompute,
                  requires_grad, true>{},
              BSplinePerformanceTest::eval<
                  iganet::UniformBSpline<iganet::perftests::real_t, 1, 3, 3>,
                  iganet::deriv::func, memory_optimized, precompute,
                  requires_grad, true>{},
              BSplinePerformanceTest::eval<
                  iganet::UniformBSpline<iganet::perftests::real_t, 1, 4, 4>,
                  iganet::deriv::func, memory_optimized, precompute,
                  requires_grad, true>{},
              BSplinePerformanceTest::eval<
                  iganet::UniformBSpline<iganet::perftests::real_t, 1, 5, 5>,
                  iganet::deriv::func, memory_optimized, precompute,
                  requires_grad, true>{}));

      std::cout << std::endl;
    }
  }
}

template <bool memory_optimized, bool precompute, bool requires_grad>
void make_test_NonUniformBSpline_parDim2() {

  std::cout << "memory_optimized : " << memory_optimized
            << ", precompute : " << precompute
            << ", requires_grad : " << requires_grad << std::endl;

  std::cout << std::scientific << std::setprecision(3);

  for (int64_t ncoeffs :
       iganet::utils::getenv("IGANET_NCOEFFS", {10, 100, 1000})) {
    for (int64_t nsamples : iganet::utils::getenv(
             "IGANET_NSAMPLES", {1, 10, 100, 1000, 10000, 25000, 50000, 100000,
                                 250000, 500000, 1000000})) {

      std::cout << std::right << std::setw(8) << ncoeffs << "\t" << std::right
                << std::setw(8) << nsamples << "\t";

      std::apply(
          [&](auto &&...args) { ((args(ncoeffs, nsamples)), ...); },
          std::make_tuple(
              BSplinePerformanceTest::eval<
                  iganet::NonUniformBSpline<iganet::perftests::real_t, 1, 1, 1>,
                  iganet::deriv::func, memory_optimized, precompute,
                  requires_grad, true>{},
              BSplinePerformanceTest::eval<
                  iganet::NonUniformBSpline<iganet::perftests::real_t, 1, 2, 2>,
                  iganet::deriv::func, memory_optimized, precompute,
                  requires_grad, true>{},
              BSplinePerformanceTest::eval<
                  iganet::NonUniformBSpline<iganet::perftests::real_t, 1, 3, 3>,
                  iganet::deriv::func, memory_optimized, precompute,
                  requires_grad, true>{},
              BSplinePerformanceTest::eval<
                  iganet::NonUniformBSpline<iganet::perftests::real_t, 1, 4, 4>,
                  iganet::deriv::func, memory_optimized, precompute,
                  requires_grad, true>{},
              BSplinePerformanceTest::eval<
                  iganet::NonUniformBSpline<iganet::perftests::real_t, 1, 5, 5>,
                  iganet::deriv::func, memory_optimized, precompute,
                  requires_grad, true>{}));

      std::cout << std::endl;
    }
  }
}

template <bool memory_optimized, bool precompute, bool requires_grad>
void make_test_UniformBSpline_parDim3() {

  std::cout << "memory_optimized : " << memory_optimized
            << ", precompute : " << precompute
            << ", requires_grad : " << requires_grad << std::endl;

  std::cout << std::scientific << std::setprecision(3);

  for (int64_t ncoeffs :
       iganet::utils::getenv("IGANET_NCOEFFS", {10, 100, 1000})) {
    for (int64_t nsamples : iganet::utils::getenv(
             "IGANET_NSAMPLES", {1, 10, 100, 1000, 10000, 25000, 50000, 100000,
                                 250000, 500000, 1000000})) {

      std::cout << std::right << std::setw(8) << ncoeffs << "\t" << std::right
                << std::setw(8) << nsamples << "\t";

      std::apply(
          [&](auto &&...args) { ((args(ncoeffs, nsamples)), ...); },
          std::make_tuple(
              BSplinePerformanceTest::eval<
                  iganet::UniformBSpline<iganet::perftests::real_t, 1, 1, 1, 1>,
                  iganet::deriv::func, memory_optimized, precompute,
                  requires_grad, true>{},
              BSplinePerformanceTest::eval<
                  iganet::UniformBSpline<iganet::perftests::real_t, 1, 2, 2, 2>,
                  iganet::deriv::func, memory_optimized, precompute,
                  requires_grad, true>{},
              BSplinePerformanceTest::eval<
                  iganet::UniformBSpline<iganet::perftests::real_t, 1, 3, 3, 3>,
                  iganet::deriv::func, memory_optimized, precompute,
                  requires_grad, true>{},
              BSplinePerformanceTest::eval<
                  iganet::UniformBSpline<iganet::perftests::real_t, 1, 4, 4, 4>,
                  iganet::deriv::func, memory_optimized, precompute,
                  requires_grad, true>{},
              BSplinePerformanceTest::eval<
                  iganet::UniformBSpline<iganet::perftests::real_t, 1, 5, 5, 5>,
                  iganet::deriv::func, memory_optimized, precompute,
                  requires_grad, true>{}));

      std::cout << std::endl;
    }
  }
}

template <bool memory_optimized, bool precompute, bool requires_grad>
void make_test_NonUniformBSpline_parDim3() {

  std::cout << "memory_optimized : " << memory_optimized
            << ", precompute : " << precompute
            << ", requires_grad : " << requires_grad << std::endl;

  std::cout << std::scientific << std::setprecision(3);

  for (int64_t ncoeffs :
       iganet::utils::getenv("IGANET_NCOEFFS", {10, 100, 1000})) {
    for (int64_t nsamples : iganet::utils::getenv(
             "IGANET_NSAMPLES", {1, 10, 100, 1000, 10000, 25000, 50000, 100000,
                                 250000, 500000, 1000000})) {

      std::cout << std::right << std::setw(8) << ncoeffs << "\t" << std::right
                << std::setw(8) << nsamples << "\t";

      std::apply([&](auto &&...args) { ((args(ncoeffs, nsamples)), ...); },
                 std::make_tuple(
                     BSplinePerformanceTest::eval<
                         iganet::NonUniformBSpline<iganet::perftests::real_t, 1,
                                                   1, 1, 1>,
                         iganet::deriv::func, memory_optimized, precompute,
                         requires_grad, true>{},
                     BSplinePerformanceTest::eval<
                         iganet::NonUniformBSpline<iganet::perftests::real_t, 1,
                                                   2, 2, 2>,
                         iganet::deriv::func, memory_optimized, precompute,
                         requires_grad, true>{},
                     BSplinePerformanceTest::eval<
                         iganet::NonUniformBSpline<iganet::perftests::real_t, 1,
                                                   3, 3, 3>,
                         iganet::deriv::func, memory_optimized, precompute,
                         requires_grad, true>{},
                     BSplinePerformanceTest::eval<
                         iganet::NonUniformBSpline<iganet::perftests::real_t, 1,
                                                   4, 4, 4>,
                         iganet::deriv::func, memory_optimized, precompute,
                         requires_grad, true>{},
                     BSplinePerformanceTest::eval<
                         iganet::NonUniformBSpline<iganet::perftests::real_t, 1,
                                                   5, 5, 5>,
                         iganet::deriv::func, memory_optimized, precompute,
                         requires_grad, true>{}));

      std::cout << std::endl;
    }
  }
}

TEST_F(BSplinePerformanceTest, UniformBSpline_parDim1_memopt_precomp_nograd) {
  make_test_UniformBSpline_parDim1<true, true, false>();
}

TEST_F(BSplinePerformanceTest, UniformBSpline_parDim1_memopt_precomp_grad) {
  make_test_UniformBSpline_parDim1<true, true, true>();
}

TEST_F(BSplinePerformanceTest, UniformBSpline_parDim1_memopt_noprecomp_nograd) {
  make_test_UniformBSpline_parDim1<true, false, false>();
}

TEST_F(BSplinePerformanceTest, UniformBSpline_parDim1_memopt_noprecomp_grad) {
  make_test_UniformBSpline_parDim1<true, false, true>();
}

TEST_F(BSplinePerformanceTest, UniformBSpline_parDim1_nomemopt_precomp_nograd) {
  make_test_UniformBSpline_parDim1<false, true, false>();
}

TEST_F(BSplinePerformanceTest, UniformBSpline_parDim1_nomemopt_precomp_grad) {
  make_test_UniformBSpline_parDim1<false, true, true>();
}

TEST_F(BSplinePerformanceTest,
       UniformBSpline_parDim1_nomemopt_noprecomp_nograd) {
  make_test_UniformBSpline_parDim1<false, false, false>();
}

TEST_F(BSplinePerformanceTest, UniformBSpline_parDim1_nomemopt_noprecomp_grad) {
  make_test_UniformBSpline_parDim1<false, false, true>();
}

TEST_F(BSplinePerformanceTest, UniformBSpline_parDim2_memopt_precomp_nograd) {
  make_test_UniformBSpline_parDim2<true, true, false>();
}

TEST_F(BSplinePerformanceTest, UniformBSpline_parDim2_memopt_precomp_grad) {
  make_test_UniformBSpline_parDim2<true, true, true>();
}

TEST_F(BSplinePerformanceTest, UniformBSpline_parDim2_memopt_noprecomp_nograd) {
  make_test_UniformBSpline_parDim2<true, false, false>();
}

TEST_F(BSplinePerformanceTest, UniformBSpline_parDim2_memopt_noprecomp_grad) {
  make_test_UniformBSpline_parDim2<true, false, true>();
}

TEST_F(BSplinePerformanceTest, UniformBSpline_parDim2_nomemopt_precomp_nograd) {
  make_test_UniformBSpline_parDim2<false, true, false>();
}

TEST_F(BSplinePerformanceTest, UniformBSpline_parDim2_nomemopt_precomp_grad) {
  make_test_UniformBSpline_parDim2<false, true, true>();
}

TEST_F(BSplinePerformanceTest,
       UniformBSpline_parDim2_nomemopt_noprecomp_nograd) {
  make_test_UniformBSpline_parDim2<false, false, false>();
}

TEST_F(BSplinePerformanceTest, UniformBSpline_parDim2_nomemopt_noprecomp_grad) {
  make_test_UniformBSpline_parDim2<false, false, true>();
}

TEST_F(BSplinePerformanceTest, UniformBSpline_parDim3_memopt_precomp_nograd) {
  make_test_UniformBSpline_parDim3<true, true, false>();
}

TEST_F(BSplinePerformanceTest, UniformBSpline_parDim3_memopt_precomp_grad) {
  make_test_UniformBSpline_parDim3<true, true, true>();
}

TEST_F(BSplinePerformanceTest, UniformBSpline_parDim3_memopt_noprecomp_nograd) {
  make_test_UniformBSpline_parDim3<true, false, false>();
}

TEST_F(BSplinePerformanceTest, UniformBSpline_parDim3_memopt_noprecomp_grad) {
  make_test_UniformBSpline_parDim3<true, false, true>();
}

TEST_F(BSplinePerformanceTest, UniformBSpline_parDim3_nomemopt_precomp_nograd) {
  make_test_UniformBSpline_parDim3<false, true, false>();
}

TEST_F(BSplinePerformanceTest, UniformBSpline_parDim3_nomemopt_precomp_grad) {
  make_test_UniformBSpline_parDim3<false, true, true>();
}

TEST_F(BSplinePerformanceTest,
       UniformBSpline_parDim3_nomemopt_noprecomp_nograd) {
  make_test_UniformBSpline_parDim3<false, false, false>();
}

TEST_F(BSplinePerformanceTest, UniformBSpline_parDim3_nomemopt_noprecomp_grad) {
  make_test_UniformBSpline_parDim3<false, false, true>();
}

TEST_F(BSplinePerformanceTest,
       NonUniformBSpline_parDim1_memopt_precomp_nograd) {
  make_test_NonUniformBSpline_parDim1<true, true, false>();
}

TEST_F(BSplinePerformanceTest, NonUniformBSpline_parDim1_memopt_precomp_grad) {
  make_test_NonUniformBSpline_parDim1<true, true, true>();
}

TEST_F(BSplinePerformanceTest,
       NonUniformBSpline_parDim1_memopt_noprecomp_nograd) {
  make_test_NonUniformBSpline_parDim1<true, false, false>();
}

TEST_F(BSplinePerformanceTest,
       NonUniformBSpline_parDim1_memopt_noprecomp_grad) {
  make_test_NonUniformBSpline_parDim1<true, false, true>();
}

TEST_F(BSplinePerformanceTest,
       NonUniformBSpline_parDim1_nomemopt_precomp_nograd) {
  make_test_NonUniformBSpline_parDim1<false, true, false>();
}

TEST_F(BSplinePerformanceTest,
       NonUniformBSpline_parDim1_nomemopt_precomp_grad) {
  make_test_NonUniformBSpline_parDim1<false, true, true>();
}

TEST_F(BSplinePerformanceTest,
       NonUniformBSpline_parDim1_nomemopt_noprecomp_nograd) {
  make_test_NonUniformBSpline_parDim1<false, false, false>();
}

TEST_F(BSplinePerformanceTest,
       NonUniformBSpline_parDim1_nomemopt_noprecomp_grad) {
  make_test_NonUniformBSpline_parDim1<false, false, true>();
}

TEST_F(BSplinePerformanceTest,
       NonUniformBSpline_parDim2_memopt_precomp_nograd) {
  make_test_NonUniformBSpline_parDim2<true, true, false>();
}

TEST_F(BSplinePerformanceTest, NonUniformBSpline_parDim2_memopt_precomp_grad) {
  make_test_NonUniformBSpline_parDim2<true, true, true>();
}

TEST_F(BSplinePerformanceTest,
       NonUniformBSpline_parDim2_memopt_noprecomp_nograd) {
  make_test_NonUniformBSpline_parDim2<true, false, false>();
}

TEST_F(BSplinePerformanceTest,
       NonUniformBSpline_parDim2_memopt_noprecomp_grad) {
  make_test_NonUniformBSpline_parDim2<true, false, true>();
}

TEST_F(BSplinePerformanceTest,
       NonUniformBSpline_parDim2_nomemopt_precomp_nograd) {
  make_test_NonUniformBSpline_parDim2<false, true, false>();
}

TEST_F(BSplinePerformanceTest,
       NonUniformBSpline_parDim2_nomemopt_precomp_grad) {
  make_test_NonUniformBSpline_parDim2<false, true, true>();
}

TEST_F(BSplinePerformanceTest,
       NonUniformBSpline_parDim2_nomemopt_noprecomp_nograd) {
  make_test_NonUniformBSpline_parDim2<false, false, false>();
}

TEST_F(BSplinePerformanceTest,
       NonUniformBSpline_parDim2_nomemopt_noprecomp_grad) {
  make_test_NonUniformBSpline_parDim2<false, false, true>();
}

TEST_F(BSplinePerformanceTest,
       NonUniformBSpline_parDim3_memopt_precomp_nograd) {
  make_test_NonUniformBSpline_parDim3<true, true, false>();
}

TEST_F(BSplinePerformanceTest, NonUniformBSpline_parDim3_memopt_precomp_grad) {
  make_test_NonUniformBSpline_parDim3<true, true, true>();
}

TEST_F(BSplinePerformanceTest,
       NonUniformBSpline_parDim3_memopt_noprecomp_nograd) {
  make_test_NonUniformBSpline_parDim3<true, false, false>();
}

TEST_F(BSplinePerformanceTest,
       NonUniformBSpline_parDim3_memopt_noprecomp_grad) {
  make_test_NonUniformBSpline_parDim3<true, false, true>();
}

TEST_F(BSplinePerformanceTest,
       NonUniformBSpline_parDim3_nomemopt_precomp_nograd) {
  make_test_NonUniformBSpline_parDim3<false, true, false>();
}

TEST_F(BSplinePerformanceTest,
       NonUniformBSpline_parDim3_nomemopt_precomp_grad) {
  make_test_NonUniformBSpline_parDim3<false, true, true>();
}

TEST_F(BSplinePerformanceTest,
       NonUniformBSpline_parDim3_nomemopt_noprecomp_nograd) {
  make_test_NonUniformBSpline_parDim3<false, false, false>();
}

TEST_F(BSplinePerformanceTest,
       NonUniformBSpline_parDim3_nomemopt_noprecomp_grad) {
  make_test_NonUniformBSpline_parDim3<false, false, true>();
}

int main(int argc, char **argv) {
  ::testing::GTEST_FLAG(filter) = ":-:*";
  ::testing::InitGoogleTest(&argc, argv);
  iganet::init();
  return RUN_ALL_TESTS();
}
