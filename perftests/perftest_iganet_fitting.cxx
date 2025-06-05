/**
   @file perftests/perftest_iganet_fitting.cxx

   @brief IgANet training and inference performance unittests

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

/// @brief Specialization of the abstract IgANet class for function fitting
template <typename Optimizer, typename GeometryMap, typename Variable,
          bool memory_optimized, bool precompute>
class Fitting;

/// @brief Specialization of the abstract IgANet class for function fitting
template <typename Optimizer, typename GeometryMap, typename Variable,
          bool memory_optimized>
class Fitting<Optimizer, GeometryMap, Variable, memory_optimized, false>
    : public iganet::IgANet<Optimizer, GeometryMap, Variable,
                            iganet::IgABaseNoRefData> {

private:
  /// @brief Type of the base class
  using Base = iganet::IgANet<Optimizer, GeometryMap, Variable,
                              iganet::IgABaseNoRefData>;

  /// @brief Collocation points
  typename Base::variable_collPts_type collPts_;

public:
  /// @brief Constructors from the base class
  using iganet::IgANet<Optimizer, GeometryMap, Variable,
                       iganet::IgABaseNoRefData>::IgANet;

  /// @brief Returns a constant reference to the collocation points
  auto const &collPts() const { return collPts_; }

  /// @brief Initializes the epoch
  ///
  /// @param[in] epoch Epoch number
  bool epoch(int64_t epoch) override {
    // In the very first epoch we need to generate the sampling points
    // for the inputs and the sampling points in the function space of
    // the variables since otherwise the respective tensors would be
    // empty. In all further epochs no updates are needed since we do
    // not change the inputs nor the variable function space.
    if (epoch == 0) {
      Base::inputs(epoch);
      collPts_ = Base::variable_collPts(iganet::collPts::greville);

      return true;
    } else
      return false;
  }

  /// @brief Computes the loss function
  ///
  /// @param[in] outputs Output of the network
  ///
  /// @param[in] epoch Epoch number
  torch::Tensor loss(const torch::Tensor &outputs, int64_t epoch) override {

    // Cast the network output (a raw tensor) into the proper
    // function-space format, i.e. B-spline objects for the interior
    // and boundary parts that can be evaluated.
    Base::u_.from_tensor(outputs);

    // Evaluate the loss function
    return torch::mse_loss(*Base::u_.eval(collPts_.first)[0],
                           sin(M_PI * collPts_.first[0]) *
                               sin(M_PI * collPts_.first[1]));
  }
};

/// @brief Specialization of the abstract IgANet class for function fitting
template <typename Optimizer, typename GeometryMap, typename Variable,
          bool memory_optimized>
class Fitting<Optimizer, GeometryMap, Variable, memory_optimized, true>
    : public iganet::IgANet<Optimizer, GeometryMap, Variable,
                            iganet::IgABaseNoRefData>,
      public iganet::IgANetCustomizable<GeometryMap, Variable> {

private:
  /// @brief Type of the base class
  using Base = iganet::IgANet<Optimizer, GeometryMap, Variable,
                              iganet::IgABaseNoRefData>;

  /// @brief Collocation points
  typename Base::variable_collPts_type collPts_;

  /// @brief Type of the customizable class
  using Customizable = iganet::IgANetCustomizable<GeometryMap, Variable>;

  /// @brief Knot indices
  typename Customizable::variable_interior_knot_indices_type knot_indices_;

  /// @brief Coefficient indices
  typename Customizable::variable_interior_coeff_indices_type coeff_indices_;

public:
  /// @brief Constructors from the base class
  using iganet::IgANet<Optimizer, GeometryMap, Variable,
                       iganet::IgABaseNoRefData>::IgANet;

  /// @brief Returns a constant reference to the collocation points
  auto const &collPts() const { return collPts_; }

  /// @brief Initializes the epoch
  ///
  /// @param[in] epoch Epoch number
  bool epoch(int64_t epoch) override {
    // In the very first epoch we need to generate the sampling points
    // for the inputs and the sampling points in the function space of
    // the variables since otherwise the respective tensors would be
    // empty. In all further epochs no updates are needed since we do
    // not change the inputs nor the variable function space.
    if (epoch == 0) {
      Base::inputs(epoch);
      collPts_ = Base::variable_collPts(iganet::collPts::greville);

      knot_indices_ =
          Base::u_.template find_knot_indices<iganet::functionspace::interior>(
              collPts_.first);
      coeff_indices_ =
          Base::u_.template find_coeff_indices<iganet::functionspace::interior>(
              knot_indices_);

      return true;
    } else
      return false;
  }

  /// @brief Computes the loss function
  ///
  /// @param[in] outputs Output of the network
  ///
  /// @param[in] epoch Epoch number
  torch::Tensor loss(const torch::Tensor &outputs, int64_t epoch) override {

    // Cast the network output (a raw tensor) into the proper
    // function-space format, i.e. B-spline objects for the interior
    // and boundary parts that can be evaluated.
    Base::u_.from_tensor(outputs);

    // Evaluate the loss function
    return torch::mse_loss(
        *Base::u_.eval(collPts_.first, knot_indices_, coeff_indices_)[0],
        sin(M_PI * collPts_.first[0]) * sin(M_PI * collPts_.first[1]));
  }
};

/// @brief Fixture for IgANet fitting performance test
class FittingPerformanceTest : public ::testing::Test {
public:
  /// @brief Evaluation functor
  ///
  /// @note GoogleTest does not support fixtures with multiple
  /// non-type template parameters. This functor is a work-around
  template <typename GeometryMap, typename Variable, bool memory_optimized,
            bool precompute>
  struct train {

    /// @brief Call operator
    void operator()(int64_t ncoeffs, int64_t nlayers, int64_t nneurons) {

      using namespace iganet::literals;
      using optimizer_t = torch::optim::Adam;

      std::vector<int64_t> layers(nlayers, nneurons);
      std::vector<std::vector<std::any>> activations(
          nlayers, std::vector<std::any>{iganet::activation::relu});
      activations.emplace_back(std::vector<std::any>{iganet::activation::none});

      // 1D
      if constexpr (GeometryMap::spline_type::parDim() == 1) {
        Fitting<optimizer_t, GeometryMap, Variable, memory_optimized,
                precompute>
            net(layers, activations,
                std::tuple(iganet::utils::to_array(
                    GeometryMap::spline_type::degree(0) + 1_i64)),
                std::tuple(iganet::utils::to_array(ncoeffs)));

        net.options().max_epoch(
            iganet::utils::getenv("IGANET_MAX_EPOCH", 1000_i64));

        net.options().min_loss(iganet::utils::getenv("IGANET_MIN_LOSS", 1e-12));

        auto t1 = std::chrono::high_resolution_clock::now();
        net.train();
        auto t2 = std::chrono::high_resolution_clock::now();

        std::cout << std::right << std::setw(10)
                  << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 -
                                                                          t1)
                             .count() /
                         net.nparameters()
                  << "\t";
      }

      // 2D
      else if constexpr (GeometryMap::spline_type::parDim() == 2) {
        Fitting<optimizer_t, GeometryMap, Variable, memory_optimized,
                precompute>
            net(layers, activations,
                std::tuple(iganet::utils::to_array(
                    GeometryMap::spline_type::degree(0) + 1_i64,
                    GeometryMap::spline_type::degree(1) + 1_i64)),
                std::tuple(iganet::utils::to_array(ncoeffs, ncoeffs)));

        net.options().max_epoch(
            iganet::utils::getenv("IGANET_MAX_EPOCH", 1000_i64));

        net.options().min_loss(iganet::utils::getenv("IGANET_MIN_LOSS", 1e-12));

        auto t1 = std::chrono::high_resolution_clock::now();
        net.train();
        auto t2 = std::chrono::high_resolution_clock::now();

        std::cout << std::right << std::setw(10)
                  << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 -
                                                                          t1)
                             .count() /
                         net.nparameters()
                  << "\t";
      }

      // 3D
      else if constexpr (GeometryMap::spline_type::parDim() == 3) {
        Fitting<optimizer_t, GeometryMap, Variable, memory_optimized,
                precompute>
            net(layers, activations,
                std::tuple(iganet::utils::to_array(
                    GeometryMap::spline_type::degree(0) + 1_i64,
                    GeometryMap::spline_type::degree(1) + 1_i64,
                    GeometryMap::spline_type::degree(2) + 1_i64)),
                std::tuple(iganet::utils::to_array(ncoeffs, ncoeffs, ncoeffs)));

        net.options().max_epoch(
            iganet::utils::getenv("IGANET_MAX_EPOCH", 1000_i64));

        net.options().min_loss(iganet::utils::getenv("IGANET_MIN_LOSS", 1e-12));

        auto t1 = std::chrono::high_resolution_clock::now();
        net.train();
        auto t2 = std::chrono::high_resolution_clock::now();

        std::cout << std::right << std::setw(10)
                  << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 -
                                                                          t1)
                             .count() /
                         net.nparameters()
                  << "\t";
      }

      // 4D
      else if constexpr (GeometryMap::spline_type::parDim() == 4) {
        Fitting<optimizer_t, GeometryMap, Variable, memory_optimized,
                precompute>
            net(layers, activations,
                std::tuple(iganet::utils::to_array(
                    GeometryMap::spline_type::degree(0) + 1_i64,
                    GeometryMap::spline_type::degree(1) + 1_i64,
                    GeometryMap::spline_type::degree(2) + 1_i64,
                    GeometryMap::spline_type::degree(3) + 1_i64)),
                std::tuple(iganet::utils::to_array(ncoeffs, ncoeffs, ncoeffs,
                                                   ncoeffs)));

        net.options().max_epoch(
            iganet::utils::getenv("IGANET_MAX_EPOCH", 1000_i64));

        net.options().min_loss(iganet::utils::getenv("IGANET_MIN_LOSS", 1e-12));

        auto t1 = std::chrono::high_resolution_clock::now();
        net.train();
        auto t2 = std::chrono::high_resolution_clock::now();

        std::cout << std::right << std::setw(10)
                  << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 -
                                                                          t1)
                             .count() /
                         net.nparameters()
                  << "\t";
      }

      else
        throw std::runtime_error("Unsupported parametric dimension");
    }
  };
};

template <bool memory_optimized, bool precompute>
void make_test_UniformBSpline_parDim1() {

  std::cout << "memory_optimized : " << memory_optimized
            << ", precompute : " << precompute << std::endl;

  for (int64_t ncoeffs : iganet::utils::getenv("IGANET_NCOEFFS", {32})) {
    for (std::vector<std::any> activation :
         {std::vector<std::any>{iganet::activation::relu}}) {
      for (int64_t nlayers : iganet::utils::getenv("IGANET_NLAYERS", {1})) {
        for (int64_t nneurons :
             iganet::utils::getenv("IGANET_NNEURONS", {10})) {

          std::cout << std::right << std::setw(8) << ncoeffs << "\t"
                    << std::right << std::setw(8) << nlayers << "\t"
                    << std::right << std::setw(8) << nneurons << "\t";

          std::apply(
              [&](auto &&...args) {
                ((args(ncoeffs, nlayers, nneurons)), ...);
              },
              std::make_tuple(FittingPerformanceTest::train<
                                  iganet::S<iganet::UniformBSpline<
                                      iganet::perftests::real_t, 1, 1>>,
                                  iganet::S<iganet::UniformBSpline<
                                      iganet::perftests::real_t, 1, 1>>,
                                  memory_optimized, precompute>{},
                              FittingPerformanceTest::train<
                                  iganet::S<iganet::UniformBSpline<
                                      iganet::perftests::real_t, 1, 2>>,
                                  iganet::S<iganet::UniformBSpline<
                                      iganet::perftests::real_t, 1, 2>>,
                                  memory_optimized, precompute>{},
                              FittingPerformanceTest::train<
                                  iganet::S<iganet::UniformBSpline<
                                      iganet::perftests::real_t, 1, 3>>,
                                  iganet::S<iganet::UniformBSpline<
                                      iganet::perftests::real_t, 1, 3>>,
                                  memory_optimized, precompute>{},
                              FittingPerformanceTest::train<
                                  iganet::S<iganet::UniformBSpline<
                                      iganet::perftests::real_t, 1, 4>>,
                                  iganet::S<iganet::UniformBSpline<
                                      iganet::perftests::real_t, 1, 4>>,
                                  memory_optimized, precompute>{},
                              FittingPerformanceTest::train<
                                  iganet::S<iganet::UniformBSpline<
                                      iganet::perftests::real_t, 1, 5>>,
                                  iganet::S<iganet::UniformBSpline<
                                      iganet::perftests::real_t, 1, 5>>,
                                  memory_optimized, precompute>{}));

          std::cout << std::endl;
        }
      }
    }
  }

  std::cout << std::scientific << std::setprecision(3);
}

template <bool memory_optimized, bool precompute>
void make_test_UniformBSpline_parDim2() {

  std::cout << "memory_optimized : " << memory_optimized
            << ", precompute : " << precompute << std::endl;

  for (int64_t ncoeffs : iganet::utils::getenv("IGANET_NCOEFFS", {32})) {
    for (std::vector<std::any> activation :
         {std::vector<std::any>{iganet::activation::relu}}) {
      for (int64_t nlayers : iganet::utils::getenv("IGANET_NLAYERS", {1})) {
        for (int64_t nneurons :
             iganet::utils::getenv("IGANET_NNEURONS", {10})) {

          std::cout << std::right << std::setw(8) << ncoeffs << "\t"
                    << std::right << std::setw(8) << nlayers << "\t"
                    << std::right << std::setw(8) << nneurons << "\t";

          std::apply(
              [&](auto &&...args) {
                ((args(ncoeffs, nlayers, nneurons)), ...);
              },
              std::make_tuple(FittingPerformanceTest::train<
                                  iganet::S<iganet::UniformBSpline<
                                      iganet::perftests::real_t, 2, 1, 1>>,
                                  iganet::S<iganet::UniformBSpline<
                                      iganet::perftests::real_t, 1, 1, 1>>,
                                  memory_optimized, precompute>{},
                              FittingPerformanceTest::train<
                                  iganet::S<iganet::UniformBSpline<
                                      iganet::perftests::real_t, 2, 2, 2>>,
                                  iganet::S<iganet::UniformBSpline<
                                      iganet::perftests::real_t, 1, 2, 2>>,
                                  memory_optimized, precompute>{},
                              FittingPerformanceTest::train<
                                  iganet::S<iganet::UniformBSpline<
                                      iganet::perftests::real_t, 2, 3, 3>>,
                                  iganet::S<iganet::UniformBSpline<
                                      iganet::perftests::real_t, 1, 3, 3>>,
                                  memory_optimized, precompute>{},
                              FittingPerformanceTest::train<
                                  iganet::S<iganet::UniformBSpline<
                                      iganet::perftests::real_t, 2, 4, 4>>,
                                  iganet::S<iganet::UniformBSpline<
                                      iganet::perftests::real_t, 1, 4, 4>>,
                                  memory_optimized, precompute>{},
                              FittingPerformanceTest::train<
                                  iganet::S<iganet::UniformBSpline<
                                      iganet::perftests::real_t, 2, 5, 5>>,
                                  iganet::S<iganet::UniformBSpline<
                                      iganet::perftests::real_t, 1, 5, 5>>,
                                  memory_optimized, precompute>{}));

          std::cout << std::endl;
        }
      }
    }
  }

  std::cout << std::scientific << std::setprecision(3);
}

template <bool memory_optimized, bool precompute>
void make_test_UniformBSpline_parDim3() {

  std::cout << "memory_optimized : " << memory_optimized
            << ", precompute : " << precompute << std::endl;

  for (int64_t ncoeffs : iganet::utils::getenv("IGANET_NCOEFFS", {32})) {
    for (std::vector<std::any> activation :
         {std::vector<std::any>{iganet::activation::relu}}) {
      for (int64_t nlayers : iganet::utils::getenv("IGANET_NLAYERS", {1})) {
        for (int64_t nneurons :
             iganet::utils::getenv("IGANET_NNEURONS", {10})) {

          std::cout << std::right << std::setw(8) << ncoeffs << "\t"
                    << std::right << std::setw(8) << nlayers << "\t"
                    << std::right << std::setw(8) << nneurons << "\t";

          std::apply(
              [&](auto &&...args) {
                ((args(ncoeffs, nlayers, nneurons)), ...);
              },
              std::make_tuple(FittingPerformanceTest::train<
                                  iganet::S<iganet::UniformBSpline<
                                      iganet::perftests::real_t, 3, 1, 1, 1>>,
                                  iganet::S<iganet::UniformBSpline<
                                      iganet::perftests::real_t, 1, 1, 1, 1>>,
                                  memory_optimized, precompute>{},
                              FittingPerformanceTest::train<
                                  iganet::S<iganet::UniformBSpline<
                                      iganet::perftests::real_t, 3, 2, 2, 2>>,
                                  iganet::S<iganet::UniformBSpline<
                                      iganet::perftests::real_t, 1, 2, 2, 2>>,
                                  memory_optimized, precompute>{},
                              FittingPerformanceTest::train<
                                  iganet::S<iganet::UniformBSpline<
                                      iganet::perftests::real_t, 3, 3, 3, 3>>,
                                  iganet::S<iganet::UniformBSpline<
                                      iganet::perftests::real_t, 1, 3, 3, 3>>,
                                  memory_optimized, precompute>{},
                              FittingPerformanceTest::train<
                                  iganet::S<iganet::UniformBSpline<
                                      iganet::perftests::real_t, 3, 4, 4, 4>>,
                                  iganet::S<iganet::UniformBSpline<
                                      iganet::perftests::real_t, 1, 4, 4, 4>>,
                                  memory_optimized, precompute>{},
                              FittingPerformanceTest::train<
                                  iganet::S<iganet::UniformBSpline<
                                      iganet::perftests::real_t, 3, 5, 5, 5>>,
                                  iganet::S<iganet::UniformBSpline<
                                      iganet::perftests::real_t, 1, 5, 5, 5>>,
                                  memory_optimized, precompute>{}));

          std::cout << std::endl;
        }
      }
    }
  }

  std::cout << std::scientific << std::setprecision(3);
}

template <bool memory_optimized, bool precompute>
void make_test_NonUniformBSpline_parDim1() {

  std::cout << "memory_optimized : " << memory_optimized
            << ", precompute : " << precompute << std::endl;

  for (int64_t ncoeffs : iganet::utils::getenv("IGANET_NCOEFFS", {32})) {
    for (std::vector<std::any> activation :
         {std::vector<std::any>{iganet::activation::relu}}) {
      for (int64_t nlayers : iganet::utils::getenv("IGANET_NLAYERS", {1})) {
        for (int64_t nneurons :
             iganet::utils::getenv("IGANET_NNEURONS", {10})) {

          std::cout << std::right << std::setw(8) << ncoeffs << "\t"
                    << std::right << std::setw(8) << nlayers << "\t"
                    << std::right << std::setw(8) << nneurons << "\t";

          std::apply(
              [&](auto &&...args) {
                ((args(ncoeffs, nlayers, nneurons)), ...);
              },
              std::make_tuple(FittingPerformanceTest::train<
                                  iganet::S<iganet::NonUniformBSpline<
                                      iganet::perftests::real_t, 1, 1>>,
                                  iganet::S<iganet::NonUniformBSpline<
                                      iganet::perftests::real_t, 1, 1>>,
                                  memory_optimized, precompute>{},
                              FittingPerformanceTest::train<
                                  iganet::S<iganet::NonUniformBSpline<
                                      iganet::perftests::real_t, 1, 2>>,
                                  iganet::S<iganet::NonUniformBSpline<
                                      iganet::perftests::real_t, 1, 2>>,
                                  memory_optimized, precompute>{},
                              FittingPerformanceTest::train<
                                  iganet::S<iganet::NonUniformBSpline<
                                      iganet::perftests::real_t, 1, 3>>,
                                  iganet::S<iganet::NonUniformBSpline<
                                      iganet::perftests::real_t, 1, 3>>,
                                  memory_optimized, precompute>{},
                              FittingPerformanceTest::train<
                                  iganet::S<iganet::NonUniformBSpline<
                                      iganet::perftests::real_t, 1, 4>>,
                                  iganet::S<iganet::NonUniformBSpline<
                                      iganet::perftests::real_t, 1, 4>>,
                                  memory_optimized, precompute>{},
                              FittingPerformanceTest::train<
                                  iganet::S<iganet::NonUniformBSpline<
                                      iganet::perftests::real_t, 1, 5>>,
                                  iganet::S<iganet::NonUniformBSpline<
                                      iganet::perftests::real_t, 1, 5>>,
                                  memory_optimized, precompute>{}));

          std::cout << std::endl;
        }
      }
    }
  }

  std::cout << std::scientific << std::setprecision(3);
}

template <bool memory_optimized, bool precompute>
void make_test_NonUniformBSpline_parDim2() {

  std::cout << "memory_optimized : " << memory_optimized
            << ", precompute : " << precompute << std::endl;

  for (int64_t ncoeffs : iganet::utils::getenv("IGANET_NCOEFFS", {32})) {
    for (std::vector<std::any> activation :
         {std::vector<std::any>{iganet::activation::relu}}) {
      for (int64_t nlayers : iganet::utils::getenv("IGANET_NLAYERS", {1})) {
        for (int64_t nneurons :
             iganet::utils::getenv("IGANET_NNEURONS", {10})) {

          std::cout << std::right << std::setw(8) << ncoeffs << "\t"
                    << std::right << std::setw(8) << nlayers << "\t"
                    << std::right << std::setw(8) << nneurons << "\t";

          std::apply(
              [&](auto &&...args) {
                ((args(ncoeffs, nlayers, nneurons)), ...);
              },
              std::make_tuple(FittingPerformanceTest::train<
                                  iganet::S<iganet::NonUniformBSpline<
                                      iganet::perftests::real_t, 2, 1, 1>>,
                                  iganet::S<iganet::NonUniformBSpline<
                                      iganet::perftests::real_t, 1, 1, 1>>,
                                  memory_optimized, precompute>{},
                              FittingPerformanceTest::train<
                                  iganet::S<iganet::NonUniformBSpline<
                                      iganet::perftests::real_t, 2, 2, 2>>,
                                  iganet::S<iganet::NonUniformBSpline<
                                      iganet::perftests::real_t, 1, 2, 2>>,
                                  memory_optimized, precompute>{},
                              FittingPerformanceTest::train<
                                  iganet::S<iganet::NonUniformBSpline<
                                      iganet::perftests::real_t, 2, 3, 3>>,
                                  iganet::S<iganet::NonUniformBSpline<
                                      iganet::perftests::real_t, 1, 3, 3>>,
                                  memory_optimized, precompute>{},
                              FittingPerformanceTest::train<
                                  iganet::S<iganet::NonUniformBSpline<
                                      iganet::perftests::real_t, 2, 4, 4>>,
                                  iganet::S<iganet::NonUniformBSpline<
                                      iganet::perftests::real_t, 1, 4, 4>>,
                                  memory_optimized, precompute>{},
                              FittingPerformanceTest::train<
                                  iganet::S<iganet::NonUniformBSpline<
                                      iganet::perftests::real_t, 2, 5, 5>>,
                                  iganet::S<iganet::NonUniformBSpline<
                                      iganet::perftests::real_t, 1, 5, 5>>,
                                  memory_optimized, precompute>{}));

          std::cout << std::endl;
        }
      }
    }
  }

  std::cout << std::scientific << std::setprecision(3);
}

template <bool memory_optimized, bool precompute>
void make_test_NonUniformBSpline_parDim3() {

  std::cout << "memory_optimized : " << memory_optimized
            << ", precompute : " << precompute << std::endl;

  for (int64_t ncoeffs : iganet::utils::getenv("IGANET_NCOEFFS", {32})) {
    for (std::vector<std::any> activation :
         {std::vector<std::any>{iganet::activation::relu}}) {
      for (int64_t nlayers : iganet::utils::getenv("IGANET_NLAYERS", {1})) {
        for (int64_t nneurons :
             iganet::utils::getenv("IGANET_NNEURONS", {10})) {

          std::cout << std::right << std::setw(8) << ncoeffs << "\t"
                    << std::right << std::setw(8) << nlayers << "\t"
                    << std::right << std::setw(8) << nneurons << "\t";

          std::apply(
              [&](auto &&...args) {
                ((args(ncoeffs, nlayers, nneurons)), ...);
              },
              std::make_tuple(FittingPerformanceTest::train<
                                  iganet::S<iganet::NonUniformBSpline<
                                      iganet::perftests::real_t, 3, 1, 1, 1>>,
                                  iganet::S<iganet::NonUniformBSpline<
                                      iganet::perftests::real_t, 1, 1, 1, 1>>,
                                  memory_optimized, precompute>{},
                              FittingPerformanceTest::train<
                                  iganet::S<iganet::NonUniformBSpline<
                                      iganet::perftests::real_t, 3, 2, 2, 2>>,
                                  iganet::S<iganet::NonUniformBSpline<
                                      iganet::perftests::real_t, 1, 2, 2, 2>>,
                                  memory_optimized, precompute>{},
                              FittingPerformanceTest::train<
                                  iganet::S<iganet::NonUniformBSpline<
                                      iganet::perftests::real_t, 3, 3, 3, 3>>,
                                  iganet::S<iganet::NonUniformBSpline<
                                      iganet::perftests::real_t, 1, 3, 3, 3>>,
                                  memory_optimized, precompute>{},
                              FittingPerformanceTest::train<
                                  iganet::S<iganet::NonUniformBSpline<
                                      iganet::perftests::real_t, 3, 4, 4, 4>>,
                                  iganet::S<iganet::NonUniformBSpline<
                                      iganet::perftests::real_t, 1, 4, 4, 4>>,
                                  memory_optimized, precompute>{},
                              FittingPerformanceTest::train<
                                  iganet::S<iganet::NonUniformBSpline<
                                      iganet::perftests::real_t, 3, 5, 5, 5>>,
                                  iganet::S<iganet::NonUniformBSpline<
                                      iganet::perftests::real_t, 1, 5, 5, 5>>,
                                  memory_optimized, precompute>{}));

          std::cout << std::endl;
        }
      }
    }
  }

  std::cout << std::scientific << std::setprecision(3);
}

TEST_F(FittingPerformanceTest, UniformBSpline_parDim1_memopt_precomp) {
  make_test_UniformBSpline_parDim1<true, true>();
}

TEST_F(FittingPerformanceTest, UniformBSpline_parDim1_memopt_noprecomp) {
  make_test_UniformBSpline_parDim1<true, false>();
}

TEST_F(FittingPerformanceTest, UniformBSpline_parDim1_nomemopt_precomp) {
  make_test_UniformBSpline_parDim1<false, true>();
}

TEST_F(FittingPerformanceTest, UniformBSpline_parDim1_nomemopt_noprecomp) {
  make_test_UniformBSpline_parDim1<false, false>();
}

TEST_F(FittingPerformanceTest, UniformBSpline_parDim2_memopt_precomp) {
  make_test_UniformBSpline_parDim2<true, true>();
}

TEST_F(FittingPerformanceTest, UniformBSpline_parDim2_memopt_noprecomp) {
  make_test_UniformBSpline_parDim2<true, false>();
}

TEST_F(FittingPerformanceTest, UniformBSpline_parDim2_nomemopt_precomp) {
  make_test_UniformBSpline_parDim2<false, true>();
}

TEST_F(FittingPerformanceTest, UniformBSpline_parDim2_nomemopt_noprecomp) {
  make_test_UniformBSpline_parDim2<false, false>();
}

TEST_F(FittingPerformanceTest, UniformBSpline_parDim3_memopt_precomp) {
  make_test_UniformBSpline_parDim2<true, true>();
}

TEST_F(FittingPerformanceTest, UniformBSpline_parDim3_memopt_noprecomp) {
  make_test_UniformBSpline_parDim2<true, false>();
}

TEST_F(FittingPerformanceTest, UniformBSpline_parDim3_nomemopt_precomp) {
  make_test_UniformBSpline_parDim2<false, true>();
}

TEST_F(FittingPerformanceTest, UniformBSpline_parDim3_nomemopt_noprecomp) {
  make_test_UniformBSpline_parDim2<false, false>();
}

TEST_F(FittingPerformanceTest, NonUniformBSpline_parDim1_memopt_precomp) {
  make_test_NonUniformBSpline_parDim1<true, true>();
}

TEST_F(FittingPerformanceTest, NonUniformBSpline_parDim1_memopt_noprecomp) {
  make_test_NonUniformBSpline_parDim1<true, false>();
}

TEST_F(FittingPerformanceTest, NonUniformBSpline_parDim1_nomemopt_precomp) {
  make_test_NonUniformBSpline_parDim1<false, true>();
}

TEST_F(FittingPerformanceTest, NonUniformBSpline_parDim1_nomemopt_noprecomp) {
  make_test_NonUniformBSpline_parDim1<false, false>();
}

TEST_F(FittingPerformanceTest, NonUniformBSpline_parDim2_memopt_precomp) {
  make_test_NonUniformBSpline_parDim2<true, true>();
}

TEST_F(FittingPerformanceTest, NonUniformBSpline_parDim2_memopt_noprecomp) {
  make_test_NonUniformBSpline_parDim2<true, false>();
}

TEST_F(FittingPerformanceTest, NonUniformBSpline_parDim2_nomemopt_precomp) {
  make_test_NonUniformBSpline_parDim2<false, true>();
}

TEST_F(FittingPerformanceTest, NonUniformBSpline_parDim2_nomemopt_noprecomp) {
  make_test_NonUniformBSpline_parDim2<false, false>();
}

TEST_F(FittingPerformanceTest, NonUniformBSpline_parDim3_memopt_precomp) {
  make_test_NonUniformBSpline_parDim2<true, true>();
}

TEST_F(FittingPerformanceTest, NonUniformBSpline_parDim3_memopt_noprecomp) {
  make_test_NonUniformBSpline_parDim2<true, false>();
}

TEST_F(FittingPerformanceTest, NonUniformBSpline_parDim3_nomemopt_precomp) {
  make_test_NonUniformBSpline_parDim2<false, true>();
}

TEST_F(FittingPerformanceTest, NonUniformBSpline_parDim3_nomemopt_noprecomp) {
  make_test_NonUniformBSpline_parDim2<false, false>();
}

int main(int argc, char **argv) {
  ::testing::GTEST_FLAG(filter) = ":-:*";
  ::testing::InitGoogleTest(&argc, argv);
  iganet::init();
  iganet::Log.setLogLevel(iganet::log::none);
  int result = RUN_ALL_TESTS();
  iganet::finalize();
  return result;
}
