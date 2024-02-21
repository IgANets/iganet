/**
   @file examples/iganet_simple_fitting.cxx

   @brief Demonstration of IgANet function fitting

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <iganet.h>
#include <iostream>

/// @brief IgANet for function fitting
template <typename optimizer_t, typename geometry_t, typename variable_t>
class fitting
  : public iganet::IgANet<optimizer_t, geometry_t, variable_t> {

private:
  /// @brief Type of the base class
  using Base = iganet::IgANet<optimizer_t, geometry_t, variable_t>;

public:
  /// @brief Constructors from the base class
  using iganet::IgANet<optimizer_t, geometry_t, variable_t>::IgANet;

  /// @brief Initializes the epoch
  ///
  /// @param[in] epoch Epoch number
  ///
  /// @param[in] status Status flag
  iganet::status epoch(int64_t epoch) override {
    std::cout << "Epoch " << std::to_string(epoch) << ": ";
    
    return iganet::status::inputs;
  }

  /// @brief Computes the loss function
  ///
  /// @param[in] outputs Output of the network
  ///
  /// @param[in] geometry_samples Sampling points for the geometry
  ///
  /// @param[in] variable_samples Sampling points for the variable
  ///
  /// @param[in] epoch Epoch number
  ///
  /// @param[in] status Status flag
  torch::Tensor
  loss(const torch::Tensor &outputs,
       const typename Base::geometry_samples_type &geometry_samples,
       const typename Base::variable_samples_type &variable_samples,
       int64_t epoch, iganet::status status) override {

    std::cout << "LOSS\n";
    
    std::cout << geometry_samples.first << std::endl;

    std::cout << variable_samples.first << std::endl;
    
    exit(0);
    
    // Evaluate loss function
    return torch::mse_loss(*Base::outputs_.eval(variable_samples.first)[0],
                           *Base::variable_.eval(variable_samples.first)[0]);
  }
};

int main() {
  iganet::init();
  iganet::verbose(std::cout);

  using namespace iganet::literals;
  using optimizer_t = torch::optim::Adam;
  using real_t = double;

  // Bi-linear B-spline geometry
  using geometry_t = iganet::S2<iganet::UniformBSpline<real_t, 2, 2, 2>>;

  // Bi-quadratic B-spline variable
  using variable_t = iganet::S2<iganet::UniformBSpline<real_t, 1, 2, 2>>;

  fitting<optimizer_t, geometry_t, variable_t>
    net( // Number of neurons per layers
         {50, 50, 50, 50, 50}
         ,
         // Activation functions
         {{iganet::activation::sigmoid},
          {iganet::activation::sigmoid},
          {iganet::activation::sigmoid},
          {iganet::activation::sigmoid},
          {iganet::activation::sigmoid},
          {iganet::activation::none}}
         ,
         // Number of B-spline coefficients of the geometry
         std::tuple(iganet::utils::to_array(7_i64, 7_i64))
         //         ,
         // Number of B-spline coefficients of the variable
         //         std::tuple(iganet::utils::to_array(7_i64, 7_i64))
         );
  
  // Impose solution value for supervised training (not right-hand side)
  net.variable().transform([](const std::array<real_t, 2> xi) {
    return std::array<real_t, 1>{static_cast<real_t>(sin(M_PI * xi[0]) * sin(M_PI * xi[1]))};
  });

  net.options().max_epoch(1000);
  net.options().min_loss(1e-8);
  net.train();

#ifdef IGANET_WITH_MATPLOT
  net.geometry().plot(net.outputs(), 50, 50);
#endif
  
  return 0;
}
