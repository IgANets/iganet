/**
   @file examples/iganet_fitting.cxx

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
  : public iganet::IgANet<optimizer_t, geometry_t, variable_t>,
    public iganet::IgANetCustomizable<optimizer_t, geometry_t, variable_t> {

private:
  /// @brief Type of the base class
  using Base = iganet::IgANet<optimizer_t, geometry_t, variable_t>;

  /// @brief Type of the customizable class
  using Customizable =
    iganet::IgANetCustomizable<optimizer_t, geometry_t, variable_t>;

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

    return (epoch == 0
            ? iganet::status::inputs + iganet::status::geometry_samples +
            iganet::status::variable_samples
            : iganet::status::inputs);
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
    
    // Update indices and precompute basis functions for geometry
    if (status & iganet::status::geometry_samples) {
      Customizable::geometry_interior_knot_indices_ =
        Base::geometry_
        .template find_knot_indices<iganet::functionspace::interior>(
                                                                     geometry_samples.first);
      Customizable::geometry_interior_coeff_indices_ =
        Base::geometry_
        .template find_coeff_indices<iganet::functionspace::interior>(
                                                                      Customizable::geometry_interior_knot_indices_);

      Customizable::geometry_boundary_knot_indices_ =
        Base::geometry_
        .template find_knot_indices<iganet::functionspace::boundary>(
                                                                     geometry_samples.second);
      Customizable::geometry_boundary_coeff_indices_ =
        Base::geometry_
        .template find_coeff_indices<iganet::functionspace::boundary>(
                                                                      Customizable::geometry_boundary_knot_indices_);
    }

    // Update indices and precompute basis functions for variable
    if (status & iganet::status::variable_samples) {
      Customizable::variable_interior_knot_indices_ =
        Base::variable_
        .template find_knot_indices<iganet::functionspace::interior>(
                                                                     variable_samples.first);
      Customizable::variable_interior_coeff_indices_ =
        Base::variable_
        .template find_coeff_indices<iganet::functionspace::interior>(
                                                                      Customizable::variable_interior_knot_indices_);

      Customizable::variable_boundary_knot_indices_ =
        Base::variable_
        .template find_knot_indices<iganet::functionspace::boundary>(
                                                                     variable_samples.second);
      Customizable::variable_boundary_coeff_indices_ =
        Base::variable_
        .template find_coeff_indices<iganet::functionspace::boundary>(
                                                                      Customizable::variable_boundary_knot_indices_);
    }

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
  using real_t = float;

  using geometry_t = iganet::S2<iganet::UniformBSpline<real_t, 2, 2, 2>>;
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
         ,
         // Number of B-spline coefficients of the variable
         std::tuple(iganet::utils::to_array(7_i64, 7_i64))
         );
  
  // Deform geometry
  net.geometry().transform([](const std::array<real_t, 2> xi) {
    return std::array<real_t, 2>{(xi[0] + 1) * cos(static_cast<real_t>(M_PI) * xi[1]),
                                 (xi[0] + 1) * sin(static_cast<real_t>(M_PI) * xi[1])};
  });

  // Impose solution value for supervised training (not right-hand side)
  net.variable().transform([](const std::array<real_t, 2> xi) {
    return std::array<real_t, 1>{static_cast<real_t>(sin(M_PI * xi[0]) * sin(M_PI * xi[1]))};
  });

  // Boundary values
  net.variable().boundary().template side<1>().transform(
                                                         [](const std::array<real_t, 1> xi) {
                                                           return std::array<real_t, 1>{0.0};
                                                         });

  net.variable().boundary().template side<2>().transform(
                                                         [](const std::array<real_t, 1> xi) {
                                                           return std::array<real_t, 1>{0.0};
                                                         });

  net.variable().boundary().template side<3>().transform(
                                                         [](const std::array<real_t, 1> xi) {
                                                           return std::array<real_t, 1>{0.0};
                                                         });

  net.variable().boundary().template side<4>().transform(
                                                         [](const std::array<real_t, 1> xi) {
                                                           return std::array<real_t, 1>{0.0};
                                                         });

  net.options().max_epoch(1000);
  net.options().min_loss(1e-8);

  net.train();

  net.geometry().plot(net.outputs(), 50, 50);
  
  return 0;
}
