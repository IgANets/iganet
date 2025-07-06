/**
   @file examples/iganet_stokes.cxx

   @brief Demonstration of IgANet Stokes solver

   This example demonstrates how to implement a simple IgANet for
   learning the Stokes equation with (non-)homogeneous Dirichlet
   boundary conditions on a square geometry.

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <iganet.h>
#include <iostream>

using namespace iganet::literals;

/// @brief Specialization of the abstract IgANet class for Stokes's equation
template <typename Optimizer, typename GeometryMap, typename Variable>
class stokes : public iganet::IgANet<Optimizer, GeometryMap, Variable,
                                     iganet::IgABaseNoRefData>,
               public iganet::IgANetCustomizable<GeometryMap, Variable> {

private:
  /// @brief Type of the base class
  using Base = iganet::IgANet<Optimizer, GeometryMap, Variable,
                              iganet::IgABaseNoRefData>;

  /// @brief Collocation points
  typename Base::variable_collPts_type collPts_;

  /// @brief Reference solution
  // Variable ref_;

  /// @brief Type of the customizable class
  using Customizable = iganet::IgANetCustomizable<GeometryMap, Variable>;

  /// @brief Knot indices of variables
  typename Customizable::variable_interior_knot_indices_type var_knot_indices_;

  /// @brief Coefficient indices of variables
  typename Customizable::variable_interior_coeff_indices_type
      var_coeff_indices_;


public:
  /// @brief Constructor
  template <typename... Args>
  stokes(std::vector<int64_t> &&layers,
         std::vector<std::vector<std::any>> &&activations, Args &&...args)
      : Base(std::forward<std::vector<int64_t>>(layers),
             std::forward<std::vector<std::vector<std::any>>>(activations),
             std::forward<Args>(args)...) {}

  /// @brief Returns a constant reference to the collocation points
  auto const &collPts() const { return collPts_; }

  /// @brief Returns a constant reference to the reference solution
  //auto const &ref() const { return ref_; }

  /// @brief Returns a non-constant reference to the reference solution
  //auto &ref() { return ref_; }

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

      var_knot_indices_ =
          Base::u_.template find_knot_indices<iganet::functionspace::interior>(
              collPts_.first);
      var_coeff_indices_ =
          Base::u_.template find_coeff_indices<iganet::functionspace::interior>(
              var_knot_indices_);

      return true;
    } else
      return false;
  }
  // define material parameters
  float_t density{1e3};
  float_t viscosity{1e-3};

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

    auto vel = Base::u_.template clone<0, 1>();
    auto p = Base::u_.template clone<0, 2>();
        
    auto vel_grad_x = vel.grad( std::get<0>(collPts_.first) ); //  g = grad( vel )[ cpt_u ] -> g_x component
    std::cout << *vel_grad_x[0] << std::endl;
    
    auto vel_grad_y = vel.grad( std::get<1>(collPts_.first) ); //  g = grad( vel )[ cpt_v ] -> g_y component
    //vel_grad_y[0];
    
    //    std::cout << vel << std::endl;
    exit(0);
    
    // Evaluate
    //    auto u_ilapl = Base::u_.template clone<0, 1>().div(std::get<0>(
    //        collPts_.first)); //, var_knot_indices_, var_coeff_indices_);
    //    auto u_ilapl =
    //        Base::u_.ilapl(Base::G_, collPts_.first, var_knot_indices_,
    //                       var_coeff_indices_, G_knot_indices_,
    //                       G_coeff_indices_);

    exit(0);

    auto u_bdr = Base::u_.template eval<iganet::functionspace::boundary>(
        collPts_.second);

    //auto bdr =
        //ref_.template eval<iganet::functionspace::boundary>(collPts_.second);

    // Define the MSE loss function with zero target
    auto mse_loss = [](const torch::Tensor &input) {
      return torch::mean(torch::square(input));
    };

    // Evaluate the loss function
    return outputs; // mse_loss(*std::get<0>(u_ilapl)[0]);
  }
};

int main() {
  iganet::init();
  //iganet::verbose(std::cout);

  nlohmann::json json;
  json["res0"] = 50;
  json["res1"] = 50;
  json["cnet"] = true;

  using namespace iganet::literals;
  using optimizer_t = torch::optim::LBFGS;
  using real_t = double;

  using geometry_t = iganet::S<iganet::UniformBSpline<real_t, 2, 1, 1>>;
  using variable_t = iganet::TH<iganet::NonUniformBSpline<real_t, 1, 1, 1>,2>;

  stokes<optimizer_t, geometry_t, variable_t>
      net( // Number of neurons per layers
          {120, 120},
          // Activation functions
          {{iganet::activation::sigmoid},
           {iganet::activation::sigmoid},
           {iganet::activation::none}},
          // Number of B-spline coefficients of the geometry, just [0,1] x [0,1]
          iganet::utils::to_array(2_i64, 2_i64),
          // Number of B-spline coefficients of the variable
          iganet::utils::to_array(22_i64, 22_i64));

    iganet::Log(iganet::log::info)
              << ", #parameters: " << net.nparameters() << std::endl;
  // Set maximum number of epochs
  net.options().max_epoch(200);

  // Set tolerance for the loss functions
  net.options().min_loss(1e-8);

  // Start time measurement
  auto t1 = std::chrono::high_resolution_clock::now();

  // Train network
  //net.train();

  // Stop time measurement
  auto t2 = std::chrono::high_resolution_clock::now();
  iganet::Log(iganet::log::info)
      << "Training took "
      << std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1)
             .count()
      << " seconds\n";

#ifdef IGANET_WITH_MATPLOT
  // Plot the solution
  // net.G().plot(net.u(), net.collPts().first, json)->show();

  // Plot the difference between the exact and predicted solutions
  // net.G().plot(net.ref().abs_diff(net.u()), net.collPts().first,
  // json)->show();
#endif

  iganet::finalize();
  return 0;
}
