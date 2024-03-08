/**
   @file examples/iganet_poisson.cxx

   @brief Demonstration of IgANet Poisson solver

   This example demonstrates how to implement a simple IgANet for
   learning the Poisson equation with (non-)homogeneous Dirichlet
   boundary conditions on a square geometry.

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <iganet.h>
#include <iostream>

/// @brief Specialization of the abstract IgANet class for Poisson's equation
template <typename Optimizer, typename GeometryMap, typename Variable>
class poisson
    : public iganet::IgANet<Optimizer, GeometryMap, Variable>,
      public iganet::IgANetCustomizable<Optimizer, GeometryMap, Variable> {

private:
  /// @brief Type of the base class
  using Base = iganet::IgANet<Optimizer, GeometryMap, Variable>;

  /// @brief Constructor
  template <typename... Args>
  poisson(Args... args) : Base(args...), ref_(Base::u_.clone()) {}

  /// @brief Collocation points
  typename Base::variable_collPts_type collPts_;

  /// @brief Reference solution
  Variable ref_;

  /// @brief Type of the customizable class
  using Customizable =
      iganet::IgANetCustomizable<Optimizer, GeometryMap, Variable>;

  /// @brief Knot indices of variables
  typename Customizable::variable_interior_knot_indices_type var_knot_indices_;

  /// @broef Coefficient indices of variables
  typename Customizable::variable_interior_coeff_indices_type
      var_coeff_indices_;

  /// @brief Knot indices of the geometry map
  typename Customizable::geometryMap_interior_knot_indices_type G_knot_indices_;

  /// @broef Coefficient indices of the geometry map
  typename Customizable::geometryMap_interior_coeff_indices_type
      G_coeff_indices_;

public:
  /// @brief Constructors from the base class
  using iganet::IgANet<Optimizer, GeometryMap, Variable>::IgANet;

  /// @brief Returns a constant reference to the collocation points
  auto const &collPts() const { return collPts_; }

  /// @brief Returns a constant reference to the reference solution
  auto const &ref() const { return ref_; }

  /// @brief Returns a non-constant reference to the reference solution
  auto &ref() { return ref_; }

  /// @brief Initializes the epoch
  ///
  /// @param[in] epoch Epoch number
  bool epoch(int64_t epoch) override {
    std::clog << "Epoch " << std::to_string(epoch) << ": ";

    // In the very first epoch we need to generate the sampling points
    // for the inputs and the sampling points in the function space of
    // the variables since otherwise the respective tensors would be
    // empty. In all further epochs no updates are needed since we do
    // not change the inputs nor the variable function space.
    if (epoch == 0) {
      Base::inputs(epoch);
      collPts_ = Base::variable_collPts(iganet::collPts::greville);

      var_knot_indices_ =
          Base::f_.template find_knot_indices<iganet::functionspace::interior>(
              collPts_.first);
      var_coeff_indices_ =
          Base::f_.template find_coeff_indices<iganet::functionspace::interior>(
              var_knot_indices_);

      G_knot_indices_ =
          Base::G_.template find_knot_indices<iganet::functionspace::interior>(
              collPts_.first);
      G_coeff_indices_ =
          Base::G_.template find_coeff_indices<iganet::functionspace::interior>(
              G_knot_indices_);

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
    Base::u_.from_tensor(outputs, false);

    // Evaluate the Laplacian operator
    auto u_ilapl =
        Base::u_.ilapl(Base::G_, collPts_.first, var_knot_indices_,
                       var_coeff_indices_, G_knot_indices_, G_coeff_indices_);

    auto f =
        Base::f_.eval(collPts_.first, var_knot_indices_, var_coeff_indices_);

    // Evaluate the loss function
    return torch::mse_loss(*u_ilapl[0], *f[0]);

    // auto loss_pde =
    //     torch::mse_loss(*sol_ilaplace[0] + *sol_ilaplace[3], *rhs[0]);

    // auto rhs = Base::f_.eval(variable_collPts.first);

    // // Evaluate solution at the boundary
    // auto bdr_pred = Base::u_.template eval<iganet::functionspace::boundary>(
    //     variable_collPts.second);
    // auto bdr_cond = Base::f_.template eval<iganet::functionspace::boundary>(
    //     variable_collPts.second);

    // // Evaluate boundary losses
    // auto loss_bdr0 =
    //     torch::mse_loss(*std::get<0>(bdr_pred)[0],
    //     *std::get<0>(bdr_cond)[0]);
    // auto loss_bdr1 =
    //     torch::mse_loss(*std::get<1>(bdr_pred)[0],
    //     *std::get<1>(bdr_cond)[0]);
    // auto loss_bdr2 =
    //     torch::mse_loss(*std::get<2>(bdr_pred)[0],
    //     *std::get<2>(bdr_cond)[0]);
    // auto loss_bdr3 =
    //     torch::mse_loss(*std::get<3>(bdr_pred)[0],
    //     *std::get<3>(bdr_cond)[0]);

    // return torch::mse_loss(*Base::u_.eval(variable_collPts.first)[0],
    //                          *rhs[0]) +
    //          0 * (loss_bdr0 + loss_bdr1 + loss_bdr2 + loss_bdr3);

    // // Evaluate pde loss
    // auto sol_ilaplace = Base::u_.ihess(Base::G_, variable_collPts.first);
    // auto loss_pde =
    //     torch::mse_loss(*sol_ilaplace[0] + *sol_ilaplace[3], *rhs[0]);

    // return loss_pde + 0 * (loss_bdr0 + loss_bdr1 + loss_bdr2 + loss_bdr3);
  }
};

int main() {
  iganet::init();
  iganet::verbose(std::cout);

  nlohmann::json json;
  json["res0"] = 50;
  json["res1"] = 50;

  using namespace iganet::literals;
  using optimizer_t = torch::optim::Adam;
  using real_t = double;

  using geometry_t = iganet::S2<iganet::UniformBSpline<real_t, 2, 1, 1>>;
  using variable_t = iganet::S2<iganet::UniformBSpline<real_t, 1, 2, 2>>;

  poisson<optimizer_t, geometry_t, variable_t>
      net( // Number of neurons per layers
          {50, 50, 50, 50, 50},
          // Activation functions
          {{iganet::activation::sigmoid},
           {iganet::activation::sigmoid},
           {iganet::activation::sigmoid},
           {iganet::activation::sigmoid},
           {iganet::activation::sigmoid},
           {iganet::activation::none}},
          // Number of B-spline coefficients of the geometry, just [0,1] x [0,1]
          std::tuple(iganet::utils::to_array(2_i64, 2_i64)),
          // Number of B-spline coefficients of the variable
          std::tuple(iganet::utils::to_array(10_i64, 10_i64)));

  // Impose the negative of the second derivative of sin(M_PI*x) *
  // sin(M_PI*y) as right-hand side vector (manufactured solution)
  net.f().transform([](const std::array<real_t, 2> xi) {
    return std::array<real_t, 1>{-2.0 * M_PI * M_PI * sin(M_PI * xi[0]) *
                                 sin(M_PI * xi[1])};
  });

  // Impose reference solution
  net.ref().transform([](const std::array<real_t, 2> xi) {
    return std::array<real_t, 1>{sin(M_PI * xi[0]) * sin(M_PI * xi[1])};
  });

  // Set maximum number of epoches
  net.options().max_epoch(5000);

  // Set tolerance for the loss functions
  net.options().min_loss(1e-8);

  // Start time measurement
  auto t1 = std::chrono::high_resolution_clock::now();

  // Train network
  net.train();

  // Stop time measurement
  auto t2 = std::chrono::high_resolution_clock::now();
  std::cout << "Training took "
            << std::chrono::duration_cast<std::chrono::duration<double>>(t2 -
                                                                         t1)
                   .count()
            << " seconds\n";

#ifdef IGANET_WITH_MATPLOT
  // Plot the solution
  net.G().plot(net.u(), net.collPts().first, json)->show();

  // Plot the difference between the exact and predicted solutions
  net.G().plot(net.ref().abs_diff(net.u()), net.collPts().first, json)->show();
#endif

  return 0;
}
