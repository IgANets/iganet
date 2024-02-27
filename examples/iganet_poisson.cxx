/**
   @file examples/iganet_poisson.cxx

   @brief Demonstration of IgANet Poisson solver

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <iganet.h>
#include <iostream>

/// @brief IgANet for Poisson's equation
template <typename Optimizer, typename GeometryMap, typename Variable>
class poisson
    : public iganet::IgANet<Optimizer, GeometryMap, Variable>,
      public iganet::IgANetCustomizable<Optimizer, GeometryMap, Variable> {
public:
  bool pde;

private:
  /// @brief Type of the base class
  using Base = iganet::IgANet<Optimizer, GeometryMap, Variable>;

  /// @brief Type of the customizable class
  using Customizable =
      iganet::IgANetCustomizable<Optimizer, GeometryMap, Variable>;

public:
  /// @brief Constructors from the base class
  using iganet::IgANet<Optimizer, GeometryMap, Variable>::IgANet;

  /// @brief Initializes the epoch
  iganet::status epoch(int64_t epoch) override {
    std::cout << "Epoch " << std::to_string(epoch) << ": ";

    return (epoch == 0
                ? iganet::status::inputs + iganet::status::geometryMap_collPts +
                      iganet::status::variable_collPts
                : iganet::status::inputs);
  }

  /// @brief Computes the loss function
  torch::Tensor
  loss(const torch::Tensor &outputs,
       const typename Base::geometryMap_collPts_type &geometryMap_collPts,
       const typename Base::variable_collPts_type &variable_collPts,
       int64_t epoch, iganet::status status) override {

    // Update indices and precompute basis functions for geometry
    if (status & iganet::status::geometryMap_collPts) {
      Customizable::geometryMap_interior_knot_indices_ =
          Base::G_.template find_knot_indices<iganet::functionspace::interior>(
              geometryMap_collPts.first);
      Customizable::geometryMap_interior_coeff_indices_ =
          Base::G_.template find_coeff_indices<iganet::functionspace::interior>(
              Customizable::geometryMap_interior_knot_indices_);

      Customizable::geometryMap_boundary_knot_indices_ =
          Base::G_.template find_knot_indices<iganet::functionspace::boundary>(
              geometryMap_collPts.second);
      Customizable::geometryMap_boundary_coeff_indices_ =
          Base::G_.template find_coeff_indices<iganet::functionspace::boundary>(
              Customizable::geometryMap_boundary_knot_indices_);
    }

    // Update indices and precompute basis functions for variable
    if (status & iganet::status::variable_collPts) {
      Customizable::variable_interior_knot_indices_ =
          Base::f_.template find_knot_indices<iganet::functionspace::interior>(
              variable_collPts.first);
      Customizable::variable_interior_coeff_indices_ =
          Base::f_.template find_coeff_indices<iganet::functionspace::interior>(
              Customizable::variable_interior_knot_indices_);

      Customizable::variable_boundary_knot_indices_ =
          Base::f_.template find_knot_indices<iganet::functionspace::boundary>(
              variable_collPts.second);
      Customizable::variable_boundary_coeff_indices_ =
          Base::f_.template find_coeff_indices<iganet::functionspace::boundary>(
              Customizable::variable_boundary_knot_indices_);
    }

    // Evaluate right-hand side in the interior
    Base::u_.from_tensor(outputs, false);
    auto rhs = Base::f_.eval(variable_collPts.first);

    // Evaluate solution at the boundary
    auto bdr_pred = Base::u_.template eval<iganet::functionspace::boundary>(
        variable_collPts.second);
    auto bdr_cond = Base::f_.template eval<iganet::functionspace::boundary>(
        variable_collPts.second);

    // Evaluate boundary losses
    auto loss_bdr0 =
        torch::mse_loss(*std::get<0>(bdr_pred)[0], *std::get<0>(bdr_cond)[0]);
    auto loss_bdr1 =
        torch::mse_loss(*std::get<1>(bdr_pred)[0], *std::get<1>(bdr_cond)[0]);
    auto loss_bdr2 =
        torch::mse_loss(*std::get<2>(bdr_pred)[0], *std::get<2>(bdr_cond)[0]);
    auto loss_bdr3 =
        torch::mse_loss(*std::get<3>(bdr_pred)[0], *std::get<3>(bdr_cond)[0]);

    // std::cout << loss_bdr0.template item<double>() << ", "
    //           << loss_bdr1.template item<double>() << ", "
    //           << loss_bdr2.template item<double>() << ", "
    //           << loss_bdr3.template item<double>() << std::endl;

    // Evaluate loss function
    if (!pde)
      return torch::mse_loss(*Base::u_.eval(variable_collPts.first)[0],
                             *rhs[0]) +
             0 * (loss_bdr0 + loss_bdr1 + loss_bdr2 + loss_bdr3);

    // Evaluate pde loss
    auto sol_ilaplace = Base::u_.ihess(Base::G_, variable_collPts.first);
    auto loss_pde =
        torch::mse_loss(*sol_ilaplace[0] + *sol_ilaplace[3], *rhs[0]);

    return loss_pde + 0 * (loss_bdr0 + loss_bdr1 + loss_bdr2 + loss_bdr3);
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

  using geometry_t = iganet::S2<iganet::UniformBSpline<real_t, 2, 2, 2>>;
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
          // Number of B-spline coefficients
          std::tuple(iganet::utils::to_array(7_i64, 7_i64)));

  // Deform geometry
  // net.G().transform([](const std::array<real_t, 2> xi) {
  //   return std::array<real_t, 2>{(xi[0] + 1) * cos(M_PI * xi[1]),
  //                                (xi[0] + 1) * sin(M_PI * xi[1])};
  // });

  // Impose solution value for supervised training (not right-hand side)
  net.f().transform([](const std::array<real_t, 2> xi) {
    return std::array<real_t, 1>{sin(M_PI * xi[0]) * sin(M_PI * xi[1])};
  });

  // Impose boundary values
  net.f().boundary().template side<1>().transform(
      [](const std::array<real_t, 1> xi) {
        return std::array<real_t, 1>{0.0};
      });

  net.f().boundary().template side<2>().transform(
      [](const std::array<real_t, 1> xi) {
        return std::array<real_t, 1>{0.0};
      });

  net.f().boundary().template side<3>().transform(
      [](const std::array<real_t, 1> xi) {
        return std::array<real_t, 1>{0.0};
      });

  net.f().boundary().template side<4>().transform(
      [](const std::array<real_t, 1> xi) {
        return std::array<real_t, 1>{0.0};
      });

  net.options().max_epoch(1000);
  net.options().min_loss(1e-8);
  net.train();

#ifdef IGANET_WITH_MATPLOT
  net.G().plot(net.u(), net.variable_collPts(0).first, json)->show();
#endif

  net.f().transform([](const std::array<real_t, 2> xi) {
    return std::array<real_t, 1>{-2.0 * M_PI * M_PI * sin(M_PI * xi[0]) *
                                 sin(M_PI * xi[1])};
  });

  net.pde = true;
  net.options().max_epoch(1000);
  net.options().min_loss(1e-10);
  net.train();

#ifdef IGANET_WITH_MATPLOT
  net.G().plot(net.u(), net.variable_collPts(0).first, json)->show();
#endif

  return 0;
}
