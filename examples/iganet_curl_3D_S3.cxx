/**
   @file examples/iganet_curl_3D.cxx

   @brief Demonstration of IgANet Curl Curl solver

  //TODO write brief description

   @author Merle Backmeyer

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <iganet.h>
#include <iostream>
#include <utils/blocktensor.hpp>
#include <utils/fqn.hpp>
#include <utils/integer_pow.hpp>
#include <utils/linalg.hpp>
#include <utils/serialize.hpp>
#include <utils/tensorarray.hpp>
#include <utils/vslice.hpp>

using namespace iganet::literals;

/// @brief Specialization of the abstract IgANet class for Poisson's equation
template <typename Optimizer, typename GeometryMap, typename Variable>
class poisson
    : public iganet::IgANet<Optimizer, GeometryMap, Variable>,
    public iganet::IgANetCustomizable<GeometryMap, Variable> {

private:
    /// @brief Type of the base class
    using Base = iganet::IgANet<Optimizer, GeometryMap, Variable>;

    /// @brief Collocation points
    typename Base::variable_collPts_type collPts_;

    /// @brief Reference solution
    Variable ref_;

    /// @brief Type of the customizable class
    using Customizable =
        iganet::IgANetCustomizable<GeometryMap, Variable>;

    /// @brief Knot indices of variables
    typename Customizable::variable_interior_knot_indices_type var_knot_indices_;

    /// @broef Coefficient indices of variables
    typename Customizable::variable_interior_coeff_indices_type
        var_coeff_indices_;

    /// @brief Knot indices of the geometry map
    typename Customizable::geometryMap_interior_knot_indices_type G_knot_indices_;

    /// @brief Coefficient indices of the geometry map
    typename Customizable::geometryMap_interior_coeff_indices_type
        G_coeff_indices_;

    /// @brief Analytic function f evaluated at physical collocation points
    typename torch::Tensor t_ana1_;
    typename torch::Tensor t_ana2_;
    typename torch::Tensor t_ana3_;


public:
    /// @brief Constructor
    template <typename... Args>
    poisson(std::vector<int64_t>&& layers,
        std::vector<std::vector<std::any>>&& activations, Args&&...args)
        : Base(std::forward<std::vector<int64_t>>(layers),
            std::forward<std::vector<std::vector<std::any>>>(activations),
            std::forward<Args>(args)...),
        ref_(iganet::utils::to_array(10_i64, 10_i64, 10_i64)) {}

    /// @brief Returns a constant reference to the collocation points
    auto const& collPts() const { return collPts_; }

    /// @brief Returns a constant reference to the reference solution
    auto const& ref() const { return ref_; }

    /// @brief Returns a non-constant reference to the reference solution
    auto& ref() { return ref_; }

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
       
            t_ana1_ = t_ana1();
            t_ana2_ = t_ana2();
            t_ana3_ = t_ana3();

            return true;
        }
        else
            return false;
    }

    /// @brief Computes the loss function
    ///
    /// @param[in] outputs Output of the network
    ///
    /// @param[in] epoch Epoch number
    torch::Tensor loss(const torch::Tensor& outputs, int64_t epoch) override {

        // Cast the network output (a raw tensor) into the proper
        // function-space format, i.e. B-spline objects for the interior
        // and boundary parts that can be evaluated.
        Base::u_.from_tensor(outputs);
        // Evaluate the weakformulation
        /*
        auto curl = Base::u_.curl(collPts_.first);
        auto sol_iweak = curl * curl.tr();

        auto rhs_weak = t_ana1_ * curl(0) + t_ana2_ * curl(1) + t_ana3_ * curl(2);

         */
        auto u_bdr = Base::u_.template eval<iganet::functionspace::boundary>(
            collPts_.second);
        auto bdr =
            ref_.template eval<iganet::functionspace::boundary>(collPts_.second);

        auto loss_pde = //torch::mean(0.5 * *sol_iweak[0] - rhs_weak) +
     //   auto curlcurl = Base::u_.curlcurl(std::get<0>(collPts_.first), var_knot_indices_, var_coeff_indices_);
     //  auto loss_pde = torch::mse_loss(*curlcurl[0], f_ana1_) + torch::mse_loss(*curlcurl[1], f_ana2_) + torch::mse_loss(*curlcurl[2], f_ana3_) + 
       //periodic bc in x direction
         //      10e1 * torch::mse_loss(*std::get<0>(std::get<2>(u_bdr))[0],  *std::get<1>(std::get<2>(u_bdr))[0]) +

       //     10e1 * torch::mse_loss( *std::get<0>(std::get<1>(u_bdr))[0], *std::get<1>(std::get<1>(u_bdr))[0]) +
          /* 10e1 * torch::mse_loss(*std::get<0>(std::get<1>(u_bdr))[0], *std::get<0>(std::get<1>(bdr))[0]) +
           10e1 * torch::mse_loss(*std::get<0>(std::get<2>(u_bdr))[0], *std::get<0>(std::get<2>(bdr))[0]) +

            10e1 * torch::mse_loss(*std::get<1>(std::get<1>(u_bdr))[0], *std::get<1>(std::get<1>(bdr))[0]) +
            10e1 * torch::mse_loss(*std::get<1>(std::get<2>(u_bdr))[0], *std::get<1>(std::get<2>(bdr))[0]) +

            10e1 * torch::mse_loss(*std::get<2>(std::get<0>(u_bdr))[0], *std::get<2>(std::get<0>(bdr))[0]) +
            10e1 * torch::mse_loss(*std::get<2>(std::get<2>(u_bdr))[0], *std::get<2>(std::get<2>(bdr))[0]) +

            10e1 * torch::mse_loss(*std::get<3>(std::get<0>(u_bdr))[0], *std::get<3>(std::get<0>(bdr))[0]) +
            10e1 * torch::mse_loss(*std::get<3>(std::get<2>(u_bdr))[0], *std::get<3>(std::get<2>(bdr))[0]) +

            10e1 * torch::mse_loss(*std::get<4>(std::get<0>(u_bdr))[0], *std::get<4>(std::get<0>(bdr))[0]) +
            10e1 * torch::mse_loss(*std::get<4>(std::get<1>(u_bdr))[0], *std::get<4>(std::get<1>(bdr))[0]) +

            10e1 * torch::mse_loss(*std::get<5>(std::get<0>(u_bdr))[0], *std::get<5>(std::get<0>(bdr))[0]) +*/
            10e1 * torch::mse_loss(*std::get<5>(u_bdr)[0], *std::get<5>(bdr)[0]);


        // Evaluate the loss function
        std::cout << loss_pde.item() << std::endl;

        return loss_pde;
    }

    torch::Tensor t_ana1() {

        auto Geo = Base::G_.eval(collPts_.first, G_knot_indices_, G_coeff_indices_);
        auto x = *Geo[0];
        auto y = *Geo[1];
        auto z = *Geo[2];

        //Allocate space for torch::Tensor with results
        torch::Tensor T = torch::zeros(x.size(0), torch::dtype(torch::kDouble));
        for (int i = 0; i < x.size(0); i++) {
            T[i] = M_PI * (2.0 * sin(2.0 * M_PI * x[i]) * cos(2.0 * M_PI * y[i]) - sin(2.0 * M_PI * x[i]) * cos(M_PI * z[i])) - sin(M_PI * x[i]) * y[i] * y[i] * (1.0 / 3.0 * y[i] - 0.5);
        };
        return T;//return tensor
    }
    torch::Tensor t_ana2() {

        auto Geo = Base::G_.eval(collPts_.first, G_knot_indices_, G_coeff_indices_);
        auto x = *Geo[0];
        auto y = *Geo[1];
        auto z = *Geo[2];

        //Allocate space for torch::Tensor with results
        torch::Tensor T = torch::zeros(x.size(0), torch::dtype(torch::kDouble));
        for (int i = 0; i < x.size(0); i++) {
            T[i] = M_PI * (-2.0 * sin(2.0 * M_PI * y[i]) * cos(2.0 * M_PI * x[i]) + sin(2 * M_PI * y[i]) * cos(M_PI * z[i])) + cos(M_PI * x[i]) * y[i] * (y[i] - 1.0);
        };
        return T;//return tensor
    }
    torch::Tensor t_ana3() {

        auto Geo = Base::G_.eval(collPts_.first, G_knot_indices_, G_coeff_indices_);
        auto x = *Geo[0];
        auto y = *Geo[1];
        auto z = *Geo[2];

        //Allocate space for torch::Tensor with results
        torch::Tensor T = torch::zeros(x.size(0), torch::dtype(torch::kDouble));
        for (int i = 0; i < x.size(0); i++) {
            T[i] = 2.0 * M_PI * (cos(2.0 * M_PI * x[i]) * sin(M_PI * z[i]) - cos(2.0 * M_PI * y[i]) * sin(M_PI * z[i]));
        };
        return T;//return tensor
    }
};



int main() {
    iganet::init();
    iganet::verbose(std::cout);

    nlohmann::json json;
    json["res0"] = 50;
    json["res1"] = 50;
    json["res2"] = 50;

    using namespace iganet::literals;
    using optimizer_t = torch::optim::LBFGS;
    using real_t = double;

    using geometry_t = iganet::S<iganet::UniformBSpline<real_t, 3, 1, 1, 1>>;
    using variable_t = iganet::S<iganet::UniformBSpline<real_t, 3, 2, 2, 2>>;

    poisson<optimizer_t, geometry_t, variable_t>
        net( // Number of neurons per layers
            { 50,50 },
            // Activation functions
            { {iganet::activation::tanh},
             {iganet::activation::tanh},
             {iganet::activation::none} },
            // Number of B-spline coefficients of the geometry, just [0,1] x [0,1]
            std::tuple(iganet::utils::to_array(2_i64, 2_i64, 2_i64)),
            // Number of B-spline coefficients of the variable
            std::tuple(iganet::utils::to_array(10_i64, 10_i64, 10_i64)));


        // Impose reference solution
    net.ref().transform([](const std::array<real_t, 3> xi) {
        return std::array<real_t, 3>{M_PI* (2.0 * sin(2.0 * M_PI * xi[0]) * cos(2.0 * M_PI * xi[1]) - sin(2.0 * M_PI * xi[0]) * cos(M_PI * xi[2])),
            M_PI* (-2.0 * sin(2.0 * M_PI * xi[1]) * cos(2.0 * M_PI * xi[0]) + sin(2.0 * M_PI * xi[1]) * cos(M_PI * xi[2])),
            M_PI * 2.0 * (cos(2.0 * M_PI * xi[0]) * sin(M_PI * xi[2]) - cos(2.0 * M_PI * xi[1]) * sin(M_PI * xi[2]))
        };
        });
    
    //Impose the boundary conditions

    net.ref().boundary().template side<1>().transform(
        [](const std::array<real_t, 2> xi) {
            return std::array<real_t, 3>{0.0, 0.0, 0.0};
        });

    net.ref().template boundary().template side<2>().transform(
        [](const std::array<real_t, 2> xi) {
            return std::array<real_t, 3>{0.0, 0.0, 0.0};
        });
 
    net.ref().template boundary().template side<3>().transform(
        [](const std::array<real_t, 2> xi) {
            return std::array<real_t, 3>{0.0, 0.0, 0.0};
        });

    net.ref().template boundary().template side<4>().transform(
        [](const std::array<real_t, 2> xi) {
            return std::array<real_t, 3>{0.0, 0.0, 0.0};
        });
  
    net.ref().template boundary().template side<5>().transform(
        [](const std::array<real_t, 2> xi) {
            return std::array<real_t, 3>{0.0, 0.0, 0.0};
        });


    net.ref().template boundary().template side<6>().transform(
        [](const std::array<real_t, 2> xi) {
            return std::array<real_t, 3>{0.0, 0.0, 0.0};
        });



  // Set maximum number of epoches
    net.options().max_epoch(10);

    // Set tolerance for the loss functions
    net.options().min_loss(-5000);

    // Start time measurement
    auto t1 = std::chrono::high_resolution_clock::now();

    net.train();
    // Stop time measurement
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Training took "
        << std::chrono::duration_cast<std::chrono::duration<double>>(t2 -
            t1)
        .count()
        << " seconds\n";

    return 0;
}
