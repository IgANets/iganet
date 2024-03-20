/**
   @file unittests/unittest_functionspace.cxx

   @brief Function space unittests

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <filesystem>
#include <iganet.h>
#include <iostream>

#include <gtest/gtest.h>
#include <unittest_config.hpp>

class FunctionSpaceTest : public ::testing::Test {
protected:
  using real_t = iganet::unittests::real_t;
  iganet::Options<real_t> options;
};

TEST_F(FunctionSpaceTest, S1_geoDim1_degrees2) {
  using iganet::deriv;
  using iganet::functionspace;
  using iganet::side;
  using BSpline = iganet::UniformBSpline<real_t, 1, 2>;
  iganet::S1<BSpline> functionspace({5}, iganet::init::greville, options);
  BSpline bspline({5}, iganet::init::greville, options);

  { // Interior

    auto xi = iganet::utils::to_tensorArray<real_t>(
        {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, options);

    // Evaluation
    EXPECT_TRUE(torch::equal(
        *(functionspace.eval<functionspace::interior, deriv::func, false>(
            xi)[0]),
        *(bspline.eval<deriv::func, false>(xi)[0])));

    EXPECT_TRUE(torch::equal(
        *(functionspace.eval<functionspace::interior, deriv::dx, false>(xi)[0]),
        *(bspline.eval<deriv::dx, false>(xi)[0])));

    EXPECT_TRUE(torch::equal(
        *(functionspace.eval<functionspace::interior, deriv::dx ^ 2, false>(
            xi)[0]),
        *(bspline.eval<deriv::dx ^ 2, false>(xi)[0])));

    // Evaluation from precomputed coefficients and basis functions
    auto knot_indices =
        functionspace.template find_knot_indices<functionspace::interior>(xi);
    auto coeff_indices =
        functionspace.template find_coeff_indices<functionspace::interior>(
            knot_indices);

    auto basfunc =
        functionspace
            .template eval_basfunc<functionspace::interior, deriv::func, false>(
                xi, knot_indices);

    EXPECT_TRUE(torch::equal(
        *(functionspace.eval_from_precomputed<functionspace::interior>(
            basfunc, coeff_indices, xi[0].numel(), xi[0].sizes())[0]),
        *(bspline.eval<deriv::func, false>(xi)[0])));

    basfunc =
        functionspace
            .template eval_basfunc<functionspace::interior, deriv::dx, false>(
                xi, knot_indices);
    EXPECT_TRUE(torch::equal(
        *(functionspace.eval_from_precomputed<functionspace::interior>(
            basfunc, coeff_indices, xi[0].numel(), xi[0].sizes())[0]),
        *(bspline.eval<deriv::dx, false>(xi)[0])));

    basfunc = functionspace.template eval_basfunc<functionspace::interior,
                                                  deriv::dx ^ 2, false>(
        xi, knot_indices);
    EXPECT_TRUE(torch::equal(
        *(functionspace.eval_from_precomputed<functionspace::interior>(
            basfunc, coeff_indices, xi[0].numel(), xi[0].sizes())[0]),
        *(bspline.eval<deriv::dx ^ 2, false>(xi)[0])));

    /// Evaluation of gradient (in parametric domain)
    auto grad = functionspace.template grad<functionspace::interior, false>(xi);
    auto grad_ref = bspline.grad(xi);

    for (std::size_t i = 0; i < grad.entries(); ++i)
      EXPECT_TRUE(torch::equal(*(grad)[i], *(grad_ref)[i]));

    /// Evaluation of gradient (in physical domain)
    auto igrad = functionspace.template igrad<functionspace::interior, false>(
        functionspace, xi);
    auto igrad_ref = bspline.igrad(bspline, xi);

    for (std::size_t i = 0; i < igrad.entries(); ++i)
      EXPECT_TRUE(torch::equal(*(igrad)[i], *(igrad_ref)[i]));

    /// Evaluation of Jacobian (in parametric domain)
    auto jac = functionspace.template jac<functionspace::interior, false>(xi);
    auto jac_ref = bspline.jac(xi);

    for (std::size_t i = 0; i < jac.entries(); ++i)
      EXPECT_TRUE(torch::equal(*(jac)[i], *(jac_ref)[i]));

    /// Evaluation of Jacobian (in physical domain)
    auto ijac = functionspace.template ijac<functionspace::interior, false>(
        functionspace, xi);
    auto ijac_ref = bspline.ijac(bspline, xi);

    for (std::size_t i = 0; i < ijac.entries(); ++i)
      EXPECT_TRUE(torch::equal(*(ijac)[i], *(ijac_ref)[i]));

    /// Evaluation of Hessian (in parametric domain)
    auto hess = functionspace.template hess<functionspace::interior, false>(xi);
    auto hess_ref = bspline.hess(xi);

    for (std::size_t i = 0; i < hess.entries(); ++i)
      EXPECT_TRUE(torch::equal(*(hess)[i], *(hess_ref)[i]));

    /// Evaluation of Hessian (in physical domain)
    auto ihess = functionspace.template ihess<functionspace::interior, false>(
        functionspace, xi);
    auto ihess_ref = bspline.ihess(bspline, xi);

    for (std::size_t i = 0; i < ihess.entries(); ++i)
      EXPECT_TRUE(torch::equal(*(ihess)[i], *(ihess_ref)[i]));
  }

  { // Boundary

    auto xi = std::tuple{std::array<torch::Tensor, 0>{},
                         std::array<torch::Tensor, 0>{}};

    // Evaluation
    auto eval =
        functionspace.eval<functionspace::boundary, deriv::func, false>(xi);

    EXPECT_TRUE(torch::equal(*std::get<side::left - 1>(eval)[0],
                             torch::zeros(1, options)));
    EXPECT_TRUE(torch::equal(*std::get<side::right - 1>(eval)[0],
                             torch::ones(1, options)));

    eval = functionspace.eval<functionspace::boundary, deriv::dx, false>(xi);

    EXPECT_TRUE(torch::equal(*std::get<side::left - 1>(eval)[0],
                             torch::zeros(1, options)));
    EXPECT_TRUE(torch::equal(*std::get<side::right - 1>(eval)[0],
                             torch::zeros(1, options)));

    eval =
        functionspace.eval<functionspace::boundary, deriv::dx ^ 2, false>(xi);

    EXPECT_TRUE(torch::equal(*std::get<side::left - 1>(eval)[0],
                             torch::zeros(1, options)));
    EXPECT_TRUE(torch::equal(*std::get<side::right - 1>(eval)[0],
                             torch::zeros(1, options)));

    // Evaluation from precomputed coefficients and basis functions
    auto knot_indices =
        functionspace.template find_knot_indices<functionspace::boundary>(xi);
    auto coeff_indices =
        functionspace.template find_coeff_indices<functionspace::boundary>(
            knot_indices);

    auto numel = [](const auto &xi) {
      return std::tuple<int64_t, int64_t>{1, 1};
    };
    auto sizes = [](const auto &xi) {
      return std::tuple{torch::IntArrayRef{}, torch::IntArrayRef{}};
    };

    auto basfunc =
        functionspace
            .template eval_basfunc<functionspace::boundary, deriv::func, false>(
                xi, knot_indices);
    eval =
        functionspace.template eval_from_precomputed<functionspace::boundary>(
            basfunc, coeff_indices, numel(xi), sizes(xi));

    EXPECT_TRUE(torch::allclose(*std::get<side::left - 1>(eval)[0],
                                torch::zeros(1, options)));
    EXPECT_TRUE(torch::allclose(*std::get<side::right - 1>(eval)[0],
                                torch::ones(1, options)));

    basfunc =
        functionspace
            .template eval_basfunc<functionspace::boundary, deriv::dx, false>(
                xi, knot_indices);
    eval =
        functionspace.template eval_from_precomputed<functionspace::boundary>(
            basfunc, coeff_indices, numel(xi), sizes(xi));

    EXPECT_TRUE(torch::equal(*std::get<side::left - 1>(eval)[0],
                             torch::zeros({}, options)));
    EXPECT_TRUE(torch::equal(*std::get<side::right - 1>(eval)[0],
                             torch::zeros({}, options)));

    basfunc = functionspace.template eval_basfunc<functionspace::boundary,
                                                  deriv::dx ^ 2, false>(
        xi, knot_indices);
    eval =
        functionspace.template eval_from_precomputed<functionspace::boundary>(
            basfunc, coeff_indices, numel(xi), sizes(xi));

    EXPECT_TRUE(torch::equal(*std::get<side::left - 1>(eval)[0],
                             torch::zeros({}, options)));
    EXPECT_TRUE(torch::equal(*std::get<side::right - 1>(eval)[0],
                             torch::zeros({}, options)));

    /// Evaluation of gradient (in parametric domain)
    auto grad = functionspace.template grad<functionspace::boundary, false>(xi);

    for (std::size_t i = 0; i < std::get<0>(grad).entries(); ++i) {
      EXPECT_TRUE(torch::equal(*std::get<side::left - 1>(grad)[i],
                               torch::zeros({1}, options)));
      EXPECT_TRUE(torch::equal(*std::get<side::right - 1>(grad)[i],
                               torch::zeros({1}, options)));
    }

    /// Evaluation of gradient (in physical domain)
    auto igrad = functionspace.template igrad<functionspace::boundary, false>(
        functionspace, xi);

    for (std::size_t i = 0; i < std::get<0>(igrad).entries(); ++i) {
      EXPECT_TRUE(torch::equal(*std::get<side::left - 1>(igrad)[i],
                               torch::zeros({1}, options)));
      EXPECT_TRUE(torch::equal(*std::get<side::right - 1>(igrad)[i],
                               torch::zeros({1}, options)));
    }

    /// Evaluation of Jacobian (in parametric domain)
    auto jac = functionspace.template jac<functionspace::boundary, false>(xi);

    for (std::size_t i = 0; i < std::get<0>(jac).entries(); ++i) {
      EXPECT_TRUE(torch::equal(*std::get<side::left - 1>(jac)[i],
                               torch::zeros({1}, options)));
      EXPECT_TRUE(torch::equal(*std::get<side::right - 1>(jac)[i],
                               torch::zeros({1}, options)));
    }

    /// Evaluation of Jacobian (in physical domain)
    auto ijac = functionspace.template ijac<functionspace::boundary, false>(
        functionspace, xi);

    for (std::size_t i = 0; i < std::get<0>(ijac).entries(); ++i) {
      EXPECT_TRUE(torch::equal(*std::get<side::left - 1>(ijac)[i],
                               torch::zeros({1}, options)));
      EXPECT_TRUE(torch::equal(*std::get<side::right - 1>(ijac)[i],
                               torch::zeros({1}, options)));
    }

    /// Evaluation of Hessian (in parametric domain)
    auto hess = functionspace.template hess<functionspace::boundary, false>(xi);

    for (std::size_t i = 0; i < std::get<0>(hess).entries(); ++i) {
      EXPECT_TRUE(torch::equal(*std::get<side::left - 1>(hess)[i],
                               torch::zeros({1}, options)));
      EXPECT_TRUE(torch::equal(*std::get<side::right - 1>(hess)[i],
                               torch::zeros({1}, options)));
    }

    /// Evaluation of Hessian (in physical domain)
    auto ihess = functionspace.template ihess<functionspace::boundary, false>(
        functionspace, xi);

    for (std::size_t i = 0; i < std::get<0>(ihess).entries(); ++i) {
      EXPECT_TRUE(torch::equal(*std::get<side::left - 1>(ihess)[i],
                               torch::zeros({1}, options)));
      EXPECT_TRUE(torch::equal(*std::get<side::right - 1>(ihess)[i],
                               torch::zeros({1}, options)));
    }
  }
}

TEST_F(FunctionSpaceTest, S2_geoDim1_degrees23) {
  using iganet::deriv;
  using iganet::functionspace;
  using iganet::side;
  using BSpline = iganet::UniformBSpline<real_t, 1, 2, 3>;
  using Geometry = iganet::UniformBSpline<real_t, 2, 2, 3>;
  iganet::S2<BSpline> functionspace({5, 4}, iganet::init::greville, options);
  iganet::S2<Geometry> S2_geometry({5, 4}, iganet::init::greville, options);
  BSpline bspline({5, 4}, iganet::init::greville, options);
  Geometry geometry({5, 4}, iganet::init::greville, options);

  { // Interior

    auto xi = iganet::utils::to_tensorArray<real_t>(
        {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0} /* u */,
        {1.0, 0.2, 0.1, 0.5, 0.9, 0.75, 0.0} /* v */, options);

    // Evaluation
    EXPECT_TRUE(torch::equal(
        *(functionspace.eval<functionspace::interior, deriv::func, false>(
            xi)[0]),
        *(bspline.eval<deriv::func, false>(xi)[0])));

    EXPECT_TRUE(torch::equal(
        *(functionspace.eval<functionspace::interior, deriv::dx, false>(xi)[0]),
        *(bspline.eval<deriv::dx, false>(xi)[0])));

    EXPECT_TRUE(torch::equal(
        *(functionspace.eval<functionspace::interior, deriv::dx ^ 2, false>(
            xi)[0]),
        *(bspline.eval<deriv::dx ^ 2, false>(xi)[0])));

    EXPECT_TRUE(torch::equal(
        *(functionspace.eval<functionspace::interior, deriv::dy, false>(xi)[0]),
        *(bspline.eval<deriv::dy, false>(xi)[0])));

    EXPECT_TRUE(torch::equal(
        *(functionspace.eval<functionspace::interior, deriv::dy ^ 2, false>(
            xi)[0]),
        *(bspline.eval<deriv::dy ^ 2, false>(xi)[0])));

    EXPECT_TRUE(
        torch::equal(*(functionspace.eval<functionspace::interior,
                                          deriv::dx + deriv::dy, false>(xi)[0]),
                     *(bspline.eval<deriv::dx + deriv::dy, false>(xi)[0])));

    // Evaluation from precomputed coefficients and basis functions
    auto knot_indices =
        functionspace.template find_knot_indices<functionspace::interior>(xi);
    auto coeff_indices =
        functionspace.template find_coeff_indices<functionspace::interior>(
            knot_indices);

    auto basfunc =
        functionspace
            .template eval_basfunc<functionspace::interior, deriv::func, false>(
                xi, knot_indices);
    EXPECT_TRUE(torch::equal(
        *(functionspace.eval_from_precomputed<functionspace::interior>(
            basfunc, coeff_indices, xi[0].numel(), xi[0].sizes())[0]),
        *(bspline.eval<deriv::func, false>(xi)[0])));

    basfunc =
        functionspace
            .template eval_basfunc<functionspace::interior, deriv::dx, false>(
                xi, knot_indices);
    EXPECT_TRUE(torch::equal(
        *(functionspace.eval_from_precomputed<functionspace::interior>(
            basfunc, coeff_indices, xi[0].numel(), xi[0].sizes())[0]),
        *(bspline.eval<deriv::dx, false>(xi)[0])));

    basfunc = functionspace.template eval_basfunc<functionspace::interior,
                                                  deriv::dx ^ 2, false>(
        xi, knot_indices);
    EXPECT_TRUE(torch::equal(
        *(functionspace.eval_from_precomputed<functionspace::interior>(
            basfunc, coeff_indices, xi[0].numel(), xi[0].sizes())[0]),
        *(bspline.eval<deriv::dx ^ 2, false>(xi)[0])));

    basfunc =
        functionspace
            .template eval_basfunc<functionspace::interior, deriv::dy, false>(
                xi, knot_indices);
    EXPECT_TRUE(torch::equal(
        *(functionspace.eval_from_precomputed<functionspace::interior>(
            basfunc, coeff_indices, xi[0].numel(), xi[0].sizes())[0]),
        *(bspline.eval<deriv::dy, false>(xi)[0])));

    basfunc = functionspace.template eval_basfunc<functionspace::interior,
                                                  deriv::dy ^ 2, false>(
        xi, knot_indices);
    EXPECT_TRUE(torch::equal(
        *(functionspace.eval_from_precomputed<functionspace::interior>(
            basfunc, coeff_indices, xi[0].numel(), xi[0].sizes())[0]),
        *(bspline.eval<deriv::dy ^ 2, false>(xi)[0])));

    basfunc = functionspace.template eval_basfunc<functionspace::interior,
                                                  deriv::dx + deriv::dy, false>(
        xi, knot_indices);
    EXPECT_TRUE(torch::equal(
        *(functionspace.eval_from_precomputed<functionspace::interior>(
            basfunc, coeff_indices, xi[0].numel(), xi[0].sizes())[0]),
        *(bspline.eval<deriv::dx + deriv::dy, false>(xi)[0])));

    /// Evaluation of gradient (in parametric domain)
    auto grad = functionspace.template grad<functionspace::interior, false>(xi);
    auto grad_ref = bspline.grad(xi);

    for (std::size_t i = 0; i < grad.entries(); ++i)
      EXPECT_TRUE(torch::equal(*(grad)[i], *(grad_ref)[i]));

    /// Evaluation of gradient (in physical domain)
    auto igrad = functionspace.template igrad<functionspace::interior, false>(
        S2_geometry, xi);
    auto igrad_ref = bspline.igrad(geometry, xi);

    for (std::size_t i = 0; i < igrad.entries(); ++i)
      EXPECT_TRUE(torch::equal(*(igrad)[i], *(igrad_ref)[i]));

    /// Evaluation of Jacobian (in parametric domain)
    auto jac = functionspace.template jac<functionspace::interior>(xi);
    auto jac_ref = bspline.jac(xi);

    for (std::size_t i = 0; i < jac.entries(); ++i)
      EXPECT_TRUE(torch::equal(*(jac)[i], *(jac_ref)[i]));

    /// Evaluation of Jacobian (in physical domain)
    auto ijac = functionspace.template ijac<functionspace::interior, false>(
        S2_geometry, xi);
    auto ijac_ref = bspline.ijac(geometry, xi);

    for (std::size_t i = 0; i < ijac.entries(); ++i)
      EXPECT_TRUE(torch::equal(*(ijac)[i], *(ijac_ref)[i]));

    /// Evaluation of Hessian (in parametric domain)
    auto hess = functionspace.template hess<functionspace::interior, false>(xi);
    auto hess_ref = bspline.hess(xi);

    for (std::size_t i = 0; i < hess.entries(); ++i)
      EXPECT_TRUE(torch::equal(*(hess)[i], *(hess_ref)[i]));

    /// Evaluation of Hessian (in physical domain)
    auto ihess = functionspace.template ihess<functionspace::interior, false>(
        S2_geometry, xi);
    auto ihess_ref = bspline.ihess(geometry, xi);

    for (std::size_t i = 0; i < ihess.entries(); ++i)
      EXPECT_TRUE(torch::equal(*(ihess)[i], *(ihess_ref)[i]));
  }

  { // Boundary

    iganet::UniformBSpline<real_t, 1, 2> bspline_bdrNS(
        {5}, iganet::init::greville, options);
    iganet::UniformBSpline<real_t, 1, 3> bspline_bdrEW(
        {4}, iganet::init::greville, options);

    auto xi = std::tuple{
        iganet::utils::to_tensorArray<real_t>(
            {1.0, 0.2, 0.1, 0.5, 0.9, 0.75, 0.0}, options) /* west  */,
        iganet::utils::to_tensorArray<real_t>(
            {1.0, 0.2, 0.1, 0.5, 0.9, 0.75, 0.0}, options) /* east  */,
        iganet::utils::to_tensorArray<real_t>(
            {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, options) /* south */,
        iganet::utils::to_tensorArray<real_t>(
            {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, options) /* north */};

    // Evaluation
    auto eval =
        functionspace.eval<functionspace::boundary, deriv::func, false>(xi);

    EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::func, false>(
                                 std::get<side::north - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::func, false>(
                                 std::get<side::south - 1>(xi))[0])));
    EXPECT_TRUE(torch::allclose(*(std::get<side::east - 1>(eval)[0]),
                                torch::ones(7, options)));
    EXPECT_TRUE(torch::allclose(*(std::get<side::west - 1>(eval)[0]),
                                torch::zeros(7, options)));

    eval = functionspace.eval<functionspace::boundary, deriv::dx, false>(xi);

    EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dx, false>(
                                 std::get<side::north - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dx, false>(
                                 std::get<side::south - 1>(xi))[0])));
    EXPECT_TRUE(torch::allclose(*(std::get<side::east - 1>(eval)[0]),
                                torch::zeros(7, options)));
    EXPECT_TRUE(torch::allclose(*(std::get<side::west - 1>(eval)[0]),
                                torch::zeros(7, options)));

    eval =
        functionspace.eval<functionspace::boundary, deriv::dx ^ 2, false>(xi);

    EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dx ^ 2, false>(
                                 std::get<side::north - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dx ^ 2, false>(
                                 std::get<side::south - 1>(xi))[0])));
    EXPECT_TRUE(torch::allclose(*(std::get<side::east - 1>(eval)[0]),
                                torch::zeros(7, options)));
    EXPECT_TRUE(torch::allclose(*(std::get<side::west - 1>(eval)[0]),
                                torch::zeros(7, options)));

    eval = functionspace.eval<functionspace::boundary, deriv::dy, false>(xi);
    iganet::verbose(std::cout);
    std::cout << functionspace.boundary() << std::endl;
    EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dy, false>(
                                 std::get<side::north - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dy, false>(
                                 std::get<side::south - 1>(xi))[0])));
    EXPECT_TRUE(torch::allclose(*(std::get<side::east - 1>(eval)[0]),
                                torch::zeros(7, options)));
    std::cout << *(std::get<side::east - 1>(eval)[0]) << std::endl;
    EXPECT_TRUE(torch::allclose(*(std::get<side::west - 1>(eval)[0]),
                                torch::zeros(7, options)));

    eval =
        functionspace.eval<functionspace::boundary, deriv::dy ^ 2, false>(xi);

    EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dy ^ 2, false>(
                                 std::get<side::north - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dy ^ 2, false>(
                                 std::get<side::south - 1>(xi))[0])));
    EXPECT_TRUE(torch::allclose(*(std::get<side::east - 1>(eval)[0]),
                                torch::zeros(7, options)));
    std::cout << *(std::get<side::east - 1>(eval)[0]) << std::endl;
    EXPECT_TRUE(torch::allclose(*(std::get<side::west - 1>(eval)[0]),
                                torch::zeros(7, options)));

    eval = functionspace
               .eval<functionspace::boundary, deriv::dx + deriv::dy, false>(xi);

    EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dx + deriv::dy, false>(
                                 std::get<side::north - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dx + deriv::dy, false>(
                                 std::get<side::south - 1>(xi))[0])));
    EXPECT_TRUE(torch::allclose(*(std::get<side::east - 1>(eval)[0]),
                                torch::zeros(7, options)));
    EXPECT_TRUE(torch::allclose(*(std::get<side::west - 1>(eval)[0]),
                                torch::zeros(7, options)));

    // Evaluation from precomputed coefficients and basis functions
    auto knot_indices =
        functionspace.template find_knot_indices<functionspace::boundary>(xi);
    auto coeff_indices =
        functionspace.template find_coeff_indices<functionspace::boundary>(
            knot_indices);

    auto numel = [](const auto &xi) {
      return std::tuple{std::get<0>(xi)[0].numel(), std::get<1>(xi)[0].numel(),
                        std::get<2>(xi)[0].numel(), std::get<3>(xi)[0].numel()};
    };
    auto sizes = [](const auto &xi) {
      return std::tuple{std::get<0>(xi)[0].sizes(), std::get<1>(xi)[0].sizes(),
                        std::get<2>(xi)[0].sizes(), std::get<3>(xi)[0].sizes()};
    };

    auto basfunc =
        functionspace
            .template eval_basfunc<functionspace::boundary, deriv::func, false>(
                xi, knot_indices);
    eval =
        functionspace.template eval_from_precomputed<functionspace::boundary>(
            basfunc, coeff_indices, numel(xi), sizes(xi));

    EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::func, false>(
                                 std::get<side::north - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::func, false>(
                                 std::get<side::south - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::func, false>(
                                 std::get<side::east - 1>(xi))[0])));
    std::cout << *(std::get<side::east - 1>(eval)[0]) << std::endl;
    EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::func, false>(
                                 std::get<side::west - 1>(xi))[0])));
    std::cout << *(std::get<side::west - 1>(eval)[0]) << std::endl;
    exit(0);
    basfunc =
        functionspace
            .template eval_basfunc<functionspace::boundary, deriv::dx, false>(
                xi, knot_indices);
    eval =
        functionspace.template eval_from_precomputed<functionspace::boundary>(
            basfunc, coeff_indices, numel(xi), sizes(xi));

    EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dx, false>(
                                 std::get<side::north - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dx, false>(
                                 std::get<side::south - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dx, false>(
                                 std::get<side::east - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dx, false>(
                                 std::get<side::west - 1>(xi))[0])));

    basfunc = functionspace.template eval_basfunc<functionspace::boundary,
                                                  deriv::dx ^ 2, false>(
        xi, knot_indices);
    eval =
        functionspace.template eval_from_precomputed<functionspace::boundary>(
            basfunc, coeff_indices, numel(xi), sizes(xi));

    EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dx ^ 2, false>(
                                 std::get<side::north - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dx ^ 2, false>(
                                 std::get<side::south - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dx ^ 2, false>(
                                 std::get<side::east - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dx ^ 2, false>(
                                 std::get<side::west - 1>(xi))[0])));

    basfunc =
        functionspace
            .template eval_basfunc<functionspace::boundary, deriv::dy, false>(
                xi, knot_indices);
    eval =
        functionspace.template eval_from_precomputed<functionspace::boundary>(
            basfunc, coeff_indices, numel(xi), sizes(xi));

    EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dy, false>(
                                 std::get<side::north - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dy, false>(
                                 std::get<side::south - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dy, false>(
                                 std::get<side::east - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dy, false>(
                                 std::get<side::west - 1>(xi))[0])));

    basfunc = functionspace.template eval_basfunc<functionspace::boundary,
                                                  deriv::dy ^ 2, false>(
        xi, knot_indices);
    eval =
        functionspace.template eval_from_precomputed<functionspace::boundary>(
            basfunc, coeff_indices, numel(xi), sizes(xi));

    EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dy ^ 2, false>(
                                 std::get<side::north - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dy ^ 2, false>(
                                 std::get<side::south - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dy ^ 2, false>(
                                 std::get<side::east - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dy ^ 2, false>(
                                 std::get<side::west - 1>(xi))[0])));

    basfunc = functionspace.template eval_basfunc<functionspace::boundary,
                                                  deriv::dx + deriv::dy, false>(
        xi, knot_indices);
    eval =
        functionspace.template eval_from_precomputed<functionspace::boundary>(
            basfunc, coeff_indices, numel(xi), sizes(xi));

    EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dx + deriv::dy, false>(
                                 std::get<side::north - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dx + deriv::dy, false>(
                                 std::get<side::south - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dx + deriv::dy, false>(
                                 std::get<side::east - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dx + deriv::dy, false>(
                                 std::get<side::west - 1>(xi))[0])));

    /// Evaluation of gradient (in parametric domain)
    auto grad = functionspace.template grad<functionspace::boundary, false>(xi);

    for (std::size_t i = 0; i < std::get<0>(grad).entries(); ++i) {
      EXPECT_TRUE(torch::equal(
          *(std::get<side::north - 1>(grad)[i]),
          *(bspline_bdrNS.grad<false>(std::get<side::north - 1>(xi))[i])));
      EXPECT_TRUE(torch::equal(
          *(std::get<side::south - 1>(grad)[i]),
          *(bspline_bdrNS.grad<false>(std::get<side::south - 1>(xi))[i])));
      EXPECT_TRUE(torch::equal(
          *(std::get<side::east - 1>(grad)[i]),
          *(bspline_bdrEW.grad<false>(std::get<side::east - 1>(xi))[i])));
      EXPECT_TRUE(torch::equal(
          *(std::get<side::west - 1>(grad)[i]),
          *(bspline_bdrEW.grad<false>(std::get<side::west - 1>(xi))[i])));
    }

    /// Evaluation of gradient (in physical domain)
    auto igrad = functionspace.template igrad<functionspace::boundary, false>(
        functionspace, xi);

    for (std::size_t i = 0; i < std::get<0>(igrad).entries(); ++i) {
      EXPECT_TRUE(
          torch::equal(*(std::get<side::north - 1>(igrad)[i]),
                       *(bspline_bdrNS.igrad<false>(
                           bspline_bdrNS, std::get<side::north - 1>(xi))[i])));
      EXPECT_TRUE(
          torch::equal(*(std::get<side::south - 1>(igrad)[i]),
                       *(bspline_bdrNS.igrad<false>(
                           bspline_bdrNS, std::get<side::south - 1>(xi))[i])));
      EXPECT_TRUE(
          torch::equal(*(std::get<side::east - 1>(igrad)[i]),
                       *(bspline_bdrEW.igrad<false>(
                           bspline_bdrEW, std::get<side::east - 1>(xi))[i])));
      EXPECT_TRUE(
          torch::equal(*(std::get<side::west - 1>(igrad)[i]),
                       *(bspline_bdrEW.igrad<false>(
                           bspline_bdrEW, std::get<side::west - 1>(xi))[i])));
    }

    /// Evaluation of Jacobian (in parametric domain)
    auto jac = functionspace.template jac<functionspace::boundary, false>(xi);

    for (std::size_t i = 0; i < std::get<0>(jac).entries(); ++i) {
      EXPECT_TRUE(torch::equal(
          *(std::get<side::north - 1>(jac)[i]),
          *(bspline_bdrNS.jac<false>(std::get<side::north - 1>(xi))[i])));
      EXPECT_TRUE(torch::equal(
          *(std::get<side::south - 1>(jac)[i]),
          *(bspline_bdrNS.jac<false>(std::get<side::south - 1>(xi))[i])));
      EXPECT_TRUE(torch::equal(
          *(std::get<side::east - 1>(jac)[i]),
          *(bspline_bdrEW.jac<false>(std::get<side::east - 1>(xi))[i])));
      EXPECT_TRUE(torch::equal(
          *(std::get<side::west - 1>(jac)[i]),
          *(bspline_bdrEW.jac<false>(std::get<side::west - 1>(xi))[i])));
    }

    /// Evaluation of Jacobian (in physical domain)
    auto ijac = functionspace.template ijac<functionspace::boundary, false>(
        functionspace, xi);

    for (std::size_t i = 0; i < std::get<0>(ijac).entries(); ++i) {
      EXPECT_TRUE(
          torch::equal(*(std::get<side::north - 1>(ijac)[i]),
                       *(bspline_bdrNS.ijac<false>(
                           bspline_bdrNS, std::get<side::north - 1>(xi))[i])));
      EXPECT_TRUE(
          torch::equal(*(std::get<side::south - 1>(ijac)[i]),
                       *(bspline_bdrNS.ijac<false>(
                           bspline_bdrNS, std::get<side::south - 1>(xi))[i])));
      EXPECT_TRUE(
          torch::equal(*(std::get<side::east - 1>(ijac)[i]),
                       *(bspline_bdrEW.ijac<false>(
                           bspline_bdrEW, std::get<side::east - 1>(xi))[i])));
      EXPECT_TRUE(
          torch::equal(*(std::get<side::west - 1>(ijac)[i]),
                       *(bspline_bdrEW.ijac<false>(
                           bspline_bdrEW, std::get<side::west - 1>(xi))[i])));
    }

    /// Evaluation of Hessian (in parametric domain)
    auto hess = functionspace.template hess<functionspace::boundary>(xi);

    for (std::size_t i = 0; i < std::get<0>(hess).entries(); ++i) {
      EXPECT_TRUE(torch::equal(
          *(std::get<side::north - 1>(hess)[i]),
          *(bspline_bdrNS.hess<false>(std::get<side::north - 1>(xi))[i])));
      EXPECT_TRUE(torch::equal(
          *(std::get<side::south - 1>(hess)[i]),
          *(bspline_bdrNS.hess<false>(std::get<side::south - 1>(xi))[i])));
      EXPECT_TRUE(torch::equal(
          *(std::get<side::east - 1>(hess)[i]),
          *(bspline_bdrEW.hess<false>(std::get<side::east - 1>(xi))[i])));
      EXPECT_TRUE(torch::equal(
          *(std::get<side::west - 1>(hess)[i]),
          *(bspline_bdrEW.hess<false>(std::get<side::west - 1>(xi))[i])));
    }

    /// Evaluation of Hessian (in physical domain)
    auto ihess = functionspace.template ihess<functionspace::boundary, false>(
        functionspace, xi);

    for (std::size_t i = 0; i < std::get<0>(ihess).entries(); ++i) {
      EXPECT_TRUE(
          torch::equal(*(std::get<side::north - 1>(ihess)[i]),
                       *(bspline_bdrNS.ihess<false>(
                           bspline_bdrNS, std::get<side::north - 1>(xi))[i])));
      EXPECT_TRUE(
          torch::equal(*(std::get<side::south - 1>(ihess)[i]),
                       *(bspline_bdrNS.ihess<false>(
                           bspline_bdrNS, std::get<side::south - 1>(xi))[i])));
      EXPECT_TRUE(
          torch::equal(*(std::get<side::east - 1>(ihess)[i]),
                       *(bspline_bdrEW.ihess<false>(
                           bspline_bdrEW, std::get<side::east - 1>(xi))[i])));
      EXPECT_TRUE(
          torch::equal(*(std::get<side::west - 1>(ihess)[i]),
                       *(bspline_bdrEW.ihess<false>(
                           bspline_bdrEW, std::get<side::west - 1>(xi))[i])));
    }
  }
}

TEST_F(FunctionSpaceTest, S3_geoDim1_degrees234) {
  using iganet::deriv;
  using iganet::functionspace;
  using iganet::side;
  using BSpline = iganet::UniformBSpline<real_t, 1, 2, 3, 4>;
  using Geometry = iganet::UniformBSpline<real_t, 3, 2, 3, 4>;
  iganet::S3<BSpline> functionspace({5, 4, 7}, iganet::init::greville, options);
  iganet::S3<Geometry> S3_geometry({5, 4, 7}, iganet::init::greville, options);
  BSpline bspline({5, 4, 7}, iganet::init::greville, options);
  Geometry geometry({5, 4, 7}, iganet::init::greville, options);

  { // Interior

    auto xi = iganet::utils::to_tensorArray<real_t>(
        {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0} /* u */,
        {1.0, 0.2, 0.1, 0.5, 0.9, 0.75, 0.0} /* v */,
        {0.2, 0.5, 0.75, 0.9, 1.0, 0.0, 0.1} /* w */, options);

    // Evaluation
    EXPECT_TRUE(torch::equal(
        *(functionspace.eval<functionspace::interior, deriv::func, false>(
            xi)[0]),
        *(bspline.eval<deriv::func, false>(xi)[0])));

    EXPECT_TRUE(torch::equal(
        *(functionspace.eval<functionspace::interior, deriv::dx, false>(xi)[0]),
        *(bspline.eval<deriv::dx, false>(xi)[0])));

    EXPECT_TRUE(torch::equal(
        *(functionspace.eval<functionspace::interior, deriv::dx ^ 2, false>(
            xi)[0]),
        *(bspline.eval<deriv::dx ^ 2, false>(xi)[0])));

    EXPECT_TRUE(torch::equal(
        *(functionspace.eval<functionspace::interior, deriv::dy, false>(xi)[0]),
        *(bspline.eval<deriv::dy, false>(xi)[0])));

    EXPECT_TRUE(torch::equal(
        *(functionspace.eval<functionspace::interior, deriv::dy ^ 2, false>(
            xi)[0]),
        *(bspline.eval<deriv::dy ^ 2, false>(xi)[0])));

    EXPECT_TRUE(torch::equal(
        *(functionspace.eval<functionspace::interior, deriv::dz, false>(xi)[0]),
        *(bspline.eval<deriv::dz, false>(xi)[0])));

    EXPECT_TRUE(torch::equal(
        *(functionspace.eval<functionspace::interior, deriv::dz ^ 2, false>(
            xi)[0]),
        *(bspline.eval<deriv::dz ^ 2, false>(xi)[0])));

    EXPECT_TRUE(
        torch::equal(*(functionspace.eval<functionspace::interior,
                                          deriv::dx + deriv::dy, false>(xi)[0]),
                     *(bspline.eval<deriv::dx + deriv::dy, false>(xi)[0])));

    EXPECT_TRUE(
        torch::equal(*(functionspace.eval<functionspace::interior,
                                          deriv::dx + deriv::dz, false>(xi)[0]),
                     *(bspline.eval<deriv::dx + deriv::dz, false>(xi)[0])));

    EXPECT_TRUE(
        torch::equal(*(functionspace.eval<functionspace::interior,
                                          deriv::dy + deriv::dz, false>(xi)[0]),
                     *(bspline.eval<deriv::dy + deriv::dz, false>(xi)[0])));

    // Evaluation from precomputed coefficients and basis functions
    auto knot_indices =
        functionspace.template find_knot_indices<functionspace::interior>(xi);
    auto coeff_indices =
        functionspace.template find_coeff_indices<functionspace::interior>(
            knot_indices);

    auto basfunc =
        functionspace
            .template eval_basfunc<functionspace::interior, deriv::func, false>(
                xi, knot_indices);
    EXPECT_TRUE(torch::equal(
        *(functionspace.eval_from_precomputed<functionspace::interior>(
            basfunc, coeff_indices, xi[0].numel(), xi[0].sizes())[0]),
        *(bspline.eval<deriv::func, false>(xi)[0])));

    basfunc =
        functionspace
            .template eval_basfunc<functionspace::interior, deriv::dx, false>(
                xi, knot_indices);
    EXPECT_TRUE(torch::equal(
        *(functionspace.eval_from_precomputed<functionspace::interior>(
            basfunc, coeff_indices, xi[0].numel(), xi[0].sizes())[0]),
        *(bspline.eval<deriv::dx, false>(xi)[0])));

    basfunc = functionspace.template eval_basfunc<functionspace::interior,
                                                  deriv::dx ^ 2, false>(
        xi, knot_indices);
    EXPECT_TRUE(torch::equal(
        *(functionspace.eval_from_precomputed<functionspace::interior>(
            basfunc, coeff_indices, xi[0].numel(), xi[0].sizes())[0]),
        *(bspline.eval<deriv::dx ^ 2, false>(xi)[0])));

    basfunc =
        functionspace
            .template eval_basfunc<functionspace::interior, deriv::dy, false>(
                xi, knot_indices);
    EXPECT_TRUE(torch::equal(
        *(functionspace.eval_from_precomputed<functionspace::interior>(
            basfunc, coeff_indices, xi[0].numel(), xi[0].sizes())[0]),
        *(bspline.eval<deriv::dy, false>(xi)[0])));

    basfunc = functionspace.template eval_basfunc<functionspace::interior,
                                                  deriv::dy ^ 2, false>(
        xi, knot_indices);
    EXPECT_TRUE(torch::equal(
        *(functionspace.eval_from_precomputed<functionspace::interior>(
            basfunc, coeff_indices, xi[0].numel(), xi[0].sizes())[0]),
        *(bspline.eval<deriv::dy ^ 2, false>(xi)[0])));

    basfunc =
        functionspace
            .template eval_basfunc<functionspace::interior, deriv::dz, false>(
                xi, knot_indices);
    EXPECT_TRUE(torch::equal(
        *(functionspace.eval_from_precomputed<functionspace::interior>(
            basfunc, coeff_indices, xi[0].numel(), xi[0].sizes())[0]),
        *(bspline.eval<deriv::dz, false>(xi)[0])));

    basfunc = functionspace.template eval_basfunc<functionspace::interior,
                                                  deriv::dz ^ 2, false>(
        xi, knot_indices);
    EXPECT_TRUE(torch::equal(
        *(functionspace.eval_from_precomputed<functionspace::interior>(
            basfunc, coeff_indices, xi[0].numel(), xi[0].sizes())[0]),
        *(bspline.eval<deriv::dz ^ 2, false>(xi)[0])));

    basfunc = functionspace.template eval_basfunc<functionspace::interior,
                                                  deriv::dx + deriv::dy, false>(
        xi, knot_indices);
    EXPECT_TRUE(torch::equal(
        *(functionspace.eval_from_precomputed<functionspace::interior>(
            basfunc, coeff_indices, xi[0].numel(), xi[0].sizes())[0]),
        *(bspline.eval<deriv::dx + deriv::dy, false>(xi)[0])));

    basfunc = functionspace.template eval_basfunc<functionspace::interior,
                                                  deriv::dx + deriv::dz, false>(
        xi, knot_indices);
    EXPECT_TRUE(torch::equal(
        *(functionspace.eval_from_precomputed<functionspace::interior>(
            basfunc, coeff_indices, xi[0].numel(), xi[0].sizes())[0]),
        *(bspline.eval<deriv::dx + deriv::dz, false>(xi)[0])));

    basfunc = functionspace.template eval_basfunc<functionspace::interior,
                                                  deriv::dy + deriv::dz, false>(
        xi, knot_indices);
    EXPECT_TRUE(torch::equal(
        *(functionspace.eval_from_precomputed<functionspace::interior>(
            basfunc, coeff_indices, xi[0].numel(), xi[0].sizes())[0]),
        *(bspline.eval<deriv::dy + deriv::dz, false>(xi)[0])));

    /// Evaluation of gradient (in parametric domain)
    auto grad = functionspace.template grad<functionspace::interior, false>(xi);
    auto grad_ref = bspline.grad(xi);

    for (std::size_t i = 0; i < grad.entries(); ++i)
      EXPECT_TRUE(torch::equal(*(grad)[i], *(grad_ref)[i]));

    /// Evaluation of gradient (in physical domain)
    auto igrad = functionspace.template igrad<functionspace::interior, false>(
        S3_geometry, xi);
    auto igrad_ref = bspline.igrad(geometry, xi);

    for (std::size_t i = 0; i < igrad.entries(); ++i)
      EXPECT_TRUE(torch::equal(*(igrad)[i], *(igrad_ref)[i]));

    /// Evaluation of Jacobian (in parametric domain)
    auto jac = functionspace.template jac<functionspace::interior, false>(xi);
    auto jac_ref = bspline.jac(xi);

    for (std::size_t i = 0; i < jac.entries(); ++i)
      EXPECT_TRUE(torch::equal(*(jac)[i], *(jac_ref)[i]));

    /// Evaluation of Jacobian (in physical domain)
    auto ijac = functionspace.template ijac<functionspace::interior, false>(
        S3_geometry, xi);
    auto ijac_ref = bspline.ijac(geometry, xi);

    for (std::size_t i = 0; i < ijac.entries(); ++i)
      EXPECT_TRUE(torch::equal(*(ijac)[i], *(ijac_ref)[i]));

    /// Evaluation of Hessian (in parametric domain)
    auto hess = functionspace.template hess<functionspace::interior, false>(xi);
    auto hess_ref = bspline.hess(xi);

    for (std::size_t i = 0; i < hess.entries(); ++i)
      EXPECT_TRUE(torch::equal(*(hess)[i], *(hess_ref)[i]));

    /// Evaluation of Hessian (in physical domain)
    auto ihess = functionspace.template ihess<functionspace::interior, false>(
        S3_geometry, xi);
    auto ihess_ref = bspline.ihess(geometry, xi);

    for (std::size_t i = 0; i < ihess.entries(); ++i)
      EXPECT_TRUE(torch::equal(*(ihess)[i], *(ihess_ref)[i]));
  }

  { // Boundary

    iganet::UniformBSpline<real_t, 1, 2, 4> bspline_bdrNS(
        {5, 7}, iganet::init::greville, options);
    iganet::UniformBSpline<real_t, 1, 3, 4> bspline_bdrEW(
        {4, 7}, iganet::init::greville, options);
    iganet::UniformBSpline<real_t, 1, 2, 3> bspline_bdrFB(
        {5, 4}, iganet::init::greville, options);
    iganet::UniformBSpline<real_t, 3, 2, 4> geometry_bdrNS(
        {5, 7}, iganet::init::greville, options);
    iganet::UniformBSpline<real_t, 3, 3, 4> geometry_bdrEW(
        {4, 7}, iganet::init::greville, options);
    iganet::UniformBSpline<real_t, 3, 2, 3> geometry_bdrFB(
        {5, 4}, iganet::init::greville, options);

    auto xi = std::tuple{
        iganet::utils::to_tensorArray<real_t>(
            {1.0, 0.2, 0.1, 0.5, 0.9, 0.75, 0.0} /* v */,
            {0.2, 0.5, 0.75, 0.9, 1.0, 0.0, 0.1} /* w */, options) /* west  */,
        iganet::utils::to_tensorArray<real_t>(
            {1.0, 0.2, 0.1, 0.5, 0.9, 0.75, 0.0} /* v */,
            {0.2, 0.5, 0.75, 0.9, 1.0, 0.0, 0.1} /* w */, options) /* east  */,
        iganet::utils::to_tensorArray<real_t>(
            {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0} /* u */,
            {0.2, 0.5, 0.75, 0.9, 1.0, 0.0, 0.1} /* w */, options) /* south */,
        iganet::utils::to_tensorArray<real_t>(
            {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0} /* u */,
            {0.2, 0.5, 0.75, 0.9, 1.0, 0.0, 0.1} /* w */, options) /* north */,
        iganet::utils::to_tensorArray<real_t>(
            {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0} /* u */,
            {1.0, 0.2, 0.1, 0.5, 0.9, 0.75, 0.0} /* v */, options) /* front */,
        iganet::utils::to_tensorArray<real_t>(
            {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0} /* u */,
            {1.0, 0.2, 0.1, 0.5, 0.9, 0.75, 0.0} /* v */, options) /* back  */};

    // Evaluation
    auto eval =
        functionspace.eval<functionspace::boundary, deriv::func, false>(xi);

    EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::func, false>(
                                 std::get<side::north - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::func, false>(
                                 std::get<side::south - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::func, false>(
                                 std::get<side::east - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::func, false>(
                                 std::get<side::west - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::front - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::func, false>(
                                 std::get<side::front - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::back - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::func, false>(
                                 std::get<side::back - 1>(xi))[0])));

    eval = functionspace.eval<functionspace::boundary, deriv::dx, false>(xi);

    EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dx, false>(
                                 std::get<side::north - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dx, false>(
                                 std::get<side::south - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dx, false>(
                                 std::get<side::east - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dx, false>(
                                 std::get<side::west - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::front - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dx, false>(
                                 std::get<side::front - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::back - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dx, false>(
                                 std::get<side::back - 1>(xi))[0])));

    eval =
        functionspace.eval<functionspace::boundary, deriv::dx ^ 2, false>(xi);

    EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dx ^ 2, false>(
                                 std::get<side::north - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dx ^ 2, false>(
                                 std::get<side::south - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dx ^ 2, false>(
                                 std::get<side::east - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dx ^ 2, false>(
                                 std::get<side::west - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::front - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dx ^ 2, false>(
                                 std::get<side::front - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::back - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dx ^ 2, false>(
                                 std::get<side::back - 1>(xi))[0])));

    eval = functionspace.eval<functionspace::boundary, deriv::dy, false>(xi);

    EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dy, false>(
                                 std::get<side::north - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dy, false>(
                                 std::get<side::south - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dy, false>(
                                 std::get<side::east - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dy, false>(
                                 std::get<side::west - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::front - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dy, false>(
                                 std::get<side::front - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::back - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dy, false>(
                                 std::get<side::back - 1>(xi))[0])));

    eval =
        functionspace.eval<functionspace::boundary, deriv::dy ^ 2, false>(xi);

    EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dy ^ 2, false>(
                                 std::get<side::north - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dy ^ 2, false>(
                                 std::get<side::south - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dy ^ 2, false>(
                                 std::get<side::east - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dy ^ 2, false>(
                                 std::get<side::west - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::front - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dy ^ 2, false>(
                                 std::get<side::front - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::back - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dy ^ 2, false>(
                                 std::get<side::back - 1>(xi))[0])));

    eval = functionspace.eval<functionspace::boundary, deriv::dz, false>(xi);

    EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dz, false>(
                                 std::get<side::north - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dz, false>(
                                 std::get<side::south - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dz, false>(
                                 std::get<side::east - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dz, false>(
                                 std::get<side::west - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::front - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dz, false>(
                                 std::get<side::front - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::back - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dz, false>(
                                 std::get<side::back - 1>(xi))[0])));

    eval =
        functionspace.eval<functionspace::boundary, deriv::dz ^ 2, false>(xi);

    EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dz ^ 2, false>(
                                 std::get<side::north - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dz ^ 2, false>(
                                 std::get<side::south - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dz ^ 2, false>(
                                 std::get<side::east - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dz ^ 2, false>(
                                 std::get<side::west - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::front - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dz ^ 2, false>(
                                 std::get<side::front - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::back - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dz ^ 2, false>(
                                 std::get<side::back - 1>(xi))[0])));

    eval = functionspace
               .eval<functionspace::boundary, deriv::dx + deriv::dy, false>(xi);

    EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dx + deriv::dy, false>(
                                 std::get<side::north - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dx + deriv::dy, false>(
                                 std::get<side::south - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dx + deriv::dy, false>(
                                 std::get<side::east - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dx + deriv::dy, false>(
                                 std::get<side::west - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::front - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dx + deriv::dy, false>(
                                 std::get<side::front - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::back - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dx + deriv::dy, false>(
                                 std::get<side::back - 1>(xi))[0])));

    eval = functionspace
               .eval<functionspace::boundary, deriv::dx + deriv::dz, false>(xi);

    EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dx + deriv::dz, false>(
                                 std::get<side::north - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dx + deriv::dz, false>(
                                 std::get<side::south - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dx + deriv::dz, false>(
                                 std::get<side::east - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dx + deriv::dz, false>(
                                 std::get<side::west - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::front - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dx + deriv::dz, false>(
                                 std::get<side::front - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::back - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dx + deriv::dz, false>(
                                 std::get<side::back - 1>(xi))[0])));

    eval = functionspace
               .eval<functionspace::boundary, deriv::dy + deriv::dz, false>(xi);

    EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dy + deriv::dz, false>(
                                 std::get<side::north - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dy + deriv::dz, false>(
                                 std::get<side::south - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dy + deriv::dz, false>(
                                 std::get<side::east - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dy + deriv::dz, false>(
                                 std::get<side::west - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::front - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dy + deriv::dz, false>(
                                 std::get<side::front - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::back - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dy + deriv::dz, false>(
                                 std::get<side::back - 1>(xi))[0])));

    // Evaluation from precomputed coefficients and basis functions
    auto knot_indices =
        functionspace.template find_knot_indices<functionspace::boundary>(xi);
    auto coeff_indices =
        functionspace.template find_coeff_indices<functionspace::boundary>(
            knot_indices);

    auto numel = [](const auto &xi) {
      return std::tuple{std::get<0>(xi)[0].numel(), std::get<1>(xi)[0].numel(),
                        std::get<2>(xi)[0].numel(), std::get<3>(xi)[0].numel(),
                        std::get<4>(xi)[0].numel(), std::get<5>(xi)[0].numel()};
    };
    auto sizes = [](const auto &xi) {
      return std::tuple{std::get<0>(xi)[0].sizes(), std::get<1>(xi)[0].sizes(),
                        std::get<2>(xi)[0].sizes(), std::get<3>(xi)[0].sizes(),
                        std::get<4>(xi)[0].sizes(), std::get<5>(xi)[0].sizes()};
    };

    auto basfunc =
        functionspace
            .template eval_basfunc<functionspace::boundary, deriv::func, false>(
                xi, knot_indices);
    eval =
        functionspace.template eval_from_precomputed<functionspace::boundary>(
            basfunc, coeff_indices, numel(xi), sizes(xi));

    EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::func, false>(
                                 std::get<side::north - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::func, false>(
                                 std::get<side::south - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::func, false>(
                                 std::get<side::east - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::func, false>(
                                 std::get<side::west - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::front - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::func, false>(
                                 std::get<side::front - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::back - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::func, false>(
                                 std::get<side::back - 1>(xi))[0])));

    basfunc =
        functionspace
            .template eval_basfunc<functionspace::boundary, deriv::dx, false>(
                xi, knot_indices);
    eval =
        functionspace.template eval_from_precomputed<functionspace::boundary>(
            basfunc, coeff_indices, numel(xi), sizes(xi));

    EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dx, false>(
                                 std::get<side::north - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dx, false>(
                                 std::get<side::south - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dx, false>(
                                 std::get<side::east - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dx, false>(
                                 std::get<side::west - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::front - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dx, false>(
                                 std::get<side::front - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::back - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dx, false>(
                                 std::get<side::back - 1>(xi))[0])));

    basfunc = functionspace.template eval_basfunc<functionspace::boundary,
                                                  deriv::dx ^ 2, false>(
        xi, knot_indices);
    eval =
        functionspace.template eval_from_precomputed<functionspace::boundary>(
            basfunc, coeff_indices, numel(xi), sizes(xi));

    EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dx ^ 2, false>(
                                 std::get<side::north - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dx ^ 2, false>(
                                 std::get<side::south - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dx ^ 2, false>(
                                 std::get<side::east - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dx ^ 2, false>(
                                 std::get<side::west - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::front - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dx ^ 2, false>(
                                 std::get<side::front - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::back - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dx ^ 2, false>(
                                 std::get<side::back - 1>(xi))[0])));

    basfunc =
        functionspace
            .template eval_basfunc<functionspace::boundary, deriv::dy, false>(
                xi, knot_indices);
    eval =
        functionspace.template eval_from_precomputed<functionspace::boundary>(
            basfunc, coeff_indices, numel(xi), sizes(xi));

    EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dy, false>(
                                 std::get<side::north - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dy, false>(
                                 std::get<side::south - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dy, false>(
                                 std::get<side::east - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dy, false>(
                                 std::get<side::west - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::front - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dy, false>(
                                 std::get<side::front - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::back - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dy, false>(
                                 std::get<side::back - 1>(xi))[0])));

    basfunc = functionspace.template eval_basfunc<functionspace::boundary,
                                                  deriv::dy ^ 2, false>(
        xi, knot_indices);
    eval =
        functionspace.template eval_from_precomputed<functionspace::boundary>(
            basfunc, coeff_indices, numel(xi), sizes(xi));

    EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dy ^ 2, false>(
                                 std::get<side::north - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dy ^ 2, false>(
                                 std::get<side::south - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dy ^ 2, false>(
                                 std::get<side::east - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dy ^ 2, false>(
                                 std::get<side::west - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::front - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dy ^ 2, false>(
                                 std::get<side::front - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::back - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dy ^ 2, false>(
                                 std::get<side::back - 1>(xi))[0])));

    basfunc =
        functionspace
            .template eval_basfunc<functionspace::boundary, deriv::dz, false>(
                xi, knot_indices);
    eval =
        functionspace.template eval_from_precomputed<functionspace::boundary>(
            basfunc, coeff_indices, numel(xi), sizes(xi));

    EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dz, false>(
                                 std::get<side::north - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dz, false>(
                                 std::get<side::south - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dz, false>(
                                 std::get<side::east - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dz, false>(
                                 std::get<side::west - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::front - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dz, false>(
                                 std::get<side::front - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::back - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dz, false>(
                                 std::get<side::back - 1>(xi))[0])));

    basfunc = functionspace.template eval_basfunc<functionspace::boundary,
                                                  deriv::dz ^ 2, false>(
        xi, knot_indices);
    eval =
        functionspace.template eval_from_precomputed<functionspace::boundary>(
            basfunc, coeff_indices, numel(xi), sizes(xi));

    EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dz ^ 2, false>(
                                 std::get<side::north - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dz ^ 2, false>(
                                 std::get<side::south - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dz ^ 2, false>(
                                 std::get<side::east - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dz ^ 2, false>(
                                 std::get<side::west - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::front - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dz ^ 2, false>(
                                 std::get<side::front - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::back - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dz ^ 2, false>(
                                 std::get<side::back - 1>(xi))[0])));

    basfunc = functionspace.template eval_basfunc<functionspace::boundary,
                                                  deriv::dx + deriv::dy, false>(
        xi, knot_indices);
    eval =
        functionspace.template eval_from_precomputed<functionspace::boundary>(
            basfunc, coeff_indices, numel(xi), sizes(xi));

    EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dx + deriv::dy, false>(
                                 std::get<side::north - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dx + deriv::dy, false>(
                                 std::get<side::south - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dx + deriv::dy, false>(
                                 std::get<side::east - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dx + deriv::dy, false>(
                                 std::get<side::west - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::front - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dx + deriv::dy, false>(
                                 std::get<side::front - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::back - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dx + deriv::dy, false>(
                                 std::get<side::back - 1>(xi))[0])));

    basfunc = functionspace.template eval_basfunc<functionspace::boundary,
                                                  deriv::dx + deriv::dz, false>(
        xi, knot_indices);
    eval =
        functionspace.template eval_from_precomputed<functionspace::boundary>(
            basfunc, coeff_indices, numel(xi), sizes(xi));

    EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dx + deriv::dz, false>(
                                 std::get<side::north - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dx + deriv::dz, false>(
                                 std::get<side::south - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dx + deriv::dz, false>(
                                 std::get<side::east - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dx + deriv::dz, false>(
                                 std::get<side::west - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::front - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dx + deriv::dz, false>(
                                 std::get<side::front - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::back - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dx + deriv::dz, false>(
                                 std::get<side::back - 1>(xi))[0])));

    basfunc = functionspace.template eval_basfunc<functionspace::boundary,
                                                  deriv::dy + deriv::dz, false>(
        xi, knot_indices);
    eval =
        functionspace.template eval_from_precomputed<functionspace::boundary>(
            basfunc, coeff_indices, numel(xi), sizes(xi));

    EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dy + deriv::dz, false>(
                                 std::get<side::north - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dy + deriv::dz, false>(
                                 std::get<side::south - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dy + deriv::dz, false>(
                                 std::get<side::east - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dy + deriv::dz, false>(
                                 std::get<side::west - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::front - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dy + deriv::dz, false>(
                                 std::get<side::front - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::back - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dy + deriv::dz, false>(
                                 std::get<side::back - 1>(xi))[0])));

    /// Evaluation of gradient (in parametric domain)
    auto grad = functionspace.template grad<functionspace::boundary, false>(xi);

    for (std::size_t i = 0; i < std::get<0>(grad).entries(); ++i) {
      EXPECT_TRUE(torch::equal(
          *(std::get<side::north - 1>(grad)[i]),
          *(bspline_bdrNS.grad<false>(std::get<side::north - 1>(xi))[i])));
      EXPECT_TRUE(torch::equal(
          *(std::get<side::south - 1>(grad)[i]),
          *(bspline_bdrNS.grad<false>(std::get<side::south - 1>(xi))[i])));
      EXPECT_TRUE(torch::equal(
          *(std::get<side::east - 1>(grad)[i]),
          *(bspline_bdrEW.grad<false>(std::get<side::east - 1>(xi))[i])));
      EXPECT_TRUE(torch::equal(
          *(std::get<side::west - 1>(grad)[i]),
          *(bspline_bdrEW.grad<false>(std::get<side::west - 1>(xi))[i])));
      EXPECT_TRUE(torch::equal(
          *(std::get<side::front - 1>(grad)[i]),
          *(bspline_bdrFB.grad<false>(std::get<side::front - 1>(xi))[i])));
      EXPECT_TRUE(torch::equal(
          *(std::get<side::back - 1>(grad)[i]),
          *(bspline_bdrFB.grad<false>(std::get<side::back - 1>(xi))[i])));
    }

    /// Evaluation of gradient (in physical domain)
    auto igrad = functionspace.template igrad<functionspace::boundary, false>(
        S3_geometry, xi);

    for (std::size_t i = 0; i < std::get<0>(igrad).entries(); ++i) {
      EXPECT_TRUE(
          torch::equal(*(std::get<side::north - 1>(igrad)[i]),
                       *(bspline_bdrNS.igrad<false>(
                           geometry_bdrNS, std::get<side::north - 1>(xi))[i])));
      EXPECT_TRUE(
          torch::equal(*(std::get<side::south - 1>(igrad)[i]),
                       *(bspline_bdrNS.igrad<false>(
                           geometry_bdrNS, std::get<side::south - 1>(xi))[i])));
      EXPECT_TRUE(
          torch::equal(*(std::get<side::east - 1>(igrad)[i]),
                       *(bspline_bdrEW.igrad<false>(
                           geometry_bdrEW, std::get<side::east - 1>(xi))[i])));
      EXPECT_TRUE(
          torch::equal(*(std::get<side::west - 1>(igrad)[i]),
                       *(bspline_bdrEW.igrad<false>(
                           geometry_bdrEW, std::get<side::west - 1>(xi))[i])));
      EXPECT_TRUE(
          torch::equal(*(std::get<side::front - 1>(igrad)[i]),
                       *(bspline_bdrFB.igrad<false>(
                           geometry_bdrFB, std::get<side::front - 1>(xi))[i])));
      EXPECT_TRUE(
          torch::equal(*(std::get<side::back - 1>(igrad)[i]),
                       *(bspline_bdrFB.igrad<false>(
                           geometry_bdrFB, std::get<side::back - 1>(xi))[i])));
    }

    /// Evaluation of Jacobian (in parametric domain)
    auto jac = functionspace.template jac<functionspace::boundary, false>(xi);

    for (std::size_t i = 0; i < std::get<0>(jac).entries(); ++i) {
      EXPECT_TRUE(torch::equal(
          *(std::get<side::north - 1>(jac)[i]),
          *(bspline_bdrNS.jac<false>(std::get<side::north - 1>(xi))[i])));
      EXPECT_TRUE(torch::equal(
          *(std::get<side::south - 1>(jac)[i]),
          *(bspline_bdrNS.jac<false>(std::get<side::south - 1>(xi))[i])));
      EXPECT_TRUE(torch::equal(
          *(std::get<side::east - 1>(jac)[i]),
          *(bspline_bdrEW.jac<false>(std::get<side::east - 1>(xi))[i])));
      EXPECT_TRUE(torch::equal(
          *(std::get<side::west - 1>(jac)[i]),
          *(bspline_bdrEW.jac<false>(std::get<side::west - 1>(xi))[i])));
      EXPECT_TRUE(torch::equal(
          *(std::get<side::front - 1>(jac)[i]),
          *(bspline_bdrFB.jac<false>(std::get<side::front - 1>(xi))[i])));
      EXPECT_TRUE(torch::equal(
          *(std::get<side::back - 1>(jac)[i]),
          *(bspline_bdrFB.jac<false>(std::get<side::back - 1>(xi))[i])));
    }

    /// Evaluation of Jacobian (in physical domain)
    auto ijac = functionspace.template ijac<functionspace::boundary, false>(
        S3_geometry, xi);

    for (std::size_t i = 0; i < std::get<0>(ijac).entries(); ++i) {
      EXPECT_TRUE(
          torch::equal(*(std::get<side::north - 1>(ijac)[i]),
                       *(bspline_bdrNS.ijac<false>(
                           geometry_bdrNS, std::get<side::north - 1>(xi))[i])));
      EXPECT_TRUE(
          torch::equal(*(std::get<side::south - 1>(ijac)[i]),
                       *(bspline_bdrNS.ijac<false>(
                           geometry_bdrNS, std::get<side::south - 1>(xi))[i])));
      EXPECT_TRUE(
          torch::equal(*(std::get<side::east - 1>(ijac)[i]),
                       *(bspline_bdrEW.ijac<false>(
                           geometry_bdrEW, std::get<side::east - 1>(xi))[i])));
      EXPECT_TRUE(
          torch::equal(*(std::get<side::west - 1>(ijac)[i]),
                       *(bspline_bdrEW.ijac<false>(
                           geometry_bdrEW, std::get<side::west - 1>(xi))[i])));
      EXPECT_TRUE(
          torch::equal(*(std::get<side::front - 1>(ijac)[i]),
                       *(bspline_bdrFB.ijac<false>(
                           geometry_bdrFB, std::get<side::front - 1>(xi))[i])));
      EXPECT_TRUE(
          torch::equal(*(std::get<side::back - 1>(ijac)[i]),
                       *(bspline_bdrFB.ijac<false>(
                           geometry_bdrFB, std::get<side::back - 1>(xi))[i])));
    }

    /// Evaluation of Hessian (in parametric domain)
    auto hess = functionspace.template hess<functionspace::boundary>(xi);

    for (std::size_t i = 0; i < std::get<0>(hess).entries(); ++i) {
      EXPECT_TRUE(torch::equal(
          *(std::get<side::north - 1>(hess)[i]),
          *(bspline_bdrNS.hess<false>(std::get<side::north - 1>(xi))[i])));
      EXPECT_TRUE(torch::equal(
          *(std::get<side::south - 1>(hess)[i]),
          *(bspline_bdrNS.hess<false>(std::get<side::south - 1>(xi))[i])));
      EXPECT_TRUE(torch::equal(
          *(std::get<side::east - 1>(hess)[i]),
          *(bspline_bdrEW.hess<false>(std::get<side::east - 1>(xi))[i])));
      EXPECT_TRUE(torch::equal(
          *(std::get<side::west - 1>(hess)[i]),
          *(bspline_bdrEW.hess<false>(std::get<side::west - 1>(xi))[i])));
      EXPECT_TRUE(torch::equal(
          *(std::get<side::front - 1>(hess)[i]),
          *(bspline_bdrFB.hess<false>(std::get<side::front - 1>(xi))[i])));
      EXPECT_TRUE(torch::equal(
          *(std::get<side::back - 1>(hess)[i]),
          *(bspline_bdrFB.hess<false>(std::get<side::back - 1>(xi))[i])));
    }

    /// Evaluation of Hessian (in physical domain)
    auto ihess =
        functionspace.template ihess<functionspace::boundary>(S3_geometry, xi);

    for (std::size_t i = 0; i < std::get<0>(ihess).entries(); ++i) {
      EXPECT_TRUE(
          torch::equal(*(std::get<side::north - 1>(ihess)[i]),
                       *(bspline_bdrNS.ihess<false>(
                           geometry_bdrNS, std::get<side::north - 1>(xi))[i])));
      EXPECT_TRUE(
          torch::equal(*(std::get<side::south - 1>(ihess)[i]),
                       *(bspline_bdrNS.ihess<false>(
                           geometry_bdrNS, std::get<side::south - 1>(xi))[i])));
      EXPECT_TRUE(
          torch::equal(*(std::get<side::east - 1>(ihess)[i]),
                       *(bspline_bdrEW.ihess<false>(
                           geometry_bdrEW, std::get<side::east - 1>(xi))[i])));
      EXPECT_TRUE(
          torch::equal(*(std::get<side::west - 1>(ihess)[i]),
                       *(bspline_bdrEW.ihess<false>(
                           geometry_bdrEW, std::get<side::west - 1>(xi))[i])));
      EXPECT_TRUE(
          torch::equal(*(std::get<side::front - 1>(ihess)[i]),
                       *(bspline_bdrFB.ihess<false>(
                           geometry_bdrFB, std::get<side::front - 1>(xi))[i])));
      EXPECT_TRUE(
          torch::equal(*(std::get<side::back - 1>(ihess)[i]),
                       *(bspline_bdrFB.ihess<false>(
                           geometry_bdrFB, std::get<side::back - 1>(xi))[i])));
    }
  }
}

TEST_F(FunctionSpaceTest, S4_geoDim1_degrees2341) {
  using iganet::deriv;
  using iganet::functionspace;
  using iganet::side;
  using BSpline = iganet::UniformBSpline<real_t, 1, 2, 3, 4, 1>;
  using Geometry = iganet::UniformBSpline<real_t, 4, 2, 3, 4, 1>;
  iganet::S4<BSpline> functionspace({5, 4, 7, 3}, iganet::init::greville,
                                    options);
  iganet::S4<Geometry> S4_geometry({5, 4, 7, 3}, iganet::init::greville,
                                   options);
  BSpline bspline({5, 4, 7, 3}, iganet::init::greville, options);
  Geometry geometry({5, 4, 7, 3}, iganet::init::greville, options);

  { // Interior

    auto xi = iganet::utils::to_tensorArray<real_t>(
        {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0} /* u */,
        {1.0, 0.2, 0.1, 0.5, 0.9, 0.75, 0.0} /* v */,
        {0.2, 0.5, 0.75, 0.9, 1.0, 0.0, 0.1} /* w */,
        {0.1, 0.1, 0.2, 0.3, 0.3, 0.0, 0.1} /* t */, options);

    // Evaluation
    EXPECT_TRUE(torch::equal(
        *(functionspace.eval<functionspace::interior, deriv::func, false>(
            xi)[0]),
        *(bspline.eval<deriv::func, false>(xi)[0])));

    EXPECT_TRUE(torch::equal(
        *(functionspace.eval<functionspace::interior, deriv::dx, false>(xi)[0]),
        *(bspline.eval<deriv::dx, false>(xi)[0])));

    EXPECT_TRUE(torch::equal(
        *(functionspace.eval<functionspace::interior, deriv::dx ^ 2, false>(
            xi)[0]),
        *(bspline.eval<deriv::dx ^ 2, false>(xi)[0])));

    EXPECT_TRUE(torch::equal(
        *(functionspace.eval<functionspace::interior, deriv::dy, false>(xi)[0]),
        *(bspline.eval<deriv::dy, false>(xi)[0])));

    EXPECT_TRUE(torch::equal(
        *(functionspace.eval<functionspace::interior, deriv::dy ^ 2, false>(
            xi)[0]),
        *(bspline.eval<deriv::dy ^ 2, false>(xi)[0])));

    EXPECT_TRUE(torch::equal(
        *(functionspace.eval<functionspace::interior, deriv::dz, false>(xi)[0]),
        *(bspline.eval<deriv::dz, false>(xi)[0])));

    EXPECT_TRUE(torch::equal(
        *(functionspace.eval<functionspace::interior, deriv::dz ^ 2, false>(
            xi)[0]),
        *(bspline.eval<deriv::dz ^ 2, false>(xi)[0])));

    EXPECT_TRUE(torch::equal(
        *(functionspace.eval<functionspace::interior, deriv::dt, false>(xi)[0]),
        *(bspline.eval<deriv::dt, false>(xi)[0])));

    EXPECT_TRUE(torch::equal(
        *(functionspace.eval<functionspace::interior, deriv::dt ^ 2, false>(
            xi)[0]),
        *(bspline.eval<deriv::dt ^ 2, false>(xi)[0])));

    EXPECT_TRUE(
        torch::equal(*(functionspace.eval<functionspace::interior,
                                          deriv::dx + deriv::dy, false>(xi)[0]),
                     *(bspline.eval<deriv::dx + deriv::dy, false>(xi)[0])));

    EXPECT_TRUE(
        torch::equal(*(functionspace.eval<functionspace::interior,
                                          deriv::dx + deriv::dz, false>(xi)[0]),
                     *(bspline.eval<deriv::dx + deriv::dz, false>(xi)[0])));

    EXPECT_TRUE(
        torch::equal(*(functionspace.eval<functionspace::interior,
                                          deriv::dx + deriv::dt, false>(xi)[0]),
                     *(bspline.eval<deriv::dx + deriv::dt, false>(xi)[0])));

    EXPECT_TRUE(
        torch::equal(*(functionspace.eval<functionspace::interior,
                                          deriv::dy + deriv::dz, false>(xi)[0]),
                     *(bspline.eval<deriv::dy + deriv::dz, false>(xi)[0])));

    EXPECT_TRUE(
        torch::equal(*(functionspace.eval<functionspace::interior,
                                          deriv::dy + deriv::dt, false>(xi)[0]),
                     *(bspline.eval<deriv::dy + deriv::dt, false>(xi)[0])));

    EXPECT_TRUE(
        torch::equal(*(functionspace.eval<functionspace::interior,
                                          deriv::dz + deriv::dt, false>(xi)[0]),
                     *(bspline.eval<deriv::dz + deriv::dt, false>(xi)[0])));

    // Evaluation from precomputed coefficients and basis functions
    auto knot_indices =
        functionspace.template find_knot_indices<functionspace::interior>(xi);
    auto coeff_indices =
        functionspace.template find_coeff_indices<functionspace::interior>(
            knot_indices);

    auto basfunc =
        functionspace
            .template eval_basfunc<functionspace::interior, deriv::func, false>(
                xi, knot_indices);
    EXPECT_TRUE(torch::equal(
        *(functionspace.eval_from_precomputed<functionspace::interior>(
            basfunc, coeff_indices, xi[0].numel(), xi[0].sizes())[0]),
        *(bspline.eval<deriv::func, false>(xi)[0])));

    basfunc =
        functionspace
            .template eval_basfunc<functionspace::interior, deriv::dx, false>(
                xi, knot_indices);
    EXPECT_TRUE(torch::equal(
        *(functionspace.eval_from_precomputed<functionspace::interior>(
            basfunc, coeff_indices, xi[0].numel(), xi[0].sizes())[0]),
        *(bspline.eval<deriv::dx, false>(xi)[0])));

    basfunc = functionspace.template eval_basfunc<functionspace::interior,
                                                  deriv::dx ^ 2, false>(
        xi, knot_indices);
    EXPECT_TRUE(torch::equal(
        *(functionspace.eval_from_precomputed<functionspace::interior>(
            basfunc, coeff_indices, xi[0].numel(), xi[0].sizes())[0]),
        *(bspline.eval<deriv::dx ^ 2, false>(xi)[0])));

    basfunc =
        functionspace
            .template eval_basfunc<functionspace::interior, deriv::dy, false>(
                xi, knot_indices);
    EXPECT_TRUE(torch::equal(
        *(functionspace.eval_from_precomputed<functionspace::interior>(
            basfunc, coeff_indices, xi[0].numel(), xi[0].sizes())[0]),
        *(bspline.eval<deriv::dy, false>(xi)[0])));

    basfunc = functionspace.template eval_basfunc<functionspace::interior,
                                                  deriv::dy ^ 2, false>(
        xi, knot_indices);
    EXPECT_TRUE(torch::equal(
        *(functionspace.eval_from_precomputed<functionspace::interior>(
            basfunc, coeff_indices, xi[0].numel(), xi[0].sizes())[0]),
        *(bspline.eval<deriv::dy ^ 2, false>(xi)[0])));

    basfunc =
        functionspace
            .template eval_basfunc<functionspace::interior, deriv::dz, false>(
                xi, knot_indices);
    EXPECT_TRUE(torch::equal(
        *(functionspace.eval_from_precomputed<functionspace::interior>(
            basfunc, coeff_indices, xi[0].numel(), xi[0].sizes())[0]),
        *(bspline.eval<deriv::dz, false>(xi)[0])));

    basfunc = functionspace.template eval_basfunc<functionspace::interior,
                                                  deriv::dz ^ 2, false>(
        xi, knot_indices);
    EXPECT_TRUE(torch::equal(
        *(functionspace.eval_from_precomputed<functionspace::interior>(
            basfunc, coeff_indices, xi[0].numel(), xi[0].sizes())[0]),
        *(bspline.eval<deriv::dz ^ 2, false>(xi)[0])));

    basfunc =
        functionspace
            .template eval_basfunc<functionspace::interior, deriv::dt, false>(
                xi, knot_indices);
    EXPECT_TRUE(torch::equal(
        *(functionspace.eval_from_precomputed<functionspace::interior>(
            basfunc, coeff_indices, xi[0].numel(), xi[0].sizes())[0]),
        *(bspline.eval<deriv::dt, false>(xi)[0])));

    basfunc = functionspace.template eval_basfunc<functionspace::interior,
                                                  deriv::dt ^ 2, false>(
        xi, knot_indices);
    EXPECT_TRUE(torch::equal(
        *(functionspace.eval_from_precomputed<functionspace::interior>(
            basfunc, coeff_indices, xi[0].numel(), xi[0].sizes())[0]),
        *(bspline.eval<deriv::dt ^ 2, false>(xi)[0])));

    basfunc = functionspace.template eval_basfunc<functionspace::interior,
                                                  deriv::dx + deriv::dy, false>(
        xi, knot_indices);
    EXPECT_TRUE(torch::equal(
        *(functionspace.eval_from_precomputed<functionspace::interior>(
            basfunc, coeff_indices, xi[0].numel(), xi[0].sizes())[0]),
        *(bspline.eval<deriv::dx + deriv::dy, false>(xi)[0])));

    basfunc = functionspace.template eval_basfunc<functionspace::interior,
                                                  deriv::dx + deriv::dz, false>(
        xi, knot_indices);
    EXPECT_TRUE(torch::equal(
        *(functionspace.eval_from_precomputed<functionspace::interior>(
            basfunc, coeff_indices, xi[0].numel(), xi[0].sizes())[0]),
        *(bspline.eval<deriv::dx + deriv::dz, false>(xi)[0])));

    basfunc = functionspace.template eval_basfunc<functionspace::interior,
                                                  deriv::dx + deriv::dt, false>(
        xi, knot_indices);
    EXPECT_TRUE(torch::equal(
        *(functionspace.eval_from_precomputed<functionspace::interior>(
            basfunc, coeff_indices, xi[0].numel(), xi[0].sizes())[0]),
        *(bspline.eval<deriv::dx + deriv::dt, false>(xi)[0])));

    basfunc = functionspace.template eval_basfunc<functionspace::interior,
                                                  deriv::dy + deriv::dz, false>(
        xi, knot_indices);
    EXPECT_TRUE(torch::equal(
        *(functionspace.eval_from_precomputed<functionspace::interior>(
            basfunc, coeff_indices, xi[0].numel(), xi[0].sizes())[0]),
        *(bspline.eval<deriv::dy + deriv::dz, false>(xi)[0])));

    basfunc = functionspace.template eval_basfunc<functionspace::interior,
                                                  deriv::dy + deriv::dt, false>(
        xi, knot_indices);
    EXPECT_TRUE(torch::equal(
        *(functionspace.eval_from_precomputed<functionspace::interior>(
            basfunc, coeff_indices, xi[0].numel(), xi[0].sizes())[0]),
        *(bspline.eval<deriv::dy + deriv::dt, false>(xi)[0])));

    basfunc = functionspace.template eval_basfunc<functionspace::interior,
                                                  deriv::dz + deriv::dt, false>(
        xi, knot_indices);
    EXPECT_TRUE(torch::equal(
        *(functionspace.eval_from_precomputed<functionspace::interior>(
            basfunc, coeff_indices, xi[0].numel(), xi[0].sizes())[0]),
        *(bspline.eval<deriv::dz + deriv::dt, false>(xi)[0])));

    /// Evaluation of gradient (in parametric domain)
    auto grad = functionspace.template grad<functionspace::interior, false>(xi);
    auto grad_ref = bspline.grad(xi);

    for (std::size_t i = 0; i < grad.entries(); ++i)
      EXPECT_TRUE(torch::equal(*(grad)[i], *(grad_ref)[i]));

    /// Evaluation of gradient (in physical domain)
    auto igrad = functionspace.template igrad<functionspace::interior, false>(
        S4_geometry, xi);
    auto igrad_ref = bspline.igrad(geometry, xi);

    for (std::size_t i = 0; i < igrad.entries(); ++i)
      EXPECT_TRUE(torch::equal(*(igrad)[i], *(igrad_ref)[i]));

    /// Evaluation of Jacobian (in parametric domain)
    auto jac = functionspace.template jac<functionspace::interior, false>(xi);
    auto jac_ref = bspline.jac(xi);

    for (std::size_t i = 0; i < jac.entries(); ++i)
      EXPECT_TRUE(torch::equal(*(jac)[i], *(jac_ref)[i]));

    /// Evaluation of Jacobian (in physical domain)
    auto ijac = functionspace.template ijac<functionspace::interior, false>(
        S4_geometry, xi);
    auto ijac_ref = bspline.ijac(geometry, xi);

    for (std::size_t i = 0; i < ijac.entries(); ++i)
      EXPECT_TRUE(torch::equal(*(ijac)[i], *(ijac_ref)[i]));

    /// Evaluation of Hessian (in parametric domain)
    auto hess = functionspace.template hess<functionspace::interior, false>(xi);
    auto hess_ref = bspline.hess(xi);

    for (std::size_t i = 0; i < hess.entries(); ++i)
      EXPECT_TRUE(torch::equal(*(hess)[i], *(hess_ref)[i]));

    /// Evaluation of Hessian (in physical domain)
    auto ihess = functionspace.template ihess<functionspace::interior, false>(
        S4_geometry, xi);
    auto ihess_ref = bspline.ihess(geometry, xi);

    for (std::size_t i = 0; i < ihess.entries(); ++i)
      EXPECT_TRUE(torch::equal(*(ihess)[i], *(ihess_ref)[i]));
  }

  { // Boundary

    iganet::UniformBSpline<real_t, 1, 2, 4, 1> bspline_bdrNS(
        {5, 7, 3}, iganet::init::greville, options);
    iganet::UniformBSpline<real_t, 1, 3, 4, 1> bspline_bdrEW(
        {4, 7, 3}, iganet::init::greville, options);
    iganet::UniformBSpline<real_t, 1, 2, 3, 1> bspline_bdrFB(
        {5, 4, 3}, iganet::init::greville, options);
    iganet::UniformBSpline<real_t, 1, 2, 3, 4> bspline_bdrSE(
        {5, 4, 7}, iganet::init::greville, options);
    iganet::UniformBSpline<real_t, 4, 2, 4, 1> geometry_bdrNS(
        {5, 7, 3}, iganet::init::greville, options);
    iganet::UniformBSpline<real_t, 4, 3, 4, 1> geometry_bdrEW(
        {4, 7, 3}, iganet::init::greville, options);
    iganet::UniformBSpline<real_t, 4, 2, 3, 1> geometry_bdrFB(
        {5, 4, 3}, iganet::init::greville, options);
    iganet::UniformBSpline<real_t, 4, 2, 3, 4> geometry_bdrSE(
        {5, 4, 7}, iganet::init::greville, options);

    auto xi = std::tuple{
        iganet::utils::to_tensorArray<real_t>(
            {1.0, 0.2, 0.1, 0.5, 0.9, 0.75, 0.0} /* v */,
            {0.2, 0.5, 0.75, 0.9, 1.0, 0.0, 0.1} /* w */,
            {0.1, 0.1, 0.2, 0.3, 0.3, 0.0, 0.1} /* t */, options) /* west  */,
        iganet::utils::to_tensorArray<real_t>(
            {1.0, 0.2, 0.1, 0.5, 0.9, 0.75, 0.0} /* v */,
            {0.2, 0.5, 0.75, 0.9, 1.0, 0.0, 0.1} /* w */,
            {0.1, 0.1, 0.2, 0.3, 0.3, 0.0, 0.1} /* t */, options) /* east  */,
        iganet::utils::to_tensorArray<real_t>(
            {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0} /* u */,
            {0.2, 0.5, 0.75, 0.9, 1.0, 0.0, 0.1} /* w */,
            {0.1, 0.1, 0.2, 0.3, 0.3, 0.0, 0.1} /* t */, options) /* south */,
        iganet::utils::to_tensorArray<real_t>(
            {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0} /* u */,
            {0.2, 0.5, 0.75, 0.9, 1.0, 0.0, 0.1} /* w */,
            {0.1, 0.1, 0.2, 0.3, 0.3, 0.0, 0.1} /* t */, options) /* north */,
        iganet::utils::to_tensorArray<real_t>(
            {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0} /* u */,
            {1.0, 0.2, 0.1, 0.5, 0.9, 0.75, 0.0} /* v */,
            {0.1, 0.1, 0.2, 0.3, 0.3, 0.0, 0.1} /* t */, options) /* front */,
        iganet::utils::to_tensorArray<real_t>(
            {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0} /* u */,
            {1.0, 0.2, 0.1, 0.5, 0.9, 0.75, 0.0} /* v */,
            {0.1, 0.1, 0.2, 0.3, 0.3, 0.0, 0.1} /* t */, options) /* back  */,
        iganet::utils::to_tensorArray<real_t>(
            {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0} /* u */,
            {1.0, 0.2, 0.1, 0.5, 0.9, 0.75, 0.0} /* v */,
            {0.2, 0.5, 0.75, 0.9, 1.0, 0.0, 0.1} /* w */, options) /* stime */,
        iganet::utils::to_tensorArray<real_t>(
            {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0} /* u */,
            {1.0, 0.2, 0.1, 0.5, 0.9, 0.75, 0.0} /* v */,
            {0.2, 0.5, 0.75, 0.9, 1.0, 0.0, 0.1} /* w */, options) /* etime */};

    // Evaluation
    auto eval =
        functionspace.eval<functionspace::boundary, deriv::func, false>(xi);

    EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::func, false>(
                                 std::get<side::north - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::func, false>(
                                 std::get<side::south - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::func, false>(
                                 std::get<side::east - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::func, false>(
                                 std::get<side::west - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::front - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::func, false>(
                                 std::get<side::front - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::back - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::func, false>(
                                 std::get<side::back - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::stime - 1>(eval)[0]),
                             *(bspline_bdrSE.eval<deriv::func, false>(
                                 std::get<side::stime - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::etime - 1>(eval)[0]),
                             *(bspline_bdrSE.eval<deriv::func, false>(
                                 std::get<side::etime - 1>(xi))[0])));

    eval = functionspace.eval<functionspace::boundary, deriv::dx, false>(xi);

    EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dx, false>(
                                 std::get<side::north - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dx, false>(
                                 std::get<side::south - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dx, false>(
                                 std::get<side::east - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dx, false>(
                                 std::get<side::west - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::front - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dx, false>(
                                 std::get<side::front - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::back - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dx, false>(
                                 std::get<side::back - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::stime - 1>(eval)[0]),
                             *(bspline_bdrSE.eval<deriv::dx, false>(
                                 std::get<side::stime - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::etime - 1>(eval)[0]),
                             *(bspline_bdrSE.eval<deriv::dx, false>(
                                 std::get<side::etime - 1>(xi))[0])));

    eval =
        functionspace.eval<functionspace::boundary, deriv::dx ^ 2, false>(xi);

    EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dx ^ 2, false>(
                                 std::get<side::north - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dx ^ 2, false>(
                                 std::get<side::south - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dx ^ 2, false>(
                                 std::get<side::east - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dx ^ 2, false>(
                                 std::get<side::west - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::front - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dx ^ 2, false>(
                                 std::get<side::front - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::back - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dx ^ 2, false>(
                                 std::get<side::back - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::stime - 1>(eval)[0]),
                             *(bspline_bdrSE.eval<deriv::dx ^ 2, false>(
                                 std::get<side::stime - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::etime - 1>(eval)[0]),
                             *(bspline_bdrSE.eval<deriv::dx ^ 2, false>(
                                 std::get<side::etime - 1>(xi))[0])));

    eval = functionspace.eval<functionspace::boundary, deriv::dy, false>(xi);

    EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dy, false>(
                                 std::get<side::north - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dy, false>(
                                 std::get<side::south - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dy, false>(
                                 std::get<side::east - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dy, false>(
                                 std::get<side::west - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::front - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dy, false>(
                                 std::get<side::front - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::back - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dy, false>(
                                 std::get<side::back - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::stime - 1>(eval)[0]),
                             *(bspline_bdrSE.eval<deriv::dy, false>(
                                 std::get<side::stime - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::etime - 1>(eval)[0]),
                             *(bspline_bdrSE.eval<deriv::dy, false>(
                                 std::get<side::etime - 1>(xi))[0])));

    eval =
        functionspace.eval<functionspace::boundary, deriv::dy ^ 2, false>(xi);

    EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dy ^ 2, false>(
                                 std::get<side::north - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dy ^ 2, false>(
                                 std::get<side::south - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dy ^ 2, false>(
                                 std::get<side::east - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dy ^ 2, false>(
                                 std::get<side::west - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::front - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dy ^ 2, false>(
                                 std::get<side::front - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::back - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dy ^ 2, false>(
                                 std::get<side::back - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::stime - 1>(eval)[0]),
                             *(bspline_bdrSE.eval<deriv::dy ^ 2, false>(
                                 std::get<side::stime - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::etime - 1>(eval)[0]),
                             *(bspline_bdrSE.eval<deriv::dy ^ 2, false>(
                                 std::get<side::etime - 1>(xi))[0])));

    eval = functionspace.eval<functionspace::boundary, deriv::dz, false>(xi);

    EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dz, false>(
                                 std::get<side::north - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dz, false>(
                                 std::get<side::south - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dz, false>(
                                 std::get<side::east - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dz, false>(
                                 std::get<side::west - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::front - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dz, false>(
                                 std::get<side::front - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::back - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dz, false>(
                                 std::get<side::back - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::stime - 1>(eval)[0]),
                             *(bspline_bdrSE.eval<deriv::dz, false>(
                                 std::get<side::stime - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::etime - 1>(eval)[0]),
                             *(bspline_bdrSE.eval<deriv::dz, false>(
                                 std::get<side::etime - 1>(xi))[0])));

    eval =
        functionspace.eval<functionspace::boundary, deriv::dz ^ 2, false>(xi);

    EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dz ^ 2, false>(
                                 std::get<side::north - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dz ^ 2, false>(
                                 std::get<side::south - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dz ^ 2, false>(
                                 std::get<side::east - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dz ^ 2, false>(
                                 std::get<side::west - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::front - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dz ^ 2, false>(
                                 std::get<side::front - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::back - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dz ^ 2, false>(
                                 std::get<side::back - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::stime - 1>(eval)[0]),
                             *(bspline_bdrSE.eval<deriv::dz ^ 2, false>(
                                 std::get<side::stime - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::etime - 1>(eval)[0]),
                             *(bspline_bdrSE.eval<deriv::dz ^ 2, false>(
                                 std::get<side::etime - 1>(xi))[0])));

    eval = functionspace.eval<functionspace::boundary, deriv::dt, false>(xi);

    EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dt, false>(
                                 std::get<side::north - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dt, false>(
                                 std::get<side::south - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dt, false>(
                                 std::get<side::east - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dt, false>(
                                 std::get<side::west - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::front - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dt, false>(
                                 std::get<side::front - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::back - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dt, false>(
                                 std::get<side::back - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::stime - 1>(eval)[0]),
                             *(bspline_bdrSE.eval<deriv::dt, false>(
                                 std::get<side::stime - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::etime - 1>(eval)[0]),
                             *(bspline_bdrSE.eval<deriv::dt, false>(
                                 std::get<side::etime - 1>(xi))[0])));

    eval =
        functionspace.eval<functionspace::boundary, deriv::dt ^ 2, false>(xi);

    EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dt ^ 2, false>(
                                 std::get<side::north - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dt ^ 2, false>(
                                 std::get<side::south - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dt ^ 2, false>(
                                 std::get<side::east - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dt ^ 2, false>(
                                 std::get<side::west - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::front - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dt ^ 2, false>(
                                 std::get<side::front - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::back - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dt ^ 2, false>(
                                 std::get<side::back - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::stime - 1>(eval)[0]),
                             *(bspline_bdrSE.eval<deriv::dt ^ 2, false>(
                                 std::get<side::stime - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::etime - 1>(eval)[0]),
                             *(bspline_bdrSE.eval<deriv::dt ^ 2, false>(
                                 std::get<side::etime - 1>(xi))[0])));

    eval = functionspace
               .eval<functionspace::boundary, deriv::dx + deriv::dy, false>(xi);

    EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dx + deriv::dy, false>(
                                 std::get<side::north - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dx + deriv::dy, false>(
                                 std::get<side::south - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dx + deriv::dy, false>(
                                 std::get<side::east - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dx + deriv::dy, false>(
                                 std::get<side::west - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::front - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dx + deriv::dy, false>(
                                 std::get<side::front - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::back - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dx + deriv::dy, false>(
                                 std::get<side::back - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::stime - 1>(eval)[0]),
                             *(bspline_bdrSE.eval<deriv::dx + deriv::dy, false>(
                                 std::get<side::stime - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::etime - 1>(eval)[0]),
                             *(bspline_bdrSE.eval<deriv::dx + deriv::dy, false>(
                                 std::get<side::etime - 1>(xi))[0])));

    eval = functionspace
               .eval<functionspace::boundary, deriv::dx + deriv::dz, false>(xi);

    EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dx + deriv::dz, false>(
                                 std::get<side::north - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dx + deriv::dz, false>(
                                 std::get<side::south - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dx + deriv::dz, false>(
                                 std::get<side::east - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dx + deriv::dz, false>(
                                 std::get<side::west - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::front - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dx + deriv::dz, false>(
                                 std::get<side::front - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::back - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dx + deriv::dz, false>(
                                 std::get<side::back - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::stime - 1>(eval)[0]),
                             *(bspline_bdrSE.eval<deriv::dx + deriv::dz, false>(
                                 std::get<side::stime - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::etime - 1>(eval)[0]),
                             *(bspline_bdrSE.eval<deriv::dx + deriv::dz, false>(
                                 std::get<side::etime - 1>(xi))[0])));

    eval = functionspace
               .eval<functionspace::boundary, deriv::dx + deriv::dt, false>(xi);

    EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dx + deriv::dt, false>(
                                 std::get<side::north - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dx + deriv::dt, false>(
                                 std::get<side::south - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dx + deriv::dt, false>(
                                 std::get<side::east - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dx + deriv::dt, false>(
                                 std::get<side::west - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::front - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dx + deriv::dt, false>(
                                 std::get<side::front - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::back - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dx + deriv::dt, false>(
                                 std::get<side::back - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::stime - 1>(eval)[0]),
                             *(bspline_bdrSE.eval<deriv::dx + deriv::dt, false>(
                                 std::get<side::stime - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::etime - 1>(eval)[0]),
                             *(bspline_bdrSE.eval<deriv::dx + deriv::dt, false>(
                                 std::get<side::etime - 1>(xi))[0])));

    eval = functionspace
               .eval<functionspace::boundary, deriv::dy + deriv::dz, false>(xi);

    EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dy + deriv::dz, false>(
                                 std::get<side::north - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dy + deriv::dz, false>(
                                 std::get<side::south - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dy + deriv::dz, false>(
                                 std::get<side::east - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dy + deriv::dz, false>(
                                 std::get<side::west - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::front - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dy + deriv::dz, false>(
                                 std::get<side::front - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::back - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dy + deriv::dz, false>(
                                 std::get<side::back - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::stime - 1>(eval)[0]),
                             *(bspline_bdrSE.eval<deriv::dy + deriv::dz, false>(
                                 std::get<side::stime - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::etime - 1>(eval)[0]),
                             *(bspline_bdrSE.eval<deriv::dy + deriv::dz, false>(
                                 std::get<side::etime - 1>(xi))[0])));

    eval = functionspace
               .eval<functionspace::boundary, deriv::dy + deriv::dt, false>(xi);

    EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dy + deriv::dt, false>(
                                 std::get<side::north - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dy + deriv::dt, false>(
                                 std::get<side::south - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dy + deriv::dt, false>(
                                 std::get<side::east - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dy + deriv::dt, false>(
                                 std::get<side::west - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::front - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dy + deriv::dt, false>(
                                 std::get<side::front - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::back - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dy + deriv::dt, false>(
                                 std::get<side::back - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::stime - 1>(eval)[0]),
                             *(bspline_bdrSE.eval<deriv::dy + deriv::dt, false>(
                                 std::get<side::stime - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::etime - 1>(eval)[0]),
                             *(bspline_bdrSE.eval<deriv::dy + deriv::dt, false>(
                                 std::get<side::etime - 1>(xi))[0])));

    eval = functionspace
               .eval<functionspace::boundary, deriv::dz + deriv::dt, false>(xi);

    EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dz + deriv::dt, false>(
                                 std::get<side::north - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dz + deriv::dt, false>(
                                 std::get<side::south - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dz + deriv::dt, false>(
                                 std::get<side::east - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dz + deriv::dt, false>(
                                 std::get<side::west - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::front - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dz + deriv::dt, false>(
                                 std::get<side::front - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::back - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dz + deriv::dt, false>(
                                 std::get<side::back - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::stime - 1>(eval)[0]),
                             *(bspline_bdrSE.eval<deriv::dz + deriv::dt, false>(
                                 std::get<side::stime - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::etime - 1>(eval)[0]),
                             *(bspline_bdrSE.eval<deriv::dz + deriv::dt, false>(
                                 std::get<side::etime - 1>(xi))[0])));

    // Evaluation from precomputed coefficients and basis functions
    auto knot_indices =
        functionspace.template find_knot_indices<functionspace::boundary>(xi);
    auto coeff_indices =
        functionspace.template find_coeff_indices<functionspace::boundary>(
            knot_indices);

    auto numel = [](const auto &xi) {
      return std::tuple{std::get<0>(xi)[0].numel(), std::get<1>(xi)[0].numel(),
                        std::get<2>(xi)[0].numel(), std::get<3>(xi)[0].numel(),
                        std::get<4>(xi)[0].numel(), std::get<5>(xi)[0].numel(),
                        std::get<6>(xi)[0].numel(), std::get<7>(xi)[0].numel()};
    };
    auto sizes = [](const auto &xi) {
      return std::tuple{std::get<0>(xi)[0].sizes(), std::get<1>(xi)[0].sizes(),
                        std::get<2>(xi)[0].sizes(), std::get<3>(xi)[0].sizes(),
                        std::get<4>(xi)[0].sizes(), std::get<5>(xi)[0].sizes(),
                        std::get<6>(xi)[0].sizes(), std::get<7>(xi)[0].sizes()};
    };

    auto basfunc =
        functionspace
            .template eval_basfunc<functionspace::boundary, deriv::func, false>(
                xi, knot_indices);
    eval =
        functionspace.template eval_from_precomputed<functionspace::boundary>(
            basfunc, coeff_indices, numel(xi), sizes(xi));

    EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::func, false>(
                                 std::get<side::north - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::func, false>(
                                 std::get<side::south - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::func, false>(
                                 std::get<side::east - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::func, false>(
                                 std::get<side::west - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::front - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::func, false>(
                                 std::get<side::front - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::back - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::func, false>(
                                 std::get<side::back - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::stime - 1>(eval)[0]),
                             *(bspline_bdrSE.eval<deriv::func, false>(
                                 std::get<side::stime - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::etime - 1>(eval)[0]),
                             *(bspline_bdrSE.eval<deriv::func, false>(
                                 std::get<side::etime - 1>(xi))[0])));

    basfunc =
        functionspace
            .template eval_basfunc<functionspace::boundary, deriv::dx, false>(
                xi, knot_indices);
    eval =
        functionspace.template eval_from_precomputed<functionspace::boundary>(
            basfunc, coeff_indices, numel(xi), sizes(xi));

    EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dx, false>(
                                 std::get<side::north - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dx, false>(
                                 std::get<side::south - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dx, false>(
                                 std::get<side::east - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dx, false>(
                                 std::get<side::west - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::front - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dx, false>(
                                 std::get<side::front - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::back - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dx, false>(
                                 std::get<side::back - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::stime - 1>(eval)[0]),
                             *(bspline_bdrSE.eval<deriv::dx, false>(
                                 std::get<side::stime - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::etime - 1>(eval)[0]),
                             *(bspline_bdrSE.eval<deriv::dx, false>(
                                 std::get<side::etime - 1>(xi))[0])));

    basfunc = functionspace.template eval_basfunc<functionspace::boundary,
                                                  deriv::dx ^ 2, false>(
        xi, knot_indices);
    eval =
        functionspace.template eval_from_precomputed<functionspace::boundary>(
            basfunc, coeff_indices, numel(xi), sizes(xi));

    EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dx ^ 2, false>(
                                 std::get<side::north - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dx ^ 2, false>(
                                 std::get<side::south - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dx ^ 2, false>(
                                 std::get<side::east - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dx ^ 2, false>(
                                 std::get<side::west - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::front - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dx ^ 2, false>(
                                 std::get<side::front - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::back - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dx ^ 2, false>(
                                 std::get<side::back - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::stime - 1>(eval)[0]),
                             *(bspline_bdrSE.eval<deriv::dx ^ 2, false>(
                                 std::get<side::stime - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::etime - 1>(eval)[0]),
                             *(bspline_bdrSE.eval<deriv::dx ^ 2, false>(
                                 std::get<side::etime - 1>(xi))[0])));

    basfunc =
        functionspace
            .template eval_basfunc<functionspace::boundary, deriv::dy, false>(
                xi, knot_indices);
    eval =
        functionspace.template eval_from_precomputed<functionspace::boundary>(
            basfunc, coeff_indices, numel(xi), sizes(xi));

    EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dy, false>(
                                 std::get<side::north - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dy, false>(
                                 std::get<side::south - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dy, false>(
                                 std::get<side::east - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dy, false>(
                                 std::get<side::west - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::front - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dy, false>(
                                 std::get<side::front - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::back - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dy, false>(
                                 std::get<side::back - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::stime - 1>(eval)[0]),
                             *(bspline_bdrSE.eval<deriv::dy, false>(
                                 std::get<side::stime - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::etime - 1>(eval)[0]),
                             *(bspline_bdrSE.eval<deriv::dy, false>(
                                 std::get<side::etime - 1>(xi))[0])));

    basfunc = functionspace.template eval_basfunc<functionspace::boundary,
                                                  deriv::dy ^ 2, false>(
        xi, knot_indices);
    eval =
        functionspace.template eval_from_precomputed<functionspace::boundary>(
            basfunc, coeff_indices, numel(xi), sizes(xi));

    EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dy ^ 2, false>(
                                 std::get<side::north - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dy ^ 2, false>(
                                 std::get<side::south - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dy ^ 2, false>(
                                 std::get<side::east - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dy ^ 2, false>(
                                 std::get<side::west - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::front - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dy ^ 2, false>(
                                 std::get<side::front - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::back - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dy ^ 2, false>(
                                 std::get<side::back - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::stime - 1>(eval)[0]),
                             *(bspline_bdrSE.eval<deriv::dy ^ 2, false>(
                                 std::get<side::stime - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::etime - 1>(eval)[0]),
                             *(bspline_bdrSE.eval<deriv::dy ^ 2, false>(
                                 std::get<side::etime - 1>(xi))[0])));

    basfunc =
        functionspace
            .template eval_basfunc<functionspace::boundary, deriv::dz, false>(
                xi, knot_indices);
    eval =
        functionspace.template eval_from_precomputed<functionspace::boundary>(
            basfunc, coeff_indices, numel(xi), sizes(xi));

    EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dz, false>(
                                 std::get<side::north - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dz, false>(
                                 std::get<side::south - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dz, false>(
                                 std::get<side::east - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dz, false>(
                                 std::get<side::west - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::front - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dz, false>(
                                 std::get<side::front - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::back - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dz, false>(
                                 std::get<side::back - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::stime - 1>(eval)[0]),
                             *(bspline_bdrSE.eval<deriv::dz, false>(
                                 std::get<side::stime - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::etime - 1>(eval)[0]),
                             *(bspline_bdrSE.eval<deriv::dz, false>(
                                 std::get<side::etime - 1>(xi))[0])));

    basfunc = functionspace.template eval_basfunc<functionspace::boundary,
                                                  deriv::dz ^ 2, false>(
        xi, knot_indices);
    eval =
        functionspace.template eval_from_precomputed<functionspace::boundary>(
            basfunc, coeff_indices, numel(xi), sizes(xi));

    EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dz ^ 2, false>(
                                 std::get<side::north - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dz ^ 2, false>(
                                 std::get<side::south - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dz ^ 2, false>(
                                 std::get<side::east - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dz ^ 2, false>(
                                 std::get<side::west - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::front - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dz ^ 2, false>(
                                 std::get<side::front - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::back - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dz ^ 2, false>(
                                 std::get<side::back - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::stime - 1>(eval)[0]),
                             *(bspline_bdrSE.eval<deriv::dz ^ 2, false>(
                                 std::get<side::stime - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::etime - 1>(eval)[0]),
                             *(bspline_bdrSE.eval<deriv::dz ^ 2, false>(
                                 std::get<side::etime - 1>(xi))[0])));

    basfunc =
        functionspace
            .template eval_basfunc<functionspace::boundary, deriv::dt, false>(
                xi, knot_indices);
    eval =
        functionspace.template eval_from_precomputed<functionspace::boundary>(
            basfunc, coeff_indices, numel(xi), sizes(xi));

    EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dt, false>(
                                 std::get<side::north - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dt, false>(
                                 std::get<side::south - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dt, false>(
                                 std::get<side::east - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dt, false>(
                                 std::get<side::west - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::front - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dt, false>(
                                 std::get<side::front - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::back - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dt, false>(
                                 std::get<side::back - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::stime - 1>(eval)[0]),
                             *(bspline_bdrSE.eval<deriv::dt, false>(
                                 std::get<side::stime - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::etime - 1>(eval)[0]),
                             *(bspline_bdrSE.eval<deriv::dt, false>(
                                 std::get<side::etime - 1>(xi))[0])));

    basfunc = functionspace.template eval_basfunc<functionspace::boundary,
                                                  deriv::dt ^ 2, false>(
        xi, knot_indices);
    eval =
        functionspace.template eval_from_precomputed<functionspace::boundary>(
            basfunc, coeff_indices, numel(xi), sizes(xi));

    EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dt ^ 2, false>(
                                 std::get<side::north - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dt ^ 2, false>(
                                 std::get<side::south - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dt ^ 2, false>(
                                 std::get<side::east - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dt ^ 2, false>(
                                 std::get<side::west - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::front - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dt ^ 2, false>(
                                 std::get<side::front - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::back - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dt ^ 2, false>(
                                 std::get<side::back - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::stime - 1>(eval)[0]),
                             *(bspline_bdrSE.eval<deriv::dt ^ 2, false>(
                                 std::get<side::stime - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::etime - 1>(eval)[0]),
                             *(bspline_bdrSE.eval<deriv::dt ^ 2, false>(
                                 std::get<side::etime - 1>(xi))[0])));

    basfunc = functionspace.template eval_basfunc<functionspace::boundary,
                                                  deriv::dx + deriv::dy, false>(
        xi, knot_indices);
    eval =
        functionspace.template eval_from_precomputed<functionspace::boundary>(
            basfunc, coeff_indices, numel(xi), sizes(xi));

    EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dx + deriv::dy, false>(
                                 std::get<side::north - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dx + deriv::dy, false>(
                                 std::get<side::south - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dx + deriv::dy, false>(
                                 std::get<side::east - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dx + deriv::dy, false>(
                                 std::get<side::west - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::front - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dx + deriv::dy, false>(
                                 std::get<side::front - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::back - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dx + deriv::dy, false>(
                                 std::get<side::back - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::stime - 1>(eval)[0]),
                             *(bspline_bdrSE.eval<deriv::dx + deriv::dy, false>(
                                 std::get<side::stime - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::etime - 1>(eval)[0]),
                             *(bspline_bdrSE.eval<deriv::dx + deriv::dy, false>(
                                 std::get<side::etime - 1>(xi))[0])));

    basfunc = functionspace.template eval_basfunc<functionspace::boundary,
                                                  deriv::dx + deriv::dz, false>(
        xi, knot_indices);
    eval =
        functionspace.template eval_from_precomputed<functionspace::boundary>(
            basfunc, coeff_indices, numel(xi), sizes(xi));

    EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dx + deriv::dz, false>(
                                 std::get<side::north - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dx + deriv::dz, false>(
                                 std::get<side::south - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dx + deriv::dz, false>(
                                 std::get<side::east - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dx + deriv::dz, false>(
                                 std::get<side::west - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::front - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dx + deriv::dz, false>(
                                 std::get<side::front - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::back - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dx + deriv::dz, false>(
                                 std::get<side::back - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::stime - 1>(eval)[0]),
                             *(bspline_bdrSE.eval<deriv::dx + deriv::dz, false>(
                                 std::get<side::stime - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::etime - 1>(eval)[0]),
                             *(bspline_bdrSE.eval<deriv::dx + deriv::dz, false>(
                                 std::get<side::etime - 1>(xi))[0])));

    basfunc = functionspace.template eval_basfunc<functionspace::boundary,
                                                  deriv::dx + deriv::dt, false>(
        xi, knot_indices);
    eval =
        functionspace.template eval_from_precomputed<functionspace::boundary>(
            basfunc, coeff_indices, numel(xi), sizes(xi));

    EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dx + deriv::dt, false>(
                                 std::get<side::north - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dx + deriv::dt, false>(
                                 std::get<side::south - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dx + deriv::dt, false>(
                                 std::get<side::east - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dx + deriv::dt, false>(
                                 std::get<side::west - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::front - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dx + deriv::dt, false>(
                                 std::get<side::front - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::back - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dx + deriv::dt, false>(
                                 std::get<side::back - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::stime - 1>(eval)[0]),
                             *(bspline_bdrSE.eval<deriv::dx + deriv::dt, false>(
                                 std::get<side::stime - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::etime - 1>(eval)[0]),
                             *(bspline_bdrSE.eval<deriv::dx + deriv::dt, false>(
                                 std::get<side::etime - 1>(xi))[0])));

    basfunc = functionspace.template eval_basfunc<functionspace::boundary,
                                                  deriv::dy + deriv::dz, false>(
        xi, knot_indices);
    eval =
        functionspace.template eval_from_precomputed<functionspace::boundary>(
            basfunc, coeff_indices, numel(xi), sizes(xi));

    EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dy + deriv::dz, false>(
                                 std::get<side::north - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dy + deriv::dz, false>(
                                 std::get<side::south - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dy + deriv::dz, false>(
                                 std::get<side::east - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dy + deriv::dz, false>(
                                 std::get<side::west - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::front - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dy + deriv::dz, false>(
                                 std::get<side::front - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::back - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dy + deriv::dz, false>(
                                 std::get<side::back - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::stime - 1>(eval)[0]),
                             *(bspline_bdrSE.eval<deriv::dy + deriv::dz, false>(
                                 std::get<side::stime - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::etime - 1>(eval)[0]),
                             *(bspline_bdrSE.eval<deriv::dy + deriv::dz, false>(
                                 std::get<side::etime - 1>(xi))[0])));

    basfunc = functionspace.template eval_basfunc<functionspace::boundary,
                                                  deriv::dy + deriv::dt, false>(
        xi, knot_indices);
    eval =
        functionspace.template eval_from_precomputed<functionspace::boundary>(
            basfunc, coeff_indices, numel(xi), sizes(xi));

    EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dy + deriv::dt, false>(
                                 std::get<side::north - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dy + deriv::dt, false>(
                                 std::get<side::south - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dy + deriv::dt, false>(
                                 std::get<side::east - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dy + deriv::dt, false>(
                                 std::get<side::west - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::front - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dy + deriv::dt, false>(
                                 std::get<side::front - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::back - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dy + deriv::dt, false>(
                                 std::get<side::back - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::stime - 1>(eval)[0]),
                             *(bspline_bdrSE.eval<deriv::dy + deriv::dt, false>(
                                 std::get<side::stime - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::etime - 1>(eval)[0]),
                             *(bspline_bdrSE.eval<deriv::dy + deriv::dt, false>(
                                 std::get<side::etime - 1>(xi))[0])));

    basfunc = functionspace.template eval_basfunc<functionspace::boundary,
                                                  deriv::dz + deriv::dt, false>(
        xi, knot_indices);
    eval =
        functionspace.template eval_from_precomputed<functionspace::boundary>(
            basfunc, coeff_indices, numel(xi), sizes(xi));

    EXPECT_TRUE(torch::equal(*(std::get<side::north - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dz + deriv::dt, false>(
                                 std::get<side::north - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::south - 1>(eval)[0]),
                             *(bspline_bdrNS.eval<deriv::dz + deriv::dt, false>(
                                 std::get<side::south - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::east - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dz + deriv::dt, false>(
                                 std::get<side::east - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::west - 1>(eval)[0]),
                             *(bspline_bdrEW.eval<deriv::dz + deriv::dt, false>(
                                 std::get<side::west - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::front - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dz + deriv::dt, false>(
                                 std::get<side::front - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::back - 1>(eval)[0]),
                             *(bspline_bdrFB.eval<deriv::dz + deriv::dt, false>(
                                 std::get<side::back - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::stime - 1>(eval)[0]),
                             *(bspline_bdrSE.eval<deriv::dz + deriv::dt, false>(
                                 std::get<side::stime - 1>(xi))[0])));
    EXPECT_TRUE(torch::equal(*(std::get<side::etime - 1>(eval)[0]),
                             *(bspline_bdrSE.eval<deriv::dz + deriv::dt, false>(
                                 std::get<side::etime - 1>(xi))[0])));

    /// Evaluation of gradient (in parametric domain)
    auto grad = functionspace.template grad<functionspace::boundary, false>(xi);

    for (std::size_t i = 0; i < std::get<0>(grad).entries(); ++i) {
      EXPECT_TRUE(torch::equal(
          *(std::get<side::north - 1>(grad)[i]),
          *(bspline_bdrNS.grad<false>(std::get<side::north - 1>(xi))[i])));
      EXPECT_TRUE(torch::equal(
          *(std::get<side::south - 1>(grad)[i]),
          *(bspline_bdrNS.grad<false>(std::get<side::south - 1>(xi))[i])));
      EXPECT_TRUE(torch::equal(
          *(std::get<side::east - 1>(grad)[i]),
          *(bspline_bdrEW.grad<false>(std::get<side::east - 1>(xi))[i])));
      EXPECT_TRUE(torch::equal(
          *(std::get<side::west - 1>(grad)[i]),
          *(bspline_bdrEW.grad<false>(std::get<side::west - 1>(xi))[i])));
      EXPECT_TRUE(torch::equal(
          *(std::get<side::front - 1>(grad)[i]),
          *(bspline_bdrFB.grad<false>(std::get<side::front - 1>(xi))[i])));
      EXPECT_TRUE(torch::equal(
          *(std::get<side::back - 1>(grad)[i]),
          *(bspline_bdrFB.grad<false>(std::get<side::back - 1>(xi))[i])));
      EXPECT_TRUE(torch::equal(
          *(std::get<side::stime - 1>(grad)[i]),
          *(bspline_bdrSE.grad<false>(std::get<side::stime - 1>(xi))[i])));
      EXPECT_TRUE(torch::equal(
          *(std::get<side::etime - 1>(grad)[i]),
          *(bspline_bdrSE.grad<false>(std::get<side::etime - 1>(xi))[i])));
    }

    /// Evaluation of gradient (in physical domain)
    auto igrad = functionspace.template igrad<functionspace::boundary, false>(
        S4_geometry, xi);

    for (std::size_t i = 0; i < std::get<0>(igrad).entries(); ++i) {
      EXPECT_TRUE(
          torch::equal(*(std::get<side::north - 1>(igrad)[i]),
                       *(bspline_bdrNS.igrad<false>(
                           geometry_bdrNS, std::get<side::north - 1>(xi))[i])));
      EXPECT_TRUE(
          torch::equal(*(std::get<side::south - 1>(igrad)[i]),
                       *(bspline_bdrNS.igrad<false>(
                           geometry_bdrNS, std::get<side::south - 1>(xi))[i])));
      EXPECT_TRUE(
          torch::equal(*(std::get<side::east - 1>(igrad)[i]),
                       *(bspline_bdrEW.igrad<false>(
                           geometry_bdrEW, std::get<side::east - 1>(xi))[i])));
      EXPECT_TRUE(
          torch::equal(*(std::get<side::west - 1>(igrad)[i]),
                       *(bspline_bdrEW.igrad<false>(
                           geometry_bdrEW, std::get<side::west - 1>(xi))[i])));
      EXPECT_TRUE(
          torch::equal(*(std::get<side::front - 1>(igrad)[i]),
                       *(bspline_bdrFB.igrad<false>(
                           geometry_bdrFB, std::get<side::front - 1>(xi))[i])));
      EXPECT_TRUE(
          torch::equal(*(std::get<side::back - 1>(igrad)[i]),
                       *(bspline_bdrFB.igrad<false>(
                           geometry_bdrFB, std::get<side::back - 1>(xi))[i])));
      EXPECT_TRUE(
          torch::equal(*(std::get<side::stime - 1>(igrad)[i]),
                       *(bspline_bdrSE.igrad<false>(
                           geometry_bdrSE, std::get<side::stime - 1>(xi))[i])));
      EXPECT_TRUE(
          torch::equal(*(std::get<side::etime - 1>(igrad)[i]),
                       *(bspline_bdrSE.igrad<false>(
                           geometry_bdrSE, std::get<side::etime - 1>(xi))[i])));
    }

    /// Evaluation of Jacobian (in parametric domain)
    auto jac = functionspace.template jac<functionspace::boundary, false>(xi);

    for (std::size_t i = 0; i < std::get<0>(jac).entries(); ++i) {
      EXPECT_TRUE(torch::equal(
          *(std::get<side::north - 1>(jac)[i]),
          *(bspline_bdrNS.jac<false>(std::get<side::north - 1>(xi))[i])));
      EXPECT_TRUE(torch::equal(
          *(std::get<side::south - 1>(jac)[i]),
          *(bspline_bdrNS.jac<false>(std::get<side::south - 1>(xi))[i])));
      EXPECT_TRUE(torch::equal(
          *(std::get<side::east - 1>(jac)[i]),
          *(bspline_bdrEW.jac<false>(std::get<side::east - 1>(xi))[i])));
      EXPECT_TRUE(torch::equal(
          *(std::get<side::west - 1>(jac)[i]),
          *(bspline_bdrEW.jac<false>(std::get<side::west - 1>(xi))[i])));
      EXPECT_TRUE(torch::equal(
          *(std::get<side::front - 1>(jac)[i]),
          *(bspline_bdrFB.jac<false>(std::get<side::front - 1>(xi))[i])));
      EXPECT_TRUE(torch::equal(
          *(std::get<side::back - 1>(jac)[i]),
          *(bspline_bdrFB.jac<false>(std::get<side::back - 1>(xi))[i])));
      EXPECT_TRUE(torch::equal(
          *(std::get<side::stime - 1>(jac)[i]),
          *(bspline_bdrSE.jac<false>(std::get<side::stime - 1>(xi))[i])));
      EXPECT_TRUE(torch::equal(
          *(std::get<side::etime - 1>(jac)[i]),
          *(bspline_bdrSE.jac<false>(std::get<side::etime - 1>(xi))[i])));
    }

    /// Evaluation of Jacobian (in physical domain)
    auto ijac = functionspace.template ijac<functionspace::boundary, false>(
        S4_geometry, xi);

    for (std::size_t i = 0; i < std::get<0>(ijac).entries(); ++i) {
      EXPECT_TRUE(
          torch::equal(*(std::get<side::north - 1>(ijac)[i]),
                       *(bspline_bdrNS.ijac<false>(
                           geometry_bdrNS, std::get<side::north - 1>(xi))[i])));
      EXPECT_TRUE(
          torch::equal(*(std::get<side::south - 1>(ijac)[i]),
                       *(bspline_bdrNS.ijac<false>(
                           geometry_bdrNS, std::get<side::south - 1>(xi))[i])));
      EXPECT_TRUE(
          torch::equal(*(std::get<side::east - 1>(ijac)[i]),
                       *(bspline_bdrEW.ijac<false>(
                           geometry_bdrEW, std::get<side::east - 1>(xi))[i])));
      EXPECT_TRUE(
          torch::equal(*(std::get<side::west - 1>(ijac)[i]),
                       *(bspline_bdrEW.ijac<false>(
                           geometry_bdrEW, std::get<side::west - 1>(xi))[i])));
      EXPECT_TRUE(
          torch::equal(*(std::get<side::front - 1>(ijac)[i]),
                       *(bspline_bdrFB.ijac<false>(
                           geometry_bdrFB, std::get<side::front - 1>(xi))[i])));
      EXPECT_TRUE(
          torch::equal(*(std::get<side::back - 1>(ijac)[i]),
                       *(bspline_bdrFB.ijac<false>(
                           geometry_bdrFB, std::get<side::back - 1>(xi))[i])));
      EXPECT_TRUE(
          torch::equal(*(std::get<side::stime - 1>(ijac)[i]),
                       *(bspline_bdrSE.ijac<false>(
                           geometry_bdrSE, std::get<side::stime - 1>(xi))[i])));
      EXPECT_TRUE(
          torch::equal(*(std::get<side::etime - 1>(ijac)[i]),
                       *(bspline_bdrSE.ijac<false>(
                           geometry_bdrSE, std::get<side::etime - 1>(xi))[i])));
    }

    /// Evaluation of Hessian (in parametric domain)
    auto hess = functionspace.template hess<functionspace::boundary>(xi);

    for (std::size_t i = 0; i < std::get<0>(hess).entries(); ++i) {
      EXPECT_TRUE(torch::equal(
          *(std::get<side::north - 1>(hess)[i]),
          *(bspline_bdrNS.hess<false>(std::get<side::north - 1>(xi))[i])));
      EXPECT_TRUE(torch::equal(
          *(std::get<side::south - 1>(hess)[i]),
          *(bspline_bdrNS.hess<false>(std::get<side::south - 1>(xi))[i])));
      EXPECT_TRUE(torch::equal(
          *(std::get<side::east - 1>(hess)[i]),
          *(bspline_bdrEW.hess<false>(std::get<side::east - 1>(xi))[i])));
      EXPECT_TRUE(torch::equal(
          *(std::get<side::west - 1>(hess)[i]),
          *(bspline_bdrEW.hess<false>(std::get<side::west - 1>(xi))[i])));
      EXPECT_TRUE(torch::equal(
          *(std::get<side::front - 1>(hess)[i]),
          *(bspline_bdrFB.hess<false>(std::get<side::front - 1>(xi))[i])));
      EXPECT_TRUE(torch::equal(
          *(std::get<side::back - 1>(hess)[i]),
          *(bspline_bdrFB.hess<false>(std::get<side::back - 1>(xi))[i])));
      EXPECT_TRUE(torch::equal(
          *(std::get<side::stime - 1>(hess)[i]),
          *(bspline_bdrSE.hess<false>(std::get<side::stime - 1>(xi))[i])));
      EXPECT_TRUE(torch::equal(
          *(std::get<side::etime - 1>(hess)[i]),
          *(bspline_bdrSE.hess<false>(std::get<side::etime - 1>(xi))[i])));
    }

    /// Evaluation of Hessian (in physical domain)
    auto ihess =
        functionspace.template ihess<functionspace::boundary>(S4_geometry, xi);

    for (std::size_t i = 0; i < std::get<0>(ihess).entries(); ++i) {
      EXPECT_TRUE(
          torch::equal(*(std::get<side::north - 1>(ihess)[i]),
                       *(bspline_bdrNS.ihess<false>(
                           geometry_bdrNS, std::get<side::north - 1>(xi))[i])));
      EXPECT_TRUE(
          torch::equal(*(std::get<side::south - 1>(ihess)[i]),
                       *(bspline_bdrNS.ihess<false>(
                           geometry_bdrNS, std::get<side::south - 1>(xi))[i])));
      EXPECT_TRUE(
          torch::equal(*(std::get<side::east - 1>(ihess)[i]),
                       *(bspline_bdrEW.ihess<false>(
                           geometry_bdrEW, std::get<side::east - 1>(xi))[i])));
      EXPECT_TRUE(
          torch::equal(*(std::get<side::west - 1>(ihess)[i]),
                       *(bspline_bdrEW.ihess<false>(
                           geometry_bdrEW, std::get<side::west - 1>(xi))[i])));
      EXPECT_TRUE(
          torch::equal(*(std::get<side::front - 1>(ihess)[i]),
                       *(bspline_bdrFB.ihess<false>(
                           geometry_bdrFB, std::get<side::front - 1>(xi))[i])));
      EXPECT_TRUE(
          torch::equal(*(std::get<side::back - 1>(ihess)[i]),
                       *(bspline_bdrFB.ihess<false>(
                           geometry_bdrFB, std::get<side::back - 1>(xi))[i])));
      EXPECT_TRUE(
          torch::equal(*(std::get<side::stime - 1>(ihess)[i]),
                       *(bspline_bdrSE.ihess<false>(
                           geometry_bdrSE, std::get<side::stime - 1>(xi))[i])));
      EXPECT_TRUE(
          torch::equal(*(std::get<side::etime - 1>(ihess)[i]),
                       *(bspline_bdrSE.ihess<false>(
                           geometry_bdrSE, std::get<side::etime - 1>(xi))[i])));
    }
  }
}

TEST_F(FunctionSpaceTest, RT1_geoDim1_degrees2) {
  using iganet::deriv;
  using iganet::functionspace;
  using iganet::side;
  using BSpline = iganet::NonUniformBSpline<real_t, 1, 2>;
  using Geometry = iganet::NonUniformBSpline<real_t, 1, 2>;
  iganet::RT1<BSpline> functionspace({5}, iganet::init::greville, options);
  iganet::RT1<Geometry> RT1_geometry({5}, iganet::init::greville, options);

  iganet::NonUniformBSpline<real_t, 1, 3> bspline0(
      {5 + 1}, iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 1, 2> bspline1({5}, iganet::init::greville,
                                                   options);
  iganet::NonUniformBSpline<real_t, 1, 3> geometry0(
      {5 + 1}, iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 1, 2> geometry1({5}, iganet::init::greville,
                                                    options);

  { // Interior

    auto xi_ = iganet::utils::to_tensorArray<real_t>(
        {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0} /* u */, options);
    auto xi = std::tuple{xi_, xi_};

    // Evaluation
    auto eval =
        functionspace.eval<functionspace::interior, deriv::func, false>(xi);

    EXPECT_TRUE(torch::equal(*(std::get<0>(eval)[0]),
                             *(bspline0.eval<deriv::func, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<1>(eval)[0]),
                             *(bspline1.eval<deriv::func, false>(xi_)[0])));

    eval = functionspace.eval<functionspace::interior, deriv::dx, false>(xi);

    EXPECT_TRUE(torch::equal(*(std::get<0>(eval)[0]),
                             *(bspline0.eval<deriv::dx, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<1>(eval)[0]),
                             *(bspline1.eval<deriv::dx, false>(xi_)[0])));

    eval =
        functionspace.eval<functionspace::interior, deriv::dx ^ 2, false>(xi);

    EXPECT_TRUE(torch::equal(*(std::get<0>(eval)[0]),
                             *(bspline0.eval<deriv::dx ^ 2, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<1>(eval)[0]),
                             *(bspline1.eval<deriv::dx ^ 2, false>(xi_)[0])));

    // Evaluation from precomputed coefficients and basis functions
    auto knot_indices =
        functionspace.template find_knot_indices<functionspace::interior>(xi);
    auto coeff_indices =
        functionspace.template find_coeff_indices<functionspace::interior>(
            knot_indices);

    auto numel = [](const auto &xi) {
      return std::tuple{std::get<0>(xi)[0].numel(), std::get<1>(xi)[0].numel()};
    };
    auto sizes = [](const auto &xi) {
      return std::tuple{std::get<0>(xi)[0].sizes(), std::get<1>(xi)[0].sizes()};
    };

    auto basfunc =
        functionspace
            .template eval_basfunc<functionspace::interior, deriv::func, false>(
                xi, knot_indices);
    eval = functionspace.eval_from_precomputed<functionspace::interior>(
        basfunc, coeff_indices, numel(xi), sizes(xi));

    EXPECT_TRUE(torch::equal(*(std::get<0>(eval)[0]),
                             *(bspline0.eval<deriv::func, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<1>(eval)[0]),
                             *(bspline1.eval<deriv::func, false>(xi_)[0])));

    basfunc =
        functionspace
            .template eval_basfunc<functionspace::interior, deriv::dx, false>(
                xi, knot_indices);
    eval = functionspace.eval_from_precomputed<functionspace::interior>(
        basfunc, coeff_indices, numel(xi), sizes(xi));

    EXPECT_TRUE(torch::equal(*(std::get<0>(eval)[0]),
                             *(bspline0.eval<deriv::dx, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<1>(eval)[0]),
                             *(bspline1.eval<deriv::dx, false>(xi_)[0])));

    basfunc = functionspace.template eval_basfunc<functionspace::interior,
                                                  deriv::dx ^ 2, false>(
        xi, knot_indices);
    eval = functionspace.eval_from_precomputed<functionspace::interior>(
        basfunc, coeff_indices, numel(xi), sizes(xi));

    EXPECT_TRUE(torch::equal(*(std::get<0>(eval)[0]),
                             *(bspline0.eval<deriv::dx ^ 2, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<1>(eval)[0]),
                             *(bspline1.eval<deriv::dx ^ 2, false>(xi_)[0])));

    /// Evaluation of gradient (in parametric domain)
    auto grad = functionspace.template grad<functionspace::interior, false>(xi);
    auto grad_ref0 = bspline0.grad(xi_);
    auto grad_ref1 = bspline1.grad(xi_);

    for (std::size_t i = 0; i < grad_ref0.entries(); ++i) {
      EXPECT_TRUE(torch::equal(*(std::get<0>(grad)[i]), *(grad_ref0)[i]));
      EXPECT_TRUE(torch::equal(*(std::get<1>(grad)[i]), *(grad_ref1)[i]));
    }

    /// Evaluation of gradient (in physical domain)
    auto igrad = functionspace.template igrad<functionspace::interior, false>(
        RT1_geometry, xi);
    auto igrad_ref0 = bspline0.igrad(geometry0, xi_);
    auto igrad_ref1 = bspline1.igrad(geometry1, xi_);

    for (std::size_t i = 0; i < igrad_ref0.entries(); ++i) {
      EXPECT_TRUE(torch::equal(*(std::get<0>(igrad)[i]), *(igrad_ref0)[i]));
      EXPECT_TRUE(torch::equal(*(std::get<1>(igrad)[i]), *(igrad_ref1)[i]));
    }

    /// Evaluation of Jacobian (in parametric domain)
    auto jac = functionspace.template jac<functionspace::interior, false>(xi);
    auto jac_ref0 = bspline0.jac(xi_);
    auto jac_ref1 = bspline1.jac(xi_);

    for (std::size_t i = 0; i < jac_ref0.entries(); ++i) {
      EXPECT_TRUE(torch::equal(*(std::get<0>(jac)[i]), *(jac_ref0)[i]));
      EXPECT_TRUE(torch::equal(*(std::get<1>(jac)[i]), *(jac_ref1)[i]));
    }

    /// Evaluation of Jacobian (in physical domain)
    auto ijac = functionspace.template ijac<functionspace::interior, false>(
        RT1_geometry, xi);
    auto ijac_ref0 = bspline0.ijac(geometry0, xi_);
    auto ijac_ref1 = bspline1.ijac(geometry1, xi_);

    for (std::size_t i = 0; i < ijac_ref0.entries(); ++i) {
      EXPECT_TRUE(torch::equal(*(std::get<0>(ijac)[i]), *(ijac_ref0)[i]));
      EXPECT_TRUE(torch::equal(*(std::get<1>(ijac)[i]), *(ijac_ref1)[i]));
    }

    /// Evaluation of Hessian (in parametric domain)
    auto hess = functionspace.template hess<functionspace::interior, false>(xi);
    auto hess_ref0 = bspline0.hess(xi_);
    auto hess_ref1 = bspline1.hess(xi_);

    for (std::size_t i = 0; i < hess_ref0.entries(); ++i) {
      EXPECT_TRUE(torch::equal(*(std::get<0>(hess)[i]), *(hess_ref0)[i]));
      EXPECT_TRUE(torch::equal(*(std::get<1>(hess)[i]), *(hess_ref1)[i]));
    }

    /// Evaluation of Hessian (in physical domain)
    auto ihess = functionspace.template ihess<functionspace::interior, false>(
        RT1_geometry, xi);
    auto ihess_ref0 = bspline0.ihess(geometry0, xi_);
    auto ihess_ref1 = bspline1.ihess(geometry1, xi_);

    for (std::size_t i = 0; i < ihess_ref0.entries(); ++i) {
      EXPECT_TRUE(torch::equal(*(std::get<0>(ihess)[i]), *(ihess_ref0)[i]));
      EXPECT_TRUE(torch::equal(*(std::get<1>(ihess)[i]), *(ihess_ref1)[i]));
    }
  }
}

TEST_F(FunctionSpaceTest, RT2_geoDim1_degrees23) {
  using iganet::deriv;
  using iganet::functionspace;
  using iganet::side;
  using BSpline = iganet::NonUniformBSpline<real_t, 1, 2, 3>;
  using Geometry = iganet::NonUniformBSpline<real_t, 2, 2, 3>;
  iganet::RT2<BSpline> functionspace({5, 6}, iganet::init::greville, options);
  iganet::RT2<Geometry> RT2_geometry({5, 6}, iganet::init::greville, options);

  iganet::NonUniformBSpline<real_t, 1, 3, 3> bspline0(
      {5 + 1, 6}, iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 1, 2, 4> bspline1(
      {5, 6 + 1}, iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 1, 2, 3> bspline2(
      {5, 6}, iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 2, 3, 3> geometry0(
      {5 + 1, 6}, iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 2, 2, 4> geometry1(
      {5, 6 + 1}, iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 2, 2, 3> geometry2(
      {5, 6}, iganet::init::greville, options);

  { // Interior

    auto xi_ = iganet::utils::to_tensorArray<real_t>(
        {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0} /* u */,
        {1.0, 0.2, 0.1, 0.5, 0.9, 0.75, 0.0} /* v */, options);
    auto xi = std::tuple{xi_, xi_, xi_};

    // Evaluation
    auto eval =
        functionspace.eval<functionspace::interior, deriv::func, false>(xi);

    EXPECT_TRUE(torch::equal(*(std::get<0>(eval)[0]),
                             *(bspline0.eval<deriv::func, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<1>(eval)[0]),
                             *(bspline1.eval<deriv::func, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<2>(eval)[0]),
                             *(bspline2.eval<deriv::func, false>(xi_)[0])));

    eval = functionspace.eval<functionspace::interior, deriv::dx, false>(xi);

    EXPECT_TRUE(torch::equal(*(std::get<0>(eval)[0]),
                             *(bspline0.eval<deriv::dx, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<1>(eval)[0]),
                             *(bspline1.eval<deriv::dx, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<2>(eval)[0]),
                             *(bspline2.eval<deriv::dx, false>(xi_)[0])));

    eval =
        functionspace.eval<functionspace::interior, deriv::dx ^ 2, false>(xi);

    EXPECT_TRUE(torch::equal(*(std::get<0>(eval)[0]),
                             *(bspline0.eval<deriv::dx ^ 2, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<1>(eval)[0]),
                             *(bspline1.eval<deriv::dx ^ 2, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<2>(eval)[0]),
                             *(bspline2.eval<deriv::dx ^ 2, false>(xi_)[0])));

    eval = functionspace.eval<functionspace::interior, deriv::dy, false>(xi);

    EXPECT_TRUE(torch::equal(*(std::get<0>(eval)[0]),
                             *(bspline0.eval<deriv::dy, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<1>(eval)[0]),
                             *(bspline1.eval<deriv::dy, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<2>(eval)[0]),
                             *(bspline2.eval<deriv::dy, false>(xi_)[0])));

    eval =
        functionspace.eval<functionspace::interior, deriv::dy ^ 2, false>(xi);

    EXPECT_TRUE(torch::equal(*(std::get<0>(eval)[0]),
                             *(bspline0.eval<deriv::dy ^ 2, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<1>(eval)[0]),
                             *(bspline1.eval<deriv::dy ^ 2, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<2>(eval)[0]),
                             *(bspline2.eval<deriv::dy ^ 2, false>(xi_)[0])));

    eval = functionspace
               .eval<functionspace::interior, deriv::dx + deriv::dy, false>(xi);

    EXPECT_TRUE(
        torch::equal(*(std::get<0>(eval)[0]),
                     *(bspline0.eval<deriv::dx + deriv::dy, false>(xi_)[0])));
    EXPECT_TRUE(
        torch::equal(*(std::get<1>(eval)[0]),
                     *(bspline1.eval<deriv::dx + deriv::dy, false>(xi_)[0])));
    EXPECT_TRUE(
        torch::equal(*(std::get<2>(eval)[0]),
                     *(bspline2.eval<deriv::dx + deriv::dy, false>(xi_)[0])));

    // Evaluation from precomputed coefficients and basis functions
    auto knot_indices =
        functionspace.template find_knot_indices<functionspace::interior>(xi);
    auto coeff_indices =
        functionspace.template find_coeff_indices<functionspace::interior>(
            knot_indices);

    auto numel = [](const auto &xi) {
      return std::tuple{std::get<0>(xi)[0].numel(), std::get<1>(xi)[0].numel(),
                        std::get<2>(xi)[0].numel()};
    };
    auto sizes = [](const auto &xi) {
      return std::tuple{std::get<0>(xi)[0].sizes(), std::get<1>(xi)[0].sizes(),
                        std::get<2>(xi)[0].sizes()};
    };

    auto basfunc =
        functionspace
            .template eval_basfunc<functionspace::interior, deriv::func, false>(
                xi, knot_indices);
    eval = functionspace.eval_from_precomputed<functionspace::interior>(
        basfunc, coeff_indices, numel(xi), sizes(xi));

    EXPECT_TRUE(torch::equal(*(std::get<0>(eval)[0]),
                             *(bspline0.eval<deriv::func, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<1>(eval)[0]),
                             *(bspline1.eval<deriv::func, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<2>(eval)[0]),
                             *(bspline2.eval<deriv::func, false>(xi_)[0])));

    basfunc =
        functionspace
            .template eval_basfunc<functionspace::interior, deriv::dx, false>(
                xi, knot_indices);
    eval = functionspace.eval_from_precomputed<functionspace::interior>(
        basfunc, coeff_indices, numel(xi), sizes(xi));

    EXPECT_TRUE(torch::equal(*(std::get<0>(eval)[0]),
                             *(bspline0.eval<deriv::dx, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<1>(eval)[0]),
                             *(bspline1.eval<deriv::dx, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<2>(eval)[0]),
                             *(bspline2.eval<deriv::dx, false>(xi_)[0])));

    basfunc = functionspace.template eval_basfunc<functionspace::interior,
                                                  deriv::dx ^ 2, false>(
        xi, knot_indices);
    eval = functionspace.eval_from_precomputed<functionspace::interior>(
        basfunc, coeff_indices, numel(xi), sizes(xi));

    EXPECT_TRUE(torch::equal(*(std::get<0>(eval)[0]),
                             *(bspline0.eval<deriv::dx ^ 2, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<1>(eval)[0]),
                             *(bspline1.eval<deriv::dx ^ 2, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<2>(eval)[0]),
                             *(bspline2.eval<deriv::dx ^ 2, false>(xi_)[0])));

    basfunc =
        functionspace
            .template eval_basfunc<functionspace::interior, deriv::dy, false>(
                xi, knot_indices);
    eval = functionspace.eval_from_precomputed<functionspace::interior>(
        basfunc, coeff_indices, numel(xi), sizes(xi));

    EXPECT_TRUE(torch::equal(*(std::get<0>(eval)[0]),
                             *(bspline0.eval<deriv::dy, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<1>(eval)[0]),
                             *(bspline1.eval<deriv::dy, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<2>(eval)[0]),
                             *(bspline2.eval<deriv::dy, false>(xi_)[0])));

    basfunc = functionspace.template eval_basfunc<functionspace::interior,
                                                  deriv::dy ^ 2, false>(
        xi, knot_indices);
    eval = functionspace.eval_from_precomputed<functionspace::interior>(
        basfunc, coeff_indices, numel(xi), sizes(xi));

    EXPECT_TRUE(torch::equal(*(std::get<0>(eval)[0]),
                             *(bspline0.eval<deriv::dy ^ 2, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<1>(eval)[0]),
                             *(bspline1.eval<deriv::dy ^ 2, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<2>(eval)[0]),
                             *(bspline2.eval<deriv::dy ^ 2, false>(xi_)[0])));

    basfunc = functionspace.template eval_basfunc<functionspace::interior,
                                                  deriv::dx + deriv::dy, false>(
        xi, knot_indices);
    eval = functionspace.eval_from_precomputed<functionspace::interior>(
        basfunc, coeff_indices, numel(xi), sizes(xi));

    EXPECT_TRUE(
        torch::equal(*(std::get<0>(eval)[0]),
                     *(bspline0.eval<deriv::dx + deriv::dy, false>(xi_)[0])));
    EXPECT_TRUE(
        torch::equal(*(std::get<1>(eval)[0]),
                     *(bspline1.eval<deriv::dx + deriv::dy, false>(xi_)[0])));
    EXPECT_TRUE(
        torch::equal(*(std::get<2>(eval)[0]),
                     *(bspline2.eval<deriv::dx + deriv::dy, false>(xi_)[0])));

    /// Evaluation of gradient (in parametric domain)
    auto grad = functionspace.template grad<functionspace::interior, false>(xi);
    auto grad_ref0 = bspline0.grad(xi_);
    auto grad_ref1 = bspline1.grad(xi_);
    auto grad_ref2 = bspline2.grad(xi_);

    for (std::size_t i = 0; i < grad_ref0.entries(); ++i) {
      EXPECT_TRUE(torch::equal(*(std::get<0>(grad)[i]), *(grad_ref0)[i]));
      EXPECT_TRUE(torch::equal(*(std::get<1>(grad)[i]), *(grad_ref1)[i]));
      EXPECT_TRUE(torch::equal(*(std::get<2>(grad)[i]), *(grad_ref2)[i]));
    }

    /// Evaluation of gradient (in physical domain)
    auto igrad = functionspace.template igrad<functionspace::interior, false>(
        RT2_geometry, xi);
    auto igrad_ref0 = bspline0.igrad(geometry0, xi_);
    auto igrad_ref1 = bspline1.igrad(geometry1, xi_);
    auto igrad_ref2 = bspline2.igrad(geometry2, xi_);

    for (std::size_t i = 0; i < igrad_ref0.entries(); ++i) {
      EXPECT_TRUE(torch::equal(*(std::get<0>(igrad)[i]), *(igrad_ref0)[i]));
      EXPECT_TRUE(torch::equal(*(std::get<1>(igrad)[i]), *(igrad_ref1)[i]));
      EXPECT_TRUE(torch::equal(*(std::get<2>(igrad)[i]), *(igrad_ref2)[i]));
    }

    /// Evaluation of Jacobian (in parametric domain)
    auto jac = functionspace.template jac<functionspace::interior, false>(xi);
    auto jac_ref0 = bspline0.jac(xi_);
    auto jac_ref1 = bspline1.jac(xi_);
    auto jac_ref2 = bspline2.jac(xi_);

    for (std::size_t i = 0; i < jac_ref0.entries(); ++i) {
      EXPECT_TRUE(torch::equal(*(std::get<0>(jac)[i]), *(jac_ref0)[i]));
      EXPECT_TRUE(torch::equal(*(std::get<1>(jac)[i]), *(jac_ref1)[i]));
      EXPECT_TRUE(torch::equal(*(std::get<2>(jac)[i]), *(jac_ref2)[i]));
    }

    /// Evaluation of Jacobian (in physical domain)
    auto ijac = functionspace.template ijac<functionspace::interior, false>(
        RT2_geometry, xi);
    auto ijac_ref0 = bspline0.ijac(geometry0, xi_);
    auto ijac_ref1 = bspline1.ijac(geometry1, xi_);
    auto ijac_ref2 = bspline2.ijac(geometry2, xi_);

    for (std::size_t i = 0; i < ijac_ref0.entries(); ++i) {
      EXPECT_TRUE(torch::equal(*(std::get<0>(ijac)[i]), *(ijac_ref0)[i]));
      EXPECT_TRUE(torch::equal(*(std::get<1>(ijac)[i]), *(ijac_ref1)[i]));
      EXPECT_TRUE(torch::equal(*(std::get<2>(ijac)[i]), *(ijac_ref2)[i]));
    }

    /// Evaluation of Hessian (in parametric domain)
    auto hess = functionspace.template hess<functionspace::interior, false>(xi);
    auto hess_ref0 = bspline0.hess(xi_);
    auto hess_ref1 = bspline1.hess(xi_);
    auto hess_ref2 = bspline2.hess(xi_);

    for (std::size_t i = 0; i < hess_ref0.entries(); ++i) {
      EXPECT_TRUE(torch::equal(*(std::get<0>(hess)[i]), *(hess_ref0)[i]));
      EXPECT_TRUE(torch::equal(*(std::get<1>(hess)[i]), *(hess_ref1)[i]));
      EXPECT_TRUE(torch::equal(*(std::get<2>(hess)[i]), *(hess_ref2)[i]));
    }

    /// Evaluation of Hessian (in physical domain)
    auto ihess = functionspace.template ihess<functionspace::interior, false>(
        RT2_geometry, xi);
    auto ihess_ref0 = bspline0.ihess(geometry0, xi_);
    auto ihess_ref1 = bspline1.ihess(geometry1, xi_);
    auto ihess_ref2 = bspline2.ihess(geometry2, xi_);

    for (std::size_t i = 0; i < ihess_ref0.entries(); ++i) {
      EXPECT_TRUE(torch::equal(*(std::get<0>(ihess)[i]), *(ihess_ref0)[i]));
      EXPECT_TRUE(torch::equal(*(std::get<1>(ihess)[i]), *(ihess_ref1)[i]));
      EXPECT_TRUE(torch::equal(*(std::get<2>(ihess)[i]), *(ihess_ref2)[i]));
    }
  }
}

TEST_F(FunctionSpaceTest, RT3_geoDim1_degrees234) {
  using iganet::deriv;
  using iganet::functionspace;
  using iganet::side;
  using BSpline = iganet::NonUniformBSpline<real_t, 1, 2, 3, 4>;
  using Geometry = iganet::NonUniformBSpline<real_t, 3, 2, 3, 4>;
  iganet::RT3<BSpline> functionspace({5, 6, 7}, iganet::init::greville,
                                     options);
  iganet::RT3<Geometry> RT3_geometry({5, 6, 7}, iganet::init::greville,
                                     options);

  iganet::NonUniformBSpline<real_t, 1, 3, 3, 4> bspline0(
      {5 + 1, 6, 7}, iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 1, 2, 4, 4> bspline1(
      {5, 6 + 1, 7}, iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 1, 2, 3, 5> bspline2(
      {5, 6, 7 + 1}, iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 1, 2, 3, 4> bspline3(
      {5, 6, 7}, iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 3, 3, 3, 4> geometry0(
      {5 + 1, 6, 7}, iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 3, 2, 4, 4> geometry1(
      {5, 6 + 1, 7}, iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 3, 2, 3, 5> geometry2(
      {5, 6, 7 + 1}, iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 3, 2, 3, 4> geometry3(
      {5, 6, 7}, iganet::init::greville, options);

  { // Interior

    auto xi_ = iganet::utils::to_tensorArray<real_t>(
        {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0} /* u */,
        {1.0, 0.2, 0.1, 0.5, 0.9, 0.75, 0.0} /* v */,
        {0.2, 0.5, 0.75, 0.9, 1.0, 0.0, 0.1} /* w */, options);
    auto xi = std::tuple{xi_, xi_, xi_, xi_};

    // Evaluation
    auto eval =
        functionspace.eval<functionspace::interior, deriv::func, false>(xi);

    EXPECT_TRUE(torch::equal(*(std::get<0>(eval)[0]),
                             *(bspline0.eval<deriv::func, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<1>(eval)[0]),
                             *(bspline1.eval<deriv::func, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<2>(eval)[0]),
                             *(bspline2.eval<deriv::func, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<3>(eval)[0]),
                             *(bspline3.eval<deriv::func, false>(xi_)[0])));

    eval = functionspace.eval<functionspace::interior, deriv::dx, false>(xi);

    EXPECT_TRUE(torch::equal(*(std::get<0>(eval)[0]),
                             *(bspline0.eval<deriv::dx, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<1>(eval)[0]),
                             *(bspline1.eval<deriv::dx, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<2>(eval)[0]),
                             *(bspline2.eval<deriv::dx, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<3>(eval)[0]),
                             *(bspline3.eval<deriv::dx, false>(xi_)[0])));

    eval =
        functionspace.eval<functionspace::interior, deriv::dx ^ 2, false>(xi);

    EXPECT_TRUE(torch::equal(*(std::get<0>(eval)[0]),
                             *(bspline0.eval<deriv::dx ^ 2, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<1>(eval)[0]),
                             *(bspline1.eval<deriv::dx ^ 2, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<2>(eval)[0]),
                             *(bspline2.eval<deriv::dx ^ 2, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<3>(eval)[0]),
                             *(bspline3.eval<deriv::dx ^ 2, false>(xi_)[0])));

    eval = functionspace.eval<functionspace::interior, deriv::dy, false>(xi);

    EXPECT_TRUE(torch::equal(*(std::get<0>(eval)[0]),
                             *(bspline0.eval<deriv::dy, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<1>(eval)[0]),
                             *(bspline1.eval<deriv::dy, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<2>(eval)[0]),
                             *(bspline2.eval<deriv::dy, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<3>(eval)[0]),
                             *(bspline3.eval<deriv::dy, false>(xi_)[0])));

    eval =
        functionspace.eval<functionspace::interior, deriv::dy ^ 2, false>(xi);

    EXPECT_TRUE(torch::equal(*(std::get<0>(eval)[0]),
                             *(bspline0.eval<deriv::dy ^ 2, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<1>(eval)[0]),
                             *(bspline1.eval<deriv::dy ^ 2, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<2>(eval)[0]),
                             *(bspline2.eval<deriv::dy ^ 2, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<3>(eval)[0]),
                             *(bspline3.eval<deriv::dy ^ 2, false>(xi_)[0])));

    eval = functionspace.eval<functionspace::interior, deriv::dz, false>(xi);

    EXPECT_TRUE(torch::equal(*(std::get<0>(eval)[0]),
                             *(bspline0.eval<deriv::dz, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<1>(eval)[0]),
                             *(bspline1.eval<deriv::dz, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<2>(eval)[0]),
                             *(bspline2.eval<deriv::dz, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<3>(eval)[0]),
                             *(bspline3.eval<deriv::dz, false>(xi_)[0])));

    eval =
        functionspace.eval<functionspace::interior, deriv::dz ^ 2, false>(xi);

    EXPECT_TRUE(torch::equal(*(std::get<0>(eval)[0]),
                             *(bspline0.eval<deriv::dz ^ 2, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<1>(eval)[0]),
                             *(bspline1.eval<deriv::dz ^ 2, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<2>(eval)[0]),
                             *(bspline2.eval<deriv::dz ^ 2, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<3>(eval)[0]),
                             *(bspline3.eval<deriv::dz ^ 2, false>(xi_)[0])));

    eval = functionspace
               .eval<functionspace::interior, deriv::dx + deriv::dy, false>(xi);

    EXPECT_TRUE(
        torch::equal(*(std::get<0>(eval)[0]),
                     *(bspline0.eval<deriv::dx + deriv::dy, false>(xi_)[0])));
    EXPECT_TRUE(
        torch::equal(*(std::get<1>(eval)[0]),
                     *(bspline1.eval<deriv::dx + deriv::dy, false>(xi_)[0])));
    EXPECT_TRUE(
        torch::equal(*(std::get<2>(eval)[0]),
                     *(bspline2.eval<deriv::dx + deriv::dy, false>(xi_)[0])));
    EXPECT_TRUE(
        torch::equal(*(std::get<3>(eval)[0]),
                     *(bspline3.eval<deriv::dx + deriv::dy, false>(xi_)[0])));

    eval = functionspace
               .eval<functionspace::interior, deriv::dx + deriv::dz, false>(xi);

    EXPECT_TRUE(
        torch::equal(*(std::get<0>(eval)[0]),
                     *(bspline0.eval<deriv::dx + deriv::dz, false>(xi_)[0])));
    EXPECT_TRUE(
        torch::equal(*(std::get<1>(eval)[0]),
                     *(bspline1.eval<deriv::dx + deriv::dz, false>(xi_)[0])));
    EXPECT_TRUE(
        torch::equal(*(std::get<2>(eval)[0]),
                     *(bspline2.eval<deriv::dx + deriv::dz, false>(xi_)[0])));
    EXPECT_TRUE(
        torch::equal(*(std::get<3>(eval)[0]),
                     *(bspline3.eval<deriv::dx + deriv::dz, false>(xi_)[0])));

    eval = functionspace
               .eval<functionspace::interior, deriv::dy + deriv::dz, false>(xi);

    EXPECT_TRUE(
        torch::equal(*(std::get<0>(eval)[0]),
                     *(bspline0.eval<deriv::dy + deriv::dz, false>(xi_)[0])));
    EXPECT_TRUE(
        torch::equal(*(std::get<1>(eval)[0]),
                     *(bspline1.eval<deriv::dy + deriv::dz, false>(xi_)[0])));
    EXPECT_TRUE(
        torch::equal(*(std::get<2>(eval)[0]),
                     *(bspline2.eval<deriv::dy + deriv::dz, false>(xi_)[0])));
    EXPECT_TRUE(
        torch::equal(*(std::get<3>(eval)[0]),
                     *(bspline3.eval<deriv::dy + deriv::dz, false>(xi_)[0])));

    // Evaluation from precomputed coefficients and basis functions
    auto knot_indices =
        functionspace.template find_knot_indices<functionspace::interior>(xi);
    auto coeff_indices =
        functionspace.template find_coeff_indices<functionspace::interior>(
            knot_indices);

    auto numel = [](const auto &xi) {
      return std::tuple{std::get<0>(xi)[0].numel(), std::get<1>(xi)[0].numel(),
                        std::get<2>(xi)[0].numel(), std::get<3>(xi)[0].numel()};
    };
    auto sizes = [](const auto &xi) {
      return std::tuple{std::get<0>(xi)[0].sizes(), std::get<1>(xi)[0].sizes(),
                        std::get<2>(xi)[0].sizes(), std::get<3>(xi)[0].sizes()};
    };

    auto basfunc =
        functionspace
            .template eval_basfunc<functionspace::interior, deriv::func, false>(
                xi, knot_indices);
    eval = functionspace.eval_from_precomputed<functionspace::interior>(
        basfunc, coeff_indices, numel(xi), sizes(xi));

    EXPECT_TRUE(torch::equal(*(std::get<0>(eval)[0]),
                             *(bspline0.eval<deriv::func, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<1>(eval)[0]),
                             *(bspline1.eval<deriv::func, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<2>(eval)[0]),
                             *(bspline2.eval<deriv::func, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<3>(eval)[0]),
                             *(bspline3.eval<deriv::func, false>(xi_)[0])));

    basfunc =
        functionspace
            .template eval_basfunc<functionspace::interior, deriv::dx, false>(
                xi, knot_indices);
    eval = functionspace.eval_from_precomputed<functionspace::interior>(
        basfunc, coeff_indices, numel(xi), sizes(xi));

    EXPECT_TRUE(torch::equal(*(std::get<0>(eval)[0]),
                             *(bspline0.eval<deriv::dx, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<1>(eval)[0]),
                             *(bspline1.eval<deriv::dx, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<2>(eval)[0]),
                             *(bspline2.eval<deriv::dx, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<3>(eval)[0]),
                             *(bspline3.eval<deriv::dx, false>(xi_)[0])));

    basfunc = functionspace.template eval_basfunc<functionspace::interior,
                                                  deriv::dx ^ 2, false>(
        xi, knot_indices);
    eval = functionspace.eval_from_precomputed<functionspace::interior>(
        basfunc, coeff_indices, numel(xi), sizes(xi));

    EXPECT_TRUE(torch::equal(*(std::get<0>(eval)[0]),
                             *(bspline0.eval<deriv::dx ^ 2, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<1>(eval)[0]),
                             *(bspline1.eval<deriv::dx ^ 2, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<2>(eval)[0]),
                             *(bspline2.eval<deriv::dx ^ 2, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<3>(eval)[0]),
                             *(bspline3.eval<deriv::dx ^ 2, false>(xi_)[0])));

    basfunc =
        functionspace
            .template eval_basfunc<functionspace::interior, deriv::dy, false>(
                xi, knot_indices);
    eval = functionspace.eval_from_precomputed<functionspace::interior>(
        basfunc, coeff_indices, numel(xi), sizes(xi));

    EXPECT_TRUE(torch::equal(*(std::get<0>(eval)[0]),
                             *(bspline0.eval<deriv::dy, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<1>(eval)[0]),
                             *(bspline1.eval<deriv::dy, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<2>(eval)[0]),
                             *(bspline2.eval<deriv::dy, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<3>(eval)[0]),
                             *(bspline3.eval<deriv::dy, false>(xi_)[0])));

    basfunc = functionspace.template eval_basfunc<functionspace::interior,
                                                  deriv::dy ^ 2, false>(
        xi, knot_indices);
    eval = functionspace.eval_from_precomputed<functionspace::interior>(
        basfunc, coeff_indices, numel(xi), sizes(xi));

    EXPECT_TRUE(torch::equal(*(std::get<0>(eval)[0]),
                             *(bspline0.eval<deriv::dy ^ 2, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<1>(eval)[0]),
                             *(bspline1.eval<deriv::dy ^ 2, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<2>(eval)[0]),
                             *(bspline2.eval<deriv::dy ^ 2, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<3>(eval)[0]),
                             *(bspline3.eval<deriv::dy ^ 2, false>(xi_)[0])));

    basfunc =
        functionspace
            .template eval_basfunc<functionspace::interior, deriv::dz, false>(
                xi, knot_indices);
    eval = functionspace.eval_from_precomputed<functionspace::interior>(
        basfunc, coeff_indices, numel(xi), sizes(xi));

    EXPECT_TRUE(torch::equal(*(std::get<0>(eval)[0]),
                             *(bspline0.eval<deriv::dz, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<1>(eval)[0]),
                             *(bspline1.eval<deriv::dz, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<2>(eval)[0]),
                             *(bspline2.eval<deriv::dz, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<3>(eval)[0]),
                             *(bspline3.eval<deriv::dz, false>(xi_)[0])));

    basfunc = functionspace.template eval_basfunc<functionspace::interior,
                                                  deriv::dz ^ 2, false>(
        xi, knot_indices);
    eval = functionspace.eval_from_precomputed<functionspace::interior>(
        basfunc, coeff_indices, numel(xi), sizes(xi));

    EXPECT_TRUE(torch::equal(*(std::get<0>(eval)[0]),
                             *(bspline0.eval<deriv::dz ^ 2, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<1>(eval)[0]),
                             *(bspline1.eval<deriv::dz ^ 2, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<2>(eval)[0]),
                             *(bspline2.eval<deriv::dz ^ 2, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<3>(eval)[0]),
                             *(bspline3.eval<deriv::dz ^ 2, false>(xi_)[0])));

    basfunc = functionspace.template eval_basfunc<functionspace::interior,
                                                  deriv::dx + deriv::dy, false>(
        xi, knot_indices);
    eval = functionspace.eval_from_precomputed<functionspace::interior>(
        basfunc, coeff_indices, numel(xi), sizes(xi));

    EXPECT_TRUE(
        torch::equal(*(std::get<0>(eval)[0]),
                     *(bspline0.eval<deriv::dx + deriv::dy, false>(xi_)[0])));
    EXPECT_TRUE(
        torch::equal(*(std::get<1>(eval)[0]),
                     *(bspline1.eval<deriv::dx + deriv::dy, false>(xi_)[0])));
    EXPECT_TRUE(
        torch::equal(*(std::get<2>(eval)[0]),
                     *(bspline2.eval<deriv::dx + deriv::dy, false>(xi_)[0])));
    EXPECT_TRUE(
        torch::equal(*(std::get<3>(eval)[0]),
                     *(bspline3.eval<deriv::dx + deriv::dy, false>(xi_)[0])));

    basfunc = functionspace.template eval_basfunc<functionspace::interior,
                                                  deriv::dx + deriv::dz, false>(
        xi, knot_indices);
    eval = functionspace.eval_from_precomputed<functionspace::interior>(
        basfunc, coeff_indices, numel(xi), sizes(xi));

    EXPECT_TRUE(
        torch::equal(*(std::get<0>(eval)[0]),
                     *(bspline0.eval<deriv::dx + deriv::dz, false>(xi_)[0])));
    EXPECT_TRUE(
        torch::equal(*(std::get<1>(eval)[0]),
                     *(bspline1.eval<deriv::dx + deriv::dz, false>(xi_)[0])));
    EXPECT_TRUE(
        torch::equal(*(std::get<2>(eval)[0]),
                     *(bspline2.eval<deriv::dx + deriv::dz, false>(xi_)[0])));
    EXPECT_TRUE(
        torch::equal(*(std::get<3>(eval)[0]),
                     *(bspline3.eval<deriv::dx + deriv::dz, false>(xi_)[0])));

    basfunc = functionspace.template eval_basfunc<functionspace::interior,
                                                  deriv::dy + deriv::dz, false>(
        xi, knot_indices);
    eval = functionspace.eval_from_precomputed<functionspace::interior>(
        basfunc, coeff_indices, numel(xi), sizes(xi));

    EXPECT_TRUE(
        torch::equal(*(std::get<0>(eval)[0]),
                     *(bspline0.eval<deriv::dy + deriv::dz, false>(xi_)[0])));
    EXPECT_TRUE(
        torch::equal(*(std::get<1>(eval)[0]),
                     *(bspline1.eval<deriv::dy + deriv::dz, false>(xi_)[0])));
    EXPECT_TRUE(
        torch::equal(*(std::get<2>(eval)[0]),
                     *(bspline2.eval<deriv::dy + deriv::dz, false>(xi_)[0])));
    EXPECT_TRUE(
        torch::equal(*(std::get<3>(eval)[0]),
                     *(bspline3.eval<deriv::dy + deriv::dz, false>(xi_)[0])));

    /// Evaluation of gradient (in parametric domain)
    auto grad = functionspace.template grad<functionspace::interior, false>(xi);
    auto grad_ref0 = bspline0.grad(xi_);
    auto grad_ref1 = bspline1.grad(xi_);
    auto grad_ref2 = bspline2.grad(xi_);
    auto grad_ref3 = bspline3.grad(xi_);

    for (std::size_t i = 0; i < grad_ref0.entries(); ++i) {
      EXPECT_TRUE(torch::equal(*(std::get<0>(grad)[i]), *(grad_ref0)[i]));
      EXPECT_TRUE(torch::equal(*(std::get<1>(grad)[i]), *(grad_ref1)[i]));
      EXPECT_TRUE(torch::equal(*(std::get<2>(grad)[i]), *(grad_ref2)[i]));
      EXPECT_TRUE(torch::equal(*(std::get<3>(grad)[i]), *(grad_ref3)[i]));
    }

    /// Evaluation of gradient (in physical domain)
    auto igrad = functionspace.template igrad<functionspace::interior, false>(
        RT3_geometry, xi);
    auto igrad_ref0 = bspline0.igrad(geometry0, xi_);
    auto igrad_ref1 = bspline1.igrad(geometry1, xi_);
    auto igrad_ref2 = bspline2.igrad(geometry2, xi_);
    auto igrad_ref3 = bspline3.igrad(geometry3, xi_);

    for (std::size_t i = 0; i < igrad_ref0.entries(); ++i) {
      EXPECT_TRUE(torch::equal(*(std::get<0>(igrad)[i]), *(igrad_ref0)[i]));
      EXPECT_TRUE(torch::equal(*(std::get<1>(igrad)[i]), *(igrad_ref1)[i]));
      EXPECT_TRUE(torch::equal(*(std::get<2>(igrad)[i]), *(igrad_ref2)[i]));
      EXPECT_TRUE(torch::equal(*(std::get<3>(igrad)[i]), *(igrad_ref3)[i]));
    }

    /// Evaluation of Jacobian (in parametric domain)
    auto jac = functionspace.template jac<functionspace::interior, false>(xi);
    auto jac_ref0 = bspline0.jac(xi_);
    auto jac_ref1 = bspline1.jac(xi_);
    auto jac_ref2 = bspline2.jac(xi_);
    auto jac_ref3 = bspline3.jac(xi_);

    for (std::size_t i = 0; i < jac_ref0.entries(); ++i) {
      EXPECT_TRUE(torch::equal(*(std::get<0>(jac)[i]), *(jac_ref0)[i]));
      EXPECT_TRUE(torch::equal(*(std::get<1>(jac)[i]), *(jac_ref1)[i]));
      EXPECT_TRUE(torch::equal(*(std::get<2>(jac)[i]), *(jac_ref2)[i]));
      EXPECT_TRUE(torch::equal(*(std::get<3>(jac)[i]), *(jac_ref3)[i]));
    }

    /// Evaluation of Jacobian (in physical domain)
    auto ijac = functionspace.template ijac<functionspace::interior, false>(
        RT3_geometry, xi);
    auto ijac_ref0 = bspline0.ijac(geometry0, xi_);
    auto ijac_ref1 = bspline1.ijac(geometry1, xi_);
    auto ijac_ref2 = bspline2.ijac(geometry2, xi_);
    auto ijac_ref3 = bspline3.ijac(geometry3, xi_);

    for (std::size_t i = 0; i < ijac_ref0.entries(); ++i) {
      EXPECT_TRUE(torch::equal(*(std::get<0>(ijac)[i]), *(ijac_ref0)[i]));
      EXPECT_TRUE(torch::equal(*(std::get<1>(ijac)[i]), *(ijac_ref1)[i]));
      EXPECT_TRUE(torch::equal(*(std::get<2>(ijac)[i]), *(ijac_ref2)[i]));
      EXPECT_TRUE(torch::equal(*(std::get<3>(ijac)[i]), *(ijac_ref3)[i]));
    }

    /// Evaluation of Hessian (in parametric domain)
    auto hess = functionspace.template hess<functionspace::interior, false>(xi);
    auto hess_ref0 = bspline0.hess(xi_);
    auto hess_ref1 = bspline1.hess(xi_);
    auto hess_ref2 = bspline2.hess(xi_);
    auto hess_ref3 = bspline3.hess(xi_);

    for (std::size_t i = 0; i < hess_ref0.entries(); ++i) {
      EXPECT_TRUE(torch::equal(*(std::get<0>(hess)[i]), *(hess_ref0)[i]));
      EXPECT_TRUE(torch::equal(*(std::get<1>(hess)[i]), *(hess_ref1)[i]));
      EXPECT_TRUE(torch::equal(*(std::get<2>(hess)[i]), *(hess_ref2)[i]));
      EXPECT_TRUE(torch::equal(*(std::get<3>(hess)[i]), *(hess_ref3)[i]));
    }

    /// Evaluation of Hessian (in physical domain)
    auto ihess = functionspace.template ihess<functionspace::interior, false>(
        RT3_geometry, xi);
    auto ihess_ref0 = bspline0.ihess(geometry0, xi_);
    auto ihess_ref1 = bspline1.ihess(geometry1, xi_);
    auto ihess_ref2 = bspline2.ihess(geometry2, xi_);
    auto ihess_ref3 = bspline3.ihess(geometry3, xi_);

    for (std::size_t i = 0; i < ihess_ref0.entries(); ++i) {
      EXPECT_TRUE(torch::equal(*(std::get<0>(ihess)[i]), *(ihess_ref0)[i]));
      EXPECT_TRUE(torch::equal(*(std::get<1>(ihess)[i]), *(ihess_ref1)[i]));
      EXPECT_TRUE(torch::equal(*(std::get<2>(ihess)[i]), *(ihess_ref2)[i]));
      EXPECT_TRUE(torch::equal(*(std::get<3>(ihess)[i]), *(ihess_ref3)[i]));
    }
  }
}

TEST_F(FunctionSpaceTest, RT4_geoDim1_degrees2341) {
  using iganet::deriv;
  using iganet::functionspace;
  using iganet::side;
  using BSpline = iganet::NonUniformBSpline<real_t, 1, 2, 3, 4, 1>;
  using Geometry = iganet::NonUniformBSpline<real_t, 4, 2, 3, 4, 1>;
  iganet::RT4<BSpline> functionspace({5, 6, 7, 4}, iganet::init::greville,
                                     options);
  iganet::RT4<Geometry> RT4_geometry({5, 6, 7, 4}, iganet::init::greville,
                                     options);

  iganet::NonUniformBSpline<real_t, 1, 3, 3, 4, 1> bspline0(
      {5 + 1, 6, 7, 4}, iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 1, 2, 4, 4, 1> bspline1(
      {5, 6 + 1, 7, 4}, iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 1, 2, 3, 5, 1> bspline2(
      {5, 6, 7 + 1, 4}, iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 1, 2, 3, 4, 2> bspline3(
      {5, 6, 7, 4 + 1}, iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 1, 2, 3, 4, 1> bspline4(
      {5, 6, 7, 4}, iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 4, 3, 3, 4, 1> geometry0(
      {5 + 1, 6, 7, 4}, iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 4, 2, 4, 4, 1> geometry1(
      {5, 6 + 1, 7, 4}, iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 4, 2, 3, 5, 1> geometry2(
      {5, 6, 7 + 1, 4}, iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 4, 2, 3, 4, 2> geometry3(
      {5, 6, 7, 4 + 1}, iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 4, 2, 3, 4, 1> geometry4(
      {5, 6, 7, 4}, iganet::init::greville, options);

  { // Interior

    auto xi_ = iganet::utils::to_tensorArray<real_t>(
        {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0} /* u */,
        {1.0, 0.2, 0.1, 0.5, 0.9, 0.75, 0.0} /* v */,
        {0.2, 0.5, 0.75, 0.9, 1.0, 0.0, 0.1} /* w */,
        {0.1, 0.1, 0.2, 0.3, 0.3, 0.0, 0.1} /* t */, options);
    auto xi = std::tuple{xi_, xi_, xi_, xi_, xi_};

    // Evaluation
    auto eval =
        functionspace.eval<functionspace::interior, deriv::func, false>(xi);

    EXPECT_TRUE(torch::equal(*(std::get<0>(eval)[0]),
                             *(bspline0.eval<deriv::func, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<1>(eval)[0]),
                             *(bspline1.eval<deriv::func, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<2>(eval)[0]),
                             *(bspline2.eval<deriv::func, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<3>(eval)[0]),
                             *(bspline3.eval<deriv::func, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<4>(eval)[0]),
                             *(bspline4.eval<deriv::func, false>(xi_)[0])));

    eval = functionspace.eval<functionspace::interior, deriv::dx, false>(xi);

    EXPECT_TRUE(torch::equal(*(std::get<0>(eval)[0]),
                             *(bspline0.eval<deriv::dx, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<1>(eval)[0]),
                             *(bspline1.eval<deriv::dx, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<2>(eval)[0]),
                             *(bspline2.eval<deriv::dx, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<3>(eval)[0]),
                             *(bspline3.eval<deriv::dx, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<4>(eval)[0]),
                             *(bspline4.eval<deriv::dx, false>(xi_)[0])));

    eval =
        functionspace.eval<functionspace::interior, deriv::dx ^ 2, false>(xi);

    EXPECT_TRUE(torch::equal(*(std::get<0>(eval)[0]),
                             *(bspline0.eval<deriv::dx ^ 2, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<1>(eval)[0]),
                             *(bspline1.eval<deriv::dx ^ 2, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<2>(eval)[0]),
                             *(bspline2.eval<deriv::dx ^ 2, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<3>(eval)[0]),
                             *(bspline3.eval<deriv::dx ^ 2, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<4>(eval)[0]),
                             *(bspline4.eval<deriv::dx ^ 2, false>(xi_)[0])));

    eval = functionspace.eval<functionspace::interior, deriv::dy, false>(xi);

    EXPECT_TRUE(torch::equal(*(std::get<0>(eval)[0]),
                             *(bspline0.eval<deriv::dy, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<1>(eval)[0]),
                             *(bspline1.eval<deriv::dy, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<2>(eval)[0]),
                             *(bspline2.eval<deriv::dy, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<3>(eval)[0]),
                             *(bspline3.eval<deriv::dy, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<4>(eval)[0]),
                             *(bspline4.eval<deriv::dy, false>(xi_)[0])));

    eval =
        functionspace.eval<functionspace::interior, deriv::dy ^ 2, false>(xi);

    EXPECT_TRUE(torch::equal(*(std::get<0>(eval)[0]),
                             *(bspline0.eval<deriv::dy ^ 2, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<1>(eval)[0]),
                             *(bspline1.eval<deriv::dy ^ 2, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<2>(eval)[0]),
                             *(bspline2.eval<deriv::dy ^ 2, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<3>(eval)[0]),
                             *(bspline3.eval<deriv::dy ^ 2, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<4>(eval)[0]),
                             *(bspline4.eval<deriv::dy ^ 2, false>(xi_)[0])));

    eval = functionspace.eval<functionspace::interior, deriv::dz, false>(xi);

    EXPECT_TRUE(torch::equal(*(std::get<0>(eval)[0]),
                             *(bspline0.eval<deriv::dz, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<1>(eval)[0]),
                             *(bspline1.eval<deriv::dz, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<2>(eval)[0]),
                             *(bspline2.eval<deriv::dz, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<3>(eval)[0]),
                             *(bspline3.eval<deriv::dz, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<4>(eval)[0]),
                             *(bspline4.eval<deriv::dz, false>(xi_)[0])));

    eval =
        functionspace.eval<functionspace::interior, deriv::dz ^ 2, false>(xi);

    EXPECT_TRUE(torch::equal(*(std::get<0>(eval)[0]),
                             *(bspline0.eval<deriv::dz ^ 2, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<1>(eval)[0]),
                             *(bspline1.eval<deriv::dz ^ 2, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<2>(eval)[0]),
                             *(bspline2.eval<deriv::dz ^ 2, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<3>(eval)[0]),
                             *(bspline3.eval<deriv::dz ^ 2, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<4>(eval)[0]),
                             *(bspline4.eval<deriv::dz ^ 2, false>(xi_)[0])));

    eval = functionspace.eval<functionspace::interior, deriv::dt, false>(xi);

    EXPECT_TRUE(torch::equal(*(std::get<0>(eval)[0]),
                             *(bspline0.eval<deriv::dt, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<1>(eval)[0]),
                             *(bspline1.eval<deriv::dt, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<2>(eval)[0]),
                             *(bspline2.eval<deriv::dt, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<3>(eval)[0]),
                             *(bspline3.eval<deriv::dt, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<4>(eval)[0]),
                             *(bspline4.eval<deriv::dt, false>(xi_)[0])));

    eval =
        functionspace.eval<functionspace::interior, deriv::dt ^ 2, false>(xi);

    EXPECT_TRUE(torch::equal(*(std::get<0>(eval)[0]),
                             *(bspline0.eval<deriv::dt ^ 2, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<1>(eval)[0]),
                             *(bspline1.eval<deriv::dt ^ 2, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<2>(eval)[0]),
                             *(bspline2.eval<deriv::dt ^ 2, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<3>(eval)[0]),
                             *(bspline3.eval<deriv::dt ^ 2, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<4>(eval)[0]),
                             *(bspline4.eval<deriv::dt ^ 2, false>(xi_)[0])));

    eval = functionspace
               .eval<functionspace::interior, deriv::dx + deriv::dy, false>(xi);

    EXPECT_TRUE(
        torch::equal(*(std::get<0>(eval)[0]),
                     *(bspline0.eval<deriv::dx + deriv::dy, false>(xi_)[0])));
    EXPECT_TRUE(
        torch::equal(*(std::get<1>(eval)[0]),
                     *(bspline1.eval<deriv::dx + deriv::dy, false>(xi_)[0])));
    EXPECT_TRUE(
        torch::equal(*(std::get<2>(eval)[0]),
                     *(bspline2.eval<deriv::dx + deriv::dy, false>(xi_)[0])));
    EXPECT_TRUE(
        torch::equal(*(std::get<3>(eval)[0]),
                     *(bspline3.eval<deriv::dx + deriv::dy, false>(xi_)[0])));
    EXPECT_TRUE(
        torch::equal(*(std::get<4>(eval)[0]),
                     *(bspline4.eval<deriv::dx + deriv::dy, false>(xi_)[0])));

    eval = functionspace
               .eval<functionspace::interior, deriv::dx + deriv::dz, false>(xi);

    EXPECT_TRUE(
        torch::equal(*(std::get<0>(eval)[0]),
                     *(bspline0.eval<deriv::dx + deriv::dz, false>(xi_)[0])));
    EXPECT_TRUE(
        torch::equal(*(std::get<1>(eval)[0]),
                     *(bspline1.eval<deriv::dx + deriv::dz, false>(xi_)[0])));
    EXPECT_TRUE(
        torch::equal(*(std::get<2>(eval)[0]),
                     *(bspline2.eval<deriv::dx + deriv::dz, false>(xi_)[0])));
    EXPECT_TRUE(
        torch::equal(*(std::get<3>(eval)[0]),
                     *(bspline3.eval<deriv::dx + deriv::dz, false>(xi_)[0])));
    EXPECT_TRUE(
        torch::equal(*(std::get<4>(eval)[0]),
                     *(bspline4.eval<deriv::dx + deriv::dz, false>(xi_)[0])));

    eval = functionspace
               .eval<functionspace::interior, deriv::dx + deriv::dt, false>(xi);

    EXPECT_TRUE(
        torch::equal(*(std::get<0>(eval)[0]),
                     *(bspline0.eval<deriv::dx + deriv::dt, false>(xi_)[0])));
    EXPECT_TRUE(
        torch::equal(*(std::get<1>(eval)[0]),
                     *(bspline1.eval<deriv::dx + deriv::dt, false>(xi_)[0])));
    EXPECT_TRUE(
        torch::equal(*(std::get<2>(eval)[0]),
                     *(bspline2.eval<deriv::dx + deriv::dt, false>(xi_)[0])));
    EXPECT_TRUE(
        torch::equal(*(std::get<3>(eval)[0]),
                     *(bspline3.eval<deriv::dx + deriv::dt, false>(xi_)[0])));
    EXPECT_TRUE(
        torch::equal(*(std::get<4>(eval)[0]),
                     *(bspline4.eval<deriv::dx + deriv::dt, false>(xi_)[0])));

    eval = functionspace
               .eval<functionspace::interior, deriv::dy + deriv::dz, false>(xi);

    EXPECT_TRUE(
        torch::equal(*(std::get<0>(eval)[0]),
                     *(bspline0.eval<deriv::dy + deriv::dz, false>(xi_)[0])));
    EXPECT_TRUE(
        torch::equal(*(std::get<1>(eval)[0]),
                     *(bspline1.eval<deriv::dy + deriv::dz, false>(xi_)[0])));
    EXPECT_TRUE(
        torch::equal(*(std::get<2>(eval)[0]),
                     *(bspline2.eval<deriv::dy + deriv::dz, false>(xi_)[0])));
    EXPECT_TRUE(
        torch::equal(*(std::get<3>(eval)[0]),
                     *(bspline3.eval<deriv::dy + deriv::dz, false>(xi_)[0])));
    EXPECT_TRUE(
        torch::equal(*(std::get<4>(eval)[0]),
                     *(bspline4.eval<deriv::dy + deriv::dz, false>(xi_)[0])));

    eval = functionspace
               .eval<functionspace::interior, deriv::dy + deriv::dt, false>(xi);

    EXPECT_TRUE(
        torch::equal(*(std::get<0>(eval)[0]),
                     *(bspline0.eval<deriv::dy + deriv::dt, false>(xi_)[0])));
    EXPECT_TRUE(
        torch::equal(*(std::get<1>(eval)[0]),
                     *(bspline1.eval<deriv::dy + deriv::dt, false>(xi_)[0])));
    EXPECT_TRUE(
        torch::equal(*(std::get<2>(eval)[0]),
                     *(bspline2.eval<deriv::dy + deriv::dt, false>(xi_)[0])));
    EXPECT_TRUE(
        torch::equal(*(std::get<3>(eval)[0]),
                     *(bspline3.eval<deriv::dy + deriv::dt, false>(xi_)[0])));
    EXPECT_TRUE(
        torch::equal(*(std::get<4>(eval)[0]),
                     *(bspline4.eval<deriv::dy + deriv::dt, false>(xi_)[0])));

    eval = functionspace
               .eval<functionspace::interior, deriv::dz + deriv::dt, false>(xi);

    EXPECT_TRUE(
        torch::equal(*(std::get<0>(eval)[0]),
                     *(bspline0.eval<deriv::dz + deriv::dt, false>(xi_)[0])));
    EXPECT_TRUE(
        torch::equal(*(std::get<1>(eval)[0]),
                     *(bspline1.eval<deriv::dz + deriv::dt, false>(xi_)[0])));
    EXPECT_TRUE(
        torch::equal(*(std::get<2>(eval)[0]),
                     *(bspline2.eval<deriv::dz + deriv::dt, false>(xi_)[0])));
    EXPECT_TRUE(
        torch::equal(*(std::get<3>(eval)[0]),
                     *(bspline3.eval<deriv::dz + deriv::dt, false>(xi_)[0])));
    EXPECT_TRUE(
        torch::equal(*(std::get<4>(eval)[0]),
                     *(bspline4.eval<deriv::dz + deriv::dt, false>(xi_)[0])));

    // Evaluation from precomputed coefficients and basis functions
    auto knot_indices =
        functionspace.template find_knot_indices<functionspace::interior>(xi);
    auto coeff_indices =
        functionspace.template find_coeff_indices<functionspace::interior>(
            knot_indices);

    auto numel = [](const auto &xi) {
      return std::tuple{std::get<0>(xi)[0].numel(), std::get<1>(xi)[0].numel(),
                        std::get<2>(xi)[0].numel(), std::get<3>(xi)[0].numel(),
                        std::get<4>(xi)[0].numel()};
    };
    auto sizes = [](const auto &xi) {
      return std::tuple{std::get<0>(xi)[0].sizes(), std::get<1>(xi)[0].sizes(),
                        std::get<2>(xi)[0].sizes(), std::get<3>(xi)[0].sizes(),
                        std::get<4>(xi)[0].sizes()};
    };

    auto basfunc =
        functionspace
            .template eval_basfunc<functionspace::interior, deriv::func, false>(
                xi, knot_indices);
    eval = functionspace.eval_from_precomputed<functionspace::interior>(
        basfunc, coeff_indices, numel(xi), sizes(xi));

    EXPECT_TRUE(torch::equal(*(std::get<0>(eval)[0]),
                             *(bspline0.eval<deriv::func, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<1>(eval)[0]),
                             *(bspline1.eval<deriv::func, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<2>(eval)[0]),
                             *(bspline2.eval<deriv::func, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<3>(eval)[0]),
                             *(bspline3.eval<deriv::func, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<4>(eval)[0]),
                             *(bspline4.eval<deriv::func, false>(xi_)[0])));

    basfunc =
        functionspace
            .template eval_basfunc<functionspace::interior, deriv::dx, false>(
                xi, knot_indices);
    eval = functionspace.eval_from_precomputed<functionspace::interior>(
        basfunc, coeff_indices, numel(xi), sizes(xi));

    EXPECT_TRUE(torch::equal(*(std::get<0>(eval)[0]),
                             *(bspline0.eval<deriv::dx, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<1>(eval)[0]),
                             *(bspline1.eval<deriv::dx, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<2>(eval)[0]),
                             *(bspline2.eval<deriv::dx, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<3>(eval)[0]),
                             *(bspline3.eval<deriv::dx, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<4>(eval)[0]),
                             *(bspline4.eval<deriv::dx, false>(xi_)[0])));

    basfunc = functionspace.template eval_basfunc<functionspace::interior,
                                                  deriv::dx ^ 2, false>(
        xi, knot_indices);
    eval = functionspace.eval_from_precomputed<functionspace::interior>(
        basfunc, coeff_indices, numel(xi), sizes(xi));

    EXPECT_TRUE(torch::equal(*(std::get<0>(eval)[0]),
                             *(bspline0.eval<deriv::dx ^ 2, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<1>(eval)[0]),
                             *(bspline1.eval<deriv::dx ^ 2, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<2>(eval)[0]),
                             *(bspline2.eval<deriv::dx ^ 2, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<3>(eval)[0]),
                             *(bspline3.eval<deriv::dx ^ 2, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<4>(eval)[0]),
                             *(bspline4.eval<deriv::dx ^ 2, false>(xi_)[0])));

    basfunc =
        functionspace
            .template eval_basfunc<functionspace::interior, deriv::dy, false>(
                xi, knot_indices);
    eval = functionspace.eval_from_precomputed<functionspace::interior>(
        basfunc, coeff_indices, numel(xi), sizes(xi));

    EXPECT_TRUE(torch::equal(*(std::get<0>(eval)[0]),
                             *(bspline0.eval<deriv::dy, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<1>(eval)[0]),
                             *(bspline1.eval<deriv::dy, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<2>(eval)[0]),
                             *(bspline2.eval<deriv::dy, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<3>(eval)[0]),
                             *(bspline3.eval<deriv::dy, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<4>(eval)[0]),
                             *(bspline4.eval<deriv::dy, false>(xi_)[0])));

    basfunc = functionspace.template eval_basfunc<functionspace::interior,
                                                  deriv::dy ^ 2, false>(
        xi, knot_indices);
    eval = functionspace.eval_from_precomputed<functionspace::interior>(
        basfunc, coeff_indices, numel(xi), sizes(xi));

    EXPECT_TRUE(torch::equal(*(std::get<0>(eval)[0]),
                             *(bspline0.eval<deriv::dy ^ 2, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<1>(eval)[0]),
                             *(bspline1.eval<deriv::dy ^ 2, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<2>(eval)[0]),
                             *(bspline2.eval<deriv::dy ^ 2, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<3>(eval)[0]),
                             *(bspline3.eval<deriv::dy ^ 2, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<4>(eval)[0]),
                             *(bspline4.eval<deriv::dy ^ 2, false>(xi_)[0])));

    basfunc =
        functionspace
            .template eval_basfunc<functionspace::interior, deriv::dz, false>(
                xi, knot_indices);
    eval = functionspace.eval_from_precomputed<functionspace::interior>(
        basfunc, coeff_indices, numel(xi), sizes(xi));

    EXPECT_TRUE(torch::equal(*(std::get<0>(eval)[0]),
                             *(bspline0.eval<deriv::dz, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<1>(eval)[0]),
                             *(bspline1.eval<deriv::dz, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<2>(eval)[0]),
                             *(bspline2.eval<deriv::dz, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<3>(eval)[0]),
                             *(bspline3.eval<deriv::dz, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<4>(eval)[0]),
                             *(bspline4.eval<deriv::dz, false>(xi_)[0])));

    basfunc = functionspace.template eval_basfunc<functionspace::interior,
                                                  deriv::dz ^ 2, false>(
        xi, knot_indices);
    eval = functionspace.eval_from_precomputed<functionspace::interior>(
        basfunc, coeff_indices, numel(xi), sizes(xi));

    EXPECT_TRUE(torch::equal(*(std::get<0>(eval)[0]),
                             *(bspline0.eval<deriv::dz ^ 2, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<1>(eval)[0]),
                             *(bspline1.eval<deriv::dz ^ 2, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<2>(eval)[0]),
                             *(bspline2.eval<deriv::dz ^ 2, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<3>(eval)[0]),
                             *(bspline3.eval<deriv::dz ^ 2, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<4>(eval)[0]),
                             *(bspline4.eval<deriv::dz ^ 2, false>(xi_)[0])));

    basfunc =
        functionspace
            .template eval_basfunc<functionspace::interior, deriv::dt, false>(
                xi, knot_indices);
    eval = functionspace.eval_from_precomputed<functionspace::interior>(
        basfunc, coeff_indices, numel(xi), sizes(xi));

    EXPECT_TRUE(torch::equal(*(std::get<0>(eval)[0]),
                             *(bspline0.eval<deriv::dt, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<1>(eval)[0]),
                             *(bspline1.eval<deriv::dt, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<2>(eval)[0]),
                             *(bspline2.eval<deriv::dt, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<3>(eval)[0]),
                             *(bspline3.eval<deriv::dt, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<4>(eval)[0]),
                             *(bspline4.eval<deriv::dt, false>(xi_)[0])));

    basfunc = functionspace.template eval_basfunc<functionspace::interior,
                                                  deriv::dt ^ 2, false>(
        xi, knot_indices);
    eval = functionspace.eval_from_precomputed<functionspace::interior>(
        basfunc, coeff_indices, numel(xi), sizes(xi));

    EXPECT_TRUE(torch::equal(*(std::get<0>(eval)[0]),
                             *(bspline0.eval<deriv::dt ^ 2, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<1>(eval)[0]),
                             *(bspline1.eval<deriv::dt ^ 2, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<2>(eval)[0]),
                             *(bspline2.eval<deriv::dt ^ 2, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<3>(eval)[0]),
                             *(bspline3.eval<deriv::dt ^ 2, false>(xi_)[0])));
    EXPECT_TRUE(torch::equal(*(std::get<4>(eval)[0]),
                             *(bspline4.eval<deriv::dt ^ 2, false>(xi_)[0])));

    basfunc = functionspace.template eval_basfunc<functionspace::interior,
                                                  deriv::dx + deriv::dy, false>(
        xi, knot_indices);
    eval = functionspace.eval_from_precomputed<functionspace::interior>(
        basfunc, coeff_indices, numel(xi), sizes(xi));

    EXPECT_TRUE(
        torch::equal(*(std::get<0>(eval)[0]),
                     *(bspline0.eval<deriv::dx + deriv::dy, false>(xi_)[0])));
    EXPECT_TRUE(
        torch::equal(*(std::get<1>(eval)[0]),
                     *(bspline1.eval<deriv::dx + deriv::dy, false>(xi_)[0])));
    EXPECT_TRUE(
        torch::equal(*(std::get<2>(eval)[0]),
                     *(bspline2.eval<deriv::dx + deriv::dy, false>(xi_)[0])));
    EXPECT_TRUE(
        torch::equal(*(std::get<3>(eval)[0]),
                     *(bspline3.eval<deriv::dx + deriv::dy, false>(xi_)[0])));
    EXPECT_TRUE(
        torch::equal(*(std::get<4>(eval)[0]),
                     *(bspline4.eval<deriv::dx + deriv::dy, false>(xi_)[0])));

    basfunc = functionspace.template eval_basfunc<functionspace::interior,
                                                  deriv::dx + deriv::dz, false>(
        xi, knot_indices);
    eval = functionspace.eval_from_precomputed<functionspace::interior>(
        basfunc, coeff_indices, numel(xi), sizes(xi));

    EXPECT_TRUE(
        torch::equal(*(std::get<0>(eval)[0]),
                     *(bspline0.eval<deriv::dx + deriv::dz, false>(xi_)[0])));
    EXPECT_TRUE(
        torch::equal(*(std::get<1>(eval)[0]),
                     *(bspline1.eval<deriv::dx + deriv::dz, false>(xi_)[0])));
    EXPECT_TRUE(
        torch::equal(*(std::get<2>(eval)[0]),
                     *(bspline2.eval<deriv::dx + deriv::dz, false>(xi_)[0])));
    EXPECT_TRUE(
        torch::equal(*(std::get<3>(eval)[0]),
                     *(bspline3.eval<deriv::dx + deriv::dz, false>(xi_)[0])));
    EXPECT_TRUE(
        torch::equal(*(std::get<4>(eval)[0]),
                     *(bspline4.eval<deriv::dx + deriv::dz, false>(xi_)[0])));

    basfunc = functionspace.template eval_basfunc<functionspace::interior,
                                                  deriv::dx + deriv::dt, false>(
        xi, knot_indices);
    eval = functionspace.eval_from_precomputed<functionspace::interior>(
        basfunc, coeff_indices, numel(xi), sizes(xi));

    EXPECT_TRUE(
        torch::equal(*(std::get<0>(eval)[0]),
                     *(bspline0.eval<deriv::dx + deriv::dt, false>(xi_)[0])));
    EXPECT_TRUE(
        torch::equal(*(std::get<1>(eval)[0]),
                     *(bspline1.eval<deriv::dx + deriv::dt, false>(xi_)[0])));
    EXPECT_TRUE(
        torch::equal(*(std::get<2>(eval)[0]),
                     *(bspline2.eval<deriv::dx + deriv::dt, false>(xi_)[0])));
    EXPECT_TRUE(
        torch::equal(*(std::get<3>(eval)[0]),
                     *(bspline3.eval<deriv::dx + deriv::dt, false>(xi_)[0])));
    EXPECT_TRUE(
        torch::equal(*(std::get<4>(eval)[0]),
                     *(bspline4.eval<deriv::dx + deriv::dt, false>(xi_)[0])));

    basfunc = functionspace.template eval_basfunc<functionspace::interior,
                                                  deriv::dy + deriv::dz, false>(
        xi, knot_indices);
    eval = functionspace.eval_from_precomputed<functionspace::interior>(
        basfunc, coeff_indices, numel(xi), sizes(xi));

    EXPECT_TRUE(
        torch::equal(*(std::get<0>(eval)[0]),
                     *(bspline0.eval<deriv::dy + deriv::dz, false>(xi_)[0])));
    EXPECT_TRUE(
        torch::equal(*(std::get<1>(eval)[0]),
                     *(bspline1.eval<deriv::dy + deriv::dz, false>(xi_)[0])));
    EXPECT_TRUE(
        torch::equal(*(std::get<2>(eval)[0]),
                     *(bspline2.eval<deriv::dy + deriv::dz, false>(xi_)[0])));
    EXPECT_TRUE(
        torch::equal(*(std::get<3>(eval)[0]),
                     *(bspline3.eval<deriv::dy + deriv::dz, false>(xi_)[0])));
    EXPECT_TRUE(
        torch::equal(*(std::get<4>(eval)[0]),
                     *(bspline4.eval<deriv::dy + deriv::dz, false>(xi_)[0])));

    basfunc = functionspace.template eval_basfunc<functionspace::interior,
                                                  deriv::dy + deriv::dt, false>(
        xi, knot_indices);
    eval = functionspace.eval_from_precomputed<functionspace::interior>(
        basfunc, coeff_indices, numel(xi), sizes(xi));

    EXPECT_TRUE(
        torch::equal(*(std::get<0>(eval)[0]),
                     *(bspline0.eval<deriv::dy + deriv::dt, false>(xi_)[0])));
    EXPECT_TRUE(
        torch::equal(*(std::get<1>(eval)[0]),
                     *(bspline1.eval<deriv::dy + deriv::dt, false>(xi_)[0])));
    EXPECT_TRUE(
        torch::equal(*(std::get<2>(eval)[0]),
                     *(bspline2.eval<deriv::dy + deriv::dt, false>(xi_)[0])));
    EXPECT_TRUE(
        torch::equal(*(std::get<3>(eval)[0]),
                     *(bspline3.eval<deriv::dy + deriv::dt, false>(xi_)[0])));
    EXPECT_TRUE(
        torch::equal(*(std::get<4>(eval)[0]),
                     *(bspline4.eval<deriv::dy + deriv::dt, false>(xi_)[0])));

    basfunc = functionspace.template eval_basfunc<functionspace::interior,
                                                  deriv::dz + deriv::dt, false>(
        xi, knot_indices);
    eval = functionspace.eval_from_precomputed<functionspace::interior>(
        basfunc, coeff_indices, numel(xi), sizes(xi));

    EXPECT_TRUE(
        torch::equal(*(std::get<0>(eval)[0]),
                     *(bspline0.eval<deriv::dz + deriv::dt, false>(xi_)[0])));
    EXPECT_TRUE(
        torch::equal(*(std::get<1>(eval)[0]),
                     *(bspline1.eval<deriv::dz + deriv::dt, false>(xi_)[0])));
    EXPECT_TRUE(
        torch::equal(*(std::get<2>(eval)[0]),
                     *(bspline2.eval<deriv::dz + deriv::dt, false>(xi_)[0])));
    EXPECT_TRUE(
        torch::equal(*(std::get<3>(eval)[0]),
                     *(bspline3.eval<deriv::dz + deriv::dt, false>(xi_)[0])));
    EXPECT_TRUE(
        torch::equal(*(std::get<4>(eval)[0]),
                     *(bspline4.eval<deriv::dz + deriv::dt, false>(xi_)[0])));

    /// Evaluation of gradient (in parametric domain)
    auto grad = functionspace.template grad<functionspace::interior, false>(xi);
    auto grad_ref0 = bspline0.grad(xi_);
    auto grad_ref1 = bspline1.grad(xi_);
    auto grad_ref2 = bspline2.grad(xi_);
    auto grad_ref3 = bspline3.grad(xi_);
    auto grad_ref4 = bspline4.grad(xi_);

    for (std::size_t i = 0; i < grad_ref0.entries(); ++i) {
      EXPECT_TRUE(torch::equal(*(std::get<0>(grad)[i]), *(grad_ref0)[i]));
      EXPECT_TRUE(torch::equal(*(std::get<1>(grad)[i]), *(grad_ref1)[i]));
      EXPECT_TRUE(torch::equal(*(std::get<2>(grad)[i]), *(grad_ref2)[i]));
      EXPECT_TRUE(torch::equal(*(std::get<3>(grad)[i]), *(grad_ref3)[i]));
      EXPECT_TRUE(torch::equal(*(std::get<4>(grad)[i]), *(grad_ref4)[i]));
    }

    /// Evaluation of gradient (in physical domain)
    auto igrad = functionspace.template igrad<functionspace::interior, false>(
        RT4_geometry, xi);
    auto igrad_ref0 = bspline0.igrad(geometry0, xi_);
    auto igrad_ref1 = bspline1.igrad(geometry1, xi_);
    auto igrad_ref2 = bspline2.igrad(geometry2, xi_);
    auto igrad_ref3 = bspline3.igrad(geometry3, xi_);
    auto igrad_ref4 = bspline4.igrad(geometry4, xi_);

    for (std::size_t i = 0; i < igrad_ref0.entries(); ++i) {
      EXPECT_TRUE(torch::equal(*(std::get<0>(igrad)[i]), *(igrad_ref0)[i]));
      EXPECT_TRUE(torch::equal(*(std::get<1>(igrad)[i]), *(igrad_ref1)[i]));
      EXPECT_TRUE(torch::equal(*(std::get<2>(igrad)[i]), *(igrad_ref2)[i]));
      EXPECT_TRUE(torch::equal(*(std::get<3>(igrad)[i]), *(igrad_ref3)[i]));
      EXPECT_TRUE(torch::equal(*(std::get<4>(igrad)[i]), *(igrad_ref4)[i]));
    }

    /// Evaluation of Jacobian (in parametric domain)
    auto jac = functionspace.template jac<functionspace::interior, false>(xi);
    auto jac_ref0 = bspline0.jac(xi_);
    auto jac_ref1 = bspline1.jac(xi_);
    auto jac_ref2 = bspline2.jac(xi_);
    auto jac_ref3 = bspline3.jac(xi_);
    auto jac_ref4 = bspline4.jac(xi_);

    for (std::size_t i = 0; i < jac_ref0.entries(); ++i) {
      EXPECT_TRUE(torch::equal(*(std::get<0>(jac)[i]), *(jac_ref0)[i]));
      EXPECT_TRUE(torch::equal(*(std::get<1>(jac)[i]), *(jac_ref1)[i]));
      EXPECT_TRUE(torch::equal(*(std::get<2>(jac)[i]), *(jac_ref2)[i]));
      EXPECT_TRUE(torch::equal(*(std::get<3>(jac)[i]), *(jac_ref3)[i]));
      EXPECT_TRUE(torch::equal(*(std::get<4>(jac)[i]), *(jac_ref4)[i]));
    }

    /// Evaluation of Jacobian (in physical domain)
    auto ijac = functionspace.template ijac<functionspace::interior, false>(
        RT4_geometry, xi);
    auto ijac_ref0 = bspline0.ijac(geometry0, xi_);
    auto ijac_ref1 = bspline1.ijac(geometry1, xi_);
    auto ijac_ref2 = bspline2.ijac(geometry2, xi_);
    auto ijac_ref3 = bspline3.ijac(geometry3, xi_);
    auto ijac_ref4 = bspline4.ijac(geometry4, xi_);

    for (std::size_t i = 0; i < ijac_ref0.entries(); ++i) {
      EXPECT_TRUE(torch::equal(*(std::get<0>(ijac)[i]), *(ijac_ref0)[i]));
      EXPECT_TRUE(torch::equal(*(std::get<1>(ijac)[i]), *(ijac_ref1)[i]));
      EXPECT_TRUE(torch::equal(*(std::get<2>(ijac)[i]), *(ijac_ref2)[i]));
      EXPECT_TRUE(torch::equal(*(std::get<3>(ijac)[i]), *(ijac_ref3)[i]));
      EXPECT_TRUE(torch::equal(*(std::get<4>(ijac)[i]), *(ijac_ref4)[i]));
    }

    /// Evaluation of Hessian (in parametric domain)
    auto hess = functionspace.template hess<functionspace::interior, false>(xi);
    auto hess_ref0 = bspline0.hess(xi_);
    auto hess_ref1 = bspline1.hess(xi_);
    auto hess_ref2 = bspline2.hess(xi_);
    auto hess_ref3 = bspline3.hess(xi_);
    auto hess_ref4 = bspline4.hess(xi_);

    for (std::size_t i = 0; i < hess_ref0.entries(); ++i) {
      EXPECT_TRUE(torch::equal(*(std::get<0>(hess)[i]), *(hess_ref0)[i]));
      EXPECT_TRUE(torch::equal(*(std::get<1>(hess)[i]), *(hess_ref1)[i]));
      EXPECT_TRUE(torch::equal(*(std::get<2>(hess)[i]), *(hess_ref2)[i]));
      EXPECT_TRUE(torch::equal(*(std::get<3>(hess)[i]), *(hess_ref3)[i]));
      EXPECT_TRUE(torch::equal(*(std::get<4>(hess)[i]), *(hess_ref4)[i]));
    }

    /// Evaluation of Hessian (in physical domain)
    auto ihess = functionspace.template ihess<functionspace::interior, false>(
        RT4_geometry, xi);
    auto ihess_ref0 = bspline0.ihess(geometry0, xi_);
    auto ihess_ref1 = bspline1.ihess(geometry1, xi_);
    auto ihess_ref2 = bspline2.ihess(geometry2, xi_);
    auto ihess_ref3 = bspline3.ihess(geometry3, xi_);
    auto ihess_ref4 = bspline4.ihess(geometry4, xi_);

    for (std::size_t i = 0; i < ihess_ref0.entries(); ++i) {
      EXPECT_TRUE(torch::equal(*(std::get<0>(ihess)[i]), *(ihess_ref0)[i]));
      EXPECT_TRUE(torch::equal(*(std::get<1>(ihess)[i]), *(ihess_ref1)[i]));
      EXPECT_TRUE(torch::equal(*(std::get<2>(ihess)[i]), *(ihess_ref2)[i]));
      EXPECT_TRUE(torch::equal(*(std::get<3>(ihess)[i]), *(ihess_ref3)[i]));
      EXPECT_TRUE(torch::equal(*(std::get<4>(ihess)[i]), *(ihess_ref4)[i]));
    }
  }
}

TEST_F(FunctionSpaceTest, FunctionSpace_init) {}

TEST_F(FunctionSpaceTest, FunctionSpace_refine) {}

TEST_F(FunctionSpaceTest, FunctionSpace_copy_constructor) {}

TEST_F(FunctionSpaceTest, FunctionSpace_clone_constructor) {}

TEST_F(FunctionSpaceTest, FunctionSpace_move_constructor) {}

TEST_F(FunctionSpaceTest, FunctionSpace_read_write) {}

TEST_F(FunctionSpaceTest, FunctionSpace_to_from_xml) {}

TEST_F(FunctionSpaceTest, FunctionSpace_load_from_xml) {}

TEST_F(FunctionSpaceTest, FunctionSpace_to_from_json) {}

TEST_F(FunctionSpaceTest, FunctionSpace_query_property) {}

TEST_F(FunctionSpaceTest, FunctionSpace_requires_grad) {}

TEST_F(FunctionSpaceTest, FunctionSpace_to_dtype) {}

TEST_F(FunctionSpaceTest, FunctionSpace_to_device) {}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  iganet::init();
  return RUN_ALL_TESTS();
}
