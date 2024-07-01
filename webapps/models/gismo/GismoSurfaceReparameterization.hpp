/**
   @file webapps/models/gismo/GismoSurfaceReparameterization.hpp

   @brief G+Smo surface reparameterization tools

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <iganet.h>

namespace iganet {

/// @brief Computes the Mobius transformation
template <typename T>
inline void mobiusTransform(const gsAsConstVector<T> &c,
                            const gsVector<T, 2> &uv, gsVector<T, 2> &xieta,
                            gsMatrix<T, 2, 2> &jac) {
  T s = uv(0), t = uv(1);
  T alpha_1 = c(0), alpha_2 = c(1), beta_1 = c(2), beta_2 = c(3);

  T alpha = alpha_1 * t + alpha_2 * (1 - t);
  T beta = beta_1 * s + beta_2 * (1 - s);

  T xi_denominator = 2 * alpha * s - s - alpha;
  xieta(0) = (alpha - 1) * s / xi_denominator;
  T eta_denominator = 2 * beta * t - t - beta;
  xieta(1) = (beta - 1) * t / eta_denominator;

  // jac = [dxids, dxidt; detads, detadt]
  jac(0, 0) = (alpha - 1) * (xi_denominator - (2 * alpha - 1) * s) /
              (xi_denominator * xi_denominator);
  jac(0, 1) = (alpha_1 - alpha_2) * s *
              (xi_denominator - (alpha - 1) * (2 * s - 1)) /
              (xi_denominator * xi_denominator);
  jac(1, 0) = (beta_1 - beta_2) * t *
              (eta_denominator - (beta - 1) * (2 * t - 1)) /
              (eta_denominator * eta_denominator);
  jac(1, 1) = (beta - 1) * (eta_denominator - (2 * beta - 1) * t) /
              (eta_denominator * eta_denominator);
}

/// @brief Objective function for surface reparameterization
template <typename T> class gsObjFuncSurface : public gsOptProblem<T> {

private:
  typedef typename gismo::gsExprAssembler<T>::geometryMap geometryMap;
  typedef typename gismo::gsExprAssembler<T>::space space;
  typedef typename gismo::gsExprAssembler<T>::solution solution;

public:
  /// @brief Constructor
  explicit gsObjFuncSurface(const gismo::gsMultiPatch<T> &patches,
                            const gismo::gsMobiusDomain<2, T> &mobiusDomain)
      : m_mp(patches), m_MobiusDomain(mobiusDomain) {
    defaultOptions();

    gismo::gsMatrix<T> bbox;
    m_mp.boundingBox(bbox);
    m_mp.patch(0).translate(-bbox.col(0));
    gsVector<T> scaleFactor = bbox.col(1) - bbox.col(0);
    for (int i = 0; i != scaleFactor.size(); ++i) {
      if (abs(scaleFactor(i)) < 1e-5)
        scaleFactor(i) = (T)(1.0);
    }
    m_mp.patch(0).scale(1 / scaleFactor.array());

    //    const gsBasis<T> & tbasis = m_mp.basis(0); // basis(u,v) -> deriv will
    //    give dphi/du ,dphi/dv const gsGeometry<T> & tgeom = m_mp.patch(0);
    //    //G(u,v) -> deriv will give dG/du, dG/dv
    //    // The basis is composed by the square domain
    //    gsComposedBasis<T> cbasis(mobiusDomain, tbasis); // basis(u,v) =
    //    basis(sigma(xi,eta)) -> deriv will give dphi/dxi, dphi/deta
    //
    //    gsMultiBasis<T> mbasis(cbasis);

    gismo::gsComposedGeometry<T> cgeom(m_MobiusDomain, m_mp.patch(0));

    gismo::gsMultiBasis<T> dbasis(cgeom.basis());
    //    m_assembler.setIntegrationElements(dbasis);
    m_evaluator.setIntegrationElements(dbasis);

    // Set the geometry map
    geometryMap G = m_evaluator.getMap(cgeom);
    m_area = m_evaluator.integral(meas(G));

    //    m_evaluator.setIntegrationElements( mbasis );
    //    geometryMap G = m_evaluator.getMap(  );
    //    gsDebugVar( m_evaluator.integral(meas(G)) );
  }

  /// @brief Evaluates the objective function
  T evalObj(const gismo::gsAsConstVector<T> &coefsM) const override {

    m_MobiusDomain.updateGeom(coefsM);

    //  gsComposedGeometry<real_t> cgeom(mobiusDomain, tgeom); // G(u,v) =
    //  G(sigma(xi,eta))  -> deriv will give dG/dxi, dG/deta
    gismo::gsComposedGeometry<> cgeom(m_MobiusDomain, m_mp.patch(0));

    gismo::gsMultiBasis<> dbasis(cgeom.basis());
    //    m_assembler.setIntegrationElements(dbasis);
    m_evaluator.setIntegrationElements(dbasis);

    // Set the geometry map
    geometryMap G = m_evaluator.getMap(cgeom);
    auto FFF = gismo::expr::jac(G).tr() * gismo::expr::jac(G);

    gsVector<> pt(2);
    pt.setConstant(0.5);

    auto m_integration = (FFF.trace() / gismo::expr::meas(G)).val() +
                         pow(FFF.det().val(), 2) / pow(m_area, 2);
    //  auto m_integration = (FFF.trace() / meas(G)).val() + 1e-4 *
    //  FFF.det().val(); auto m_integration = pow(FFF.det().val(), 2); T val =
    //  m_evaluator.integral(m_integration);

    //  gsMatrix<T> uv;
    //  getSamplingPts(51, m_mp, uv);
    ////  gsDebugVar(uv);
    ////  T val = (m_evaluator.eval(m_integration, uv))/51;
    ////  auto val1 = m_evaluator.eval(m_integration, uv);
    //  T val1 = 0;
    //  for (int i = 0; i < uv.cols(); ++i) {
    //    val1 += m_evaluator.eval(m_integration, uv.col(i)).value();
    //  }
    ////  gsDebugVar( m_evaluator.eval(m_integration, uv.col(88)).value() );
    //  //  T val = val1.sum()/51;
    //  T val = val1/51/51;

    //  gsDebugVar(val);

    //  m_evaluator.integral(meas(G));
    //  T val = 0;
    //
    //  gsMatrix<T> uv;
    //  getSamplingPts(51, m_mp, uv);
    //
    //  gsVector<T,2> tempUV, xieta;
    //  gsMatrix<T,2,2> jacUV, jac2;
    //  gsMatrix<T,2,3> jacXIETA, jac;
    //  for (auto ipt=0;  ipt != uv.cols(); ++ipt) {
    //    // map sigma
    //    tempUV = uv.col(ipt);
    //    mobiusTransform(coefsM, tempUV, xieta, jacUV);
    //
    //    // map surface
    //    auto basisDerivs =
    //    m_mp.basis(0).collocationMatrixWithDeriv(m_mp.basis(0), xieta);
    //
    //    jacXIETA.row(0) = basisDerivs[1] * m_mp.patch(0).coefs();
    //    jacXIETA.row(1) = basisDerivs[2] * m_mp.patch(0).coefs();
    //
    //    jac = jacUV.transpose() * jacXIETA;
    //    jac2 = jac * jac.transpose();
    //
    //    // TODO: fix this balance parameter 1e-2
    //    T detJac = jac2.determinant();
    //    val += (jac2.trace())/(sqrt(jac2.determinant())+1e-8) + 1e-4*detJac;
    //  }
    //  val /= uv.cols();

    return m_evaluator.integral(m_integration);
  }

  /// @brief Evaluates the gradient of the objective function
  void gradObj_into(const gismo::gsAsConstVector<T> &u,
                    gismo::gsAsVector<T> &result) const override {

    const std::size_t n = u.rows();
    // GISMO_ASSERT((index_t)m_numDesignVars == n*m, "Wrong design.");

    gismo::gsMatrix<T> uu = u; // copy
    gismo::gsAsVector<T> tmp(uu.data(), n);
    gismo::gsAsConstVector<T> ctmp(uu.data(), n);
    std::size_t c = 0;

    //  const T e0 = this->evalObj(ctmp);
    // for all partial derivatives (column-wise)
    for (std::size_t i = 0; i != n; i++) {
      tmp[i] += T(1e-6);
      const T e1 = this->evalObj(ctmp);
      tmp[i] = u[i] - T(1e-6);
      const T e2 = this->evalObj(ctmp);
      tmp[i] = u[i];
      result[c++] = ((e1 - e2)) / T(2e-6);
    }
  }

  /// @brief Sets the tolerance
  void setEps(T tol) { m_eps = tol; }

  /// @brief Returns a reference to the option list
  gismo::gsOptionList &options() { return m_options; }

  /// @brief Sets the default options
  void defaultOptions() {
    // @Ye, make this reasonable default options
    m_options.addReal("qi_lambda1", "Sets the lambda 1 value", 1.0);
    m_options.addReal("qi_lambda2", "Sets the lambda 2 value", 1.0);
  }

  /// @brief Adds an option to the option list
  void addOptions(const gismo::gsOptionList &options) {
    m_options.update(options, gsOptionList::addIfUnknown);
  }

  /// @brief Applies an option list
  void applyOptions(const gismo::gsOptionList &options) {
    m_options.update(options, gsOptionList::addIfUnknown);
    m_lambda1 = m_options.getReal("qi_lambda1");
    m_lambda2 = m_options.getReal("qi_lambda2");
    m_evaluator.options().update(m_options, gismo::gsOptionList::addIfUnknown);
  }

protected:
  const gismo::gsMultiPatch<T> m_mp;
  const gismo::gsDofMapper m_mapper;
  const gismo::gsMultiBasis<T> m_mb;

  mutable gismo::gsExprEvaluator<T> m_evaluator;
  mutable gismo::gsExprAssembler<T> m_assembler;

  gismo::gsOptionList m_options;

  T m_lambda1 = 1.0, m_lambda2 = 1.0, m_eps = 1e-3;
  T m_area = 1;

  gismo::gsComposedGeometry<T> m_cgeom;
  mutable gismo::gsMobiusDomain<2, T> m_MobiusDomain;
};

/// @brief Converts a matrix of coefficients into a multi-patch B-spline object
template <typename T>
gsMultiPatch<T> convertIntoBSpline(const gsMultiPatch<T> &mp,
                                   const gsMatrix<T> &coefsMobiusIn) {
  gsMultiPatch<T> result;

  for (int ipatch = 0; ipatch < mp.nPatches(); ++ipatch) {

    gsMatrix<T> uv =
        gsPointGrid(mp.parameterRange(0), mp.patch(ipatch).basis().size() * 4);

    gsVector<T, 2> tempUV, xieta;
    gsMatrix<T> eval_geo;
    eval_geo.resize(3, uv.cols());
    gsMatrix<T, 2, 2> jacUV;

    gsAsConstVector<T> coefsMobius(coefsMobiusIn.data(), 4);

    for (size_t ipt = 0; ipt != uv.cols(); ++ipt) {
      // map sigma
      tempUV = uv.col(ipt);
      mobiusTransform(coefsMobius, tempUV, xieta, jacUV);

      // map surface
      eval_geo.col(ipt) = mp.patch(ipatch).eval(xieta);
    }

    gsTensorBSplineBasis<2, T> bbasis =
        static_cast<gsTensorBSplineBasis<2, T> &>(mp.patch(ipatch).basis());
    gsFitting<> fittingSurface(uv, eval_geo, bbasis);
    fittingSurface.compute();
    // fittingSurface.parameterCorrection();

    result.addPatch(*fittingSurface.result());
  }

  result.computeTopology();

  return result;
}

} // namespace iganet
