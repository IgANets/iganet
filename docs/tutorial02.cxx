//! [Include namespace]
#include <iganet.h>

using namespace iganet;
//! [Include namespace]

int main() {
  //! [Initialize internals]
  init();
  //! [Initialize internals]

  //! [Univariate uniform B-spline of degree 2]
  UniformBSpline<double, 1, 2> C({6});
  //! [Univariate uniform B-spline of degree 2]

  //! [Print Greville points of univariate uniform B-spline of degree 2]
  Log() << "\n\nGreville points\n" << C.greville() << "\n";
  //! [Print Greville points of univariate uniform B-spline of degree 2]

  //! [Print univariate uniform B-spline of degree 2]
  Log() << "\n\nB-spline curve\n" << C << "\n";
  //! [Print univariate uniform B-spline of degree 2]

  //! [Print verbose univariate uniform B-spline of degree 2]
  verbose(std::cout);
  Log() << "\n\nB-spline curve (verbose)\n" << C << "\n";
  //! [Print verbose univariate uniform B-spline of degree 2]

  //! [Print knots of univariate uniform B-spline of degree 2]
  Log() << "\n\nNumber of knots\n"
        << C.nknots() << "\nKnot vector\n"
        << C.knots() << "\n";
  //! [Print knots of univariate uniform B-spline of degree 2]

  //! [Print knots of univariate uniform B-spline of degree 2 per dimension]
  Log() << "\n\nNumber of knots in 0-th dimension\n"
        << C.nknots(0) << "\nKnot vector in 0-th dimension\n"
        << C.knots(0) << "\n";
  //! [Print knots of univariate uniform B-spline of degree 2 per dimension]

  //! [Multivariate uniform B-splines]
  UniformBSpline<double, 3, 2, 3> S({6, 8});
  UniformBSpline<double, 3, 2, 3, 2> V({6, 8, 5});
  //! [Multivariate uniform B-splines]

  //! [Print multivariate uniform B-splines]
  Log() << "\n\nB-spline surface\n"
        << S << "\n\nB-spline volume\n"
        << V << "\n";
  //! [Print multivariate uniform B-splines]

  //! [Multivariate uniform B-splines linear vs linspace]
  UniformBSpline<double, 3, 2, 3> S_linear({5, 5}, init::linear);
  UniformBSpline<double, 3, 2, 3> S_linspace({5, 5}, init::linspace);

  Log() << "\n\nB-spline surface with linear initialization of the control "
           "points\n"
        << S_linear.coeffs_view() << "\n";
  Log() << "\n\nB-spline surface with linspace initialization of the control "
           "points\n"
        << S_linspace.coeffs_view() << "\n";
  //! [Multivariate uniform B-splines linear vs linspace]

  //! [Multivariate uniform B-spline with externally defined control points]
  utils::TensorArray<2> coeffs = utils::to_tensorArray(
      {0.0000, 0.1667, 0.5000, 0.8333, 1.0000, 0.0000, 0.1667, 0.5000, 0.8333,
       1.0000, 0.0000, 0.1667, 0.5000, 0.8333, 1.0000, 0.0000, 0.1667, 0.5000,
       0.8333, 1.0000, 0.0000, 0.1667, 0.5000, 0.8333, 1.0000},
      {0.0000, 0.1667, 0.5000, 0.8333, 1.0000, 0.0000, 0.1667, 0.5000, 0.8333,
       1.0000, 0.0000, 0.1667, 0.5000, 0.8333, 1.0000, 0.0000, 0.1667, 0.5000,
       0.8333, 1.0000, 0.0000, 0.1667, 0.5000, 0.8333, 1.0000});

  UniformBSpline<double, 2, 2, 3> S_cpts({5, 5}, coeffs);

  Log() << "\n\nB-spline surface with externally defined control point "
           "coordinates\n"
        << S_cpts << "\n";
  //! [Multivariate uniform B-spline with externally defined control points]

  //! [Multivariate uniform B-spline with externally defined control points updated]
  coeffs[0][3] = 10;

  Log() << "\n\nUpdate control point coordinates" << S_cpts.coeffs_view()
        << "\n";
  //! [Multivariate uniform B-spline with externally defined control points updated]

  //! [Multivariate uniform B-spline with externally defined and cloned control points]
  UniformBSpline<double, 2, 2, 3> S_cpts_cloned({5, 5}, coeffs, true);
  //! [Multivariate uniform B-spline with externally defined and cloned control points]

  //! [Duplicating multivariate uniform B-spline]
  auto S_soft_copy(S);
  auto S_deep_copy(S.clone());

  Log() << "\n\nS == S_soft_copy: " << (S == S_soft_copy)
        << "\n\nS == S_deep_copy: " << (S == S_deep_copy) << "\n";
  //! [Duplicating multivariate uniform B-spline]

  //! [Clean up internals]
  finalize();
  //! [Clean up internals]

  return 0;
}
