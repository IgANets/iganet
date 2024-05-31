# IgANets - Isogeometric Analysis Networks

**IgANets** (Isogeometric Analysis Networks) is a C++ library that
combines Isogeometric Analysis with deep operator learning. It builds
upon the C++ API of the Torch library and is written in C++20. The
library aims to provide an easy to use, user-friendly and yet
computationally efficient framework for implementing IgANet
applications.

## Mathematical notation

### B-Spline basis functions

- \f$ \xi_d \f$ is the value of the parametric coordinate in the \f$ d \f$-th parametric dimension
- \f$ \boldsymbol{\xi} = \left( \xi_1, \dots, \xi_{d_\text{par}} \right)^\top \f$ is the vector of parametric coordinates in all parametric dimension
- \f$ B_{i_d,p_d}(\xi_d) \f$ is the \f$ i_d \f$-th univariate B-spline basis function in the \f$ d \f$-th parametric dimension evaluated at \f$ \xi_d \f$
- \f$ B_I(\boldsymbol{\xi}) = \bigotimes_{d=1}^{d_\text{par}} B_{i_d,p_d}(\xi_d) \f$ is the \f$ I \f$-th multivariate B-spline basis function
- \f$ d_\text{geo} \f$ is the total number of geometric dimension
- \f$ d_\text{par} \f$ is the total number of parametric dimension
- \f$ i_d \f$ is local index refering to the \f$ d \f$-th dimension
- \f$ \mathbf{i} = \left(i_1, \dots, i_d \right) \f$ is a local multi-index
- \f$ I \f$ is a global index
- \f$ n_d \f$ is the number of univariate B-spline basis functions in the \f$ d \f$-th dimension
- \f$ N = n_1\cdot \dots \cdot n_{d_\text{par}} \f$ is the total number of multivariable B-splines basis functions

### B-Spline function spaces

- \f$ S^{p}_{\boldsymbol{\alpha}} = \text{span} \left\{ B_{i,p} \right\}_{i=1}^n \f$ is the function space that is spanned by a univariate B-spline basis of degree \f$ p \f$ and regularity vector \f$ \boldsymbol{\alpha} = \left(\alpha_1, \dots, \alpha_n\right) \f$. By default, we assume maximal regularity, i.e. \f$ \alpha_i = p-1 \f$ for all \f$ i = 1, \dots, n \f$
- \f$ S^{p_1,\dots,p_{d_\text{par}}}_{\boldsymbol{\alpha}_1,\dots,\boldsymbol{\alpha}_{d_\text{par}}} \f$ is the function space that is spanned by a multivariate B-spline basis of degree \f$ \mathbf{p} = \left(p_1, \dots, p_{d_\text{par}}\right) \f$ with regularity vectors \f$ \boldsymbol{\alpha}_d \f$ for each parametric dimension \f$ d = 1, \dots d_\text{par} \f$. By default, we assume maximal regularity (see above)

## Documentation

- The list of [Examples](../examples/README.md)
- The list of [PerfTests](../perftests/README.md)
- The list of [UnitTests](../unittests/README.md)
- The list of [WebApps](../webapps/README.md)

## Copyright

Copyright (c) 2021-2024 Matthias Möller (m.moller@tudelft.nl).
