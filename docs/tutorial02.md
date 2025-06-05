# Tutorial 02: Working with B-spline functions

## TL;DR

In this tutorial you will learn how to create and manipulate B-spline
function objects and to evaluate them and their derivatives.

## Creating B-spline function objects

### Univariate uniform B-spline function objects

Let us start by creating a *univariate uniform B-spline function* of
degree \f$ p=2 \f$ with 6 control points in \f$ \mathbb{R}^1 \f$

\snippet tutorial02.cxx Univariate uniform B-spline of degree 2

That is, a function \f$ C : [0,1] \to \mathbb{R}^1 \f$ that is defined
through its 6 control points \f$ x_k \in \mathbb{R}^1, k=0, \dots, 5
\f$, i.e.

\f[
  C(\xi) = \sum_{i=0}^5 = x_k B_{i,2}(\xi)
\f]

Here, univariate refers to the single variable \f$ \xi \f$ and uniform
means that the knot vector \f$ \Xi = [0, 0, 0, 0.25, 0.5, 0.75, 1, 1,
1] \f$ is uniform.

By default, the control points are initialized to the so-called
*Greville abscissae*

\f[
  x_k^* = \frac{1}{p-1}\left(\xi_{i+1} + \cdots + \xi_{i+p-1} \right), \quad k=0, \dots, 5
\f]

which can be obtained using the `greville()` function

\snippet tutorial02.cxx Print Greville points of univariate uniform B-spline of degree 2

<details>
  <summary>Click to see output</summary>

```
[INFO]

Greville points
std::__1::array<at::Tensor, 1ul>(
 0.0000
 0.1250
 0.3750
 0.6250
 0.8750
 1.0000
[ CPUDoubleType{6} ]
)
```

</details>

All IgANets object have the member function `pretty_print(std::ostream
&os)` and an overload of the `operator<<` so that the curve \f$ C \f$
can be printed as follows

\snippet tutorial02.cxx Print univariate uniform B-spline of degree 2

<details>
  <summary>Click to see output</summary>

```
[INFO]

B-spline curve
iganet::BSplineCommon<iganet::UniformBSplineCore<double, (short)1, (short)2>>(
parDim = 1, geoDim = 1, degrees = 2, knots = 9, coeffs = 6, options = TensorOptions(dtype=double, device=cpu, layout=Strided (default), requires_grad=false (default), pinned_memory=false (default), memory_format=(nullopt))
)
```

</details>

Here

- `parDim = 1` is for the dimension of the parameter domain \f$
[0,1]^1 \f$,

- `geoDim = 1` is the dimension of the geometric domain \f$ C \subset
\mathbb{R}^1 \f$,

- `degrees = 2` is the degree of the B-spline function,

- `knots = 9` is the number of knots in the knot vector \f$ \Xi \f$, and

- `coeffs = 6` is the number of control points \f$ c_k \f$.

The general relation for an open knot vector is `knots = coeffs +
degrees + 1`.

A more verbose output that also prints the knot and control point
vectors can be produced by

\snippet tutorial02.cxx Print verbose univariate uniform B-spline of degree 2

<details>
  <summary>Click to see output</summary>

```
[INFO]

B-spline curve (verbose)
iganet::BSplineCommon<iganet::UniformBSplineCore<double, (short)1, (short)2>>(
parDim = 1, geoDim = 1, degrees = 2, knots = 9, coeffs = 6, options = TensorOptions(dtype=double, device=cpu, layout=Strided (default), requires_grad=false (default), pinned_memory=false (default), memory_format=(nullopt))
knots [ owns/cont ] = std::__1::array<at::Tensor, 1ul>(
 0.0000
 0.0000
 0.0000
 0.2500
 0.5000
 0.7500
 1.0000
 1.0000
 1.0000
[ CPUDoubleType{9} ]
[ TensorOptions(dtype=double, device=cpu, layout=Strided, requires_grad=false (default), pinned_memory=false (default), memory_format=(nullopt)) ]
)
coeffs [ owns/cont ] = std::__1::array<at::Tensor, 1ul>(
 0.0000
 0.1250
 0.3750
 0.6250
 0.8750
 1.0000
[ CPUDoubleType{6} ]
[ TensorOptions(dtype=double, device=cpu, layout=Strided, requires_grad=false (default), pinned_memory=false (default), memory_format=(nullopt)) ]
)
)
```

</details>

Here, `[ owns/cont ]` indicates that the underlying data is owned by
the tensor container and stored contiguously in memory. This output is
determined by Torch's `is_view()` and `is_contiguous()` function.

All properties of the B-spline function object can also be addressed
individually, e.g.

\snippet tutorial02.cxx Print knots of univariate uniform B-spline of degree 2

<details>
  <summary>Click to see output</summary>

```
[INFO]

Number of knots
std::__1::array<long long, 1ul>(9)
Knot vector
std::__1::array<at::Tensor, 1ul>(
 0.0000
 0.0000
 0.0000
 0.2500
 0.5000
 0.7500
 1.0000
 1.0000
 1.0000
[ CPUDoubleType{9} ]
[ TensorOptions(dtype=double, device=cpu, layout=Strided, requires_grad=false (default), pinned_memory=false (default), memory_format=(nullopt)) ]
)
```

</details>

Note that IgANets has overload of the `operator<<` for the standard
containers `std::array` and `std::vector`. If you want to address
individual properties restricted to a single parametric or geometric
dimension type

\snippet tutorial02.cxx Print knots of univariate uniform B-spline of degree 2 per dimension

<details>
  <summary>Click to see output</summary>

```
[INFO]

Number of knots in 0-th dimension
9
Knot vector in 0-th dimension
 0.0000
 0.0000
 0.0000
 0.2500
 0.5000
 0.7500
 1.0000
 1.0000
 1.0000
[ CPUDoubleType{9} ]
```

</details>

The following properties of the B-spline function object can be access
individually

- `coeffs()` returns the array of control points for all dimensions as flat tensor
- `coeffs(i)` returns the control points of the `i`-th dimension as flat tensor
- `coeffs_view()` returns the array of control points for all dimensions as multidimensional tensor view
- `coeffs_view(i)` returns the control points of the `i`-th dimension as multidimensional tensor view
- `degree(i)` returns the degree of the `i`-th dimension
- `degrees()` returns the array of degrees for all dimensions
- `geoDim()` returns the geometric dimension
- `knots()` returns the array of knot vectors for all dimensions
- `knots(i)` returns the knot vector of the `i`-th dimension
- `ncoeffs()` returns the numer of control points for all dimensions
- `ncoeffs(i)` returns the number of control points of the `i`-th dimension
- `ncumcoeffs()` returns the total number of control points
- `nknots()` returns the number of knots for all dimension
- `nknots(i)` returns the number of knots of the `i`-th dimension
- `parDim()` returns the parametric dimension

### Multivariate uniform B-spline function objects

Like in the univariate case, a multivariate uniform B-spline function
object is created by specifying the geometric dimension in the first
template parameter followed by a sequence of integers specifying the
individual degrees from which the parametric dimension is deduced.

As an example, the following code snipped creates

- a *bivariate B-spline surface* \f$ S : [0,1]^2 \to \mathbb{R}^3 \f$
with \f$ 6 \times 8 \f$ control points and

- a *trivariate B-spline volume* \f$ V : [0,1]^3 \to \mathbb{R}^3 \f$
with \f$ 6 \times 8 \times 5 \f$ control points

\snippet tutorial02.cxx Multivariate uniform B-splines

\snippet tutorial02.cxx Print multivariate uniform B-splines

<details>
  <summary>Click to see output</summary>

```
[INFO]

B-spline surface
iganet::BSplineCommon<iganet::UniformBSplineCore<double, (short)3, (short)2, (short)3>>(
parDim = 2, geoDim = 3, degrees = 2x3, knots = 9x12, coeffs = 6x8, options = TensorOptions(dtype=double, device=cpu, layout=Strided (default), requires_grad=false (default), pinned_memory=false (default), memory_format=(nullopt))
knots [ owns/cont owns/cont ] = std::__1::array<at::Tensor, 2ul>(
 0.0000
 0.0000
 0.0000
 0.2500
 0.5000
 0.7500
 1.0000
 1.0000
 1.0000
[ CPUDoubleType{9} ]
[ TensorOptions(dtype=double, device=cpu, layout=Strided, requires_grad=false (default), pinned_memory=false (default), memory_format=(nullopt)) ]
 0.0000
 0.0000
 0.0000
 0.0000
 0.2000
 0.4000
 0.6000
 0.8000
 1.0000
 1.0000
 1.0000
 1.0000
[ CPUDoubleType{12} ]
[ TensorOptions(dtype=double, device=cpu, layout=Strided, requires_grad=false (default), pinned_memory=false (default), memory_format=(nullopt)) ]
)
coeffs [ owns/cont owns/cont owns/cont ] = std::__1::array<at::Tensor, 3ul>(
 0.0000  0.1250  0.3750  0.6250  0.8750  1.0000
 0.0000  0.1250  0.3750  0.6250  0.8750  1.0000
 0.0000  0.1250  0.3750  0.6250  0.8750  1.0000
 0.0000  0.1250  0.3750  0.6250  0.8750  1.0000
 0.0000  0.1250  0.3750  0.6250  0.8750  1.0000
 0.0000  0.1250  0.3750  0.6250  0.8750  1.0000
 0.0000  0.1250  0.3750  0.6250  0.8750  1.0000
 0.0000  0.1250  0.3750  0.6250  0.8750  1.0000
[ CPUDoubleType{8,6} ]
[ TensorOptions(dtype=double, device=cpu, layout=Strided, requires_grad=false (default), pinned_memory=false (default), memory_format=(nullopt)) ]
 0.0000  0.0000  0.0000  0.0000  0.0000  0.0000
 0.0667  0.0667  0.0667  0.0667  0.0667  0.0667
 0.2000  0.2000  0.2000  0.2000  0.2000  0.2000
 0.4000  0.4000  0.4000  0.4000  0.4000  0.4000
 0.6000  0.6000  0.6000  0.6000  0.6000  0.6000
 0.8000  0.8000  0.8000  0.8000  0.8000  0.8000
 0.9333  0.9333  0.9333  0.9333  0.9333  0.9333
 1.0000  1.0000  1.0000  1.0000  1.0000  1.0000
[ CPUDoubleType{8,6} ]
[ TensorOptions(dtype=double, device=cpu, layout=Strided, requires_grad=false (default), pinned_memory=false (default), memory_format=(nullopt)) ]
 1  1  1  1  1  1
 1  1  1  1  1  1
 1  1  1  1  1  1
 1  1  1  1  1  1
 1  1  1  1  1  1
 1  1  1  1  1  1
 1  1  1  1  1  1
 1  1  1  1  1  1
[ CPUDoubleType{8,6} ]
[ TensorOptions(dtype=double, device=cpu, layout=Strided, requires_grad=false (default), pinned_memory=false (default), memory_format=(nullopt)) ]
)
)

B-spline volume
iganet::BSplineCommon<iganet::UniformBSplineCore<double, (short)3, (short)2, (short)3, (short)2>>(
parDim = 3, geoDim = 3, degrees = 2x3x2, knots = 9x12x8, coeffs = 6x8x5, options = TensorOptions(dtype=double, device=cpu, layout=Strided (default), requires_grad=false (default), pinned_memory=false (default), memory_format=(nullopt))
knots [ owns/cont owns/cont owns/cont ] = std::__1::array<at::Tensor, 3ul>(
 0.0000
 0.0000
 0.0000
 0.2500
 0.5000
 0.7500
 1.0000
 1.0000
 1.0000
[ CPUDoubleType{9} ]
[ TensorOptions(dtype=double, device=cpu, layout=Strided, requires_grad=false (default), pinned_memory=false (default), memory_format=(nullopt)) ]
 0.0000
 0.0000
 0.0000
 0.0000
 0.2000
 0.4000
 0.6000
 0.8000
 1.0000
 1.0000
 1.0000
 1.0000
[ CPUDoubleType{12} ]
[ TensorOptions(dtype=double, device=cpu, layout=Strided, requires_grad=false (default), pinned_memory=false (default), memory_format=(nullopt)) ]
 0.0000
 0.0000
 0.0000
 0.3333
 0.6667
 1.0000
 1.0000
 1.0000
[ CPUDoubleType{8} ]
[ TensorOptions(dtype=double, device=cpu, layout=Strided, requires_grad=false (default), pinned_memory=false (default), memory_format=(nullopt)) ]
)
coeffs [ owns/cont owns/cont owns/cont ] = std::__1::array<at::Tensor, 3ul>(
(1,.,.) =
  0.0000  0.1250  0.3750  0.6250  0.8750  1.0000
  0.0000  0.1250  0.3750  0.6250  0.8750  1.0000
  0.0000  0.1250  0.3750  0.6250  0.8750  1.0000
  0.0000  0.1250  0.3750  0.6250  0.8750  1.0000
  0.0000  0.1250  0.3750  0.6250  0.8750  1.0000
  0.0000  0.1250  0.3750  0.6250  0.8750  1.0000
  0.0000  0.1250  0.3750  0.6250  0.8750  1.0000
  0.0000  0.1250  0.3750  0.6250  0.8750  1.0000

(2,.,.) =
  0.0000  0.1250  0.3750  0.6250  0.8750  1.0000
  0.0000  0.1250  0.3750  0.6250  0.8750  1.0000
  0.0000  0.1250  0.3750  0.6250  0.8750  1.0000
  0.0000  0.1250  0.3750  0.6250  0.8750  1.0000
  0.0000  0.1250  0.3750  0.6250  0.8750  1.0000
  0.0000  0.1250  0.3750  0.6250  0.8750  1.0000
  0.0000  0.1250  0.3750  0.6250  0.8750  1.0000
  0.0000  0.1250  0.3750  0.6250  0.8750  1.0000

(3,.,.) =
  0.0000  0.1250  0.3750  0.6250  0.8750  1.0000
  0.0000  0.1250  0.3750  0.6250  0.8750  1.0000
  0.0000  0.1250  0.3750  0.6250  0.8750  1.0000
  0.0000  0.1250  0.3750  0.6250  0.8750  1.0000
  0.0000  0.1250  0.3750  0.6250  0.8750  1.0000
  0.0000  0.1250  0.3750  0.6250  0.8750  1.0000
  0.0000  0.1250  0.3750  0.6250  0.8750  1.0000
  0.0000  0.1250  0.3750  0.6250  0.8750  1.0000

(4,.,.) =
  0.0000  0.1250  0.3750  0.6250  0.8750  1.0000
  0.0000  0.1250  0.3750  0.6250  0.8750  1.0000
  0.0000  0.1250  0.3750  0.6250  0.8750  1.0000
  0.0000  0.1250  0.3750  0.6250  0.8750  1.0000
  0.0000  0.1250  0.3750  0.6250  0.8750  1.0000
  0.0000  0.1250  0.3750  0.6250  0.8750  1.0000
  0.0000  0.1250  0.3750  0.6250  0.8750  1.0000
  0.0000  0.1250  0.3750  0.6250  0.8750  1.0000

(5,.,.) =
  0.0000  0.1250  0.3750  0.6250  0.8750  1.0000
  0.0000  0.1250  0.3750  0.6250  0.8750  1.0000
  0.0000  0.1250  0.3750  0.6250  0.8750  1.0000
  0.0000  0.1250  0.3750  0.6250  0.8750  1.0000
  0.0000  0.1250  0.3750  0.6250  0.8750  1.0000
  0.0000  0.1250  0.3750  0.6250  0.8750  1.0000
  0.0000  0.1250  0.3750  0.6250  0.8750  1.0000
  0.0000  0.1250  0.3750  0.6250  0.8750  1.0000
[ CPUDoubleType{5,8,6} ]
[ TensorOptions(dtype=double, device=cpu, layout=Strided, requires_grad=false (default), pinned_memory=false (default), memory_format=(nullopt)) ]
(1,.,.) =
  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000
  0.0667  0.0667  0.0667  0.0667  0.0667  0.0667
  0.2000  0.2000  0.2000  0.2000  0.2000  0.2000
  0.4000  0.4000  0.4000  0.4000  0.4000  0.4000
  0.6000  0.6000  0.6000  0.6000  0.6000  0.6000
  0.8000  0.8000  0.8000  0.8000  0.8000  0.8000
  0.9333  0.9333  0.9333  0.9333  0.9333  0.9333
  1.0000  1.0000  1.0000  1.0000  1.0000  1.0000

(2,.,.) =
  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000
  0.0667  0.0667  0.0667  0.0667  0.0667  0.0667
  0.2000  0.2000  0.2000  0.2000  0.2000  0.2000
  0.4000  0.4000  0.4000  0.4000  0.4000  0.4000
  0.6000  0.6000  0.6000  0.6000  0.6000  0.6000
  0.8000  0.8000  0.8000  0.8000  0.8000  0.8000
  0.9333  0.9333  0.9333  0.9333  0.9333  0.9333
  1.0000  1.0000  1.0000  1.0000  1.0000  1.0000

(3,.,.) =
  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000
  0.0667  0.0667  0.0667  0.0667  0.0667  0.0667
  0.2000  0.2000  0.2000  0.2000  0.2000  0.2000
  0.4000  0.4000  0.4000  0.4000  0.4000  0.4000
  0.6000  0.6000  0.6000  0.6000  0.6000  0.6000
  0.8000  0.8000  0.8000  0.8000  0.8000  0.8000
  0.9333  0.9333  0.9333  0.9333  0.9333  0.9333
  1.0000  1.0000  1.0000  1.0000  1.0000  1.0000

(4,.,.) =
  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000
  0.0667  0.0667  0.0667  0.0667  0.0667  0.0667
  0.2000  0.2000  0.2000  0.2000  0.2000  0.2000
  0.4000  0.4000  0.4000  0.4000  0.4000  0.4000
  0.6000  0.6000  0.6000  0.6000  0.6000  0.6000
  0.8000  0.8000  0.8000  0.8000  0.8000  0.8000
  0.9333  0.9333  0.9333  0.9333  0.9333  0.9333
  1.0000  1.0000  1.0000  1.0000  1.0000  1.0000

(5,.,.) =
  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000
  0.0667  0.0667  0.0667  0.0667  0.0667  0.0667
  0.2000  0.2000  0.2000  0.2000  0.2000  0.2000
  0.4000  0.4000  0.4000  0.4000  0.4000  0.4000
  0.6000  0.6000  0.6000  0.6000  0.6000  0.6000
  0.8000  0.8000  0.8000  0.8000  0.8000  0.8000
  0.9333  0.9333  0.9333  0.9333  0.9333  0.9333
  1.0000  1.0000  1.0000  1.0000  1.0000  1.0000
[ CPUDoubleType{5,8,6} ]
[ TensorOptions(dtype=double, device=cpu, layout=Strided, requires_grad=false (default), pinned_memory=false (default), memory_format=(nullopt)) ]
(1,.,.) =
  0  0  0  0  0  0
  0  0  0  0  0  0
  0  0  0  0  0  0
  0  0  0  0  0  0
  0  0  0  0  0  0
  0  0  0  0  0  0
  0  0  0  0  0  0
  0  0  0  0  0  0

(2,.,.) =
  0.1667  0.1667  0.1667  0.1667  0.1667  0.1667
  0.1667  0.1667  0.1667  0.1667  0.1667  0.1667
  0.1667  0.1667  0.1667  0.1667  0.1667  0.1667
  0.1667  0.1667  0.1667  0.1667  0.1667  0.1667
  0.1667  0.1667  0.1667  0.1667  0.1667  0.1667
  0.1667  0.1667  0.1667  0.1667  0.1667  0.1667
  0.1667  0.1667  0.1667  0.1667  0.1667  0.1667
  0.1667  0.1667  0.1667  0.1667  0.1667  0.1667

(3,.,.) =
  0.5000  0.5000  0.5000  0.5000  0.5000  0.5000
  0.5000  0.5000  0.5000  0.5000  0.5000  0.5000
  0.5000  0.5000  0.5000  0.5000  0.5000  0.5000
  0.5000  0.5000  0.5000  0.5000  0.5000  0.5000
  0.5000  0.5000  0.5000  0.5000  0.5000  0.5000
  0.5000  0.5000  0.5000  0.5000  0.5000  0.5000
  0.5000  0.5000  0.5000  0.5000  0.5000  0.5000
  0.5000  0.5000  0.5000  0.5000  0.5000  0.5000

(4,.,.) =
  0.8333  0.8333  0.8333  0.8333  0.8333  0.8333
  0.8333  0.8333  0.8333  0.8333  0.8333  0.8333
  0.8333  0.8333  0.8333  0.8333  0.8333  0.8333
  0.8333  0.8333  0.8333  0.8333  0.8333  0.8333
  0.8333  0.8333  0.8333  0.8333  0.8333  0.8333
  0.8333  0.8333  0.8333  0.8333  0.8333  0.8333
  0.8333  0.8333  0.8333  0.8333  0.8333  0.8333
  0.8333  0.8333  0.8333  0.8333  0.8333  0.8333

(5,.,.) =
  1  1  1  1  1  1
  1  1  1  1  1  1
  1  1  1  1  1  1
  1  1  1  1  1  1
  1  1  1  1  1  1
  1  1  1  1  1  1
  1  1  1  1  1  1
  1  1  1  1  1  1
[ CPUDoubleType{5,8,6} ]
[ TensorOptions(dtype=double, device=cpu, layout=Strided, requires_grad=false (default), pinned_memory=false (default), memory_format=(nullopt)) ]
)
)
```

</details>

All functionality available for univariate uniform B-spline function
objects are also available for multivariate ones.

During creation of the B-spline function object, the values of the
control points can be specified either by defining one of the
following initialization schemes

- `init::greville` (default) sets the control points to the Greville points
- `init::linear` sets the control points linearly per direction between zero and one
- `init::linspace` sets the control points to a special linspace pattern
- `init::none` leaves the control points uninitialized
- `init::ones` sets all control points to one
- `init::random` sets all control points to a random value
- `init::zeros` sets all control points to zero

While `init::linear` initializes the coordinates of the control points
linearly along the respective parametric dimension (e.g., comparable
to the `meshgrid` function in Matlab) and sets all control point
coordinates for which `geoDim > parDim` to one, `init::linspace`
initializes all coordinates with an increasing sequence of
integers. The latter is mostly useful for debugging purposes to
identify control point coordinates unambiguously. The following code
snippet illustrates the difference between `init::linear` and
`init::linspace`

\snippet tutorial02.cxx Multivariate uniform B-splines linear vs linspace

<details>
  <summary>Click to see output</summary>

```
[INFO]

B-spline surface with linear initialization of the control points
std::__1::array<at::Tensor, 3ul>(
 0.0000  0.2500  0.5000  0.7500  1.0000
 0.0000  0.2500  0.5000  0.7500  1.0000
 0.0000  0.2500  0.5000  0.7500  1.0000
 0.0000  0.2500  0.5000  0.7500  1.0000
 0.0000  0.2500  0.5000  0.7500  1.0000
[ CPUDoubleType{5,5} ]
[ TensorOptions(dtype=double, device=cpu, layout=Strided, requires_grad=false (default), pinned_memory=false (default), memory_format=(nullopt)) ]
 0.0000  0.0000  0.0000  0.0000  0.0000
 0.2500  0.2500  0.2500  0.2500  0.2500
 0.5000  0.5000  0.5000  0.5000  0.5000
 0.7500  0.7500  0.7500  0.7500  0.7500
 1.0000  1.0000  1.0000  1.0000  1.0000
[ CPUDoubleType{5,5} ]
[ TensorOptions(dtype=double, device=cpu, layout=Strided, requires_grad=false (default), pinned_memory=false (default), memory_format=(nullopt)) ]
 1  1  1  1  1
 1  1  1  1  1
 1  1  1  1  1
 1  1  1  1  1
 1  1  1  1  1
[ CPUDoubleType{5,5} ]
[ TensorOptions(dtype=double, device=cpu, layout=Strided, requires_grad=false (default), pinned_memory=false (default), memory_format=(nullopt)) ]
)
[INFO]

B-spline surface with linspace initialization of the control points
std::__1::array<at::Tensor, 3ul>(
  0   1   2   3   4
  5   6   7   8   9
 10  11  12  13  14
 15  16  17  18  19
 20  21  22  23  24
[ CPUDoubleType{5,5} ]
[ TensorOptions(dtype=double, device=cpu, layout=Strided, requires_grad=false (default), pinned_memory=false (default), memory_format=(nullopt)) ]
   0   10   20   30   40
  50   60   70   80   90
 100  110  120  130  140
 150  160  170  180  190
 200  210  220  230  240
[ CPUDoubleType{5,5} ]
[ TensorOptions(dtype=double, device=cpu, layout=Strided, requires_grad=false (default), pinned_memory=false (default), memory_format=(nullopt)) ]
    0   100   200   300   400
  500   600   700   800   900
 1000  1100  1200  1300  1400
 1500  1600  1700  1800  1900
 2000  2100  2200  2300  2400
[ CPUDoubleType{5,5} ]
[ TensorOptions(dtype=double, device=cpu, layout=Strided, requires_grad=false (default), pinned_memory=false (default), memory_format=(nullopt)) ]
)
```

</details>

Instead of providing an initialization stratey for the control points,
it is also also possible to provide the coordinates of the control
points as an array of tensors, whereby the size of the array must be
equal to `geoDim` and the size of the tensor must be equal to
`ncumcoeffs`, e.g.

\snippet tutorial02.cxx Multivariate uniform B-spline with externally defined control points

<details>
  <summary>Click to see output</summary>

```
[INFO]

B-spline surface with externally defined control point coordinates
iganet::BSplineCommon<iganet::UniformBSplineCore<double, (short)2, (short)2, (short)3>>(
parDim = 2, geoDim = 2, degrees = 2x3, knots = 8x9, coeffs = 5x5, options = TensorOptions(dtype=double, device=cpu, layout=Strided (default), requires_grad=false (default), pinned_memory=false (default), memory_format=(nullopt))
knots [ owns/cont owns/cont ] = std::__1::array<at::Tensor, 2ul>(
 0.0000
 0.0000
 0.0000
 0.3333
 0.6667
 1.0000
 1.0000
 1.0000
[ CPUDoubleType{8} ]
[ TensorOptions(dtype=double, device=cpu, layout=Strided, requires_grad=false (default), pinned_memory=false (default), memory_format=(nullopt)) ]
 0.0000
 0.0000
 0.0000
 0.0000
 0.5000
 1.0000
 1.0000
 1.0000
 1.0000
[ CPUDoubleType{9} ]
[ TensorOptions(dtype=double, device=cpu, layout=Strided, requires_grad=false (default), pinned_memory=false (default), memory_format=(nullopt)) ]
)
coeffs [ owns/cont owns/cont ] = std::__1::array<at::Tensor, 2ul>(
 0.0000  0.1667  0.5000  0.8333  1.0000
 0.0000  0.1667  0.5000  0.8333  1.0000
 0.0000  0.1667  0.5000  0.8333  1.0000
 0.0000  0.1667  0.5000  0.8333  1.0000
 0.0000  0.1667  0.5000  0.8333  1.0000
[ CPUDoubleType{5,5} ]
[ TensorOptions(dtype=double, device=cpu, layout=Strided, requires_grad=false (default), pinned_memory=false (default), memory_format=(nullopt)) ]
 0.0000  0.1667  0.5000  0.8333  1.0000
 0.0000  0.1667  0.5000  0.8333  1.0000
 0.0000  0.1667  0.5000  0.8333  1.0000
 0.0000  0.1667  0.5000  0.8333  1.0000
 0.0000  0.1667  0.5000  0.8333  1.0000
[ CPUDoubleType{5,5} ]
[ TensorOptions(dtype=double, device=cpu, layout=Strided, requires_grad=false (default), pinned_memory=false (default), memory_format=(nullopt)) ]
)
)
```
</details>


By default, the B-spline function constructor makes a soft copy of the
control points meaning that the `coeffs` object and the control point
coefficients of the B-spline function object share the same
data. Hence, changing the value of an entry in `coeffs` will lead to a
change in the coordinates of the control points, e.g.,

\snippet tutorial02.cxx Multivariate uniform B-spline with externally defined control points updated

<details>
  <summary>Click to see output</summary>

```
[INFO]

Update control point coordinatesstd::__1::array<at::Tensor, 2ul>(
  0.0000   0.1667   0.5000  10.0000   1.0000
  0.0000   0.1667   0.5000   0.8333   1.0000
  0.0000   0.1667   0.5000   0.8333   1.0000
  0.0000   0.1667   0.5000   0.8333   1.0000
  0.0000   0.1667   0.5000   0.8333   1.0000
[ CPUDoubleType{5,5} ]
[ TensorOptions(dtype=double, device=cpu, layout=Strided, requires_grad=false (default), pinned_memory=false (default), memory_format=(nullopt)) ]
 0.0000  0.1667  0.5000  0.8333  1.0000
 0.0000  0.1667  0.5000  0.8333  1.0000
 0.0000  0.1667  0.5000  0.8333  1.0000
 0.0000  0.1667  0.5000  0.8333  1.0000
 0.0000  0.1667  0.5000  0.8333  1.0000
[ CPUDoubleType{5,5} ]
[ TensorOptions(dtype=double, device=cpu, layout=Strided, requires_grad=false (default), pinned_memory=false (default), memory_format=(nullopt)) ]
)
```
</details>

This behavior can be changed by providing a third argument
(`clone=true`) to the constructor so that a deep copy of the control
point coordinates is made

\snippet tutorial02.cxx Multivariate uniform B-spline with externally defined and cloned control points

As shown in the above code snippet, the `coeffs_view()` function
returns the control point coordinates as a tensor view that is
reshaped according to the values given by `ncoeffs()`.

It is moreover possible to construct B-spline function objects from
existing ones. To ensure conformity with Torch, the copy constructor
makes only soft copies of the internal data. A deep copy can be made
by using the `clone()` function on the existing B-spline function
object and thereby passing it to the move constructor, e.g.,

\snippet tutorial02.cxx Duplicating multivariate uniform B-spline

### Query

Query attributes

device()
device_index()
dtype()
layout()
requires_griad()
pinned_memory()
is_sparse()

is_uniform()
is_nonuniform()

set_requires_grad()

options()


Constructors (ncoeff, init, options)
Constructors (ncoeff, coeffs, options)
Constructor(other, options)



as_tensor()
from_tensor()
as_tensor_size()

eval_from_precomputed(basfunc, coeff_indices, numeval, sizes)
eval(xi)
eval(xi, knot_indices)
eval(xi, knot_indices, coeff_indices)

find_knot_indices(xi)
find_coeff_indices(indices)


eval_basfunc(xi)
eval_basfunc(xi, knot_indices)

transform()

to_json()
knots_to_json()
coeffs_to_json()

from_json()

to_xml()
from_xml()

load()
save()

read()
write()

isclose()
==
!=

uniform_refine()
init_knots()
init_coeffs()

to_gismo()
from_gismo()

clone()
to()
diff()
abs_diff()
scale()
translate()
rotate()
boundingBox()

curl()
icurl()
div()
idiv()
grad()
igrad()
hess()
ihess()
ja()
ijac()
lapl()
ilapl()


plot()
