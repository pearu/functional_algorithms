# Functional algorithms

Implementing a math function in a software is a non-trival task when
requiring that the function must return correct or very close to
correct results for all possible inputs including complex infinities
and extremely small or extremely large values of floating point
numbers. Such algorithms typically use different approximations of the
function depending on the inputs locations in the real line or complex
plane.

This project provides a tool for defining functional algorithms for
math functions and generating implementations of the algorithms to
various programming languages and math libraries. The aim is to
provide algorithms that guarantee the correctness of the function
evaluation on the whole complex plane or real line.

The motivation for this project raises from the need to implement
sophisticated algorithms for various math libraries that can be a
tedious and errorprone task. For instance, the definition of the
algorithm for computing arcus sine for Python or NumPy target has LOC
about 45 but for the StableHLO target the LOC is 186. Implementing
such an algorithm for StableHLO by hand would be just unhuman.


## Supported algorithms

Currently, algorithms are provided for the following math functions
that are correct on the whole complex plane or real line:

- square, complex and float inputs
- hypot, float inputs
- asin, complex inputs, using modified [Hull et al](https://dl.acm.org/doi/10.1145/275323.275324) algorithm. 

## Supported targets

Currently, the implementations of supported algorithms are provided
for the following target libraries and languages:

- Python, using math functions on real inputs
- NumPy, using numpy functions on real inputs
- StableHLO, using its existing decompositions and operations

## Testing algorithms and its implementations

To ensure the correctness as well as accuracy of provided algorithms,
we'll use [MPMath](https://github.com/mpmath/mpmath/) as a reference
library of math functions. We assume that mpmath implementations
produce correct results to math functions with arbitrary precision -
this is the prerequisity for ensuring accuracy. To ensure correctness,
we'll verify this assumption for each function case separately to
eliminate the possibility of false-positives due to possible bugs in
MPMath.

The algorithms are typically validated with 32 and 64-bit floating
point numbers and their complex compositions using
[NumPy](https://github.com/numpy/numpy/). The evaluation of the numpy
target implementation is performed on logarithmic-uniform samples that
represent the whole complex plane or real line including complex
infinities, and extremly small and large values.

## A case study: square

A naive implementation of the square function can be defined as
```python
def square(z):
    return z * z
```

which produces correct results on the real line, however, in the case
of complex inputs, there exists regions in complex plane where the
given algorithm of using a plain complex multiplication produces
incorrect values. For example:
```python
>>> def square(z):
...   return z * z
... 
>>> z = complex(1e170, 1e170)
>>> square(z)
(nan+infj)
```
where the imaginary part being `inf` is expected due to overflow from
`1e170 * 1e170` but the real part ought to be zero but here the `nan`
real part originates from the following computation of `1e170 *
1e170 - 1e170 * 1e170 -> inf - inf -> nan`.

Btw, we cannot rely on NumPy square function as a reference because it
produces incorrect value as well (likely in a platform-dependent way):
```python
>>> numpy.square(z)
(-inf+infj)
```

In this project, the square function uses the following algorithm:
```python
def square(ctx, z):
    if z.is_complex:
        real = ctx.select(abs(z.real) == abs(z.imag), 0, (z.real - z.imag) * (z.real + z.imag))
        imag = 2 * (z.real * z.imag)
        return ctx.complex(real, imag)
    return z * z
```
from which implementations for different libraries and programming
languages can be generated. For example, to generate a square function
for Python, we'll use
```python
>>> import functional_algorithms as fa
>>> ctx = fa.Context()
>>> square_graph = ctx.trace(square, complex)
>>> py_square = fa.targets.python.as_function(square_graph)
>>> py_square(z)
infj
```
In general, `py_square` produces correct results on the whole complex
plane.

### Digging into details

Let us look into some of the details of the above example. First,
`square_graph` is an `Expr` instance that represents the traced
function using a pure functional form:
```python
>>> print(square_graph)
apply(square: float, (z: complex), complex(select(eq(abs(real(z: complex)), abs(imag(z: complex))), 0, multiply(subtract(real(z: complex), imag(z: complex)), add(real(z: complex), imag(z: complex)))), multiply(2, multiply(real(z: complex), imag(z: complex)))))
```

The module object `fa.targets.python` defines the so-called Python target
implementation. There exists other targets such as `fa.targets.numpy`,
`fa.targets.stablehlo`, etc.

To visualize the implementation for the given target, say,
`fa.targets.python`, we'll use `tostring(<target>)` method:
```python
>>> print(square_graph.tostring(fa.targets.python))
def square(z: complex) -> complex:
  real_z: float = (z).real
  imag_z: float = (z).imag
  eq_abs_real_z_abs_imag_z: bool = (abs(real_z)) == (abs(imag_z))
  return complex((0) if (eq_abs_real_z_abs_imag_z) else (((real_z) - (imag_z)) * ((real_z) + (imag_z))), (2) * ((real_z) * (imag_z)))
```
which is actually the definition of the Python function used above
when evaluating `py_square(z)`.

Similarly, we can generate implementations for other targets, for instance:
```python
>>> print(square_graph.tostring(fa.targets.stablehlo))
def : Pat<(CHLO_Square $z),
  (StableHLO_ComplexOp
    (StableHLO_SelectOp
      (StableHLO_CompareOp
       (StableHLO_AbsOp
         (StableHLO_RealOp:$real_z $z)),
       (StableHLO_AbsOp
         (StableHLO_ImagOp:$imag_z $z)),
        StableHLO_ComparisonDirectionValue<"EQ">,
        (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
      (StableHLO_ConstantLike<"0"> $real_z),
      (StableHLO_MulOp
        (StableHLO_SubtractOp $real_z, $imag_z),
        (StableHLO_AddOp $real_z, $imag_z))),
    (StableHLO_MulOp
      (StableHLO_ConstantLike<"2"> $real_z),
      (StableHLO_MulOp $real_z, $imag_z)))>;
```

In the case of the NumPy target, the arguments types must include
bit-width information:
```python
>>> np_square_graph = ctx.trace(square, numpy.complex64)
>>> print(np_square_graph.tostring(fa.targets.numpy))
def square(z: numpy.complex64) -> numpy.complex64:
  with warnings.catch_warnings(action="ignore"):
    z = numpy.complex64(z)
    real_z: numpy.float32 = (z).real
    imag_z: numpy.float32 = (z).imag
    result = make_complex((numpy.float32(0)) if (numpy.equal(numpy.abs(real_z), numpy.abs(imag_z), dtype=numpy.bool_)) else (((real_z) - (imag_z)) * ((real_z) + (imag_z))), (numpy.float32(2)) * ((real_z) * (imag_z)))
    return result
>>> fa.targets.numpy.as_function(np_square_graph)(z)
infj
```

## Useful tips

### Debugging NumPy target implementations

A useful feature in the `tostring` method is the `debug`
kw-argument. When it is greater than 0, type checking statements are
inserted into the function implementation:
```python
>>> print(np_square_graph.tostring(fa.targets.numpy, debug=1))
def square(z: numpy.complex64) -> numpy.complex64:
  with warnings.catch_warnings(action="ignore"):
    z = numpy.complex64(z)
    real_z: numpy.float32 = (z).real
    assert real_z.dtype == numpy.float32, (real_z.dtype, numpy.float32)
    imag_z: numpy.float32 = (z).imag
    assert imag_z.dtype == numpy.float32, (imag_z.dtype, numpy.float32)
    result = make_complex((numpy.float32(0)) if (numpy.equal(numpy.abs(real_z), numpy.abs(imag_z), dtype=numpy.bool_)) else (((real_z) - (imag_z)) * ((real_z) + (imag_z))), (numpy.float32(2)) * ((real_z) * (imag_z)))
    assert result.dtype == numpy.complex64, (result.dtype,)
    return result
```
When `debug=2`, the values of all variables are printed out when
calling the function:
```python
>>> fa.targets.numpy.as_function(np_square_graph, debug=2)(3 + 4j)
def square(z: numpy.complex64) -> numpy.complex64:
  with warnings.catch_warnings(action="ignore"):
    z = numpy.complex64(z)
    print("z=", z)
    real_z: numpy.float32 = (z).real
    print("real_z=",  real_z)
    assert real_z.dtype == numpy.float32, (real_z.dtype, numpy.float32)
    imag_z: numpy.float32 = (z).imag
    print("imag_z=",  imag_z)
    assert imag_z.dtype == numpy.float32, (imag_z.dtype, numpy.float32)
    result = make_complex((numpy.float32(0)) if (numpy.equal(numpy.abs(real_z), numpy.abs(imag_z), dtype=numpy.bool_)) else (((real_z) - (imag_z)) * ((real_z) + (imag_z))), (numpy.float32(2)) * ((real_z) * (imag_z)))
    print("result=", result)
    assert result.dtype == numpy.complex64, (result.dtype,)
    return result
z= (3+4j)
real_z= 3.0
imag_z= 4.0
result= (-7+24j)
(-7+24j)
```

### Intermediate variables in Python and NumPy target implementations

When generating implementations, one can control the naming of
intermediate variables as well as their appearance. By default,
intermediate variables are generated only for expressions that are
used multiple times as subexpressions. However, one can also force the
creation of intermediate variables for better visualization of the
implementations. For that, we'll redefine the square algorithm as follows:
```python
def square(ctx, z):
    if z.is_complex:
        x = abs(z.real)
        y = abs(z.imag)
        real = ctx.select(x == y, 0, ((x - y) * (y + y)).props_(force_ref=True, ref="real_part"))
        imag = 2 * (x * y)
        r = ctx.complex(real.props_(force_ref=True), imag.props_(force_ref=True))
        ctx.update_refs()
        return r
    return z * z
```
The generated implementation for the Python targer of the above definition is
```python
>>> square_graph = ctx.trace(square, complex)
>>> print(square_graph.tostring(fa.targets.python))
def square(z: complex) -> complex:
  real_z: float = (z).real
  x: float = abs(real_z)
  y: float = abs((z).imag)
  real_part: float = ((x) - (y)) * ((y) + (y))
  real: float = (0) if ((x) == (y)) else (real_part)
  imag: float = (2) * ((x) * (y))
  return complex(real, imag)
```
which is more expressive than the one shown above.
