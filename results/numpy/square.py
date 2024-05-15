
# This file is generated using functional_algorithms tool (0.1.dev6+g8b58236), see
#   https://github.com/pearu/functional_algorithms
def square(z: numpy.complex128) -> numpy.complex128:
  with warnings.catch_warnings(action="ignore"):
    z = numpy.complex128(z)
    x: numpy.float64 = (z).real
    y: numpy.float64 = (z).imag
    result = make_complex((numpy.float64(0)) if (numpy.equal(numpy.abs(x), numpy.abs(y), dtype=numpy.bool_)) else (((x) - (y)) * ((x) + (y))), (numpy.float64(2)) * ((x) * (y)))
    return result