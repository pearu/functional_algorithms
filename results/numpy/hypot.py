
# This file is generated using functional_algorithms tool (0.1.dev6+g8b58236), see
#   https://github.com/pearu/functional_algorithms
def hypot(x: numpy.float64, y: numpy.float64) -> numpy.float64:
  with warnings.catch_warnings(action="ignore"):
    x = numpy.float64(x)
    y = numpy.float64(y)
    abs_x: numpy.float64 = numpy.abs(x)
    abs_y: numpy.float64 = numpy.abs(y)
    mx: numpy.float64 = max(abs_x, abs_y)
    mn: numpy.float64 = min(abs_x, abs_y)
    z: numpy.float64 = (mn) / (mx)
    result = ((mx) * (numpy.sqrt(numpy.float64(2)))) if (numpy.equal(mx, mn, dtype=numpy.bool_)) else ((mx) * (numpy.sqrt(((z) * (z)) + (numpy.float64(1)))))
    return result