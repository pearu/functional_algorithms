
# This file is generated using functional_algorithms tool (0.1.dev6+g8b58236), see
#   https://github.com/pearu/functional_algorithms
def square(z: complex) -> complex:
  x: float = (z).real
  y: float = (z).imag
  return complex((0) if ((abs(x)) == (abs(y))) else (((x) - (y)) * ((x) + (y))), (2) * ((x) * (y)))