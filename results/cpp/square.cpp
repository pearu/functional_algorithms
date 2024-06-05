// This file is generated using functional_algorithms tool (0.1.2.dev7+g332df57.d20240604), see
//   https://github.com/pearu/functional_algorithms
// for more information.



#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>


float square_0(float z) { return (z) * (z); }

double square_1(double z) { return (z) * (z); }

std::complex<float> square_2(std::complex<float> z) {
  float x = (z).real();
  float y = (z).imag();
  return std::complex<float>(
      (((std::abs(x)) == (std::abs(y))) ? (0) : (((x) - (y)) * ((x) + (y)))),
      (2) * ((x) * (y)));
}

std::complex<double> square_3(std::complex<double> z) {
  double x = (z).real();
  double y = (z).imag();
  return std::complex<double>(
      (((std::abs(x)) == (std::abs(y))) ? (0) : (((x) - (y)) * ((x) + (y)))),
      (2) * ((x) * (y)));
}