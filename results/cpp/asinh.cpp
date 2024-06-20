// This file is generated using functional_algorithms tool (0.4.0), see
//   https://github.com/pearu/functional_algorithms
// for more information.



#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>


float asinh_0(float z) {
  float ax = std::abs(z);
  float ax2 = (ax) * (ax);
  float one = 1;
  return ((z == 0 ? z : std::copysign(1, z))) *
         ((((ax) >= (std::sqrt(std::numeric_limits<float>::max())))
               ? ((std::log(2)) + (std::log(ax)))
               : (std::log1p((ax) +
                             ((ax2) / ((one) + (std::sqrt((one) + (ax2)))))))));
}

double asinh_1(double z) {
  double ax = std::abs(z);
  double ax2 = (ax) * (ax);
  double one = 1;
  return ((z == 0 ? z : std::copysign(1, z))) *
         ((((ax) >= (std::sqrt(std::numeric_limits<double>::max())))
               ? ((std::log(2)) + (std::log(ax)))
               : (std::log1p((ax) +
                             ((ax2) / ((one) + (std::sqrt((one) + (ax2)))))))));
}

std::complex<float> asinh_2(std::complex<float> z) {
  std::complex<float> w =
      std::asin(std::complex<float>(-((z).imag()), (z).real()));
  return std::complex<float>((w).imag(), -((w).real()));
}

std::complex<double> asinh_3(std::complex<double> z) {
  std::complex<double> w =
      std::asin(std::complex<double>(-((z).imag()), (z).real()));
  return std::complex<double>((w).imag(), -((w).real()));
}