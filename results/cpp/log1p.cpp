// This file is generated using functional_algorithms tool (0.10.2.dev9+g7001467.d20241002), see
//   https://github.com/pearu/functional_algorithms
// for more information.



#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>


std::complex<float> log1p_2(std::complex<float> z) {
  float x = (z).real();
  float one = 1.0;
  float xp1 = (x) + (one);
  float y = (z).imag();
  float ay = std::abs(y);
  float axp1 = std::abs(xp1);
  float mx = std::max(axp1, ay);
  float mn = std::min(axp1, ay);
  float r = (mn) / (mx);
  return std::complex<float>(
      (std::log1p((((xp1) >= (ay)) ? (x) : ((mx) - (one))))) +
          ((0.5) * (std::log1p((((mn) == (mx)) ? (one) : ((r) * (r)))))),
      std::atan2(y, xp1));
}

std::complex<double> log1p_3(std::complex<double> z) {
  double x = (z).real();
  double one = 1.0;
  double xp1 = (x) + (one);
  double y = (z).imag();
  double ay = std::abs(y);
  double axp1 = std::abs(xp1);
  double mx = std::max(axp1, ay);
  double mn = std::min(axp1, ay);
  double r = (mn) / (mx);
  return std::complex<double>(
      (std::log1p((((xp1) >= (ay)) ? (x) : ((mx) - (one))))) +
          ((0.5) * (std::log1p((((mn) == (mx)) ? (one) : ((r) * (r)))))),
      std::atan2(y, xp1));
}