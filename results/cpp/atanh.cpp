// This file is generated using functional_algorithms tool (0.10.2.dev9+g7001467.d20241002), see
//   https://github.com/pearu/functional_algorithms
// for more information.



#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>


std::complex<float> atanh_2(std::complex<float> z) {
  float x = (z).real();
  float zero = 0;
  float one = 1;
  float constant_neg1 = -1;
  float ax = std::abs(x);
  float largest = std::numeric_limits<float>::max();
  float inv_negeps =
      (((largest) > (1e+308))
           ? (9007199254740994.0)
           : ((((largest) > (1e+38)) ? (16777218.0) : (2050.0))));
  float safe_max = (inv_negeps) * (inv_negeps);
  float y = (z).imag();
  float ay = std::abs(y);
  bool in_safe_region = ((ax) < (safe_max)) && ((ay) < (safe_max));
  float naxm1 = (one) - (ax);
  float y2 = (y) * (y);
  float constant_posinf = std::numeric_limits<float>::infinity();
  float constant_neginf = -std::numeric_limits<float>::infinity();
  return std::complex<float>(
      (((((x) >= (zero)) ? (one) : (constant_neg1))) *
       (std::log1p(
           (4) *
           (((in_safe_region)
                 ? ((ax) / (((naxm1) * (naxm1)) + (y2)))
                 : (((((ay) * (inv_negeps)) < (ax))
                         ? ((one) / (ax))
                         : ((((((x) == (constant_posinf)) ||
                               ((x) == (constant_neginf))) ||
                              (((y) == (constant_posinf)) ||
                               ((y) == (constant_neginf))))
                                 ? (zero)
                                 : (((one) / (((ax) / (y)) + ((y) / (ax)))) /
                                    (y))))))))))) *
          (0.25),
      (((in_safe_region)
            ? (std::atan2((y) + (y), ((naxm1) * ((one) + (ax))) - (y2)))
            : (((((y) >= (0)) ? (one) : (constant_neg1))) * (M_PI)))) *
          (0.5));
}

std::complex<double> atanh_3(std::complex<double> z) {
  double x = (z).real();
  double zero = 0;
  double one = 1;
  double constant_neg1 = -1;
  double ax = std::abs(x);
  double largest = std::numeric_limits<double>::max();
  double inv_negeps =
      (((largest) > (1e+308))
           ? (9007199254740994.0)
           : ((((largest) > (1e+38)) ? (16777218.0) : (2050.0))));
  double safe_max = (inv_negeps) * (inv_negeps);
  double y = (z).imag();
  double ay = std::abs(y);
  bool in_safe_region = ((ax) < (safe_max)) && ((ay) < (safe_max));
  double naxm1 = (one) - (ax);
  double y2 = (y) * (y);
  double constant_posinf = std::numeric_limits<double>::infinity();
  double constant_neginf = -std::numeric_limits<double>::infinity();
  return std::complex<double>(
      (((((x) >= (zero)) ? (one) : (constant_neg1))) *
       (std::log1p(
           (4) *
           (((in_safe_region)
                 ? ((ax) / (((naxm1) * (naxm1)) + (y2)))
                 : (((((ay) * (inv_negeps)) < (ax))
                         ? ((one) / (ax))
                         : ((((((x) == (constant_posinf)) ||
                               ((x) == (constant_neginf))) ||
                              (((y) == (constant_posinf)) ||
                               ((y) == (constant_neginf))))
                                 ? (zero)
                                 : (((one) / (((ax) / (y)) + ((y) / (ax)))) /
                                    (y))))))))))) *
          (0.25),
      (((in_safe_region)
            ? (std::atan2((y) + (y), ((naxm1) * ((one) + (ax))) - (y2)))
            : (((((y) >= (0)) ? (one) : (constant_neg1))) * (M_PI)))) *
          (0.5));
}