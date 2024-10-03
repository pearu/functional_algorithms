// This file is generated using functional_algorithms tool (0.10.2.dev9+g7001467.d20241002), see
//   https://github.com/pearu/functional_algorithms
// for more information.



#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>


std::complex<float> atan_2(std::complex<float> z) {
  float imag_z = (z).imag();
  float x = -(imag_z);
  float ax = std::abs(x);
  float safe_max = 281475040000000.0;
  float y = (z).real();
  float ay = std::abs(y);
  bool in_safe_region = ((ax) < (safe_max)) && ((ay) < (safe_max));
  float one = 1.0;
  float naxm1 = (one) - (ax);
  float y2 = (y) * (y);
  float constant_fneg1 = -1.0;
  float zero = 0.0;
  float constant_posinf = std::numeric_limits<float>::infinity();
  float constant_neginf = (-std::numeric_limits<float>::infinity());
  return std::complex<float>(
      (((in_safe_region)
            ? (std::atan2((y) + (y), ((naxm1) * ((one) + (ax))) - (y2)))
            : (((((y) >= (0.0)) ? (one) : (constant_fneg1))) * (3.1415927)))) *
          (0.5),
      -((((((x) >= (zero)) ? (one) : (constant_fneg1))) *
         (std::log1p(
             (4.0) *
             (((in_safe_region)
                   ? ((ax) / (((naxm1) * (naxm1)) + (y2)))
                   : (((((ay) * (16777218.0)) < (ax))
                           ? ((one) / (ax))
                           : ((((((x) == (constant_posinf)) ||
                                 ((x) == (constant_neginf))) ||
                                (((y) == (constant_posinf)) ||
                                 ((y) == (constant_neginf))))
                                   ? (zero)
                                   : (((one) / (((ax) / (y)) + ((y) / (ax)))) /
                                      (y))))))))))) *
        (0.25)));
}

std::complex<double> atan_3(std::complex<double> z) {
  double imag_z = (z).imag();
  double x = -(imag_z);
  double ax = std::abs(x);
  double safe_max = 8.112963841460672e+31;
  double y = (z).real();
  double ay = std::abs(y);
  bool in_safe_region = ((ax) < (safe_max)) && ((ay) < (safe_max));
  double one = 1.0;
  double naxm1 = (one) - (ax);
  double y2 = (y) * (y);
  double constant_fneg1 = -1.0;
  double zero = 0.0;
  double constant_posinf = std::numeric_limits<double>::infinity();
  double constant_neginf = (-std::numeric_limits<double>::infinity());
  return std::complex<double>(
      (((in_safe_region)
            ? (std::atan2((y) + (y), ((naxm1) * ((one) + (ax))) - (y2)))
            : (((((y) >= (0.0)) ? (one) : (constant_fneg1))) *
               (3.141592653589793)))) *
          (0.5),
      -((((((x) >= (zero)) ? (one) : (constant_fneg1))) *
         (std::log1p(
             (4.0) *
             (((in_safe_region)
                   ? ((ax) / (((naxm1) * (naxm1)) + (y2)))
                   : (((((ay) * (9007199254740994.0)) < (ax))
                           ? ((one) / (ax))
                           : ((((((x) == (constant_posinf)) ||
                                 ((x) == (constant_neginf))) ||
                                (((y) == (constant_posinf)) ||
                                 ((y) == (constant_neginf))))
                                   ? (zero)
                                   : (((one) / (((ax) / (y)) + ((y) / (ax)))) /
                                      (y))))))))))) *
        (0.25)));
}