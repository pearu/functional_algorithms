// This file is generated using functional_algorithms tool (0.4.0), see
//   https://github.com/pearu/functional_algorithms
// for more information.



#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>


float atan_0(float z) { return std::atan(z); }

double atan_1(double z) { return std::atan(z); }

std::complex<float> atan_2(std::complex<float> z) {
  float imag_z = (z).imag();
  float x = -(imag_z);
  float ax = std::abs(x);
  float largest = std::numeric_limits<float>::max();
  float inv_negeps =
      (((largest) > (1e+308))
           ? (9007199254740994.0)
           : ((((largest) > (1e+38)) ? (16777218.0) : (2050.0))));
  float safe_max = (inv_negeps) * (inv_negeps);
  float y = (z).real();
  float ay = std::abs(y);
  bool in_safe_region = ((ax) < (safe_max)) && ((ay) < (safe_max));
  float one = 1.0;
  float naxm1 = (one) - (ax);
  float y2 = (y) * (y);
  float constant_0 = 0;
  float constant_fxbff0000000000000 = -1.0;
  float constant_posinf = std::numeric_limits<float>::infinity();
  float constant_neginf = -std::numeric_limits<float>::infinity();
  return std::complex<float>(
      (((in_safe_region)
            ? (std::atan2((y) + (y), ((naxm1) * ((one) + (ax))) - (y2)))
            : (((((y) >= (constant_0)) ? (one)
                                       : (constant_fxbff0000000000000))) *
               (M_PI)))) *
          (0.5),
      -((((((x) >= (constant_0)) ? (one) : (constant_fxbff0000000000000))) *
         (std::log1p(
             (4.0) *
             (((in_safe_region)
                   ? ((ax) / (((naxm1) * (naxm1)) + (y2)))
                   : (((((ay) * (inv_negeps)) < (ax))
                           ? ((one) / (ax))
                           : ((((((x) == (constant_posinf)) ||
                                 ((x) == (constant_neginf))) ||
                                (((y) == (constant_posinf)) ||
                                 ((y) == (constant_neginf))))
                                   ? (0.0)
                                   : (((one) / (((ax) / (y)) + ((y) / (ax)))) /
                                      (y))))))))))) *
        (0.25)));
}

std::complex<double> atan_3(std::complex<double> z) {
  double imag_z = (z).imag();
  double x = -(imag_z);
  double ax = std::abs(x);
  double largest = std::numeric_limits<double>::max();
  double inv_negeps =
      (((largest) > (1e+308))
           ? (9007199254740994.0)
           : ((((largest) > (1e+38)) ? (16777218.0) : (2050.0))));
  double safe_max = (inv_negeps) * (inv_negeps);
  double y = (z).real();
  double ay = std::abs(y);
  bool in_safe_region = ((ax) < (safe_max)) && ((ay) < (safe_max));
  double one = 1.0;
  double naxm1 = (one) - (ax);
  double y2 = (y) * (y);
  double constant_0 = 0;
  double constant_fxbff0000000000000 = -1.0;
  double constant_posinf = std::numeric_limits<double>::infinity();
  double constant_neginf = -std::numeric_limits<double>::infinity();
  return std::complex<double>(
      (((in_safe_region)
            ? (std::atan2((y) + (y), ((naxm1) * ((one) + (ax))) - (y2)))
            : (((((y) >= (constant_0)) ? (one)
                                       : (constant_fxbff0000000000000))) *
               (M_PI)))) *
          (0.5),
      -((((((x) >= (constant_0)) ? (one) : (constant_fxbff0000000000000))) *
         (std::log1p(
             (4.0) *
             (((in_safe_region)
                   ? ((ax) / (((naxm1) * (naxm1)) + (y2)))
                   : (((((ay) * (inv_negeps)) < (ax))
                           ? ((one) / (ax))
                           : ((((((x) == (constant_posinf)) ||
                                 ((x) == (constant_neginf))) ||
                                (((y) == (constant_posinf)) ||
                                 ((y) == (constant_neginf))))
                                   ? (0.0)
                                   : (((one) / (((ax) / (y)) + ((y) / (ax)))) /
                                      (y))))))))))) *
        (0.25)));
}