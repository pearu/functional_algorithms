// This file is generated using functional_algorithms tool (0.4.0), see
//   https://github.com/pearu/functional_algorithms
// for more information.



#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>


float asinh_0(float z) {
  float xoy = 0;
  float y = std::abs(signed_y);
  float safe_max = (std::sqrt(std::numeric_limits<float>::max())) / (8);
  float safe_max_opt = (safe_max) * (1e-06);
  bool y_gt_safe_max_opt = (y) >= (safe_max_opt);
  float mx = ((y_gt_safe_max_opt) ? (y) : (xoy));
  float two = 2;
  bool lt_y_safe_min =
      (y) < ((std::sqrt(std::numeric_limits<float>::min())) * (4));
  float xm1 = -1;
  float half = 0.5;
  float xp1 = 1;
  float _hypot_2_mx = std::max(xp1, y);
  float _hypot_2_mn_over_mx = (std::min(xp1, y)) / (_hypot_2_mx);
  float _hypot_2_r = (_hypot_2_mn_over_mx) * (_hypot_2_mn_over_mx);
  float _hypot_2_sqa = std::sqrt((xp1) + (_hypot_2_r));
  float s =
      (((xp1) == (y))
           ? ((std::sqrt(two)) * (_hypot_2_mx))
           : (((((_hypot_2_sqa) == (xp1)) && ((_hypot_2_r) > (xoy)))
                   ? ((_hypot_2_mx) + (((_hypot_2_mx) * (_hypot_2_r)) / (two)))
                   : ((_hypot_2_mx) * (_hypot_2_sqa)))));
  float apx = (half) * ((s) + (s));
  float ap1 = (apx) + (xp1);
  float half_yy = (half) * ((y) * (y));
  float x_ge_1_or_not =
      (((apx) <= (1.5))
           ? (((half_yy) / ((s) + (xp1))) + ((half_yy) / ((s) - (xm1))))
           : ((apx) - (xp1)));
  float am1 = ((lt_y_safe_min) ? (-((xm1) / (ap1))) : (x_ge_1_or_not));
  float sq = std::sqrt((am1) * (ap1));
  float imag =
      (((mx) >= (((y_gt_safe_max_opt) ? (safe_max_opt) : (safe_max))))
           ? ((std::log(two)) + (std::log(mx)))
           : (((lt_y_safe_min) ? ((y) / (sq)) : (std::log1p((am1) + (sq))))));
  return (((signed_y) < (xoy)) ? (-(imag)) : (imag));
}

double asinh_1(double z) {
  double xoy = 0;
  double y = std::abs(signed_y);
  double safe_max = (std::sqrt(std::numeric_limits<double>::max())) / (8);
  double safe_max_opt = (safe_max) * (1e-06);
  bool y_gt_safe_max_opt = (y) >= (safe_max_opt);
  double mx = ((y_gt_safe_max_opt) ? (y) : (xoy));
  double two = 2;
  bool lt_y_safe_min =
      (y) < ((std::sqrt(std::numeric_limits<double>::min())) * (4));
  double xm1 = -1;
  double half = 0.5;
  double xp1 = 1;
  double _hypot_2_mx = std::max(xp1, y);
  double _hypot_2_mn_over_mx = (std::min(xp1, y)) / (_hypot_2_mx);
  double _hypot_2_r = (_hypot_2_mn_over_mx) * (_hypot_2_mn_over_mx);
  double _hypot_2_sqa = std::sqrt((xp1) + (_hypot_2_r));
  double s =
      (((xp1) == (y))
           ? ((std::sqrt(two)) * (_hypot_2_mx))
           : (((((_hypot_2_sqa) == (xp1)) && ((_hypot_2_r) > (xoy)))
                   ? ((_hypot_2_mx) + (((_hypot_2_mx) * (_hypot_2_r)) / (two)))
                   : ((_hypot_2_mx) * (_hypot_2_sqa)))));
  double apx = (half) * ((s) + (s));
  double ap1 = (apx) + (xp1);
  double half_yy = (half) * ((y) * (y));
  double x_ge_1_or_not =
      (((apx) <= (1.5))
           ? (((half_yy) / ((s) + (xp1))) + ((half_yy) / ((s) - (xm1))))
           : ((apx) - (xp1)));
  double am1 = ((lt_y_safe_min) ? (-((xm1) / (ap1))) : (x_ge_1_or_not));
  double sq = std::sqrt((am1) * (ap1));
  double imag =
      (((mx) >= (((y_gt_safe_max_opt) ? (safe_max_opt) : (safe_max))))
           ? ((std::log(two)) + (std::log(mx)))
           : (((lt_y_safe_min) ? ((y) / (sq)) : (std::log1p((am1) + (sq))))));
  return (((signed_y) < (xoy)) ? (-(imag)) : (imag));
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