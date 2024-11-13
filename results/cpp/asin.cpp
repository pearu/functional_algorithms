// This file is generated using functional_algorithms tool (0.11.0), see
//   https://github.com/pearu/functional_algorithms
// for more information.



#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>


float asin_0(float z) {
  float one = 1.0;
  float ta = std::atan2(z, (one) + (std::sqrt(((one) - (z)) * ((one) + (z)))));
  return (ta) + (ta);
}

double asin_1(double z) {
  double one = 1.0;
  double ta = std::atan2(z, (one) + (std::sqrt(((one) - (z)) * ((one) + (z)))));
  return (ta) + (ta);
}

std::complex<float> asin_2(std::complex<float> z) {
  float signed_x = (z).real();
  float x = std::abs(signed_x);
  float signed_y = (z).imag();
  float y = std::abs(signed_y);
  float safe_max = (1.8446743e+19) / (8.0);
  float one = 1.0;
  float half = 0.5;
  float xp1 = (x) + (one);
  float abs_xp1 = std::abs(xp1);
  float mx = std::max(abs_xp1, y);
  float mn = std::min(abs_xp1, y);
  float sqrt_two = 1.4142135;
  float mn_over_mx = (mn) / (mx);
  float _r_0_ = (mn_over_mx) * (mn_over_mx);
  float sqa = std::sqrt((one) + (_r_0_));
  float constant_f0 = 0.0;
  float two = 2.0;
  float r = (((mx) == (mn)) ? ((sqrt_two) * (mx))
                            : (((((sqa) == (one)) && ((_r_0_) > (constant_f0)))
                                    ? ((mx) + (((mx) * (_r_0_)) / (two)))
                                    : ((mx) * (sqa)))));
  float xm1 = (x) - (one);
  float abs_xm1 = std::abs(xm1);
  float _mx_0_ = std::max(abs_xm1, y);
  float _mn_0_ = std::min(abs_xm1, y);
  float _mn_over_mx_0_ = (_mn_0_) / (_mx_0_);
  float _r_1_ = (_mn_over_mx_0_) * (_mn_over_mx_0_);
  float _sqa_0_ = std::sqrt((one) + (_r_1_));
  float s = (((_mx_0_) == (_mn_0_))
                 ? ((sqrt_two) * (_mx_0_))
                 : (((((_sqa_0_) == (one)) && ((_r_1_) > (constant_f0)))
                         ? ((_mx_0_) + (((_mx_0_) * (_r_1_)) / (two)))
                         : ((_mx_0_) * (_sqa_0_)))));
  float a = (half) * ((r) + (s));
  float half_apx = (half) * ((a) + (x));
  float yy = (y) * (y);
  float rpxp1 = (r) + (xp1);
  float smxm1 = (s) - (xm1);
  float spxm1 = (s) + (xm1);
  float real = std::atan2(
      signed_x,
      (((std::max(x, y)) >= (safe_max))
           ? (y)
           : ((((x) <= (one))
                   ? (std::sqrt((half_apx) * (((yy) / (rpxp1)) + (smxm1))))
                   : ((y) * (std::sqrt(((half_apx) / (rpxp1)) +
                                       ((half_apx) / (spxm1)))))))));
  float safe_max_opt =
      (((x) < ((safe_max) * (1000000000000.0))) ? ((safe_max) * (1e-06))
                                                : ((safe_max) * (100.0)));
  bool y_gt_safe_max_opt = (y) >= (safe_max_opt);
  float _mx_1_ = ((y_gt_safe_max_opt) ? (y) : (x));
  float xoy = (((y_gt_safe_max_opt) &&
                (!((y) == (std::numeric_limits<float>::infinity()))))
                   ? ((x) / (y))
                   : (constant_f0));
  bool logical_and_lt_y_safe_min_lt_x_one =
      ((y) < (4.3368087e-19)) && ((x) < (one));
  float ap1 = (a) + (one);
  float half_yy = (half) * (yy);
  float divide_half_yy_rpxp1 = (half_yy) / (rpxp1);
  float x_ge_1_or_not =
      (((x) >= (one))
           ? ((divide_half_yy_rpxp1) + ((half) * (spxm1)))
           : ((((a) <= (1.5)) ? ((divide_half_yy_rpxp1) + ((half_yy) / (smxm1)))
                              : ((a) - (one)))));
  float am1 =
      ((logical_and_lt_y_safe_min_lt_x_one) ? (-(((xp1) * (xm1)) / (ap1)))
                                            : (x_ge_1_or_not));
  float sq = std::sqrt((am1) * (ap1));
  float imag =
      (((_mx_1_) >= (((y_gt_safe_max_opt) ? (safe_max_opt) : (safe_max))))
           ? (((std::log(two)) + (std::log(_mx_1_))) +
              ((half) * (std::log1p((xoy) * (xoy)))))
           : (((logical_and_lt_y_safe_min_lt_x_one)
                   ? ((y) / (sq))
                   : (std::log1p((am1) + (sq))))));
  return std::complex<float>(
      real, (((signed_y) < (constant_f0)) ? (-(imag)) : (imag)));
}

std::complex<double> asin_3(std::complex<double> z) {
  double signed_x = (z).real();
  double x = std::abs(signed_x);
  double signed_y = (z).imag();
  double y = std::abs(signed_y);
  double safe_max = (1.3407807929942596e+154) / (8.0);
  double one = 1.0;
  double half = 0.5;
  double xp1 = (x) + (one);
  double abs_xp1 = std::abs(xp1);
  double mx = std::max(abs_xp1, y);
  double mn = std::min(abs_xp1, y);
  double sqrt_two = 1.4142135623730951;
  double mn_over_mx = (mn) / (mx);
  double _r_0_ = (mn_over_mx) * (mn_over_mx);
  double sqa = std::sqrt((one) + (_r_0_));
  double constant_f0 = 0.0;
  double two = 2.0;
  double r = (((mx) == (mn)) ? ((sqrt_two) * (mx))
                             : (((((sqa) == (one)) && ((_r_0_) > (constant_f0)))
                                     ? ((mx) + (((mx) * (_r_0_)) / (two)))
                                     : ((mx) * (sqa)))));
  double xm1 = (x) - (one);
  double abs_xm1 = std::abs(xm1);
  double _mx_0_ = std::max(abs_xm1, y);
  double _mn_0_ = std::min(abs_xm1, y);
  double _mn_over_mx_0_ = (_mn_0_) / (_mx_0_);
  double _r_1_ = (_mn_over_mx_0_) * (_mn_over_mx_0_);
  double _sqa_0_ = std::sqrt((one) + (_r_1_));
  double s = (((_mx_0_) == (_mn_0_))
                  ? ((sqrt_two) * (_mx_0_))
                  : (((((_sqa_0_) == (one)) && ((_r_1_) > (constant_f0)))
                          ? ((_mx_0_) + (((_mx_0_) * (_r_1_)) / (two)))
                          : ((_mx_0_) * (_sqa_0_)))));
  double a = (half) * ((r) + (s));
  double half_apx = (half) * ((a) + (x));
  double yy = (y) * (y);
  double rpxp1 = (r) + (xp1);
  double smxm1 = (s) - (xm1);
  double spxm1 = (s) + (xm1);
  double real = std::atan2(
      signed_x,
      (((std::max(x, y)) >= (safe_max))
           ? (y)
           : ((((x) <= (one))
                   ? (std::sqrt((half_apx) * (((yy) / (rpxp1)) + (smxm1))))
                   : ((y) * (std::sqrt(((half_apx) / (rpxp1)) +
                                       ((half_apx) / (spxm1)))))))));
  double safe_max_opt =
      (((x) < ((safe_max) * (1000000000000.0))) ? ((safe_max) * (1e-06))
                                                : ((safe_max) * (100.0)));
  bool y_gt_safe_max_opt = (y) >= (safe_max_opt);
  double _mx_1_ = ((y_gt_safe_max_opt) ? (y) : (x));
  double xoy = (((y_gt_safe_max_opt) &&
                 (!((y) == (std::numeric_limits<double>::infinity()))))
                    ? ((x) / (y))
                    : (constant_f0));
  bool logical_and_lt_y_safe_min_lt_x_one =
      ((y) < (5.966672584960166e-154)) && ((x) < (one));
  double ap1 = (a) + (one);
  double half_yy = (half) * (yy);
  double divide_half_yy_rpxp1 = (half_yy) / (rpxp1);
  double x_ge_1_or_not =
      (((x) >= (one))
           ? ((divide_half_yy_rpxp1) + ((half) * (spxm1)))
           : ((((a) <= (1.5)) ? ((divide_half_yy_rpxp1) + ((half_yy) / (smxm1)))
                              : ((a) - (one)))));
  double am1 =
      ((logical_and_lt_y_safe_min_lt_x_one) ? (-(((xp1) * (xm1)) / (ap1)))
                                            : (x_ge_1_or_not));
  double sq = std::sqrt((am1) * (ap1));
  double imag =
      (((_mx_1_) >= (((y_gt_safe_max_opt) ? (safe_max_opt) : (safe_max))))
           ? (((std::log(two)) + (std::log(_mx_1_))) +
              ((half) * (std::log1p((xoy) * (xoy)))))
           : (((logical_and_lt_y_safe_min_lt_x_one)
                   ? ((y) / (sq))
                   : (std::log1p((am1) + (sq))))));
  return std::complex<double>(
      real, (((signed_y) < (constant_f0)) ? (-(imag)) : (imag)));
}