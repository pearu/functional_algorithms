// This file is generated using functional_algorithms tool (0.4.0), see
//   https://github.com/pearu/functional_algorithms
// for more information.



#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>


float acosh_0(float z) {
  float constant_2 = 2;
  float constant_1 = 1;
  float sqrt_subtract_z_constant_1 = std::sqrt((z) - (constant_1));
  return (((z) >= ((std::numeric_limits<float>::max()) / (constant_2)))
              ? ((std::log(constant_2)) + (std::log(z)))
              : (std::log1p((sqrt_subtract_z_constant_1) *
                            ((std::sqrt((z) + (constant_1))) +
                             (sqrt_subtract_z_constant_1)))));
}

double acosh_1(double z) {
  double constant_2 = 2;
  double constant_1 = 1;
  double sqrt_subtract_z_constant_1 = std::sqrt((z) - (constant_1));
  return (((z) >= ((std::numeric_limits<double>::max()) / (constant_2)))
              ? ((std::log(constant_2)) + (std::log(z)))
              : (std::log1p((sqrt_subtract_z_constant_1) *
                            ((std::sqrt((z) + (constant_1))) +
                             (sqrt_subtract_z_constant_1)))));
}

std::complex<float> acosh_2(std::complex<float> z) {
  float signed_y = (z).imag();
  float zero = 0;
  float y = std::abs(signed_y);
  float signed_x = (z).real();
  float x = std::abs(signed_x);
  float safe_max = (std::sqrt(std::numeric_limits<float>::max())) / (8);
  float safe_max_opt =
      (((x) < ((safe_max) * (1000000000000.0))) ? ((safe_max) * (1e-06))
                                                : ((safe_max) * (100.0)));
  bool y_gt_safe_max_opt = (y) >= (safe_max_opt);
  float mx = ((y_gt_safe_max_opt) ? (y) : (x));
  float two = 2;
  float half = 0.5;
  float xoy = (((y_gt_safe_max_opt) &&
                (!((y) == (std::numeric_limits<float>::infinity()))))
                   ? ((x) / (y))
                   : (zero));
  float one = 1;
  bool logical_and_lt_y_safe_min_lt_x_one =
      ((y) < ((std::sqrt(std::numeric_limits<float>::min())) * (4))) &&
      ((x) < (one));
  float xp1 = (x) + (one);
  float xm1 = (x) - (one);
  float abs_xp1 = std::abs(xp1);
  float _hypot_1_mx = std::max(abs_xp1, y);
  float mn = std::min(abs_xp1, y);
  float sqrt_two = std::sqrt(two);
  float mn_over_mx = (mn) / (_hypot_1_mx);
  float _hypot_1_r = (mn_over_mx) * (mn_over_mx);
  float sqa = std::sqrt((one) + (_hypot_1_r));
  float r =
      (((_hypot_1_mx) == (mn))
           ? ((sqrt_two) * (_hypot_1_mx))
           : (((((sqa) == (one)) && ((_hypot_1_r) > (zero)))
                   ? ((_hypot_1_mx) + (((_hypot_1_mx) * (_hypot_1_r)) / (two)))
                   : ((_hypot_1_mx) * (sqa)))));
  float abs_xm1 = std::abs(xm1);
  float _hypot_2_mx = std::max(abs_xm1, y);
  float _hypot_2_mn = std::min(abs_xm1, y);
  float _hypot_2_mn_over_mx = (_hypot_2_mn) / (_hypot_2_mx);
  float _hypot_2_r = (_hypot_2_mn_over_mx) * (_hypot_2_mn_over_mx);
  float _hypot_2_sqa = std::sqrt((one) + (_hypot_2_r));
  float s =
      (((_hypot_2_mx) == (_hypot_2_mn))
           ? ((sqrt_two) * (_hypot_2_mx))
           : (((((_hypot_2_sqa) == (one)) && ((_hypot_2_r) > (zero)))
                   ? ((_hypot_2_mx) + (((_hypot_2_mx) * (_hypot_2_r)) / (two)))
                   : ((_hypot_2_mx) * (_hypot_2_sqa)))));
  float a = (half) * ((r) + (s));
  float ap1 = (a) + (one);
  float yy = (y) * (y);
  float half_yy = (half) * (yy);
  float rpxp1 = (r) + (xp1);
  float divide_half_yy_rpxp1 = (half_yy) / (rpxp1);
  float spxm1 = (s) + (xm1);
  float smxm1 = (s) - (xm1);
  float x_ge_1_or_not =
      (((x) >= (one))
           ? ((divide_half_yy_rpxp1) + ((half) * (spxm1)))
           : ((((a) <= (1.5)) ? ((divide_half_yy_rpxp1) + ((half_yy) / (smxm1)))
                              : ((a) - (one)))));
  float am1 =
      ((logical_and_lt_y_safe_min_lt_x_one) ? (-(((xp1) * (xm1)) / (ap1)))
                                            : (x_ge_1_or_not));
  float sq = std::sqrt((am1) * (ap1));
  float imag = (((mx) >= (((y_gt_safe_max_opt) ? (safe_max_opt) : (safe_max))))
                    ? (((std::log(two)) + (std::log(mx))) +
                       ((half) * (std::log1p((xoy) * (xoy)))))
                    : (((logical_and_lt_y_safe_min_lt_x_one)
                            ? ((y) / (sq))
                            : (std::log1p((am1) + (sq))))));
  float half_apx = (half) * ((a) + (x));
  float acos_real = std::atan2(
      (((std::max(x, y)) >= (safe_max))
           ? (y)
           : ((((x) <= (one))
                   ? (std::sqrt((half_apx) * (((yy) / (rpxp1)) + (smxm1))))
                   : ((y) * (std::sqrt(((half_apx) / (rpxp1)) +
                                       ((half_apx) / (spxm1)))))))),
      signed_x);
  std::complex<float> complex_negative_acos_signed_imag_acos_real =
      std::complex<float>(-((((signed_y) < (zero)) ? (imag) : (-(imag)))),
                          acos_real);
  return (((signed_y) < (0)) ? (-(complex_negative_acos_signed_imag_acos_real))
                             : (complex_negative_acos_signed_imag_acos_real));
}

std::complex<double> acosh_3(std::complex<double> z) {
  double signed_y = (z).imag();
  double zero = 0;
  double y = std::abs(signed_y);
  double signed_x = (z).real();
  double x = std::abs(signed_x);
  double safe_max = (std::sqrt(std::numeric_limits<double>::max())) / (8);
  double safe_max_opt =
      (((x) < ((safe_max) * (1000000000000.0))) ? ((safe_max) * (1e-06))
                                                : ((safe_max) * (100.0)));
  bool y_gt_safe_max_opt = (y) >= (safe_max_opt);
  double mx = ((y_gt_safe_max_opt) ? (y) : (x));
  double two = 2;
  double half = 0.5;
  double xoy = (((y_gt_safe_max_opt) &&
                 (!((y) == (std::numeric_limits<double>::infinity()))))
                    ? ((x) / (y))
                    : (zero));
  double one = 1;
  bool logical_and_lt_y_safe_min_lt_x_one =
      ((y) < ((std::sqrt(std::numeric_limits<double>::min())) * (4))) &&
      ((x) < (one));
  double xp1 = (x) + (one);
  double xm1 = (x) - (one);
  double abs_xp1 = std::abs(xp1);
  double _hypot_1_mx = std::max(abs_xp1, y);
  double mn = std::min(abs_xp1, y);
  double sqrt_two = std::sqrt(two);
  double mn_over_mx = (mn) / (_hypot_1_mx);
  double _hypot_1_r = (mn_over_mx) * (mn_over_mx);
  double sqa = std::sqrt((one) + (_hypot_1_r));
  double r =
      (((_hypot_1_mx) == (mn))
           ? ((sqrt_two) * (_hypot_1_mx))
           : (((((sqa) == (one)) && ((_hypot_1_r) > (zero)))
                   ? ((_hypot_1_mx) + (((_hypot_1_mx) * (_hypot_1_r)) / (two)))
                   : ((_hypot_1_mx) * (sqa)))));
  double abs_xm1 = std::abs(xm1);
  double _hypot_2_mx = std::max(abs_xm1, y);
  double _hypot_2_mn = std::min(abs_xm1, y);
  double _hypot_2_mn_over_mx = (_hypot_2_mn) / (_hypot_2_mx);
  double _hypot_2_r = (_hypot_2_mn_over_mx) * (_hypot_2_mn_over_mx);
  double _hypot_2_sqa = std::sqrt((one) + (_hypot_2_r));
  double s =
      (((_hypot_2_mx) == (_hypot_2_mn))
           ? ((sqrt_two) * (_hypot_2_mx))
           : (((((_hypot_2_sqa) == (one)) && ((_hypot_2_r) > (zero)))
                   ? ((_hypot_2_mx) + (((_hypot_2_mx) * (_hypot_2_r)) / (two)))
                   : ((_hypot_2_mx) * (_hypot_2_sqa)))));
  double a = (half) * ((r) + (s));
  double ap1 = (a) + (one);
  double yy = (y) * (y);
  double half_yy = (half) * (yy);
  double rpxp1 = (r) + (xp1);
  double divide_half_yy_rpxp1 = (half_yy) / (rpxp1);
  double spxm1 = (s) + (xm1);
  double smxm1 = (s) - (xm1);
  double x_ge_1_or_not =
      (((x) >= (one))
           ? ((divide_half_yy_rpxp1) + ((half) * (spxm1)))
           : ((((a) <= (1.5)) ? ((divide_half_yy_rpxp1) + ((half_yy) / (smxm1)))
                              : ((a) - (one)))));
  double am1 =
      ((logical_and_lt_y_safe_min_lt_x_one) ? (-(((xp1) * (xm1)) / (ap1)))
                                            : (x_ge_1_or_not));
  double sq = std::sqrt((am1) * (ap1));
  double imag = (((mx) >= (((y_gt_safe_max_opt) ? (safe_max_opt) : (safe_max))))
                     ? (((std::log(two)) + (std::log(mx))) +
                        ((half) * (std::log1p((xoy) * (xoy)))))
                     : (((logical_and_lt_y_safe_min_lt_x_one)
                             ? ((y) / (sq))
                             : (std::log1p((am1) + (sq))))));
  double half_apx = (half) * ((a) + (x));
  double acos_real = std::atan2(
      (((std::max(x, y)) >= (safe_max))
           ? (y)
           : ((((x) <= (one))
                   ? (std::sqrt((half_apx) * (((yy) / (rpxp1)) + (smxm1))))
                   : ((y) * (std::sqrt(((half_apx) / (rpxp1)) +
                                       ((half_apx) / (spxm1)))))))),
      signed_x);
  std::complex<double> complex_negative_acos_signed_imag_acos_real =
      std::complex<double>(-((((signed_y) < (zero)) ? (imag) : (-(imag)))),
                           acos_real);
  return (((signed_y) < (0)) ? (-(complex_negative_acos_signed_imag_acos_real))
                             : (complex_negative_acos_signed_imag_acos_real));
}