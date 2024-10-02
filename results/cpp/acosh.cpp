// This file is generated using functional_algorithms tool (0.10.2.dev9+g7001467.d20241002), see
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
  float zero = 0;
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
  float mx__0 = std::max(abs_xp1, y);
  float mn = std::min(abs_xp1, y);
  float sqrt_two = std::sqrt(two);
  float mn_over_mx = (mn) / (mx__0);
  float r__0 = (mn_over_mx) * (mn_over_mx);
  float sqa = std::sqrt((one) + (r__0));
  float r =
      (((mx__0) == (mn)) ? ((sqrt_two) * (mx__0))
                         : (((((sqa) == (one)) && ((r__0) > (zero)))
                                 ? ((mx__0) + (((mx__0) * (r__0)) / (two)))
                                 : ((mx__0) * (sqa)))));
  float abs_xm1 = std::abs(xm1);
  float mx__1 = std::max(abs_xm1, y);
  float mn__0 = std::min(abs_xm1, y);
  float mn_over_mx__0 = (mn__0) / (mx__1);
  float r__1 = (mn_over_mx__0) * (mn_over_mx__0);
  float sqa__0 = std::sqrt((one) + (r__1));
  float s =
      (((mx__1) == (mn__0)) ? ((sqrt_two) * (mx__1))
                            : (((((sqa__0) == (one)) && ((r__1) > (zero)))
                                    ? ((mx__1) + (((mx__1) * (r__1)) / (two)))
                                    : ((mx__1) * (sqa__0)))));
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
  float half_apx = (half) * ((a) + (x));
  float imag__0 = std::atan2(
      (((std::max(x, y)) >= (safe_max))
           ? (y)
           : ((((x) <= (one))
                   ? (std::sqrt((half_apx) * (((yy) / (rpxp1)) + (smxm1))))
                   : ((y) * (std::sqrt(((half_apx) / (rpxp1)) +
                                       ((half_apx) / (spxm1)))))))),
      signed_x);
  return std::complex<float>(
      (((mx) >= (((y_gt_safe_max_opt) ? (safe_max_opt) : (safe_max))))
           ? (((std::log(two)) + (std::log(mx))) +
              ((half) * (std::log1p((xoy) * (xoy)))))
           : (((logical_and_lt_y_safe_min_lt_x_one)
                   ? ((y) / (sq))
                   : (std::log1p((am1) + (sq)))))),
      (((signed_y) < (0)) ? (-(imag__0)) : (imag__0)));
}

std::complex<double> acosh_3(std::complex<double> z) {
  double signed_y = (z).imag();
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
  double zero = 0;
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
  double mx__0 = std::max(abs_xp1, y);
  double mn = std::min(abs_xp1, y);
  double sqrt_two = std::sqrt(two);
  double mn_over_mx = (mn) / (mx__0);
  double r__0 = (mn_over_mx) * (mn_over_mx);
  double sqa = std::sqrt((one) + (r__0));
  double r =
      (((mx__0) == (mn)) ? ((sqrt_two) * (mx__0))
                         : (((((sqa) == (one)) && ((r__0) > (zero)))
                                 ? ((mx__0) + (((mx__0) * (r__0)) / (two)))
                                 : ((mx__0) * (sqa)))));
  double abs_xm1 = std::abs(xm1);
  double mx__1 = std::max(abs_xm1, y);
  double mn__0 = std::min(abs_xm1, y);
  double mn_over_mx__0 = (mn__0) / (mx__1);
  double r__1 = (mn_over_mx__0) * (mn_over_mx__0);
  double sqa__0 = std::sqrt((one) + (r__1));
  double s =
      (((mx__1) == (mn__0)) ? ((sqrt_two) * (mx__1))
                            : (((((sqa__0) == (one)) && ((r__1) > (zero)))
                                    ? ((mx__1) + (((mx__1) * (r__1)) / (two)))
                                    : ((mx__1) * (sqa__0)))));
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
  double half_apx = (half) * ((a) + (x));
  double imag__0 = std::atan2(
      (((std::max(x, y)) >= (safe_max))
           ? (y)
           : ((((x) <= (one))
                   ? (std::sqrt((half_apx) * (((yy) / (rpxp1)) + (smxm1))))
                   : ((y) * (std::sqrt(((half_apx) / (rpxp1)) +
                                       ((half_apx) / (spxm1)))))))),
      signed_x);
  return std::complex<double>(
      (((mx) >= (((y_gt_safe_max_opt) ? (safe_max_opt) : (safe_max))))
           ? (((std::log(two)) + (std::log(mx))) +
              ((half) * (std::log1p((xoy) * (xoy)))))
           : (((logical_and_lt_y_safe_min_lt_x_one)
                   ? ((y) / (sq))
                   : (std::log1p((am1) + (sq)))))),
      (((signed_y) < (0)) ? (-(imag__0)) : (imag__0)));
}