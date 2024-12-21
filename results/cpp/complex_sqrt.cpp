// This file is generated using functional_algorithms tool (0.11.1), see
//   https://github.com/pearu/functional_algorithms
// for more information.



#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>


std::complex<double> complex_sqrt_0(std::complex<double> z) {
  double x = (z).real();
  double constant_f0 = 0.0;
  double ax = std::abs(x);
  double y = (z).imag();
  double ay = std::abs(y);
  bool eq_ax_ay = (ax) == (ay);
  double sq_ax = std::sqrt(ax);
  double sq_2 = 1.4142135623730951;
  double mx = std::max(ax, ay);
  double mn = std::min(ax, ay);
  double one = 1.0;
  double mn_over_mx = (mn) / (mx);
  double r = (mn_over_mx) * (mn_over_mx);
  double sqa = std::sqrt((one) + (r));
  double two = 2.0;
  double u_general = std::sqrt(
      (((((mx) == (mn)) ? ((sq_2) * (mx))
                        : (((((sqa) == (one)) && ((r) > (constant_f0)))
                                ? ((mx) + (((mx) * (r)) / (two)))
                                : ((mx) * (sqa)))))) /
       (two)) +
      ((ax) / (two)));
  bool logical_or_eq_u_general_constant_f0_eq_u_general_constant_posinf =
      ((u_general) == (constant_f0)) ||
      ((u_general) == (std::numeric_limits<double>::infinity()));
  bool gt_ax_ay = (ax) > (ay);
  bool lt_ax_ay = (ax) < (ay);
  double _r_0_ =
      ((eq_ax_ay) ? (one) : (((lt_ax_ay) ? ((ax) / (ay)) : ((ay) / (ax)))));
  double abs__r_0_ = std::abs(_r_0_);
  double _mx_0_ = std::max(one, abs__r_0_);
  double _mn_0_ = std::min(one, abs__r_0_);
  double _mn_over_mx_0_ = (_mn_0_) / (_mx_0_);
  double _r_1_ = (_mn_over_mx_0_) * (_mn_over_mx_0_);
  double _sqa_0_ = std::sqrt((one) + (_r_1_));
  double h = (((_mx_0_) == (_mn_0_))
                  ? ((sq_2) * (_mx_0_))
                  : (((((_sqa_0_) == (one)) && ((_r_1_) > (constant_f0)))
                          ? ((_mx_0_) + (((_mx_0_) * (_r_1_)) / (two)))
                          : ((_mx_0_) * (_sqa_0_)))));
  double sq_1h = std::sqrt((one) + (h));
  double sq_ay = std::sqrt(ay);
  double sq_rh = std::sqrt((_r_0_) + (h));
  double u =
      ((eq_ax_ay)
           ? (((sq_ax) * (1.5537739740300374)) / (sq_2))
           : (((logical_or_eq_u_general_constant_f0_eq_u_general_constant_posinf)
                   ? (((gt_ax_ay) ? ((sq_ax) * ((sq_1h) / (sq_2)))
                                  : ((sq_ay) * ((sq_rh) / (sq_2)))))
                   : (u_general))));
  double ay_div_u =
      ((eq_ax_ay)
           ? ((sq_ay) / (2.19736822693562))
           : (((logical_or_eq_u_general_constant_f0_eq_u_general_constant_posinf)
                   ? (((gt_ax_ay)
                           ? (((sq_ay) *
                               (((eq_ax_ay)
                                     ? (one)
                                     : (((lt_ax_ay) ? ((sq_ax) / (sq_ay))
                                                    : ((sq_ay) / (sq_ax))))))) /
                              ((sq_1h) * (sq_2)))
                           : ((sq_ay) / ((sq_rh) * (sq_2)))))
                   : ((ay) / ((u_general) * (two))))));
  bool lt_y_constant_f0 = (y) < (constant_f0);
  return std::complex<double>(
      (((x) >= (constant_f0)) ? (u) : (ay_div_u)),
      (((x) < (constant_f0))
           ? (((lt_y_constant_f0) ? (-(u)) : (u)))
           : (((lt_y_constant_f0) ? (-(ay_div_u)) : (ay_div_u)))));
}