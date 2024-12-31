// This file is generated using functional_algorithms tool (0.14.1.dev0+ge22be68.d20241231), see
//   https://github.com/pearu/functional_algorithms
// for more information.



#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>


std::complex<double> complex_log_0(std::complex<double> z) {
  double constant_fneg1 = -1.0;
  double y = (z).imag();
  double square_dekker_high = (y) * (y);
  double x = (z).real();
  double _square_dekker_high_0_ = (x) * (x);
  bool gt_square_dekker_high__square_dekker_high_0_ =
      (square_dekker_high) > (_square_dekker_high_0_);
  double mxh = ((gt_square_dekker_high__square_dekker_high_0_)
                    ? (square_dekker_high)
                    : (_square_dekker_high_0_));
  double _add_fast2sum_high_2_ = (constant_fneg1) + (mxh);
  double mnh =
      ((gt_square_dekker_high__square_dekker_high_0_) ? (_square_dekker_high_0_)
                                                      : (square_dekker_high));
  double _add_fast2sum_high_1_ = (_add_fast2sum_high_2_) + (mnh);
  double largest = std::numeric_limits<double>::max();
  double veltkamp_splitter_constant =
      (((largest) > (1e+308)) ? (134217729.0)
                              : ((((largest) > (1e+38)) ? (4097.0) : (65.0))));
  double multiply_veltkamp_splitter_constant_y =
      (veltkamp_splitter_constant) * (y);
  double yh = (multiply_veltkamp_splitter_constant_y) +
              ((y) - (multiply_veltkamp_splitter_constant_y));
  double yl = (y) - (yh);
  double multiply_yh_yl = (yh) * (yl);
  double square_dekker_low =
      ((((-(square_dekker_high)) + ((yh) * (yh))) + (multiply_yh_yl)) +
       (multiply_yh_yl)) +
      ((yl) * (yl));
  double _add_fast2sum_high_0_ = (_add_fast2sum_high_1_) + (square_dekker_low);
  double multiply_veltkamp_splitter_constant_x =
      (veltkamp_splitter_constant) * (x);
  double xh = (multiply_veltkamp_splitter_constant_x) +
              ((x) - (multiply_veltkamp_splitter_constant_x));
  double xl = (x) - (xh);
  double multiply_xh_xl = (xh) * (xl);
  double _square_dekker_low_0_ =
      ((((-(_square_dekker_high_0_)) + ((xh) * (xh))) + (multiply_xh_xl)) +
       (multiply_xh_xl)) +
      ((xl) * (xl));
  double add_fast2sum_high = (_add_fast2sum_high_0_) + (_square_dekker_low_0_);
  double add_fast2sum_low =
      (mxh) - ((_add_fast2sum_high_2_) - (constant_fneg1));
  double _add_fast2sum_low_0_ =
      (mnh) - ((_add_fast2sum_high_1_) - (_add_fast2sum_high_2_));
  double _add_fast2sum_low_1_ =
      (square_dekker_low) - ((_add_fast2sum_high_0_) - (_add_fast2sum_high_1_));
  double _add_fast2sum_low_2_ =
      (_square_dekker_low_0_) - ((add_fast2sum_high) - (_add_fast2sum_high_0_));
  double sum_fast2sum_high =
      (add_fast2sum_high) + ((((add_fast2sum_low) + (_add_fast2sum_low_0_)) +
                              (_add_fast2sum_low_1_)) +
                             (_add_fast2sum_low_2_));
  double half = 0.5;
  double abs_x = std::abs(x);
  double abs_y = std::abs(y);
  double mx = std::max(abs_x, abs_y);
  double mn = std::min(abs_x, abs_y);
  double mn_over_mx = (((mn) == (mx)) ? (1.0) : ((mn) / (mx)));
  return std::complex<double>(
      (((std::abs(sum_fast2sum_high)) < (half))
           ? ((half) * (std::log1p(sum_fast2sum_high)))
           : ((std::log(mx)) +
              ((half) * (std::log1p((mn_over_mx) * (mn_over_mx)))))),
      std::atan2(y, x));
}