// This file is generated using functional_algorithms tool (0.11.1), see
//   https://github.com/pearu/functional_algorithms
// for more information.



#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>


std::complex<float> log1p_2(std::complex<float> z) {
  float x = (z).real();
  float ax = std::abs(x);
  float y = (z).imag();
  float ay = std::abs(y);
  float mx = std::max(ax, ay);
  float half = 0.5;
  float mn = std::min(ax, ay);
  float one = 1.0;
  float r = (mn) / (mx);
  float xp1 = (x) + (one);
  float square_dekker_high = (y) * (y);
  float x2h = (x) + (x);
  float _add_2sum_high_2_ = (x2h) + (square_dekker_high);
  float _square_dekker_high_0_ = (x) * (x);
  float _add_2sum_high_1_ = (_add_2sum_high_2_) + (_square_dekker_high_0_);
  float veltkamp_splitter_constant = 4097.0;
  float multiply_veltkamp_splitter_constant_y =
      (veltkamp_splitter_constant) * (y);
  float yh = (multiply_veltkamp_splitter_constant_y) +
             ((y) - (multiply_veltkamp_splitter_constant_y));
  float yl = (y) - (yh);
  float multiply_yh_yl = (yh) * (yl);
  float square_dekker_low =
      ((((-(square_dekker_high)) + ((yh) * (yh))) + (multiply_yh_yl)) +
       (multiply_yh_yl)) +
      ((yl) * (yl));
  float _add_2sum_high_0_ = (_add_2sum_high_1_) + (square_dekker_low);
  float multiply_veltkamp_splitter_constant_x =
      (veltkamp_splitter_constant) * (x);
  float xh = (multiply_veltkamp_splitter_constant_x) +
             ((x) - (multiply_veltkamp_splitter_constant_x));
  float xl = (x) - (xh);
  float multiply_xh_xl = (xh) * (xl);
  float _square_dekker_low_0_ =
      ((((-(_square_dekker_high_0_)) + ((xh) * (xh))) + (multiply_xh_xl)) +
       (multiply_xh_xl)) +
      ((xl) * (xl));
  float add_2sum_high = (_add_2sum_high_0_) + (_square_dekker_low_0_);
  float subtract__add_2sum_high_2__x2h = (_add_2sum_high_2_) - (x2h);
  float add_2sum_low =
      ((x2h) - ((_add_2sum_high_2_) - (subtract__add_2sum_high_2__x2h))) +
      ((square_dekker_high) - (subtract__add_2sum_high_2__x2h));
  float subtract__add_2sum_high_1___add_2sum_high_2_ =
      (_add_2sum_high_1_) - (_add_2sum_high_2_);
  float _add_2sum_low_0_ =
      ((_add_2sum_high_2_) -
       ((_add_2sum_high_1_) - (subtract__add_2sum_high_1___add_2sum_high_2_))) +
      ((_square_dekker_high_0_) -
       (subtract__add_2sum_high_1___add_2sum_high_2_));
  float subtract__add_2sum_high_0___add_2sum_high_1_ =
      (_add_2sum_high_0_) - (_add_2sum_high_1_);
  float _add_2sum_low_1_ =
      ((_add_2sum_high_1_) -
       ((_add_2sum_high_0_) - (subtract__add_2sum_high_0___add_2sum_high_1_))) +
      ((square_dekker_low) - (subtract__add_2sum_high_0___add_2sum_high_1_));
  float subtract_add_2sum_high__add_2sum_high_0_ =
      (add_2sum_high) - (_add_2sum_high_0_);
  float _add_2sum_low_2_ =
      ((_add_2sum_high_0_) -
       ((add_2sum_high) - (subtract_add_2sum_high__add_2sum_high_0_))) +
      ((_square_dekker_low_0_) - (subtract_add_2sum_high__add_2sum_high_0_));
  float sum_2sum_high =
      (add_2sum_high) +
      ((((add_2sum_low) + (_add_2sum_low_0_)) + (_add_2sum_low_1_)) +
       (_add_2sum_low_2_));
  return std::complex<float>(
      (((mx) > (1.8446742e+17))
           ? ((std::log(mx)) +
              ((half) * (std::log1p((((mn) == (mx)) ? (one) : ((r) * (r)))))))
           : (((((std::abs(xp1)) + (ay)) < (0.2))
                   ? ((half) *
                      (std::log(((xp1) * (xp1)) + (square_dekker_high))))
                   : ((half) * (std::log1p(sum_2sum_high)))))),
      std::atan2(y, xp1));
}

std::complex<double> log1p_3(std::complex<double> z) {
  double x = (z).real();
  double ax = std::abs(x);
  double y = (z).imag();
  double ay = std::abs(y);
  double mx = std::max(ax, ay);
  double half = 0.5;
  double mn = std::min(ax, ay);
  double one = 1.0;
  double r = (mn) / (mx);
  double xp1 = (x) + (one);
  double square_dekker_high = (y) * (y);
  double x2h = (x) + (x);
  double _add_2sum_high_2_ = (x2h) + (square_dekker_high);
  double _square_dekker_high_0_ = (x) * (x);
  double _add_2sum_high_1_ = (_add_2sum_high_2_) + (_square_dekker_high_0_);
  double veltkamp_splitter_constant = 134217729.0;
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
  double _add_2sum_high_0_ = (_add_2sum_high_1_) + (square_dekker_low);
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
  double add_2sum_high = (_add_2sum_high_0_) + (_square_dekker_low_0_);
  double subtract__add_2sum_high_2__x2h = (_add_2sum_high_2_) - (x2h);
  double add_2sum_low =
      ((x2h) - ((_add_2sum_high_2_) - (subtract__add_2sum_high_2__x2h))) +
      ((square_dekker_high) - (subtract__add_2sum_high_2__x2h));
  double subtract__add_2sum_high_1___add_2sum_high_2_ =
      (_add_2sum_high_1_) - (_add_2sum_high_2_);
  double _add_2sum_low_0_ =
      ((_add_2sum_high_2_) -
       ((_add_2sum_high_1_) - (subtract__add_2sum_high_1___add_2sum_high_2_))) +
      ((_square_dekker_high_0_) -
       (subtract__add_2sum_high_1___add_2sum_high_2_));
  double subtract__add_2sum_high_0___add_2sum_high_1_ =
      (_add_2sum_high_0_) - (_add_2sum_high_1_);
  double _add_2sum_low_1_ =
      ((_add_2sum_high_1_) -
       ((_add_2sum_high_0_) - (subtract__add_2sum_high_0___add_2sum_high_1_))) +
      ((square_dekker_low) - (subtract__add_2sum_high_0___add_2sum_high_1_));
  double subtract_add_2sum_high__add_2sum_high_0_ =
      (add_2sum_high) - (_add_2sum_high_0_);
  double _add_2sum_low_2_ =
      ((_add_2sum_high_0_) -
       ((add_2sum_high) - (subtract_add_2sum_high__add_2sum_high_0_))) +
      ((_square_dekker_low_0_) - (subtract_add_2sum_high__add_2sum_high_0_));
  double sum_2sum_high =
      (add_2sum_high) +
      ((((add_2sum_low) + (_add_2sum_low_0_)) + (_add_2sum_low_1_)) +
       (_add_2sum_low_2_));
  return std::complex<double>(
      (((mx) > (1.3407807929942595e+152))
           ? ((std::log(mx)) +
              ((half) * (std::log1p((((mn) == (mx)) ? (one) : ((r) * (r)))))))
           : (((((std::abs(xp1)) + (ay)) < (0.2))
                   ? ((half) *
                      (std::log(((xp1) * (xp1)) + (square_dekker_high))))
                   : ((half) * (std::log1p(sum_2sum_high)))))),
      std::atan2(y, xp1));
}