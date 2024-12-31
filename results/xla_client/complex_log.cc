// This file is generated using functional_algorithms tool (0.14.1.dev0+ge22be68.d20241231), see
//   https://github.com/pearu/functional_algorithms
// for more information.



#include "xla/client/lib/constants.h"
#include "xla/client/xla_builder.h"
#include <limits>


template <typename FloatType>
XlaOp complex_log_0(XlaOp z) {
  XlaOp constant_constant_fneg1 = ScalarLike(x, -1.0);
  XlaOp y = Imag(z);
  XlaOp square_dekker_high = Mul(y, y);
  XlaOp x = Real(z);
  XlaOp _square_dekker_high_0_ = Mul(x, x);
  XlaOp gt_square_dekker_high__square_dekker_high_0_ =
      Gt(square_dekker_high, _square_dekker_high_0_);
  XlaOp mxh = Select(gt_square_dekker_high__square_dekker_high_0_,
                     square_dekker_high, _square_dekker_high_0_);
  XlaOp _add_fast2sum_high_2_ = Add(constant_constant_fneg1, mxh);
  XlaOp mnh = Select(gt_square_dekker_high__square_dekker_high_0_,
                     _square_dekker_high_0_, square_dekker_high);
  XlaOp _add_fast2sum_high_1_ = Add(_add_fast2sum_high_2_, mnh);
  FloatType constant_largest = std::numeric_limits<FloatType>::max();
  XlaOp veltkamp_splitter_constant =
      ScalarLike(x, (((constant_largest) > (1e+308))
                         ? (134217729)
                         : ((((constant_largest) > (1e+38)) ? (4097) : (65)))));
  XlaOp multiply_veltkamp_splitter_constant_y =
      Mul(veltkamp_splitter_constant, y);
  XlaOp yh = Add(multiply_veltkamp_splitter_constant_y,
                 Sub(y, multiply_veltkamp_splitter_constant_y));
  XlaOp yl = Sub(y, yh);
  XlaOp multiply_yh_yl = Mul(yh, yl);
  XlaOp square_dekker_low =
      Add(Add(Add(Add(Neg(square_dekker_high), Mul(yh, yh)), multiply_yh_yl),
              multiply_yh_yl),
          Mul(yl, yl));
  XlaOp _add_fast2sum_high_0_ = Add(_add_fast2sum_high_1_, square_dekker_low);
  XlaOp multiply_veltkamp_splitter_constant_x =
      Mul(veltkamp_splitter_constant, x);
  XlaOp xh = Add(multiply_veltkamp_splitter_constant_x,
                 Sub(x, multiply_veltkamp_splitter_constant_x));
  XlaOp xl = Sub(x, xh);
  XlaOp multiply_xh_xl = Mul(xh, xl);
  XlaOp _square_dekker_low_0_ = Add(
      Add(Add(Add(Neg(_square_dekker_high_0_), Mul(xh, xh)), multiply_xh_xl),
          multiply_xh_xl),
      Mul(xl, xl));
  XlaOp add_fast2sum_high = Add(_add_fast2sum_high_0_, _square_dekker_low_0_);
  XlaOp add_fast2sum_low =
      Sub(mxh, Sub(_add_fast2sum_high_2_, constant_constant_fneg1));
  XlaOp _add_fast2sum_low_0_ =
      Sub(mnh, Sub(_add_fast2sum_high_1_, _add_fast2sum_high_2_));
  XlaOp _add_fast2sum_low_1_ =
      Sub(square_dekker_low, Sub(_add_fast2sum_high_0_, _add_fast2sum_high_1_));
  XlaOp _add_fast2sum_low_2_ =
      Sub(_square_dekker_low_0_, Sub(add_fast2sum_high, _add_fast2sum_high_0_));
  XlaOp sum_fast2sum_high = Add(
      add_fast2sum_high, Add(Add(Add(add_fast2sum_low, _add_fast2sum_low_0_),
                                 _add_fast2sum_low_1_),
                             _add_fast2sum_low_2_));
  XlaOp half = ScalarLike(x, 0.5);
  XlaOp abs_x = Abs(x);
  XlaOp abs_y = Abs(y);
  XlaOp mx = Max(abs_x, abs_y);
  XlaOp mn = Min(abs_x, abs_y);
  XlaOp mn_over_mx = Select(Eq(mn, mx), ScalarLike(x, 1.0), Div(mn, mx));
  return Complex(
      Select(Lt(Abs(sum_fast2sum_high), half),
             Mul(half, Log1p(sum_fast2sum_high)),
             Add(Log(mx), Mul(half, Log1p(Mul(mn_over_mx, mn_over_mx))))),
      Atan2(y, x));
}