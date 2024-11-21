// This file is generated using functional_algorithms tool (0.11.1), see
//   https://github.com/pearu/functional_algorithms
// for more information.



#include "xla/client/lib/constants.h"
#include "xla/client/xla_builder.h"
#include <limits>


template <typename FloatType>
XlaOp complex_log1p_0(XlaOp z) {
  XlaOp x = Real(z);
  XlaOp ax = Abs(x);
  XlaOp y = Imag(z);
  XlaOp ay = Abs(y);
  XlaOp mx = Max(ax, ay);
  FloatType constant_largest = std::numeric_limits<FloatType>::max();
  XlaOp half = ScalarLike(x, 0.5);
  XlaOp mn = Min(ax, ay);
  XlaOp one = ScalarLike(x, 1);
  XlaOp r = Div(mn, mx);
  XlaOp xp1 = Add(x, one);
  XlaOp square_dekker_high = Mul(y, y);
  XlaOp x2h = Add(x, x);
  XlaOp _add_2sum_high_2_ = Add(x2h, square_dekker_high);
  XlaOp _square_dekker_high_0_ = Mul(x, x);
  XlaOp _add_2sum_high_1_ = Add(_add_2sum_high_2_, _square_dekker_high_0_);
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
  XlaOp _add_2sum_high_0_ = Add(_add_2sum_high_1_, square_dekker_low);
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
  XlaOp add_2sum_high = Add(_add_2sum_high_0_, _square_dekker_low_0_);
  XlaOp subtract__add_2sum_high_2__x2h = Sub(_add_2sum_high_2_, x2h);
  XlaOp add_2sum_low =
      Add(Sub(x2h, Sub(_add_2sum_high_2_, subtract__add_2sum_high_2__x2h)),
          Sub(square_dekker_high, subtract__add_2sum_high_2__x2h));
  XlaOp subtract__add_2sum_high_1___add_2sum_high_2_ =
      Sub(_add_2sum_high_1_, _add_2sum_high_2_);
  XlaOp _add_2sum_low_0_ = Add(
      Sub(_add_2sum_high_2_,
          Sub(_add_2sum_high_1_, subtract__add_2sum_high_1___add_2sum_high_2_)),
      Sub(_square_dekker_high_0_,
          subtract__add_2sum_high_1___add_2sum_high_2_));
  XlaOp subtract__add_2sum_high_0___add_2sum_high_1_ =
      Sub(_add_2sum_high_0_, _add_2sum_high_1_);
  XlaOp _add_2sum_low_1_ = Add(
      Sub(_add_2sum_high_1_,
          Sub(_add_2sum_high_0_, subtract__add_2sum_high_0___add_2sum_high_1_)),
      Sub(square_dekker_low, subtract__add_2sum_high_0___add_2sum_high_1_));
  XlaOp subtract_add_2sum_high__add_2sum_high_0_ =
      Sub(add_2sum_high, _add_2sum_high_0_);
  XlaOp _add_2sum_low_2_ =
      Add(Sub(_add_2sum_high_0_,
              Sub(add_2sum_high, subtract_add_2sum_high__add_2sum_high_0_)),
          Sub(_square_dekker_low_0_, subtract_add_2sum_high__add_2sum_high_0_));
  XlaOp sum_2sum_high =
      Add(add_2sum_high,
          Add(Add(Add(add_2sum_low, _add_2sum_low_0_), _add_2sum_low_1_),
              _add_2sum_low_2_));
  return Complex(
      Select(Gt(mx, ScalarLike(x, (std::sqrt(constant_largest)) * (0.01))),
             Add(Log(mx), Mul(half, Log1p(Select(Eq(mn, mx), one, Mul(r, r))))),
             Select(Lt(Add(Abs(xp1), ay), ScalarLike(x, 0.2)),
                    Mul(half, Log(Add(Mul(xp1, xp1), square_dekker_high))),
                    Mul(half, Log1p(sum_2sum_high)))),
      Atan2(y, xp1));
}