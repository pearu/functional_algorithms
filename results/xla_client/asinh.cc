// This file is generated using functional_algorithms tool (0.4.0), see
//   https://github.com/pearu/functional_algorithms
// for more information.



#include "xla/client/lib/constants.h"
#include "xla/client/xla_builder.h"
#include <limits>


template <typename FloatType>
XlaOp asinh_0(XlaOp z) {
  FloatType zero_ = 0;
  XlaOp signed_x = ScalarLike(signed_y, zero_);
  XlaOp y = Abs(signed_y);
  FloatType abs_zero_ = std::abs(zero_);
  FloatType safe_max_ =
      (std::sqrt(std::numeric_limits<FloatType>::max())) / (8);
  XlaOp safe_max_opt =
      ScalarLike(signed_y, (((abs_zero_) < ((safe_max_) * (1000000000000.0)))
                                ? ((safe_max_) * (1e-06))
                                : ((safe_max_) * (100.0))));
  XlaOp y_gt_safe_max_opt = Ge(y, safe_max_opt);
  XlaOp x = ScalarLike(signed_y, abs_zero_);
  XlaOp mx = Select(y_gt_safe_max_opt, y, x);
  FloatType two_ = 2;
  XlaOp half = ScalarLike(signed_y, 0.5);
  XlaOp xoy = Select(
      And(y_gt_safe_max_opt,
          Not(Eq(y, ScalarLike(signed_y,
                               std::numeric_limits<FloatType>::infinity())))),
      Div(x, y), signed_x);
  FloatType one_ = 1;
  XlaOp logical_and_lt_y_safe_min_constant_lt_abs_zero__one_ = And(
      Lt(y,
         ScalarLike(signed_y,
                    (std::sqrt(std::numeric_limits<FloatType>::min())) * (4))),
      ScalarLike(signed_y, (abs_zero_) < (one_)));
  FloatType add_abs_zero__one_ = (abs_zero_) + (one_);
  FloatType subtract_abs_zero__one_ = (abs_zero_) - (one_);
  XlaOp constant_abs_add_abs_zero__one_ =
      ScalarLike(signed_y, std::abs(add_abs_zero__one_));
  XlaOp _hypot_1_mx = Max(constant_abs_add_abs_zero__one_, y);
  XlaOp mn = Min(constant_abs_add_abs_zero__one_, y);
  XlaOp sqrt_two = ScalarLike(signed_y, std::sqrt(two_));
  XlaOp one = ScalarLike(signed_y, one_);
  XlaOp _hypot_1_r = Square(Div(mn, _hypot_1_mx));
  XlaOp sqa = Sqrt(Add(one, _hypot_1_r));
  XlaOp two = ScalarLike(signed_y, two_);
  XlaOp r =
      Select(Eq(_hypot_1_mx, mn), Mul(sqrt_two, _hypot_1_mx),
             Select(And(Eq(sqa, one), Gt(_hypot_1_r, signed_x)),
                    Add(_hypot_1_mx, Div(Mul(_hypot_1_mx, _hypot_1_r), two)),
                    Mul(_hypot_1_mx, sqa)));
  XlaOp constant_abs_subtract_abs_zero__one_ =
      ScalarLike(signed_y, std::abs(subtract_abs_zero__one_));
  XlaOp _hypot_2_mx = Max(constant_abs_subtract_abs_zero__one_, y);
  XlaOp _hypot_2_mn = Min(constant_abs_subtract_abs_zero__one_, y);
  XlaOp _hypot_2_r = Square(Div(_hypot_2_mn, _hypot_2_mx));
  XlaOp _hypot_2_sqa = Sqrt(Add(one, _hypot_2_r));
  XlaOp s =
      Select(Eq(_hypot_2_mx, _hypot_2_mn), Mul(sqrt_two, _hypot_2_mx),
             Select(And(Eq(_hypot_2_sqa, one), Gt(_hypot_2_r, signed_x)),
                    Add(_hypot_2_mx, Div(Mul(_hypot_2_mx, _hypot_2_r), two)),
                    Mul(_hypot_2_mx, _hypot_2_sqa)));
  XlaOp a = Mul(half, Add(r, s));
  XlaOp ap1 = Add(a, one);
  XlaOp half_yy = Mul(half, Mul(y, y));
  XlaOp divide_half_yy_rpxp1 =
      Div(half_yy, Add(r, ScalarLike(signed_y, add_abs_zero__one_)));
  XlaOp xm1 = ScalarLike(signed_y, subtract_abs_zero__one_);
  XlaOp x_ge_1_or_not =
      Select(ScalarLike(signed_y, (abs_zero_) >= (one_)),
             Add(divide_half_yy_rpxp1, Mul(half, Add(s, xm1))),
             Select(Le(a, ScalarLike(signed_y, 1.5)),
                    Add(divide_half_yy_rpxp1, Div(half_yy, Sub(s, xm1))),
                    Sub(a, one)));
  XlaOp am1 =
      Select(logical_and_lt_y_safe_min_constant_lt_abs_zero__one_,
             Neg(Div(ScalarLike(signed_y, (add_abs_zero__one_) *
                                              (subtract_abs_zero__one_)),
                     ap1)),
             x_ge_1_or_not);
  XlaOp sq = Sqrt(Mul(am1, ap1));
  XlaOp imag =
      Select(Ge(mx, Select(y_gt_safe_max_opt, safe_max_opt,
                           ScalarLike(signed_y, safe_max_))),
             Add(Add(ScalarLike(signed_y, std::log(two_)), Log(mx)),
                 Mul(half, Log1p(Mul(xoy, xoy)))),
             Select(logical_and_lt_y_safe_min_constant_lt_abs_zero__one_,
                    Div(y, sq), Log1p(Add(am1, sq))));
  return Select(Lt(signed_y, signed_x), Neg(imag), imag);
}

template <typename FloatType>
XlaOp asinh_1(XlaOp z) {
  XlaOp w = Asin(Complex(Neg(Imag(z)), Real(z)));
  return Complex(Imag(w), Neg(Real(w)));
}