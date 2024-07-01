// This file is generated using functional_algorithms tool (0.4.0), see
//   https://github.com/pearu/functional_algorithms
// for more information.



#include "xla/client/lib/constants.h"
#include "xla/client/xla_builder.h"
#include <limits>


template <typename FloatType>
XlaOp acosh_0(XlaOp z) {
  FloatType constant_2 = 2;
  XlaOp constant_constant_1 = ScalarLike(z, 1);
  XlaOp sqrt_subtract_z_constant_constant_1 = Sqrt(Sub(z, constant_constant_1));
  return Select(Ge(z, ScalarLike(z, (std::numeric_limits<FloatType>::max()) /
                                        (constant_2))),
                Add(ScalarLike(z, std::log(constant_2)), Log(z)),
                Log1p(Mul(sqrt_subtract_z_constant_constant_1,
                          Add(Sqrt(Add(z, constant_constant_1)),
                              sqrt_subtract_z_constant_constant_1))));
}

template <typename FloatType>
XlaOp acosh_1(XlaOp z) {
  XlaOp signed_y = Imag(z);
  XlaOp y = Abs(signed_y);
  XlaOp signed_x = Real(z);
  XlaOp x = Abs(signed_x);
  FloatType safe_max_ =
      (std::sqrt(std::numeric_limits<FloatType>::max())) / (8);
  XlaOp safe_max_opt =
      Select(Lt(x, ScalarLike(signed_x, (safe_max_) * (1000000000000.0))),
             ScalarLike(signed_x, (safe_max_) * (1e-06)),
             ScalarLike(signed_x, (safe_max_) * (100.0)));
  XlaOp y_gt_safe_max_opt = Ge(y, safe_max_opt);
  XlaOp mx = Select(y_gt_safe_max_opt, y, x);
  XlaOp safe_max = ScalarLike(signed_x, safe_max_);
  FloatType two_ = 2;
  XlaOp half = ScalarLike(signed_x, 0.5);
  FloatType zero_ = 0;
  XlaOp zero = ScalarLike(signed_x, zero_);
  XlaOp xoy = Select(
      And(y_gt_safe_max_opt,
          Not(Eq(y, ScalarLike(signed_y,
                               std::numeric_limits<FloatType>::infinity())))),
      Div(x, y), zero);
  XlaOp one = ScalarLike(signed_x, 1);
  XlaOp logical_and_lt_y_safe_min_lt_x_one = And(
      Lt(y,
         ScalarLike(signed_x,
                    (std::sqrt(std::numeric_limits<FloatType>::min())) * (4))),
      Lt(x, one));
  XlaOp xp1 = Add(x, one);
  XlaOp xm1 = Sub(x, one);
  XlaOp abs_xp1 = Abs(xp1);
  XlaOp _hypot_1_mx = Max(abs_xp1, y);
  XlaOp mn = Min(abs_xp1, y);
  XlaOp sqrt_two = ScalarLike(signed_x, std::sqrt(two_));
  XlaOp _hypot_1_r = Square(Div(mn, _hypot_1_mx));
  XlaOp sqa = Sqrt(Add(one, _hypot_1_r));
  XlaOp two = ScalarLike(signed_x, two_);
  XlaOp r =
      Select(Eq(_hypot_1_mx, mn), Mul(sqrt_two, _hypot_1_mx),
             Select(And(Eq(sqa, one), Gt(_hypot_1_r, zero)),
                    Add(_hypot_1_mx, Div(Mul(_hypot_1_mx, _hypot_1_r), two)),
                    Mul(_hypot_1_mx, sqa)));
  XlaOp abs_xm1 = Abs(xm1);
  XlaOp _hypot_2_mx = Max(abs_xm1, y);
  XlaOp _hypot_2_mn = Min(abs_xm1, y);
  XlaOp _hypot_2_r = Square(Div(_hypot_2_mn, _hypot_2_mx));
  XlaOp _hypot_2_sqa = Sqrt(Add(one, _hypot_2_r));
  XlaOp s =
      Select(Eq(_hypot_2_mx, _hypot_2_mn), Mul(sqrt_two, _hypot_2_mx),
             Select(And(Eq(_hypot_2_sqa, one), Gt(_hypot_2_r, zero)),
                    Add(_hypot_2_mx, Div(Mul(_hypot_2_mx, _hypot_2_r), two)),
                    Mul(_hypot_2_mx, _hypot_2_sqa)));
  XlaOp a = Mul(half, Add(r, s));
  XlaOp ap1 = Add(a, one);
  XlaOp yy = Mul(y, y);
  XlaOp half_yy = Mul(half, yy);
  XlaOp rpxp1 = Add(r, xp1);
  XlaOp divide_half_yy_rpxp1 = Div(half_yy, rpxp1);
  XlaOp spxm1 = Add(s, xm1);
  XlaOp smxm1 = Sub(s, xm1);
  XlaOp x_ge_1_or_not = Select(
      Ge(x, one), Add(divide_half_yy_rpxp1, Mul(half, spxm1)),
      Select(Le(a, ScalarLike(signed_x, 1.5)),
             Add(divide_half_yy_rpxp1, Div(half_yy, smxm1)), Sub(a, one)));
  XlaOp am1 = Select(logical_and_lt_y_safe_min_lt_x_one,
                     Neg(Div(Mul(xp1, xm1), ap1)), x_ge_1_or_not);
  XlaOp sq = Sqrt(Mul(am1, ap1));
  XlaOp half_apx = Mul(half, Add(a, x));
  XlaOp imag = Atan2(
      Select(Ge(Max(x, y), safe_max), y,
             Select(Le(x, one), Sqrt(Mul(half_apx, Add(Div(yy, rpxp1), smxm1))),
                    Mul(y, Sqrt(Add(Div(half_apx, rpxp1),
                                    Div(half_apx, spxm1)))))),
      signed_x);
  return Complex(
      Select(Ge(mx, Select(y_gt_safe_max_opt, safe_max_opt, safe_max)),
             Add(Add(ScalarLike(signed_x, std::log(two_)), Log(mx)),
                 Mul(half, Log1p(Mul(xoy, xoy)))),
             Select(logical_and_lt_y_safe_min_lt_x_one, Div(y, sq),
                    Log1p(Add(am1, sq)))),
      Select(Lt(signed_y, ScalarLike(signed_y, zero_)), Neg(imag), imag));
}