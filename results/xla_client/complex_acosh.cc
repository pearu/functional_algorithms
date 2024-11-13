// This file is generated using functional_algorithms tool (0.11.0), see
//   https://github.com/pearu/functional_algorithms
// for more information.



#include "xla/client/lib/constants.h"
#include "xla/client/xla_builder.h"
#include <limits>


template <typename FloatType>
XlaOp complex_acosh_0(XlaOp z) {
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
  XlaOp _mx_0_ = Max(abs_xp1, y);
  XlaOp mn = Min(abs_xp1, y);
  XlaOp sqrt_two = ScalarLike(signed_x, std::sqrt(2));
  XlaOp _r_0_ = Square(Div(mn, _mx_0_));
  XlaOp sqa = Sqrt(Add(one, _r_0_));
  XlaOp two = ScalarLike(signed_x, two_);
  XlaOp r = Select(
      Eq(_mx_0_, mn), Mul(sqrt_two, _mx_0_),
      Select(And(Eq(sqa, one), Gt(_r_0_, zero)),
             Add(_mx_0_, Div(Mul(_mx_0_, _r_0_), two)), Mul(_mx_0_, sqa)));
  XlaOp abs_xm1 = Abs(xm1);
  XlaOp _mx_1_ = Max(abs_xm1, y);
  XlaOp _mn_0_ = Min(abs_xm1, y);
  XlaOp _r_1_ = Square(Div(_mn_0_, _mx_1_));
  XlaOp _sqa_0_ = Sqrt(Add(one, _r_1_));
  XlaOp s = Select(
      Eq(_mx_1_, _mn_0_), Mul(sqrt_two, _mx_1_),
      Select(And(Eq(_sqa_0_, one), Gt(_r_1_, zero)),
             Add(_mx_1_, Div(Mul(_mx_1_, _r_1_), two)), Mul(_mx_1_, _sqa_0_)));
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
  XlaOp _imag_0_ = Atan2(
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
      Select(Lt(signed_y, ScalarLike(signed_y, zero_)), Neg(_imag_0_),
             _imag_0_));
}