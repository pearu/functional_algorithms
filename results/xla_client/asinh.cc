// This file is generated using functional_algorithms tool (0.10.2.dev9+g7001467.d20241002), see
//   https://github.com/pearu/functional_algorithms
// for more information.



#include "xla/client/lib/constants.h"
#include "xla/client/xla_builder.h"
#include <limits>


template <typename FloatType>
XlaOp asinh_0(XlaOp z) {
  XlaOp ax = Abs(z);
  XlaOp ax2 = Mul(ax, ax);
  XlaOp one = ScalarLike(z, 1);
  return Mul(
      Sign(z),
      Select(Ge(ax, ScalarLike(
                        z, std::sqrt(std::numeric_limits<FloatType>::max()))),
             Add(ScalarLike(z, std::log(2)), Log(ax)),
             Log1p(Add(ax, Div(ax2, Add(one, Sqrt(Add(one, ax2))))))));
}

template <typename FloatType>
XlaOp asinh_1(XlaOp z) {
  XlaOp signed_y = Real(z);
  FloatType zero_ = 0;
  XlaOp y = Abs(signed_y);
  XlaOp signed_y__0 = Imag(z);
  XlaOp x = Abs(Neg(signed_y__0));
  FloatType safe_max_ =
      (std::sqrt(std::numeric_limits<FloatType>::max())) / (8);
  XlaOp safe_max_opt =
      Select(Lt(x, ScalarLike(signed_y__0, (safe_max_) * (1000000000000.0))),
             ScalarLike(signed_y__0, (safe_max_) * (1e-06)),
             ScalarLike(signed_y__0, (safe_max_) * (100.0)));
  XlaOp y_gt_safe_max_opt = Ge(y, safe_max_opt);
  XlaOp mx = Select(y_gt_safe_max_opt, y, x);
  XlaOp safe_max = ScalarLike(signed_y__0, safe_max_);
  FloatType two_ = 2;
  XlaOp half = ScalarLike(signed_y__0, 0.5);
  XlaOp zero = ScalarLike(signed_y__0, zero_);
  XlaOp xoy = Select(
      And(y_gt_safe_max_opt,
          Not(Eq(y, ScalarLike(signed_y,
                               std::numeric_limits<FloatType>::infinity())))),
      Div(x, y), zero);
  XlaOp one = ScalarLike(signed_y__0, 1);
  XlaOp logical_and_lt_y_safe_min_lt_x_one = And(
      Lt(y,
         ScalarLike(signed_y__0,
                    (std::sqrt(std::numeric_limits<FloatType>::min())) * (4))),
      Lt(x, one));
  XlaOp xp1 = Add(x, one);
  XlaOp xm1 = Sub(x, one);
  XlaOp abs_xp1 = Abs(xp1);
  XlaOp mx__0 = Max(abs_xp1, y);
  XlaOp mn = Min(abs_xp1, y);
  XlaOp sqrt_two = ScalarLike(signed_y__0, std::sqrt(2));
  XlaOp r__0 = Square(Div(mn, mx__0));
  XlaOp sqa = Sqrt(Add(one, r__0));
  XlaOp two = ScalarLike(signed_y__0, two_);
  XlaOp r =
      Select(Eq(mx__0, mn), Mul(sqrt_two, mx__0),
             Select(And(Eq(sqa, one), Gt(r__0, zero)),
                    Add(mx__0, Div(Mul(mx__0, r__0), two)), Mul(mx__0, sqa)));
  XlaOp abs_xm1 = Abs(xm1);
  XlaOp mx__1 = Max(abs_xm1, y);
  XlaOp mn__0 = Min(abs_xm1, y);
  XlaOp r__1 = Square(Div(mn__0, mx__1));
  XlaOp sqa__0 = Sqrt(Add(one, r__1));
  XlaOp s = Select(
      Eq(mx__1, mn__0), Mul(sqrt_two, mx__1),
      Select(And(Eq(sqa__0, one), Gt(r__1, zero)),
             Add(mx__1, Div(Mul(mx__1, r__1), two)), Mul(mx__1, sqa__0)));
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
      Select(Le(a, ScalarLike(signed_y__0, 1.5)),
             Add(divide_half_yy_rpxp1, Div(half_yy, smxm1)), Sub(a, one)));
  XlaOp am1 = Select(logical_and_lt_y_safe_min_lt_x_one,
                     Neg(Div(Mul(xp1, xm1), ap1)), x_ge_1_or_not);
  XlaOp sq = Sqrt(Mul(am1, ap1));
  XlaOp imag__0 =
      Select(Ge(mx, Select(y_gt_safe_max_opt, safe_max_opt, safe_max)),
             Add(Add(ScalarLike(signed_y__0, std::log(two_)), Log(mx)),
                 Mul(half, Log1p(Mul(xoy, xoy)))),
             Select(logical_and_lt_y_safe_min_lt_x_one, Div(y, sq),
                    Log1p(Add(am1, sq))));
  XlaOp half_apx = Mul(half, Add(a, x));
  return Complex(
      Select(Lt(signed_y, ScalarLike(signed_y, zero_)), Neg(imag__0), imag__0),
      Atan2(signed_y__0,
            Select(Ge(Max(x, y), safe_max), y,
                   Select(Le(x, one),
                          Sqrt(Mul(half_apx, Add(Div(yy, rpxp1), smxm1))),
                          Mul(y, Sqrt(Add(Div(half_apx, rpxp1),
                                          Div(half_apx, spxm1))))))));
}