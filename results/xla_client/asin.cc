// This file is generated using functional_algorithms tool (0.11.0), see
//   https://github.com/pearu/functional_algorithms
// for more information.



#include "xla/client/lib/constants.h"
#include "xla/client/xla_builder.h"
#include <limits>


template <typename FloatType>
XlaOp asin_0(XlaOp z) {
  XlaOp one = ScalarLike(z, 1);
  XlaOp ta = Atan2(z, Add(one, Sqrt(Mul(Sub(one, z), Add(one, z)))));
  return Add(ta, ta);
}

template <typename FloatType>
XlaOp asin_1(XlaOp z) {
  XlaOp signed_x = Real(z);
  XlaOp x = Abs(signed_x);
  XlaOp signed_y = Imag(z);
  XlaOp y = Abs(signed_y);
  FloatType safe_max_ =
      (std::sqrt(std::numeric_limits<FloatType>::max())) / (8);
  XlaOp safe_max = ScalarLike(signed_x, safe_max_);
  XlaOp one = ScalarLike(signed_x, 1);
  XlaOp half = ScalarLike(signed_x, 0.5);
  XlaOp xp1 = Add(x, one);
  XlaOp abs_xp1 = Abs(xp1);
  XlaOp mx = Max(abs_xp1, y);
  XlaOp mn = Min(abs_xp1, y);
  XlaOp sqrt_two = ScalarLike(signed_x, std::sqrt(2));
  XlaOp _r_0_ = Square(Div(mn, mx));
  XlaOp sqa = Sqrt(Add(one, _r_0_));
  XlaOp constant_constant_0 = ScalarLike(signed_x, 0);
  FloatType two_ = 2;
  XlaOp two = ScalarLike(signed_x, two_);
  XlaOp r = Select(Eq(mx, mn), Mul(sqrt_two, mx),
                   Select(And(Eq(sqa, one), Gt(_r_0_, constant_constant_0)),
                          Add(mx, Div(Mul(mx, _r_0_), two)), Mul(mx, sqa)));
  XlaOp xm1 = Sub(x, one);
  XlaOp abs_xm1 = Abs(xm1);
  XlaOp _mx_0_ = Max(abs_xm1, y);
  XlaOp _mn_0_ = Min(abs_xm1, y);
  XlaOp _r_1_ = Square(Div(_mn_0_, _mx_0_));
  XlaOp _sqa_0_ = Sqrt(Add(one, _r_1_));
  XlaOp s = Select(
      Eq(_mx_0_, _mn_0_), Mul(sqrt_two, _mx_0_),
      Select(And(Eq(_sqa_0_, one), Gt(_r_1_, constant_constant_0)),
             Add(_mx_0_, Div(Mul(_mx_0_, _r_1_), two)), Mul(_mx_0_, _sqa_0_)));
  XlaOp a = Mul(half, Add(r, s));
  XlaOp half_apx = Mul(half, Add(a, x));
  XlaOp yy = Mul(y, y);
  XlaOp rpxp1 = Add(r, xp1);
  XlaOp smxm1 = Sub(s, xm1);
  XlaOp spxm1 = Add(s, xm1);
  XlaOp real = Atan2(
      signed_x,
      Select(Ge(Max(x, y), safe_max), y,
             Select(Le(x, one), Sqrt(Mul(half_apx, Add(Div(yy, rpxp1), smxm1))),
                    Mul(y, Sqrt(Add(Div(half_apx, rpxp1),
                                    Div(half_apx, spxm1)))))));
  XlaOp safe_max_opt =
      Select(Lt(x, ScalarLike(signed_x, (safe_max_) * (1000000000000.0))),
             ScalarLike(signed_x, (safe_max_) * (1e-06)),
             ScalarLike(signed_x, (safe_max_) * (100.0)));
  XlaOp y_gt_safe_max_opt = Ge(y, safe_max_opt);
  XlaOp __asin_acos_kernel_1_mx_0_ = Select(y_gt_safe_max_opt, y, x);
  XlaOp xoy = Select(
      And(y_gt_safe_max_opt,
          Not(Eq(y, ScalarLike(signed_y,
                               std::numeric_limits<FloatType>::infinity())))),
      Div(x, y), constant_constant_0);
  XlaOp logical_and_lt_y_safe_min_lt_x_one = And(
      Lt(y,
         ScalarLike(signed_x,
                    (std::sqrt(std::numeric_limits<FloatType>::min())) * (4))),
      Lt(x, one));
  XlaOp ap1 = Add(a, one);
  XlaOp half_yy = Mul(half, yy);
  XlaOp divide_half_yy_rpxp1 = Div(half_yy, rpxp1);
  XlaOp x_ge_1_or_not = Select(
      Ge(x, one), Add(divide_half_yy_rpxp1, Mul(half, spxm1)),
      Select(Le(a, ScalarLike(signed_x, 1.5)),
             Add(divide_half_yy_rpxp1, Div(half_yy, smxm1)), Sub(a, one)));
  XlaOp am1 = Select(logical_and_lt_y_safe_min_lt_x_one,
                     Neg(Div(Mul(xp1, xm1), ap1)), x_ge_1_or_not);
  XlaOp sq = Sqrt(Mul(am1, ap1));
  XlaOp imag = Select(Ge(__asin_acos_kernel_1_mx_0_,
                         Select(y_gt_safe_max_opt, safe_max_opt, safe_max)),
                      Add(Add(ScalarLike(signed_x, std::log(two_)),
                              Log(__asin_acos_kernel_1_mx_0_)),
                          Mul(half, Log1p(Mul(xoy, xoy)))),
                      Select(logical_and_lt_y_safe_min_lt_x_one, Div(y, sq),
                             Log1p(Add(am1, sq))));
  return Complex(real,
                 Select(Lt(signed_y, constant_constant_0), Neg(imag), imag));
}