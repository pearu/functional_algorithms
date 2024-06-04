// This file is generated using functional_algorithms tool (0.1.2.dev2+g1428951.d20240525), see
//   https://github.com/pearu/functional_algorithms
// for more information.



#include <limits>


XLAOp complex_asin_0(XLAOp z) {
  XLAOp signed_x = Real(z);
  XLAOp x = Abs(signed_x);
  XLAOp signed_y = Imag(z);
  XLAOp y = Abs(signed_y);
  XLAOp safe_max =
      Div(Sqrt(ScalarLike(signed_x, largest)), ScalarLike(signed_x, 8));
  XLAOp one = ScalarLike(signed_x, 1);
  XLAOp half = ScalarLike(signed_x, 0.5);
  XLAOp xp1 = Add(x, one);
  XLAOp abs_xp1 = Abs(xp1);
  XLAOp _hypot_1_mx = Max(abs_xp1, y);
  XLAOp mn = Min(abs_xp1, y);
  XLAOp two = ScalarLike(signed_x, 2);
  XLAOp sqrt_two = Sqrt(two);
  XLAOp _hypot_1_r = Square(Div(mn, _hypot_1_mx));
  XLAOp sqa = Sqrt(Add(one, _hypot_1_r));
  XLAOp zero = ScalarLike(signed_x, 0);
  XLAOp r =
      Select(Eq(_hypot_1_mx, mn), Mul(sqrt_two, _hypot_1_mx),
             Select(And(Eq(sqa, one), Gt(_hypot_1_r, zero)),
                    Add(_hypot_1_mx, Div(Mul(_hypot_1_mx, _hypot_1_r), two)),
                    Mul(_hypot_1_mx, sqa)));
  XLAOp xm1 = Sub(x, one);
  XLAOp abs_xm1 = Abs(xm1);
  XLAOp _hypot_2_mx = Max(abs_xm1, y);
  XLAOp _hypot_2_mn = Min(abs_xm1, y);
  XLAOp _hypot_2_r = Square(Div(_hypot_2_mn, _hypot_2_mx));
  XLAOp _hypot_2_sqa = Sqrt(Add(one, _hypot_2_r));
  XLAOp s =
      Select(Eq(_hypot_2_mx, _hypot_2_mn), Mul(sqrt_two, _hypot_2_mx),
             Select(And(Eq(_hypot_2_sqa, one), Gt(_hypot_2_r, zero)),
                    Add(_hypot_2_mx, Div(Mul(_hypot_2_mx, _hypot_2_r), two)),
                    Mul(_hypot_2_mx, _hypot_2_sqa)));
  XLAOp a = Mul(half, Add(r, s));
  XLAOp half_apx = Mul(half, Add(a, x));
  XLAOp yy = Mul(y, y);
  XLAOp rpxp1 = Add(r, xp1);
  XLAOp smxm1 = Sub(s, xm1);
  XLAOp spxm1 = Add(s, xm1);
  XLAOp real = Atan2(
      signed_x,
      Select(Ge(Max(x, y), safe_max), y,
             Select(Le(x, one), Sqrt(Mul(half_apx, Add(Div(yy, rpxp1), smxm1))),
                    Mul(y, Sqrt(Add(Div(half_apx, rpxp1),
                                    Div(half_apx, spxm1)))))));
  XLAOp safe_max_opt =
      Select(Lt(x, Mul(safe_max, ScalarLike(signed_x, 1000000000000.0))),
             Mul(safe_max, ScalarLike(signed_x, 1e-06)),
             Mul(safe_max, ScalarLike(signed_x, 100.0)));
  XLAOp y_gt_safe_max_opt = Ge(y, safe_max_opt);
  XLAOp mx = Select(y_gt_safe_max_opt, y, x);
  XLAOp xoy =
      Select(And(y_gt_safe_max_opt, Not(Eq(y, ScalarLike(signed_y, posinf)))),
             Div(x, y), zero);
  XLAOp logical_and_lt_y_safe_min_lt_x_one = And(
      Lt(y, Mul(Sqrt(ScalarLike(signed_x, smallest)), ScalarLike(signed_x, 4))),
      Lt(x, one));
  XLAOp ap1 = Add(a, one);
  XLAOp half_yy = Mul(half, yy);
  XLAOp divide_half_yy_rpxp1 = Div(half_yy, rpxp1);
  XLAOp x_ge_1_or_not = Select(
      Ge(x, one), Add(divide_half_yy_rpxp1, Mul(half, spxm1)),
      Select(Le(a, ScalarLike(signed_x, 1.5)),
             Add(divide_half_yy_rpxp1, Div(half_yy, smxm1)), Sub(a, one)));
  XLAOp am1 = Select(logical_and_lt_y_safe_min_lt_x_one,
                     Neg(Div(Mul(xp1, xm1), ap1)), x_ge_1_or_not);
  XLAOp sq = Sqrt(Mul(am1, ap1));
  XLAOp imag =
      Select(Ge(mx, Select(y_gt_safe_max_opt, safe_max_opt, safe_max)),
             Add(Add(Log(two), Log(mx)), Mul(half, Log1p(Mul(xoy, xoy)))),
             Select(logical_and_lt_y_safe_min_lt_x_one, Div(y, sq),
                    Log1p(Add(am1, sq))));
  return Complex(real, Select(Lt(signed_y, zero), Neg(imag), imag));
}