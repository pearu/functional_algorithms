// This file is generated using functional_algorithms tool (0.11.1), see
//   https://github.com/pearu/functional_algorithms
// for more information.



#include "xla/client/lib/constants.h"
#include "xla/client/xla_builder.h"
#include <limits>


template <typename FloatType>
XlaOp complex_sqrt_0(XlaOp z) {
  XlaOp x = Real(z);
  XlaOp constant_constant_0 = ScalarLike(x, 0);
  XlaOp ax = Abs(x);
  XlaOp y = Imag(z);
  XlaOp ay = Abs(y);
  XlaOp eq_ax_ay = Eq(ax, ay);
  XlaOp sq_ax = Sqrt(ax);
  FloatType sq_2_ = std::sqrt(2);
  FloatType sq_12_ = std::sqrt((1) + (sq_2_));
  XlaOp sq_2 = ScalarLike(x, sq_2_);
  XlaOp mx = Max(ax, ay);
  XlaOp mn = Min(ax, ay);
  XlaOp one = ScalarLike(x, 1);
  XlaOp r = Square(Div(mn, mx));
  XlaOp sqa = Sqrt(Add(one, r));
  XlaOp two = ScalarLike(x, 2);
  XlaOp u_general =
      Sqrt(Add(Div(Select(Eq(mx, mn), Mul(sq_2, mx),
                          Select(And(Eq(sqa, one), Gt(r, constant_constant_0)),
                                 Add(mx, Div(Mul(mx, r), two)), Mul(mx, sqa))),
                   two),
               Div(ax, two)));
  XlaOp
      logical_or_eq_u_general_constant_constant_0_eq_u_general_constant_constant_posinf =
          Or(Eq(u_general, constant_constant_0),
             Eq(u_general,
                ScalarLike(x, std::numeric_limits<FloatType>::infinity())));
  XlaOp gt_ax_ay = Gt(ax, ay);
  XlaOp lt_ax_ay = Lt(ax, ay);
  XlaOp _r_0_ =
      Select(eq_ax_ay, one, Select(lt_ax_ay, Div(ax, ay), Div(ay, ax)));
  XlaOp abs__r_0_ = Abs(_r_0_);
  XlaOp _mx_0_ = Max(one, abs__r_0_);
  XlaOp _mn_0_ = Min(one, abs__r_0_);
  XlaOp _r_1_ = Square(Div(_mn_0_, _mx_0_));
  XlaOp _sqa_0_ = Sqrt(Add(one, _r_1_));
  XlaOp h = Select(
      Eq(_mx_0_, _mn_0_), Mul(sq_2, _mx_0_),
      Select(And(Eq(_sqa_0_, one), Gt(_r_1_, constant_constant_0)),
             Add(_mx_0_, Div(Mul(_mx_0_, _r_1_), two)), Mul(_mx_0_, _sqa_0_)));
  XlaOp sq_1h = Sqrt(Add(one, h));
  XlaOp sq_ay = Sqrt(ay);
  XlaOp sq_rh = Sqrt(Add(_r_0_, h));
  XlaOp u = Select(
      eq_ax_ay, Div(Mul(sq_ax, ScalarLike(x, sq_12_)), sq_2),
      Select(
          logical_or_eq_u_general_constant_constant_0_eq_u_general_constant_constant_posinf,
          Select(gt_ax_ay, Mul(sq_ax, Div(sq_1h, sq_2)),
                 Mul(sq_ay, Div(sq_rh, sq_2))),
          u_general));
  XlaOp ay_div_u = Select(
      eq_ax_ay, Div(sq_ay, ScalarLike(x, (sq_12_) * (sq_2_))),
      Select(
          logical_or_eq_u_general_constant_constant_0_eq_u_general_constant_constant_posinf,
          Select(gt_ax_ay,
                 Div(Mul(sq_ay, Select(eq_ax_ay, one,
                                       Select(lt_ax_ay, Div(sq_ax, sq_ay),
                                              Div(sq_ay, sq_ax)))),
                     Mul(sq_1h, sq_2)),
                 Div(sq_ay, Mul(sq_rh, sq_2))),
          Div(ay, Mul(u_general, two))));
  XlaOp lt_y_constant_constant_0 = Lt(y, constant_constant_0);
  return Complex(
      Select(Ge(x, constant_constant_0), u, ay_div_u),
      Select(Lt(x, constant_constant_0),
             Select(lt_y_constant_constant_0, Neg(u), u),
             Select(lt_y_constant_constant_0, Neg(ay_div_u), ay_div_u)));
}