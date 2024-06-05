// This file is generated using functional_algorithms tool (0.1.2.dev7+g332df57.d20240604), see
//   https://github.com/pearu/functional_algorithms
// for more information.



#include <limits>


template <typename FloatType>
XLAOp hypot_0(XLAOp x, XLAOp y) {
  XLAOp abs_x = Abs(x);
  XLAOp abs_y = Abs(y);
  XLAOp mx = Max(abs_x, abs_y);
  XLAOp mn = Min(abs_x, abs_y);
  FloatType constant_2 = 2;
  XLAOp constant_constant_1 = ScalarLike(x, 1);
  XLAOp r = Square(Div(mn, mx));
  XLAOp sqa = Sqrt(Add(constant_constant_1, r));
  return Select(
      Eq(mx, mn), Mul(ScalarLike(x, std::sqrt(constant_2)), mx),
      Select(And(Eq(sqa, constant_constant_1), Gt(r, ScalarLike(x, 0))),
             Add(mx, Div(Mul(mx, r), ScalarLike(x, constant_2))),
             Mul(mx, sqa)));
}