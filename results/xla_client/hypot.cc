// This file is generated using functional_algorithms tool (0.1.2.dev2+g1428951.d20240525), see
//   https://github.com/pearu/functional_algorithms
// for more information.



#include <limits>


XLAOp hypot_0(XLAOp x, XLAOp y) {
  XLAOp abs_x = Abs(x);
  XLAOp abs_y = Abs(y);
  XLAOp mx = Max(abs_x, abs_y);
  XLAOp mn = Min(abs_x, abs_y);
  XLAOp constant_2 = ScalarLike(x, 2);
  XLAOp constant_1 = ScalarLike(x, 1);
  XLAOp r = Square(Div(mn, mx));
  XLAOp sqa = Sqrt(Add(constant_1, r));
  return Select(Eq(mx, mn), Mul(Sqrt(constant_2), mx),
                Select(And(Eq(sqa, constant_1), Gt(r, ScalarLike(x, 0))),
                       Add(mx, Div(Mul(mx, r), constant_2)), Mul(mx, sqa)));
}