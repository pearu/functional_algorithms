// This file is generated using functional_algorithms tool (0.4.0), see
//   https://github.com/pearu/functional_algorithms
// for more information.



#include "xla/client/lib/constants.h"
#include "xla/client/xla_builder.h"
#include <limits>


template <typename FloatType>
XlaOp hypot_0(XlaOp x, XlaOp y) {
  XlaOp abs_x = Abs(x);
  XlaOp abs_y = Abs(y);
  XlaOp mx = Max(abs_x, abs_y);
  XlaOp mn = Min(abs_x, abs_y);
  FloatType constant_2 = 2;
  XlaOp constant_constant_1 = ScalarLike(x, 1);
  XlaOp r = Square(Div(mn, mx));
  XlaOp sqa = Sqrt(Add(constant_constant_1, r));
  return Select(
      Eq(mx, mn), Mul(ScalarLike(x, std::sqrt(constant_2)), mx),
      Select(And(Eq(sqa, constant_constant_1), Gt(r, ScalarLike(x, 0))),
             Add(mx, Div(Mul(mx, r), ScalarLike(x, constant_2))),
             Mul(mx, sqa)));
}