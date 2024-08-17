// This file is generated using functional_algorithms tool (0.4.0), see
//   https://github.com/pearu/functional_algorithms
// for more information.



#include "xla/client/lib/constants.h"
#include "xla/client/xla_builder.h"
#include <limits>


template <typename FloatType>
XlaOp complex_log1p_0(XlaOp z) {
  XlaOp x = Real(z);
  XlaOp one = ScalarLike(x, 1);
  XlaOp xp1 = Add(x, one);
  XlaOp y = Imag(z);
  XlaOp ay = Abs(y);
  XlaOp axp1 = Abs(xp1);
  XlaOp mx = Max(axp1, ay);
  XlaOp mn = Min(axp1, ay);
  XlaOp r = Div(mn, mx);
  return Complex(
      Add(Log1p(Select(Ge(xp1, ay), x, Sub(mx, one))),
          Mul(ScalarLike(x, 0.5), Log1p(Select(Eq(mn, mx), one, Mul(r, r))))),
      Atan2(y, xp1));
}