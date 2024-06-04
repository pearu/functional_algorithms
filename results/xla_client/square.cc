// This file is generated using functional_algorithms tool (0.1.2.dev2+g1428951.d20240525), see
//   https://github.com/pearu/functional_algorithms
// for more information.



#include <limits>


XLAOp square_0(XLAOp z) { return Mul(z, z); }

XLAOp square_1(XLAOp z) {
  XLAOp x = Real(z);
  XLAOp y = Imag(z);
  return Complex(
      Select(Eq(Abs(x), Abs(y)), ScalarLike(x, 0), Mul(Sub(x, y), Add(x, y))),
      Mul(ScalarLike(x, 2), Mul(x, y)));
}