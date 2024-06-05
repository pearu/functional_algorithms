// This file is generated using functional_algorithms tool (0.1.2.dev7+g332df57.d20240604), see
//   https://github.com/pearu/functional_algorithms
// for more information.



#include <limits>


template <typename FloatType>
XLAOp square_0(XLAOp z) {
  return Mul(z, z);
}

template <typename FloatType>
XLAOp square_1(XLAOp z) {
  XLAOp x = Real(z);
  XLAOp y = Imag(z);
  return Complex(
      Select(Eq(Abs(x), Abs(y)), ScalarLike(x, 0), Mul(Sub(x, y), Add(x, y))),
      Mul(ScalarLike(x, 2), Mul(x, y)));
}