// This file is generated using functional_algorithms tool (0.10.2.dev9+g7001467.d20241002), see
//   https://github.com/pearu/functional_algorithms
// for more information.



#include "xla/client/lib/constants.h"
#include "xla/client/xla_builder.h"
#include <limits>


template <typename FloatType>
XlaOp square_0(XlaOp z) {
  return Mul(z, z);
}

template <typename FloatType>
XlaOp square_1(XlaOp z) {
  XlaOp x = Real(z);
  XlaOp y = Imag(z);
  return Complex(Select(And(IsFinite(x), Eq(Abs(x), Abs(y))), ScalarLike(x, 0),
                        Mul(Sub(x, y), Add(x, y))),
                 Mul(ScalarLike(x, 2), Mul(x, y)));
}