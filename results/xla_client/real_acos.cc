// This file is generated using functional_algorithms tool (0.4.0), see
//   https://github.com/pearu/functional_algorithms
// for more information.



#include "xla/client/lib/constants.h"
#include "xla/client/xla_builder.h"
#include <limits>


template <typename FloatType>
XlaOp real_acos_0(XlaOp x) {
  XlaOp constant_constant_1 = ScalarLike(x, 1);
  return Atan2(
      Sqrt(Mul(Sub(constant_constant_1, x), Add(constant_constant_1, x))), x);
}