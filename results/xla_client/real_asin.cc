// This file is generated using functional_algorithms tool (0.4.0), see
//   https://github.com/pearu/functional_algorithms
// for more information.



#include "xla/client/lib/constants.h"
#include "xla/client/xla_builder.h"
#include <limits>


template <typename FloatType>
XlaOp real_asin_0(XlaOp x) {
  XlaOp one = ScalarLike(x, 1);
  return Mul(ScalarLike(x, 2),
             Atan2(x, Add(one, Sqrt(Mul(Sub(one, x), Add(one, x))))));
}