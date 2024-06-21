// This file is generated using functional_algorithms tool (0.4.0), see
//   https://github.com/pearu/functional_algorithms
// for more information.



#include "xla/client/lib/constants.h"
#include "xla/client/xla_builder.h"
#include <limits>


template <typename FloatType>
XlaOp real_acos_0(XlaOp x) {
  XlaOp one = ScalarLike(x, 1);
  XlaOp add_one_x = Add(one, x);
  return Mul(ScalarLike(x, 2),
             Atan2(Sqrt(Mul(Sub(one, x), add_one_x)), add_one_x));
}