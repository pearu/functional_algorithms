// This file is generated using functional_algorithms tool (0.4.0), see
//   https://github.com/pearu/functional_algorithms
// for more information.



#include "xla/client/lib/constants.h"
#include "xla/client/xla_builder.h"
#include <limits>


template <typename FloatType>
XlaOp real_acosh_0(XlaOp x) {
  FloatType constant_2 = 2;
  XlaOp constant_constant_1 = ScalarLike(x, 1);
  XlaOp sqrt_subtract_x_constant_constant_1 = Sqrt(Sub(x, constant_constant_1));
  return Select(Ge(x, ScalarLike(x, (std::numeric_limits<FloatType>::max()) /
                                        (constant_2))),
                Add(ScalarLike(x, std::log(constant_2)), Log(x)),
                Log1p(Mul(sqrt_subtract_x_constant_constant_1,
                          Add(Sqrt(Add(x, constant_constant_1)),
                              sqrt_subtract_x_constant_constant_1))));
}