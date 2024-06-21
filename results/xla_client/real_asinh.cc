// This file is generated using functional_algorithms tool (0.4.0), see
//   https://github.com/pearu/functional_algorithms
// for more information.



#include "xla/client/lib/constants.h"
#include "xla/client/xla_builder.h"
#include <limits>


template <typename FloatType>
XlaOp real_asinh_0(XlaOp x) {
  XlaOp ax = Abs(x);
  XlaOp ax2 = Mul(ax, ax);
  XlaOp one = ScalarLike(x, 1);
  return Mul(
      Sign(x),
      Select(Ge(ax, ScalarLike(
                        x, std::sqrt(std::numeric_limits<FloatType>::max()))),
             Add(ScalarLike(x, std::log(2)), Log(ax)),
             Log1p(Add(ax, Div(ax2, Add(one, Sqrt(Add(one, ax2))))))));
}