// This file is generated using functional_algorithms tool (0.1.2.dev2+g1428951.d20240525), see
//   https://github.com/pearu/functional_algorithms
// for more information.



#include <limits>


XLAOp real_asin_0(XLAOp x) {
  XLAOp one = ScalarLike(x, 1);
  return Mul(ScalarLike(x, 2),
             Atan2(x, Add(one, Sqrt(Mul(Sub(one, x), Add(one, x))))));
}