// This file is generated using functional_algorithms tool (0.4.0), see
//   https://github.com/pearu/functional_algorithms
// for more information.



#include "xla/client/lib/constants.h"
#include "xla/client/xla_builder.h"
#include <limits>


template <typename FloatType>
XlaOp asinh_0(XlaOp z) {
  XlaOp ax = Abs(z);
  XlaOp ax2 = Mul(ax, ax);
  XlaOp one = ScalarLike(z, 1);
  return Mul(
      Sign(z),
      Select(Ge(ax, ScalarLike(
                        z, std::sqrt(std::numeric_limits<FloatType>::max()))),
             Add(ScalarLike(z, std::log(2)), Log(ax)),
             Log1p(Add(ax, Div(ax2, Add(one, Sqrt(Add(one, ax2))))))));
}

template <typename FloatType>
XlaOp asinh_1(XlaOp z) {
  XlaOp w = Asin(Complex(Neg(Imag(z)), Real(z)));
  return Complex(Imag(w), Neg(Real(w)));
}