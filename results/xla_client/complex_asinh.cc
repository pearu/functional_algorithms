// This file is generated using functional_algorithms tool (0.4.0), see
//   https://github.com/pearu/functional_algorithms
// for more information.



#include "xla/client/lib/constants.h"
#include "xla/client/xla_builder.h"
#include <limits>


template <typename FloatType>
XlaOp complex_asinh_0(XlaOp z) {
  XlaOp w = Asin(Complex(Neg(Imag(z)), Real(z)));
  return Complex(Imag(w), Neg(Real(w)));
}