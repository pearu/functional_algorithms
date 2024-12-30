// This file is generated using functional_algorithms tool (0.11.1), see
//   https://github.com/pearu/functional_algorithms
// for more information.



#include "xla/client/lib/constants.h"
#include "xla/client/xla_builder.h"
#include <limits>


template <typename FloatType>
XlaOp complex_log10_0(XlaOp z) {
  XlaOp lnz = Log(z);
  XlaOp x = Real(lnz);
  XlaOp ln10 = ScalarLike(x, std::log(10));
  return Complex(Div(x, ln10), Div(Imag(lnz), ln10));
}