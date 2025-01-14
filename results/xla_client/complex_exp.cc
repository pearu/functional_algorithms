// This file is generated using functional_algorithms tool (0.15.1.dev3+ge93b47e.d20250113), see
//   https://github.com/pearu/functional_algorithms
// for more information.



#include "xla/client/lib/constants.h"
#include "xla/client/xla_builder.h"
#include <limits>


template <typename FloatType>
XlaOp complex_exp_0(XlaOp z) {
  XlaOp x = Real(z);
  XlaOp e = Exp(x);
  XlaOp eq_e_constant_constant_posinf =
      Eq(e, ScalarLike(x, std::numeric_limits<FloatType>::infinity()));
  XlaOp e2 = Exp(Mul(x, ScalarLike(x, 0.5)));
  XlaOp y = Imag(z);
  XlaOp cs = Cos(y);
  XlaOp zero = ScalarLike(x, 0);
  XlaOp sn = Sin(y);
  return Complex(
      Select(eq_e_constant_constant_posinf, Mul(Mul(e2, cs), e2), Mul(e, cs)),
      Select(Eq(y, zero), zero,
             Select(eq_e_constant_constant_posinf, Mul(Mul(e2, sn), e2),
                    Mul(e, sn))));
}