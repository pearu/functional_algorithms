// This file is generated using functional_algorithms tool (0.15.1.dev20+g7dbbfd7db.d20261016), see
//   https://github.com/pearu/functional_algorithms
// for more information.



#include "xla/client/lib/constants.h"
#include "xla/client/xla_builder.h"
#include <limits>


template <typename FloatType>
XlaOp complex_exp_0(XlaOp z) {
  XlaOp y = Imag(z);
  XlaOp y_is_not_finite = Not(IsFinite(y));
  XlaOp x = Real(z);
  XlaOp constant_constant_posinf =
      ScalarLike(x, std::numeric_limits<FloatType>::infinity());
  XlaOp eq_x_constant_constant_neginf =
      Eq(x, ScalarLike(x, -std::numeric_limits<FloatType>::infinity()));
  XlaOp e = Exp(x);
  XlaOp eq_e_constant_constant_posinf = Eq(e, constant_constant_posinf);
  XlaOp e2 = Exp(Mul(x, ScalarLike(x, 0.5)));
  XlaOp cs = Cos(y);
  XlaOp zero = ScalarLike(x, 0);
  XlaOp sn = Sin(y);
  return Complex(
      Select(And(y_is_not_finite, Or(Eq(x, constant_constant_posinf),
                                     eq_x_constant_constant_neginf)),
             e,
             Select(eq_e_constant_constant_posinf, Mul(Mul(e2, cs), e2),
                    Mul(e, cs))),
      Select(And(Not(And(eq_x_constant_constant_neginf, y_is_not_finite)),
                 Ne(y, zero)),
             Select(eq_e_constant_constant_posinf, Mul(Mul(e2, sn), e2),
                    Mul(e, sn)),
             zero));
}