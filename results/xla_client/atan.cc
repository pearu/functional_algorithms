// This file is generated using functional_algorithms tool (0.10.2.dev1+g24430b3.d20240905), see
//   https://github.com/pearu/functional_algorithms
// for more information.



#include "xla/client/lib/constants.h"
#include "xla/client/xla_builder.h"
#include <limits>


template <typename FloatType>
XlaOp atan_1(XlaOp z) {
  XlaOp imag_z = Imag(z);
  XlaOp x = Neg(imag_z);
  XlaOp ax = Abs(x);
  FloatType constant_largest = std::numeric_limits<FloatType>::max();
  FloatType inv_negeps_ =
      (((constant_largest) > (1e+308))
           ? (9007199254740994.0)
           : ((((constant_largest) > (1e+38)) ? (16777218.0) : (2050.0))));
  XlaOp safe_max = ScalarLike(imag_z, (inv_negeps_) * (inv_negeps_));
  XlaOp y = Real(z);
  XlaOp ay = Abs(y);
  XlaOp in_safe_region = And(Lt(ax, safe_max), Lt(ay, safe_max));
  XlaOp one = ScalarLike(imag_z, 1);
  XlaOp naxm1 = Sub(one, ax);
  XlaOp y2 = Mul(y, y);
  FloatType zero_ = 0;
  XlaOp constant_constant_neg1 = ScalarLike(imag_z, -1);
  XlaOp zero = ScalarLike(imag_z, zero_);
  XlaOp constant_constant_posinf =
      ScalarLike(imag_z, std::numeric_limits<FloatType>::infinity());
  XlaOp constant_constant_neginf =
      ScalarLike(imag_z, -std::numeric_limits<FloatType>::infinity());
  return Complex(
      Mul(Select(in_safe_region,
                 Atan2(Add(y, y), Sub(Mul(naxm1, Add(one, ax)), y2)),
                 Mul(Select(Ge(y, ScalarLike(y, zero_)), one,
                            constant_constant_neg1),
                     ScalarLike(imag_z, M_PI))),
          ScalarLike(imag_z, 0.5)),
      Neg(Mul(
          Mul(Select(Ge(x, zero), one, constant_constant_neg1),
              Log1p(Mul(
                  ScalarLike(imag_z, 4),
                  Select(
                      in_safe_region, Div(ax, Add(Mul(naxm1, naxm1), y2)),
                      Select(Lt(Mul(ay, ScalarLike(imag_z, inv_negeps_)), ax),
                             Div(one, ax),
                             Select(Or(Or(Eq(x, constant_constant_posinf),
                                          Eq(x, constant_constant_neginf)),
                                       Or(Eq(y, constant_constant_posinf),
                                          Eq(y, constant_constant_neginf))),
                                    zero,
                                    Div(Div(one, Add(Div(ax, y), Div(y, ax))),
                                        y))))))),
          ScalarLike(imag_z, 0.25))));
}