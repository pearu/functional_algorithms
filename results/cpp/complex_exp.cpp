// This file is generated using functional_algorithms tool (0.15.1.dev3+ge93b47e.d20250113), see
//   https://github.com/pearu/functional_algorithms
// for more information.



#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>


std::complex<double> complex_exp_0(std::complex<double> z) {
  double x = (z).real();
  double e = std::exp(x);
  bool eq_e_constant_posinf = (e) == (std::numeric_limits<double>::infinity());
  double e2 = std::exp((x) * (0.5));
  double y = (z).imag();
  double cs = std::cos(y);
  double zero = 0.0;
  double sn = std::sin(y);
  return std::complex<double>(
      ((eq_e_constant_posinf) ? (((e2) * (cs)) * (e2)) : ((e) * (cs))),
      (((y) == (zero)) ? (zero)
                       : (((eq_e_constant_posinf) ? (((e2) * (sn)) * (e2))
                                                  : ((e) * (sn))))));
}