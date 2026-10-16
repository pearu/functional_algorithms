// This file is generated using functional_algorithms tool (0.15.1.dev20+g7dbbfd7db.d20261016), see
//   https://github.com/pearu/functional_algorithms
// for more information.



#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>


std::complex<double> complex_exp_0(std::complex<double> z) {
  double y = (z).imag();
  bool y_is_not_finite = !(std::isfinite(y));
  double x = (z).real();
  double constant_posinf = std::numeric_limits<double>::infinity();
  bool eq_x_constant_neginf = (x) == (-std::numeric_limits<double>::infinity());
  double e = std::exp(x);
  bool eq_e_constant_posinf = (e) == (constant_posinf);
  double e2 = std::exp((x) * (0.5));
  double cs = std::cos(y);
  double zero = 0.0;
  double sn = std::sin(y);
  return std::complex<double>(
      (((y_is_not_finite) &&
        (((x) == (constant_posinf)) || (eq_x_constant_neginf)))
           ? (e)
           : (((eq_e_constant_posinf) ? (((e2) * (cs)) * (e2))
                                      : ((e) * (cs))))),
      (((!((eq_x_constant_neginf) && (y_is_not_finite))) && ((y) != (zero)))
           ? (((eq_e_constant_posinf) ? (((e2) * (sn)) * (e2)) : ((e) * (sn))))
           : (zero)));
}