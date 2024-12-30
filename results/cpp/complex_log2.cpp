// This file is generated using functional_algorithms tool (0.13.3.dev1+g8d134ad.d20241230), see
//   https://github.com/pearu/functional_algorithms
// for more information.



#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>


std::complex<double> complex_log2_0(std::complex<double> z) {
  std::complex<double> lnz = std::log(z);
  double x = (lnz).real();
  double ln2 = std::log(2.0);
  return std::complex<double>((x) / (ln2), ((lnz).imag()) / (ln2));
}