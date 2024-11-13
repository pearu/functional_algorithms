// This file is generated using functional_algorithms tool (0.11.0), see
//   https://github.com/pearu/functional_algorithms
// for more information.



#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>


std::complex<float> atan_2(std::complex<float> z) {
  std::complex<float> w =
      std::atanh(std::complex<float>(-((z).imag()), (z).real()));
  return std::complex<float>((w).imag(), -((w).real()));
}

std::complex<double> atan_3(std::complex<double> z) {
  std::complex<double> w =
      std::atanh(std::complex<double>(-((z).imag()), (z).real()));
  return std::complex<double>((w).imag(), -((w).real()));
}