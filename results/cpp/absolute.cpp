// This file is generated using functional_algorithms tool (0.1.2.dev7+g332df57.d20240604), see
//   https://github.com/pearu/functional_algorithms
// for more information.



#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>


float absolute_0(float z) { return std::abs(z); }

double absolute_1(double z) { return std::abs(z); }

float absolute_2(std::complex<float> z) {
  float x = (z).real();
  float abs_x = std::abs(x);
  float abs_y = std::abs((z).imag());
  float mx = std::max(abs_x, abs_y);
  float mn = std::min(abs_x, abs_y);
  float constant_2 = 2;
  float constant_1 = 1;
  float mn_over_mx = (mn) / (mx);
  float r = (mn_over_mx) * (mn_over_mx);
  float sqa = std::sqrt((constant_1) + (r));
  return (((mx) == (mn)) ? ((std::sqrt(constant_2)) * (mx))
                         : (((((sqa) == (constant_1)) && ((r) > (0)))
                                 ? ((mx) + (((mx) * (r)) / (constant_2)))
                                 : ((mx) * (sqa)))));
}

double absolute_3(std::complex<double> z) {
  double x = (z).real();
  double abs_x = std::abs(x);
  double abs_y = std::abs((z).imag());
  double mx = std::max(abs_x, abs_y);
  double mn = std::min(abs_x, abs_y);
  double constant_2 = 2;
  double constant_1 = 1;
  double mn_over_mx = (mn) / (mx);
  double r = (mn_over_mx) * (mn_over_mx);
  double sqa = std::sqrt((constant_1) + (r));
  return (((mx) == (mn)) ? ((std::sqrt(constant_2)) * (mx))
                         : (((((sqa) == (constant_1)) && ((r) > (0)))
                                 ? ((mx) + (((mx) * (r)) / (constant_2)))
                                 : ((mx) * (sqa)))));
}