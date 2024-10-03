// This file is generated using functional_algorithms tool (0.10.2.dev9+g7001467.d20241002), see
//   https://github.com/pearu/functional_algorithms
// for more information.



#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>


float hypot_0(float x, float y) {
  float abs_x = std::abs(x);
  float abs_y = std::abs(y);
  float mx = std::max(abs_x, abs_y);
  float mn = std::min(abs_x, abs_y);
  float constant_f1 = 1.0;
  float mn_over_mx = (mn) / (mx);
  float r = (mn_over_mx) * (mn_over_mx);
  float sqa = std::sqrt((constant_f1) + (r));
  return (((mx) == (mn)) ? ((1.4142135) * (mx))
                         : (((((sqa) == (constant_f1)) && ((r) > (0.0)))
                                 ? ((mx) + (((mx) * (r)) / (2.0)))
                                 : ((mx) * (sqa)))));
}

double hypot_1(double x, double y) {
  double abs_x = std::abs(x);
  double abs_y = std::abs(y);
  double mx = std::max(abs_x, abs_y);
  double mn = std::min(abs_x, abs_y);
  double constant_f1 = 1.0;
  double mn_over_mx = (mn) / (mx);
  double r = (mn_over_mx) * (mn_over_mx);
  double sqa = std::sqrt((constant_f1) + (r));
  return (((mx) == (mn)) ? ((1.4142135623730951) * (mx))
                         : (((((sqa) == (constant_f1)) && ((r) > (0.0)))
                                 ? ((mx) + (((mx) * (r)) / (2.0)))
                                 : ((mx) * (sqa)))));
}