// This file is generated using functional_algorithms tool (0.1.2.dev2+g1428951.d20240525), see
//   https://github.com/pearu/functional_algorithms
// for more information.



#include <limits>
#include <cmath>
#include <algorithm>
#include  <cstdint>


double hypot_0(double x, double y) {
  double abs_x = std::abs(x);
  double abs_y = std::abs(y);
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