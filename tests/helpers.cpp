/// Copyright (c) 2018-2021, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "tests/helpers.h"

sia::LinearGaussian createTestSystem() {
  Eigen::Matrix<double, 1, 1> A, B, C, H, Q, R;
  A << 0.9;
  B << 0.1;
  C << 1;
  H << 1;
  Q << 0.1;
  R << 0.01;
  return sia::LinearGaussian(A, B, C, H, Q, R);
}
