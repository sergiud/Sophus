// Copyright 2026 Sergiu Deitsch <sergiu.deitsch@gmail.com>

// Constructing a Cartesian from a fixed-size matrix that is neither an
// M-vector nor an (M+1)x(M+1) matrix must not compile.

#include <sophus/cartesian.hpp>

int main() {
  int constexpr kRows = 2;
  int constexpr kCols = 5;
  using Matrix = Eigen::Matrix<double, kRows, kCols>;
  Matrix const m = Matrix::Zero();
  Sophus::Cartesian3d const c(m);
  return static_cast<int>(c.params().size());
}
