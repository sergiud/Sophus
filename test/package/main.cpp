// Copyright 2026 Sergiu Deitsch <sergiu.deitsch@gmail.com>

#include <cstdlib>

#include <sophus/so3.hpp>

int main() {
  Sophus::SO3d const R = Sophus::SO3d::exp(Eigen::Vector3d(0.1, 0.2, 0.3));
  double constexpr kTolerance = 1e-12;
  return (R.log() - Eigen::Vector3d(0.1, 0.2, 0.3)).norm() < kTolerance
             ? EXIT_SUCCESS
             : EXIT_FAILURE;
}
