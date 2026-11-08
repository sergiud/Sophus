#include <cmath>
#include <cstdlib>
#include <iostream>
#include "sophus/geometry.hpp"

int main() {
  // The following demonstrates the group multiplication of rotation matrices

  // Create rotation matrices from rotations around the x and y and z axes:
  const double kPi = Sophus::Constants<double>::pi();
  Sophus::SO3d R1 = Sophus::SO3d::rotX(kPi / 4);
  Sophus::SO3d R2 = Sophus::SO3d::rotY(kPi / 6);
  Sophus::SO3d R3 = Sophus::SO3d::rotZ(-kPi / 3);

  std::cout << "The rotation matrices are" << std::endl;
  std::cout << "R1:\n" << R1.matrix() << std::endl;
  std::cout << "R2:\n" << R2.matrix() << std::endl;
  std::cout << "R3:\n" << R3.matrix() << std::endl;
  std::cout << "Their product R1*R2*R3:\n"
            << (R1 * R2 * R3).matrix() << std::endl;
  std::cout << std::endl;

  // Rotation matrices can act on vectors
  Eigen::Vector3d x;
  x << 0.0, 0.0, 1.0;
  std::cout << "Rotation matrices can act on vectors" << std::endl;
  std::cout << "x\n" << x << std::endl;
  std::cout << "R2*x\n" << R2 * x << std::endl;
  std::cout << "R1*(R2*x)\n" << R1 * (R2 * x) << std::endl;
  std::cout << "(R1*R2)*x\n" << (R1 * R2) * x << std::endl;
  std::cout << std::endl;

  // SO(3) are internally represented as unit quaternions.
  std::cout << "R1 in matrix form:\n" << R1.matrix() << std::endl;
  std::cout << "R1 in unit quaternion form:\n"
            << R1.unit_quaternion().coeffs() << std::endl;
  // Note that the order of coefficients of Eigen's quaternion class is
  // (imag0, imag1, imag2, real)
  std::cout << std::endl;

  // The rotation around the x axis has a closed form, and the product of
  // rotations is the product of their matrices. The example fails unless
  // both hold.
  double constexpr kTolerance = 1e-12;
  double const c = std::cos(kPi / 4);
  double const s = std::sin(kPi / 4);
  Eigen::Matrix3d R1_closed_form;
  R1_closed_form << 1, 0, 0, 0, c, -s, 0, s, c;
  bool const matches_closed_form =
      (R1.matrix() - R1_closed_form).norm() < kTolerance;
  bool const multiplies_like_matrices =
      ((R1 * R2 * R3).matrix() - R1.matrix() * R2.matrix() * R3.matrix())
          .norm() < kTolerance;
  std::cout << std::boolalpha
            << "R1 matches its closed form: " << matches_closed_form << '\n'
            << "R1*R2*R3 is the product of their matrices: "
            << multiplies_like_matrices << std::endl;
  return matches_closed_form && multiplies_like_matrices ? EXIT_SUCCESS
                                                         : EXIT_FAILURE;
}