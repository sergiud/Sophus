// Copyright 2026 Sergiu Deitsch <sergiu.deitsch@gmail.com>

#include <sophus/se3.hpp>

#include "tests.hpp"

template <>
struct RotationalPart<Sophus::SO3d> {
  static double Norm(const typename Sophus::SO3d::Tangent& t) {
    return t.norm();
  }
};

namespace Sophus {

template <>
struct LieGroupCeresTestData<SO3d> {
  using SO3Type = Sophus::SO3<double>;
  using Point = SO3Type::Point;
  using Tangent = SO3Type::Tangent;

  static StdVector<SO3d> groupElements() {
    double const kPi = Sophus::Constants<double>::pi();
    double const kEpsilon = Sophus::Constants<double>::epsilon();
    SO3Type const C_0 = SO3Type::exp(Tangent(0.1, 0.05, -0.7));
    Tangent const axis_0(0.18005924, -0.54563405, 0.81845107);

    StdVector<SO3d> group_vec;
    // Generic tests
    group_vec.push_back(SO3Type::exp(Point(0.2, 0.5, 0.0)));
    group_vec.push_back(SO3Type::exp(Point(0.2, 0.5, -1.0)));
    group_vec.push_back(SO3Type::exp(Point(0., 0., 0.)));
    group_vec.push_back(SO3Type::exp(Point(0., 0., 0.00001)));
    group_vec.push_back(SO3Type::exp(Point(kPi, 0, 0)));
    group_vec.push_back(SO3Type::exp(Point(0.2, 0.5, 0.0)) *
                        SO3Type::exp(Point(kPi, 0, 0)) *
                        SO3Type::exp(Point(-0.2, -0.5, -0.0)));
    group_vec.push_back(SO3Type::exp(Point(0.3, 0.5, 0.1)) *
                        SO3Type::exp(Point(kPi, 0, 0)) *
                        SO3Type::exp(Point(-0.3, -0.5, -0.1)));

    // Checks if ceres optimization will proceed correctly given problematic
    // or close-to-singular initial conditions, i.e. approx. 180-deg rotation,
    // which trips a flaw in old implementation of SO3::log() where the
    // tangent vector's magnitude is set to a constant close to \pi whenever
    // the input rotation's rotation angle is within some tolerance of \pi,
    // giving zero gradients wrt scalar part of quaternion.
    group_vec.push_back(C_0);
    // Generic rotation angle < pi
    group_vec.push_back(C_0 * SO3Type::exp(axis_0 * 1.0));
    group_vec.push_back(C_0 * SO3Type::exp(axis_0 * -1.0));
    // Generic rotation angle > pi
    group_vec.push_back(C_0 * SO3Type::exp(axis_0 * 4.0));
    group_vec.push_back(C_0 * SO3Type::exp(axis_0 * -4.0));
    // Singular rotation angle = pi
    group_vec.push_back(C_0 * SO3Type::exp(axis_0 * kPi));
    group_vec.push_back(C_0 * SO3Type::exp(axis_0 * -kPi));
    group_vec.push_back(C_0 * SO3Type::exp(axis_0 * kPi * (1.0 + kEpsilon)));
    group_vec.push_back(C_0 * SO3Type::exp(axis_0 * kPi * (1.0 - kEpsilon)));
    group_vec.push_back(C_0 * SO3Type::exp(axis_0 * -kPi * (1.0 + kEpsilon)));
    group_vec.push_back(C_0 * SO3Type::exp(axis_0 * -kPi * (1.0 - kEpsilon)));
    return group_vec;
  }

  static StdVector<Point> points() {
    StdVector<Point> point_vec;
    point_vec.push_back(Point(1.012, 2.73, -1.4));
    point_vec.push_back(Point(9.2, -7.3, -4.4));
    point_vec.push_back(Point(2.5, 0.1, 9.1));
    point_vec.push_back(Point(12.3, 1.9, 3.8));
    point_vec.push_back(Point(-3.21, 3.42, 2.3));
    point_vec.push_back(Point(-8.0, 6.1, -1.1));
    point_vec.push_back(Point(0.0, 2.5, 5.9));
    point_vec.push_back(Point(7.1, 7.8, -14));
    point_vec.push_back(Point(5.8, 9.2, 0.0));
    return point_vec;
  }
};

namespace {

[[maybe_unused]] bool const kRegistered =
    LieGroupCeresTests<SO3d>::registerAll("SO3");

}  // namespace
}  // namespace Sophus
