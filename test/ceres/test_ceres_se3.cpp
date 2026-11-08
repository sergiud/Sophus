// Copyright 2026 Sergiu Deitsch <sergiu.deitsch@gmail.com>

#include <sophus/se3.hpp>

#include "tests.hpp"

template <>
struct RotationalPart<Sophus::SE3d> {
  static double Norm(const typename Sophus::SE3d::Tangent& t) {
    return t.template tail<3>().norm();
  }
};

namespace Sophus {

template <>
struct LieGroupCeresTestData<SE3d> {
  using SO3d = Sophus::SO3d;
  using Point = SE3d::Point;

  static StdVector<SE3d> groupElements() {
    double const kPi = Sophus::Constants<double>::pi();

    StdVector<SE3d> group_vec;
    group_vec.push_back(SE3d(SO3d::exp(Point(0.2, 0.5, 0.0)), Point(0, 0, 0)));
    group_vec.push_back(
        SE3d(SO3d::exp(Point(0.2, 0.5, -1.0)), Point(10, 0, 0)));
    group_vec.push_back(SE3d(SO3d::exp(Point(0., 0., 0.)), Point(0, 100, 5)));
    group_vec.push_back(
        SE3d(SO3d::exp(Point(0., 0., 0.00001)), Point(0, 0, 0)));
    group_vec.push_back(SE3d(SO3d::exp(Point(0., 0., 0.00001)),
                             Point(0, -0.00000001, 0.0000000001)));
    group_vec.push_back(
        SE3d(SO3d::exp(Point(0., 0., 0.00001)), Point(0.01, 0, 0)));
    group_vec.push_back(SE3d(SO3d::exp(Point(kPi, 0, 0)), Point(4, -5, 0)));
    group_vec.push_back(
        SE3d(SO3d::exp(Point(0.2, 0.5, 0.0)), Point(0, 0, 0)) *
        SE3d(SO3d::exp(Point(kPi, 0, 0)), Point(0, 0, 0)) *
        SE3d(SO3d::exp(Point(-0.2, -0.5, -0.0)), Point(0, 0, 0)));
    group_vec.push_back(
        SE3d(SO3d::exp(Point(0.3, 0.5, 0.1)), Point(2, 0, -7)) *
        SE3d(SO3d::exp(Point(kPi, 0, 0)), Point(0, 0, 0)) *
        SE3d(SO3d::exp(Point(-0.3, -0.5, -0.1)), Point(0, 6, 0)));
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
    LieGroupCeresTests<SE3d>::registerAll("SE3");

}  // namespace
}  // namespace Sophus
