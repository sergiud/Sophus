// Copyright 2026 Sergiu Deitsch <sergiu.deitsch@gmail.com>

#include <sophus/so2.hpp>

#include "tests.hpp"

template <>
struct RotationalPart<Sophus::SO2d> {
  static double Norm(const typename Sophus::SO2d::Tangent& t) {
    return std::abs(t);
  }
};

namespace Sophus {

template <>
struct LieGroupCeresTestData<SO2d> {
  using Point = SO2d::Point;

  static StdVector<SO2d> groupElements() {
    double const kPi = Sophus::Constants<double>::pi();

    StdVector<SO2d> group_vec;
    group_vec.emplace_back(SO2d::exp(0.0));
    group_vec.emplace_back(SO2d::exp(0.2));
    group_vec.emplace_back(SO2d::exp(10.));
    group_vec.emplace_back(SO2d::exp(0.00001));
    group_vec.emplace_back(SO2d::exp(kPi));
    group_vec.emplace_back(SO2d::exp(0.2) * SO2d::exp(kPi) * SO2d::exp(-0.2));
    group_vec.emplace_back(SO2d::exp(-0.3) * SO2d::exp(kPi) * SO2d::exp(0.3));
    return group_vec;
  }

  static StdVector<Point> points() {
    StdVector<Point> point_vec;
    point_vec.emplace_back(Point(1.012, 2.73));
    point_vec.emplace_back(Point(9.2, -7.3));
    point_vec.emplace_back(Point(2.5, 0.1));
    point_vec.emplace_back(Point(12.3, 1.9));
    point_vec.emplace_back(Point(-3.21, 3.42));
    point_vec.emplace_back(Point(-8.0, 6.1));
    point_vec.emplace_back(Point(0.0, 2.5));
    point_vec.emplace_back(Point(7.1, 7.8));
    point_vec.emplace_back(Point(5.8, 9.2));
    return point_vec;
  }
};

namespace {

[[maybe_unused]] bool const kRegistered =
    LieGroupCeresTests<SO2d>::registerAll("SO2");

}  // namespace
}  // namespace Sophus
