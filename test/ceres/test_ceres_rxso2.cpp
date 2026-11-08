// Copyright 2026 Sergiu Deitsch <sergiu.deitsch@gmail.com>

#include <sophus/rxso2.hpp>

#include "tests.hpp"

template <>
struct RotationalPart<Sophus::RxSO2d> {
  static double Norm(const typename Sophus::RxSO2d::Tangent& t) {
    return std::abs(t[0]);
  }
};

namespace Sophus {

template <>
struct LieGroupCeresTestData<RxSO2d> {
  using Tangent = RxSO2d::Tangent;
  using Point = RxSO2d::Point;

  static StdVector<RxSO2d> groupElements() {
    double const kPi = Sophus::Constants<double>::pi();

    StdVector<RxSO2d> group_vec;
    group_vec.push_back(RxSO2d::exp(Tangent(0.2, 1.)));
    group_vec.push_back(RxSO2d::exp(Tangent(0.2, 1.1)));
    group_vec.push_back(RxSO2d::exp(Tangent(0., 1.1)));
    group_vec.push_back(RxSO2d::exp(Tangent(0.00001, 0.)));
    group_vec.push_back(RxSO2d::exp(Tangent(0.00001, 0.00001)));
    group_vec.push_back(RxSO2d::exp(Tangent(kPi, 0.9)));
    group_vec.push_back(RxSO2d::exp(Tangent(0.2, 0)) *
                        RxSO2d::exp(Tangent(kPi, 0.0)) *
                        RxSO2d::exp(Tangent(-0.2, 0)));
    group_vec.push_back(RxSO2d::exp(Tangent(0.3, 0)) *
                        RxSO2d::exp(Tangent(kPi, 0.001)) *
                        RxSO2d::exp(Tangent(-0.3, 0)));
    return group_vec;
  }

  static StdVector<Point> points() {
    StdVector<Point> point_vec;
    point_vec.push_back(Point(1.012, 2.73));
    point_vec.push_back(Point(9.2, -7.3));
    point_vec.push_back(Point(2.5, 0.1));
    point_vec.push_back(Point(12.3, 1.9));
    point_vec.push_back(Point(-3.21, 3.42));
    point_vec.push_back(Point(-8.0, 6.1));
    point_vec.push_back(Point(0.0, 2.5));
    point_vec.push_back(Point(7.1, 7.8));
    point_vec.push_back(Point(5.8, 9.2));
    return point_vec;
  }
};

namespace {

[[maybe_unused]] bool const kRegistered =
    LieGroupCeresTests<RxSO2d>::registerAll("RxSO2");

}  // namespace
}  // namespace Sophus
