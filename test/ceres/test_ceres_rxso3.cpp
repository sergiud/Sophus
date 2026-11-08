// Copyright 2026 Sergiu Deitsch <sergiu.deitsch@gmail.com>

#include <sophus/rxso3.hpp>

#include "tests.hpp"

template <>
struct RotationalPart<Sophus::RxSO3d> {
  static double Norm(const typename Sophus::RxSO3d::Tangent& t) {
    return t.template head<3>().norm();
  }
};

namespace Sophus {

template <>
struct LieGroupCeresTestData<RxSO3d> {
  using Point = RxSO3d::Point;
  using Tangent = RxSO3d::Tangent;

  static StdVector<RxSO3d> groupElements() {
    double const kPi = Sophus::Constants<double>::pi();

    StdVector<RxSO3d> group_vec;
    group_vec.push_back(RxSO3d::exp(Tangent(0.2, 0.5, 0.0, 1.)));
    group_vec.push_back(RxSO3d::exp(Tangent(0.2, 0.5, -1.0, 1.1)));
    group_vec.push_back(RxSO3d::exp(Tangent(0., 0., 0., 1.1)));
    group_vec.push_back(RxSO3d::exp(Tangent(0., 0., 0.00001, 0.)));
    group_vec.push_back(RxSO3d::exp(Tangent(0., 0., 0.00001, 0.00001)));
    group_vec.push_back(RxSO3d::exp(Tangent(0., 0., 0.00001, 0)));
    group_vec.push_back(RxSO3d::exp(Tangent(kPi, 0, 0, 0.9)));
    group_vec.push_back(RxSO3d::exp(Tangent(0.2, -0.5, 0, 0)) *
                        RxSO3d::exp(Tangent(kPi, 0, 0, 0)) *
                        RxSO3d::exp(Tangent(-0.2, -0.5, 0, 0)));
    group_vec.push_back(RxSO3d::exp(Tangent(0.3, 0.5, 0.1, 0)) *
                        RxSO3d::exp(Tangent(kPi, 0, 0, 0)) *
                        RxSO3d::exp(Tangent(-0.3, -0.5, -0.1, 0)));
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
    LieGroupCeresTests<RxSO3d>::registerAll("RxSO3");

}  // namespace
}  // namespace Sophus
