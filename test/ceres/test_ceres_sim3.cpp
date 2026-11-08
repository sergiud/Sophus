// Copyright 2026 Sergiu Deitsch <sergiu.deitsch@gmail.com>

#include <sophus/sim3.hpp>

#include "tests.hpp"

template <>
struct RotationalPart<Sophus::Sim3d> {
  static double Norm(const typename Sophus::Sim3d::Tangent& t) {
    return t.template segment<3>(3).norm();
  }
};

namespace Sophus {

template <>
struct LieGroupCeresTestData<Sim3d> {
  using RxSO3d = Sophus::RxSO3d;
  using Point = Sim3d::Point;
  using Vector4d = Eigen::Vector4d;

  static StdVector<Sim3d> groupElements() {
    double const kPi = Sophus::Constants<double>::pi();

    StdVector<Sim3d> group_vec;
    group_vec.push_back(
        Sim3d(RxSO3d::exp(Vector4d(0.2, 0.5, 0.0, 1.)), Point(0, 0, 0)));
    group_vec.push_back(
        Sim3d(RxSO3d::exp(Vector4d(0.2, 0.5, -1.0, 1.1)), Point(10, 0, 0)));
    group_vec.push_back(
        Sim3d(RxSO3d::exp(Vector4d(0., 0., 0., 0.001)), Point(0, 10, 5)));
    group_vec.push_back(
        Sim3d(RxSO3d::exp(Vector4d(0., 0., 0., 1.1)), Point(0, 10, 5)));
    group_vec.push_back(
        Sim3d(RxSO3d::exp(Vector4d(0., 0., 0.00001, 0.)), Point(0, 0, 0)));
    group_vec.push_back(Sim3d(RxSO3d::exp(Vector4d(0., 0., 0.00001, 0.0000001)),
                              Point(1, -1.00000001, 2.0000000001)));
    group_vec.push_back(
        Sim3d(RxSO3d::exp(Vector4d(0., 0., 0.00001, 0)), Point(0.01, 0, 0)));
    group_vec.push_back(
        Sim3d(RxSO3d::exp(Vector4d(kPi, 0, 0, 0.9)), Point(4, -5, 0)));
    group_vec.push_back(
        Sim3d(RxSO3d::exp(Vector4d(0.2, 0.5, 0.0, 0)), Point(0, 0, 0)) *
        Sim3d(RxSO3d::exp(Vector4d(kPi, 0, 0, 0)), Point(0, 0, 0)) *
        Sim3d(RxSO3d::exp(Vector4d(-0.2, -0.5, -0.0, 0)), Point(0, 0, 0)));
    group_vec.push_back(
        Sim3d(RxSO3d::exp(Vector4d(0.3, 0.5, 0.1, 0)), Point(2, 0, -7)) *
        Sim3d(RxSO3d::exp(Vector4d(kPi, 0, 0, 0)), Point(0, 0, 0)) *
        Sim3d(RxSO3d::exp(Vector4d(-0.3, -0.5, -0.1, 0)), Point(0, 6, 0)));
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
    LieGroupCeresTests<Sim3d>::registerAll("Sim3");

}  // namespace
}  // namespace Sophus
