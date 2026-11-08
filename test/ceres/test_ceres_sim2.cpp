// Copyright 2026 Sergiu Deitsch <sergiu.deitsch@gmail.com>

#include <sophus/sim2.hpp>

#include "tests.hpp"

template <>
struct RotationalPart<Sophus::Sim2d> {
  static double Norm(const typename Sophus::Sim2d::Tangent& t) {
    return std::abs(t[2]);
  }
};

namespace Sophus {

template <>
struct LieGroupCeresTestData<Sim2d> {
  using RxSO2d = Sophus::RxSO2d;
  using Point = Sim2d::Point;
  using Vector2d = Sophus::Vector2d;

  static StdVector<Sim2d> groupElements() {
    double const kPi = Sophus::Constants<double>::pi();

    StdVector<Sim2d> group_vec;
    group_vec.push_back(Sim2d(RxSO2d::exp(Vector2d(0.2, 1.)), Point(0, 0)));
    group_vec.push_back(Sim2d(RxSO2d::exp(Vector2d(0.2, 1.1)), Point(10, 0)));
    group_vec.push_back(Sim2d(RxSO2d::exp(Vector2d(0., 0.)), Point(0, 10)));
    group_vec.push_back(Sim2d(RxSO2d::exp(Vector2d(0.00001, 0.)), Point(0, 0)));
    group_vec.push_back(Sim2d(RxSO2d::exp(Vector2d(0.00001, 0.0000001)),
                              Point(1, -1.00000001)));
    group_vec.push_back(Sim2d(RxSO2d::exp(Vector2d(0., 0.)), Point(0.01, 0)));
    group_vec.push_back(Sim2d(RxSO2d::exp(Vector2d(kPi, 0.9)), Point(4, 0)));
    group_vec.push_back(Sim2d(RxSO2d::exp(Vector2d(0.2, 0)), Point(0, 0)) *
                        Sim2d(RxSO2d::exp(Vector2d(kPi, 0)), Point(0, 0)) *
                        Sim2d(RxSO2d::exp(Vector2d(-0.2, 0)), Point(0, 0)));
    group_vec.push_back(Sim2d(RxSO2d::exp(Vector2d(0.3, 0)), Point(2, -7)) *
                        Sim2d(RxSO2d::exp(Vector2d(kPi, 0)), Point(0, 0)) *
                        Sim2d(RxSO2d::exp(Vector2d(-0.3, 0)), Point(0, 6)));
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
    LieGroupCeresTests<Sim2d>::registerAll("Sim2");

}  // namespace
}  // namespace Sophus
