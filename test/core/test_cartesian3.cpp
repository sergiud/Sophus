// Copyright 2026 Sergiu Deitsch <sergiu.deitsch@gmail.com>

#include <sophus/cartesian.hpp>

#include "tests.hpp"

// Explicit instantiate all class templates so that all member methods
// get compiled and for code coverage analysis.
namespace Eigen {
template class Map<Sophus::Cartesian<double, 3>>;
template class Map<Sophus::Cartesian<double, 3> const>;
template class Map<Sophus::Cartesian<double, 3>, Eigen::Aligned>;
template class Map<Sophus::Cartesian<double, 3> const, Eigen::Aligned>;
}  // namespace Eigen

namespace Sophus {

template class Cartesian<double, 3, Eigen::AutoAlign>;
template class Cartesian<float, 3, Eigen::DontAlign>;
#if SOPHUS_CERES
template class Cartesian<ceres::Jet<double, 3>, 3>;
#endif

template <class Scalar>
struct LieGroupTestData<Cartesian<Scalar, 3>> {
  using CartesianType = Cartesian<Scalar, 3>;
  using Point = typename CartesianType::Point;
  using Tangent = typename CartesianType::Tangent;

  static StdVector<Tangent> tangents() {
    StdVector<Tangent> tangent_vec;
    tangent_vec.push_back(Tangent(Scalar(0), Scalar(0), Scalar(0)));
    tangent_vec.push_back(Tangent(Scalar(1), Scalar(0), Scalar(0)));
    tangent_vec.push_back(Tangent(Scalar(1), Scalar(0), Scalar(0.1)));
    tangent_vec.push_back(Tangent(Scalar(0), Scalar(1), Scalar(0.1)));
    tangent_vec.push_back(Tangent(Scalar(0), Scalar(0), Scalar(-0.1)));
    tangent_vec.push_back(Tangent(Scalar(-1), Scalar(1), Scalar(-0.1)));
    tangent_vec.push_back(Tangent(Scalar(20), Scalar(-1), Scalar(2)));
    return tangent_vec;
  }

  static StdVector<CartesianType> groupElements() {
    StdVector<CartesianType> cartesian_vec;
    for (Tangent const& t : tangents()) {
      cartesian_vec.push_back(CartesianType(t));
    }
    return cartesian_vec;
  }

  static StdVector<Point> points() {
    StdVector<Point> point_vec;
    point_vec.push_back(Point(Scalar(1), Scalar(2), Scalar(4)));
    point_vec.push_back(Point(Scalar(1), Scalar(-3), Scalar(0.5)));
    point_vec.push_back(Point(Scalar(-5), Scalar(-6), Scalar(7)));
    return point_vec;
  }
};

namespace {

[[maybe_unused]] bool const kRegistered =
    registerLieGroupTests<Cartesian<double, 3>, Cartesian<float, 3>
#if SOPHUS_CERES
                          ,
                          Cartesian<ceres::Jet<double, 3>, 3>
#endif
                          >("Cartesian3");

}  // namespace
}  // namespace Sophus
