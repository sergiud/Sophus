// Copyright 2022, 2026 Sergiu Deitsch <sergiu.deitsch@gmail.com>

/// @file
/// Basis spline implementation on Lie Group following:
/// S. Lovegrove, A. Patron-Perez, G. Sibley, BMVC 2013
/// http://www.bmva.org/bmvc/2013/Papers/paper0093/paper0093.pdf

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdlib>

#include "types.hpp"

namespace Sophus {

template <class Scalar>
class SplineBasisFunction {
 public:
  static SOPHUS_FUNC Eigen::Matrix<Scalar, 3, 4> C() {
    return sixfoldC() / Scalar(kCDenominator);
  }

  static SOPHUS_FUNC Vector3<Scalar> B(Scalar const& u) {
    Scalar u_square(u * u);
    return sixfoldC() * Vector4<Scalar>(Scalar(1), u, u_square, u * u_square) /
           Scalar(kCDenominator);
  }

  static SOPHUS_FUNC Vector3<Scalar> Dt_B(Scalar const& u,
                                          Scalar const& delta_t) {
    return sixfoldC() *
           Vector4<Scalar>(Scalar(0), Scalar(1), Scalar(2) * u,
                           Scalar(3) * u * u) /
           (Scalar(kCDenominator) * delta_t);
  }

  static SOPHUS_FUNC Vector3<Scalar> Dt2_B(Scalar const& u,
                                           Scalar const& delta_t) {
    return sixfoldC() *
           Vector4<Scalar>(Scalar(0), Scalar(0), Scalar(2), Scalar(6) * u) /
           (Scalar(kCDenominator) * delta_t * delta_t);
  }

 private:
  /// The common denominator of the entries of C().
  static constexpr int kCDenominator = 6;

  /// Returns C() times kCDenominator, whose entries are integers and thus
  /// exact, unlike the entries of C().
  static SOPHUS_FUNC Eigen::Matrix<Scalar, 3, 4> sixfoldC() {
    Eigen::Matrix<Scalar, 3, 4> C;
    // clang-format off
    C << Scalar(5), Scalar(3), Scalar(-3), Scalar(1),
         Scalar(1), Scalar(3), Scalar(3),  Scalar(-2),
         Scalar(0), Scalar(0), Scalar(0),  Scalar(1);
    // clang-format on
    return C;
  }
};

template <class LieGroup_>
class BasisSplineFn {
 public:
  using LieGroup = LieGroup_;
  using Scalar = typename LieGroup::Scalar;
  using Transformation = typename LieGroup::Transformation;
  using Tangent = typename LieGroup::Tangent;

  static LieGroup parent_T_spline(
      const LieGroup& parent_Ts_control_point,
      std::tuple<Tangent, Tangent, Tangent> const& control_tangent_vectors,
      double u) {
    auto AA = A(control_tangent_vectors, u);
    return parent_Ts_control_point * std::get<0>(AA) * std::get<1>(AA) *
           std::get<2>(AA);
  }

  static Transformation Dt_parent_T_spline(
      const LieGroup& parent_Ts_control_point,
      std::tuple<Tangent, Tangent, Tangent> const& control_tangent_vectors,
      double u, double delta_t) {
    auto AA = A(control_tangent_vectors, u);
    auto Dt_AA = Dt_A(AA, control_tangent_vectors, u, delta_t);
    return parent_Ts_control_point.matrix() *
           ((std::get<0>(Dt_AA) * std::get<1>(AA).matrix() *
             std::get<2>(AA).matrix()) +
            (std::get<0>(AA).matrix() * std::get<1>(Dt_AA) *
             std::get<2>(AA).matrix()) +
            (std::get<0>(AA).matrix() * std::get<1>(AA).matrix() *
             std::get<2>(Dt_AA)));
  }

  static Transformation Dt2_parent_T_spline(
      const LieGroup& parent_Ts_control_point,
      std::tuple<Tangent, Tangent, Tangent> const& control_tangent_vectors,
      double u, double delta_t) {
    using Scalar = typename LieGroup::Scalar;
    auto AA = A(control_tangent_vectors, u);
    auto Dt_AA = Dt_A(AA, control_tangent_vectors, u, delta_t);
    auto Dt2_AA = Dt2_A(AA, Dt_AA, control_tangent_vectors, u, delta_t);

    return parent_Ts_control_point.matrix() *
           ((std::get<0>(Dt2_AA) * std::get<1>(AA).matrix() *
             std::get<2>(AA).matrix()) +
            (std::get<0>(AA).matrix() * std::get<1>(Dt2_AA) *
             std::get<2>(AA).matrix()) +
            (std::get<0>(AA).matrix() * std::get<1>(AA).matrix() *
             std::get<2>(Dt2_AA)) +
            Scalar(2) * ((std::get<0>(Dt_AA) * std::get<1>(Dt_AA) *
                          std::get<2>(AA).matrix()) +
                         (std::get<0>(Dt_AA) * std::get<1>(AA).matrix() *
                          std::get<2>(Dt_AA)) +
                         (std::get<0>(AA).matrix() * std::get<1>(Dt_AA) *
                          std::get<2>(Dt_AA))));
  }

 private:
  static std::tuple<LieGroup, LieGroup, LieGroup> A(
      std::tuple<Tangent, Tangent, Tangent> const& control_tangent_vectors,
      double u) {
    Vector3<Scalar> const B = SplineBasisFunction<double>::B(u).cast<Scalar>();
    return std::make_tuple(
        LieGroup::exp(B[0] * std::get<0>(control_tangent_vectors)),
        LieGroup::exp(B[1] * std::get<1>(control_tangent_vectors)),
        LieGroup::exp(B[2] * std::get<2>(control_tangent_vectors)));
  }

  static std::tuple<Transformation, Transformation, Transformation> Dt_A(
      std::tuple<LieGroup, LieGroup, LieGroup> const& AA,
      const std::tuple<Tangent, Tangent, Tangent>& control_tangent_vectors,
      double u, double delta_t) {
    Eigen::Vector3d Dt_B = SplineBasisFunction<double>::Dt_B(u, delta_t);
    return std::make_tuple(
        Dt_B[0] * std::get<0>(AA).matrix() *
            LieGroup::hat(std::get<0>(control_tangent_vectors)),
        Dt_B[1] * std::get<1>(AA).matrix() *
            LieGroup::hat(std::get<1>(control_tangent_vectors)),
        Dt_B[2] * std::get<2>(AA).matrix() *
            LieGroup::hat(std::get<2>(control_tangent_vectors)));
  }

  static std::tuple<Transformation, Transformation, Transformation> Dt2_A(
      std::tuple<LieGroup, LieGroup, LieGroup> const& AA,
      std::tuple<Transformation, Transformation, Transformation> const& Dt_AA,
      std::tuple<Tangent, Tangent, Tangent> const& control_tangent_vectors,
      double u, double delta_t) {
    Eigen::Vector3d Dt_B = SplineBasisFunction<double>::Dt_B(u, delta_t);
    Eigen::Vector3d Dt2_B = SplineBasisFunction<double>::Dt2_B(u, delta_t);

    return std::make_tuple(
        (Dt_B[0] * std::get<0>(Dt_AA).matrix() +
         Dt2_B[0] * std::get<0>(AA).matrix()) *
            LieGroup::hat(std::get<0>(control_tangent_vectors)),
        (Dt_B[1] * std::get<1>(Dt_AA).matrix() +
         Dt2_B[1] * std::get<1>(AA).matrix()) *
            LieGroup::hat(std::get<1>(control_tangent_vectors)),
        (Dt_B[2] * std::get<2>(Dt_AA).matrix() +
         Dt2_B[2] * std::get<2>(AA).matrix()) *
            LieGroup::hat(std::get<2>(control_tangent_vectors)));
  }
};

enum class SegmentCase { first, normal, last };

template <class LieGroup>
struct BasisSplineSegment {
 public:
  using T = typename LieGroup::Scalar;
  using Transformation = typename LieGroup::Transformation;

  BasisSplineSegment(SegmentCase segment_case, T const* const raw_ptr0,
                     T const* const raw_ptr1, T const* const raw_ptr2,
                     T const* const raw_ptr3)
      : segment_case_(segment_case),
        raw_params0_(raw_ptr0),
        raw_params1_(raw_ptr1),
        raw_params2_(raw_ptr2),
        raw_params3_(raw_ptr3) {}

  Eigen::Map<LieGroup const> const world_pose_foo_prev() const {
    return Eigen::Map<LieGroup const>(raw_params0_);
  }
  Eigen::Map<LieGroup const> const world_pose_foo_0() const {
    return Eigen::Map<LieGroup const>(raw_params1_);
  }

  Eigen::Map<LieGroup const> const world_pose_foo_1() const {
    return Eigen::Map<LieGroup const>(raw_params2_);
  }

  Eigen::Map<LieGroup const> const world_pose_foo_2() const {
    return Eigen::Map<LieGroup const>(raw_params3_);
  }
  LieGroup parent_T_spline(double u) {
    switch (segment_case_) {
      case SegmentCase::first:
        return BasisSplineFn<LieGroup>::parent_T_spline(
            world_pose_foo_0(),
            std::make_tuple(
                (world_pose_foo_0().inverse() * world_pose_foo_0()).log(),
                (world_pose_foo_0().inverse() * world_pose_foo_1()).log(),
                (world_pose_foo_1().inverse() * world_pose_foo_2()).log()),
            u);
      case SegmentCase::normal:
        return BasisSplineFn<LieGroup>::parent_T_spline(
            world_pose_foo_prev(),
            std::make_tuple(
                (world_pose_foo_prev().inverse() * world_pose_foo_0()).log(),
                (world_pose_foo_0().inverse() * world_pose_foo_1()).log(),
                (world_pose_foo_1().inverse() * world_pose_foo_2()).log()),
            u);
      case SegmentCase::last:
        return BasisSplineFn<LieGroup>::parent_T_spline(
            world_pose_foo_prev(),
            std::make_tuple(
                (world_pose_foo_prev().inverse() * world_pose_foo_0()).log(),
                (world_pose_foo_0().inverse() * world_pose_foo_1()).log(),
                (world_pose_foo_1().inverse() * world_pose_foo_1()).log()),
            u);
    }
    SOPHUS_ENSURE(false, "invalid segment case {}", int(segment_case_));
    // Disabled ensures and ensure handlers return, so the function must not
    // fall off its end.
    std::abort();
  }

  Transformation Dt_parent_T_spline(double u, double delta_t) {
    switch (segment_case_) {
      case SegmentCase::first:
        return BasisSplineFn<LieGroup>::Dt_parent_T_spline(
            world_pose_foo_0(),
            std::make_tuple(
                (world_pose_foo_0().inverse() * world_pose_foo_0()).log(),
                (world_pose_foo_0().inverse() * world_pose_foo_1()).log(),
                (world_pose_foo_1().inverse() * world_pose_foo_2()).log()),
            u, delta_t);
      case SegmentCase::normal:
        return BasisSplineFn<LieGroup>::Dt_parent_T_spline(
            world_pose_foo_prev(),
            std::make_tuple(
                (world_pose_foo_prev().inverse() * world_pose_foo_0()).log(),
                (world_pose_foo_0().inverse() * world_pose_foo_1()).log(),
                (world_pose_foo_1().inverse() * world_pose_foo_2()).log()),
            u, delta_t);
      case SegmentCase::last:
        return BasisSplineFn<LieGroup>::Dt_parent_T_spline(
            world_pose_foo_prev(),
            std::make_tuple(
                (world_pose_foo_prev().inverse() * world_pose_foo_0()).log(),
                (world_pose_foo_0().inverse() * world_pose_foo_1()).log(),
                (world_pose_foo_1().inverse() * world_pose_foo_1()).log()),
            u, delta_t);
    }
    SOPHUS_ENSURE(false, "invalid segment case {}", int(segment_case_));
    // Disabled ensures and ensure handlers return, so the function must not
    // fall off its end.
    std::abort();
  }

  Transformation Dt2_parent_T_spline(double u, double delta_t) {
    switch (segment_case_) {
      case SegmentCase::first:
        return BasisSplineFn<LieGroup>::Dt2_parent_T_spline(
            world_pose_foo_0(),
            std::make_tuple(
                (world_pose_foo_0().inverse() * world_pose_foo_0()).log(),
                (world_pose_foo_0().inverse() * world_pose_foo_1()).log(),
                (world_pose_foo_1().inverse() * world_pose_foo_2()).log()),
            u, delta_t);
      case SegmentCase::normal:
        return BasisSplineFn<LieGroup>::Dt2_parent_T_spline(
            world_pose_foo_prev(),
            std::make_tuple(
                (world_pose_foo_prev().inverse() * world_pose_foo_0()).log(),
                (world_pose_foo_0().inverse() * world_pose_foo_1()).log(),
                (world_pose_foo_1().inverse() * world_pose_foo_2()).log()),
            u, delta_t);
      case SegmentCase::last:
        return BasisSplineFn<LieGroup>::Dt2_parent_T_spline(
            world_pose_foo_prev(),
            std::make_tuple(
                (world_pose_foo_prev().inverse() * world_pose_foo_0()).log(),
                (world_pose_foo_0().inverse() * world_pose_foo_1()).log(),
                (world_pose_foo_1().inverse() * world_pose_foo_1()).log()),
            u, delta_t);
    }
    SOPHUS_ENSURE(false, "invalid segment case {}", int(segment_case_));
    // Disabled ensures and ensure handlers return, so the function must not
    // fall off its end.
    std::abort();
  }

 private:
  SegmentCase segment_case_;
  T const* raw_params0_;
  T const* raw_params1_;
  T const* raw_params2_;
  T const* raw_params3_;
};

template <class LieGroup_>
class BasisSplineImpl {
 public:
  using LieGroup = LieGroup_;
  using Scalar = typename LieGroup::Scalar;
  using Transformation = typename LieGroup::Transformation;
  using Tangent = typename LieGroup::Tangent;

  BasisSplineImpl(const std::vector<LieGroup>& parent_Ts_control_point,
                  double delta_t)
      : parent_Ts_control_point_(parent_Ts_control_point), delta_t_(delta_t) {
    SOPHUS_ENSURE(parent_Ts_control_point_.size() >= 2u, ", but {}",
                  parent_Ts_control_point_.size());
  }

  LieGroup parent_T_spline(int i, double u) const {
    return segment(i).parent_T_spline(u);
  }

  Transformation Dt_parent_T_spline(int i, double u) const {
    return segment(i).Dt_parent_T_spline(u, delta_t_);
  }

  Transformation Dt2_parent_T_spline(int i, double u) const {
    return segment(i).Dt2_parent_T_spline(u, delta_t_);
  }

  const std::vector<LieGroup>& parent_Ts_control_point() const {
    return parent_Ts_control_point_;
  }

  std::vector<LieGroup>& parent_Ts_control_point() {
    return parent_Ts_control_point_;
  }

  [[nodiscard]] int getNumSegments() const {
    return int(parent_Ts_control_point_.size()) - 1;
  }

  [[nodiscard]] double delta_t() const { return delta_t_; }

 private:
  /// Returns segment ``i`` with its neighboring control points, clamped to
  /// the first and the last control point at the ends of the spline.
  BasisSplineSegment<LieGroup> segment(int i) const {
    SOPHUS_ENSURE(i >= 0 && i < this->getNumSegments(),
                  "i is {}, but must be in [0, {}].", i,
                  this->getNumSegments() - 1);

    SegmentCase const segment_case =
        i == 0 ? SegmentCase::first
               : (i == this->getNumSegments() - 1 ? SegmentCase::last
                                                  : SegmentCase::normal);

    std::size_t const index = static_cast<std::size_t>(i);
    std::size_t const last = parent_Ts_control_point_.size() - 1;
    return BasisSplineSegment<LieGroup>(
        segment_case,
        parent_Ts_control_point_[index > 0 ? index - 1 : 0].data(),
        parent_Ts_control_point_[index].data(),
        parent_Ts_control_point_[index + 1].data(),
        parent_Ts_control_point_[std::min(index + 2, last)].data());
  }

  std::vector<LieGroup> parent_Ts_control_point_;
  double delta_t_;
};

struct IndexAndU {
  int i;
  double u;
};

template <class LieGroup_>
class BasisSpline {
 public:
  using LieGroup = LieGroup_;
  using Scalar = typename LieGroup::Scalar;
  using Transformation = typename LieGroup::Transformation;
  using Tangent = typename LieGroup::Tangent;

  BasisSpline(std::vector<LieGroup> parent_Ts_control_point, double t0,
              double delta_t)
      : impl_(std::move(parent_Ts_control_point), delta_t), t0_(t0) {}

  LieGroup parent_T_spline(double t) const {
    IndexAndU index_and_u = this->index_and_u(t);

    return impl_.parent_T_spline(index_and_u.i, index_and_u.u);
  }

  Transformation Dt_parent_T_spline(double t) const {
    IndexAndU index_and_u = this->index_and_u(t);
    return impl_.Dt_parent_T_spline(index_and_u.i, index_and_u.u);
  }

  Transformation Dt2_parent_T_spline(double t) const {
    IndexAndU index_and_u = this->index_and_u(t);
    return impl_.Dt2_parent_T_spline(index_and_u.i, index_and_u.u);
  }

  [[nodiscard]] double t0() const { return t0_; }

  [[nodiscard]] double tmax() const {
    return t0_ + impl_.delta_t() * getNumSegments();
  }

  const std::vector<LieGroup>& parent_Ts_control_point() const {
    return impl_.parent_Ts_control_point();
  }

  std::vector<LieGroup>& parent_Ts_control_point() {
    return impl_.parent_Ts_control_point();
  }

  [[nodiscard]] int getNumSegments() const { return impl_.getNumSegments(); }

  [[nodiscard]] double s(double t) const { return (t - t0_) / impl_.delta_t(); }

  double delta_t() const { return impl_.delta_t(); }

  IndexAndU index_and_u(double t) const {
    SOPHUS_ENSURE(t >= t0_, "{} vs. {}", t, t0_);
    SOPHUS_ENSURE(t <= this->tmax(), "{} vs. {}", t, this->tmax());

    double s = this->s(t);
    double i;
    IndexAndU index_and_u;
    index_and_u.u = std::modf(s, &i);
    index_and_u.i = int(i);
    if (index_and_u.u > Sophus::Constants<double>::epsilon()) {
      return index_and_u;
    }

    // u ~=~ 0.0. The spline parameter s counts segments from t0, so its
    // midpoint is half the number of segments.
    if (s < double(this->getNumSegments()) / 2) {
      // First half of spline, keep as is (i, 0.0).
      return index_and_u;
    }
    // Second half of spline, use (i-1, 1.0) instead. This way we can represent
    // t == tmax (and not just t<tmax).
    index_and_u.u += 1.0;
    --index_and_u.i;

    return index_and_u;
  }

 private:
  BasisSplineImpl<LieGroup> impl_;

  double t0_;
};

}  // namespace Sophus
