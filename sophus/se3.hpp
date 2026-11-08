// Copyright 2021-2022, 2026 Sergiu Deitsch <sergiu.deitsch@gmail.com>

/// @file
/// Special Euclidean group SE(3) - rotation and translation in 3d.

#ifndef SOPHUS_SE3_HPP
#define SOPHUS_SE3_HPP

#include "so3.hpp"

namespace Sophus {
template <class Scalar_, int Options = 0>
class SE3;
using SE3d = SE3<double>;
using SE3f = SE3<float>;
}  // namespace Sophus

namespace Eigen {
namespace internal {

template <class Scalar_, int Options>
struct traits<Sophus::SE3<Scalar_, Options>> {
  using Scalar = Scalar_;
  using TranslationType = Sophus::Vector3<Scalar, Options>;
  using SO3Type = Sophus::SO3<Scalar, Options>;
};

template <class Scalar_, int Options>
struct traits<Map<Sophus::SE3<Scalar_>, Options>>
    : traits<Sophus::SE3<Scalar_, Options>> {
  using Scalar = Scalar_;
  using TranslationType = Map<Sophus::Vector3<Scalar>, Options>;
  using SO3Type = Map<Sophus::SO3<Scalar>, Options>;
};

template <class Scalar_, int Options>
struct traits<Map<Sophus::SE3<Scalar_> const, Options>>
    : traits<Sophus::SE3<Scalar_, Options> const> {
  using Scalar = Scalar_;
  using TranslationType = Map<Sophus::Vector3<Scalar> const, Options>;
  using SO3Type = Map<Sophus::SO3<Scalar> const, Options>;
};
}  // namespace internal
}  // namespace Eigen

namespace Sophus {

/// SE3 base type - implements SE3 class but is storage agnostic.
///
/// SE(3) is the group of rotations  and translation in 3d. It is the
/// semi-direct product of SO(3) and the 3d Euclidean vector space.  The class
/// is represented using a composition of SO3  for rotation and a one 3-vector
/// for translation.
///
/// SE(3) is neither compact, nor a commutative group.
///
/// See SO3 for more details of the rotation representation in 3d.
///
template <class Derived>
class SE3Base {
 public:
  using Scalar = typename Eigen::internal::traits<Derived>::Scalar;
  using TranslationType =
      typename Eigen::internal::traits<Derived>::TranslationType;
  using SO3Type = typename Eigen::internal::traits<Derived>::SO3Type;
  using QuaternionType = typename SO3Type::QuaternionType;
  /// Degrees of freedom of manifold, number of dimensions in tangent space
  /// (three for translation, three for rotation).
  static int constexpr DoF = 6;
  /// Number of internal parameters used (4-tuple for quaternion, three for
  /// translation).
  static int constexpr num_parameters = 7;
  /// Group transformations are 4x4 matrices.
  static int constexpr N = 4;
  /// Points are 3-dimensional
  static int constexpr Dim = 3;
  using Transformation = Matrix<Scalar, N, N>;
  using Point = Vector3<Scalar>;
  using HomogeneousPoint = Vector4<Scalar>;
  using Line = ParametrizedLine3<Scalar>;
  using Hyperplane = Hyperplane3<Scalar>;
  using Tangent = Vector<Scalar, DoF>;
  using Adjoint = Matrix<Scalar, DoF, DoF>;

  /// For binary operations the return type is determined with the
  /// ScalarBinaryOpTraits feature of Eigen. This allows mixing concrete and Map
  /// types, as well as other compatible scalar types such as Ceres::Jet and
  /// double scalars with SE3 operations.
  template <typename OtherDerived>
  using ReturnScalar = typename Eigen::ScalarBinaryOpTraits<
      Scalar, typename OtherDerived::Scalar>::ReturnType;

  template <typename OtherDerived>
  using SE3Product = SE3<ReturnScalar<OtherDerived>>;

  template <typename PointDerived>
  using PointProduct = Vector3<ReturnScalar<PointDerived>>;

  template <typename HPointDerived>
  using HomogeneousPointProduct = Vector4<ReturnScalar<HPointDerived>>;

  /// Adjoint transformation
  ///
  /// This function return the adjoint transformation ``Ad`` of the group
  /// element ``A`` such that for all ``x`` it holds that
  /// ``hat(Ad_A * x) = A * hat(x) A^{-1}``. See hat-operator below.
  ///
  [[nodiscard]] SOPHUS_FUNC Adjoint Adj() const {
    Sophus::Matrix3<Scalar> const R = so3().matrix();
    Adjoint res;
    // clang-format off
    res << R, SO3<Scalar>::hat(translation()) * R,
           Matrix3<Scalar>::Zero(), R;
    // clang-format on
    return res;
  }

  /// Extract rotation angle about canonical X-axis
  ///
  [[nodiscard]] Scalar angleX() const { return so3().angleX(); }

  /// Extract rotation angle about canonical Y-axis
  ///
  [[nodiscard]] Scalar angleY() const { return so3().angleY(); }

  /// Extract rotation angle about canonical Z-axis
  ///
  [[nodiscard]] Scalar angleZ() const { return so3().angleZ(); }

  /// Returns copy of instance casted to NewScalarType.
  ///
  template <class NewScalarType>
  [[nodiscard]] SOPHUS_FUNC SE3<NewScalarType> cast() const {
    return SE3<NewScalarType>(so3().template cast<NewScalarType>(),
                              translation().template cast<NewScalarType>());
  }

  /// Returns derivative of  this * exp(x)  wrt x at x=0.
  ///
  [[nodiscard]] SOPHUS_FUNC Matrix<Scalar, num_parameters, DoF>
  Dx_this_mul_exp_x_at_0() const {
    Matrix<Scalar, num_parameters, DoF> J;
    Eigen::Quaternion<Scalar> const q = unit_quaternion();
    Scalar const c0 = q.w() / Scalar(2);
    Scalar const c1 = q.z() / Scalar(2);
    Scalar const c2 = -c1;
    Scalar const c3 = q.y() / Scalar(2);
    Scalar const c4 = q.x() / Scalar(2);
    Scalar const c5 = -c4;
    Scalar const c6 = -c3;
    Scalar const c7 = q.w() * q.w();
    Scalar const c8 = q.x() * q.x();
    Scalar const c9 = q.y() * q.y();
    Scalar const c10 = -c9;
    Scalar const c11 = q.z() * q.z();
    Scalar const c12 = -c11;
    Scalar const c13 = Scalar(2) * q.w();
    Scalar const c14 = c13 * q.z();
    Scalar const c15 = Scalar(2) * q.x();
    Scalar const c16 = c15 * q.y();
    Scalar const c17 = c13 * q.y();
    Scalar const c18 = c15 * q.z();
    Scalar const c19 = c7 - c8;
    Scalar const c20 = c13 * q.x();
    Scalar const c21 = Scalar(2) * q.y() * q.z();
    J(0, 0) = 0;
    J(0, 1) = 0;
    J(0, 2) = 0;
    J(0, 3) = c0;
    J(0, 4) = c2;
    J(0, 5) = c3;
    J(1, 0) = 0;
    J(1, 1) = 0;
    J(1, 2) = 0;
    J(1, 3) = c1;
    J(1, 4) = c0;
    J(1, 5) = c5;
    J(2, 0) = 0;
    J(2, 1) = 0;
    J(2, 2) = 0;
    J(2, 3) = c6;
    J(2, 4) = c4;
    J(2, 5) = c0;
    J(3, 0) = 0;
    J(3, 1) = 0;
    J(3, 2) = 0;
    J(3, 3) = c5;
    J(3, 4) = c6;
    J(3, 5) = c2;
    J(4, 0) = c10 + c12 + c7 + c8;
    J(4, 1) = -c14 + c16;
    J(4, 2) = c17 + c18;
    J(4, 3) = 0;
    J(4, 4) = 0;
    J(4, 5) = 0;
    J(5, 0) = c14 + c16;
    J(5, 1) = c12 + c19 + c9;
    J(5, 2) = -c20 + c21;
    J(5, 3) = 0;
    J(5, 4) = 0;
    J(5, 5) = 0;
    J(6, 0) = -c17 + c18;
    J(6, 1) = c20 + c21;
    J(6, 2) = c10 + c11 + c19;
    J(6, 3) = 0;
    J(6, 4) = 0;
    J(6, 5) = 0;
    return J;
  }

  /// Returns derivative of log(this^{-1} * x) by x at x=this.
  ///
  SOPHUS_FUNC Matrix<Scalar, DoF, num_parameters> Dx_log_this_inv_by_x_at_this()
      const {
    Matrix<Scalar, DoF, num_parameters> J;
    J.template block<3, 4>(0, 0).setZero();
    J.template block<3, 3>(0, 4) = so3().inverse().matrix();
    J.template block<3, 4>(3, 0) = so3().Dx_log_this_inv_by_x_at_this();
    J.template block<3, 3>(3, 4).setZero();
    return J;
  }

  /// Returns group inverse.
  ///
  [[nodiscard]] SOPHUS_FUNC SE3<Scalar> inverse() const {
    SO3<Scalar> invR = so3().inverse();
    return SE3<Scalar>(invR, invR * (translation() * Scalar(-1)));
  }

  /// Logarithmic map
  ///
  /// Computes the logarithm, the inverse of the group exponential which maps
  /// element of the group (rigid body transformations) to elements of the
  /// tangent space (twist).
  ///
  /// To be specific, this function computes ``vee(logmat(.))`` with
  /// ``logmat(.)`` being the matrix logarithm and ``vee(.)`` the vee-operator
  /// of SE(3).
  ///
  [[nodiscard]] SOPHUS_FUNC Tangent log() const {
    // For the derivation of the logarithm of SE(3), see
    // J. Gallier, D. Xu, "Computing exponentials of skew symmetric matrices
    // and logarithms of orthogonal matrices", IJRA 2002.
    // https://pdfs.semanticscholar.org/cfe3/e4b39de63c8cabd89bf3feff7f5449fc981d.pdf
    // (Sec. 6., pp. 8)
    using std::abs;
    using std::cos;
    using std::sin;
    Tangent upsilon_omega;
    auto omega_and_theta = so3().logAndTheta();
    Scalar theta = omega_and_theta.theta;
    Vector3<Scalar> const& omega = omega_and_theta.tangent;
    upsilon_omega.template tail<3>() = omega;
    Matrix3<Scalar> V_inv = SO3<Scalar>::leftJacobianInverse(omega, theta);
    upsilon_omega.template head<3>() = V_inv * translation();
    return upsilon_omega;
  }

  /// It re-normalizes the SO3 element.
  ///
  /// Note: Because of the class invariant of SO3, there is typically no need to
  /// call this function directly.
  ///
  SOPHUS_FUNC void normalize() { so3().normalize(); }

  /// Returns 4x4 matrix representation of the instance.
  ///
  /// It has the following form:
  ///
  ///   | R t |
  ///   | o 1 |
  ///
  /// where ``R`` is a 3x3 rotation matrix, ``t`` a translation 3-vector and
  /// ``o`` a 3-column vector of zeros.
  ///
  [[nodiscard]] SOPHUS_FUNC Transformation matrix() const {
    Transformation homogenious_matrix;
    homogenious_matrix << matrix3x4(), Matrix<Scalar, 1, 4>::UnitW();
    return homogenious_matrix;
  }

  /// Returns the significant first three rows of the matrix above.
  ///
  [[nodiscard]] SOPHUS_FUNC Matrix<Scalar, 3, 4> matrix3x4() const {
    Matrix<Scalar, 3, 4> matrix;
    matrix << rotationMatrix(), translation();
    return matrix;
  }

  /// Assignment-like operator from OtherDerived.
  ///
  template <class OtherDerived>
  SOPHUS_FUNC SE3Base<Derived>& operator=(SE3Base<OtherDerived> const& other) {
    so3() = other.so3();
    translation() = other.translation();
    return *this;
  }

  /// Group multiplication, which is rotation concatenation.
  ///
  template <typename OtherDerived>
  [[nodiscard]] SOPHUS_FUNC SE3Product<OtherDerived> operator*(
      SE3Base<OtherDerived> const& other) const {
    return SE3Product<OtherDerived>(
        so3() * other.so3(), translation() + so3() * other.translation());
  }

  /// Group action on 3-points.
  ///
  /// This function rotates and translates a three dimensional point ``p`` by
  /// the SE(3) element ``bar_T_foo = (bar_R_foo, t_bar)`` (= rigid body
  /// transformation):
  ///
  ///   ``p_bar = bar_R_foo * p_foo + t_bar``.
  ///
  template <
      typename PointDerived,
      typename = std::enable_if_t<IsFixedSizeVector<PointDerived, 3>::value>>
  [[nodiscard]] SOPHUS_FUNC PointProduct<PointDerived> operator*(
      Eigen::MatrixBase<PointDerived> const& p) const {
    return so3() * p + translation();
  }

  /// Group action on homogeneous 3-points. See above for more details.
  ///
  template <
      typename HPointDerived,
      typename = std::enable_if_t<IsFixedSizeVector<HPointDerived, 4>::value>>
  [[nodiscard]] SOPHUS_FUNC HomogeneousPointProduct<HPointDerived> operator*(
      Eigen::MatrixBase<HPointDerived> const& p) const {
    const PointProduct<HPointDerived> tp =
        so3() * p.template head<3>() + p(3) * translation();
    return HomogeneousPointProduct<HPointDerived>(tp(0), tp(1), tp(2), p(3));
  }

  /// Group action on lines.
  ///
  /// This function rotates and translates a parametrized line
  /// ``l(t) = o + t * d`` by the SE(3) element:
  ///
  /// Origin is transformed using SE(3) action
  /// Direction is transformed using rotation part
  ///
  [[nodiscard]] SOPHUS_FUNC Line operator*(Line const& l) const {
    return Line((*this) * l.origin(), so3() * l.direction());
  }

  /// Group action on planes.
  ///
  /// This function rotates and translates a plane
  /// ``n.x + d = 0`` by the SE(3) element:
  ///
  /// Normal vector ``n`` is rotated
  /// Offset ``d`` is adjusted for translation
  ///
  [[nodiscard]] SOPHUS_FUNC Hyperplane operator*(Hyperplane const& p) const {
    Hyperplane const rotated = so3() * p;
    return Hyperplane(rotated.normal(),
                      rotated.offset() - translation().dot(rotated.normal()));
  }

  /// In-place group multiplication. This method is only valid if the return
  /// type of the multiplication is compatible with this SE3's Scalar type.
  ///
  template <typename OtherDerived, typename = std::enable_if_t<std::is_same_v<
                                       Scalar, ReturnScalar<OtherDerived>>>>
  SOPHUS_FUNC SE3Base<Derived>& operator*=(SE3Base<OtherDerived> const& other) {
    *static_cast<Derived*>(this) = *this * other;
    return *this;
  }

  /// Returns rotation matrix.
  ///
  [[nodiscard]] SOPHUS_FUNC Matrix3<Scalar> rotationMatrix() const {
    return so3().matrix();
  }

  /// Mutator of SO3 group.
  ///
  [[nodiscard]] SOPHUS_FUNC SO3Type& so3() {
    return static_cast<Derived*>(this)->so3();
  }

  /// Accessor of SO3 group.
  ///
  [[nodiscard]] SOPHUS_FUNC SO3Type const& so3() const {
    return static_cast<const Derived*>(this)->so3();
  }

  /// Takes in quaternion, and normalizes it.
  ///
  /// Precondition: The quaternion must not be close to zero.
  ///
  SOPHUS_FUNC void setQuaternion(Eigen::Quaternion<Scalar> const& quat) {
    so3().setQuaternion(quat);
  }

  /// Sets ``so3`` using ``rotation_matrix``.
  ///
  /// Precondition: ``R`` must be orthogonal and ``det(R)=1``.
  ///
  SOPHUS_FUNC void setRotationMatrix(Matrix3<Scalar> const& R) {
    SOPHUS_ENSURE(isOrthogonal(R), "R is not orthogonal:\n {}",
                  SOPHUS_FMT_ARG(R));
    SOPHUS_ENSURE(R.determinant() > Scalar(0), "det(R) is not positive: {}",
                  SOPHUS_FMT_ARG(R.determinant()));
    so3().setQuaternion(Eigen::Quaternion<Scalar>(R));
  }

  /// Returns internal parameters of SE(3).
  ///
  /// It returns (q.imag[0], q.imag[1], q.imag[2], q.real, t[0], t[1], t[2]),
  /// with q being the unit quaternion, t the translation 3-vector.
  ///
  [[nodiscard]] SOPHUS_FUNC Sophus::Vector<Scalar, num_parameters> params()
      const {
    Sophus::Vector<Scalar, num_parameters> p;
    p << so3().params(), translation();
    return p;
  }

  /// Mutator of translation vector.
  ///
  [[nodiscard]] SOPHUS_FUNC TranslationType& translation() {
    return static_cast<Derived*>(this)->translation();
  }

  /// Accessor of translation vector
  ///
  [[nodiscard]] SOPHUS_FUNC TranslationType const& translation() const {
    return static_cast<Derived const*>(this)->translation();
  }

  /// Accessor of unit quaternion.
  ///
  [[nodiscard]] SOPHUS_FUNC QuaternionType const& unit_quaternion() const {
    return this->so3().unit_quaternion();
  }
};

/// SE3 using default storage; derived from SE3Base.
template <class Scalar_, int Options>
class SE3 : public SE3Base<SE3<Scalar_, Options>> {
  using Base = SE3Base<SE3<Scalar_, Options>>;

 public:
  static int constexpr DoF = Base::DoF;
  static int constexpr num_parameters = Base::num_parameters;

  using Scalar = Scalar_;
  using Transformation = typename Base::Transformation;
  using Point = typename Base::Point;
  using HomogeneousPoint = typename Base::HomogeneousPoint;
  using Tangent = typename Base::Tangent;
  using Adjoint = typename Base::Adjoint;
  using SO3Member = SO3<Scalar, Options>;
  using TranslationMember = Vector3<Scalar, Options>;

  using Base::operator=;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /// Default constructor initializes rigid body motion to the identity.
  ///
  [[nodiscard]] SOPHUS_FUNC SE3();

  /// Copy-like constructor from OtherDerived.
  ///
  template <class OtherDerived>
  [[nodiscard]] SOPHUS_FUNC SE3(SE3Base<OtherDerived> const& other)
      : so3_(other.so3()), translation_(other.translation()) {
    static_assert(std::is_same_v<typename OtherDerived::Scalar, Scalar>,
                  "must be same Scalar type");
  }

  /// Constructor from SO3 and translation vector
  ///
  template <class OtherDerived, class D>
  [[nodiscard]] SOPHUS_FUNC SE3(SO3Base<OtherDerived> const& so3,
                                Eigen::MatrixBase<D> const& translation)
      : so3_(so3), translation_(translation) {
    static_assert(std::is_same_v<typename OtherDerived::Scalar, Scalar>,
                  "must be same Scalar type");
    static_assert(std::is_same_v<typename D::Scalar, Scalar>,
                  "must be same Scalar type");
  }

  /// Constructor from rotation matrix and translation vector
  ///
  /// Precondition: Rotation matrix needs to be orthogonal with determinant
  ///               of 1.
  ///
  [[nodiscard]] SOPHUS_FUNC SE3(Matrix3<Scalar> const& rotation_matrix,
                                Point const& translation)
      : so3_(rotation_matrix), translation_(translation) {}

  /// Constructor from quaternion and translation vector.
  ///
  /// Precondition: ``quaternion`` must not be close to zero.
  ///
  [[nodiscard]] SOPHUS_FUNC SE3(Eigen::Quaternion<Scalar> const& quaternion,
                                Point const& translation)
      : so3_(quaternion), translation_(translation) {}

  /// Constructor from 4x4 matrix
  ///
  /// Precondition: Rotation matrix needs to be orthogonal with determinant
  ///               of 1. The last row must be ``(0, 0, 0, 1)``.
  ///
  [[nodiscard]] SOPHUS_FUNC explicit SE3(Matrix4<Scalar> const& T)
      : so3_(T.template topLeftCorner<3, 3>()),
        translation_(T.template rightCols<1>().template head<3>()) {
    SOPHUS_ENSURE((T.row(3) - Matrix<Scalar, 1, 4>::UnitW()).squaredNorm() <
                      Constants<Scalar>::epsilon(),
                  "Last row is not (0,0,0,1), but ({}).",
                  SOPHUS_FMT_ARG(T.row(3)));
  }

  /// This provides unsafe read/write access to internal data. SO(3) is
  /// represented by an Eigen::Quaternion (four parameters). When using direct
  /// write access, the user needs to take care of that the quaternion stays
  /// normalized.
  ///
  [[nodiscard]] SOPHUS_FUNC Scalar* data() {
    // so3_ and translation_ are laid out sequentially with no padding
    return so3_.data();
  }

  /// Const version of data() above.
  ///
  [[nodiscard]] SOPHUS_FUNC Scalar const* data() const {
    // so3_ and translation_ are laid out sequentially with no padding
    return so3_.data();
  }

  /// Mutator of SO3
  ///
  [[nodiscard]] SOPHUS_FUNC SO3Member& so3() { return so3_; }

  /// Accessor of SO3
  ///
  [[nodiscard]] SOPHUS_FUNC SO3Member const& so3() const { return so3_; }

  /// Mutator of translation vector
  ///
  [[nodiscard]] SOPHUS_FUNC TranslationMember& translation() {
    return translation_;
  }

  /// Accessor of translation vector
  ///
  [[nodiscard]] SOPHUS_FUNC TranslationMember const& translation() const {
    return translation_;
  }

  [[nodiscard]] SOPHUS_FUNC static Matrix3<Scalar> jacobianUpperRightBlock(
      Vector3<Scalar> const& upsilon, Vector3<Scalar> const& omega) {
    using std::cos;
    using std::hypot;
    using std::sin;

    Scalar const theta_sq = omega.squaredNorm();
    Matrix3<Scalar> const Upsilon = SO3<Scalar>::hat(upsilon);

    // The coefficients
    //   c1 = (theta - sin(theta)) / theta^3,
    //   c2 = (theta^2 / 2 + cos(theta) - 1) / theta^4,
    //   c3 = (theta + theta cos(theta) / 2 - 3 sin(theta) / 2) / theta^5
    // cancel for small angles faster than the powers of Omega they multiply
    // vanish, so their Taylor series replace them there.
    Scalar c1;
    Scalar c2;
    Scalar c3;
    if (theta_sq < Scalar(details::kSeriesThetaSqBound)) {
      c1 = details::alternatingSeries(theta_sq, 3, [](int) { return 1; });
      c2 = details::alternatingSeries(theta_sq, 4, [](int) { return 1; });
      c3 = details::alternatingSeries(theta_sq, 5, [](int k) { return k + 1; });
    } else {
      Scalar const theta = hypot(omega.x(), omega.y(), omega.z());
      Scalar const theta_po4 = theta_sq * theta_sq;
      Scalar const st = sin(theta);
      Scalar const ct = cos(theta);
      c1 = (theta - st) / (theta_sq * theta);
      c2 = (theta_sq / Scalar(2) + ct - Scalar(1)) / theta_po4;
      c3 = (theta + theta * ct / Scalar(2) - Scalar(1.5) * st) /
           (theta_po4 * theta);
    }

    Matrix3<Scalar> const Omega = SO3<Scalar>::hat(omega);
    Matrix3<Scalar> const OmegaUpsilon = Omega * Upsilon;
    Matrix3<Scalar> const OmegaUpsilonOmega = OmegaUpsilon * Omega;
    return Upsilon / Scalar(2) +
           c1 * (OmegaUpsilon + Upsilon * Omega + OmegaUpsilonOmega) -
           c2 * (theta_sq * Upsilon + Scalar(2) * OmegaUpsilonOmega) +
           c3 * (OmegaUpsilonOmega * Omega + Omega * OmegaUpsilonOmega);
  }

  [[nodiscard]] SOPHUS_FUNC static Sophus::Matrix<Scalar, DoF, DoF>
  leftJacobian(Tangent const& upsilon_omega) {
    Vector3<Scalar> const upsilon = upsilon_omega.template head<3>();
    Vector3<Scalar> const omega = upsilon_omega.template tail<3>();
    Matrix3<Scalar> const J = SO3<Scalar>::leftJacobian(omega);
    Matrix3<Scalar> Q = jacobianUpperRightBlock(upsilon, omega);
    Matrix6<Scalar> U;
    U << J, Q, Matrix3<Scalar>::Zero(), J;
    return U;
  }

  [[nodiscard]] SOPHUS_FUNC static Sophus::Matrix<Scalar, DoF, DoF>
  leftJacobianInverse(Tangent const& upsilon_omega) {
    Vector3<Scalar> const upsilon = upsilon_omega.template head<3>();
    Vector3<Scalar> const omega = upsilon_omega.template tail<3>();
    Matrix3<Scalar> const J_inv = SO3<Scalar>::leftJacobianInverse(omega);
    Matrix3<Scalar> Q = jacobianUpperRightBlock(upsilon, omega);
    Matrix6<Scalar> U;
    U << J_inv, -J_inv * Q * J_inv, Matrix3<Scalar>::Zero(), J_inv;
    return U;
  }

  /// Returns derivative of exp(x) wrt. x.
  ///
  [[nodiscard]] SOPHUS_FUNC static Sophus::Matrix<Scalar, num_parameters, DoF>
  Dx_exp_x(Tangent const& upsilon_omega) {
    Vector3<Scalar> const upsilon = upsilon_omega.template head<3>();
    Vector3<Scalar> const omega = upsilon_omega.template tail<3>();
    // The translation t = J(omega) upsilon of exp(x) changes by
    // J(omega) d_upsilon + Q(upsilon, omega) d_omega - t x (J(omega) d_omega),
    // since exp(x + dx) = exp(J_SE3(x) dx) exp(x) to first order with the
    // left Jacobian J_SE3 = [J Q; 0 J], and a left-multiplied rotation by
    // J(omega) d_omega moves t by its cross product with t. Building on the
    // left Jacobian avoids the cancellation of a closed form.
    Matrix3<Scalar> const J_omega = SO3<Scalar>::leftJacobian(omega);
    Vector3<Scalar> const translation = J_omega * upsilon;
    Sophus::Matrix<Scalar, num_parameters, DoF> J;
    J.template topLeftCorner<4, 3>().setZero();
    J.template topRightCorner<4, 3>() = SO3<Scalar>::Dx_exp_x(omega);
    J.template bottomLeftCorner<3, 3>() = J_omega;
    J.template bottomRightCorner<3, 3>() =
        jacobianUpperRightBlock(upsilon, omega) -
        SO3<Scalar>::hat(translation) * J_omega;
    return J;
  }

  /// Returns derivative of exp(x) wrt. x_i at x=0.
  ///
  [[nodiscard]] SOPHUS_FUNC static Sophus::Matrix<Scalar, num_parameters, DoF>
  Dx_exp_x_at_0() {
    Sophus::Matrix<Scalar, num_parameters, DoF> J;
    Scalar const o(0);
    Scalar const h(0.5);
    Scalar const i(1);

    // clang-format off
    J << o, o, o, h, o, o,
         o, o, o, o, h, o,
	 o, o, o, o, o, h,
	 o, o, o, o, o, o,
	 i, o, o, o, o, o,
	 o, i, o, o, o, o,
	 o, o, i, o, o, o;
    // clang-format on
    return J;
  }

  /// Returns derivative of exp(x).matrix() wrt. ``x_i at x=0``.
  ///
  [[nodiscard]] SOPHUS_FUNC static Transformation Dxi_exp_x_matrix_at_0(int i) {
    return generator(i);
  }

  /// Returns derivative of exp(x) * p wrt. x_i at x=0.
  ///
  SOPHUS_FUNC static Sophus::Matrix<Scalar, 3, DoF> Dx_exp_x_times_point_at_0(
      Point const& point) {
    Sophus::Matrix<Scalar, 3, DoF> J;
    J << Sophus::Matrix3<Scalar>::Identity(),
        Sophus::SO3<Scalar>::Dx_exp_x_times_point_at_0(point);
    return J;
  }

  /// Group exponential
  ///
  /// This functions takes in an element of tangent space (= twist ``a``) and
  /// returns the corresponding element of the group SE(3).
  ///
  /// The first three components of ``a`` represent the translational part
  /// ``upsilon`` in the tangent space of SE(3), while the last three components
  /// of ``a`` represents the rotation vector ``omega``.
  /// To be more specific, this function computes ``expmat(hat(a))`` with
  /// ``expmat(.)`` being the matrix exponential and ``hat(.)`` the hat-operator
  /// of SE(3), see below.
  ///
  [[nodiscard]] SOPHUS_FUNC static SE3<Scalar> exp(Tangent const& a) {
    using std::cos;
    using std::sin;
    Vector3<Scalar> const omega = a.template tail<3>();

    Scalar theta;
    SO3<Scalar> const so3 = SO3<Scalar>::expAndTheta(omega, &theta);
    Matrix3<Scalar> const V = SO3<Scalar>::leftJacobian(omega, theta);
    return SE3<Scalar>(so3, V * a.template head<3>());
  }

  /// Returns closest SE3 given arbitrary 4x4 matrix.
  ///
  template <class S = Scalar>
  [[nodiscard]] SOPHUS_FUNC static std::enable_if_t<std::is_floating_point_v<S>,
                                                    SE3>
  fitToSE3(Matrix4<Scalar> const& T) {
    return SE3(SO3<Scalar>::fitToSO3(T.template topLeftCorner<3, 3>()),
               T.template rightCols<1>().template head<3>());
  }

  /// Returns the ith infinitesimal generators of SE(3).
  ///
  /// The infinitesimal generators of SE(3) are:
  ///
  /// ```
  ///         |  0  0  0  1 |
  ///   G_0 = |  0  0  0  0 |
  ///         |  0  0  0  0 |
  ///         |  0  0  0  0 |
  ///
  ///         |  0  0  0  0 |
  ///   G_1 = |  0  0  0  1 |
  ///         |  0  0  0  0 |
  ///         |  0  0  0  0 |
  ///
  ///         |  0  0  0  0 |
  ///   G_2 = |  0  0  0  0 |
  ///         |  0  0  0  1 |
  ///         |  0  0  0  0 |
  ///
  ///         |  0  0  0  0 |
  ///   G_3 = |  0  0 -1  0 |
  ///         |  0  1  0  0 |
  ///         |  0  0  0  0 |
  ///
  ///         |  0  0  1  0 |
  ///   G_4 = |  0  0  0  0 |
  ///         | -1  0  0  0 |
  ///         |  0  0  0  0 |
  ///
  ///         |  0 -1  0  0 |
  ///   G_5 = |  1  0  0  0 |
  ///         |  0  0  0  0 |
  ///         |  0  0  0  0 |
  /// ```
  ///
  /// Precondition: ``i`` must be in [0, 5].
  ///
  [[nodiscard]] SOPHUS_FUNC static Transformation generator(int i) {
    SOPHUS_ENSURE(i >= 0 && i <= 5, "i should be in range [0,5].");
    Tangent e;
    e.setZero();
    e[i] = Scalar(1);
    return hat(e);
  }

  /// hat-operator
  ///
  /// It takes in the 6-vector representation (= twist) and returns the
  /// corresponding matrix representation of Lie algebra element.
  ///
  /// Formally, the hat()-operator of SE(3) is defined as
  ///
  ///   ``hat(.): R^6 -> R^{4x4},  hat(a) = sum_i a_i * G_i``  (for i=0,...,5)
  ///
  /// with ``G_i`` being the ith infinitesimal generator of SE(3).
  ///
  /// The corresponding inverse is the vee()-operator, see below.
  ///
  [[nodiscard]] SOPHUS_FUNC static Transformation hat(Tangent const& a) {
    Transformation Omega;
    Omega << SO3<Scalar>::hat(a.template tail<3>()), a.template head<3>(),
        Matrix<Scalar, 1, 4>::Zero();
    return Omega;
  }

  /// Lie bracket
  ///
  /// It computes the Lie bracket of SE(3). To be more specific, it computes
  ///
  ///   ``[omega_1, omega_2]_se3 := vee([hat(omega_1), hat(omega_2)])``
  ///
  /// with ``[A,B] := AB-BA`` being the matrix commutator, ``hat(.)`` the
  /// hat()-operator and ``vee(.)`` the vee()-operator of SE(3).
  ///
  [[nodiscard]] SOPHUS_FUNC static Tangent lieBracket(Tangent const& a,
                                                      Tangent const& b) {
    Vector3<Scalar> const upsilon1 = a.template head<3>();
    Vector3<Scalar> const upsilon2 = b.template head<3>();
    Vector3<Scalar> const omega1 = a.template tail<3>();
    Vector3<Scalar> const omega2 = b.template tail<3>();

    Tangent res;
    res.template head<3>() = omega1.cross(upsilon2) + upsilon1.cross(omega2);
    res.template tail<3>() = omega1.cross(omega2);

    return res;
  }

  /// Construct x-axis rotation.
  ///
  [[nodiscard]] static SOPHUS_FUNC SE3 rotX(Scalar const& x) {
    return SE3(SO3<Scalar>::rotX(x), Sophus::Vector3<Scalar>::Zero());
  }

  /// Construct y-axis rotation.
  ///
  [[nodiscard]] static SOPHUS_FUNC SE3 rotY(Scalar const& y) {
    return SE3(SO3<Scalar>::rotY(y), Sophus::Vector3<Scalar>::Zero());
  }

  /// Construct z-axis rotation.
  ///
  [[nodiscard]] static SOPHUS_FUNC SE3 rotZ(Scalar const& z) {
    return SE3(SO3<Scalar>::rotZ(z), Sophus::Vector3<Scalar>::Zero());
  }

  /// Draw uniform sample from SE(3) manifold.
  ///
  /// Translations are drawn component-wise from the range [-1, 1].
  ///
  template <class UniformRandomBitGenerator>
  [[nodiscard]] static SE3 sampleUniform(UniformRandomBitGenerator& generator) {
    std::uniform_real_distribution<Scalar> uniform(Scalar(-1), Scalar(1));
    return SE3(SO3<Scalar>::sampleUniform(generator),
               Vector3<Scalar>(uniform(generator), uniform(generator),
                               uniform(generator)));
  }

  /// Construct a translation only SE3 instance.
  ///
  template <class T0, class T1, class T2>
  [[nodiscard]] static SOPHUS_FUNC SE3 trans(T0 const& x, T1 const& y,
                                             T2 const& z) {
    return SE3(SO3<Scalar>(), Vector3<Scalar>(x, y, z));
  }

  [[nodiscard]] static SOPHUS_FUNC SE3 trans(Vector3<Scalar> const& xyz) {
    return SE3(SO3<Scalar>(), xyz);
  }

  /// Construct x-axis translation.
  ///
  [[nodiscard]] static SOPHUS_FUNC SE3 transX(Scalar const& x) {
    return SE3::trans(x, Scalar(0), Scalar(0));
  }

  /// Construct y-axis translation.
  ///
  [[nodiscard]] static SOPHUS_FUNC SE3 transY(Scalar const& y) {
    return SE3::trans(Scalar(0), y, Scalar(0));
  }

  /// Construct z-axis translation.
  ///
  [[nodiscard]] static SOPHUS_FUNC SE3 transZ(Scalar const& z) {
    return SE3::trans(Scalar(0), Scalar(0), z);
  }

  /// vee-operator
  ///
  /// It takes 4x4-matrix representation ``Omega`` and maps it to the
  /// corresponding 6-vector representation of Lie algebra.
  ///
  /// This is the inverse of the hat()-operator, see above.
  ///
  /// Precondition: ``Omega`` must have the following structure:
  ///
  ///                |  0 -f  e  a |
  ///                |  f  0 -d  b |
  ///                | -e  d  0  c
  ///                |  0  0  0  0 | .
  ///
  [[nodiscard]] SOPHUS_FUNC static Tangent vee(Transformation const& Omega) {
    Tangent upsilon_omega;
    upsilon_omega << Omega.template rightCols<1>().template head<3>(),
        SO3<Scalar>::vee(Omega.template topLeftCorner<3, 3>());
    return upsilon_omega;
  }

 protected:
  SO3Member so3_;
  TranslationMember translation_;
};

template <class Scalar, int Options>
SOPHUS_FUNC SE3<Scalar, Options>::SE3()
    : translation_(TranslationMember::Zero()) {
  static_assert(std::is_standard_layout_v<SE3>,
                "Assume standard layout for the use of offsetof check below.");
  static_assert(
      offsetof(SE3, so3_) + sizeof(Scalar) * SO3<Scalar>::num_parameters ==
          offsetof(SE3, translation_),
      "This class assumes packed storage and hence will only work "
      "correctly depending on the compiler (options) - in "
      "particular when using [this->data(), this-data() + "
      "num_parameters] to access the raw data in a contiguous fashion.");
}
}  // namespace Sophus

namespace Eigen {

/// Specialization of Eigen::Map for ``SE3``; derived from SE3Base.
///
/// Allows us to wrap SE3 objects around POD array.
template <class Scalar_, int Options>
class Map<Sophus::SE3<Scalar_>, Options>
    : public Sophus::SE3Base<Map<Sophus::SE3<Scalar_>, Options>> {
 public:
  using Base = Sophus::SE3Base<Map<Sophus::SE3<Scalar_>, Options>>;
  using Scalar = Scalar_;
  using Transformation = typename Base::Transformation;
  using Point = typename Base::Point;
  using HomogeneousPoint = typename Base::HomogeneousPoint;
  using Tangent = typename Base::Tangent;
  using Adjoint = typename Base::Adjoint;

  using Base::operator=;
  using Base::operator*=;
  using Base::operator*;

  [[nodiscard]] SOPHUS_FUNC explicit Map(Scalar* coeffs)
      : so3_(coeffs),
        translation_(coeffs + Sophus::SO3<Scalar>::num_parameters) {}

  /// Mutator of SO3
  ///
  [[nodiscard]] SOPHUS_FUNC Map<Sophus::SO3<Scalar>, Options>& so3() {
    return so3_;
  }

  /// Accessor of SO3
  ///
  [[nodiscard]] SOPHUS_FUNC Map<Sophus::SO3<Scalar>, Options> const& so3()
      const {
    return so3_;
  }

  /// Mutator of translation vector
  ///
  [[nodiscard]] SOPHUS_FUNC Map<Sophus::Vector3<Scalar>, Options>&
  translation() {
    return translation_;
  }

  /// Accessor of translation vector
  ///
  [[nodiscard]] SOPHUS_FUNC Map<Sophus::Vector3<Scalar>, Options> const&
  translation() const {
    return translation_;
  }

 protected:
  Map<Sophus::SO3<Scalar>, Options> so3_;
  Map<Sophus::Vector3<Scalar>, Options> translation_;
};

/// Specialization of Eigen::Map for ``SE3 const``; derived from SE3Base.
///
/// Allows us to wrap SE3 objects around POD array.
template <class Scalar_, int Options>
class Map<Sophus::SE3<Scalar_> const, Options>
    : public Sophus::SE3Base<Map<Sophus::SE3<Scalar_> const, Options>> {
 public:
  using Base = Sophus::SE3Base<Map<Sophus::SE3<Scalar_> const, Options>>;
  using Scalar = Scalar_;
  using Transformation = typename Base::Transformation;
  using Point = typename Base::Point;
  using HomogeneousPoint = typename Base::HomogeneousPoint;
  using Tangent = typename Base::Tangent;
  using Adjoint = typename Base::Adjoint;

  using Base::operator*=;
  using Base::operator*;

  [[nodiscard]] SOPHUS_FUNC explicit Map(Scalar const* coeffs)
      : so3_(coeffs),
        translation_(coeffs + Sophus::SO3<Scalar>::num_parameters) {}

  /// Accessor of SO3
  ///
  [[nodiscard]] SOPHUS_FUNC Map<Sophus::SO3<Scalar> const, Options> const& so3()
      const {
    return so3_;
  }

  /// Accessor of translation vector
  ///
  [[nodiscard]] SOPHUS_FUNC Map<Sophus::Vector3<Scalar> const, Options> const&
  translation() const {
    return translation_;
  }

 protected:
  Map<Sophus::SO3<Scalar> const, Options> const so3_;
  Map<Sophus::Vector3<Scalar> const, Options> const translation_;
};
}  // namespace Eigen

#endif
