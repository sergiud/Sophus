// Copyright 2021, 2026 Sergiu Deitsch <sergiu.deitsch@gmail.com>

#pragma once

#include "types.hpp"

namespace Sophus {
namespace details {

/// Below this value of @f$\theta^2 + \sigma^2@f$, W comes from its Taylor
/// series, since the closed forms of its coefficients cancel when the angle
/// and the log scale are both small.
constexpr double kWSeriesBound = 1;

/// The number of terms of the Taylor series of W. For
/// @f$\theta^2 + \sigma^2 < 1@f$, the first omitted term is below
/// @f$10^{-17}@f$.
constexpr int kWSeriesTerms = 18;

/**
 * Returns the Taylor series
 * @f[
 *   W = \sum_{k=0}^{n-1} \frac{X^k}{(k + 1)!}, \quad X = \sigma I + \Omega
 * @f]
 * with @f$n@f$ = kWSeriesTerms, evaluated by Horner's scheme. Since X is a
 * normal matrix with norm @f$\sqrt{\theta^2 + \sigma^2}@f$, the series has no
 * cancellation below kWSeriesBound.
 */
template <class Scalar, int N>
Matrix<Scalar, N, N> calcWSeries(Matrix<Scalar, N, N> const &Omega,
                                 Scalar const sigma) {
  Matrix<Scalar, N, N> const I = Matrix<Scalar, N, N>::Identity();
  Matrix<Scalar, N, N> const X = Omega + sigma * I;
  Matrix<Scalar, N, N> W = I;
  for (int k = kWSeriesTerms - 1; k >= 1; --k) {
    W = I + X * W / Scalar(k + 1);
  }
  return W;
}

template <class Scalar, int N>
Matrix<Scalar, N, N> calcW(Matrix<Scalar, N, N> const &Omega,
                           Scalar const theta, Scalar const sigma) {
  using std::cos;
  using std::exp;
  using std::expm1;
  using std::fpclassify;
  using std::sin;
  static Scalar const one(1);
  Scalar const theta_sq = theta * theta;
  Scalar const sigma_sq = sigma * sigma;
  if (theta_sq + sigma_sq < Scalar(kWSeriesBound)) {
    return calcWSeries(Omega, sigma);
  }

  Matrix<Scalar, N, N> const Omega2 = Omega * Omega;
  Scalar const scale = exp(sigma);
  // C = (exp(sigma) - 1) / sigma. Computing the numerator via expm1 avoids
  // the cancellation that exp(sigma) - 1 would incur for small sigma, and
  // expm1(sigma) / sigma is itself well-conditioned for any non-zero sigma
  // (it tends to 1 continuously), so the only case that needs to be
  // special-cased is the removable singularity at sigma == 0 itself.
  Scalar const C = (fpclassify(sigma) == FP_ZERO) ? one : expm1(sigma) / sigma;

  // Outside the series, theta or sigma is at least 1/sqrt(2). The closed
  // forms below then only cancel for small theta, where the error of A and B
  // is weighted by the powers of Omega they multiply.
  Scalar A;
  Scalar B;
  if (theta_sq < Constants<Scalar>::epsilon()) {
    A = ((sigma - one) * scale + one) / sigma_sq;
    B = (scale * sigma_sq / Scalar(2) + scale - one - sigma * scale) /
        (sigma_sq * sigma);
  } else {
    Scalar const a = scale * sin(theta);
    Scalar const b = scale * cos(theta);
    Scalar const c = theta_sq + sigma_sq;
    A = (a * sigma + (one - b) * theta) / (theta * c);
    B = (C - ((b - one) * sigma + a * theta) / c) / theta_sq;
  }
  return A * Omega + B * Omega2 + C * Matrix<Scalar, N, N>::Identity();
}

/**
 * Returns @f$\varphi_1(M) = \sum_k M^k / (k + 1)!@f$ of a square matrix.
 *
 * The series is evaluated for @f$Y = M / 2^s@f$ with a 1-norm below 1/2,
 * then doubled s times with
 * @f$\varphi_1(2Y) = \tfrac{1}{2}(e^Y + I)\varphi_1(Y)@f$ and
 * @f$e^{2Y} = (e^Y)^2@f$, where @f$e^Y = I + Y\varphi_1(Y)@f$. This keeps the
 * series free of cancellation for any norm of M.
 */
template <class Scalar, int N>
Matrix<Scalar, N, N> phi1(Matrix<Scalar, N, N> const &M) {
  using std::abs;
  Matrix<Scalar, N, N> const I = Matrix<Scalar, N, N>::Identity();
  Scalar norm(0);
  for (int col = 0; col < N; ++col) {
    Scalar column_sum(0);
    for (int row = 0; row < N; ++row) {
      column_sum += abs(M(row, col));
    }
    if (column_sum > norm) {
      norm = column_sum;
    }
  }
  Matrix<Scalar, N, N> Y = M;
  int doublings = 0;
  while (norm > Scalar(0.5)) {
    norm /= Scalar(2);
    Y /= Scalar(2);
    ++doublings;
  }
  Matrix<Scalar, N, N> P = I;
  for (int k = kWSeriesTerms - 1; k >= 1; --k) {
    P = I + Y * P / Scalar(k + 1);
  }
  Matrix<Scalar, N, N> E = I + Y * P;
  for (int i = 0; i < doublings; ++i) {
    P = (E + I) * P / Scalar(2);
    E = E * E;
  }
  return P;
}

/**
 * Returns the derivative of W = @f$\varphi_1(\sigma I + \Omega)@f$ in the
 * direction @f$\Delta@f$ of @f$\sigma I + \Omega@f$.
 *
 * For an analytic function f, the upper right block of
 * @f$f\left(\begin{smallmatrix} X & \Delta \\ 0 & X\end{smallmatrix}
 * \right)@f$ is the Frechet derivative of f at X in the direction
 * @f$\Delta@f$, which avoids differentiating the closed forms of W.
 */
template <class Scalar, int N>
Matrix<Scalar, N, N> calcWDerivative(Matrix<Scalar, N, N> const &Omega,
                                     Scalar const sigma,
                                     Matrix<Scalar, N, N> const &Delta) {
  Matrix<Scalar, 2 * N, 2 * N> block = Matrix<Scalar, 2 * N, 2 * N>::Zero();
  Matrix<Scalar, N, N> const X =
      Omega + sigma * Matrix<Scalar, N, N>::Identity();
  block.template topLeftCorner<N, N>() = X;
  block.template bottomRightCorner<N, N>() = X;
  block.template topRightCorner<N, N>() = Delta;
  return phi1(block).template topRightCorner<N, N>();
}

template <class Scalar, int N>
Matrix<Scalar, N, N> calcWInv(Matrix<Scalar, N, N> const &Omega,
                              Scalar const theta, Scalar const sigma,
                              Scalar const scale) {
  using std::cos;
  using std::expm1;
  using std::fpclassify;
  using std::sin;
  using std::sinh;
  static Scalar const one(1);
  static Scalar const two(2);
  Scalar const theta_sq = theta * theta;
  if (theta_sq + sigma * sigma < Scalar(kWSeriesBound)) {
    // W is close to the identity there, so its inverse is accurate.
    return calcWSeries(Omega, sigma).inverse();
  }

  Matrix<Scalar, N, N> const Omega2 = Omega * Omega;
  Scalar const scale_sq = scale * scale;
  Scalar const sin_theta = sin(theta);
  Scalar const cos_theta = cos(theta);

  // c = sigma / (scale - 1) = sigma / expm1(sigma). expm1(sigma) is
  // well-conditioned even for small sigma (unlike scale - 1, which loses
  // precision the same way exp(sigma) - 1 does), and sigma / expm1(sigma)
  // is itself well-conditioned for any non-zero sigma (it tends to 1
  // continuously), so, as with C in calcW() above, only the removable
  // singularity at sigma == 0 needs to be handled explicitly.
  Scalar const expm1_sigma = expm1(sigma);
  Scalar const c = (fpclassify(sigma) == FP_ZERO) ? one : sigma / expm1_sigma;

  // Outside the series, theta or sigma is at least 1/sqrt(2), so the
  // branch for small theta has sigma bounded away from zero.
  Scalar a;
  Scalar b;
  Scalar const scale_cu = scale_sq * scale;
  if (theta_sq < Constants<Scalar>::epsilon()) {
    // Rewritten in terms of expm1(sigma) rather than scale - 1 directly,
    // for the same reason c above uses expm1(sigma).
    a = (expm1_sigma - sigma * scale) / (expm1_sigma * expm1_sigma);
    b = (scale_sq * sigma - two * scale_sq + scale * sigma + two * scale) /
        (two * scale_cu - Scalar(6) * scale_sq + Scalar(6) * scale - two);
  } else {
    // The direct formulas
    //
    //   a = (theta*s_cos_theta - theta - sigma*s_sin_theta)
    //       / (theta*(scale^2 - 2*s_cos_theta + 1))
    //   b = -scale*(theta*s_sin_theta - theta*sin(theta)
    //               + sigma*s_cos_theta - scale*sigma + sigma*cos(theta)
    //               - sigma)
    //       / (theta^2*(scale^3 - 2*scale*s_cos_theta - scale^2
    //                   + 2*s_cos_theta + scale - 1))
    //
    // (with s_sin_theta = scale*sin(theta), s_cos_theta = scale*
    // cos(theta)) lose precision when theta or sigma is small: both
    // denominators reduce to
    // theta^2*(scale - 1)*(scale^2 - 2*scale*cos(theta) + 1), and each
    // factor of that, as well as the numerators above, is a
    // near-cancelling difference of two order-1 quantities.
    //
    // scale^2 - 2*scale*cos(theta) + 1 ==
    //     4*scale*(sinh(sigma/2)^2 + sin(theta/2)^2)
    // exactly (verified in sympy/sophus/sim_details.py): a sum of two
    // non-negative, individually well-conditioned squares, avoiding
    // that cancellation entirely. The numerators below are likewise
    // rewritten in terms of expm1(sigma) and half-angle sin/cos, and the
    // factor expm1(sigma) of the numerator and the denominator of b
    // cancels into c, which keeps b finite at sigma == 0.
    Scalar const half_theta = theta / two;
    Scalar const sin_half_theta = sin(half_theta);
    Scalar const cos_half_theta = cos(half_theta);
    Scalar const half_sigma = sigma / two;
    Scalar const sinh_half_sigma = sinh(half_sigma);
    Scalar const D =
        Scalar(4) * scale *
        (sinh_half_sigma * sinh_half_sigma +
         sin_half_theta * sin_half_theta);  // == scale^2 -
                                            // 2*scale*cos(theta) + 1

    a = (two * half_theta * expm1_sigma * cos_theta -
         Scalar(4) * half_theta * sin_half_theta * sin_half_theta -
         Scalar(4) * half_sigma * scale * sin_half_theta * cos_half_theta) /
        (theta * D);
    b = -scale *
        (theta * sin_theta -
         two * c * sin_half_theta * sin_half_theta * (scale + one)) /
        (theta_sq * D);
  }
  return a * Omega + b * Omega2 + c * Matrix<Scalar, N, N>::Identity();
}

}  // namespace details
}  // namespace Sophus
