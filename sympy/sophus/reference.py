# Copyright 2026 Sergiu Deitsch <sergiu.deitsch@gmail.com>

"""High-precision reference values of the Lie group functions.

The values come from the matrix exponential of the Lie algebra element,
evaluated with mpmath at 60 significant digits, independently of the closed
forms of the C++ headers and of the derivations in this package. Jacobians
are central differences at that precision, left Jacobians the series
sum_k ad^k / (k + 1)! and its inverse sum_k B_k ad^k / k!.

Every input is a single-precision number, so that float and double tests
evaluate the same point. Running

    python3 -m sophus.reference ../test/core

in the sympy directory writes one header per group into test/core.
"""

import struct
import sys
from pathlib import Path

import mpmath as mp

DIGITS = 60
OUTPUT_DIGITS = 25

DIM = {"SO2": 2, "RxSO2": 2, "SE2": 2, "Sim2": 2,
       "SO3": 3, "RxSO3": 3, "SE3": 3, "Sim3": 3}
DOF = {"SO2": 1, "RxSO2": 2, "SE2": 3, "Sim2": 4,
       "SO3": 3, "RxSO3": 4, "SE3": 6, "Sim3": 7}
NUM_PARAMETERS = {"SO2": 2, "RxSO2": 2, "SE2": 4, "Sim2": 4,
                  "SO3": 4, "RxSO3": 4, "SE3": 7, "Sim3": 7}
TRANSLATING = ("SE2", "SE3", "Sim2", "Sim3")
SCALING = ("RxSO2", "RxSO3", "Sim2", "Sim3")

ANGLES = ["0", "1e-12", "1e-9", "1e-7", "1e-5", "1e-3", "0.1", "1", "3"]
SIGMAS = ["0", "1e-9", "1e-5", "0.2", "-1"]
# A unit vector with components exactly representable in binary.
DIRECTION = ["0.48", "-0.6", "0.64"]
UPSILON = ["0.5", "-1.25", "2"]


def to_float32(value):
    """Rounds a number to the nearest single-precision number."""
    return mp.mpf(struct.unpack("f", struct.pack("f", float(value)))[0])


def split(group, x):
    """Splits a tangent into translation, rotation and log scale."""
    n = DIM[group]
    rotation_dof = 1 if n == 2 else 3
    translation = x[:n] if group in TRANSLATING else []
    offset = len(translation)
    rotation = x[offset:offset + rotation_dof]
    sigma = x[offset + rotation_dof] if group in SCALING else None
    return translation, rotation, sigma


def join(group, translation, rotation, sigma):
    return translation + rotation + ([sigma] if group in SCALING else [])


def hat(group, x):
    """The (n+1)x(n+1) matrix of a Lie algebra element."""
    n = DIM[group]
    translation, w, sigma = split(group, x)
    if n == 2:
        rotation = mp.matrix([[0, -w[0]], [w[0], 0]])
    else:
        rotation = mp.matrix([[0, -w[2], w[1]], [w[2], 0, -w[0]],
                              [-w[1], w[0], 0]])
    if sigma is not None:
        rotation += sigma * mp.eye(n)
    result = mp.zeros(n + 1, n + 1)
    result[0:n, 0:n] = rotation
    for row, value in enumerate(translation):
        result[row, n] = value
    return result


def vee(group, matrix):
    n = DIM[group]
    if n == 2:
        rotation = [(matrix[1, 0] - matrix[0, 1]) / 2]
    else:
        rotation = [(matrix[2, 1] - matrix[1, 2]) / 2,
                    (matrix[0, 2] - matrix[2, 0]) / 2,
                    (matrix[1, 0] - matrix[0, 1]) / 2]
    sigma = sum(matrix[i, i] for i in range(n)) / n
    translation = [matrix[row, n] for row in range(n)]
    return join(group, translation if group in TRANSLATING else [], rotation,
                sigma)


def quaternion_from_rotation(rotation):
    """Unit quaternion (x, y, z, w) with w >= 0 of a rotation matrix."""
    r = rotation
    trace = r[0, 0] + r[1, 1] + r[2, 2]
    largest = max(range(4), key=lambda i: [trace, r[0, 0], r[1, 1],
                                           r[2, 2]][i])
    if largest == 0:
        w = mp.sqrt(1 + trace) / 2
        x = (r[2, 1] - r[1, 2]) / (4 * w)
        y = (r[0, 2] - r[2, 0]) / (4 * w)
        z = (r[1, 0] - r[0, 1]) / (4 * w)
    elif largest == 1:
        x = mp.sqrt(1 + 2 * r[0, 0] - trace) / 2
        w = (r[2, 1] - r[1, 2]) / (4 * x)
        y = (r[0, 1] + r[1, 0]) / (4 * x)
        z = (r[0, 2] + r[2, 0]) / (4 * x)
    elif largest == 2:
        y = mp.sqrt(1 + 2 * r[1, 1] - trace) / 2
        w = (r[0, 2] - r[2, 0]) / (4 * y)
        x = (r[0, 1] + r[1, 0]) / (4 * y)
        z = (r[1, 2] + r[2, 1]) / (4 * y)
    else:
        z = mp.sqrt(1 + 2 * r[2, 2] - trace) / 2
        w = (r[1, 0] - r[0, 1]) / (4 * z)
        x = (r[0, 2] + r[2, 0]) / (4 * z)
        y = (r[1, 2] + r[2, 1]) / (4 * z)
    sign = -1 if w < 0 else 1
    return [sign * x, sign * y, sign * z, sign * w]


def params_from_matrix(group, matrix):
    """The parameters in the layout of the C++ classes."""
    n = DIM[group]
    linear = matrix[0:n, 0:n]
    if n == 2:
        params = [linear[0, 0], linear[1, 0]]
    else:
        scale = mp.cbrt(mp.det(linear))
        params = quaternion_from_rotation(linear / scale)
        if group in SCALING:
            params = [mp.sqrt(scale) * c for c in params]
    if group in TRANSLATING:
        params += [matrix[row, n] for row in range(n)]
    return params


def exp(group, x):
    return params_from_matrix(group, mp.expm(hat(group, x)))


def log(group, params):
    """Logarithm of possibly unnormalized parameters. The angle comes from
    atan2 and the translation from W^-1 t, where W = int_0^1 exp(s X) ds is
    the upper right block of a block exponential, avoiding the branch
    selection of a matrix logarithm near pi."""
    n = DIM[group]
    if n == 2:
        rotation = [mp.atan2(params[1], params[0])]
        sigma = mp.log(mp.hypot(params[0], params[1]))
        translation = params[2:4]
    else:
        vector_norm = mp.sqrt(sum(c * c for c in params[0:3]))
        norm = mp.sqrt(vector_norm ** 2 + params[3] ** 2)
        angle = 2 * mp.atan2(vector_norm, params[3])
        if angle > mp.pi:
            angle -= 2 * mp.pi
        rotation = ([angle * c / vector_norm for c in params[0:3]]
                    if vector_norm != 0 else [mp.mpf(0)] * 3)
        sigma = 2 * mp.log(norm)
        translation = params[4:7]
    if group not in TRANSLATING:
        return join(group, [], rotation, sigma)
    linear = hat(group, join(group, [0] * n, rotation, sigma))[0:n, 0:n]
    block = mp.zeros(2 * n, 2 * n)
    block[0:n, 0:n] = linear
    block[0:n, n:2 * n] = mp.eye(n)
    w = mp.expm(block)[0:n, n:2 * n]
    upsilon = mp.lu_solve(w, mp.matrix(translation))
    return join(group, [upsilon[i] for i in range(n)], rotation, sigma)


def jacobian(function, x, rows):
    step = mp.mpf(10) ** (-(DIGITS // 2 - 5))
    result = mp.zeros(rows, len(x))
    for col in range(len(x)):
        forward = list(x)
        backward = list(x)
        forward[col] += step
        backward[col] -= step
        f = function(forward)
        b = function(backward)
        for row in range(rows):
            result[row, col] = (f[row] - b[row]) / (2 * step)
    return result


def adjoint(group, x):
    """The matrix of ad_x on the tangent space."""
    dof = DOF[group]
    matrix = mp.zeros(dof, dof)
    left = hat(group, x)
    for col in range(dof):
        unit = [0] * dof
        unit[col] = 1
        right = hat(group, unit)
        bracket = vee(group, left * right - right * left)
        for row in range(dof):
            matrix[row, col] = bracket[row]
    return matrix


def left_jacobian(group, x, inverse=False):
    ad = adjoint(group, x)
    result = mp.zeros(DOF[group], DOF[group])
    power = mp.eye(DOF[group])
    for k in range(400):
        if inverse:
            coefficient = mp.bernoulli(k) / mp.factorial(k)
            if k == 1:
                coefficient = mp.mpf(-1) / 2
        else:
            coefficient = 1 / mp.factorial(k + 1)
        result += coefficient * power
        power = power * ad
    return result


def tangents(group):
    n = DIM[group]
    result = []
    sigmas = SIGMAS if group in SCALING else [None]
    for angle in ANGLES:
        for sigma in sigmas:
            if n == 2:
                rotation = [mp.mpf(angle)]
            else:
                rotation = [mp.mpf(angle) * mp.mpf(c) for c in DIRECTION]
            translation = ([mp.mpf(c) for c in UPSILON[:n]]
                           if group in TRANSLATING else [])
            x = join(group, translation, rotation,
                     mp.mpf(sigma) if sigma is not None else None)
            result.append([to_float32(c) for c in x])
    return result


def flatten(matrix):
    if isinstance(matrix, list):
        return matrix
    return [matrix[r, c] for r in range(matrix.rows)
            for c in range(matrix.cols)]


def exact(value):
    """A hexadecimal literal of a binary floating-point number, which
    represents it exactly."""
    return float(value).hex() + "L"


def rounded(value):
    return mp.nstr(value, OUTPUT_DIGITS) + "L"


def table(name, cases):
    lines = [f"inline std::vector<Case> const {name} = {{"]
    for inputs, outputs in cases:
        lines.append("    {{" + ", ".join(exact(v) for v in inputs) + "},")
        lines.append("     {" + ", ".join(rounded(v) for v in outputs) +
                     "}},")
    lines.append("};")
    return "\n".join(lines)


def header(group):
    xs = tangents(group)
    tables = []
    tables.append(table(f"k{group}Exp", [(x, exp(group, x)) for x in xs]))
    tables.append(table(f"k{group}DxExpX", [
        (x, flatten(jacobian(lambda v, g=group: exp(g, v), x,
                             NUM_PARAMETERS[group]))) for x in xs]))
    logs = []
    for x in xs:
        params = [to_float32(c) for c in exp(group, x)]
        logs.append((params, log(group, params)))
    tables.append(table(f"k{group}Log", logs))
    if group in ("SO3", "SE3"):
        tables.append(table(f"k{group}LeftJacobian", [
            (x, flatten(left_jacobian(group, x))) for x in xs]))
        tables.append(table(f"k{group}LeftJacobianInverse", [
            (x, flatten(left_jacobian(group, x, inverse=True)))
            for x in xs]))
    return "\n".join([
        "// Copyright 2026 Sergiu Deitsch <sergiu.deitsch@gmail.com>",
        "//",
        "// Generated by `python3 -m sophus.reference ../test/core` in the",
        "// sympy directory. Do not edit.",
        "",
        "#pragma once",
        "",
        "#include <vector>",
        "",
        '#include "reference.hpp"',
        "",
        "namespace Sophus {",
        "namespace reference {",
        "",
        "\n\n".join(tables),
        "",
        "}  // namespace reference",
        "}  // namespace Sophus",
        ""])


def main(directory):
    mp.mp.dps = DIGITS
    for group in DIM:
        path = Path(directory) / f"reference_{group.lower()}.hpp"
        path.write_text(header(group))


if __name__ == "__main__":
    main(sys.argv[1])
