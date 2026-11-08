|GithubCICpp|_ |GithubCISympy|_ |ci_cov|_


Sophus
======

Overview
--------

This is a C++ implementation of Lie groups commonly used for 2D and 3D
geometric problems, e.g., for Computer Vision or Robotics applications.
Among others, this package includes the special orthogonal groups SO(2) and
SO(3) to represent rotations in 2D and 3D as well as the special Euclidean
groups SE(2) and SE(3) to represent rigid body transformations, i.e., rotations
and translations, in 2D and 3D.

Cross platform support
----------------------

Sophus compiles with GCC and Clang on Linux, AppleClang on macOS, and MSVC and
MinGW-w64 on Windows. The specific compiler and operating system versions which
are supported are the ones which are used in the Continuous Integration (CI):
See GitHubCI_ for details.

However, it should work with no to minor modifications on many other modern
configurations as long as they support C++17, CMake 3.28 or later, Eigen 3.4.1
to 5 and fmt.
Building the tests additionally requires GoogleTest 1.10 or later.

.. _GitHubCI: https://github.com/sergiud/Sophus/actions

.. |ci_cov| image:: https://codecov.io/gh/sergiud/Sophus/branch/master/graph/badge.svg
.. _ci_cov: https://codecov.io/gh/sergiud/Sophus

.. |GithubCICpp| image:: https://github.com/sergiud/Sophus/actions/workflows/ci.yml/badge.svg?branch=master
.. _GithubCICpp: https://github.com/sergiud/Sophus/actions/workflows/ci.yml?query=branch%3Amaster

.. |GithubCISympy| image:: https://github.com/sergiud/Sophus/actions/workflows/sympy.yml/badge.svg?branch=master
.. _GithubCISympy: https://github.com/sergiud/Sophus/actions/workflows/sympy.yml?query=branch%3Amaster
