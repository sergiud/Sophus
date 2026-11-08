# Copyright 2026 Sergiu Deitsch <sergiu.deitsch@gmail.com>
#
# A project that consumes the installed Sophus package, built by the test
# package_consumer of the top-level project.

cmake_minimum_required (VERSION 3.28...4.4)
project (SophusPackageConsumer LANGUAGES CXX)

find_package (Sophus 2.0 REQUIRED CONFIG)

add_executable (consumer main.cpp)
target_link_libraries (consumer PRIVATE Sophus::Sophus)
