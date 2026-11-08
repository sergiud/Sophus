#!/bin/bash
# Copyright 2026 Sergiu Deitsch <sergiu.deitsch@gmail.com>
# Formats the tracked C++ sources outside the generated code in sympy/.
set -euo pipefail
cd "$(dirname "$0")"
git ls-files -z -- '*.hpp' '*.cpp' ':!:sympy/' | xargs -0 clang-format -i
