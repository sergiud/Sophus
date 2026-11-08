#!/bin/bash
# Copyright 2026 Sergiu Deitsch <sergiu.deitsch@gmail.com>
# Builds the HTML documentation with Doxygen, doxyrest and Sphinx. doxyrest
# is expected as a build of its source in doxyrest_b/.
set -euo pipefail
cd "$(dirname "$0")"
doxygen doxyfile
doxyrest_b/build/doxyrest/bin/Release/doxyrest -c doxyrest-config.lua
sphinx-build -b html rst-dir html-dir
