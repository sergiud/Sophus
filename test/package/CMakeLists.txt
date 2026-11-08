# Copyright 2026 Sergiu Deitsch <sergiu.deitsch@gmail.com>
#
# A project that consumes the installed Sophus package, built by the test
# package_consumer of the top-level project.

cmake_minimum_required (VERSION 3.28...4.4)
project (SophusPackageConsumer LANGUAGES CXX)

find_package (Sophus 2.0 REQUIRED CONFIG)

# Every header of the source tree must be installed, including those the
# consumer below does not include.
if (NOT IS_DIRECTORY "${SOPHUS_SOURCE_DIR}/sophus")
  message (FATAL_ERROR
    "SOPHUS_SOURCE_DIR is '${SOPHUS_SOURCE_DIR}', but must be the source "
    "directory of Sophus")
endif (NOT IS_DIRECTORY "${SOPHUS_SOURCE_DIR}/sophus")

file (GLOB SOPHUS_SOURCE_HEADERS
  RELATIVE ${SOPHUS_SOURCE_DIR}
  ${SOPHUS_SOURCE_DIR}/sophus/*.hpp
)
get_target_property (SOPHUS_HEADER_DIRS Sophus::Sophus HEADER_DIRS)

if (NOT SOPHUS_SOURCE_HEADERS OR NOT SOPHUS_HEADER_DIRS)
  message (FATAL_ERROR
    "Found the source headers '${SOPHUS_SOURCE_HEADERS}' and the installed "
    "header directories '${SOPHUS_HEADER_DIRS}', but expected both to be "
    "non-empty")
endif (NOT SOPHUS_SOURCE_HEADERS OR NOT SOPHUS_HEADER_DIRS)

foreach (header IN LISTS SOPHUS_SOURCE_HEADERS)
  set (header_found OFF)

  foreach (header_dir IN LISTS SOPHUS_HEADER_DIRS)
    if (EXISTS ${header_dir}/${header})
      set (header_found ON)
    endif (EXISTS ${header_dir}/${header})
  endforeach (header_dir)

  if (NOT header_found)
    message (FATAL_ERROR
      "${header} is missing from the installed header directories "
      "'${SOPHUS_HEADER_DIRS}'")
  endif (NOT header_found)
endforeach (header)

add_executable (consumer main.cpp)
target_link_libraries (consumer PRIVATE Sophus::Sophus)
