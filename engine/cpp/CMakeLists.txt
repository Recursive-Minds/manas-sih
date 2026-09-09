cmake_minimum_required(VERSION 3.14)
project(idr_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags
if(MSVC)
  add_compile_options(/O2 /W4)
else()
  add_compile_options(-O3 -Wall -Wextra -fPIC)
endif()

# Include directories
include_directories(include)

# Find optional Eigen3 package
find_package(Eigen3 3.3 QUIET NO_MODULE)
if(TARGET Eigen3::Eigen)
  message(STATUS "Eigen3 found. Linking Eigen3::Eigen.")
else()
  message(STATUS "Eigen3 not found. Using standalone optimized fallback matrix operations.")
endif()

# Shared library
add_library(idr_core SHARED src/idr_core.cpp)

if(TARGET Eigen3::Eigen)
  target_link_libraries(idr_core PRIVATE Eigen3::Eigen)
endif()

target_include_directories(idr_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

set_target_properties(idr_core PROPERTIES
  VERSION 1.0.0
  SOVERSION 1
  OUTPUT_NAME "idr_core"
)
