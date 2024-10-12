cmake_minimum_required(VERSION 3.25)
project(_lib LANGUAGES CXX)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(SKBUILD)
  set(LIBRARY_OUTPUT_PATH ${SKBUILD_PLATLIB_DIR}/coreforecast)
else()
  set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/python/coreforecast)
endif()

find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
set(PYBIND11_FINDPYTHON ON)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE pybind11_DIR)
find_package(pybind11 CONFIG REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m pybind11 --includes
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE pybind11_INCLUDES)

include_directories(include/coreforecast external_libs/skiplist/src/cpp external_libs/stl-cpp/include ${pybind11_INCLUDES})
file(GLOB SOURCES src/*.cpp external_libs/skiplist/src/cpp/SkipList.cpp)
pybind11_add_module(_lib ${SOURCES})

if(MSVC)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE "${OUTPUT_DIRECTORY}")
endif()

install(TARGETS _lib DESTINATION .)
