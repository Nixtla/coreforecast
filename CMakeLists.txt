cmake_minimum_required(VERSION 3.25)
project(ga LANGUAGES CXX)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
set(PYBIND11_FINDPYTHON ON)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE pybind11_DIR)
find_package(pybind11 CONFIG REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m pybind11 --includes
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE pybind11_INCLUDES)

include_directories(include external/skiplist/src/cpp external/stl-cpp/include ${pybind11_INCLUDES})
pybind11_add_module(ga src/coreforecast.cpp external/skiplist/src/cpp/SkipList.cpp)
install(TARGETS ga DESTINATION .)
