cmake_minimum_required(VERSION 3.25)
project(coreforecast LANGUAGES CXX)

include(FetchContent)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR
)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

include_directories(include)
FetchContent_Declare(
    stl-cpp
    GIT_REPOSITORY https://github.com/jmoralez/stl-cpp.git
    GIT_TAG 13d26c0d0653ddcdbf853de3f92f56faa831a330
)
FetchContent_GetProperties(stl-cpp)

if(NOT stl-cpp_POPULATED)
    FetchContent_Populate(stl-cpp)
    include_directories(${stl-cpp_SOURCE_DIR}/include)
endif()

FetchContent_Declare(
    skiplist
    GIT_REPOSITORY https://github.com/paulross/skiplist.git
    GIT_TAG aace571a8564067820ff7a5e34ba68d0a9782153
)
FetchContent_GetProperties(skiplist)

if(NOT skiplist_POPULATED)
    FetchContent_Populate(skiplist)
    include_directories(${skiplist_SOURCE_DIR}/src/cpp)
endif()

file(GLOB SOURCES src/*.cpp ${skiplist_SOURCE_DIR}/src/cpp/SkipList.cpp)
nanobind_add_module(_lib ${SOURCES})
install(TARGETS _lib LIBRARY DESTINATION coreforecast)
