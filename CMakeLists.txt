cmake_minimum_required(VERSION 3.25)
project(coreforecast)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-fPIC -Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

if(SKBUILD)
    set(LIBRARY_OUTPUT_PATH ${SKBUILD_PLATLIB_DIR}/coreforecast/lib)
else()
    set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/coreforecast/lib)
endif()

include_directories(include)
add_library(coreforecast SHARED src/coreforecast.cpp)
