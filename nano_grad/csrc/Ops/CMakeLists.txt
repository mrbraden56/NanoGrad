cmake_minimum_required(VERSION 3.22.1)

add_library(ops ops.cpp ops.h)
set_target_properties(ops PROPERTIES POSITION_INDEPENDENT_CODE ON)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
target_include_directories(ops PRIVATE ${Python3_INCLUDE_DIRS})
target_link_libraries(ops ${Python3_LIBRARIES})