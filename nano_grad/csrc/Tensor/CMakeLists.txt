cmake_minimum_required(VERSION 3.22.1)

add_library(tensor tensor.cpp tensor.h)
set_target_properties(tensor PROPERTIES POSITION_INDEPENDENT_CODE ON)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
target_include_directories(tensor PRIVATE ${Python3_INCLUDE_DIRS})
target_link_libraries(tensor ${Python3_LIBRARIES})