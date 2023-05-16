cmake_minimum_required(VERSION 3.22.1)

add_library(network network.cpp network.h)
set_target_properties(network PROPERTIES POSITION_INDEPENDENT_CODE ON)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
target_include_directories(network PRIVATE ${Python3_INCLUDE_DIRS})
target_link_libraries(network ${Python3_LIBRARIES})