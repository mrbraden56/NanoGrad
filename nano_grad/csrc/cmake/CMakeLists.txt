cmake_minimum_required(VERSION 3.22.1)
set(SRC_DIR ../nano_grad)
project(nano_grad)
add_library(${PROJECT_NAME} SHARED ${SRC_DIR}/engine/tester.cpp SUFFIX .so)