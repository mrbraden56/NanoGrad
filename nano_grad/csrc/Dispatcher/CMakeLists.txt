cmake_minimum_required(VERSION 3.22.1)

add_library(dispatcher dispatcher.cpp dispatcher.h)
set_target_properties(dispatcher PROPERTIES POSITION_INDEPENDENT_CODE ON)
# target_link_libraries(dispatcher python_tensor)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
target_include_directories(dispatcher PRIVATE ${Python3_INCLUDE_DIRS})
target_link_libraries(dispatcher ${Python3_LIBRARIES})

find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)
target_include_directories(dispatcher PRIVATE ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(dispatcher ${Python3_NumPy_LIBRARIES})