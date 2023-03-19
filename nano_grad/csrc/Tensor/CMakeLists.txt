find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

add_library(python_tensor SHARED python_tensor.cpp python_tensor.h)

target_include_directories(python_tensor PRIVATE ${Python3_INCLUDE_DIRS})
target_link_libraries(python_tensor ${Python3_LIBRARIES})