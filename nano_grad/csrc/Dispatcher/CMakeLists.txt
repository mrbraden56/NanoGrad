find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

add_library(dispatcher SHARED dispatcher.cpp dispatcher.h)

target_include_directories(dispatcher PRIVATE ${Python3_INCLUDE_DIRS})
target_link_libraries(dispatcher ${Python3_LIBRARIES})