cmake_minimum_required(VERSION 3.22.1)
set(SRC_DIR /home/bradenlock83/Projects/NanoGrad/nano_grad/csrc)
project(nano_grad_backend)

# Create the object file
add_library(${PROJECT_NAME}_obj OBJECT ${SRC_DIR}/main.cpp)
set_target_properties(${PROJECT_NAME}_obj PROPERTIES POSITION_INDEPENDENT_CODE ON)  # Add this line

# Create the executable
add_executable(${PROJECT_NAME} $<TARGET_OBJECTS:${PROJECT_NAME}_obj>)

# Create the shared library (.so) using the object file
add_library(${PROJECT_NAME}_shared SHARED $<TARGET_OBJECTS:${PROJECT_NAME}_obj>)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

add_subdirectory(${SRC_DIR}/Autograd autograd_build)
add_subdirectory(${SRC_DIR}/Dispatcher dispatcher_build)
add_subdirectory(${SRC_DIR}/Tensor tensor_build)

target_link_libraries(${PROJECT_NAME} autograd dispatcher python_tensor)


target_link_directories(${PROJECT_NAME} PUBLIC ${SRC_DIR}/Autograd ${SRC_DIR}/Dispatcher ${SRC_DIR}/Tensor)


# Add the Python include directories for the object file and executable targets
target_include_directories(${PROJECT_NAME}_obj PRIVATE ${Python3_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} PRIVATE ${Python3_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${Python3_LIBRARIES})
