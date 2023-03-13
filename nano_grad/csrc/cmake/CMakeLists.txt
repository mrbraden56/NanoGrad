cmake_minimum_required(VERSION 3.22.1)
set(SRC_DIR ../)
project(nano_grad_backend)
add_executable(${PROJECT_NAME} ${SRC_DIR}/main.cpp)

target_include_directories(${PROJECT_NAME} PUBLIC ${SRC_DIR}/Autograd)

target_link_directories(${PROJECT_NAME} PRIVATE ${SRC_DIR}/Autograd)

target_link_libraries(${PROJECT_NAME} ${SRC_DIR}/autograd)