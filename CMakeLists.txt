cmake_minimum_required(VERSION 3.5.1)
project(cupix)

find_package(CUDA 9.0 REQUIRED)
find_package(OpenGL REQUIRED)

# C++ list
set(CMAKE_CXX_STANDARD 11)
list(APPEND CMAKE_CXX_FLAGS "-std=c++11 -Wall -O3")

option(BUILD_UTILS "utilities" OFF)
add_subdirectory(src/thirdparty/glew-2.1.0/build/cmake)

option(GLFW_BUILD_EXAMPLES "Build the GLFW example programs" OFF)
option(GLFW_BUILD_TESTS "Build the GLFW test programs" OFF)
option(GLFW_BUILD_DOCS "Build the GLFW documentation" OFF)
add_subdirectory(src/thirdparty/glfw-3.2.1)

# CUDA list
list(APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_30,code=sm_30)
#list(APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_50,code=sm_50)
#list(APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_52,code=sm_52)
#list(APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_52,code=compute_52)
#list(APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_60,code=sm_60)
list(APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_61,code=sm_61)
#list(APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_62,code=sm_62)

# set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)
set(CUDA_SEPARABLE_COMPILATION ON)

include_directories(
	src/core
	src/thirdparty
	src/thirdparty/glew-2.1.0/include
	src/thirdparty/glfw-3.2.1/include
	src/thirdparty/glm-0.9.8.5
	${OPENGL_INCLUDE_DIR}
)

cuda_add_library(core
	src/core/shader.cu
	src/core/cupix.cu
)

cuda_add_executable(cupix
	src/main.cpp
	src/util/camera.cpp
)
target_link_libraries(cupix
	core
	glew_s
	glfw
	${CUDA_LIBRARIES}
	${OPENGL_gl_LIBRARY}
)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/util/raw2mp4.sh ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
