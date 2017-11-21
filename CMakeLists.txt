cmake_minimum_required(VERSION 3.5.1)
project(cupix)

# C++ list
list(APPEND CMAKE_CXX_FLAGS "-std=c++11 -Wall")

# CUDA list
list(APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_30,code=sm_30)
#list(APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_50,code=sm_50)
#list(APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_52,code=sm_52)
#list(APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_52,code=compute_52)
#list(APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_60,code=sm_60)
list(APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_61,code=sm_61)
#list(APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_62,code=sm_62)

find_package(CUDA REQUIRED)
find_package(glfw3 3.3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(OpenCV REQUIRED)

cuda_add_executable(cupix cupix.cu cupix.hpp main.cpp)
target_include_directories(cupix PUBLIC ${OPENGL_INCLUDE_DIR})
target_link_libraries(cupix
	${CUDA_LIBRARIES}
	${OPENGL_gl_LIBRARY}
	glfw
	GLEW
	${OpenCV_LIBS}
)
