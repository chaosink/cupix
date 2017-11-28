cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cupix LANGUAGES CXX CUDA)

find_package(CUDA 9.0 REQUIRED)
find_package(OpenGL REQUIRED)

# C++ list
list(APPEND CMAKE_CXX_FLAGS " -Wall -O3 ")

# CUDA list
string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_30,code=sm_30 ")
# string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_32,code=sm_32 ")
# string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_35,code=sm_35 ")
# string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_37,code=sm_37 ")
# string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_50,code=sm_50 ")
# string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_52,code=sm_52 ")
# string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_53,code=sm_53 ")
# string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_60,code=sm_60 ")
string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_61,code=sm_61 ")
# string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_62,code=sm_62 ")
# string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_70,code=sm_70 ")

include_directories(
	src/core
	src/utility
	src/thirdparty
	src/thirdparty/glew-2.1.0/include
	src/thirdparty/glfw-3.2.1/include
	src/thirdparty/glm-0.9.8.5
	${CUDA_INCLUDE_DIRS}
	${OPENGL_INCLUDE_DIR}
)

add_library(core
	src/core/CUPix.cu
	src/core/CUPixCore.cu
	src/core/VertexShader.cu
	src/core/FragmentShader.cu
	src/core/Shadertoy.cu
)
target_compile_features(core PUBLIC cxx_std_11)
set_target_properties(core PROPERTIES
	CUDA_SEPARABLE_COMPILATION ON
)

add_executable(cupix
	src/main.cpp
	src/utility/Camera.cpp
	src/utility/Model.cpp
	src/utility/FPS.cpp
	src/utility/Texture.cpp
	src/utility/Video.cpp
	src/utility/Toggle.cpp
)
target_link_libraries(cupix
	core
	glew_s
	glfw
	${CUDA_LIBRARIES}
	${OPENGL_gl_LIBRARY}
)
if(APPLE)
	# We need to add the default path to the driver (libcuda.dylib) as an rpath,
	# so that the static cuda runtime can find it at runtime.
	set_property(TARGET cupix PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/utility/raw2mp4.sh ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/font/bitmap_font.data ${CMAKE_CURRENT_BINARY_DIR}/font/bitmap_font.data COPYONLY)
file(GLOB textures RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/texture/*)
foreach(tex ${textures})
	configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${tex} ${CMAKE_CURRENT_BINARY_DIR}/${tex} COPYONLY)
endforeach()



option(BUILD_UTILS "utilities" OFF)
add_subdirectory(src/thirdparty/glew-2.1.0/build/cmake)

option(GLFW_BUILD_EXAMPLES "Build the GLFW example programs" OFF)
option(GLFW_BUILD_TESTS "Build the GLFW test programs" OFF)
option(GLFW_BUILD_DOCS "Build the GLFW documentation" OFF)
add_subdirectory(src/thirdparty/glfw-3.2.1)
