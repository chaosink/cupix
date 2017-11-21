#pragma once

#include <cstdlib>
#include <cstdio>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

namespace cupix {

const double PI  = 3.14159265358979323846;

class CUPix {
	int window_w_, window_h_;
	cudaGraphicsResource *buffer_resource_;
	unsigned char *buffer_ptr_;
public:
	CUPix(int window_w, int window_h, GLuint buffer);
	~CUPix();
	void MapResources();
	void UnmapResources();
	void Clear();
	void Draw();
};

class FPS {
	int c_frame_ = 0;
	float time_old_ = glfwGetTime();
	float time_new_;
public:
	void Update();
};


namespace cu { // CUDA kernels

__global__
void Clear(unsigned char *buffer, int w, int h);

__global__
void T1(unsigned char *buffer, int w, int h, double time);

}

}
