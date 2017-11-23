#pragma once

#include <cuda_runtime.h>
#include <glm/glm.hpp>

namespace cupix {

const double PI  = 3.14159265358979323846;

struct VertexIn {
	glm::vec3 position;
	glm::vec3 normal;
	glm::vec3 color;
};
struct VertexOut {
	glm::vec4 position;
	glm::vec3 normal;
	glm::vec3 color;
};
struct FragmentIn {
	glm::ivec2 position;
	float depth;
	glm::vec3 normal;
	glm::vec3 color;
};
struct AABB {
	glm::ivec2 v[2];
};
struct Triangle {
	VertexOut v[3];
};

class CUPix {
	int window_w_, window_h_;
	cudaGraphicsResource *pbo_resource_;
	unsigned char *pbo_ptr_;
	glm::vec4 clear_color_;
	int n_triangle_, n_vertex_;
	VertexIn *vertex_in_;
	VertexOut *vertex_out_;
	AABB *aabb_buf_, *aabb_;
	glm::mat4 *mvp_buf_;
	float *frame_buf_;
	float *depth_buf_;
public:
	CUPix(int window_w, int window_h, unsigned int buffer);
	~CUPix();
	void MapResources();
	void UnmapResources();
	void ClearColor(float r, float g, float b, float a);
	void Clear();
	void Draw();
	void VertexData(int size, float *position, float *normal);
	void MVP(glm::mat4 &mvp);
};

// namespace cu { // CUDA kernels

// __global__
// void Clear(unsigned char *buffer, int w, int h, glm::vec4 color);

// __global__
// void T1(unsigned char *buffer, int w, int h);

// __global__
// void Test(glm::vec3 c);

// }

}
