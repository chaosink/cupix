#pragma once

#include <cuda_runtime.h>
#include <glm/glm.hpp>

namespace cupix {

const double PI  = 3.14159265358979323846;

struct VertexIn {
	glm::vec3 position;
	glm::vec3 normal;
	glm::vec3 color;
	glm::vec2 uv;
};
struct VertexOut {
	glm::vec4 position;
	glm::vec3 normal;
	glm::vec3 color;
	glm::vec2 uv;
};
struct FragmentIn {
	glm::ivec2 position;
	float z;
	float depth;
	glm::vec3 normal;
	glm::vec3 color;
	glm::vec2 uv;
};
struct AABB {
	glm::ivec2 v[2];
};

enum Flag {
	DEPTH_TEST,
	BLEND,
};

class CUPix {
	int window_w_, window_h_;
	cudaGraphicsResource *pbo_resource_;
	unsigned char *pbo_ptr_;
	bool record_;
	glm::vec4 clear_color_;
	int n_triangle_, n_vertex_;
	VertexIn *vertex_in_;
	VertexOut *vertex_out_;
	AABB *aabb_buf_, *aabb_;
	glm::mat4 *mvp_buf_;
	float *frame_buf_;
	float *depth_buf_;
	unsigned char *texture_buf_;
	unsigned char *frame_;
public:
	CUPix(int window_w, int window_h, unsigned int buffer, bool record);
	~CUPix();
	void MapResources();
	void UnmapResources();
	void Enable(Flag flag);
	void Disable(Flag flag);
	void ClearColor(float r, float g, float b, float a);
	void Clear();
	void Draw();
	void DrawFPS(int fps);
	void VertexData(int size, float *position, float *normal, float *uv);
	void MVP(glm::mat4 &mvp);
	void Time(double time);
	void Texture(unsigned char *data, int w, int h, bool gamma_correction);
	unsigned char* frame() {
		return frame_;
	}
};

}
