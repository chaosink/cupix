#pragma once

#include <cuda_runtime.h>
#define GLM_FORCE_SWIZZLE
#include <glm/glm.hpp>

namespace cupix {

namespace cu{

struct Light {
	float position[3];
	float color[3];
	float power;
};

}

const double PI  = 3.14159265358979323846;

enum Winding : unsigned char {
	CCW,
	CW
};

enum Face : unsigned char {
	BACK,
	FRONT,
	FRONT_AND_BACK
};

enum Flag : unsigned char {
	DEPTH_TEST,
	BLEND,
	CULL_FACE,
};

struct VertexIn {
	glm::vec3 position;
	glm::vec3 normal;
	glm::vec3 color;
	glm::vec2 uv;
};
struct VertexOut {
	glm::vec3 position;
	glm::vec3 normal;
	glm::vec3 color;
	glm::vec2 uv;
};
struct Vertex {
	glm::vec4 position;
};
struct Triangle {
	glm::ivec2 v[2];
	Winding winding;
	bool empty;
};
struct FragmentIn {
	glm::vec2 coord;
	float z;

	glm::vec3 position;
	glm::vec3 normal;
	glm::vec3 color;
	glm::vec2 uv;
};

class CUPix {
	int window_w_, window_h_;
	cudaGraphicsResource *pbo_resource_;
	unsigned char *pbo_buf_;
	bool record_;
	unsigned char *frame_;

	glm::vec4 clear_color_;
	bool cull_ = true;
	Face cull_face_ = BACK;
	Winding front_face_ = CCW;

	int n_triangle_, n_vertex_;
	Triangle *triangle_;

	VertexIn *vertex_in_;
	VertexOut *vertex_out_;
	Vertex *vertex_buf_;
	Triangle *triangle_buf_;
	unsigned char *frame_buf_;
	float *depth_buf_;
	unsigned char *texture_buf_;

public:
	CUPix(int window_w, int window_h, unsigned int buffer, bool record);
	~CUPix();
	void MapResources();
	void UnmapResources();
	void Enable(Flag flag);
	void Disable(Flag flag);
	void CullFace(Face face);
	void FrontFace(Winding winding);
	void ClearColor(float r, float g, float b, float a);
	void Clear();
	void Draw();
	void DrawFPS(int fps);
	void VertexData(int size, float *position, float *normal, float *uv);
	void MVP(glm::mat4 &mvp);
	void MV(glm::mat4 &mv);
	void Time(double time);
	void Texture(unsigned char *data, int w, int h, bool gamma_correction);
	void Light(int n, cu::Light *light);
	void Toggle(bool toggle);
	unsigned char* frame() {
		return frame_;
	}
};

}
