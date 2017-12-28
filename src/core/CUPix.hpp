#pragma once

#include <cuda_runtime.h>
#include "glm/glm.hpp"

#define VERSION_MAJOR 0
#define VERSION_MINOR 1
#define VERSION_PATCH 2

#define _QUOTE(S) #S
#define _STR(S) _QUOTE(S)
#define VERSION_STRING _STR(VERSION_MAJOR) "." _STR(VERSION_MINOR) "." _STR(VERSION_PATCH)

namespace cupix {

enum AA : unsigned char {
	NOAA,
	MSAA,
	SSAA
};
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
	glm::vec2 uv;
};
struct VertexOut {
	glm::vec3 position;
	glm::vec3 normal;
	glm::vec2 uv;
};
struct Vertex {
	glm::vec4 position;
};
struct Triangle {
	glm::ivec2 aabb[2];
	Winding winding;
	bool empty;
};
struct FragmentIn {
	glm::vec2 coord;
	float z;

	glm::vec3 position;
	glm::vec3 normal;
	glm::vec2 uv;
};
struct Light {
	float position[3];
	float color[3];
	float power;
};

class CUPix {
	int window_w_, window_h_;
	cudaGraphicsResource *pbo_resource_;
	unsigned char *pbo_buf_;
	bool record_;
	unsigned char *frame_;

	int frame_w_, frame_h_;
	AA aa_;
	bool cull_ = true;
	Face cull_face_ = BACK;
	Winding front_face_ = CCW;
	int n_triangle_, n_vertex_;
	Triangle *triangle_ = NULL;
	unsigned char *frame_buf_;
	float *depth_buf_;

	VertexIn *vertex_in_ = NULL;
	VertexOut *vertex_out_ = NULL;
	Vertex *vertex_buf_ = NULL;
	Triangle *triangle_buf_ = NULL;
	unsigned char *texture_buf_ = NULL;

public:
	CUPix(int window_w, int window_h, unsigned int buffer, AA aa, bool record);
	~CUPix();
	void BeforeDraw();
	void AfterDraw();
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
	void Time(float time);
	void Texture(unsigned char *data, int w, int h, bool gamma_correction);
	void Lights(int n, Light *light);
	void Toggle(bool toggle);
	unsigned char* frame() {
		return frame_;
	}
};

}
