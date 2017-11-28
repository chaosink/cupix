#include "CUPix.hpp"

namespace cupix {

namespace core {

extern __constant__ __device__ int w, h;
extern __constant__ __device__ int n_triangle;
extern __constant__ __device__ bool depth_test;
extern __constant__ __device__ bool blend;
extern __constant__ __device__ unsigned char clear_color[4];

__constant__ __device__ unsigned char bit[8] = {
	0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01};
extern __constant__ __device__ char bitmap[];

__device__
void VertexShader(VertexIn &in, VertexOut &out, Vertex &v);
__device__
void FragmentShader(FragmentIn &in, glm::vec4 &color);

__global__
void Clear(unsigned char *frame_buf, float *depth_buf) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if(x >= w || y >= h) return;
	int i_pixel = y * w + x;
	frame_buf[i_pixel * 3 + 0] = clear_color[0];
	frame_buf[i_pixel * 3 + 1] = clear_color[1];
	frame_buf[i_pixel * 3 + 2] = clear_color[2];
	depth_buf[i_pixel] = 0;
}

__global__
void NormalSpace(VertexIn *in, VertexOut *out, Vertex *v) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	if(x >= n_triangle * 3) return;

	VertexShader(in[x], out[x], v[x]);
}

__global__
void WindowSpace(Vertex *v) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	if(x >= n_triangle * 3) return;

	float w_inv = 1.f / v[x].position.w;
	glm::mat4 m;
	glm::vec3 p = v[x].position * w_inv;
	p.x = (p.x * 0.5f + 0.5f) * w;
	p.y = (p.y * 0.5f + 0.5f) * h;
	v[x].position.x = p.x;
	v[x].position.y = p.y;
	v[x].position.z = p.z;
}

__global__
void AssemTriangle(Vertex *v, Triangle *triangle) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	if(x >= n_triangle) return;

	glm::vec2
		p0(v[x * 3 + 0].position.x, v[x * 3 + 0].position.y),
		p1(v[x * 3 + 1].position.x, v[x * 3 + 1].position.y),
		p2(v[x * 3 + 2].position.x, v[x * 3 + 2].position.y);
	glm::vec2
		v_min = glm::min(glm::min(p0, p1), p2),
		v_max = glm::max(glm::max(p0, p1), p2);
	glm::ivec2
		c0 = glm::ivec2(0, 0),
		c1 = glm::ivec2(w - 1, h - 1),
		iv_min = v_min + 0.5f,
		iv_max = v_max + 0.5f;

	triangle[x].empty = (iv_min.x >= w || iv_min.y >= h || iv_max.x < 0 || iv_max.y < 0
		|| v[0].position.z > 1 && v[1].position.z > 1 && v[2].position.z > 1
		|| v[0].position.z <-1 && v[1].position.z <-1 && v[2].position.z <-1);
	triangle[x].winding = Winding((p1.x - p0.x) * (p2.y - p1.y) - (p1.y - p0.y) * (p2.x - p1.x) < 0);

	iv_min = glm::clamp(iv_min, c0, c1);
	iv_max = glm::clamp(iv_max, c0, c1);

	triangle[x].aabb[0] = iv_min;
	triangle[x].aabb[1] = iv_max;
}

__device__
void Interpolate(Vertex *v, VertexOut *va, glm::vec3 &e, FragmentIn *f) { // va: vertex attibute
	glm::vec3 we(
		e.x / v[0].position.w,
		e.y / v[1].position.w,
		e.z / v[2].position.w);
	float w = 1.f / (we.x + we.y + we.z);

	f->position = (
		va[0].position * we.x +
		va[1].position * we.y +
		va[2].position * we.z) * w;
	f->normal = (
		va[0].normal * we.x +
		va[1].normal * we.y +
		va[2].normal * we.z) * w;
	f->color = (
		va[0].color * we.x +
		va[1].color * we.y +
		va[2].color * we.z) * w;
	f->uv = (
		va[0].uv * we.x +
		va[1].uv * we.y +
		va[2].uv * we.z) * w;

	f->z =
		e.x * v[0].position.z +
		e.y * v[1].position.z +
		e.z * v[2].position.z;
}

__global__
void Rasterize(glm::ivec2 corner, glm::ivec2 dim, Vertex *v, VertexOut *va, float *depth_buf, unsigned char* frame_buf) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if(x >= dim.x || y >= dim.y) return;
	x += corner.x;
	y += corner.y;
	if(x >= w || y >= h) return;
	int i_pixel = y * w + x;

	glm::vec2
		p0 = glm::vec2(v[0].position.x, v[0].position.y),
		p1 = glm::vec2(v[1].position.x, v[1].position.y),
		p2 = glm::vec2(v[2].position.x, v[2].position.y);
	glm::vec2 d01 = p1 - p0, d12 = p2 - p1, d20 = p0 - p2;
	float e0 = glm::dot(d12, glm::vec2(y + 0.5f - p1.y, p1.x - x - 0.5f));
	float e1 = glm::dot(d20, glm::vec2(y + 0.5f - p2.y, p2.x - x - 0.5f));
	float e2 = glm::dot(d01, glm::vec2(y + 0.5f - p0.y, p0.x - x - 0.5f));

	if(e0 >= 0 && e1 >= 0 && e2 >= 0
	|| e0 <= 0 && e1 <= 0 && e2 <= 0) {
		FragmentIn fragment = {glm::vec2(x, y)};
		glm::vec3 e = glm::vec3(e0, e1, e2) / (e0 + e1 + e2);
		Interpolate(v, va, e, &fragment);
		if(fragment.z > 1 || fragment.z < -1) return; // need 3D clipping
		if(!depth_test || 1 - fragment.z > depth_buf[i_pixel]) {
			depth_buf[i_pixel] = 1 - fragment.z;
			glm::vec4 color;
			FragmentShader(fragment, color);
			glm::ivec4 icolor;
			if(blend) {
				float alpha = color.a;
				glm::vec4 color_old = glm::vec4(
					frame_buf[i_pixel * 3 + 0],
					frame_buf[i_pixel * 3 + 1],
					frame_buf[i_pixel * 3 + 2],
					0.f
				);
				icolor = color * 255.f * alpha + color_old * (1.f - alpha);
			} else {
				icolor = color * 255.f;
			}
			icolor = glm::clamp(icolor, glm::ivec4(0), glm::ivec4(255));
			frame_buf[i_pixel * 3 + 0] = icolor.r;
			frame_buf[i_pixel * 3 + 1] = icolor.g;
			frame_buf[i_pixel * 3 + 2] = icolor.b;
		}
	}
}

__global__
void RasterizeMSAA(glm::ivec2 corner, glm::ivec2 dim, Vertex *v, VertexOut *va, float *depth_buf, unsigned char* frame_buf) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if(x >= dim.x || y >= dim.y) return;
	x = (x + corner.x) * 2;
	y = (y + corner.y) * 2;
	if(x >= w || y >= h) return;

	float e0, e1, e2;

	glm::vec2
		p0 = glm::vec2(v[0].position.x, v[0].position.y),
		p1 = glm::vec2(v[1].position.x, v[1].position.y),
		p2 = glm::vec2(v[2].position.x, v[2].position.y);
	glm::vec2 d01 = p1 - p0, d12 = p2 - p1, d20 = p0 - p2;
	bool cover[2][2];
	bool covered = false;
	for(int i = 0; i < 2; i++)
		for(int j = 0; j < 2; j++) {
			cover[i][j] = false;
			int xx = x + j, yy = y + i;
			int i_pixel = yy * w + xx;
			e0 = glm::dot(d12, glm::vec2(yy + 0.5f - p1.y, p1.x - xx - 0.5f));
			e1 = glm::dot(d20, glm::vec2(yy + 0.5f - p2.y, p2.x - xx - 0.5f));
			e2 = glm::dot(d01, glm::vec2(yy + 0.5f - p0.y, p0.x - xx - 0.5f));
			if(e0 >= 0 && e1 >= 0 && e2 >= 0
			|| e0 <= 0 && e1 <= 0 && e2 <= 0) {
				glm::vec3 e = glm::vec3(e0, e1, e2) / (e0 + e1 + e2);
				float z = e.x * v[0].position.z + e.y * v[1].position.z + e.z * v[2].position.z;
				if(z > 1 || z < -1) continue;
				if(!depth_test || 1 - z > depth_buf[i_pixel]) {
					depth_buf[i_pixel] = 1 - z;
					cover[i][j] = covered = true;
				}
			}
		}
	if(!covered) return;

	float xx = x + 0.5f, yy = y + 0.5f;
	e0 = glm::dot(d12, glm::vec2(yy + 0.5f - p1.y, p1.x - xx - 0.5f));
	e1 = glm::dot(d20, glm::vec2(yy + 0.5f - p2.y, p2.x - xx - 0.5f));
	e2 = glm::dot(d01, glm::vec2(yy + 0.5f - p0.y, p0.x - xx - 0.5f));

	FragmentIn fragment = {glm::vec2(xx, yy)};
	glm::vec3 e = glm::vec3(e0, e1, e2) / (e0 + e1 + e2);
	Interpolate(v, va, e, &fragment);
	// if(fragment.z > 1 || fragment.z < -1) return; // extrapolation may cause z not in [-1,1]
	glm::vec4 color;
	FragmentShader(fragment, color); // run fragment shader noly once
	glm::ivec4 icolor;
	for(int i = 0; i < 2; i++)
		for(int j = 0; j < 2; j++)
			if(cover[i][j]) {
				int xx = x + j, yy = y + i;
				int i_pixel = yy * w + xx;
				if(blend) {
					float alpha = color.a;
					glm::vec4 color_old = glm::vec4(
						frame_buf[i_pixel * 3 + 0],
						frame_buf[i_pixel * 3 + 1],
						frame_buf[i_pixel * 3 + 2],
						0.f
					);
					icolor = color * 255.f * alpha + color_old * (1.f - alpha);
				} else {
					icolor = color * 255.f;
				}
				icolor = glm::clamp(icolor, glm::ivec4(0), glm::ivec4(255));
				frame_buf[i_pixel * 3 + 0] = icolor.r;
				frame_buf[i_pixel * 3 + 1] = icolor.g;
				frame_buf[i_pixel * 3 + 2] = icolor.b;
			}
}

__global__
void DrawCharater(int ch, int x0, int y0, bool ssaa, unsigned char *frame_buf) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int i_pixel = w * (y + y0) + x + x0;

	int offset = ch * 32;
	char c = bitmap[offset + (15 - y) * 2 + x / 8];
	if(!(c & bit[x % 8])) return;
	if(ssaa) {
		int p0 = (((y + y0) * 2 + 0) * w + (x + x0) * 2 + 0) * 3;
		int p1 = (((y + y0) * 2 + 0) * w + (x + x0) * 2 + 1) * 3;
		int p2 = (((y + y0) * 2 + 1) * w + (x + x0) * 2 + 0) * 3;
		int p3 = (((y + y0) * 2 + 1) * w + (x + x0) * 2 + 1) * 3;
		frame_buf[p0 + 0] = frame_buf[p1 + 0] = frame_buf[p2 + 0] = frame_buf[p3 + 0] = 255 - clear_color[0];
		frame_buf[p0 + 1] = frame_buf[p1 + 1] = frame_buf[p2 + 1] = frame_buf[p3 + 1] = 255 - clear_color[1];
		frame_buf[p0 + 2] = frame_buf[p1 + 2] = frame_buf[p2 + 2] = frame_buf[p3 + 2] = 255 - clear_color[2];
	} else {
		frame_buf[i_pixel * 3 + 0] = 255 - clear_color[0];
		frame_buf[i_pixel * 3 + 1] = 255 - clear_color[1];
		frame_buf[i_pixel * 3 + 2] = 255 - clear_color[2];
	}
}

__global__
void DownSample(unsigned char *frame_buf, unsigned char *pbo_buf) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if(x >= w / 2 || y >= h / 2) return;
	int i_pixel = w / 2 * y + x;

	int p0 = ((y * 2 + 0) * w + x * 2 + 0) * 3;
	int p1 = ((y * 2 + 0) * w + x * 2 + 1) * 3;
	int p2 = ((y * 2 + 1) * w + x * 2 + 0) * 3;
	int p3 = ((y * 2 + 1) * w + x * 2 + 1) * 3;

	int r = (frame_buf[p0 + 0] + frame_buf[p1 + 0] + frame_buf[p2 + 0] + frame_buf[p3 + 0]) / 4;
	int g = (frame_buf[p0 + 1] + frame_buf[p1 + 1] + frame_buf[p2 + 1] + frame_buf[p3 + 1]) / 4;
	int b = (frame_buf[p0 + 2] + frame_buf[p1 + 2] + frame_buf[p2 + 2] + frame_buf[p3 + 2]) / 4;

	pbo_buf[i_pixel * 3 + 0] = glm::clamp(r, 0, 255);
	pbo_buf[i_pixel * 3 + 1] = glm::clamp(g, 0, 255);
	pbo_buf[i_pixel * 3 + 2] = glm::clamp(b, 0, 255);
}

}

}
