#include <cstdio>
#include <iostream>
using namespace std;

#include "cupix.hpp"

#include <cuda_gl_interop.h>

namespace cupix {

namespace cu {

texture<uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat> texture;
__constant__ __device__ int w, h;
__constant__ __device__ Light light;
__constant__ __device__ float mvp[16];
__constant__ __device__ float mv[16];
__constant__ __device__ float time;
__constant__ __device__ bool toggle;

__constant__ __device__ int n_triangle;
__constant__ __device__ bool depth_test = true;
__constant__ __device__ bool blend = false;
__constant__ __device__ unsigned char clear_color[4];

const int zb16_file_size = 269888;
__constant__ __device__ unsigned char bit[8] = {
	0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01};
__device__ char zb16[zb16_file_size];

extern __device__
void VertexShader(VertexIn &in, VertexOut &out, Vertex &v);

extern __device__
void FragmentShader(FragmentIn &in, glm::vec4 &color);

__global__
void Clear(unsigned char *frame_buf, float *depth_buf) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if(x >= w || y >= h) return;
	int i_thread = y * w + x;
	frame_buf[i_thread * 3 + 0] = clear_color[0];
	frame_buf[i_thread * 3 + 1] = clear_color[1];
	frame_buf[i_thread * 3 + 2] = clear_color[2];
	depth_buf[i_thread] = 0;
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

	iv_min = glm::clamp(iv_min, c0, c1);
	iv_max = glm::clamp(iv_max, c0, c1);

	triangle[x].v[0] = iv_min;
	triangle[x].v[1] = iv_max;
	triangle[x].winding = Winding((p1.x - p0.x) * (p2.y - p1.y) - (p1.y - p0.y) * (p2.x - p1.x) < 0);
	triangle[x].empty = (iv_min.x == iv_max.x || iv_min.y == iv_max.y
		|| v[0].position.z > 1 && v[1].position.z > 1 && v[2].position.z > 1
		|| v[0].position.z <-1 && v[1].position.z <-1 && v[2].position.z <-1);
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
	int i_thread = y * w + x;

	// if(v[0].position.z > 1 && v[1].position.z > 1 && v[2].position.z > 1
	// || v[0].position.z <-1 && v[1].position.z <-1 && v[2].position.z <-1)
	// 	return; // cause FPS reduction of about 1
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
		FragmentIn fragment = {glm::ivec2(x, y)};
		glm::vec3 e = glm::vec3(e0, e1, e2) / (e0 + e1 + e2);
		Interpolate(v, va, e, &fragment);
		if(fragment.z > 1 || fragment.z < -1) return;
		if(!depth_test || 1 - fragment.z > depth_buf[i_thread]) {
			depth_buf[i_thread] = 1 - fragment.z;
			glm::vec4 color;
			FragmentShader(fragment, color);
			glm::ivec4 icolor;
			if(blend) {
				float alpha = color.a;
				glm::vec4 color_old = glm::vec4(
					frame_buf[i_thread * 3 + 0],
					frame_buf[i_thread * 3 + 1],
					frame_buf[i_thread * 3 + 2],
					0.0f
				);
				icolor = color * 255.f * alpha + color_old * (1.f - alpha);
			} else {
				icolor = color * 255.f;
			}
			icolor = glm::clamp(icolor, glm::ivec4(0), glm::ivec4(255));
			frame_buf[i_thread * 3 + 0] = icolor.r;
			frame_buf[i_thread * 3 + 1] = icolor.g;
			frame_buf[i_thread * 3 + 2] = icolor.b;
		}
	}
}

__global__
void Font(int ch, int x0, int y0, unsigned char *frame_buf) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int i_pixel = w * (y + y0) + x + x0;

	int offset = (ch + 155) * 32 + 0x2000;
	char c = zb16[offset + (16 - y) * 2 + x / 8];
	if(!(c & bit[(x % 8)])) return;
	frame_buf[i_pixel * 3 + 0] = 255 - clear_color[0];
	frame_buf[i_pixel * 3 + 1] = 255 - clear_color[1];
	frame_buf[i_pixel * 3 + 2] = 255 - clear_color[2];
}

}



CUPix::CUPix(int window_w, int window_h, GLuint pbo, bool record = false)
	: window_w_(window_w), window_h_(window_h), record_(record) {
	frame_ = new unsigned char[window_w_ * window_h_ * 3];
	cudaMalloc(&depth_buf_, sizeof(float) * window_w_ * window_h_);
	// cudaMalloc(&frame_buf_, sizeof(float) * window_w_ * window_h_);
	cudaMemcpyToSymbol(cu::w, &window_w_, sizeof(int));
	cudaMemcpyToSymbol(cu::h, &window_h_, sizeof(int));
	cudaGraphicsGLRegisterBuffer(&pbo_resource_, pbo, cudaGraphicsMapFlagsNone);

	FILE *zb16_file = fopen("../font/zb16.data", "rb");
	char zb16[cu::zb16_file_size];
	size_t r = fread(zb16, 1, cu::zb16_file_size, zb16_file);
	fclose(zb16_file);
	cudaMemcpyToSymbol(cu::zb16, zb16, cu::zb16_file_size);
}

CUPix::~CUPix() {
	delete[] frame_;
	cudaFree(depth_buf_);
	// cudaFree(frame_buf_);
	cudaGraphicsUnregisterResource(pbo_resource_);
}

void CUPix::MapResources() {
	size_t size;
	cudaGraphicsMapResources(1, &pbo_resource_, NULL);
	cudaGraphicsResourceGetMappedPointer((void**)&pbo_buf_, &size, pbo_resource_);
	frame_buf_ = pbo_buf_;
}

void CUPix::UnmapResources() {
	if(record_)
		cudaMemcpy(frame_, frame_buf_, window_w_ * window_h_ * 3, cudaMemcpyDeviceToHost);
	cudaGraphicsUnmapResources(1, &pbo_resource_, NULL);
}

void CUPix::Enable(Flag flag) {
	bool b = true;
	switch(flag) {
		case DEPTH_TEST:
			cudaMemcpyToSymbol(cu::depth_test, &b, 1); return;
		case BLEND:
			cudaMemcpyToSymbol(cu::blend, &b, 1); return;
		case CULL_FACE:
			cull_ = b;
	}
}

void CUPix::Disable(Flag flag) {
	bool b = false;
	switch(flag) {
		case DEPTH_TEST:
			cudaMemcpyToSymbol(cu::depth_test, &b, 1); return;
		case BLEND:
			cudaMemcpyToSymbol(cu::blend, &b, 1); return;
		case CULL_FACE:
			cull_ = b; return;
	}
}

void CUPix::CullFace(Face face) {
	cull_face_ = face;
}

void CUPix::FrontFace(Winding winding) {
	front_face_ = winding;
}

void CUPix::ClearColor(float r, float g, float b, float a) {
	glm::ivec4 color = glm::vec4(r * 255.f, g * 255.f, b * 255.f, a * 255.f);
	color = glm::clamp(color, glm::ivec4(0), glm::ivec4(255));
	glm::u8vec4 clear_color = color;
	cudaMemcpyToSymbol(cu::clear_color, &clear_color, 4);
}

void CUPix::Clear() {
	cu::Clear<<<dim3((window_w_-1)/32+1, (window_h_-1)/32+1), dim3(32, 32)>>>(frame_buf_, depth_buf_);
}

void CUPix::Draw() {
	cu::NormalSpace<<<(n_triangle_*3-1)/32+1, 32>>>(vertex_in_, vertex_out_, vertex_buf_);
	cu::WindowSpace<<<(n_triangle_*3-1)/32+1, 32>>>(vertex_buf_);
	cu::AssemTriangle<<<(n_triangle_-1)/32+1, 32>>>(vertex_buf_, triangle_buf_);
	cudaMemcpy(triangle_, triangle_buf_, sizeof(Triangle) * n_triangle_, cudaMemcpyDeviceToHost);
	for(int i = 0; i < n_triangle_; i++)
		if(!triangle_[i].empty)
			if(!cull_ || (cull_face_ != FRONT_AND_BACK
			&& (triangle_[i].winding == front_face_ != cull_face_))) {
				glm::ivec2 dim = triangle_[i].v[1] - triangle_[i].v[0] + 1;
				cu::Rasterize<<<dim3((dim.x-1)/4+1, (dim.y-1)/8+1), dim3(4, 8)>>>
					(triangle_[i].v[0], dim, vertex_buf_ + i * 3, vertex_out_ + i * 3, depth_buf_, frame_buf_);
		}
}

void CUPix::DrawFPS(int fps) {
	cu::Font<<<1, dim3(16, 16)>>>('F',  0, 0, frame_buf_);
	cu::Font<<<1, dim3(16, 16)>>>('P', 16, 0, frame_buf_);
	cu::Font<<<1, dim3(16, 16)>>>('S', 32 - 3, 0, frame_buf_);
	cu::Font<<<1, dim3(16, 16)>>>(fps % 1000 / 100 + 48, 48 + 5, 0, frame_buf_);
	cu::Font<<<1, dim3(16, 16)>>>(fps % 100 / 10   + 48, 64 + 5, 0, frame_buf_);
	cu::Font<<<1, dim3(16, 16)>>>(fps % 10         + 48, 80 + 5, 0, frame_buf_);
}

void CUPix::VertexData(int size, float *position, float *normal, float *uv) {
	n_vertex_ = size;
	n_triangle_ = n_vertex_ / 3;
	VertexIn v[n_vertex_];
	for(int i = 0; i < n_vertex_; i++) {
		v[i].position = glm::vec3(position[i * 3], position[i * 3 + 1], position[i * 3 + 2]);
		v[i].normal = glm::vec3(normal[i * 3], normal[i * 3 + 1], normal[i * 3 + 2]);
		v[i].color = glm::vec3(rand() / RAND_MAX, rand() / RAND_MAX, rand() / RAND_MAX);
		v[i].uv = glm::vec2(uv[i * 2], uv[i * 2 + 1]);
	}
	cudaMalloc(&vertex_in_, sizeof(v));
	cudaMemcpy(vertex_in_, v, sizeof(v), cudaMemcpyHostToDevice);
	cudaMalloc(&vertex_buf_, sizeof(Vertex) * n_vertex_);
	cudaMalloc(&triangle_buf_, sizeof(Triangle) * n_triangle_);
	triangle_ = new Triangle[n_triangle_];
	cudaMalloc(&vertex_out_, sizeof(VertexOut) * n_vertex_);
	cudaMemcpyToSymbol(cu::n_triangle, &n_triangle_, sizeof(int));
}

void CUPix::MVP(glm::mat4 &mvp) {
	cudaMemcpyToSymbol(cu::mvp, &mvp, sizeof(glm::mat4));
}

void CUPix::MV(glm::mat4 &mv) {
	cudaMemcpyToSymbol(cu::mv, &mv, sizeof(glm::mat4));
}

void CUPix::Time(double time) {
	float t = time;
	cudaMemcpyToSymbol(cu::time, &t, sizeof(float));
}

void CUPix::Texture(unsigned char *d, int w, int h, bool gamma_correction) {
	unsigned char data[w * h * 4];
	for(int i = 0; i < w * h; i++) {
		data[i * 4 + 0] = d[i * 3 + 0];
		data[i * 4 + 1] = d[i * 3 + 1];
		data[i * 4 + 2] = d[i * 3 + 2];
		data[i * 4 + 3] = 0;
	}
	size_t pitch;
	cudaMallocPitch((void**)&texture_buf_, &pitch, w * 4, h);
	cudaMemcpy2D(
		texture_buf_, pitch,
		data, w * 4,
		w * 4, h,
		cudaMemcpyHostToDevice);

	cu::texture.normalized = true;
	cu::texture.sRGB = gamma_correction;
	cu::texture.filterMode = cudaFilterModeLinear;
	cu::texture.addressMode[0] = cudaAddressModeWrap;
	cu::texture.addressMode[1] = cudaAddressModeWrap;
	cudaChannelFormatDesc desc =
		cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);

	cudaBindTexture2D(NULL, cu::texture, texture_buf_, desc, w, h, pitch);
}

void CUPix::Light(cu::Light &light) {
	cudaMemcpyToSymbol(cu::light, &light, sizeof(light));
}

void CUPix::Toggle(bool toggle) {
	cudaMemcpyToSymbol(cu::toggle, &toggle, sizeof(toggle));
}

}
