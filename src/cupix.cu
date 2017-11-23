#include <cstdio>
#include <iostream>
using namespace std;

#include "cupix.hpp"

#include <cuda_gl_interop.h>

namespace cupix {

namespace cu {

__constant__ __device__ int w, h;
__constant__ __device__ int n_triangle;

__global__
void Test(glm::vec3 c) {

}

__global__
void Clear(unsigned char *buffer, glm::vec3 color) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if(x >= w || y >= h) return;
	int i_thread = y * w + x;
	buffer[i_thread * 3 + 0] = glm::clamp(color.r, 0.f, 1.f) * 255;
	buffer[i_thread * 3 + 1] = glm::clamp(color.g, 0.f, 1.f) * 255;
	buffer[i_thread * 3 + 2] = glm::clamp(color.b, 0.f, 1.f) * 255;
}

__global__
void VertexShader(VertexIn *in, VertexOut *out, glm::mat4 *mvp) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	if(x >= n_triangle * 3) return;

	out[x].position = *mvp * glm::vec4(in[x].position, 1.f);
	out[x].normal = in[x].normal;
}

__global__
void GetAABB(VertexOut *in, AABB *aabb) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	if(x >= n_triangle) return;

	glm::vec2
		p1 = glm::vec2(in[x * 3 + 0].position.x, in[x * 3 + 0].position.y) / in[x * 3 + 0].position.w,
		p2 = glm::vec2(in[x * 3 + 1].position.x, in[x * 3 + 1].position.y) / in[x * 3 + 1].position.w,
		p3 = glm::vec2(in[x * 3 + 2].position.x, in[x * 3 + 2].position.y) / in[x * 3 + 2].position.w;
	p1 = (p1 * 0.5f + 0.5f) * glm::vec2(w, h);
	p2 = (p2 * 0.5f + 0.5f) * glm::vec2(w, h);
	p3 = (p3 * 0.5f + 0.5f) * glm::vec2(w, h);
	glm::vec2
		v_min = glm::min(glm::min(p1, p2), p3),
		v_max = glm::max(glm::max(p1, p2), p3);
	glm::ivec2
		c0 = glm::ivec2(0, 0),
		c1 = glm::ivec2(w - 1, h - 1),
		iv_min = glm::ceil(v_min),
		iv_max = glm::floor(v_max);

	iv_min = glm::clamp(iv_min, c0, c1);
	iv_max = glm::clamp(iv_max, c0, c1);

	aabb[x].v[0] = iv_min;
	aabb[x].v[1] = iv_max;
}

__device__
glm::vec3 GetBarycentric() {

}

__device__
void Interpolate(VertexOut *v, FragmentIn *f) {
	// glm::vec3 b = GetBarycentric(
	// 	v[0].position,
	// 	v[1].position,
	// 	v[2].position);
}

__global__
void Rasterize(VertexOut *v, float *depth_buf, unsigned char* frame_buf, int i, glm::ivec2 corner, glm::ivec2 dim) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if(x >= dim.x || y >= dim.y) return;
	x += corner.x;
	y += corner.y;
	if(x >= w || y >= h) return;
	int i_thread = y * w + x;

	if(v[i * 3 + 0].position.z < 0
	|| v[i * 3 + 1].position.z < 0
	|| v[i * 3 + 2].position.z < 0) return;
	glm::vec2
		p1 = glm::vec2(v[i * 3 + 0].position.x, v[i * 3 + 0].position.y) / v[i * 3 + 0].position.w,
		p2 = glm::vec2(v[i * 3 + 1].position.x, v[i * 3 + 1].position.y) / v[i * 3 + 1].position.w,
		p3 = glm::vec2(v[i * 3 + 2].position.x, v[i * 3 + 2].position.y) / v[i * 3 + 2].position.w;
	p1 = (p1 * 0.5f + 0.5f) * glm::vec2(w, h);
	p2 = (p2 * 0.5f + 0.5f) * glm::vec2(w, h);
	p3 = (p3 * 0.5f + 0.5f) * glm::vec2(w, h);

	glm::vec2 d12 = p1 - p2, d23 = p2 - p3, d31 = p3 - p1;
	float e1 = glm::dot(d12, glm::vec2(y - p1.y, p1.x - x));
	float e2 = glm::dot(d23, glm::vec2(y - p2.y, p2.x - x));
	float e3 = glm::dot(d31, glm::vec2(y - p3.y, p3.x - x));

	if(e1 <= 0 && e2 <= 0 && e3 <= 0) {
		FragmentIn fragment = {glm::ivec2(x, y)};
		Interpolate(v + i * 3, &fragment);
		glm::vec3 c = v[i * 3 + 0].normal + v[i * 3 + 1].normal + v[i * 3 + 2].normal;
		c = (c / 3.f + 1.f) * 0.5f * 255.f;
		frame_buf[i_thread * 3 + 0] = c.r;
		frame_buf[i_thread * 3 + 1] = c.g;
		frame_buf[i_thread * 3 + 2] = c.b;
	}

	// for(int i = 0; i < n_triangle; i++) {
	// 	if(in[i * 3 + 0].position.z < 0
	// 	|| in[i * 3 + 1].position.z < 0
	// 	|| in[i * 3 + 2].position.z < 0) continue;
	// 	// glm::vec2 p1(0.f, 0.f), p2(0.1f, 0.f), p3(0.f, 0.1f);
	// 	// p1 = (p1 * 0.5f + 0.5f) * glm::vec2(w, h);
	// 	// p2 = (p2 * 0.5f + 0.5f) * glm::vec2(w, h);
	// 	// p3 = (p3 * 0.5f + 0.5f) * glm::vec2(w, h);
	// 	glm::vec2
	// 		// p1(10.2f, 10.8f), p2(300.3f, 30.1f), p3(200.7f, 300.5f);
	// 		p1 = glm::vec2(in[i * 3 + 0].position.x, in[i * 3 + 0].position.y) / in[i * 3 + 0].position.w,
	// 		p2 = glm::vec2(in[i * 3 + 1].position.x, in[i * 3 + 1].position.y) / in[i * 3 + 1].position.w,
	// 		p3 = glm::vec2(in[i * 3 + 2].position.x, in[i * 3 + 2].position.y) / in[i * 3 + 2].position.w;
	// 	p1 = (p1 * 0.5f + 0.5f) * glm::vec2(w, h);
	// 	p2 = (p2 * 0.5f + 0.5f) * glm::vec2(w, h);
	// 	p3 = (p3 * 0.5f + 0.5f) * glm::vec2(w, h);

	// 	glm::vec2 d12 = p1 - p2, d23 = p2 - p3, d31 = p3 - p1;
	// 	glm::ivec2 p_min = min(min(p1, p2), p3);
	// 	glm::ivec2 p_max = max(max(p1, p2), p3);
	// 	float c1 = d12.y * p1.x - d12.x * p1.y;
	// 	float c2 = d23.y * p2.x - d23.x * p2.y;
	// 	float c3 = d31.y * p3.x - d31.x * p3.y;

	// 	float cy1 = c1 + d12.x * p_min.y - d12.y * p_min.x;
	// 	float cy2 = c2 + d23.x * p_min.y - d23.y * p_min.x;
	// 	float cy3 = c3 + d31.x * p_min.y - d31.y * p_min.x;

	// // for(int y = miny; y < maxy; y++)
	// // {
	// // 	// Start value for horizontal scan
	// // 	float Cx1 = Cy1;
	// // 	float Cx2 = Cy2;
	// // 	float Cx3 = Cy3;

	// // 	for(int x = minx; x < maxx; x++)
	// // 	{
	// // 		if(Cx1 > 0 && Cx2 > 0 && Cx3 > 0)
	// // 		{
	// // 			colorBuffer[x] = 0x00FFFFFF;<< // White
	// // 		}

	// // 		Cx1 -= Dy12;
	// // 		Cx2 -= Dy23;
	// // 		Cx3 -= Dy31;
	// // 	}

	// // 	Cy1 += Dx12;
	// // 	Cy2 += Dx23;
	// // 	Cy3 += Dx31;

	// // 	(char*&)colorBuffer += stride;
	// // }

	// 	float e1 = glm::dot(d12, glm::vec2(y - p1.y, p1.x - x));
	// 	float e2 = glm::dot(d23, glm::vec2(y - p2.y, p2.x - x));
	// 	float e3 = glm::dot(d31, glm::vec2(y - p3.y, p3.x - x));

	// 	if(e1 <= 0 && e2 <= 0 && e3 <= 0) {
	// 		buffer[i_thread * 3 + 0] = 0;
	// 		buffer[i_thread * 3 + 1] = 255;
	// 		buffer[i_thread * 3 + 2] = 0;
	// 	}
	// }

	// glm::vec2 fragCoord(x, y);
	// glm::vec2 iResolution(w, h);
	// glm::vec4 fragColor;

	// glm::vec2 uv = fragCoord - iResolution / 2.f;
	// float d = glm::dot(uv, uv);
	// //float d = sqrt(dot(uv, uv));
	// fragColor = glm::vec4(0.5f + 0.5f * cos(d / 5.f + 10.f));

	// buffer[i_thread * 3 + 0] = fragColor.x * 255;
	// buffer[i_thread * 3 + 1] = fragColor.y * 255;
	// buffer[i_thread * 3 + 2] = fragColor.z * 255;
}

}



CUPix::CUPix(int window_w, int window_h, GLuint pbo) : window_w_(window_w), window_h_(window_h) {
	cudaMalloc(&depth_buf_, sizeof(float) * window_w_ * window_h_);
	cudaMalloc(&frame_buf_, sizeof(float) * window_w_ * window_h_);
	cudaMalloc(&mvp_buf_, sizeof(glm::mat4));
	cudaGraphicsGLRegisterBuffer(&pbo_resource_, pbo, cudaGraphicsMapFlagsNone);
	cudaMemcpyToSymbol(cu::w, &window_w_, sizeof(int));
	cudaMemcpyToSymbol(cu::h, &window_h_, sizeof(int));
}

CUPix::~CUPix() {
	cudaGraphicsUnregisterResource(pbo_resource_);
}

void CUPix::MapResources() {
	size_t size;
	cudaGraphicsMapResources(1, &pbo_resource_, NULL);
	cudaGraphicsResourceGetMappedPointer((void**)&pbo_ptr_, &size, pbo_resource_);
}

void CUPix::UnmapResources() {
	cudaGraphicsUnmapResources(1, &pbo_resource_, NULL);
}

void CUPix::ClearColor(float r, float g, float b, float a) {
	clear_color_ = glm::vec4(r, g, b, a);
}

void CUPix::Clear() {
	cu::Clear<<<
		dim3((window_w_-1)/32+1, (window_h_-1)/32+1),
		dim3(32, 32)>>>(
			pbo_ptr_,
			glm::vec3(clear_color_.r, clear_color_.g, clear_color_.b));
}

void CUPix::Draw() {
	cu::VertexShader<<<(n_triangle_*3-1)/32+1, 32>>>(vertex_in_, vertex_out_, mvp_buf_);
	cu::GetAABB<<<(n_triangle_-1)/32+1, 32>>>(vertex_out_, aabb_buf_);
	cudaMemcpy(aabb_, aabb_buf_, sizeof(AABB) * n_triangle_, cudaMemcpyDeviceToHost);
	for(int i = 0; i < n_triangle_; i++) {
		glm::ivec2 dim = aabb_[i].v[1] - aabb_[i].v[0] + 1;
		cu::Rasterize<<<
			dim3((dim.x-1)/4+1, (dim.y-1)/8+1),
			dim3(4, 8)>>>
			(vertex_out_, depth_buf_, pbo_ptr_, i, aabb_[i].v[0], dim);
		// FragmentShader<<<>>>();
	}
}

void CUPix::VertexData(int size, float *position, float *normal) {
	n_vertex_ = size / 3;
	n_triangle_ = n_vertex_ / 3;
	VertexIn v[n_vertex_];
	for(int i = 0; i < n_vertex_; i++) {
		v[i].position = glm::vec3(position[i * 3], position[i * 3 + 1], position[i * 3 + 2]);
		v[i].normal = glm::vec3(normal[i * 3], normal[i * 3 + 1], normal[i * 3 + 2]);
		v[i].color = glm::vec3(1.f, 1.f, 1.f);
	}
	cudaMalloc(&vertex_in_, sizeof(v));
	cudaMemcpy(vertex_in_, v, sizeof(v), cudaMemcpyHostToDevice);
	cudaMalloc(&aabb_buf_, sizeof(AABB) * n_triangle_);
	aabb_ = new AABB[n_triangle_];
	cudaMalloc(&vertex_out_, sizeof(VertexOut) * n_vertex_);
	cudaMemcpyToSymbol(cu::n_triangle, &n_triangle_, sizeof(int));
}

void CUPix::MVP(glm::mat4 &mvp) {
	cudaMemcpy(mvp_buf_, &mvp, sizeof(glm::mat4), cudaMemcpyHostToDevice);
}

}
