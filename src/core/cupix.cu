#include <cstdio>
#include <iostream>
using namespace std;

#include "cupix.hpp"

#include <cuda_gl_interop.h>

namespace cupix {

namespace cu {

__constant__ __device__ int w, h;
__constant__ __device__ int n_triangle;
texture<uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat> texture;

extern __device__
void VertexShader(VertexIn &in, VertexOut &out, glm::mat4 &mvp);

extern __device__
void FragmentShader(FragmentIn &in, glm::vec4 &color);

__global__
void Clear(unsigned char *frame_buf, float *depth_buf, glm::vec3 color) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if(x >= w || y >= h) return;
	int i_thread = y * w + x;
	frame_buf[i_thread * 3 + 0] = glm::clamp(color.r, 0.f, 1.f) * 255;
	frame_buf[i_thread * 3 + 1] = glm::clamp(color.g, 0.f, 1.f) * 255;
	frame_buf[i_thread * 3 + 2] = glm::clamp(color.b, 0.f, 1.f) * 255;
	depth_buf[i_thread] = 0;
}

__global__
void NormalSpace(VertexIn *in, VertexOut *out, glm::mat4 *mvp) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	if(x >= n_triangle * 3) return;

	VertexShader(in[x], out[x], *mvp);
}

__global__
void WindowSpace(VertexOut *v) {
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
void GetAABB(VertexOut *v, AABB *aabb) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	if(x >= n_triangle) return;

	glm::vec2
		p1(v[x * 3 + 0].position.x, v[x * 3 + 0].position.y),
		p2(v[x * 3 + 1].position.x, v[x * 3 + 1].position.y),
		p3(v[x * 3 + 2].position.x, v[x * 3 + 2].position.y);
	glm::vec2
		v_min = glm::min(glm::min(p1, p2), p3),
		v_max = glm::max(glm::max(p1, p2), p3);
	glm::ivec2
		c0 = glm::ivec2(0, 0),
		c1 = glm::ivec2(w - 1, h - 1),
		iv_min = v_min + 0.5f,
		iv_max = v_max + 0.5f;

	iv_min = glm::clamp(iv_min, c0, c1);
	iv_max = glm::clamp(iv_max, c0, c1);

	aabb[x].v[0] = iv_min;
	aabb[x].v[1] = iv_max;
}

__device__
void Interpolate(VertexOut *v, FragmentIn *f, glm::vec3 e) {
	glm::vec3 d(
		e.x / v[0].position.w,
		e.y / v[1].position.w,
		e.z / v[2].position.w);
	f->depth = 1 / (d.x + d.y + d.z);
	f->normal = (
		v[0].normal * d.x +
		v[1].normal * d.y +
		v[2].normal * d.z) * f->depth;
	f->color = (
		v[0].color * d.x +
		v[1].color * d.y +
		v[2].color * d.z) * f->depth;
	f->uv = (
		v[0].uv * d.x +
		v[1].uv * d.y +
		v[2].uv * d.z) * f->depth;
	f->z =
		e.x * v[0].position.z +
		e.y * v[1].position.z +
		e.z * v[2].position.z;
}

__global__
void Rasterize(VertexOut *v, float *depth_buf, unsigned char* frame_buf, glm::ivec2 corner, glm::ivec2 dim) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if(x >= dim.x || y >= dim.y) return;
	x += corner.x;
	y += corner.y;
	if(x >= w || y >= h) return;
	int i_thread = y * w + x;

	// if(v[0].position.z >  1 || v[1].position.z >  1 || v[2].position.z >  1
	// || v[0].position.z < -1 || v[1].position.z < -1 || v[2].position.z < -1)
	// 	return;
	glm::vec2
		p0 = glm::vec2(v[0].position.x, v[0].position.y),
		p1 = glm::vec2(v[1].position.x, v[1].position.y),
		p2 = glm::vec2(v[2].position.x, v[2].position.y);

	glm::vec2 d01 = p1 - p0, d12 = p2 - p1, d20 = p0 - p2;
	float e0 = glm::dot(d12, glm::vec2(y + 0.5f - p1.y, p1.x - x - 0.5f));
	float e1 = glm::dot(d20, glm::vec2(y + 0.5f - p2.y, p2.x - x - 0.5f));
	float e2 = glm::dot(d01, glm::vec2(y + 0.5f - p0.y, p0.x - x - 0.5f));

	if(e0 >= 0 && e1 >= 0 && e2 >= 0) {
		FragmentIn fragment = {glm::ivec2(x, y)};
		float e = e0 + e1 + e2;
		Interpolate(v, &fragment, glm::vec3(e0, e1, e2) / e);
		if(fragment.z > 1 || fragment.z < -1) return;
		if(1 - fragment.z > depth_buf[i_thread]) {
			depth_buf[i_thread] = 1 - fragment.z;
			glm::vec4 color;
			FragmentShader(fragment, color);
			glm::ivec4 icolor = color * 255.f;
			icolor = glm::clamp(icolor, glm::ivec4(0), glm::ivec4(255));
			frame_buf[i_thread * 3 + 0] = icolor.r;
			frame_buf[i_thread * 3 + 1] = icolor.g;
			frame_buf[i_thread * 3 + 2] = icolor.b;
		}
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
	cudaMemset(depth_buf_, 0, sizeof(float) * window_w_ * window_h_);
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
			depth_buf_,
			glm::vec3(clear_color_.r, clear_color_.g, clear_color_.b));
}

void CUPix::Draw() {
	cu::NormalSpace<<<(n_triangle_*3-1)/32+1, 32>>>(vertex_in_, vertex_out_, mvp_buf_);
	cu::WindowSpace<<<(n_triangle_*3-1)/32+1, 32>>>(vertex_out_);
	cu::GetAABB<<<(n_triangle_-1)/32+1, 32>>>(vertex_out_, aabb_buf_);
	cudaMemcpy(aabb_, aabb_buf_, sizeof(AABB) * n_triangle_, cudaMemcpyDeviceToHost);
	for(int i = 0; i < n_triangle_; i++) {
		glm::ivec2 dim = aabb_[i].v[1] - aabb_[i].v[0] + 1;
		cu::Rasterize<<<dim3((dim.x-1)/4+1, (dim.y-1)/8+1), dim3(4, 8)>>>
			(vertex_out_ + i * 3, depth_buf_, pbo_ptr_, aabb_[i].v[0], dim);
	}
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
	cudaMalloc(&aabb_buf_, sizeof(AABB) * n_triangle_);
	aabb_ = new AABB[n_triangle_];
	cudaMalloc(&vertex_out_, sizeof(VertexOut) * n_vertex_);
	cudaMemcpyToSymbol(cu::n_triangle, &n_triangle_, sizeof(int));
}

void CUPix::MVP(glm::mat4 &mvp) {
	cudaMemcpy(mvp_buf_, &mvp, sizeof(glm::mat4), cudaMemcpyHostToDevice);
}

void CUPix::Texture(unsigned char *d, int w, int h) {
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
	cu::texture.sRGB = true;
	cu::texture.filterMode = cudaFilterModeLinear;
	cu::texture.addressMode[0] = cudaAddressModeWrap;
	cu::texture.addressMode[1] = cudaAddressModeWrap;
	cudaChannelFormatDesc desc =
		cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);

	cudaBindTexture2D(NULL, cu::texture, texture_buf_, desc, w, h, pitch);
}

}
