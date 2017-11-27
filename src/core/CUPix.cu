#include "CUPix.hpp"

#include <cstdio>
#include <cuda_gl_interop.h>

namespace cupix {

namespace core {

texture<uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat> texture;
__constant__ __device__ int w, h;
__constant__ __device__ int n_light;
__constant__ __device__ Light light[4];
__constant__ __device__ float mvp[16];
__constant__ __device__ float mv[16];
__constant__ __device__ float time;
__constant__ __device__ bool toggle;

__constant__ __device__ int n_triangle;
__constant__ __device__ bool depth_test = true;
__constant__ __device__ bool blend = false;
__constant__ __device__ unsigned char clear_color[4];

const int bitmap_size = 128 * 32; // 128 characters, each 32 bytes
__constant__ __device__ char bitmap[bitmap_size];

__global__ void Clear(unsigned char *frame_buf, float *depth_buf);
__global__ void NormalSpace(VertexIn *in, VertexOut *out, Vertex *v);
__global__ void WindowSpace(Vertex *v);
__global__ void AssemTriangle(Vertex *v, Triangle *triangle);
__global__ void Rasterize(glm::ivec2 corner, glm::ivec2 dim, Vertex *v, VertexOut *va, float *depth_buf, unsigned char* frame_buf);
__global__ void RasterizeMSAA(glm::ivec2 corner, glm::ivec2 dim, Vertex *v, VertexOut *va, float *depth_buf, unsigned char* frame_buf);
__global__ void DrawCharater(int ch, int x0, int y0, bool ssaa, unsigned char *frame_buf);
__global__ void DownSample(unsigned char *frame_buf, unsigned char *pbo_buf);

}


CUPix::CUPix(int window_w, int window_h, GLuint pbo, AA aa = NOAA, bool record = false)
	: window_w_(window_w), window_h_(window_h), frame_w_(window_w), frame_h_(window_h)
	, aa_(aa), record_(record) {
	if(aa_ != NOAA) {
		frame_w_ *= 2;
		frame_h_ *= 2;
	}
	frame_ = new unsigned char[window_w_ * window_h_ * 3];
	cudaMalloc(&depth_buf_, sizeof(float) * frame_w_ * frame_h_);
	cudaMalloc(&frame_buf_, frame_w_ * frame_h_ * 3);
	cudaMemcpyToSymbol(core::w, &frame_w_, sizeof(int));
	cudaMemcpyToSymbol(core::h, &frame_h_, sizeof(int));
	cudaGraphicsGLRegisterBuffer(&pbo_resource_, pbo, cudaGraphicsMapFlagsNone);

	// load bitmap font into GPU memory
	FILE *font_file = fopen("../font/bitmap_font.data", "rb");
	char bitmap[core::bitmap_size];
	size_t r = fread(bitmap, 1, core::bitmap_size, font_file);
	fclose(font_file);
	cudaMemcpyToSymbol(core::bitmap, bitmap, core::bitmap_size);
}

CUPix::~CUPix() {
	delete[] triangle_;
	delete[] frame_;
	cudaFree(depth_buf_);
	cudaFree(frame_buf_);
	cudaGraphicsUnregisterResource(pbo_resource_);
}

void CUPix::MapResources() {
	size_t size;
	cudaGraphicsMapResources(1, &pbo_resource_, NULL);
	cudaGraphicsResourceGetMappedPointer((void**)&pbo_buf_, &size, pbo_resource_);
}

void CUPix::UnmapResources() {
	if(aa_ != NOAA) core::DownSample<<<dim3((window_w_-1)/32+1, (window_h_-1)/32+1), dim3(32, 32)>>>(frame_buf_, pbo_buf_);
	else cudaMemcpy(pbo_buf_, frame_buf_, window_w_ * window_h_ * 3, cudaMemcpyDeviceToDevice);
	if(record_) cudaMemcpy(frame_, pbo_buf_, window_w_ * window_h_ * 3, cudaMemcpyDeviceToHost);
	cudaGraphicsUnmapResources(1, &pbo_resource_, NULL);
}

void CUPix::Enable(Flag flag) {
	bool b = true;
	switch(flag) {
		case DEPTH_TEST:
			cudaMemcpyToSymbol(core::depth_test, &b, 1); return;
		case BLEND:
			cudaMemcpyToSymbol(core::blend, &b, 1); return;
		case CULL_FACE:
			cull_ = b; return;
	}
}

void CUPix::Disable(Flag flag) {
	bool b = false;
	switch(flag) {
		case DEPTH_TEST:
			cudaMemcpyToSymbol(core::depth_test, &b, 1); return;
		case BLEND:
			cudaMemcpyToSymbol(core::blend, &b, 1); return;
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
	cudaMemcpyToSymbol(core::clear_color, &clear_color, 4);
}

void CUPix::Clear() {
	core::Clear<<<dim3((frame_w_-1)/32+1, (frame_h_-1)/32+1), dim3(32, 32)>>>(frame_buf_, depth_buf_);
}

void CUPix::Draw() {
	core::NormalSpace<<<(n_triangle_*3-1)/32+1, 32>>>(vertex_in_, vertex_out_, vertex_buf_);
	core::WindowSpace<<<(n_triangle_*3-1)/32+1, 32>>>(vertex_buf_);
	core::AssemTriangle<<<(n_triangle_-1)/32+1, 32>>>(vertex_buf_, triangle_buf_);
	cudaMemcpy(triangle_, triangle_buf_, sizeof(Triangle) * n_triangle_, cudaMemcpyDeviceToHost);
	for(int i = 0; i < n_triangle_; i++)
		if(!triangle_[i].empty)
			if(!cull_ || (cull_face_ != FRONT_AND_BACK
			&& (triangle_[i].winding == front_face_ != cull_face_))) {
				glm::ivec2 dim = triangle_[i].v[1] - triangle_[i].v[0] + 1;
				core::Rasterize<<<dim3((dim.x-1)/4+1, (dim.y-1)/8+1), dim3(4, 8)>>>
					(triangle_[i].v[0], dim, vertex_buf_ + i * 3, vertex_out_ + i * 3, depth_buf_, frame_buf_);
		}
}

void CUPix::DrawFPS(int fps) {
	bool aa = aa_ != NOAA;
	core::DrawCharater<<<1, dim3(16, 16)>>>('F',  0, 0, aa, frame_buf_);
	core::DrawCharater<<<1, dim3(16, 16)>>>('P', 16, 0, aa, frame_buf_);
	core::DrawCharater<<<1, dim3(16, 16)>>>('S', 32 - 3, 0, aa, frame_buf_);
	core::DrawCharater<<<1, dim3(16, 16)>>>(fps % 1000 / 100 + 48, 48 + 5, 0, aa, frame_buf_);
	core::DrawCharater<<<1, dim3(16, 16)>>>(fps % 100 / 10   + 48, 64 + 5, 0, aa, frame_buf_);
	core::DrawCharater<<<1, dim3(16, 16)>>>(fps % 10         + 48, 80 + 5, 0, aa, frame_buf_);
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
	cudaFree(vertex_in_);
	cudaMalloc(&vertex_in_, sizeof(v));
	cudaMemcpy(vertex_in_, v, sizeof(v), cudaMemcpyHostToDevice);
	cudaFree(vertex_out_);
	cudaMalloc(&vertex_out_, sizeof(VertexOut) * n_vertex_);
	cudaFree(vertex_buf_);
	cudaMalloc(&vertex_buf_, sizeof(Vertex) * n_vertex_);
	cudaFree(triangle_buf_);
	cudaMalloc(&triangle_buf_, sizeof(Triangle) * n_triangle_);
	delete[] triangle_;
	triangle_ = new Triangle[n_triangle_];
	cudaMemcpyToSymbol(core::n_triangle, &n_triangle_, sizeof(int));
}

void CUPix::MVP(glm::mat4 &mvp) {
	cudaMemcpyToSymbol(core::mvp, &mvp, sizeof(glm::mat4));
}

void CUPix::MV(glm::mat4 &mv) {
	cudaMemcpyToSymbol(core::mv, &mv, sizeof(glm::mat4));
}

void CUPix::Time(float time) {
	cudaMemcpyToSymbol(core::time, &time, sizeof(float));
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
	cudaFree(texture_buf_);
	cudaMallocPitch((void**)&texture_buf_, &pitch, w * 4, h);
	cudaMemcpy2D(
		texture_buf_, pitch,
		data, w * 4,
		w * 4, h,
		cudaMemcpyHostToDevice);

	core::texture.normalized = true;
	core::texture.sRGB = gamma_correction;
	core::texture.filterMode = cudaFilterModeLinear;
	core::texture.addressMode[0] = cudaAddressModeWrap;
	core::texture.addressMode[1] = cudaAddressModeWrap;
	cudaChannelFormatDesc desc =
		cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);

	cudaBindTexture2D(NULL, core::texture, texture_buf_, desc, w, h, pitch);
}

void CUPix::Lights(int n, Light *light) {
	cudaMemcpyToSymbol(core::n_light, &n, sizeof(int));
	cudaMemcpyToSymbol(core::light, light, sizeof(Light) * n);
}

void CUPix::Toggle(bool toggle) {
	cudaMemcpyToSymbol(core::toggle, &toggle, sizeof(toggle));
}

}
