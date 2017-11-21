#include "cupix.hpp"

namespace cupix {

CUPix::CUPix(int window_w, int window_h, GLuint buffer) : window_w_(window_w), window_h_(window_h) {
	cudaGraphicsGLRegisterBuffer(&buffer_resource_, buffer, cudaGraphicsMapFlagsNone);
}

CUPix::~CUPix() {
	cudaGraphicsUnregisterResource(buffer_resource_);
}

void CUPix::MapResources() {
	size_t size;
	cudaGraphicsMapResources(1, &buffer_resource_, NULL);
	cudaGraphicsResourceGetMappedPointer((void**)&buffer_ptr_, &size, buffer_resource_);
}

void CUPix::UnmapResources() {
	cudaGraphicsUnmapResources(1, &buffer_resource_, NULL);
}

void CUPix::Clear() {
	cu::Clear<<<
		dim3((window_w_-1)/32+1, (window_h_-1)/32+1),
		dim3(32, 32)>>>
		(buffer_ptr_, window_w_, window_h_);
}

void CUPix::Draw() {
	cu::T1<<<
		dim3((window_w_-1)/32+1, (window_h_-1)/32+1),
		dim3(32, 32)>>>
		(buffer_ptr_, window_w_, window_h_, glfwGetTime());
}

void FPS::Update() {
	if(c_frame_++ % 10 == 0) {
		time_new_ = glfwGetTime();
		double fps = 10 / (time_new_ - time_old_);
		time_old_ = time_new_;
		printf("FPS: %lf\n", fps);
	}
}

namespace cu {

__global__
void Clear(unsigned char *buffer, int w, int h) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if(x >= w || y >= h) return;
	int i_thread = y * w + x;
	buffer[i_thread * 3 + 0] = 0;
	buffer[i_thread * 3 + 1] = 0;
	buffer[i_thread * 3 + 2] = 0;
}

__global__
void T1(unsigned char* buffer, int width, int height, double iTime) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if(x >= width || y >= height) return;
	int i_thread = y * width + x;

	glm::vec2 p1(10.2f, 10.8f), p2(300.3f, 30.1f), p3(200.7f, 300.5f);

	glm::vec2 d12 = p1 - p2, d23 = p2 - p3, d31 = p3 - p1;
	glm::ivec2 p_min = min(min(p1, p2), p3);
	glm::ivec2 p_max = max(max(p1, p2), p3);
	float c1 = d12.y * p1.x - d12.x * p1.y;
	float c2 = d23.y * p2.x - d23.x * p2.y;
	float c3 = d31.y * p3.x - d31.x * p3.y;

	float cy1 = c1 + d12.x * p_min.y - d12.y * p_min.x;
	float cy2 = c2 + d23.x * p_min.y - d23.y * p_min.x;
	float cy3 = c3 + d31.x * p_min.y - d31.y * p_min.x;

	// for(int y = miny; y < maxy; y++)
	// {
	// 	// Start value for horizontal scan
	// 	float Cx1 = Cy1;
	// 	float Cx2 = Cy2;
	// 	float Cx3 = Cy3;

	// 	for(int x = minx; x < maxx; x++)
	// 	{
	// 		if(Cx1 > 0 && Cx2 > 0 && Cx3 > 0)
	// 		{
	// 			colorBuffer[x] = 0x00FFFFFF;<< // White
	// 		}

	// 		Cx1 -= Dy12;
	// 		Cx2 -= Dy23;
	// 		Cx3 -= Dy31;
	// 	}

	// 	Cy1 += Dx12;
	// 	Cy2 += Dx23;
	// 	Cy3 += Dx31;

	// 	(char*&)colorBuffer += stride;
	// }

	float e1 = glm::dot(d12, glm::vec2(y - p1.y, p1.x - x));
	float e2 = glm::dot(d23, glm::vec2(y - p2.y, p2.x - x));
	float e3 = glm::dot(d31, glm::vec2(y - p3.y, p3.x - x));

	glm::vec2 fragCoord(x, y);
	glm::vec2 iResolution(width, height);
	glm::vec4 fragColor;

	glm::vec2 uv = fragCoord - iResolution / 2.f;
	float d = glm::dot(uv, uv);
	//float d = sqrt(dot(uv, uv));
	fragColor = glm::vec4(0.5f + 0.5f * cos(d / 5.f + iTime * 10.f));

	if(e1 <= 0 && e2 <= 0 && e3 <= 0) {
		// buffer[i_thread * 3 + 0] = fragColor.x * 255;
		// buffer[i_thread * 3 + 1] = fragColor.y * 255;
		// buffer[i_thread * 3 + 2] = fragColor.z * 255;
		buffer[i_thread * 3 + 0] = 0;
		buffer[i_thread * 3 + 1] = 255;
		buffer[i_thread * 3 + 2] = 0;
	}
}

}

}
