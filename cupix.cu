#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <opencv2/opencv.hpp>
using namespace cv;
#include <cstdlib>
#include <cstdio>
#include <cuda_gl_interop.h>
#include <glm/glm.hpp>
using namespace glm;

#define PI 3.14159265358979323846

static void error_callback(int error, const char* description) {
	fprintf(stderr, "Error: %s\n", description);
}

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
		glfwSetWindowShouldClose(window, GLFW_TRUE);
}

__global__
void mainImage(unsigned char* buffer, int width, int height, double iTime) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if(x >= width || y >= height) return;
	int i_thread = y * width + x;
	vec2 fragCoord(x, y);
	vec2 iResolution(width, height);
	vec4 fragColor;



	vec2 uv = fragCoord - iResolution / 2.f;
	float d = dot(uv, uv);
	//float d = sqrt(dot(uv, uv));
	fragColor = vec4(0.5f + 0.5f * cos(d / 5.f + iTime * 10.f));



	buffer[i_thread * 3 + 0] = fragColor.x * 255;
	buffer[i_thread * 3 + 1] = fragColor.y * 255;
	buffer[i_thread * 3 + 2] = fragColor.z * 255;
}

__global__
void Julia1(unsigned char* buffer, int width, int height, double time) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if(x >= width || y >= height) return;
	int i_thread = y * width + x;

	vec2 f = (vec2(x, y) * 2.f - vec2(width, height)) / (float)height;
	float fx = f.x;
	float fy = f.y;

	// float fx = (x * 2.f - width) / height;
	// float fy = (y * 2.f - height) / height;
	float c_r = sin(time), c_v = cos(time);
	float z_r = fx, z_v = fy;
	float e = 0;
	int k;
	int n = 100;
	for(k = 0; k  < n; k++) {
		if(e > 4) break;
		float temp = z_r * z_r - z_v * z_v + c_r;
		z_v = 2 * z_r * z_v + c_v;
		z_r = temp;
		e = z_r * z_r + z_v * z_v;
	}
	buffer[i_thread * 3 + 0] = (1 - k * 1.0 / n) * 255;
	buffer[i_thread * 3 + 1] = (1 - k * 1.0 / n) * 255;
	buffer[i_thread * 3 + 2] = (1 - k * 1.0 / n) * 255;
}

__global__
void Ripple1(unsigned char* buffer, int width, int height, double time) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if(x >= width || y >= height) return;
	int i_thread = y * width + x;

	float fx = x  - width / 2;
	float fy = y  - height / 2;
	float d = fx * fx + fy * fy;
	unsigned char grey = (unsigned char)(128.f + 127.f * cos(d / 5.f + time * 10.f));

	buffer[i_thread * 3 + 0] = grey;
	buffer[i_thread * 3 + 1] = grey;
	buffer[i_thread * 3 + 2] = grey;
}

__global__
void Ripple3(unsigned char* buffer, int width, int height, double time) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if(x >= width || y >= height) return;
	int i_thread = y * width + x;

	float fx = x  - width / 2;
	float fy = y  - height / 2;
	float d = fx * fx + fy * fy;
	unsigned char r = (unsigned char)(128.f + 127.f * cos(d / 5.f + time * 10.f));
	unsigned char g = (unsigned char)(128.f + 127.f * cos(d / 5.f + time * 10.f + PI / 6.f));
	unsigned char b = (unsigned char)(128.f + 127.f * cos(d / 5.f + time * 10.f + PI / 3.f));

	buffer[i_thread * 3 + 0] = r;
	buffer[i_thread * 3 + 1] = g;
	buffer[i_thread * 3 + 2] = b;
}

int main(void) {
	int width = 1280;
	int height = 720;
	Mat image(height, width, CV_8UC3);

	GLFWwindow* window;

	glfwSetErrorCallback(error_callback);
	if(!glfwInit()) exit(EXIT_FAILURE);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);
	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
	glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);
//	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	window = glfwCreateWindow(width, height, "Simple example", NULL, NULL);
	if(!window) {
		glfwTerminate();
		exit(EXIT_FAILURE);
	}
	glfwSetKeyCallback(window, key_callback);
	glfwMakeContextCurrent(window);
	if(glewInit() != GLEW_OK) {
		fprintf(stderr, "Failed to initialize GLEW\n");
		return -1;
	}
	glfwSwapInterval(1);

	GLuint buffer;
	glGenBuffers(1, &buffer);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, buffer);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 3, image.data, GL_DYNAMIC_DRAW);
	glRasterPos2f(-1, 1);
	glPixelZoom(1, -1);

	cudaGraphicsResource *resource;
	cudaGraphicsGLRegisterBuffer(&resource, buffer, cudaGraphicsMapFlagsNone);
	unsigned char *dev_ptr;
	size_t size;


	while(!glfwWindowShouldClose(window)) {
		double time = glfwGetTime();
		cudaGraphicsMapResources(1, &resource, NULL);
		cudaGraphicsResourceGetMappedPointer((void**)&dev_ptr, &size, resource);
		mainImage<<<dim3((width-1)/32+1, (height-1)/32+1), dim3(32, 32)>>>(dev_ptr, width, height, time);
		cudaGraphicsUnmapResources(1, &resource, NULL);

		glClear(GL_COLOR_BUFFER_BIT);
		glDrawPixels(width, height, GL_RGB, GL_UNSIGNED_BYTE, 0);
		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	cudaGraphicsUnregisterResource(resource);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	glDeleteBuffers(1, &buffer);
	glfwDestroyWindow(window);
	glfwTerminate();
	exit(EXIT_SUCCESS);
}
