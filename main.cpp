#include "cupix.hpp"

using namespace cupix;

void GLFWErrorCallback(int error, const char* description) {
	fprintf(stderr, "Error: %s\n", description);
}

GLFWwindow* InitGLFW(int window_w, int window_H) {
	glfwSetErrorCallback(GLFWErrorCallback);
	if(!glfwInit()) exit(EXIT_FAILURE);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);
	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
	glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);

	GLFWwindow *window = glfwCreateWindow(window_w, window_H, "CUPix", NULL, NULL);
	if(!window) {
		glfwTerminate();
		exit(EXIT_FAILURE);
	}
	glfwMakeContextCurrent(window);
	if(glewInit() != GLEW_OK) {
		glfwTerminate();
		fprintf(stderr, "Failed to initialize GLEW\n");
		exit(EXIT_FAILURE);
	}
	glfwSwapInterval(1);
	return window;
}

void TermGLFW(GLFWwindow *window, GLuint buffer) {
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	glDeleteBuffers(1, &buffer);
	glfwDestroyWindow(window);
	glfwTerminate();
}

GLuint InitGL(int window_w, int window_h) {
	GLuint buffer;
	glGenBuffers(1, &buffer);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, buffer);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, window_w * window_h * 3, NULL, GL_DYNAMIC_DRAW);
	return buffer;
}

void UpdateGL(GLFWwindow *window, int window_w, int window_h) {
	glDrawPixels(window_w, window_h, GL_RGB, GL_UNSIGNED_BYTE, 0);
	glfwSwapBuffers(window);
	glfwPollEvents();
}

int main(int argc, char *argv[]) {
	int window_w = 1280;
	int window_h = 720;

	GLFWwindow* window = InitGLFW(window_w, window_h);
	GLuint buffer = InitGL(window_w, window_h);

	CUPix pix(window_w, window_h, buffer);

	FPS fps;
	while(glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS && !glfwWindowShouldClose(window)) {
		pix.MapResources();
		pix.Clear();


		pix.Draw();
		pix.UnmapResources();

		UpdateGL(window, window_w, window_h);
		fps.Update();
	}
}
