#include <iostream>
using namespace std;

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "cupix.hpp"
#include "util/model.hpp"
#include "util/texture.hpp"
#include "util/camera.hpp"
#include "util/fps.hpp"

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

void TermGLFW(GLFWwindow *window) {
	glfwDestroyWindow(window);
	glfwTerminate();
}

GLuint InitGL(int window_w, int window_h) {
	GLuint pbo;
	glGenBuffers(1, &pbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, window_w * window_h * 3, NULL, GL_DYNAMIC_COPY);
	return pbo;
}

void TermGL(GLuint pbo) {
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	glDeleteBuffers(1, &pbo);
}

void UpdateGL(GLFWwindow *window, int window_w, int window_h) {
	glDrawPixels(window_w, window_h, GL_RGB, GL_UNSIGNED_BYTE, 0);
	glfwSwapBuffers(window);
	glfwPollEvents();
}

int main(int argc, char *argv[]) {
	if(argc != 2) {
		printf("Usage: cupix obj_file\n");
		return 0;
	}

	int window_w = 1280;
	int window_h = 720;

	GLFWwindow* window = InitGLFW(window_w, window_h);
	GLuint pbo = InitGL(window_w, window_h);

	CUPix pix(window_w, window_h, pbo);
	pix.ClearColor(0.08f, 0.16f, 0.24f, 1.f);

	Model model(argv[1]);
	pix.VertexData(model.n_vertex(), model.vertex(), model.normal(), model.uv());

	Texture texture("../texture/uv.png");
	pix.Texture(texture.data(), texture.w(), texture.h());

	Camera camera(window, window_w, window_h);
	FPS fps;
	while(glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS && !glfwWindowShouldClose(window)) {
		pix.MapResources();
		pix.Clear();

		glm::mat4 m;
		glm::mat4 v = camera.v();
		glm::mat4 p = camera.p();
		glm::mat4 mvp = p * v * m;
		pix.MVP(mvp);

		pix.Draw();
		pix.UnmapResources();

		UpdateGL(window, window_w, window_h);
		camera.Update();
		fps.Update();
	}

	TermGL(pbo);
	TermGLFW(window);
}
