#include "GL/glew.h"
#include "GLFW/glfw3.h"

#include "CUPix.hpp"
using namespace cupix;
#include "Model.hpp"
#include "Texture.hpp"
#include "Camera.hpp"
#include "FPS.hpp"
#include "Toggle.hpp"
#include "Video.hpp"

GLFWwindow* InitGLFW(int window_w, int window_h) {
	if(!glfwInit()) exit(EXIT_FAILURE);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);
	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
	glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);

	GLFWwindow *window = glfwCreateWindow(window_w, window_h, "CUPix", NULL, NULL);
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
	// glfwSwapInterval(1);
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
	if(argc < 2) {
		printf("Usage: cupix input_obj_file [output_video_file]\n");
		return 0;
	}
	bool record = false;
	if(argc == 3) record = true;

	int window_w = 1280;
	int window_h = 720;
	// int window_w = 640;
	// int window_h = 360;

	GLFWwindow* window = InitGLFW(window_w, window_h);
	GLuint pbo = InitGL(window_w, window_h);

	CUPix pix(window_w, window_h, pbo, NOAA, record);
	pix.ClearColor(0.08f, 0.16f, 0.24f, 1.f);
	// pix.Enable(CULL_FACE);
	// pix.CullFace(BACK);
	// pix.FrontFace(CW);

	Light light[2]{
		 5.f, 4.f, 3.f, // position
		 1.f, 1.f, 1.f, // color
		20.f,           // power
		-5.f, 4.f, 3.f, // position
		 1.f, 1.f, 1.f, // color
		30.f,           // power
	};
	pix.Lights(2, light);

	Model model(argv[1]);
	pix.VertexData(model.n_vertex(), model.vertex(), model.normal(), model.uv());

	Texture texture("texture/texture.jpg");
	pix.Texture(texture.data(), texture.w(), texture.h(), false); // gamma_correction = false

	double time = glfwGetTime();
	Camera camera(window, window_w, window_h, time);
	FPS fps(time);
	Toggle toggle(window, GLFW_KEY_T, true); // init state = true
	Video video(window_w, window_h);
	while(glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS && !glfwWindowShouldClose(window)) {
		time = glfwGetTime();

		pix.BeforeDraw();
		pix.Clear();

		glm::mat4 m;
		glm::mat4 vp = camera.Update(time);
		glm::mat4 mvp = vp * m;
		pix.MVP(mvp);
		glm::mat4 v = camera.v();
		glm::mat4 mv = v * m;
		pix.MV(mv);

		pix.Time(time);
		pix.Toggle(toggle.Update([] {
			printf("\nUse Blinn-Phong shading\n");
		}, [] {
			printf("\nUse Phong shading\n");
		}));

		pix.Draw();
		pix.DrawFPS(fps.Update(time) + 0.5f);
		pix.AfterDraw();

		UpdateGL(window, window_w, window_h);
		if(record) video.Add(pix.frame());
	}
	fps.Term();

	TermGL(pbo);
	TermGLFW(window);

	if(record) video.Save(argv[2]);
}
