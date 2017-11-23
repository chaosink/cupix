#pragma once

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

class Camera {
	const double PI  = 3.14159265358979323846;

	GLFWwindow *window_;
	int window_w_, window_h_;

	glm::vec3 position_ = glm::vec3(0.f, 0.f, 5.f);

	glm::mat4 v_, p_;

	float angle_horizontal_ = PI;
	float angle_vertical_ = 0.f;
	float fov_ = 45.f;

	float speed_ = 3.0f;
	float mouse_speed_ = 0.002f;

	double time_old_ = glfwGetTime(), time_new_;
	double x_old_, y_old_;
public:
	Camera(GLFWwindow *window, int window_w, int window_h)
		: window_(window), window_w_(window_w), window_h_(window_h) {
		glfwSetInputMode(window_, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
		glfwGetCursorPos(window_, &x_old_, &y_old_);
	}
	glm::mat4 v() {
		return v_;
	}
	glm::mat4 p() {
		return p_;
	}
	void Update();
};
