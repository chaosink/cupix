#pragma once

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

class Camera {
	const double PI  = 3.14159265358979323846;

	GLFWwindow *window_;
	int window_w_, window_h_;

	glm::mat4 v_, p_, vp_;

	const glm::vec3 position_init_ = glm::vec3(0.f, 0.f, 3.f);
	const float angle_horizontal_init_ = PI;
	const float angle_vertical_init_ = 0.f;

	glm::vec3 position_ = position_init_;
	float angle_horizontal_ = angle_horizontal_init_;
	float angle_vertical_ = angle_vertical_init_;

	float fov_ = PI / 4.f;
	float speed_ = 1.0f;
	float mouse_speed_ = 0.001f;

	double time_old_ = glfwGetTime(), time_new_;
	double x_old_, y_old_;

	bool fixed = false;
	bool fixed_pressed = false;
	bool print_pressed = false;
public:
	Camera(GLFWwindow *window, int window_w, int window_h)
		: window_(window), window_w_(window_w), window_h_(window_h) {
		glfwSetInputMode(window_, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
		glfwGetCursorPos(window_, &x_old_, &y_old_);
		Update();
	}
	glm::mat4 v() {
		return v_;
	}
	glm::mat4 p() {
		return p_;
	}
	glm::mat4 vp() {
		return vp_;
	}
	void Update();
};
