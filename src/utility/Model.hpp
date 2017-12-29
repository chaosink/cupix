#pragma once

#include "GLFW/glfw3.h"
#include "glm/glm.hpp"

class Model {
	GLFWwindow *window_;
	glm::mat4 m_;
	double time_;
	float turn_speed_ = 0.5f;

	int n_vertex_ = 0;
	float *vertex_;
	float *normal_;
	float *uv_;

public:
	Model(GLFWwindow *window, const char *file_name);
	~Model() {
		delete[] vertex_;
		delete[] normal_;
	}
	int n_vertex() {
		return n_vertex_;
	}
	float* vertex() {
		return vertex_;
	}
	float* normal() {
		return normal_;
	}
	float* uv() {
		return uv_;
	}
	glm::mat4 Update(double time);
};
