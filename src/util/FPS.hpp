#pragma once

#include <GLFW/glfw3.h>

class FPS {
	int c_frame_ = 0;
	float time_old_ = glfwGetTime();
	float time_new_;
	float fps_ = 0;
public:
	double Update();
	void Term();
	float fps() {
		return fps_;
	}
};
