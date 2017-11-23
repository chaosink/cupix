#pragma once

#include <cstdio>
#include <GLFW/glfw3.h>

class FPS {
	int c_frame_ = 0;
	float time_old_ = glfwGetTime();
	float time_new_;
public:
	void Update() {
		if(c_frame_++ % 10 == 0) {
			time_new_ = glfwGetTime();
			double fps = 10 / (time_new_ - time_old_);
			time_old_ = time_new_;
			printf("\rFPS: %lf", fps);
			fflush(stdout);
		}
	}
};
