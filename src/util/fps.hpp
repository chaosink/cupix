#pragma once

#include <cstdio>
#include <GLFW/glfw3.h>

class FPS {
	int c_frame_ = 0;
	float time_old_ = glfwGetTime();
	float time_new_;
	float fps_ = 0;
public:
	double Update() {
		if(c_frame_++ % 10 == 0) {
			time_new_ = glfwGetTime();
			fps_ = 10 / (time_new_ - time_old_);
			time_old_ = time_new_;
			printf("\rFPS: %f", fps_);
			fflush(stdout);
		}
		return fps_;
	}
	void Term() {
		printf("\n");
	}
	float fps() {
		return fps_;
	}
};
