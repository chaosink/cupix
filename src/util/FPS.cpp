#include "FPS.hpp"

#include <cstdio>

double FPS::Update() {
	if(c_frame_++ % 10 == 0) {
		time_new_ = glfwGetTime();
		fps_ = 10 / (time_new_ - time_old_);
		time_old_ = time_new_;
		printf("\rFPS: %f", fps_);
		fflush(stdout);
	}
	return fps_;
}

void FPS::Term() {
	printf("\n");
}
