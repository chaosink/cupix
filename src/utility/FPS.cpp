#include "FPS.hpp"

#include <cstdio>

float FPS::Update(double time) {
	if(c_frame_++ % n_frame_ == 0) {
		fps_ = n_frame_ / (time - time_);
		time_ = time;
		printf("\rFPS: %f", fps_);
		fflush(stdout);
	}
	return fps_;
}

void FPS::Term() {
	printf("\n");
}
