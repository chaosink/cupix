#pragma once

class FPS {
	int c_frame_ = 0;
	double time_ = 0;
	float fps_ = 0;
public:
	float Update(double time);
	void Term();
	float fps() {
		return fps_;
	}
};
