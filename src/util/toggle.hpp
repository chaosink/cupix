#pragma once

#include <functional>
#include <GLFW/glfw3.h>

class Toggle {
	GLFWwindow *window_;
	int key_;
	bool pressed_ = false;
	bool state_;
	int count_ = -1;
	int jitter_ = 4;
public:
	Toggle(GLFWwindow *window, int key, bool state)
		: window_(window), key_(key), state_(state) {}
	bool Update() {
		if(count_-- > 0) return state_;
		if(glfwGetKey(window_, key_) == GLFW_PRESS) {
			if(!pressed_) {
				state_ = !state_;
				pressed_ = true;
				count_ = jitter_;
			}
		} else {
			if(pressed_) {
				pressed_ = false;
				count_ = jitter_;
			}
		}
		return state_;
	}
	bool state() {
		return state_;
	}
};
