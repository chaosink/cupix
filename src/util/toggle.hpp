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

	template<typename T0, typename T1>
	bool Update(T0 On, T1 Off) {
		if(count_-- > 0) return state_;
		if(glfwGetKey(window_, key_) == GLFW_PRESS) {
			if(!pressed_) {
				state_ = !state_;
				if(state_) On(); // excute On() when state_ change from false to true
				else Off(); // excute Off() when state_ change from true to false
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

	template<typename T>
	bool Update(T OnOff) {
		if(count_-- > 0) return state_;
		if(glfwGetKey(window_, key_) == GLFW_PRESS) {
			if(!pressed_) {
				state_ = !state_;
				OnOff(); // excute OnOff() when state_ changes
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
