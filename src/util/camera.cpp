#include <cstdio>

#include "camera.hpp"
#include <glm/gtc/matrix_transform.hpp>

static double scoll = 0;
static void ScrollCallback(GLFWwindow* window, double x, double y) {
	scoll = y;
}

void PrintMat(glm::mat4 &m, const char *indent = "", const char *name = NULL) {
	printf("\n");
	if(name) printf("%s%s = ", indent, name);
	printf(  "glm::mat4(\n");
	printf("%s	%f, %f, %f, %f,\n", indent, m[0][0], m[0][1], m[0][2], m[0][3]);
	printf("%s	%f, %f, %f, %f,\n", indent, m[1][0], m[1][1], m[1][2], m[1][3]);
	printf("%s	%f, %f, %f, %f,\n", indent, m[2][0], m[2][1], m[2][2], m[2][3]);
	printf("%s	%f, %f, %f, %f\n",  indent, m[3][0], m[3][1], m[3][2], m[3][3]);
	printf("%s);\n", indent);
}

Camera::Camera(GLFWwindow *window, int window_w, int window_h)
	: window_(window), window_w_(window_w), window_h_(window_h) {
	glfwSetScrollCallback(window, ScrollCallback);
	glfwSetInputMode(window_, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	glfwGetCursorPos(window_, &x_old_, &y_old_);
	Update();
}

void Camera::Update() {
	if(glfwGetKey(window_, GLFW_KEY_F) == GLFW_PRESS) {
		if(!fixed_pressed) {
			fixed_pressed = true;
			fixed = !fixed;
			if(fixed)
				// vp_ = glm::mat4(
				// 	1.102078, 0.573894, 0.534812, 0.533743,
				// 	0.000000, 2.205375, -0.407661, -0.406847,
				// 	0.793457, -0.797114, -0.742830, -0.741346,
				// 	-0.023755, 0.045511, 0.214560, 0.413932
				// );
				vp_ = glm::mat4(
					0.964459, -1.055420, -0.552909, -0.551804,
					0.000000, 1.892313, -0.622229, -0.620986,
					-0.956018, -1.064739, -0.557791, -0.556676,
					-0.014499, 0.237664, 1.606357, 1.802948
				);
				// vp_ = glm::mat4(
				// 	1.304048, 0.505253, 0.184963, 0.184593,
				// 	-0.000000, 1.596964, -0.751460, -0.749959,
				// 	0.378961, -1.738635, -0.636478, -0.635206,
				// 	0.009863, 0.063004, 4.698840, 4.889251
				// );
		}
	} else {
		fixed_pressed = false;
	}
	if(fixed) return;

	time_new_ = glfwGetTime();
	float time = time_new_ - time_old_;

	double x, y;
	glfwGetCursorPos(window_, &x, &y);

	angle_horizontal_ += mouse_speed_ * float(x_old_ - x);
	angle_vertical_   += mouse_speed_ * float(y_old_ - y);

	// Direction: Spherical coordinates to Cartesian coordinates conversion
	glm::vec3 direction(
		cos(angle_vertical_) * sin(angle_horizontal_),
		sin(angle_vertical_),
		cos(angle_vertical_) * cos(angle_horizontal_)
	);

	// Right vector
	glm::vec3 right = glm::vec3(
		sin(angle_horizontal_ - PI / 2.0f),
		0.f,
		cos(angle_horizontal_ - PI / 2.0f)
	);

	// Up vector
	glm::vec3 up = glm::cross(right, direction);

	// forward
	if(glfwGetKey(window_, GLFW_KEY_W) == GLFW_PRESS || glfwGetKey(window_, GLFW_KEY_UP) == GLFW_PRESS) {
		position_ += direction * time * speed_;
	}
	// backward
	if(glfwGetKey(window_, GLFW_KEY_S) == GLFW_PRESS || glfwGetKey(window_, GLFW_KEY_DOWN) == GLFW_PRESS) {
		position_ -= direction * time * speed_;
	}
	// right
	if(glfwGetKey(window_, GLFW_KEY_D) == GLFW_PRESS || glfwGetKey(window_, GLFW_KEY_RIGHT) == GLFW_PRESS) {
		position_ += right * time * speed_;
	}
	// left
	if(glfwGetKey(window_, GLFW_KEY_A) == GLFW_PRESS || glfwGetKey(window_, GLFW_KEY_LEFT) == GLFW_PRESS) {
		position_ -= right * time * speed_;
	}
	if(glfwGetKey(window_, GLFW_KEY_PAGE_UP) == GLFW_PRESS) {
		speed_ += 1;
	}
	if(glfwGetKey(window_, GLFW_KEY_PAGE_DOWN) == GLFW_PRESS) {
		speed_ -= 1;
	}
	if(glfwGetKey(window_, GLFW_KEY_SPACE) == GLFW_PRESS) {
		position_ = position_init_;
		angle_horizontal_ = angle_horizontal_init_;
		angle_vertical_ = angle_vertical_init_;
		fov_ = fov_init_;
	}
	if(glfwGetKey(window_, GLFW_KEY_P) == GLFW_PRESS) {
		if(!print_pressed) {
			print_pressed = true;
			PrintMat(vp_, "\t\t\t\t", "vp_");
		}
	} else {
		print_pressed = false;
	}

	fov_ += scoll * time * scroll_speed_;
	scoll = 0;
	// Camera matrix
	v_ = glm::lookAt(
			position_,             // Camera is here
			position_ + direction, // and looks here: at the same position_, plus "direction"
			up);                   // Head is up (set to 0,-1,0 to look upside-down)
	// Projection matrix: 45° Field of View, 4:3 ratio, display range : 0.1 unit <-> 100 units
	p_ = glm::perspective(fov_, float(window_w_) / window_h_, 0.1f, 100.f);
	vp_ = p_ * v_;

	time_old_ = time_new_;
	x_old_ = x;
	y_old_ = y;
}
