#include "Camera.hpp"

#include <cstdio>
#include <glm/gtc/matrix_transform.hpp>

static double scoll = 0;
static void ScrollCallback(GLFWwindow* window, double x, double y) {
	scoll = y;
}

void PrintMat(glm::mat4 &m, const char *indent, const char *name) {
	printf("\n");
	printf("%s", indent);
	if(name) printf("%s = ", name);
	printf(  "glm::mat4(\n");
	printf("%s	%f, %f, %f, %f,\n", indent, m[0][0], m[0][1], m[0][2], m[0][3]);
	printf("%s	%f, %f, %f, %f,\n", indent, m[1][0], m[1][1], m[1][2], m[1][3]);
	printf("%s	%f, %f, %f, %f,\n", indent, m[2][0], m[2][1], m[2][2], m[2][3]);
	printf("%s	%f, %f, %f, %f\n",  indent, m[3][0], m[3][1], m[3][2], m[3][3]);
	printf("%s);\n", indent);
}

void PrintVec(glm::vec3 &v, const char *indent, const char *name) {
	printf("\n");
	printf("%s", indent);
	if(name) printf("%s = ", name);
	printf("glm::vec3(%f, %f, %f);\n", v.x, v.y, v.z);
}

Camera::Camera(GLFWwindow *window, int window_w, int window_h, double time)
	: window_(window), window_w_(window_w), window_h_(window_h), time_(time) {
	glfwSetScrollCallback(window, ScrollCallback);
	glfwSetInputMode(window_, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	glfwGetCursorPos(window_, &x_, &y_);
}

glm::mat4 Camera::Update(double time) {
	fix_.Update([this]{
		vp_ = glm::mat4(
			1.187824, -0.546560, -0.429420, -0.428562,
			0.000000, 2.134668, -0.468028, -0.467093,
			-0.658199, -0.986354, -0.774956, -0.773407,
			0.029338, 0.094435, 3.050498, 3.244203
		);
		v_ = glm::mat4(
			0.874689, -0.226393, 0.428562, 0.000000,
			0.000000, 0.884208, 0.467093, 0.000000,
			-0.484684, -0.408561, 0.773407, 0.000000,
			0.021604, 0.039116, -3.244203, 1.000000
		);
	}, [this]{
		glfwGetCursorPos(window_, &x_, &y_);
	});
	if(fix_.state()) return vp_;

	float delta_time = time - time_;
	time_ = time;

	double x, y;
	glfwGetCursorPos(window_, &x, &y);
	angle_horizontal_ += mouse_speed_ * float(x_ - x);
	angle_vertical_   += mouse_speed_ * float(y_ - y);
	x_ = x;
	y_ = y;

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
		position_ += delta_time * speed_ * direction;
	}
	// backward
	if(glfwGetKey(window_, GLFW_KEY_S) == GLFW_PRESS || glfwGetKey(window_, GLFW_KEY_DOWN) == GLFW_PRESS) {
		position_ -= delta_time * speed_ * direction;
	}
	// right
	if(glfwGetKey(window_, GLFW_KEY_D) == GLFW_PRESS || glfwGetKey(window_, GLFW_KEY_RIGHT) == GLFW_PRESS) {
		position_ += delta_time * speed_ * right;
	}
	// left
	if(glfwGetKey(window_, GLFW_KEY_A) == GLFW_PRESS || glfwGetKey(window_, GLFW_KEY_LEFT) == GLFW_PRESS) {
		position_ -= delta_time * speed_ * right;
	}
	if(glfwGetKey(window_, GLFW_KEY_EQUAL) == GLFW_PRESS) {
		speed_ += 1;
	}
	if(glfwGetKey(window_, GLFW_KEY_MINUS) == GLFW_PRESS) {
		speed_ -= 1;
	}
	if(glfwGetKey(window_, GLFW_KEY_SPACE) == GLFW_PRESS) {
		position_ = position_init_;
		angle_horizontal_ = angle_horizontal_init_;
		angle_vertical_ = angle_vertical_init_;
		fov_ = fov_init_;
	}
	print_vp_.Update([this]{
		PrintMat(vp_, "\t\t", "vp_");
		PrintMat(v_, "\t\t", "v_");
	});

	fov_ += delta_time * scroll_speed_ * scoll;
	scoll = 0;
	// Camera matrix
	v_ = glm::lookAt(
			position_,             // Camera is here
			position_ + direction, // and looks here: at the same position_, plus "direction"
			up);                   // Head is up (set to 0,-1,0 to look upside-down)
	// Projection matrix: 45° Field of View, 4:3 ratio, display range : 0.1 unit <-> 100 units
	p_ = glm::perspective(fov_, float(window_w_) / window_h_, 0.1f, 100.f);
	vp_ = p_ * v_;

	return vp_;
}
