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

Camera::Camera(GLFWwindow *window, int window_w, int window_h)
	: window_(window), window_w_(window_w), window_h_(window_h) {
	glfwSetScrollCallback(window, ScrollCallback);
	glfwSetInputMode(window_, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	glfwGetCursorPos(window_, &x_old_, &y_old_);
	Update();
}

glm::mat4 Camera::Update() {
	fix_.Update([this]{
		vp_ = glm::mat4(
			1.357115, 0.061360, 0.025538, 0.025487,
			0.000000, 1.709492, -0.707531, -0.706117,
			0.048880, -1.703614, -0.709053, -0.707636,
			0.048471, 0.505596, 2.529923, 2.724669
		);
		v_ = glm::mat4(
			0.999352, 0.025416, -0.025487, 0.000000,
			0.000000, 0.708095, 0.706117, 0.000000,
			0.035994, -0.705660, 0.707636, 0.000000,
			0.035693, 0.209425, -2.724669, 1.000000
		);
	}, [this]{
		glfwGetCursorPos(window_, &x_old_, &y_old_);
	});
	if(fix_.state()) return vp_;

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
	print_vp_.Update([this]{
		PrintMat(vp_, "\t\t", "vp_");
		PrintMat(v_, "\t\t", "v_");
	});

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

	return vp_;
}
