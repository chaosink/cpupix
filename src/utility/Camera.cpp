#include "Camera.hpp"

#include <cstdio>
#include "glm/gtc/matrix_transform.hpp"

static double scoll = 0;
static void ScrollCallback(GLFWwindow* /*window*/, double /*x*/, double y) {
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
	glfwGetCursorPos(window_, &x_, &y_);
}

glm::mat4 Camera::Update(double time) {
	fix_.Update([this]{ // call this lambda when toggle `fix_` turn on
		vp_ = glm::mat4(
			0.955822, -0.858065, -0.616271, -0.615039,
			0.000000, 2.090286, -0.501349, -0.500347,
			-0.964653, -0.850209, -0.610629, -0.609409,
			0.002771, -0.037387, 0.061824, 0.261501
		);
		v_ = glm::mat4(
			0.703848, -0.355422, 0.615039, 0.000000,
			0.000000, 0.865825, 0.500347, 0.000000,
			-0.710351, -0.352168, 0.609409, 0.000000,
			0.002040, -0.015486, -0.261501, 1.000000
		);
	}, [this]{ // call this lambda when toggle `fix_` turn off
		glfwGetCursorPos(window_, &x_, &y_);
	});
	if(fix_.state()) return vp_;

	float delta_time = time - time_;
	time_ = time;

	double x, y;
	glfwGetCursorPos(window_, &x, &y);
	angle_horizontal_ += mouse_turn_factor_ * float(x_ - x);
	angle_vertical_   += mouse_turn_factor_ * float(y_ - y);
	x_ = x;
	y_ = y;

	// turn right
	if(glfwGetKey(window_, GLFW_KEY_SEMICOLON) == GLFW_PRESS) {
		angle_horizontal_ -= delta_time * turn_speed_;
	}
	// turn left
	if(glfwGetKey(window_, GLFW_KEY_K) == GLFW_PRESS) {
		angle_horizontal_ += delta_time * turn_speed_;
	}
	// turn  up
	if(glfwGetKey(window_, GLFW_KEY_O) == GLFW_PRESS) {
		angle_vertical_ += delta_time * turn_speed_;
	}
	// turn down
	if(glfwGetKey(window_, GLFW_KEY_L) == GLFW_PRESS) {
		angle_vertical_ -= delta_time * turn_speed_;
	}

	// Direction: Spherical coordinates to Cartesian coordinates conversion
	glm::vec3 direction(
		cos(angle_vertical_) * sin(angle_horizontal_),
		sin(angle_vertical_),
		cos(angle_vertical_) * cos(angle_horizontal_)
	);
	// Right vector
	glm::vec3 right(
		sin(angle_horizontal_ - PI / 2.0f),
		0.f,
		cos(angle_horizontal_ - PI / 2.0f)
	);
	// Up vector
	glm::vec3 up = glm::cross(right, direction);

	// move forward
	if(glfwGetKey(window_, GLFW_KEY_W) == GLFW_PRESS || glfwGetKey(window_, GLFW_KEY_UP) == GLFW_PRESS)
		position_ += delta_time * move_speed_ * direction;
	// move backward
	if(glfwGetKey(window_, GLFW_KEY_S) == GLFW_PRESS || glfwGetKey(window_, GLFW_KEY_DOWN) == GLFW_PRESS)
		position_ -= delta_time * move_speed_ * direction;
	// move right
	if(glfwGetKey(window_, GLFW_KEY_D) == GLFW_PRESS || glfwGetKey(window_, GLFW_KEY_RIGHT) == GLFW_PRESS)
		position_ += delta_time * move_speed_ * right;
	// move left
	if(glfwGetKey(window_, GLFW_KEY_A) == GLFW_PRESS || glfwGetKey(window_, GLFW_KEY_LEFT) == GLFW_PRESS)
		position_ -= delta_time * move_speed_ * right;
	// move up
	if(glfwGetKey(window_, GLFW_KEY_E) == GLFW_PRESS || glfwGetKey(window_, GLFW_KEY_PAGE_UP) == GLFW_PRESS)
		position_ += delta_time * move_speed_ * up;
	// move down
	if(glfwGetKey(window_, GLFW_KEY_Q) == GLFW_PRESS || glfwGetKey(window_, GLFW_KEY_PAGE_DOWN) == GLFW_PRESS)
		position_ -= delta_time * move_speed_ * up;
	if(glfwGetKey(window_, GLFW_KEY_EQUAL) == GLFW_PRESS)
		move_speed_ *= pow(1.1f, delta_time * 20);
	if(glfwGetKey(window_, GLFW_KEY_MINUS) == GLFW_PRESS)
		move_speed_ *= pow(0.9f, delta_time * 20);
	if(glfwGetKey(window_, GLFW_KEY_RIGHT_BRACKET) == GLFW_PRESS)
		turn_speed_ *= pow(1.1f, delta_time * 20);
	if(glfwGetKey(window_, GLFW_KEY_LEFT_BRACKET) == GLFW_PRESS)
		turn_speed_ *= pow(0.9f, delta_time * 20);
	if(glfwGetKey(window_, GLFW_KEY_C) == GLFW_PRESS || glfwGetKey(window_, GLFW_KEY_SPACE) == GLFW_PRESS) {
		position_ = position_init_;
		angle_horizontal_ = angle_horizontal_init_;
		angle_vertical_ = angle_vertical_init_;
		fov_ = fov_init_;
	}
	print_vp_.Update([this]{ // call this lambda when toggle `print_vp_` turn on
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
	p_ = glm::perspective(fov_, float(window_w_) / window_h_, 0.1f, 10000.f);
	vp_ = p_ * v_;

	return vp_;
}
