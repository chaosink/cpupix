#include "GL/glew.h"
#include "GLFW/glfw3.h"

#include "CPUPix.hpp"
using namespace cpupix;
#include "Model.hpp"
#include "Texture.hpp"
#include "Camera.hpp"
#include "FPS.hpp"
#include "Toggle.hpp"
#include "Video.hpp"

GLFWwindow* InitGLFW(int window_w, int window_h) {
	if(!glfwInit()) exit(EXIT_FAILURE);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);
	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
	glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);

	GLFWwindow *window = glfwCreateWindow(window_w, window_h, "CPUPix", NULL, NULL);
	if(!window) {
		glfwTerminate();
		fprintf(stderr, "Failed to create GLFW window\n");
		exit(EXIT_FAILURE);
	}
	glfwMakeContextCurrent(window);
	if(glewInit() != GLEW_OK) {
		glfwTerminate();
		fprintf(stderr, "Failed to initialize GLEW\n");
		exit(EXIT_FAILURE);
	}
	// glfwSwapInterval(1);
	return window;
}

void TermGLFW(GLFWwindow *window) {
	glfwDestroyWindow(window);
	glfwTerminate();
}

GLuint InitGL() {
	GLuint pbo;
	glGenBuffers(1, &pbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
	return pbo;
}

void TermGL(GLuint pbo) {
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	glDeleteBuffers(1, &pbo);
}

void UpdateGL(GLFWwindow *window, int window_w, int window_h, unsigned char *frame) {
	glBufferData(GL_PIXEL_UNPACK_BUFFER, window_w * window_h * 3, frame, GL_DYNAMIC_COPY);
	glDrawPixels(window_w, window_h, GL_RGB, GL_UNSIGNED_BYTE, 0);
	glfwSwapBuffers(window);
	glfwPollEvents();
}

int main(int argc, char *argv[]) {
	if(argc < 2) {
		printf("Usage: cpupix input_obj_file [output_video_file]\n");
		return 0;
	}
	bool record = false;
	if(argc == 3) record = true;

	int window_w = 1280;
	int window_h = 720;
	// int window_w = 640;
	// int window_h = 360;

	GLFWwindow* window = InitGLFW(window_w, window_h);
	GLuint pbo = InitGL();

	CPUPix pix(window_w, window_h, AA::NOAA);
	pix.ClearColor(0.08f, 0.16f, 0.24f, 1.f);
	// pix.Disable(Flag::DEPTH_TEST);
	// pix.Enable(Flag::BLEND);
	// pix.Disable(Flag::CULL_FACE);
	// pix.CullFace(Face::FRONT);
	// pix.FrontFace(Winding::CW);

	Light light[2]{
		 5.f, 4.f, 3.f, // position
		 1.f, 1.f, 1.f, // color
		20.f,           // power
		-5.f, 4.f, 3.f, // position
		 1.f, 1.f, 1.f, // color
		30.f,           // power
	};
	pix.Lights(2, light);

	Model model(window, argv[1]);
	pix.VertexData(model.n_vertex(), model.vertex(), model.normal(), model.uv());

	Texture texture("texture/texture.jpg");
	pix.Texture(texture.data(), texture.w(), texture.h(), false); // gamma_correction = false

	Camera camera(window, window_w, window_h);
	FPS fps;
	Toggle toggle(window, GLFW_KEY_R, true); // init_state = true
	Video video(window_w, window_h);
	while(glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS && !glfwWindowShouldClose(window)) {
		double time = glfwGetTime();

		pix.Clear();

		glm::mat4 m = model.Update(time);
		glm::mat4 vp = camera.Update(time);
		glm::mat4 mvp = vp * m;
		pix.MVP(mvp);
		glm::mat4 v = camera.v();
		glm::mat4 mv = v * m;
		pix.MV(mv);

		pix.Time(time);
		pix.Toggle(toggle.Update([] {
			printf("\nUse Blinn-Phong shading\n");
		}, [] {
			printf("\nUse Phong shading\n");
		}));

		pix.Draw();
		pix.DrawFPS(fps.Update(time) + 0.5f);

		UpdateGL(window, window_w, window_h, pix.frame());
		if(record) video.Add(pix.frame());
	}
	fps.Term();

	TermGL(pbo);
	TermGLFW(window);

	if(record) video.Save(argv[2]);
}
