#include "CPUPix.hpp"

#ifdef _WIN32
#include <windows.h>
#endif

#include <cstdio>
#include <cstring>

namespace cpupix {

namespace kernel {

Texture2D texture;
int w, h;
int n_light;
Light light[4];
glm::mat4 mvp;
glm::mat4 mv;
float time;
bool toggle;

int n_triangle;
bool depth_test = true;
bool blend = false;
glm::u8vec4 clear_color;

const int bitmap_size = 128 * 32; // 128 characters, each 32 bytes
char bitmap[bitmap_size];

void Clear(unsigned char *frame_buf, float *depth_buf);
void NormalSpace(VertexIn *in, VertexOut *out, Vertex *v);
void WindowSpace(Vertex *v);
void AssemTriangle(Vertex *v, Triangle *triangle);
void ScanTriangle(Vertex *v, VertexOut *vo, Scanline *scanline_);
void DrawSegment(Scanline *scanline_, float *depth_buf, unsigned char* frame_buf);
void DrawCharater(int ch, int x0, int y0, int w, unsigned char *frame_buf);
void DownSample(unsigned char *frame_buf, unsigned char *pbo_buf);

}

Texture2D::~Texture2D() {
	delete[] data_;
}

glm::vec3 Texture2D::Sample(float u, float v) {
	int x = glm::clamp(int(u * w_), 0, w_ - 1);
	int y = glm::clamp(int(v * h_), 0, h_ - 1);
	int i = y * w_ + x;
	return data_[i];
}

void Texture2D::Bind(unsigned char *d, int w, int h, bool gamma_correction) {
	delete[] data_;
	w_ = w;
	h_ = h;
	data_ = new glm::vec3[w * h];
	if(gamma_correction)
		for(int i = 0; i < w * h; ++i)
			data_[i] = glm::pow(glm::vec3(d[i * 3] / 255.f, d[i * 3 + 1] / 255.f, d[i * 3 + 2] / 255.f), glm::vec3(1.f / 2.2f)); // Gamma correction
	else
		for(int i = 0; i < w * h; ++i)
			data_[i] = glm::vec3(d[i * 3] / 255.f, d[i * 3 + 1] / 255.f, d[i * 3 + 2] / 255.f);
}

CPUPix::CPUPix(int window_w, int window_h, AA aa = AA::NOAA)
	: window_w_(window_w), window_h_(window_h),
	frame_w_(window_w), frame_h_(window_h), aa_(aa) {
	if(aa_ == AA::MSAA) aa_ = AA::NOAA; // MSAA not supported
	if(aa_ != AA::NOAA) {
		frame_w_ *= 2;
		frame_h_ *= 2;
		frame_buf_ = new unsigned char[frame_w_ * frame_h_ * 3];
		frame_ = new unsigned char[window_w_ * window_h_ * 3];
	} else {
		frame_buf_ = new unsigned char[frame_w_ * frame_h_ * 3];
		frame_ = frame_buf_;
	}
	depth_buf_ = new float[frame_w_ * frame_h_];
	kernel::w = frame_w_;
	kernel::h = frame_h_;

	// load bitmap font into kernel memory
	FILE *font_file = fopen("font/bitmap_font.data", "rb");
	size_t r = kernel::bitmap_size; // suppress warning of fread()
	r = fread(kernel::bitmap, 1, r, font_file);
	fclose(font_file);

	scanline_ = new Scanline[frame_h_];
}

CPUPix::~CPUPix() {
	if(aa_ != AA::NOAA) delete[] frame_;
	delete[] depth_buf_;
	delete[] frame_buf_;
	delete[] scanline_;
}

void CPUPix::Enable(Flag flag) {
	bool b = true;
	switch(flag) {
		case Flag::DEPTH_TEST:
			kernel::depth_test = b;
			return;
		case Flag::BLEND:
			kernel::blend = b;
			return;
		case Flag::CULL_FACE:
			cull_ = b;
			return;
	}
}

void CPUPix::Disable(Flag flag) {
	bool b = false;
	switch(flag) {
		case Flag::DEPTH_TEST:
			kernel::depth_test = b;
			return;
		case Flag::BLEND:
			kernel::blend = b;
			return;
		case Flag::CULL_FACE:
			cull_ = b;
			return;
	}
}

void CPUPix::CullFace(Face face) {
	cull_face_ = face;
}

void CPUPix::FrontFace(Winding winding) {
	front_face_ = winding;
}

void CPUPix::ClearColor(float r, float g, float b, float a) {
	glm::ivec4 color = glm::vec4(r * 255.f, g * 255.f, b * 255.f, a * 255.f);
	color = glm::clamp(color, glm::ivec4(0), glm::ivec4(255));
	glm::u8vec4 clear_color = color;
	kernel::clear_color = clear_color;
}

void CPUPix::Clear() {
	kernel::Clear(frame_buf_, depth_buf_);
}

void CPUPix::Draw() {
	kernel::NormalSpace(vertex_in_, vertex_out_, vertex_buf_);
	kernel::WindowSpace(vertex_buf_);
	kernel::AssemTriangle(vertex_buf_, triangle_buf_);
	for(int i = 0; i < n_triangle_; i++)
		if(!triangle_buf_[i].empty)
			if(!cull_ || ((cull_face_ != Face::FRONT_AND_BACK)
			&& ((triangle_buf_[i].winding == front_face_) != (cull_face_ == Face::FRONT)))) {
				if(aa_ == AA::MSAA) {
					// MSAA not supported
				} else {
					kernel::ScanTriangle(vertex_buf_ + i * 3, vertex_out_ + i * 3, scanline_);
				}
			}
	kernel::DrawSegment(scanline_, depth_buf_, frame_buf_);
	if(aa_ != AA::NOAA) kernel::DownSample(frame_buf_, frame_);
}

void CPUPix::DrawFPS(int fps) {
	kernel::DrawCharater('F',  0, 0, window_w_, frame_);
	kernel::DrawCharater('P', 16, 0, window_w_, frame_);
	kernel::DrawCharater('S', 32 - 3, 0, window_w_, frame_);
	kernel::DrawCharater(fps % 1000 / 100 + 48, 48 + 5, 0, window_w_, frame_);
	kernel::DrawCharater(fps % 100 / 10   + 48, 64 + 5, 0, window_w_, frame_);
	kernel::DrawCharater(fps % 10         + 48, 80 + 5, 0, window_w_, frame_);
}

void CPUPix::VertexData(int size, float *position, float *normal, float *uv) {
	n_vertex_ = size;
	n_triangle_ = n_vertex_ / 3;
	kernel::n_triangle = n_triangle_;


	delete[] vertex_in_;
	vertex_in_ = new VertexIn[n_vertex_];
	for(int i = 0; i < n_vertex_; i++) {
		vertex_in_[i].position = glm::vec3(position[i * 3], position[i * 3 + 1], position[i * 3 + 2]);
		vertex_in_[i].normal = glm::vec3(normal[i * 3], normal[i * 3 + 1], normal[i * 3 + 2]);
		vertex_in_[i].uv = glm::vec2(uv[i * 2], uv[i * 2 + 1]);
	}

	delete[] vertex_out_;
	vertex_out_ = new VertexOut[n_vertex_];

	delete[] vertex_buf_;
	vertex_buf_= new Vertex[n_vertex_];

	delete[] triangle_buf_;
	triangle_buf_ = new Triangle[n_triangle_];
}

void CPUPix::MVP(glm::mat4 &mvp) {
	kernel::mvp = mvp;
}

void CPUPix::MV(glm::mat4 &mv) {
	kernel::mv = mv;
}

void CPUPix::Time(float time) {
	kernel::time = time;
}

void CPUPix::Texture(unsigned char *data, int w, int h, bool gamma_correction) {
	kernel::texture.Bind(data, w, h, gamma_correction);
}

void CPUPix::Lights(int n, Light *light) {
	kernel::n_light = n;
	memcpy(kernel::light, light, sizeof(Light) * n);
}

void CPUPix::Toggle(bool toggle) {
	kernel::toggle = toggle;
}

}
