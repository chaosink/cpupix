#pragma once

#include <vector>
#include "glm/glm.hpp"

#define VERSION_MAJOR 0
#define VERSION_MINOR 1
#define VERSION_PATCH 0

#define _QUOTE(S) #S
#define _STR(S) _QUOTE(S)
#define VERSION_STRING _STR(VERSION_MAJOR) "." _STR(VERSION_MINOR) "." _STR(VERSION_PATCH)

namespace cpupix {

enum class AA : unsigned char {
	NOAA,
	MSAA,
	SSAA
};
enum class Winding : unsigned char {
	CCW,
	CW
};
enum class Face : unsigned char {
	BACK,
	FRONT,
	FRONT_AND_BACK
};
enum class Flag : unsigned char {
	DEPTH_TEST,
	BLEND,
	CULL_FACE,
};

struct Vertex {
	glm::vec4 position;
};
struct VertexIn {
	glm::vec3 position;
	glm::vec3 normal;
	glm::vec2 uv;
};
struct VertexOut {
	glm::vec3 position;
	glm::vec3 normal;
	glm::vec2 uv;
	VertexOut operator+(const VertexOut &vo) const {
		return VertexOut {
			position + vo.position,
			normal + vo.normal,
			uv + vo.uv
		};
	}
	VertexOut operator-(const VertexOut &vo) const {
		return VertexOut {
			position - vo.position,
			normal - vo.normal,
			uv - vo.uv
		};
	}
	VertexOut operator*(const float a) const {
		return VertexOut {
			position * a,
			normal * a,
			uv * a
		};
	}
	VertexOut operator/(const float a) const {
		return VertexOut {
			position / a,
			normal / a,
			uv / a
		};
	}
	VertexOut& operator+=(const VertexOut &vo) {
		position += vo.position;
		normal += vo.normal;
		uv += vo.uv;
		return *this;
	}
};
struct Triangle {
	Winding winding;
	bool empty;
};

struct Fragment {
	float z;
	float w;
	VertexOut vo;
	Fragment operator+(const Fragment &f) {
		return Fragment {
			z + f.z,
			w + f.w,
			vo + f.vo,
		};
	}
	Fragment operator-(const Fragment &f) {
		return Fragment {
			z - f.z,
			w - f.w,
			vo - f.vo,
		};
	}
	Fragment operator*(const float a) const {
		return Fragment {
			z * a,
			w * a,
			vo * a
		};
	}
	Fragment operator/(const float a) const {
		return Fragment {
			z / a,
			w / a,
			vo / a
		};
	}
	Fragment& operator+=(const Fragment &f) {
		z += f.z;
		w += f.w;
		vo += f.vo;
		return *this;
	}
};
struct Segment {
	int x;
	int length;
	Fragment fragment;
	Fragment fragment_delta;
	float z(int xx) {
		return fragment.z + fragment_delta.z * (xx - x);
	}
	Fragment f(int xx) {
		return fragment + fragment_delta * (xx - x);
	}
};
struct ScanNode {
	bool in;
	int x;
	Segment *segment;
};
struct Scanline {
	std::vector<Segment> segment;
};
struct FragmentIn {
	glm::vec2 coord;
	float z;
	glm::vec3 position;
	glm::vec3 normal;
	glm::vec2 uv;
};
struct Light {
	float position[3];
	float color[3];
	float power;
};

class Texture2D {
	int w_, h_;
	glm::vec3 *data_ = nullptr;
public:
	~Texture2D();
	glm::vec3 Sample(float u, float v);
	void Bind(unsigned char *d, int w, int h, bool gamma_correction);
};

class CPUPix {
	int window_w_, window_h_;
	unsigned char *frame_;

	int frame_w_, frame_h_;
	AA aa_;
	bool cull_ = true;
	Face cull_face_ = Face::BACK;
	Winding front_face_ = Winding::CCW;
	int n_triangle_, n_vertex_;
	Scanline *scanline_ = nullptr;
	unsigned char *frame_buf_;
	float *depth_buf_;

	VertexIn *vertex_in_ = nullptr;
	VertexOut *vertex_out_ = nullptr;
	Vertex *vertex_buf_ = nullptr;
	Triangle *triangle_buf_ = nullptr;
	unsigned char *texture_buf_ = nullptr;

public:
	CPUPix(int window_w, int window_h, AA aa);
	~CPUPix();
	void Enable(Flag flag);
	void Disable(Flag flag);
	void CullFace(Face face);
	void FrontFace(Winding winding);
	void ClearColor(float r, float g, float b, float a);
	void Clear();
	void Draw();
	void DrawFPS(int fps);
	void VertexData(int size, float *position, float *normal, float *uv);
	void MVP(glm::mat4 &mvp);
	void MV(glm::mat4 &mv);
	void Time(float time);
	void Texture(unsigned char *data, int w, int h, bool gamma_correction);
	void Lights(int n, Light *light);
	void Toggle(bool toggle);
	unsigned char* frame() {
		return frame_;
	}
};

}
