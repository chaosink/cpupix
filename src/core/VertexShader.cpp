#include "CPUPix.hpp"

namespace cpupix {

namespace kernel {

extern glm::mat4 mvp;
extern glm::mat4 mv;

using namespace glm;

void VertexShader(VertexIn &in, VertexOut &out, Vertex &v) {
	v.position = mvp * vec4(in.position, 1.f);

	// out.position = mv * vec4(in.position, 1.f);
	// out.normal   = mv * vec4(in.normal, 0.f);
	out.position = in.position;
	out.normal   = in.normal;
	out.uv       = in.uv;
}

}
}
