#include "CUPix.hpp"

namespace cupix {

namespace core {

extern __constant__ __device__ float mvp[16];
extern __constant__ __device__ float mv[16];

using namespace glm;

__device__
void VertexShader(VertexIn &in, VertexOut &out, Vertex &v) {
	mat4 m = *((mat4*)mvp);
	v.position = m * vec4(in.position, 1.f);

	m = *((mat4*)mv);
	out.position = m * vec4(in.position, 1.f);
	out.normal   = m * vec4(in.normal, 0.f);
	out.color    = in.color;
	out.uv       = in.uv;
}

}
}
