#include "cupix.hpp"

namespace cupix {

namespace cu {

extern __constant__ __device__ int w, h;
extern __constant__ __device__ float time;
extern texture<uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat> texture;

__device__
void VertexShader(VertexIn &in, glm::mat4 &mvp, VertexOut &out, Vertex &v) {
	v.position = mvp * glm::vec4(in.position, 1.f);

	out.position = in.position;
	out.normal = in.normal;
	out.color = in.color;
	out.uv = in.uv;
}

__device__
void FragmentShader(FragmentIn &in, glm::vec4 &color) {
	// glm::vec4 c = glm::vec4(in.normal, 0.f);
	glm::vec4 c = glm::vec4(in.normal * 0.5f + 0.5f, 0.5f);
	// glm::vec4 c = glm::vec4(in.uv, 0.f, 0.f);
	// float4 c = tex2D(texture, in.uv.s, 1 - in.uv.t);

	// glm::vec2 fragCoord(in.coord.x, in.coord.y);
	// glm::vec2 iResolution(w, h);
	// glm::vec2 uv = fragCoord - iResolution / 2.f;
	// float d = glm::dot(uv, uv);
	// glm::vec4 c = glm::vec4(0.5f + 0.5f * cos(d / 5.f + 10.f * time));

	color = glm::vec4(c.x, c.y, c.z, c.w);
	// color = glm::pow(color, glm::vec4(1.f / 2.2f));
}

}
}
