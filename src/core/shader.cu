#include <cstdio>

#include "cupix.hpp"

namespace cupix {

namespace cu {

extern __constant__ __device__ int w, h;
extern __constant__ __device__ float time;
extern texture<uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat> texture;
extern __constant__ __device__ float mvp[16];
extern __constant__ __device__ float mv[16];
extern __constant__ __device__ float light[4];

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

__device__
void FragmentShader(FragmentIn &in, vec4 &color) {
	/********** Visualization of normal and uv **********/
	// vec4 c = vec4(in.normal, 0.f);
	// vec4 c = vec4(in.normal * 0.5f + 0.5f, 0.5f);
	// vec4 c = vec4(in.uv, 0.f, 0.f);

	/********** Texture sampling **********/
	// float4 c = tex2D(texture, in.uv.s, 1 - in.uv.t);

	/********** Shadertoy effect **********/
	// vec2 fragCoord(in.coord.x, in.coord.y);
	// vec2 iResolution(w, h);
	// vec2 uv = fragCoord - iResolution / 2.f;
	// float d = dot(uv, uv);
	// vec4 c = vec4(0.5f + 0.5f * cos(d / 5.f + 10.f * time));

	/********** Phong shading **********/
	mat4 m = *((mat4*)mv);
	vec3 light_position = m * vec4(light[0], light[1], light[2], 1.0f);
	vec3 position = in.position;
	vec3 normal = in.normal;
	vec3 light_direction = light_position - position;
	vec3 eye_direction = -position;

	vec3 light_color = vec3(1.f, 1.f, 1.f);
	float light_power = 40.f;

	// Material properties
	vec3 diffuse_color  = vec3(0.9f, 0.7f, 0.5f);
	vec3 ambient_color  = vec3(0.4f, 0.4f, 0.4f) * diffuse_color;
	vec3 specular_color = vec3(0.3f, 0.3f, 0.3f);

	// Distance to the light
	float distance = length(light_position - position);
	vec3 n = normalize(normal);
	vec3 l = normalize(light_direction);
	float cos_theta = clamp(dot(n, l), 0.f, 1.f);
	vec3 e = normalize(eye_direction);
	vec3 r = reflect(-l, n);
	float cos_alpha = clamp(dot(e, r), 0.f, 1.f);

	vec4 c(
		ambient_color +
		diffuse_color * light_color * light_power * cos_theta / (distance * distance) +
		specular_color * light_color * light_power * pow(cos_alpha, 5.f) / (distance * distance),
		1.f);

	/********** Output color **********/
	color = vec4(c.x, c.y, c.z, c.w);
	// color = pow(color, vec4(1.f / 2.2f)); // Gamma correction
}

}
}
