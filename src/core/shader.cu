#include <cstdio>

#include "cupix.hpp"

namespace cupix {

namespace cu {

extern texture<uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat> texture;
extern __constant__ __device__ float mvp[16];
extern __constant__ __device__ float mv[16];
extern __constant__ __device__ int w, h;
extern __constant__ __device__ float time;
extern __constant__ __device__ Light light;
extern __constant__ __device__ bool toggle;

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
	/********** Visualization of normal **********/
	// vec4 c = vec4(in.normal, 0.f);
	// vec4 c = vec4(in.normal * 0.5f + 0.5f, 0.5f);

	/********** Visualization of uv **********/
	// vec4 c = vec4(in.uv, 0.f, 0.f);

	/********** Texture sampling **********/
	// float4 c = tex2D(texture, in.uv.s, 1 - in.uv.t);

	/********** Shadertoy effect **********/
	// vec2 fragCoord(in.coord.x, in.coord.y);
	// vec2 iResolution(w, h);
	// vec2 uv = fragCoord - iResolution / 2.f;
	// float d = dot(uv, uv);
	// vec4 c = vec4(0.5f + 0.5f * cos(d / 5.f + 10.f * time));

	/********** Lighting **********/
	mat4 m = *((mat4*)mv);
	vec3 light_position = m * vec4(light.position[0], light.position[1], light.position[2], 1.0f);
	vec3 position = in.position;
	vec3 normal = normalize(in.normal);
	vec3 light_direction = light_position - position;
	float distance = length(light_direction);
	distance = distance * distance;
	light_direction = normalize(light_direction);

	vec3 light_color = vec3(light.color[0], light.color[1], light.color[2]);
	float light_power = light.power;

	const vec3 diffuse_color  = vec3(0.9f, 0.6f, 0.3f);
	const vec3 ambient_color  = vec3(0.4f, 0.4f, 0.4f) * diffuse_color;
	const vec3 specular_color = vec3(0.3f, 0.3f, 0.3f);
	const float shininess = 16.0;

	float specular = 0.f;
	float lambertian = max(dot(light_direction, normal), 0.f);
	if(lambertian > 0.f) {
		vec3 eye_direction = normalize(-position);
		if(toggle) {
			/***** Blinn-Phong shading *****/
			vec3 half = normalize(light_direction + eye_direction);
			float cos_alpha = clamp(dot(half, normal), 0.f, 1.f);
			specular = pow(cos_alpha, shininess);
		} else {
			/***** Phong shading *****/
			vec3 reflection = reflect(-light_direction, normal);
			float cos_alpha = clamp(dot(reflection, eye_direction), 0.f, 1.f);
			specular = pow(cos_alpha, shininess / 4.f); // exponent is different
		}
	}
	vec4 c(
		ambient_color +
		diffuse_color * lambertian * light_color * light_power / distance +
		specular_color * specular  * light_color * light_power / distance,
		1.f);

	/********** Output color **********/
	color = vec4(c.x, c.y, c.z, c.w);
	// color = pow(color, vec4(1.f / 2.2f)); // Gamma correction
}

}
}
