#include "cupix.hpp"

namespace cupix {

namespace cu {

extern texture<uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat> texture;
extern __constant__ __device__ int w, h;
extern __constant__ __device__ int n_light;
extern __constant__ __device__ Light light[4];
extern __constant__ __device__ float mvp[16];
extern __constant__ __device__ float mv[16];
extern __constant__ __device__ float time;
extern __constant__ __device__ bool toggle;

using namespace glm;

__device__
void VertexShader(VertexIn &in, VertexOut &out, Vertex &v) {
	mat4 m = *((mat4*)mvp);
	v.position = m * vec4(in.position, 1.f);

	m = *((mat4*)mv);
	out.position = m * vec4(in.position, 1.f);
	out.normal   = m * vec4(in.normal, 0.f);
	out.color	= in.color;
	out.uv	   = in.uv;
}

__device__
vec4 BlinnPhong(FragmentIn &in, Light &light) {
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
	float cos_theta = dot(light_direction, normal);
	// cos_theta = cos_theta * 0.5f + 0.5f; // normalized shading
	float lambertian = clamp(cos_theta, 0.f, 1.f);

	vec3 eye_direction = normalize(-position);
	if(toggle) {
		/***** Blinn-Phong shading *****/
		vec3 half = normalize(light_direction + eye_direction);
		float cos_alpha = dot(half, normal);
		// cos_alpha = cos_alpha * 0.5f + 0.5f; // normalized shading
		specular = pow(clamp(cos_alpha, 0.f, 1.f), shininess);
	} else {
		/***** Phong shading *****/
		vec3 reflection = reflect(-light_direction, normal);
		float cos_alpha = dot(reflection, eye_direction);
		// cos_alpha = cos_alpha * 0.5f + 0.5f; // normalized shading
		specular = pow(clamp(cos_alpha, 0.f, 1.f), shininess / 4.f); // exponent is different
	}

	return vec4(
		ambient_color / float(n_light) +
		diffuse_color * lambertian * light_color * light_power / distance +
		specular_color * specular  * light_color * light_power / distance,
		1.f);
}

__device__
vec4 Lighting(FragmentIn &in) {
	vec4 c;
	for(int i = 0; i < n_light; i++) {
		Light l = light[i];
		if(i == 0) { // move light 0
			l.position[0] = sinf(time) * 4.f;
			l.position[2] = cosf(time) * 4.f;
		}
		c += BlinnPhong(in, l);
	}
	return c;
}

__device__ // https://www.shadertoy.com/view/Xt2czt
vec4 FlickeringDots(FragmentIn &in) {
	vec2 fragCoord(in.coord.x, in.coord.y);
	vec2 iResolution(w, h);
	vec2 uv = fragCoord - iResolution / 2.f;
	float d = dot(uv, uv);
	return vec4(0.5f + 0.5f * cos(d / 5.f + 10.f * time));
}

__device__ // https://www.shadertoy.com/view/lljSDy
vec4 Quadtree(vec2 U) {
	vec4 o;
	o -= o;
	float r=.1f, t=time, H = h;
	U /=  H;							// object : disc(P,r)
	vec2 P = .5f+.5f*vec2(cos(t),sin(t*.7f)), fU;
	U*=.5f; P*=.5f;						// unzoom for the whole domain falls within [0,1]^n
	o.b = .25f;							// backgroud = cold blue
	for (int i=0; i<7; i++) {			// to the infinity, and beyond ! :-)
		fU = min(U,1.f-U); if (min(fU.x,fU.y) < 3.f*r/H) { o--; break; } // cell border
		if (length(P-.5f) - r > .7f) break; // cell is out of the shape
				// --- iterate to child cell
		fU = step(.5f,U);				// select child
		U = 2.f*U - fU;					// go to new local frame
		P = 2.f*P - fU;  r *= 2.f;

		o += .13f;						// getting closer, getting hotter
	}
	o.g *= smoothstep(.9f,1.f,length(P-U)/r); // draw object
	o.b *= smoothstep(.9f,1.f,length(P-U)/r);
	return o;
}

#define N 10.f
__device__ // https://www.shadertoy.com/view/4sjSRt
vec4 Sunflower(vec2 u) {
	vec4 o;
	o.x = w; o.y = h;
	u = (u+u-vec2(w,h))/o.y;
	//u = 2.*(u / iResolution.y -vec2(.9,.5));
	float t = time,
		r = length(u), a = atan(u.y,u.x),
		i = floor(r*N);
	a *= floor(pow(128.f,i/N)); 	 a += 20.f*sin(.5f*t)+123.34f*i-100.f*r*cos(.5f*t); // (r-0.*i/N)
	r +=  (.5f+.5f*cos(a)) / N;    r = floor(N*r)/N;
	o = (1.f-r)*vec4(.5f,1.f,1.5f,1.f);
	return o;
}
#undef N

__device__
void FragmentShader(FragmentIn &in, vec4 &color) {
	/********** Visualization of normal **********/
	// vec4 c = vec4(in.normal, 0.f);
	// vec4 c = vec4(in.normal * 0.5f + 0.5f, 0.5f); // normalized

	/********** Visualization of uv **********/
	// vec4 c = vec4(in.uv, 0.f, 0.f);

	/********** Texture sampling **********/
	// float4 c = tex2D(texture, in.uv.s, 1 - in.uv.t);

	/********** Shadertoy - Flickering Dots **********/
	// vec4 c = FlickeringDots(in);

	/********** Shadertoy - Quadtree **********/
	// vec4 c = Quadtree(vec2(in.coord));

	/********** Shadertoy - Sunflower **********/
	vec4 c = Sunflower(vec2(in.coord));

	/********** Phong/Blinn-Phong shading **********/
	// vec4 c = Lighting(in);

	/********** Output color **********/
	color = vec4(c.x, c.y, c.z, c.w);
	// color = pow(color, vec4(1.f / 2.2f)); // Gamma correction
}

}
}
