#include "glm/glm.hpp"

namespace cupix {

namespace kernel {

extern __constant__ __device__ int w, h;
extern __constant__ __device__ float time;

#define iTime time
#define iResolution vec2(w, h)

using namespace glm;

__device__ // https://www.shadertoy.com/view/Xt2czt
vec4 FlickeringDots(vec2 fragCoord) {
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
	r +=  (.5f+.5f*cos(a)) / N;	r = floor(N*r)/N;
	o = (1.f-r)*vec4(.5f,1.f,1.5f,1.f);
	return o;
}
#undef N

//#define BALLS
__device__ // https://www.shadertoy.com/view/4dsSzS
vec4 Mandeltunnel(vec2 fragCoord) {
	vec2 uv = -1.f + 2.f*fragCoord / iResolution;
	uv.x *= iResolution.x/iResolution.y;

	vec2 c = vec2(-uv.y-0.3f, uv.x)*0.13f;
	vec2 z = vec2(0.f);

	float sinTime = sin(iTime*1.5f);
	vec2 cinc = vec2(sinTime*0.0001f, cos(iTime)*0.0003f);
	float m = clamp((abs(uv.x)+uv.y+0.25f)*2.f,-1.f,1.f);

	float ni = 0.f;
	vec4 col = vec4(0.f,0.f,0.f,1.f);
	bool hit = false;
	for(float i=0.f; i<50.f; i++) {
		if (hit) continue;

		float f = sin(time);
		z = vec2(z.x*z.x - z.y*z.y, 2.f*z.x*z.y) + i*c;
		float r = dot(z,z);

		if(r > 1.3f+f*m - sinTime*0.1f) {
			hit = true;
			ni = 1.f - ni;

			col = vec4(1.f+f,ni,ni,1.f)*ni;

			#ifdef BALLS
				col *= 0.5f*r-ni*0.75f;
			#endif

		}
		ni += 0.02f;
		c += cinc;
	}
	return col;
}

#define MAX_ITERS 150.f
__device__ // https://www.shadertoy.com/view/Mdj3Rh
vec4 MandelbrotsDarkerSide(vec2 fragCoord) {
	vec2 uv = fragCoord / iResolution;
	vec2 c = (2.f * uv - 1.f)
			 * vec2(iResolution.x / iResolution.y, 1.f);
	// view
	c.x -=.3f;
	c *= 1.5f;

	vec2 z;

	float iters = 20.f*(1.f-cos(((.2f*iTime*6.f+6.f*log(.5f*iTime*6.f+1.f))*.9f)*.05f));

	for (float i = 0.f; i < MAX_ITERS; ++i) {
		if( i > iters ) continue;
		float alpha = clamp(iters-float(i),0.f,1.f);
		alpha = smoothstep(0.f,1.f,alpha);
		vec2 newz = vec2(z.x  * z.x - z.y * z.y, 2.f * z.x * z.y ) + c;
		// simple linear interpolation
		z = (1.f-alpha)*z + alpha*newz;
	}

	float col = (z.x*z.x+z.y*z.y);

	col = pow(col,.35f);
	col = clamp(col,0.f,1.1f);

	float vign = (1.f-.5f*dot(uv-.5f,uv-.5f));
	return vec4(vec3(.95f,.95f,.8f)*(col) * vign, 1.f);
}
#undef MAX_ITERS

__device__ // https://www.shadertoy.com/view/4dX3Rn
vec4 DeformFlower(vec2 fragCoord) {
	vec2 p = (2.0f*fragCoord-iResolution)/min(iResolution.y,iResolution.x);

	float a = atan(p.x,p.y);
	float r = length(p)*(0.8f+0.2f*sin(0.3f*iTime));

	float w = cos(2.0f*iTime-r*2.0f);
	float h = 0.5f+0.5f*cos(12.0f*a-w*7.0f+r*8.0f+ 0.7f*iTime);
	float d = 0.25f+0.75f*pow(h,1.0f*r)*(0.7f+0.3f*w);

	float f = sqrt(1.0f-r/d)*r*2.5f;
	f *= 1.25f+0.25f*cos((12.0f*a-w*7.0f+r*8.0f)/2.0f);
	f *= 1.0f - 0.35f*(0.5f+0.5f*sin(r*30.0f))*(0.5f+0.5f*cos(12.0f*a-w*7.0f+r*8.0f));

	vec3 col = vec3( f,
					 f-h*0.5f+r*.2f + 0.35f*h*(1.0f-r),
					 f-h*r + 0.1f*h*(1.0f-r) );
	col = clamp( col, 0.0f, 1.0f );

	vec3 bcol = mix( 0.5f*vec3(0.8f,0.9f,1.0f), vec3(1.0f), 0.5f+0.5f*p.y );
	col = mix( col, bcol, smoothstep(-0.3f,0.6f,r-d) );

	return vec4( col, 1.0f );
}

__device__ // https://www.shadertoy.com/view/XsfGRn
vec4 Heart2D(vec2 fragCoord) {
	vec2 p = (2.0f*fragCoord-iResolution)/min(iResolution.y,iResolution.x);

	// background color
	vec3 bcol = vec3(1.0f,0.8f,0.7f-0.07f*p.y)*(1.0f-0.25f*length(p));

	// animate
	float tt = mod(iTime,1.5f)/1.5f;
	float ss = pow(tt,.2f)*0.5f + 0.5f;
	ss = 1.0f + ss*0.5f*sin(tt*6.2831f*3.0f + p.y*0.5f)*exp(-tt*4.0f);
	p *= vec2(0.5f,1.5f) + ss*vec2(0.5f,-0.5f);

	// shape
#if 0
	p *= 0.8f;
	p.y = -0.1f - p.y*1.2f + abs(p.x)*(1.0f-abs(p.x));
	float r = length(p);
	float d = 0.5f;
#else
	p.y -= 0.25f;
	float a = atan(p.x,p.y)/3.141593f;
	float r = length(p);
	float h = abs(a);
	float d = (13.0f*h - 22.0f*h*h + 10.0f*h*h*h)/(6.0f-5.0f*h);
#endif

	// color
	float s = 0.75f + 0.75f*p.x;
	s *= 1.0f-0.4f*r;
	s = 0.3f + 0.7f*s;
	s *= 0.5f+0.5f*pow( 1.0f-clamp(r/d, 0.0f, 1.0f ), 0.1f );
	vec3 hcol = vec3(1.0f,0.5f*r,0.3f)*s;

	vec3 col = mix( bcol, hcol, smoothstep( -0.01f, 0.01f, d-r) );

	return vec4(col,1.0f);
}

}
}
