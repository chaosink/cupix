#pragma once

#include "cupix.hpp"

namespace cupix {

namespace cu {

texture<uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat> texture;
__constant__ __device__ int w, h;
__constant__ __device__ int n_light;
__constant__ __device__ Light light[4];
__constant__ __device__ float mvp[16];
__constant__ __device__ float mv[16];
__constant__ __device__ float time;
__constant__ __device__ bool toggle;

__constant__ __device__ int n_triangle;
__constant__ __device__ bool depth_test = true;
__constant__ __device__ bool blend = false;
__constant__ __device__ unsigned char clear_color[4];

__constant__ __device__ unsigned char bit[8] = {
	0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01};
__device__ char zb16[zb16_file_size];

}

}
