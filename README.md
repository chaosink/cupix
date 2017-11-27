# CUPix

A CUDA rasterizer.

The details will be listed...

# Results

* depth test and blending

![depth-t_blend-f.png](result/image/depth_blend/depth-t_blend-f.png)

![depth-f_blend-t.png](result/image/depth_blend/depth-f_blend-t.png)

* face culling

![cull_cube_back.png](result/image/face_culling/cull_cube_back.png)

![cull_cube_front.png](result/image/face_culling/cull_cube_front.png)

![cull_cube_front_back.png](result/image/face_culling/cull_cube_front_back.png)

* Gamma correction

![gamma_rgb_f.png](result/image/gamma_correction/gamma_rgb_f.png)

![gamma_rgb_t.png](result/image/gamma_correction/gamma_rgb_t.png)

* lighting, Phong / Blinn-Phong shading

![suzanne_low_Phong.png](result/image/lighting/suzanne_low_Phong.png)

![cow_smooth_Blinn-Phong.png](result/image/lighting/cow_smooth_Blinn-Phong.png)

* anti-aliasing

No AA
![suzanne_normal_noaa.png](result/image/aa/suzanne_normal_noaa.png)

MSAA
![suzanne_normal_msaa.png](result/image/aa/suzanne_normal_msaa.png)

SSAA
![suzanne_normal_ssaa.png](result/image/aa/suzanne_normal_ssaa.png)

* Shadertoy

![FlickeringDots.png](result/image/shadertoy/FlickeringDots.png)

![DeformFlower.png](result/image/shadertoy/DeformFlower.png)

More imgae results and some video results can be found in the `result` directory.
