# cpupix

A CPU rasterizer.

The details will be listed...

# Results

* depth test and blending

![depth-t_blend-f](result/image/depth_blend/depth-t_blend-f.png)

![depth-f_blend-t](result/image/depth_blend/depth-f_blend-t.png)

* face culling

![cull_cube_front_back](result/image/face_culling/cull_flower_front_back.png)

![cull_cube_front](result/image/face_culling/cull_flower_front.png)

![cull_cube_back](result/image/face_culling/cull_flower_back.png)

* Gamma correction

![gamma_rgb_f](result/image/gamma_correction/gamma_rgb_f.png)

![gamma_rgb_t](result/image/gamma_correction/gamma_rgb_t.png)

* texture

![texture_fruit](result/image/texture/texture_fruit.png)

* lighting, Phong / Blinn-Phong shading

![torus_smooth_Blinn](result/image/lighting/torus_smooth_Blinn-Phong.png)

![cow_smooth_Blinn-Phong](result/image/lighting/cow_smooth_Blinn-Phong.png)

* AA

No AA
![suzanne_normal_noaa](result/image/aa/suzanne_normal_noaa.png)

MSAA
![suzanne_normal_msaa](result/image/aa/suzanne_normal_msaa.png)

SSAA
![suzanne_normal_ssaa](result/image/aa/suzanne_normal_ssaa.png)

* Shadertoy

![FlickeringDots](result/image/shadertoy/FlickeringDots.png)

![DeformFlower](result/image/shadertoy/DeformFlower.png)

* Large number of faces

bunny, 69451 faces, 19-20 FPS

![bunny](result/image/large_number_of_faces/bunny-69451.png)

sponza, 279105 faces, 3 FPS

![sponza_top](result/image/large_number_of_faces/sponza_top-279105.png)
