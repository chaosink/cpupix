#include "CPUPix.hpp"

namespace cpupix {

namespace kernel {

extern Texture2D texture;
extern int w, h;
extern int n_light;
extern Light light[4];
extern glm::mat4 mv;
extern float time;
extern bool toggle;

using namespace glm;

/********** Shadertoy **********/
vec4 FlickeringDots(vec2);
vec4 Quadtree(vec2);
vec4 Sunflower(vec2);
vec4 Mandeltunnel(vec2);
vec4 MandelbrotsDarkerSide(vec2);
vec4 DeformFlower(vec2);
vec4 Heart2D(vec2);

vec4 BlinnPhong(FragmentIn &in, Light &light) {
	vec3 light_position = mv * vec4(light.position[0], light.position[1], light.position[2], 1.0f);
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

vec4 Lighting(FragmentIn &in) {
	vec4 color;
	for(int i = 0; i < n_light; i++) {
		Light l = light[i];
		if(i == 0) { // move light 0
			l.position[0] = sinf(time) * 4.f;
			l.position[2] = cosf(time) * 4.f;
		}
		color += BlinnPhong(in, l);
	}
	return color;
}

void FragmentShader(FragmentIn &in, vec4 &color) {
	/********** Visualization of normal **********/
	// vec4 c = vec4(in.normal, 0.f);
	// vec4 c = vec4(abs(in.normal), 1.f);
	vec4 c = vec4(in.normal * 0.5f + 0.5f, 0.5f); // normalized

	/********** Visualization of uv **********/
	// vec4 c = vec4(in.uv, 0.f, 0.f);

	/********** Visualization of z **********/
	// float depth = pow(in.z * 0.5f + 0.5f, 5);
	// vec4 c = vec4(depth, depth, depth, 1.f);

	/********** Texture sampling **********/
	// vec4 c = vec4(texture.Sample(in.uv.s, 1 - in.uv.t), 1.f);

	/********** Phong/Blinn-Phong shading **********/
	// vec4 c = Lighting(in);

	/********** Shadertoy **********/
	// vec4 c = FlickeringDots(in.coord);
	// vec4 c = Quadtree(in.coord);
	// vec4 c = Sunflower(in.coord);
	// vec4 c = Mandeltunnel(in.coord);
	// vec4 c = MandelbrotsDarkerSide(in.coord);
	// vec4 c = DeformFlower(in.coord);
	// vec4 c = Heart2D(in.coord);

	/********** Output color **********/
	color = vec4(c.x, c.y, c.z, c.w);
	// color = pow(color, vec4(1.f / 2.2f)); // Gamma correction
}

}
}
