#version 400 core

in Attribute {
	vec3 position;
	vec3 normal;
} vertexIn;

uniform mat4 mv;

out vec4 color;

// Blinn-Phong shading
vec3 BlinnPhong(vec3 light_position, float light_power, int n_light) {
	vec3 light_color = vec3(1.f, 1.f, 1.f);

	const vec3 diffuse_color  = vec3(0.9f, 0.3f, 0.3f);
	const vec3 ambient_color  = vec3(0.4f, 0.4f, 0.4f) * diffuse_color;
	const vec3 specular_color = vec3(0.3f, 0.3f, 0.3f);
	const float shininess = 16.0;

	vec3 light_direction = light_position - vertexIn.position;
	float distance = length(light_direction);
	distance *= distance;

	light_direction = normalize(light_direction);
	vec3 normal = normalize(vertexIn.normal);
	float cos_theta = clamp(dot(normal, light_direction), 0.f, 1.f);
	float lambertian = clamp(cos_theta, 0.f, 1.f);

	vec3 eye_direction = normalize(-vertexIn.position);
	vec3 half_direction = normalize(light_direction + eye_direction);
	float cos_alpha = dot(half_direction, normal);
	float specular = pow(clamp(cos_alpha, 0.f, 1.f), shininess);

	vec3 color =
		ambient_color / n_light +
		diffuse_color * lambertian * light_color * light_power / distance +
		specular_color * specular  * light_color * light_power / distance;

	return color;
}


vec3 Lighting() {
	vec3 light_position_0 = (mv * vec4( 20.f, 20.f, 10.f, 1.f)).xyz;
	vec3 light_position_1 = (mv * vec4(-20.f, 10.f,-10.f, 1.f)).xyz;

	vec3 color = vec3(0.f);

	color += BlinnPhong(light_position_0, 40.f, 2);
	color += BlinnPhong(light_position_1, 30.f, 2);

	return color;
}

void main() {
	/********** Visualization of normal **********/
	// vec4 c = vec4(in.normal, 0.f);
	// vec4 c = vec4(abs(vertexIn.normal), 1.f);
	vec4 c = vec4(vertexIn.normal * 0.5f + 0.5f, 0.5f); // normalized

	/********** Phong/Blinn-Phong shading **********/
	// vec4 c = vec4(Lighting(), 1.f);

	/********** Output color **********/
	color = c;
	// color = pow(color, vec4(1.f / 2.2f)); // Gamma correction
}
