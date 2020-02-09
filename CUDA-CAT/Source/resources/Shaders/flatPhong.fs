#version 450

uniform vec3 u_color;
uniform vec3 u_viewpos;

out vec4 FragColor;

in vec3 vPos;
in vec3 vNormal;

void main()
{
	vec3 viewDir = normalize(u_viewpos - vPos);
	vec3 normal = normalize(vNormal);

	float diff = max(dot(normal, -vec3(.2, -1, 0)), 0.0); // hard code light dir
	float spec = max(dot(normal, viewDir), 0.0) * .2;

	FragColor = vec4(u_color * (spec + diff), 1.0);
}