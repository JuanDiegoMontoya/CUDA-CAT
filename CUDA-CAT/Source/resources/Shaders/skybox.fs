#version 450 core
out vec4 FragColor;

in vec3 TexCoords;

uniform samplerCube skybox;

void main()
{
  vec3 temp = TexCoords;
  //temp.x = 1.0 - temp.x;
	FragColor = texture(skybox, temp);
}