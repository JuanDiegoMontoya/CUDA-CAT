#pragma once
#include "texture.h"
#include "vao.h"
#include "ibo.h"
#include "vbo.h"

struct Vertex
{
	Vertex(glm::vec3 p = glm::vec3(0), glm::vec3 n = glm::vec3(0), 
		glm::vec2 t = glm::vec2(0)) 
		: Position(p), Normal(n), TexCoords(t)/*, index(0)*/ {}
	glm::vec3 Position;
	glm::vec3 Normal;
	glm::vec2 TexCoords;
	//int index;
};

enum struct ProjectorFn : int
{
	Cube,
	Sphere,
	Cylinder
};

enum struct TexEntity
{
	Position = 0,
	Normal = 1
};

class Mesh
{
public:

	explicit Mesh(std::vector<Vertex> vertices, std::vector<unsigned> indices, 
		std::vector<Texture> textures, ProjectorFn fn = ProjectorFn::Sphere,
		TexEntity entity = TexEntity::Position);
	Mesh(const Mesh& other, ProjectorFn fn, TexEntity entity);

	void Draw();

	const std::vector<Vertex>& GetVertices() const { return vertices; }
	const std::vector<GLuint>& GetIndices() const { return indices; }
	const std::vector<Texture>& GetTextures() const { return textures; }

	// unadjusted min, max, and center bounding coords
	glm::vec3 min, max, mid;
protected:
	Mesh() = delete;

	VAO* vao;
	VBO* vbo;
	IBO* ibo;

  unsigned vertCount;
	bool indexed; // whether to use DrawElements or DrawArrays
	void setupBuffers();
	void generateTexcoords(ProjectorFn fn, TexEntity entity);

	std::vector<Vertex> vertices;
	std::vector<GLuint> indices;
	std::vector<Texture> textures;
};