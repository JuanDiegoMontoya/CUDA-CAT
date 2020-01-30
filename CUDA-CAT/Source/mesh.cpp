#include "stdafx.h"
#include "mesh.h"
#include "texture.h"
#include "shader.h"

using namespace std;


Mesh::Mesh(vector<Vertex> vertices, vector<unsigned> indices, 
	vector<Texture> textures, ProjectorFn fn, TexEntity entity)
{
	this->vertices = vertices;
	this->indices = indices;
	this->textures = textures;

  if (indices.size() > 0)
  {
    indexed = true;
    vertCount = indices.size();
  }
  else
  {
    indexed = false;
    vertCount = vertices.size();
  }

	generateTexcoords(fn, entity);
	setupBuffers();
}


// copy constructor that changes the texture mapping type
Mesh::Mesh(const Mesh& other, ProjectorFn fn, TexEntity entity)
{
	this->vertices = other.vertices;
	this->indices = other.indices;
	this->textures = other.textures;

	if (indices.size() > 0)
	{
		indexed = true;
		vertCount = indices.size();
	}
	else
	{
		indexed = false;
		vertCount = vertices.size();
	}

	generateTexcoords(fn, entity);
	setupBuffers();
}


void Mesh::Draw()
{
  vao->Bind();
  if (indexed)
    glDrawElements(GL_TRIANGLES, vertCount, GL_UNSIGNED_INT, 0);
  else
    glDrawArrays(GL_TRIANGLES, 0, vertCount);
  vao->Unbind();
}


// modifies vertices (the texture part)
void Mesh::generateTexcoords(ProjectorFn func, TexEntity entity)
{
	// generate bounding box
	glm::vec3 minbox(std::numeric_limits<float>::max());
	glm::vec3 maxbox(std::numeric_limits<float>::min());
	for (const auto& v : vertices)
	{
		if (v.Position.x > maxbox.x)
			maxbox.x = v.Position.x;
		if (v.Position.x < minbox.x)
			minbox.x = v.Position.x;
		if (v.Position.y > maxbox.y)
			maxbox.y = v.Position.y;
		if (v.Position.y < minbox.y)
			minbox.y = v.Position.y;
		if (v.Position.z > maxbox.z)
			maxbox.z = v.Position.z;
		if (v.Position.z < minbox.z)
			minbox.z = v.Position.z;
	}
	glm::vec3 center = (maxbox + minbox) / 2.f;
	min = minbox;
	max = maxbox;
	mid = center;
	// translate bounding box to center
	maxbox -= center;
	minbox -= center;

	switch (func)
	{
	case ProjectorFn::Cube:
		for (auto& v : vertices)
		{
			glm::vec3 absVec(0);
			if (entity == TexEntity::Normal)
				absVec = glm::abs(v.Normal);
			else
				absVec = glm::abs(v.Position);
			glm::vec2 UV(0);

			// +-X
			if (absVec.x >= absVec.y && absVec.x >= absVec.z)
			{
				(v.Position.x < 0.0) ? (UV.s = v.Position.z) : (UV.s = -v.Position.z);
				UV.t = v.Position.y;
				UV.s /= maxbox.z;
				UV.t /= maxbox.y;
				//UV.s /= absVec.x;
				//UV.t /= absVec.x;
			}
			// +-Y
			else if (absVec.y >= absVec.x && absVec.y >= absVec.z)
			{
				(v.Position.y < 0.0) ? (UV.t = v.Position.z) : (UV.t = -v.Position.z);
				UV.s = v.Position.x;
				UV.s /= maxbox.x;
				UV.t /= maxbox.z;
			}
			// +-Z
			else
			{
				(v.Position.z < 0.0) ? (UV.s = -v.Position.x) : (UV.s = v.Position.x);
				UV.t = v.Position.y;
				UV.s /= maxbox.x;
				UV.t /= maxbox.y;
			}

			UV = (UV + 1.f) / 2.f;
			v.TexCoords = UV;
		}
		break;
	case ProjectorFn::Sphere:
		for (auto& v : vertices)
		{
			glm::vec3 tpos;
			if (entity == TexEntity::Position)
				tpos = v.Position - center;
			else if (entity == TexEntity::Normal)
				tpos = v.Normal;
			float theta = glm::atan(tpos.y / tpos.x);
			float Z = (tpos.z - minbox.z) / (maxbox.z - minbox.z);
			float r = glm::length(tpos);
			float phi = glm::acos(tpos.z / r);
			float U = theta / glm::two_pi<float>(); // convert to 0-1 range
			float V = (glm::pi<float>() - phi) / (glm::pi<float>());
			v.TexCoords = { U, V };
		}
		break;
	case ProjectorFn::Cylinder:
		for (auto& v : vertices)
		{
			glm::vec3 tpos;
			if (entity == TexEntity::Position)
				tpos = v.Position - center;
			else if (entity == TexEntity::Normal)
				tpos = v.Normal;
			float theta = glm::atan(tpos.y / tpos.x);
			float Z = (tpos.z - minbox.z) / (maxbox.z - minbox.z);
			float U = theta / glm::two_pi<float>(); // convert to 0-1 range
			float V = Z;
			v.TexCoords = { U, V };
		}
		break;
	default:
		break;
	}
}


void Mesh::setupBuffers()
{
	vao = new VAO();
	vbo = new VBO(&vertices[0], vertices.size() * sizeof(Vertex));

  // pos-normal-tex layout
	VBOlayout layout;
	layout.Push<GLfloat>(3);
	layout.Push<GLfloat>(3);
	layout.Push<GLfloat>(2);
	//layout.Push<GLfloat>(1);
	vao->AddBuffer(*vbo, layout);
  
	vbo->Unbind();
  if (indexed)
	  ibo = new IBO((GLuint*)&indices[0], vertCount);
	vao->Unbind();
}
