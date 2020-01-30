#include "stdafx.h"
#include "Line.h"
#include "vao.h"
#include "vbo.h"
#include "shader.h"
#include "camera.h"
#include "pipeline.h"


Line::Line(const glm::vec3 & wpos, const glm::vec3 & dir,
						const glm::vec3 & tclr, const glm::vec3 & bclr)
	: pos_(wpos), dir_(dir), tclr_(tclr), bclr_(bclr)
{
	glm::vec3 pdata[2]; // start/end pos
	glm::vec3 cdata[2]; // start/end color

	pdata[0] = pos_;
	pdata[1] = pos_ + dir_;
	cdata[0] = bclr;
	cdata[1] = tclr;

	vao_ = new VAO();
	positions_ = new VBO(pdata, sizeof(pdata), GL_STATIC_DRAW);
	colors_ = new VBO(cdata, sizeof(cdata), GL_STATIC_DRAW);

	// do raw GL calls since AddBuffer() doesn't work with multiple buffers
	positions_->Bind();
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
	glEnableVertexAttribArray(0);

	colors_->Bind();
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
	glEnableVertexAttribArray(1);
}

Line::~Line()
{
	delete vao_;
	delete positions_;
	delete colors_;
}

void Line::Draw() const
{
	vao_->Bind();
	glDrawArrays(GL_LINES, 0, 2);
}


//#################### instanced line stuff ###########################

LineInstanced::LineInstanced(const glm::vec3& pos, const glm::vec3 & dir, const glm::vec3 & tclr, const glm::vec3 & bclr)
	: pos_(pos), dir_(dir), tclr_(tclr), bclr_(bclr)
{
	count_++;
	lines_.push_back(this);
}

LineInstanced::~LineInstanced()
{
	count_--;
	// remove self from list
	std::remove_if(lines_.begin(), lines_.end(),
		[this](LineInstanced* line)->bool { return line == this; });
}

void LineInstanced::GenBuffers()
{
	if (vao_)
		delete vao_;
	if (positions_)
		delete positions_;
	if (colors_)
		delete colors_;
	if (models_)
		delete models_;

	vao_ = new VAO();

	// generate VBOs
	std::vector<glm::vec3> positions;	// bottom then top
	std::vector<glm::vec3> colors;		// bottom then top color
	std::vector<glm::mat4> models;		// transform matrix, single
	for (auto l : lines_)
	{
		positions.push_back(l->pos_);
		positions.push_back(l->pos_ + l->dir_);
		colors.push_back(l->bclr_);
		colors.push_back(l->tclr_);
		models.push_back(l->model_);
	}

	positions_ = new VBO(&positions[0], positions.size() * sizeof(glm::vec3), GL_STATIC_DRAW);
	colors_ = new VBO(&colors[0], colors.size() * sizeof(glm::vec3), GL_STATIC_DRAW);
	models_ = new VBO(&models[0], models.size() * sizeof(glm::mat4), GL_STATIC_DRAW);

	// set up buffer pointers
	positions_->Bind();
	glEnableVertexAttribArray(0); // model space position
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);

	colors_->Bind();
	glEnableVertexAttribArray(1); // color
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);

	models_->Bind(); // transformation matrix
	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(glm::vec4), (void*)0);
	glEnableVertexAttribArray(3);
	glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(glm::vec4), (void*)sizeof(glm::vec4));
	glEnableVertexAttribArray(4);
	glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(glm::vec4), (void*)(2 * sizeof(glm::vec4)));
	glEnableVertexAttribArray(5);
	glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(glm::vec4), (void*)(3 * sizeof(glm::vec4)));

	// update these with each vertex
	glVertexAttribDivisor(0, 0);
	glVertexAttribDivisor(1, 0);
	// update elements w/ each line instance
	glVertexAttribDivisor(2, 1);
	glVertexAttribDivisor(3, 1);
	glVertexAttribDivisor(4, 1);
	glVertexAttribDivisor(5, 1);
}

void LineInstanced::DrawAll()
{
	ShaderPtr shdr = Shader::shaders["lineInstanced"];
	shdr->Use();
	shdr->setMat4("u_view", Render::GetCamera()->GetView());
	shdr->setMat4("u_proj", Render::GetCamera()->GetProj());
	vao_->Bind();
	glDrawArraysInstanced(GL_LINES, 0, 2, count_);
}

void LineInstanced::Update(glm::mat4& mdl)
{
	// update VBOs
	model_ = mdl;
	// get position of this line in buffers
	int index = 0;
	for (auto l : lines_)
	{
		if (l == this)
			break;
		index++;
	}

	glm::vec3 pdata[2]; // start/end pos
	glm::vec3 cdata[2]; // start/end color

	pdata[0] = pos_;
	pdata[1] = pos_ + dir_;
	cdata[0] = bclr_;
	cdata[1] = tclr_;

	positions_->Bind();
	glBufferSubData(GL_ARRAY_BUFFER, index * 2 * sizeof(glm::vec3), 2 * sizeof(glm::vec3), pdata);
	colors_->Bind();
	glBufferSubData(GL_ARRAY_BUFFER, index * 2 * sizeof(glm::vec3), 2 * sizeof(glm::vec3), cdata);
	models_->Bind();
	glBufferSubData(GL_ARRAY_BUFFER, index * sizeof(glm::mat4), sizeof(glm::mat4), &lines_[index]->model_[0][0]);
}

//################################# Line Pool Stuff ###############################

LinePool::LinePool(const std::vector<glm::vec3>& wposs, const std::vector<glm::vec3>& dirs,
	const std::vector<glm::vec3>& tclrs, const std::vector<glm::vec3>& bclrs)
{
	count_ = wposs.size() * 2;
	std::vector<glm::vec3> positions; // start + end
	std::vector<glm::vec3> colors;    // start + end

	// the inputs better be the same size or i'll be mad
	for (int i = 0; i < wposs.size(); i++)
	{
		positions.push_back(wposs[i]);
		positions.push_back(wposs[i] + dirs[i]);
		colors.push_back(bclrs[i]);
		colors.push_back(tclrs[i]);
	}

	vao_ = new VAO();
	positions_ = new VBO(&positions[0], positions.size() * sizeof(glm::vec3), GL_STATIC_DRAW);
	colors_ =    new VBO(&colors[0]   , colors.size() * sizeof(glm::vec3)   , GL_STATIC_DRAW);

	positions_->Bind();
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
	glEnableVertexAttribArray(0);

	colors_->Bind();
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
	glEnableVertexAttribArray(1);
}

void LinePool::Draw() const
{
	vao_->Bind();
	glDrawArrays(GL_LINES, 0, count_);
}
