#pragma once

class VAO;
class VBO;

// rendering class
class Line
{
public:
	Line(const glm::vec3& pos, const glm::vec3& dir,
				const glm::vec3& tclr, const glm::vec3& bclr);
	~Line();

	// the user must provide a shader and correct uniforms to draw
	void Draw() const;

private:
	glm::vec3 pos_; // model space position of the line
	glm::vec3 dir_;	// direction AND length (user might want to normalize)
	glm::vec3 tclr_;// top color
	glm::vec3 bclr_;// bottom color

	// rendering
	VAO* vao_;
	VBO* positions_;
	VBO* colors_;
};

// renders more quickly but with extra overhead
// operates under the assumption that rarely will new lines be added
class LineInstanced
{
public:
	LineInstanced(const glm::vec3& pos, const glm::vec3& dir,
		const glm::vec3& tclr, const glm::vec3& bclr);
	~LineInstanced();

	static void GenBuffers();
	static void DrawAll();
	void Update(glm::mat4& mdl);
private:

	static inline unsigned count_ = 0; // instances
	static inline std::vector<LineInstanced*> lines_;

	glm::mat4 model_; // transform
	glm::vec3 pos_; // world space position
	glm::vec3 dir_;
	glm::vec3 tclr_;
	glm::vec3 bclr_;

	static inline VAO* vao_;
	static inline VBO* positions_;
	static inline VBO* colors_;
	static inline VBO* models_;
};

// many lines in one buffer (efficient)
// like lineinstanced, but for one object, and not instanced
class LinePool
{
public:
	LinePool(
		const std::vector<glm::vec3>& wposs, const std::vector<glm::vec3>& dirs,
		const std::vector<glm::vec3>& tclrs, const std::vector<glm::vec3>& bclrs
	);
	~LinePool()
	{
		delete vao_;
		delete positions_;
		delete colors_;
	}

	void Draw() const;

private:
	VAO* vao_;
	VBO* positions_;
	VBO* colors_;

	GLuint count_ = 0;
};