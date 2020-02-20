#include "stdafx.h"
#include "shader.h"
#include "camera.h"
#include "level.h"
#include "pipeline.h"
#include "mesh.h"
#include "texture.h"
#include <chrono>
#include <execution>
#include "input.h"
#include "settings.h"
#include <functional>
#include <filesystem>

Level::Level(std::string name)
{
	name_ = name;
}

Level::~Level()
{
	for (auto& cam : cameras_)
		delete cam;
}

// for now this function is where we declare objects
void Level::Init()
{
	cameras_.push_back(new Camera(kControlCam));
	Render::SetCamera(cameras_[0]);
	renderer_.Init();	
}

// update every object in the level
void Level::Update(float dt)
{
	if (Input::Keyboard().pressed[GLFW_KEY_GRAVE_ACCENT])
		activeCursor = !activeCursor;
	glfwSetInputMode(game_->GetWindow(), GLFW_CURSOR, activeCursor ? GLFW_CURSOR_NORMAL : GLFW_CURSOR_DISABLED);

	// update each camera
	if (!activeCursor)
		for (auto& cam : cameras_)
			cam->Update(dt);

	renderer_.DrawAll();
	DrawImGui();
}

void Level::DrawImGui()
{
	{
		ImGui::Begin("Shaders");
		if (ImGui::Button("Recompile Postprocess Shader"))
		{
			//delete Shader::shaders["postprocess"];
			Shader::shaders["postprocess"] = new Shader("postprocess.vs", "postprocess.fs");
		}
		if (ImGui::Button("Recompile Phong Shader"))
		{
			delete Shader::shaders["flatPhong"];
			Shader::shaders["flatPhong"] = new Shader("flatPhong.vs", "flatPhong.fs");
		}

		ImGui::End();
	}

	{
		ImGui::Begin("Info");

		ImGui::Text("FPS: %.0f (%.1f ms)", 1.f / game_->GetDT(), 1000. * game_->GetDT());
		ImGui::NewLine();
		glm::vec3 pos = Render::GetCamera()->GetPos();
		//ImGui::Text("Camera Position: (%.2f, %.2f, %.2f)", pos.x, pos.y, pos.z);
		if (ImGui::InputFloat3("Camera Position", &pos[0], 2))
			Render::GetCamera()->SetPos(pos);
		pos = Render::GetCamera()->front;
		ImGui::Text("Camera Direction: (%.2f, %.2f, %.2f)", pos.x, pos.y, pos.z);
		glm::vec3 eu = Render::GetCamera()->GetEuler();
		ImGui::Text("Euler Angles: %.0f, %.0f, %.0f", eu.x, eu.y, eu.z);
		pos = pos * .5f + .5f;
		ImGui::SameLine();
		ImGui::ColorButton("visualization", ImVec4(pos.x, pos.y, pos.z, 1.f));

		ImGui::NewLine();
		ImGui::Text("Cursor: %s", !activeCursor ? "False" : "True");

		ImGui::End();
	}

	{
		ImGui::Begin("Simulation");
		ImGui::Text("Game of Life");
		if (ImGui::RadioButton("2D##gol", renderer_.automaton == (void*)&renderer_.GoL2))
			renderer_.automaton = reinterpret_cast<CAInterface*>(&renderer_.GoL2), renderer_.automaton->Init();
		ImGui::SameLine();
		if (ImGui::RadioButton("3D##gol", renderer_.automaton == (void*)&renderer_.GoL))
			renderer_.automaton = reinterpret_cast<CAInterface*>(&renderer_.GoL), renderer_.automaton->Init();

		ImGui::Text("Cave Generator");
		if (ImGui::RadioButton("2D##cave", renderer_.automaton == (void*)&renderer_.Caver))
			renderer_.automaton = reinterpret_cast<CAInterface*>(&renderer_.Caver), renderer_.automaton->Init();
		ImGui::SameLine();
		if (ImGui::RadioButton("3D##cave", renderer_.automaton == (void*)&renderer_.Caver4))
			renderer_.automaton = reinterpret_cast<CAInterface*>(&renderer_.Caver4), renderer_.automaton->Init();

		ImGui::Text("Water");
		if (ImGui::RadioButton("Small##water", renderer_.automaton == (void*)&renderer_.Water5))
			renderer_.automaton = reinterpret_cast<CAInterface*>(&renderer_.Water5), renderer_.automaton->Init();
		ImGui::SameLine();
		if (ImGui::RadioButton("Large##water", renderer_.automaton == (void*)&renderer_.Water6))
			renderer_.automaton = reinterpret_cast<CAInterface*>(&renderer_.Water6), renderer_.automaton->Init();

		ImGui::Checkbox("Pause Simulation", &renderer_.pauseSimulation);
		ImGui::SliderFloat("s/Update", &renderer_.updateFrequency, 0, 1, "%.2f");
		if (ImGui::Button("Re-init Sim"))
			renderer_.automaton->Init();
		if (ImGui::Button("1x Update Simulation"))
			renderer_.automaton->Update();
		if (ImGui::Button("20x Update Simulation"))
			for (int i = 0; i < 20; i++)
				renderer_.automaton->Update();
		if (ImGui::Button("200x Update Simulation"))
			for (int i = 0; i < 200; i++)
				renderer_.automaton->Update();
		if (ImGui::Button("2000x Update Simulation"))
			for (int i = 0; i < 2000; i++)
				renderer_.automaton->Update();
		ImGui::End();
	}

}