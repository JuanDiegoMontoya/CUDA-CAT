#include "stdafx.h"
#include "shader.h"
#include "camera.h"
#include "input.h"
#include "Pipeline.h"
#include "Engine.h"
#include "Renderer.h"
#include "Interface.h"
#include "CellularAutomata.h"

namespace Interface
{
	namespace
	{
		bool activeCursor = true;
	}


	void Init()
	{
		Engine::PushRenderCallback(DrawImGui, 1);
		Engine::PushRenderCallback(Update, 2);
	}


	void Update()
	{
		if (Input::Keyboard().pressed[GLFW_KEY_GRAVE_ACCENT])
			activeCursor = !activeCursor;
		glfwSetInputMode(Engine::GetWindow(), GLFW_CURSOR, activeCursor ? GLFW_CURSOR_NORMAL : GLFW_CURSOR_DISABLED);
	}


	void DrawImGui()
	{
		{
			ImGui::Begin("Shaders");
			if (ImGui::Button("Recompile Phong Shader"))
			{
				delete Shader::shaders["flatPhong"];
				Shader::shaders["flatPhong"] = new Shader("flatPhong.vs", "flatPhong.fs");
			}
			if (ImGui::Button("Recompile Height Shader"))
			{
				delete Shader::shaders["height"];
				Shader::shaders["height"] = new Shader("height.vs", "height.fs");
			}
			if (ImGui::Button("Recompile HeightWater Shader"))
			{
				delete Shader::shaders["heightWater"];
				Shader::shaders["heightWater"] = new Shader("height.vs", "heightWater.fs");
			}

			ImGui::End();
		}

		{
			ImGui::Begin("Info");

			ImGui::Text("FPS: %.0f (%.1f ms)", 1.f / Engine::GetDT(), 1000. * Engine::GetDT());
			ImGui::NewLine();
			glm::vec3 pos = Renderer::GetPipeline()->GetCamera(0)->GetPos();
			if (ImGui::InputFloat3("Camera Position", &pos[0], 2))
				Renderer::GetPipeline()->GetCamera(0)->SetPos(pos);
			pos = Renderer::GetPipeline()->GetCamera(0)->front;
			ImGui::Text("Camera Direction: (%.2f, %.2f, %.2f)", pos.x, pos.y, pos.z);
			glm::vec3 eu = Renderer::GetPipeline()->GetCamera(0)->GetEuler();
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
			if (ImGui::RadioButton("2D##gol", Renderer::GetAutomatonIndex() == 0))
				Renderer::SetAutomatonIndex(0);
			ImGui::SameLine();
			if (ImGui::RadioButton("3D##gol", Renderer::GetAutomatonIndex() == 1))
				Renderer::SetAutomatonIndex(1);

			ImGui::Text("Cave Generator");
			if (ImGui::RadioButton("2D##cave", Renderer::GetAutomatonIndex() == 2))
				Renderer::SetAutomatonIndex(2);
			ImGui::SameLine();
			if (ImGui::RadioButton("3D##cave", Renderer::GetAutomatonIndex() == 3))
				Renderer::SetAutomatonIndex(2);

			ImGui::Text("Water");
			if (ImGui::RadioButton("Small##water", Renderer::GetAutomatonIndex() == 4))
				Renderer::SetAutomatonIndex(4);
			ImGui::SameLine();
			if (ImGui::RadioButton("Large##water", Renderer::GetAutomatonIndex() == 5))
				Renderer::SetAutomatonIndex(5);

			ImGui::Checkbox("Pause Simulation", &Engine::GetPauseRef());
			//ImGui::SliderFloat("s/Update", &renderer_.updateFrequency, 0, 1, "%.2f");
			if (ImGui::Button("Re-init Sim"))
				Renderer::GetAutomaton()->Init();
			if (ImGui::Button("1x Update Simulation"))
				Renderer::GetAutomaton()->Update();
			if (ImGui::Button("20x Update Simulation"))
				for (int i = 0; i < 20; i++)
					Renderer::GetAutomaton()->Update();
			if (ImGui::Button("200x Update Simulation"))
				for (int i = 0; i < 200; i++)
					Renderer::GetAutomaton()->Update();
			if (ImGui::Button("2000x Update Simulation"))
				for (int i = 0; i < 2000; i++)
					Renderer::GetAutomaton()->Update();
			ImGui::End();
		}
	}


	bool IsCursorActive()
	{
		return activeCursor;
	}
}