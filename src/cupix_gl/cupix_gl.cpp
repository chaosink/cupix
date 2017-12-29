#include "OGL.hpp"
#include "Model.hpp"
#include "Camera.hpp"
#include "FPS.hpp"

int main(int argc, char *argv[]) {
	if(argc < 2) {
		printf("Usage: cupix_gl input_obj_file [output_video_file]\n");
		return 0;
	}

	int window_w = 1280;
	int window_h = 720;

	OGL ogl;
	ogl.InitGLFW("CUPix OpenGL", window_w, window_h);
	ogl.InitGL("shader/vertex.glsl", "shader/fragment.glsl");

	Model model(ogl.window(), argv[1]);
	ogl.Vertex(model.vertex(), model.n_vertex());
	ogl.Normal(model.normal(), model.n_vertex());

	Camera camera(ogl.window(), window_w, window_h);
	FPS fps;
	while(ogl.Alive()) {
		double time = ogl.time();
		ogl.Clear();

		glm::mat4 m = model.Update(time);
		glm::mat4 vp = camera.Update(time);
		glm::mat4 mvp = vp * m;
		ogl.MVP(mvp);
		glm::mat4 v = camera.v();
		glm::mat4 mv = v * m;
		ogl.MV(mv);

		ogl.Update();
		fps.Update(time);
	}
	fps.Term();

	ogl.Terminate();

	return 0;
}
