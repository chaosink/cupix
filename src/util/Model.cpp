#include "Model.hpp"

#include <cstdio>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

Model::Model(const char *file_name) {
	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;
	std::string err;
	tinyobj::LoadObj(&attrib, &shapes, &materials, &err, file_name);

	for (size_t s = 0; s < shapes.size(); s++)
		for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++)
			n_vertex_ += shapes[s].mesh.num_face_vertices[f];
	vertex_ = new float[n_vertex_ * 3];
	normal_ = new float[n_vertex_ * 3];
	uv_ = new float[n_vertex_ * 2];

	int i = 0;
	for (size_t s = 0; s < shapes.size(); s++) {
		size_t index_offset = 0;
		for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
			size_t fv = shapes[s].mesh.num_face_vertices[f];
			for (size_t v = 0; v < fv; v++) {
				tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
				vertex_[i * 3 + 0] = attrib.vertices[3 * idx.vertex_index + 0];
				vertex_[i * 3 + 1] = attrib.vertices[3 * idx.vertex_index + 1];
				vertex_[i * 3 + 2] = attrib.vertices[3 * idx.vertex_index + 2];
				normal_[i * 3 + 0] = attrib.normals[3 * idx.normal_index + 0];
				normal_[i * 3 + 1] = attrib.normals[3 * idx.normal_index + 1];
				normal_[i * 3 + 2] = attrib.normals[3 * idx.normal_index + 2];
				if(attrib.texcoords.size()) {
					uv_[i * 2 + 0] = attrib.texcoords[2 * idx.texcoord_index + 0];
					uv_[i * 2 + 1] = attrib.texcoords[2 * idx.texcoord_index + 1];
				}
				// Optional: vertex colors
				// tinyobj::real_t red = attrib.colors[3*idx.vertex_index+0];
				// tinyobj::real_t green = attrib.colors[3*idx.vertex_index+1];
				// tinyobj::real_t blue = attrib.colors[3*idx.vertex_index+2];
				i++;
			}
			index_offset += fv;
		}
	}
	printf("Model loaded. Number of faces: %d. Number of vertices: %d\n", n_vertex_ / 3, n_vertex_);
}
