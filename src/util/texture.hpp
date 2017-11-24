#pragma once

#include <cstdio>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

class Texture {
	int w_, h_;
	unsigned char *data_;
public:
	Texture(const char *file_name) {
		data_ = stbi_load(file_name, &w_, &h_, NULL, 3);
		printf("Texture loaded. Texture Size: %d x %d\n\n", w_, h_);
	}
	~Texture() {
		free(data_);
	}
	int w() {
		return w_;
	}
	int h() {
		return h_;
	}
	unsigned char* data() {
		return data_;
	}
};
