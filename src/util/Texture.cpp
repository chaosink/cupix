#include "Texture.hpp"

#include <cstdio>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

Texture::Texture(const char *file_name) {
	data_ = stbi_load(file_name, &w_, &h_, NULL, 3);
	printf("Texture loaded. Texture Size: %d x %d\n", w_, h_);
}

Texture::~Texture() {
	free(data_);
}
