#pragma once

class Texture {
	int w_, h_;
	unsigned char *data_;
public:
	Texture(const char *file_name);
	~Texture();
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
