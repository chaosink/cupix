#include "Video.hpp"

#include <cstdio>
#include <cstring>

Video::~Video() {
	for(auto f: frame_) delete[] f;
}

void Video::Add(unsigned char *frame) {
	frame_.push_back(new unsigned char[w_ * h_ * 3]);
	memcpy(frame_.back(), frame, w_ * h_ * 3);
}

void Video::Save(const char *file_name) {
	FILE *file = fopen(file_name, "wb");
	for(auto f: frame_)
		fwrite(f, 1, w_ * h_ * 3, file);
	fclose(file);
	printf("Video saved.\n");
}
