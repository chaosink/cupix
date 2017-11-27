#pragma once

#include <vector>

class Video {
	int w_, h_;
	std::vector<unsigned char*> frame_;
public:
	Video(int w, int h) : w_(w), h_(h) {}
	~Video();
	void Add(unsigned char *frame);
	void Save(const char *file_name);
};
