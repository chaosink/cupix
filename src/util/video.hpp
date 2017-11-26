#pragma once

#include <vector>
#include <fstream>

class Video {
	int w_, h_;
	std::vector<unsigned char*> frame_;
public:
	Video(int w, int h) : w_(w), h_(h) {}
	~Video() {
		for(auto f: frame_) delete[] f;
	}
	void Add(unsigned char *frame) {
		frame_.push_back(new unsigned char[w_ * h_ * 3]);
		memcpy(frame_.back(), frame, w_ * h_ * 3);
	}
	void Save(const char *file_name) {
		std::ofstream ofs(file_name, ofs.binary);
		for(auto f: frame_)
			ofs.write(reinterpret_cast<char*>(f), w_ * h_ * 3);
		ofs.close();
		printf("Video saved.\n");
	}
};
