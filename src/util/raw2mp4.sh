#!/bin/bash

ffmpeg \
	-f rawvideo \
	-pixel_format rgb24 \
	-video_size 1280x720 \
	-framerate 40 \
	-strict -2 \
	-i $1 \
	-vf "vflip" \
	$2
