#!/bin/bash

if [ -z "$1" ]; then
	echo "Usage: raw2mp4.sh video.raw"
	exit
fi

ffmpeg \
	-f rawvideo \
	-pixel_format rgb24 \
	-video_size 1280x720 \
	-framerate 40 \
	-strict -2 \
	-i $1 \
	-vf "vflip" \
	${1%.*}.mp4
