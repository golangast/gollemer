#!/bin/bash
set -e
echo "Building whisper.cpp..."
mkdir -p build_whisper
cd build_whisper
if [ ! -d "whisper.cpp" ]; then
    git clone https://github.com/ggerganov/whisper.cpp.git
fi
cd whisper.cpp
cmake -B build
cmake --build build --config Release
cd ../..

export CGO_CFLAGS="-I$(pwd)/build_whisper/whisper.cpp/include"
export CGO_LDFLAGS="-L$(pwd)/build_whisper/whisper.cpp/build/src -lwhisper"

echo "Building voice_capture..."
go build -o ./bin/voice_capture ./cmd/tools/voice_capture
echo "Done!"
