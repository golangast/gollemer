#!/bin/bash
set -e
echo "Building whisper.cpp..."
mkdir -p build_whisper
cd build_whisper
if [ -d "whisper.cpp" ]; then
    if [ ! -f "whisper.cpp/CMakeLists.txt" ]; then
        echo "⚠️  Existing whisper.cpp directory is invalid or incomplete, removing it."
        rm -rf whisper.cpp
    fi
fi
if [ ! -d "whisper.cpp" ]; then
    git clone --depth 1 https://github.com/ggerganov/whisper.cpp.git
fi
cd whisper.cpp
cmake -B build
cmake --build build --config Release
cd ../..

if [ ! -f "build_whisper/whisper.cpp/models/ggml-tiny.en.bin" ]; then
    echo "Downloading Whisper model tiny.en into build_whisper/whisper.cpp/models..."
    pushd build_whisper/whisper.cpp/models >/dev/null
    ./download-ggml-model.sh tiny.en
    popd >/dev/null
fi

export CGO_CFLAGS="-I$(pwd)/build_whisper/whisper.cpp/include"
export CGO_LDFLAGS="-L$(pwd)/build_whisper/whisper.cpp/build/bin -lwhisper"

# Ensure the linker can find libwhisper.so by creating a stable symlink if needed.
if [ -d "build_whisper/whisper.cpp/build/bin" ]; then
    cd build_whisper/whisper.cpp/build/bin
    if [ -f "libwhisper.so.1.9.1" ] && [ ! -f "libwhisper.so" ]; then
        ln -sf libwhisper.so.1.9.1 libwhisper.so
    fi
    cd - >/dev/null
fi

echo "Building voice_capture..."
go build -o ./bin/voice_capture ./cmd/tools/voice_capture
echo "Done!"
