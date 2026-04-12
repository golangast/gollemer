#!/bin/bash
set -e

echo "🔨 Rebuilding wgpu-native with compute shaders enabled..."

docker run --rm -v $(pwd):/workspace rust:latest bash -c '
  cd /workspace
  
  # Install build dependencies
  apt-get update && apt-get install -y pkg-config libvulkan-dev || true
  
  # Clone wgpu-native
  if [ ! -d wgpu-native-build ]; then
    git clone https://github.com/gfx-rs/wgpu-native.git wgpu-native-build
  fi
  
  cd wgpu-native-build
  git fetch origin
  git checkout v27.0.4.0
  
  # Build with vulkan backend and compute support
  RUSTFLAGS="-C target-feature=+crt-static" cargo build --release \
    --features vulkan,compute \
    --target x86_64-unknown-linux-gnu
  
  # Copy built library back
  cp target/x86_64-unknown-linux-gnu/release/libwgpu_native.so /workspace/lib/libwgpu_native.so
  echo "✅ Built libwgpu_native.so with compute support"
'

if [ -f lib/libwgpu_native.so ]; then
  echo "✅ libwgpu_native.so successfully built and copied"
  ls -lh lib/libwgpu_native.so
fi
