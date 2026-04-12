#!/bin/bash
# Rebuild wgpu-native with GPU compute support for AMD Radeon

set -e

WG_TAG=$(cat wgpu-native-meta/wgpu-native-git-tag)
BUILD_DIR="/tmp/wgpu-native-build"
GOLLEMER_DIR="/home/a/go/gollemer"

echo "🔨 Building wgpu-native $WG_TAG with GPU support..."

# Clone wgpu-native  
rm -rf "$BUILD_DIR"
git clone --depth 1 --branch "$WG_TAG" https://github.com/gfx-rs/wgpu-native.git "$BUILD_DIR"
cd "$BUILD_DIR"

# Build with Vulkan backend enabled (default) and metal/dx12
# For AMD Radeon: uses Vulkan via RADV driver
echo "📦 Building Vulkan backend (AMD Radeon via RADV)..."
cargo build --release --features=vulkan,naga/validate,naga/wgsl-in,naga/wgsl-out

# Copy to lib/
cp target/release/libwgpu_native.so "$GOLLEMER_DIR/lib/"
cp target/release/libwgpu_native.a "$GOLLEMER_DIR/lib/"

echo "✅ wgpu-native rebuilt with Vulkan support at $GOLLEMER_DIR/lib/"
echo "📝 Next: rebuild train_moe with:"
echo "   cd $GOLLEMER_DIR"
echo "   LD_LIBRARY_PATH=$GOLLEMER_DIR/lib:$LD_LIBRARY_PATH WGPU_BACKEND=vulkan ./train_moe --gpu --epochs 50"
