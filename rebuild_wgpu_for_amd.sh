#!/bin/bash
# Rebuild wgpu-native with AMD GPU compute support
# This creates a GPU-accelerated libwgpu_native.so for Vulkan/AMD

set -e

WGPU_TAG=$(cat wgpu-native-meta/wgpu-native-git-tag)
BUILD_DIR="/tmp/wgpu-native-rebuild"
GOLLEMER_DIR="$(pwd)"

echo "🔨 Rebuilding wgpu-native $WGPU_TAG with full compute support for AMD Radeon..."
echo "📋 Requirements: cargo, rustc"
echo ""

# Check for cargo
if ! command -v cargo &> /dev/null; then
    echo "❌ Cargo not found! Install Rust:"
    echo "   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    exit 1
fi

# Clone wgpu-native with specific tag
echo "📥 Cloning wgpu-native $WGPU_TAG..."
rm -rf "$BUILD_DIR"
git clone --depth 1 --branch "$WGPU_TAG" \
    https://github.com/gfx-rs/wgpu-native.git "$BUILD_DIR"
cd "$BUILD_DIR"

# Build with all backends enabled, including Vulkan compute
echo "⚙️  Building with Vulkan + Metal + DX12 + Compute support..."
RUSTFLAGS="-C opt-level=3" cargo build --release \
    --features=vulkan,metal,dx12,gles,naga/validate,naga/wgsl-in,naga/wgsl-out

if [ ! -f target/release/libwgpu_native.so ]; then
    echo "❌ Build failed - libwgpu_native.so not found"
    exit 1
fi

# Backup old library
if [ -f "$GOLLEMER_DIR/lib/libwgpu_native.so" ]; then
    echo "💾 Backing up old library..."
    cp "$GOLLEMER_DIR/lib/libwgpu_native.so" \
       "$GOLLEMER_DIR/lib/libwgpu_native.so.software-only"
fi

# Install new library
echo "📦 Installing GPU-accelerated library..."
cp target/release/libwgpu_native.so "$GOLLEMER_DIR/lib/"
cp target/release/libwgpu_native.a "$GOLLEMER_DIR/lib/" 2>/dev/null || true

echo ""
echo "✅ wgpu-native rebuilt successfully!"
echo ""
echo "📝 To use GPU with born/gogpu:"
echo ""
echo "  cd $GOLLEMER_DIR"
echo "  CGO_ENABLED=0 go build -o train_moe ./cmd/tools/train_moe"
echo "  LD_LIBRARY_PATH=$GOLLEMER_DIR/lib WGPU_BACKEND=vulkan ./train_moe --gpu --epochs 50"
echo ""
echo "🎯 Expected speedup: 4-10x faster than CPU"
