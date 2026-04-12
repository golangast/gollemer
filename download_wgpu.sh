#!/bin/bash
# Download prebuilt wgpu-native with GPU support
# This gets the official build from wgpu-native releases

WG_TAG=${1:-"v27.0.4.0"}
GOLLEMER_DIR="/home/a/go/gollemer"

echo "📥 Downloading wgpu-native $WG_TAG prebuilt binary..."

# Download the prebuilt binary with GPU backends from official release
cd /tmp
wget -q "https://github.com/gfx-rs/wgpu-native/releases/download/$WG_TAG/wgpu-linux-x86_64-release.zip" -O wgpu-prebuilt.zip

if [ ! -f wgpu-prebuilt.zip ]; then
    echo "❌ Failed to download. Building from source instead..."
    chmod +x rebuild_wgpu.sh
    exec $GOLLEMER_DIR/rebuild_wgpu.sh
fi

#Extract and install
unzip -o wgpu-prebuilt.zip
cp wgpu-native/target/release/libwgpu_native.so "$GOLLEMER_DIR/lib/"
cp wgpu-native/target/release/libwgpu_native.a "$GOLLEMER_DIR/lib/" 2>/dev/null || true

echo "✅ Prebuilt wgpu-native extracted to $GOLLEMER_DIR/lib/"  
echo ""
echo "🚀 Now run:"
echo "   cd $GOLLEMER_DIR"
echo "   LD_LIBRARY_PATH=$GOLLEMER_DIR/lib WGPU_BACKEND=vulkan ./train_moe --gpu --epochs 50 --batch-size 8"
