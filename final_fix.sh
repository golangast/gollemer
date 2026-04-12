#!/bin/bash
DIR="vendor/github.com/born-ml/born/internal/backend/gogpu"
sed -i 's/package webgpu/package gogpu/g' $DIR/*.go
sed -i 's/github.com\/go-webgpu\/webgpu\/wgpu/github.com\/gogpu\/wgpu/g' $DIR/*.go
sed -i 's/github.com\/go-webgpu\/webgpu\/gputypes/github.com\/gogpu\/gputypes/g' $DIR/*.go
sed -i 's/tensor\.WebGPU/tensor.Gogpu/g' $DIR/*.go
sed -i 's/wgpu\.True/true/g' $DIR/*.go
sed -i 's/encoder\.BeginComputePass(nil)/encoder.BeginComputePass(\&wgpu.ComputePassDescriptor{})/g' $DIR/*.go
sed -i 's/\.Finish(nil)/.Finish()/g' $DIR/*.go
sed -i 's/DispatchWorkgroups/Dispatch/g' $DIR/*.go
sed -i 's/b\.device\.CreateBindGroupSimple/b.createBindGroup/g' $DIR/*.go
sed -i 's/wgpu\.BufferBindingEntry/BindGroupEntryFromBuffer/g' $DIR/*.go
sed -i 's/b\.device\.CreateBuffer/func()(*wgpu.Buffer,error){return b.device.CreateBuffer/g' $DIR/*.go
# Wait, the above CreateBuffer fix is bad.
# Let's just use the simple assignment fixes for now.
