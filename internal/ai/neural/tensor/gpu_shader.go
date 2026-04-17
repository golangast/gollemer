package tensor

const matmulWGSL = `
struct Matrix {
    size: vec2<u32>,
    data: array<f32>,
};

@group(0) @binding(0) var<storage, read> a: Matrix;
@group(0) @binding(1) var<storage, read> b: Matrix;
@group(0) @binding(2) var<storage, read_write> c: Matrix;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= c.size.x || global_id.y >= c.size.y) {
        return;
    }

    var sum: f32 = 0.0;
    for (var k: u32 = 0u; k < a.size.y; k = k + 1u) {
        sum = sum + a.data[global_id.x * a.size.y + k] * b.data[k * b.size.y + global_id.y];
    }
    c.data[global_id.x * c.size.y + global_id.y] = sum;
}
`
