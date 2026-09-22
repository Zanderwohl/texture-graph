// Fills an Rgba8Unorm dst with one value. The pack stage uses it for
// constant scalar channels and an absent normal.

struct SolidParams {
    color: vec4<f32>,
    size: vec2<u32>,
    _pad: vec2<u32>,
}

@group(0) @binding(0) var<uniform> params: SolidParams;
@group(0) @binding(1) var dst: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.size.x || gid.y >= params.size.y) { return; }
    textureStore(dst, vec2<i32>(i32(gid.x), i32(gid.y)), params.color);
}
