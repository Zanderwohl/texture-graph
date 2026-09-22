// LayerKind::Color: one Oklcha value in every texel.

struct Params {
    color: vec4<f32>,   // (L, C, hue-deg, alpha), unclamped
    size: vec2<u32>,
    _pad: vec2<u32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var out_tex: texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.size.x || gid.y >= params.size.y) { return; }
    textureStore(out_tex, vec2<i32>(i32(gid.x), i32(gid.y)), params.color);
}
