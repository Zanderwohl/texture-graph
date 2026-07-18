// LayerKind::Map. Take L from `value` at the current sample, then look up
// `palette` at sample (t, 0, 0). Clamp-to-edge in u.

struct MapParams {
    size: vec2<u32>,
    _pad: vec2<u32>,
}

@group(0) @binding(0) var<uniform> params: MapParams;
@group(0) @binding(1) var out_tex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var tex_value: texture_2d<f32>;
@group(0) @binding(3) var tex_palette: texture_2d<f32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.size.x || gid.y >= params.size.y) { return; }
    let coord = vec2<i32>(i32(gid.x), i32(gid.y));
    let val = textureLoad(tex_value, coord, 0);
    let t = clamp(val.x, 0.0, 1.0);
    // Nearest-neighbor sample into palette at (t, 0). Palette is 1D-in-U.
    let px = clamp(t * f32(params.size.x), 0.0, f32(params.size.x) - 1.0);
    let pal = textureLoad(tex_palette, vec2<i32>(i32(px), 0), 0);
    textureStore(out_tex, coord, pal);
}
