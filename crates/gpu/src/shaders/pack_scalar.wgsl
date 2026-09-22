// Writes L of an Oklcha slot to an R-only texture.
//
// A render pass, not compute: `r8unorm` is not a core WebGPU storage format,
// but every scalar format offered here is renderable in core WebGPU.
//
// Writes the raw scalar, not a display-encoded gray, for sampling by other
// shaders. `r8unorm` clamps to [0, 1]; float formats keep signed values.

struct ScalarPackParams {
    size: vec2<u32>,
    _pad0: u32,
    _pad1: u32,
    // (min_u, min_v, ext_u, ext_v)
    src_dom: vec4<f32>,
}

@group(0) @binding(0) var<uniform> params: ScalarPackParams;
@group(0) @binding(1) var src: texture_2d<f32>;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let x = f32((i << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(i & 2u) * 2.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    // `pos.xy` is the pixel center, matching compute's `(gid + 0.5) / size`.
    let uv = vec2<f32>(pos.x / f32(params.size.x), pos.y / f32(params.size.y));
    let tx = clamp(
        (uv.x - params.src_dom.x) / params.src_dom.z * f32(params.size.x),
        0.0, f32(params.size.x) - 1.0);
    let ty = clamp(
        (uv.y - params.src_dom.y) / params.src_dom.w * f32(params.size.y),
        0.0, f32(params.size.y) - 1.0);
    let l = textureLoad(src, vec2<i32>(i32(tx), i32(ty)), 0).x;
    return vec4<f32>(l, 0.0, 0.0, 1.0);
}
