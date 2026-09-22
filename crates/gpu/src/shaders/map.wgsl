// LayerKind::Map. Take L from `value` at the current sample, then look up
// `palette` at sample (t, 0, 0). Clamp-to-edge in u, and linear between the
// two nearest palette texels, so a remap does not quantize to one level per
// texel: that would show as terraces once the result is a height.

struct MapParams {
    size: vec2<u32>,
    _pad: vec2<u32>,
    dom: vec4<f32>,          // own bake domain (min_u, min_v, ext_u, ext_v)
    dom_value: vec4<f32>,
    dom_palette: vec4<f32>,
}

@group(0) @binding(0) var<uniform> params: MapParams;
@group(0) @binding(1) var out_tex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var tex_value: texture_2d<f32>;
@group(0) @binding(3) var tex_palette: texture_2d<f32>;

fn dom_uv(dom: vec4<f32>, gid: vec2<u32>, size: vec2<u32>) -> vec2<f32> {
    return vec2<f32>(
        dom.x + (f32(gid.x) + 0.5) / f32(size.x) * dom.z,
        dom.y + (f32(gid.y) + 0.5) / f32(size.y) * dom.w,
    );
}

// Nearest texel of `uv` in an input baked over `dom`, clamped to its edge.
fn dom_texel(dom: vec4<f32>, uv: vec2<f32>, size: vec2<u32>) -> vec2<i32> {
    let tx = (uv.x - dom.x) / dom.z * f32(size.x);
    let ty = (uv.y - dom.y) / dom.w * f32(size.y);
    return vec2<i32>(
        i32(clamp(tx, 0.0, f32(size.x) - 1.0)),
        i32(clamp(ty, 0.0, f32(size.y) - 1.0)),
    );
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.size.x || gid.y >= params.size.y) { return; }
    let coord = vec2<i32>(i32(gid.x), i32(gid.y));
    let uv = dom_uv(params.dom, gid.xy, params.size);
    let val = textureLoad(tex_value, dom_texel(params.dom_value, uv, params.size), 0);
    let t = clamp(val.x, 0.0, 1.0);
    // The palette varies only in u. Measured from texel centers.
    let last = f32(params.size.x) - 1.0;
    let px = clamp(
        (t - params.dom_palette.x) / params.dom_palette.z * f32(params.size.x) - 0.5,
        0.0, last,
    );
    let i0 = floor(px);
    let f = px - i0;
    let a = textureLoad(tex_palette, vec2<i32>(i32(i0), 0), 0);
    let b = textureLoad(tex_palette, vec2<i32>(i32(min(i0 + 1.0, last)), 0), 0);
    // Oklch: hue takes the shorter way around.
    var dh = b.z - a.z;
    if (dh > 180.0) { dh = dh - 360.0; }
    if (dh < -180.0) { dh = dh + 360.0; }
    var h = a.z + dh * f;
    if (h < 0.0) { h = h + 360.0; }
    if (h >= 360.0) { h = h - 360.0; }
    textureStore(out_tex, coord, vec4<f32>(mix(a.x, b.x, f), mix(a.y, b.y, f), h, mix(a.w, b.w, f)));
}
