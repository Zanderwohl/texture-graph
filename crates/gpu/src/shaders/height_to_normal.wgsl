// LayerKind::HeightToNormal. Central-difference normal from the source's L,
// encoded as a color through the `normal_to_color` path.
//
// Uses ±1 pixel neighbors, not `EvalCtx::normal_epsilon` as the CPU eval
// does, so the numbers differ from the CPU but look the same.

struct H2NParams {
    size: vec2<u32>,
    strength: f32,
    _pad: u32,
    dom: vec4<f32>,      // own bake domain (min_u, min_v, ext_u, ext_v)
    dom_src: vec4<f32>,
}

@group(0) @binding(0) var<uniform> params: H2NParams;
@group(0) @binding(1) var out_tex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var src: texture_2d<f32>;

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

const PI: f32 = 3.14159265358979323846;
const RAD_TO_DEG: f32 = 180.0 / PI;

fn srgb_to_linear_component(x: f32) -> f32 {
    let cx = clamp(x, 0.0, 1.0);
    if (cx <= 0.04045) { return cx / 12.92; }
    return pow((cx + 0.055) / 1.055, 2.4);
}

fn linear_srgb_to_oklab(r: f32, g: f32, b: f32) -> vec3<f32> {
    let ll = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b;
    let mm = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b;
    let ss = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b;
    let l_ = sign(ll) * pow(abs(ll), 1.0 / 3.0);
    let m_ = sign(mm) * pow(abs(mm), 1.0 / 3.0);
    let s_ = sign(ss) * pow(abs(ss), 1.0 / 3.0);
    return vec3<f32>(
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
    );
}

fn normal_to_oklch(n: vec3<f32>) -> vec3<f32> {
    let s = n * 0.5 + vec3<f32>(0.5);
    let lin = vec3<f32>(
        srgb_to_linear_component(s.x),
        srgb_to_linear_component(s.y),
        srgb_to_linear_component(s.z),
    );
    let lab = linear_srgb_to_oklab(lin.x, lin.y, lin.z);
    let chroma = sqrt(lab.y * lab.y + lab.z * lab.z);
    var h = atan2(lab.z, lab.y) * RAD_TO_DEG;
    if (h < 0.0) { h = h + 360.0; }
    return vec3<f32>(lab.x, chroma, h);
}

fn clamp_i(v: i32, lo: i32, hi: i32) -> i32 {
    return max(lo, min(hi, v));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.size.x || gid.y >= params.size.y) { return; }
    let sx = i32(params.size.x);
    let sy = i32(params.size.y);
    let uv = dom_uv(params.dom, gid.xy, params.size);
    let c = dom_texel(params.dom_src, uv, params.size);
    let x = c.x;
    let y = c.y;
    let l_px = textureLoad(src, vec2<i32>(clamp_i(x + 1, 0, sx - 1), y), 0).x;
    let l_nx = textureLoad(src, vec2<i32>(clamp_i(x - 1, 0, sx - 1), y), 0).x;
    let l_py = textureLoad(src, vec2<i32>(x, clamp_i(y + 1, 0, sy - 1)), 0).x;
    let l_ny = textureLoad(src, vec2<i32>(x, clamp_i(y - 1, 0, sy - 1)), 0).x;
    // One source texel spans ext/size in UV.
    let dhdx = (l_px - l_nx) * f32(sx) / params.dom_src.z * 0.5;
    let dhdy = (l_py - l_ny) * f32(sy) / params.dom_src.w * 0.5;
    var n = vec3<f32>(-dhdx * params.strength, -dhdy * params.strength, 1.0);
    let mag = max(length(n), 1e-30);
    n = n / mag;
    let oklch = normal_to_oklch(n);
    textureStore(out_tex, vec2<i32>(x, y), vec4<f32>(oklch, 1.0));
}
