// LayerKind::ColorRamp: multi-stop gradient along U.
//
// A layer stop's `input_index` selects one of MAX_RAMP_INPUTS = 8 texture
// bindings. A ramp referencing more than 8 layers is rejected at bake time.

struct RampParams {
    size: vec2<u32>,
    stop_count: u32,
    space: u32,      // 0=Oklch, 1=LinearSrgb, 2=Hsv
    dom: vec4<f32>,  // own bake domain (min_u, min_v, ext_u, ext_v)
    input_doms: array<vec4<f32>, 8>,
}

struct Stop {
    color: vec4<f32>,   // Oklcha; used when kind == 0
    t: f32,
    kind: u32,          // 0 = const, 1 = sample tex[input_index]
    input_index: u32,   // 0..7
    _p0: f32,
}

@group(0) @binding(0) var<uniform> params: RampParams;
@group(0) @binding(1) var out_tex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<storage, read> stops: array<Stop>;
@group(0) @binding(3) var in0: texture_2d<f32>;
@group(0) @binding(4) var in1: texture_2d<f32>;
@group(0) @binding(5) var in2: texture_2d<f32>;
@group(0) @binding(6) var in3: texture_2d<f32>;
@group(0) @binding(7) var in4: texture_2d<f32>;
@group(0) @binding(8) var in5: texture_2d<f32>;
@group(0) @binding(9) var in6: texture_2d<f32>;
@group(0) @binding(10) var in7: texture_2d<f32>;

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
const DEG_TO_RAD: f32 = PI / 180.0;
const RAD_TO_DEG: f32 = 180.0 / PI;

fn oklch_to_linear_srgb(c: vec4<f32>) -> vec3<f32> {
    let h_rad = c.z * DEG_TO_RAD;
    let a = c.y * cos(h_rad);
    let b = c.y * sin(h_rad);
    let l = c.x;
    let l_ = l + 0.3963377774 * a + 0.2158037573 * b;
    let m_ = l - 0.1055613458 * a - 0.0638541728 * b;
    let s_ = l - 0.0894841775 * a - 1.2914855480 * b;
    let lc = l_ * l_ * l_;
    let mc = m_ * m_ * m_;
    let sc = s_ * s_ * s_;
    return vec3<f32>(
         4.0767416621 * lc - 3.3077115913 * mc + 0.2309699292 * sc,
        -1.2684380046 * lc + 2.6097574011 * mc - 0.3413193965 * sc,
        -0.0041960863 * lc - 0.7034186147 * mc + 1.7076147010 * sc,
    );
}

fn linear_srgb_to_oklch(rgb: vec3<f32>) -> vec3<f32> {
    let ll = 0.4122214708 * rgb.x + 0.5363325363 * rgb.y + 0.0514459929 * rgb.z;
    let mm = 0.2119034982 * rgb.x + 0.6806995451 * rgb.y + 0.1073969566 * rgb.z;
    let ss = 0.0883024619 * rgb.x + 0.2817188376 * rgb.y + 0.6299787005 * rgb.z;
    let l_ = sign(ll) * pow(abs(ll), 1.0 / 3.0);
    let m_ = sign(mm) * pow(abs(mm), 1.0 / 3.0);
    let s_ = sign(ss) * pow(abs(ss), 1.0 / 3.0);
    let l = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_;
    let a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_;
    let b = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_;
    let chroma = sqrt(a * a + b * b);
    var h = atan2(b, a) * RAD_TO_DEG;
    if (h < 0.0) { h = h + 360.0; }
    return vec3<f32>(l, chroma, h);
}

fn lerp(a: f32, b: f32, t: f32) -> f32 { return a + (b - a) * t; }

fn lerp_hue_deg(a: f32, b: f32, t: f32) -> f32 {
    var d = (b - a) - trunc((b - a) / 360.0) * 360.0;
    if (d > 180.0)  { d = d - 360.0; }
    if (d < -180.0) { d = d + 360.0; }
    return a + d * t;
}

fn linear_to_srgb_component(x: f32) -> f32 {
    let cx = clamp(x, 0.0, 1.0);
    if (cx <= 0.0031308) { return 12.92 * cx; }
    return 1.055 * pow(cx, 1.0 / 2.4) - 0.055;
}

fn srgb_to_linear_component(x: f32) -> f32 {
    let cx = clamp(x, 0.0, 1.0);
    if (cx <= 0.04045) { return cx / 12.92; }
    return pow((cx + 0.055) / 1.055, 2.4);
}

struct Hsv { h: f32, s: f32, v: f32, }

fn srgb_to_hsv(r: f32, g: f32, b: f32) -> Hsv {
    let max_c = max(max(r, g), b);
    let min_c = min(min(r, g), b);
    let d = max_c - min_c;
    let v = max_c;
    let s = select(0.0, d / max_c, max_c > 0.0);
    var h: f32 = 0.0;
    if (d > 0.0) {
        if (max_c == r) { h = (g - b) / d + select(0.0, 6.0, g < b); }
        else if (max_c == g) { h = (b - r) / d + 2.0; }
        else { h = (r - g) / d + 4.0; }
        h = h * 60.0;
    }
    return Hsv(h, s, v);
}

fn hsv_to_srgb(hsv: Hsv) -> vec3<f32> {
    let h = hsv.h / 60.0;
    let c = hsv.v * hsv.s;
    let x = c * (1.0 - abs((h - 2.0 * floor(h / 2.0)) - 1.0));
    let m = hsv.v - c;
    var rgb: vec3<f32>;
    if      (h < 1.0) { rgb = vec3<f32>(c, x, 0.0); }
    else if (h < 2.0) { rgb = vec3<f32>(x, c, 0.0); }
    else if (h < 3.0) { rgb = vec3<f32>(0.0, c, x); }
    else if (h < 4.0) { rgb = vec3<f32>(0.0, x, c); }
    else if (h < 5.0) { rgb = vec3<f32>(x, 0.0, c); }
    else              { rgb = vec3<f32>(c, 0.0, x); }
    return rgb + vec3<f32>(m);
}

fn blend_stops(a: vec4<f32>, b: vec4<f32>, t: f32) -> vec4<f32> {
    let alpha = lerp(a.w, b.w, t);
    switch params.space {
        case 1u: {
            let la = oklch_to_linear_srgb(a);
            let lb = oklch_to_linear_srgb(b);
            let mixed = vec3<f32>(lerp(la.x, lb.x, t), lerp(la.y, lb.y, t), lerp(la.z, lb.z, t));
            let back = linear_srgb_to_oklch(mixed);
            return vec4<f32>(back, alpha);
        }
        case 2u: {
            let la = oklch_to_linear_srgb(a);
            let lb = oklch_to_linear_srgb(b);
            let sa = vec3<f32>(linear_to_srgb_component(la.x), linear_to_srgb_component(la.y), linear_to_srgb_component(la.z));
            let sb = vec3<f32>(linear_to_srgb_component(lb.x), linear_to_srgb_component(lb.y), linear_to_srgb_component(lb.z));
            let ha = srgb_to_hsv(sa.x, sa.y, sa.z);
            let hb = srgb_to_hsv(sb.x, sb.y, sb.z);
            let mixed = Hsv(lerp_hue_deg(ha.h, hb.h, t), lerp(ha.s, hb.s, t), lerp(ha.v, hb.v, t));
            let out_srgb = hsv_to_srgb(mixed);
            let out_lin = vec3<f32>(srgb_to_linear_component(out_srgb.x), srgb_to_linear_component(out_srgb.y), srgb_to_linear_component(out_srgb.z));
            let back = linear_srgb_to_oklch(out_lin);
            return vec4<f32>(back, alpha);
        }
        default: {
            return vec4<f32>(
                lerp(a.x, b.x, t),
                lerp(a.y, b.y, t),
                lerp_hue_deg(a.z, b.z, t),
                alpha,
            );
        }
    }
}

// A switch, because indexing a texture array needs
// SAMPLED_TEXTURE_ARRAY_NON_UNIFORM_INDEXING, which is native-only.
fn sample_input(idx: u32, coord: vec2<i32>) -> vec4<f32> {
    switch idx {
        case 0u: { return textureLoad(in0, coord, 0); }
        case 1u: { return textureLoad(in1, coord, 0); }
        case 2u: { return textureLoad(in2, coord, 0); }
        case 3u: { return textureLoad(in3, coord, 0); }
        case 4u: { return textureLoad(in4, coord, 0); }
        case 5u: { return textureLoad(in5, coord, 0); }
        case 6u: { return textureLoad(in6, coord, 0); }
        default: { return textureLoad(in7, coord, 0); }
    }
}

fn stop_color(idx: u32, uv: vec2<f32>) -> vec4<f32> {
    let s = stops[idx];
    if (s.kind == 0u) {
        return s.color;
    }
    let coord = dom_texel(params.input_doms[s.input_index], uv, params.size);
    return sample_input(s.input_index, coord);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.size.x || gid.y >= params.size.y) { return; }
    let coord = vec2<i32>(i32(gid.x), i32(gid.y));
    let uv = dom_uv(params.dom, gid.xy, params.size);
    let u = uv.x;
    // Must match the CPU loop in `eval_ramp`.
    var lo: u32 = 0u;
    var hi: u32 = params.stop_count - 1u;
    for (var i: u32 = 0u; i + 1u < params.stop_count; i = i + 1u) {
        let a = stops[i].t;
        let b = stops[i + 1u].t;
        if (u >= a && u <= b) {
            lo = i; hi = i + 1u; break;
        }
        if (u < a && i == 0u) {
            lo = 0u; hi = 1u; break;
        }
        if (u > b && i + 2u == params.stop_count) {
            lo = params.stop_count - 2u;
            hi = params.stop_count - 1u;
            break;
        }
    }
    let ta = stops[lo].t;
    let tb = stops[hi].t;
    let span = tb - ta;
    var t: f32 = 0.0;
    if (abs(span) >= 1.1754944e-38) {
        // Hold the end colors instead of extrapolating, as `eval_ramp` does.
        t = clamp((u - ta) / span, 0.0, 1.0);
    }
    let a_color = stop_color(lo, uv);
    let b_color = stop_color(hi, uv);
    let out_px = blend_stops(a_color, b_color, t);
    textureStore(out_tex, coord, out_px);
}
