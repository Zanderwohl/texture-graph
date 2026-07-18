// LayerKind::Mix. Add / Sub / Multiply / Blend of two color layers.
//
// Add and Subtract operate in Oklab (chroma is a Cartesian (a, b) vector,
// so a straight component-wise sum composes fractal noise correctly).
// Multiply operates in linear sRGB (optical darkening). Blend uses the
// caller-selected space (Oklch, LinearSrgb, or HSV of gamma-encoded sRGB).
//
// factor is either a constant or the L channel of a third input layer.

struct MixParams {
    size: vec2<u32>,
    mode: u32,             // 0=Add, 1=Subtract, 2=Multiply, 3=Blend
    space: u32,            // 0=Oklch, 1=LinearSrgb, 2=Hsv (Blend only)
    factor_const: f32,
    factor_is_layer: u32,  // 0=Const, 1=Layer
    _pad: vec2<u32>,
}

@group(0) @binding(0) var<uniform> params: MixParams;
@group(0) @binding(1) var out_tex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var tex_a: texture_2d<f32>;
@group(0) @binding(3) var tex_b: texture_2d<f32>;
@group(0) @binding(4) var tex_factor: texture_2d<f32>;

const PI: f32 = 3.14159265358979323846;
const DEG_TO_RAD: f32 = PI / 180.0;
const RAD_TO_DEG: f32 = 180.0 / PI;

fn oklch_to_oklab_vec(c: vec4<f32>) -> vec4<f32> {
    let h_rad = c.z * DEG_TO_RAD;
    return vec4<f32>(c.x, c.y * cos(h_rad), c.y * sin(h_rad), c.w);
}

fn oklab_to_oklch_vec(lab: vec4<f32>) -> vec4<f32> {
    let chroma = sqrt(lab.y * lab.y + lab.z * lab.z);
    var h = atan2(lab.z, lab.y) * RAD_TO_DEG;
    if (h < 0.0) { h = h + 360.0; }
    return vec4<f32>(lab.x, chroma, h, lab.w);
}

fn oklab_to_linear_srgb(l: f32, a: f32, b: f32) -> vec3<f32> {
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

fn oklch_to_linear_srgb(c: vec4<f32>) -> vec3<f32> {
    let lab = oklch_to_oklab_vec(c);
    return oklab_to_linear_srgb(lab.x, lab.y, lab.z);
}

fn linear_srgb_to_oklch(rgb: vec3<f32>) -> vec3<f32> {
    let lab = linear_srgb_to_oklab(rgb.x, rgb.y, rgb.z);
    let chroma = sqrt(lab.y * lab.y + lab.z * lab.z);
    var h = atan2(lab.z, lab.y) * RAD_TO_DEG;
    if (h < 0.0) { h = h + 360.0; }
    return vec3<f32>(lab.x, chroma, h);
}

// Shortest-arc hue lerp in degrees.
fn lerp_hue_deg(a: f32, b: f32, t: f32) -> f32 {
    var d = (b - a) - floor((b - a) / 360.0) * 360.0; // ((b-a) mod 360), but keep sign class
    // Match Rust's `%`: rem, not modulo. Emulate with fract-of-signed:
    d = (b - a) - trunc((b - a) / 360.0) * 360.0;
    if (d > 180.0)  { d = d - 360.0; }
    if (d < -180.0) { d = d + 360.0; }
    return a + d * t;
}

fn lerp(a: f32, b: f32, t: f32) -> f32 { return a + (b - a) * t; }

// --- HSV -------------------------------------------------------------
// Palette's Srgb -> Hsv uses the standard hexagonal formulation on
// gamma-encoded sRGB values. Match its behavior.

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

fn blend_colors(a: vec4<f32>, b: vec4<f32>, t: f32, space: u32) -> vec4<f32> {
    let alpha = lerp(a.w, b.w, t);
    switch space {
        case 1u: {
            // Linear sRGB
            let la = oklch_to_linear_srgb(a);
            let lb = oklch_to_linear_srgb(b);
            let mixed = vec3<f32>(lerp(la.x, lb.x, t), lerp(la.y, lb.y, t), lerp(la.z, lb.z, t));
            let back = linear_srgb_to_oklch(mixed);
            return vec4<f32>(back, alpha);
        }
        case 2u: {
            // HSV of gamma-encoded sRGB
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
            // Oklch
            return vec4<f32>(
                lerp(a.x, b.x, t),
                lerp(a.y, b.y, t),
                lerp_hue_deg(a.z, b.z, t),
                alpha,
            );
        }
    }
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.size.x || gid.y >= params.size.y) { return; }
    let coord = vec2<i32>(i32(gid.x), i32(gid.y));
    let a = textureLoad(tex_a, coord, 0);
    let b = textureLoad(tex_b, coord, 0);
    var out: vec4<f32>;

    switch params.mode {
        case 0u: {
            // Add — in Oklab.
            let al = oklch_to_oklab_vec(a);
            let bl = oklch_to_oklab_vec(b);
            let sum = vec4<f32>(al.x + bl.x, al.y + bl.y, al.z + bl.z, a.w);
            out = oklab_to_oklch_vec(sum);
        }
        case 1u: {
            // Subtract — in Oklab.
            let al = oklch_to_oklab_vec(a);
            let bl = oklch_to_oklab_vec(b);
            let diff = vec4<f32>(al.x - bl.x, al.y - bl.y, al.z - bl.z, a.w);
            out = oklab_to_oklch_vec(diff);
        }
        case 2u: {
            // Multiply — componentwise in linear sRGB.
            let la = oklch_to_linear_srgb(a);
            let lb = oklch_to_linear_srgb(b);
            let prod = vec3<f32>(la.x * lb.x, la.y * lb.y, la.z * lb.z);
            let back = linear_srgb_to_oklch(prod);
            out = vec4<f32>(back, a.w * b.w);
        }
        default: {
            var t = params.factor_const;
            if (params.factor_is_layer == 1u) {
                let f = textureLoad(tex_factor, coord, 0);
                t = f.x;  // scalar_of takes L
            }
            out = blend_colors(a, b, t, params.space);
        }
    }
    textureStore(out_tex, coord, out);
}
