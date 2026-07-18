// LayerKind::MinMax. Per-pixel winner-take-all between two color layers.
// Whichever pixel's `criterion` is smaller (Min) or larger (Max) is emitted
// whole — all four channels flow through the winner, not just the compared
// attribute. Ties go to `a`.

struct MinMaxParams {
    size: vec2<u32>,
    mode: u32,       // 0 = Min, 1 = Max
    criterion: u32,  // 0=R 1=G 2=B 3=Saturation 4=Value 5=Luma 6=Alpha 7=Chroma
}

@group(0) @binding(0) var<uniform> params: MinMaxParams;
@group(0) @binding(1) var out_tex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var tex_a: texture_2d<f32>;
@group(0) @binding(3) var tex_b: texture_2d<f32>;

const PI: f32 = 3.14159265358979323846;
const DEG_TO_RAD: f32 = PI / 180.0;

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

fn oklch_to_linear_srgb(v: vec4<f32>) -> vec3<f32> {
    let h_rad = v.z * DEG_TO_RAD;
    let a = v.y * cos(h_rad);
    let b = v.y * sin(h_rad);
    return oklab_to_linear_srgb(v.x, a, b);
}

fn linear_to_srgb_component(x: f32) -> f32 {
    let cx = clamp(x, 0.0, 1.0);
    if (cx <= 0.0031308) { return 12.92 * cx; }
    return 1.055 * pow(cx, 1.0 / 2.4) - 0.055;
}

fn linear_to_srgb(lin: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        linear_to_srgb_component(lin.x),
        linear_to_srgb_component(lin.y),
        linear_to_srgb_component(lin.z),
    );
}

fn criterion_of(px: vec4<f32>) -> f32 {
    // Alpha and Chroma read straight off Oklcha — no conversion needed.
    switch params.criterion {
        case 6u: { return px.w; }
        case 7u: { return px.y; }
        default: {}
    }
    // Everything else compares in gamma sRGB (what a monitor shows).
    let lin = oklch_to_linear_srgb(px);
    let srgb = linear_to_srgb(lin);
    switch params.criterion {
        case 0u: { return srgb.x; }
        case 1u: { return srgb.y; }
        case 2u: { return srgb.z; }
        case 3u: {
            // HSV saturation on gamma sRGB.
            let mx = max(max(srgb.x, srgb.y), srgb.z);
            if (mx <= 0.0) { return 0.0; }
            let mn = min(min(srgb.x, srgb.y), srgb.z);
            return (mx - mn) / mx;
        }
        case 4u: {
            // HSV value on gamma sRGB.
            return max(max(srgb.x, srgb.y), srgb.z);
        }
        default: {
            // Luma (Rec.709).
            return 0.2126 * srgb.x + 0.7152 * srgb.y + 0.0722 * srgb.z;
        }
    }
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.size.x || gid.y >= params.size.y) { return; }
    let coord = vec2<i32>(i32(gid.x), i32(gid.y));
    let a = textureLoad(tex_a, coord, 0);
    let b = textureLoad(tex_b, coord, 0);
    let va = criterion_of(a);
    let vb = criterion_of(b);
    var a_wins: bool;
    if (params.mode == 0u) {
        a_wins = va <= vb;
    } else {
        a_wins = va >= vb;
    }
    var out: vec4<f32>;
    if (a_wins) { out = a; } else { out = b; }
    textureStore(out_tex, coord, out);
}
