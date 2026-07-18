// Shared WGSL helpers: Oklch <-> Oklab <-> LinSrgb <-> sRGB.
//
// Constants match Björn Ottosson's original Oklab publication and match
// `palette`'s implementation to within f32 ULP. The CPU-vs-GPU diff harness
// uses a 1/255 tolerance which absorbs any residual difference.

const PI: f32 = 3.14159265358979323846;
const DEG_TO_RAD: f32 = PI / 180.0;
const RAD_TO_DEG: f32 = 180.0 / PI;

// Oklcha (l, chroma, hue-degrees, alpha) as stored in intermediate textures.
struct Oklcha {
    l: f32,
    c: f32,
    h_deg: f32,
    a: f32,
}

fn oklcha_from_vec4(v: vec4<f32>) -> Oklcha {
    return Oklcha(v.x, v.y, v.z, v.w);
}

fn oklcha_to_vec4(c: Oklcha) -> vec4<f32> {
    return vec4<f32>(c.l, c.c, c.h_deg, c.a);
}

// Perceptual scalar-from-color rule; matches core::color::scalar_of.
fn scalar_of(v: vec4<f32>) -> f32 {
    return v.x;
}

// --- Oklch <-> Oklab ---------------------------------------------------

fn oklch_to_oklab(l: f32, c: f32, h_deg: f32) -> vec3<f32> {
    let h_rad = h_deg * DEG_TO_RAD;
    return vec3<f32>(l, c * cos(h_rad), c * sin(h_rad));
}

fn oklab_to_oklch(l: f32, a: f32, b: f32) -> vec3<f32> {
    let c = sqrt(a * a + b * b);
    var h = atan2(b, a) * RAD_TO_DEG;
    if (h < 0.0) { h = h + 360.0; }
    return vec3<f32>(l, c, h);
}

// --- Oklab <-> Linear sRGB (Ottosson constants) ------------------------

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
    let l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b;
    let m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b;
    let s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b;
    let l_ = pow(max(l, 0.0), 1.0 / 3.0);
    let m_ = pow(max(m, 0.0), 1.0 / 3.0);
    let s_ = pow(max(s, 0.0), 1.0 / 3.0);
    return vec3<f32>(
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
    );
}

// --- Linear sRGB <-> sRGB (piecewise gamma) ----------------------------

fn linear_to_srgb_component(x: f32) -> f32 {
    let clamped = clamp(x, 0.0, 1.0);
    if (clamped <= 0.0031308) {
        return 12.92 * clamped;
    } else {
        return 1.055 * pow(clamped, 1.0 / 2.4) - 0.055;
    }
}

fn linear_to_srgb(lin: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        linear_to_srgb_component(lin.x),
        linear_to_srgb_component(lin.y),
        linear_to_srgb_component(lin.z),
    );
}

// Oklcha (unclamped) -> sRGB8-ready floats [0,1] with alpha preserved.
fn oklcha_to_srgb(v: vec4<f32>) -> vec4<f32> {
    let l_clamped = clamp(v.x, 0.0, 1.0);
    let c_clamped = max(v.y, 0.0);
    let lab = oklch_to_oklab(l_clamped, c_clamped, v.z);
    let lin = oklab_to_linear_srgb(lab.x, lab.y, lab.z);
    let srgb = linear_to_srgb(lin);
    return vec4<f32>(srgb, clamp(v.w, 0.0, 1.0));
}
