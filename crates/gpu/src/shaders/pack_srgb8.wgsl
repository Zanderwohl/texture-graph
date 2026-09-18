// Final stage: read one intermediate Rgba32Float slot (Oklcha) and pack to
// a sRGB-encoded Rgba8Unorm texture ready for egui display. Storage format
// stays Rgba8Unorm because wgpu does not allow storage bindings to
// sRGB-view formats; we apply the gamma encoding manually here.
//
// Mode controls the interpretation of the source slot:
//   0 = color   — Oklcha -> sRGB gamma
//   1 = scalar-const (unused: const path baked into a Color layer instead)
//   2 = scalar-layer — take .L, put in RGB, alpha=1
//   3 = normal — assume the slot holds `normal_to_color(n)` (Oklcha of the
//                encoded normal), convert back to sRGB and display

struct PackParams {
    size: vec2<u32>,
    mode: u32,
    // 0 = composite the gray alpha checker behind partial alpha (flat
    // previews); 1 = keep the real alpha in the output so the 3D preview
    // can blend the object itself.
    alpha_object: u32,
    // used only when mode == 1
    const_value: vec4<f32>,
    // Volume slice index in texels (0 for flat bakes) — makes the
    // out-of-range checker a true 3D checkerboard across slices.
    z_px: u32,
    // Three scalar pads, NOT vec3<u32> — vec3 aligns to 16 which would
    // desync the struct layout from the Rust side.
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    // Source layer's bake domain (min_u, min_v, ext_u, ext_v).
    src_dom: vec4<f32>,
}

// common.wgsl (inlined by the pipeline creation code below at build time)
// ---------------------------------------------------------------------
const PI: f32 = 3.14159265358979323846;
const DEG_TO_RAD: f32 = PI / 180.0;

fn oklch_to_oklab(l: f32, c: f32, h_deg: f32) -> vec3<f32> {
    let h_rad = h_deg * DEG_TO_RAD;
    return vec3<f32>(l, c * cos(h_rad), c * sin(h_rad));
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

fn oklcha_to_srgb(v: vec4<f32>) -> vec4<f32> {
    let l_clamped = clamp(v.x, 0.0, 1.0);
    let c_clamped = max(v.y, 0.0);
    let lab = oklch_to_oklab(l_clamped, c_clamped, v.z);
    let lin = oklab_to_linear_srgb(lab.x, lab.y, lab.z);
    let srgb = linear_to_srgb(lin);
    return vec4<f32>(srgb, clamp(v.w, 0.0, 1.0));
}
// ---------------------------------------------------------------------

@group(0) @binding(0) var<uniform> params: PackParams;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<rgba8unorm, write>;

// Out-of-range "presentation" checker. Only affects what the user sees in
// previews — the underlying Rgba32Float intermediates that flow between
// layers stay unclamped, so a signed-noise Mix::Add stack composes exactly
// as before. We just refuse to silently clamp negatives to black etc.
//
// 10-px checker cells alternating bright magenta and black.
const CHECKER_CELL: u32 = 10u;

fn checker_srgb(gid_xy: vec2<u32>) -> vec4<f32> {
    let cell = vec2<u32>(gid_xy.x / CHECKER_CELL, gid_xy.y / CHECKER_CELL);
    // Volume bakes contribute a third cell axis so the pattern is a solid
    // 3D checkerboard, not the same 2D checker extruded through w.
    let parity = (cell.x + cell.y + params.z_px / CHECKER_CELL) & 1u;
    if (parity == 0u) {
        return vec4<f32>(1.0, 0.0, 1.0, 1.0);
    }
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}

// Alpha-backing checker — the Photoshop-style light/dark gray pattern
// shown under partially-transparent pixels so the user actually notices
// alpha < 1. Blended in gamma sRGB, which matches how apps like Photoshop
// display transparency (not physically correct, but the familiar look).
const ALPHA_CHECKER_CELL: u32 = 8u;

fn alpha_backing_srgb(gid_xy: vec2<u32>) -> vec3<f32> {
    let cell = vec2<u32>(gid_xy.x / ALPHA_CHECKER_CELL, gid_xy.y / ALPHA_CHECKER_CELL);
    let parity = (cell.x + cell.y) & 1u;
    if (parity == 0u) {
        return vec3<f32>(0.75, 0.75, 0.75);
    }
    return vec3<f32>(0.55, 0.55, 0.55);
}

// Presentation of partial alpha, switched by `alpha_object`:
// - 0: composite `packed` (foreground, straight-alpha sRGB) over the gray
//   checker; result has alpha = 1 so the flat preview never blends with
//   whatever's underneath the panel.
// - 1: pass the real alpha through untouched — the 3D preview blends the
//   object itself against the scene background instead of faking a
//   backing grid.
fn composite_alpha(packed: vec4<f32>, gid_xy: vec2<u32>) -> vec4<f32> {
    let a = clamp(packed.w, 0.0, 1.0);
    if (params.alpha_object == 1u) {
        return vec4<f32>(packed.xyz, a);
    }
    if (a >= 1.0 - RANGE_EPS) {
        return vec4<f32>(packed.xyz, 1.0);
    }
    let bg = alpha_backing_srgb(gid_xy);
    let mixed = packed.xyz * a + bg * (1.0 - a);
    return vec4<f32>(mixed, 1.0);
}

// Tolerance so f32 wobble near the boundaries doesn't flicker a valid
// image between "in gamut" and "checker".
const RANGE_EPS: f32 = 1e-4;

fn oklcha_out_of_range(v: vec4<f32>) -> bool {
    // L outside [0, 1], negative chroma, or alpha outside [0, 1].
    return v.x < -RANGE_EPS
        || v.x > 1.0 + RANGE_EPS
        || v.y < -RANGE_EPS
        || v.w < -RANGE_EPS
        || v.w > 1.0 + RANGE_EPS;
}

fn scalar_out_of_range(l: f32) -> bool {
    return l < -RANGE_EPS || l > 1.0 + RANGE_EPS;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.size.x || gid.y >= params.size.y) { return; }
    let coord = vec2<i32>(i32(gid.x), i32(gid.y));
    // Display [0, 1] UV mapped into the source's bake domain (identity
    // when the source was baked over the unit square).
    let uv = vec2<f32>(
        (f32(gid.x) + 0.5) / f32(params.size.x),
        (f32(gid.y) + 0.5) / f32(params.size.y),
    );
    let tx = clamp((uv.x - params.src_dom.x) / params.src_dom.z * f32(params.size.x), 0.0, f32(params.size.x) - 1.0);
    let ty = clamp((uv.y - params.src_dom.y) / params.src_dom.w * f32(params.size.y), 0.0, f32(params.size.y) - 1.0);
    let src_px = textureLoad(src, vec2<i32>(i32(tx), i32(ty)), 0);
    var packed: vec4<f32>;
    if (params.mode == 0u) {
        if (oklcha_out_of_range(src_px)) {
            packed = checker_srgb(gid.xy);
        } else {
            packed = composite_alpha(oklcha_to_srgb(src_px), gid.xy);
        }
    } else if (params.mode == 2u) {
        // scalar-from-layer — take L, encode as gray sRGB.
        if (scalar_out_of_range(src_px.x)) {
            packed = checker_srgb(gid.xy);
        } else {
            let g = linear_to_srgb_component(clamp(src_px.x, 0.0, 1.0));
            packed = vec4<f32>(g, g, g, 1.0);
        }
    } else {
        // mode == 3: source already holds a normal encoded via
        // normal_to_color (Oklcha of an sRGB-encoded normal). It's produced
        // by our own encoder so it's always in gamut, but check anyway in
        // case a user routed something weird into the normal slot.
        if (oklcha_out_of_range(src_px)) {
            packed = checker_srgb(gid.xy);
        } else {
            packed = composite_alpha(oklcha_to_srgb(src_px), gid.xy);
        }
    }
    textureStore(dst, coord, packed);
}
