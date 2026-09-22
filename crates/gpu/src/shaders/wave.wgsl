// LayerKind::Wave. Twin of `eval_wave` / `wave_cycle` in `core::eval`.
// `sin` differs between backends, so they agree to one sRGB step, not bit
// for bit.
//
// When `input_is_layer == 0`, `src` is a placeholder and is not read.

struct WaveParams {
    size: vec2<u32>,
    shape: u32,           // 0=Sine, 1=Triangle, 2=Square, 3=Sawtooth
    range: u32,           // 0=Unsigned, 1=Signed
    frequency: f32,
    phase: f32,
    input_const: f32,
    input_is_layer: u32,  // 0=Const, 1=Layer
    dom: vec4<f32>,       // own bake domain (min_u, min_v, ext_u, ext_v)
    dom_input: vec4<f32>,
}

@group(0) @binding(0) var<uniform> params: WaveParams;
@group(0) @binding(1) var out_tex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var src: texture_2d<f32>;

const TAU: f32 = 6.28318530717958647693;

fn dom_uv(dom: vec4<f32>, gid: vec2<u32>, size: vec2<u32>) -> vec2<f32> {
    return vec2<f32>(
        dom.x + (f32(gid.x) + 0.5) / f32(size.x) * dom.z,
        dom.y + (f32(gid.y) + 0.5) / f32(size.y) * dom.w,
    );
}

fn dom_texel(dom: vec4<f32>, uv: vec2<f32>, size: vec2<u32>) -> vec2<i32> {
    let tx = (uv.x - dom.x) / dom.z * f32(size.x);
    let ty = (uv.y - dom.y) / dom.w * f32(size.y);
    return vec2<i32>(
        i32(clamp(tx, 0.0, f32(size.x) - 1.0)),
        i32(clamp(ty, 0.0, f32(size.y) - 1.0)),
    );
}

// Not `fract`: Rust's `f32::fract` truncates toward zero, which differs
// for negative input.
fn frac(x: f32) -> f32 {
    return x - floor(x);
}

// One cycle at phase `t`, in [-1, 1]. Twin of `wave_cycle`.
fn wave_cycle(t_in: f32) -> f32 {
    let t = frac(t_in);
    switch params.shape {
        case 1u: {
            // Phase-aligned with sine: 0 at t = 0, +1 at t = 0.25.
            return 1.0 - 4.0 * abs(frac(t + 0.25) - 0.5);
        }
        case 2u: {
            if (t < 0.5) { return 1.0; }
            return -1.0;
        }
        case 3u: {
            return 2.0 * t - 1.0;
        }
        default: {
            return sin(TAU * t);
        }
    }
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.size.x || gid.y >= params.size.y) { return; }
    let coord = vec2<i32>(i32(gid.x), i32(gid.y));

    var x = params.input_const;
    if (params.input_is_layer == 1u) {
        let uv = dom_uv(params.dom, gid.xy, params.size);
        // L, as core::color::scalar_of.
        x = textureLoad(src, dom_texel(params.dom_input, uv, params.size), 0).x;
    }

    let v = wave_cycle(x * params.frequency + params.phase);
    var l = v;
    if (params.range == 0u) {
        l = v * 0.5 + 0.5;
    }
    textureStore(out_tex, coord, vec4<f32>(l, 0.0, 0.0, 1.0));
}
