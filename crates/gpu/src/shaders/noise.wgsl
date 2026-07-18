// LayerKind::Noise. Simplex noise ported to WGSL.
//
// Uses Stefan Gustavson's textureless simplex noise (permutation-polynomial
// gradient hash). This does NOT bit-match the `noise` crate's Simplex used by
// the CPU evaluator: their permutation tables differ, and CPU noise runs in
// f64 while the GPU is f32. The two share only their statistical shape —
// same range, same smoothness class, similar feature scale. GPU-baked
// graphs stay deterministic under fixed (seed, seed_offset, dims, frequency).
//
// Range mapping and Grayscale/Color output modes match `eval_noise` in
// `core::eval` verbatim.

struct NoiseParams {
    size: vec2<u32>,
    dims: u32,            // 0=D1, 1=D2, 2=D3
    range: u32,           // 0=Unsigned, 1=Signed
    output_mode: u32,     // 0=Grayscale, 1=Color
    seed_base: u32,       // ctx.seed + seed_offset
    frequency: f32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: NoiseParams;
@group(0) @binding(1) var out_tex: texture_storage_2d<rgba32float, write>;

// ---- Permutation helpers ----------------------------------------------

fn permute3(x: vec3<f32>) -> vec3<f32> {
    return (((x * 34.0) + 1.0) * x) % vec3<f32>(289.0);
}

fn permute4(x: vec4<f32>) -> vec4<f32> {
    return (((x * 34.0) + 1.0) * x) % vec4<f32>(289.0);
}

fn taylor_inv_sqrt4(r: vec4<f32>) -> vec4<f32> {
    return 1.79284291400159 - 0.85373472095314 * r;
}

// ---- 2D simplex noise (Gustavson) — returns roughly [-1, 1] ----------

fn snoise2(v: vec2<f32>) -> f32 {
    let C = vec4<f32>(
        0.211324865405187,
        0.366025403784439,
       -0.577350269189626,
        0.024390243902439,
    );
    var i  = floor(v + vec2<f32>(dot(v, C.yy)));
    let x0 = v - i + vec2<f32>(dot(i, C.xx));
    var i1: vec2<f32>;
    if (x0.x > x0.y) {
        i1 = vec2<f32>(1.0, 0.0);
    } else {
        i1 = vec2<f32>(0.0, 1.0);
    }
    var x12 = vec4<f32>(x0.x, x0.y, x0.x, x0.y) + vec4<f32>(C.x, C.x, C.z, C.z);
    x12 = vec4<f32>(x12.x - i1.x, x12.y - i1.y, x12.z, x12.w);
    i = i % vec2<f32>(289.0);
    let p = permute3(
        permute3(i.y + vec3<f32>(0.0, i1.y, 1.0)) + i.x + vec3<f32>(0.0, i1.x, 1.0)
    );
    var m = max(
        vec3<f32>(0.5) - vec3<f32>(dot(x0, x0), dot(x12.xy, x12.xy), dot(x12.zw, x12.zw)),
        vec3<f32>(0.0),
    );
    m = m * m; m = m * m;
    let x  = 2.0 * fract(p * C.www) - 1.0;
    let h  = abs(x) - 0.5;
    let ox = floor(x + 0.5);
    let a0 = x - ox;
    m = m * (vec3<f32>(1.79284291400159) - 0.85373472095314 * (a0 * a0 + h * h));
    let g = vec3<f32>(
        a0.x * x0.x + h.x * x0.y,
        a0.y * x12.x + h.y * x12.y,
        a0.z * x12.z + h.z * x12.w,
    );
    return 130.0 * dot(m, g);
}

// ---- 3D simplex noise (Gustavson) — returns roughly [-1, 1] ----------

fn snoise3(v: vec3<f32>) -> f32 {
    let C = vec2<f32>(1.0 / 6.0, 1.0 / 3.0);
    let D = vec4<f32>(0.0, 0.5, 1.0, 2.0);
    var i  = floor(v + vec3<f32>(dot(v, vec3<f32>(C.y))));
    let x0 = v - i + vec3<f32>(dot(i, vec3<f32>(C.x)));
    let g = step(x0.yzx, x0.xyz);
    let l = 1.0 - g;
    let i1 = min(g.xyz, l.zxy);
    let i2 = max(g.xyz, l.zxy);
    let x1 = x0 - i1 + vec3<f32>(C.x);
    let x2 = x0 - i2 + vec3<f32>(C.y);
    let x3 = x0 - vec3<f32>(D.y);
    i = i % vec3<f32>(289.0);
    let p = permute4(
        permute4(
            permute4(i.z + vec4<f32>(0.0, i1.z, i2.z, 1.0))
                + i.y + vec4<f32>(0.0, i1.y, i2.y, 1.0)
        ) + i.x + vec4<f32>(0.0, i1.x, i2.x, 1.0)
    );
    let ns = vec3<f32>(0.142857142857) * D.wyz - D.xzx;
    let j  = p - 49.0 * floor(p * ns.z * ns.z);
    let x_ = floor(j * ns.z);
    let y_ = floor(j - 7.0 * x_);
    let x  = x_ * ns.x + vec4<f32>(ns.y);
    let y  = y_ * ns.x + vec4<f32>(ns.y);
    let h  = 1.0 - abs(x) - abs(y);
    let b0 = vec4<f32>(x.x, x.y, y.x, y.y);
    let b1 = vec4<f32>(x.z, x.w, y.z, y.w);
    let s0 = floor(b0) * 2.0 + 1.0;
    let s1 = floor(b1) * 2.0 + 1.0;
    let sh = -step(h, vec4<f32>(0.0));
    let a0 = vec4<f32>(b0.x, b0.z, b0.y, b0.w) + vec4<f32>(s0.x, s0.z, s0.y, s0.w) * vec4<f32>(sh.x, sh.x, sh.y, sh.y);
    let a1 = vec4<f32>(b1.x, b1.z, b1.y, b1.w) + vec4<f32>(s1.x, s1.z, s1.y, s1.w) * vec4<f32>(sh.z, sh.z, sh.w, sh.w);
    var p0 = vec3<f32>(a0.x, a0.y, h.x);
    var p1 = vec3<f32>(a0.z, a0.w, h.y);
    var p2 = vec3<f32>(a1.x, a1.y, h.z);
    var p3 = vec3<f32>(a1.z, a1.w, h.w);
    let norm = taylor_inv_sqrt4(vec4<f32>(
        dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)));
    p0 = p0 * norm.x;
    p1 = p1 * norm.y;
    p2 = p2 * norm.z;
    p3 = p3 * norm.w;
    var m = max(
        vec4<f32>(0.6) - vec4<f32>(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)),
        vec4<f32>(0.0),
    );
    m = m * m;
    return 42.0 * dot(m * m, vec4<f32>(dot(p0, x0), dot(p1, x1), dot(p2, x2), dot(p3, x3)));
}

// ---- Sampling ---------------------------------------------------------

// Seed mixed into the input as a translation. Not cryptographic — different
// (seed, seed_offset) pairs produce visually distinct noise, which is
// what the UI needs. Wanghash of the raw seed provides three independent
// low-bit-mixed offsets.
fn wang_hash(seed: u32) -> u32 {
    var x = seed;
    x = (x ^ 61u) ^ (x >> 16u);
    x = x + (x << 3u);
    x = x ^ (x >> 4u);
    x = x * 0x27d4eb2du;
    x = x ^ (x >> 15u);
    return x;
}

fn hash_to_float01(x: u32) -> f32 {
    // Map to [0, 1) by taking the top 24 bits into an f32 mantissa.
    return f32(x >> 8u) * (1.0 / 16777216.0);
}

fn seeded_uv(u: f32, v: f32, extra_seed_offset: u32) -> vec2<f32> {
    let s = params.seed_base + extra_seed_offset;
    let ox = hash_to_float01(wang_hash(s ^ 0xA1B2C3D4u)) * 256.0;
    let oy = hash_to_float01(wang_hash(s ^ 0x51F0E7A9u)) * 256.0;
    return vec2<f32>(u + ox, v + oy);
}

fn seeded_uvw(u: f32, v: f32, w: f32, extra_seed_offset: u32) -> vec3<f32> {
    let s = params.seed_base + extra_seed_offset;
    let ox = hash_to_float01(wang_hash(s ^ 0xA1B2C3D4u)) * 256.0;
    let oy = hash_to_float01(wang_hash(s ^ 0x51F0E7A9u)) * 256.0;
    let oz = hash_to_float01(wang_hash(s ^ 0x0BADC0DEu)) * 256.0;
    return vec3<f32>(u + ox, v + oy, w + oz);
}

fn sample_noise(u: f32, v: f32, w: f32, extra_seed_offset: u32) -> f32 {
    let f = params.frequency;
    var raw: f32;
    switch params.dims {
        case 0u: {
            // D1: sample 2D with y=0.
            let sh = seeded_uv(u, 0.0, extra_seed_offset);
            raw = snoise2(vec2<f32>(sh.x * f, sh.y));
        }
        case 2u: {
            let p = seeded_uvw(u, v, w, extra_seed_offset) * f;
            raw = snoise3(p);
        }
        default: {
            // D2 (and any other value).
            let p = seeded_uv(u, v, extra_seed_offset) * f;
            raw = snoise2(p);
        }
    }
    // range: 0=Unsigned -> [0,1], 1=Signed -> [-1, 1].
    if (params.range == 0u) {
        return raw * 0.5 + 0.5;
    }
    return raw;
}

const CHROMA_SCALE: f32 = 0.15;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.size.x || gid.y >= params.size.y) { return; }
    // Pixel-center sample coordinates in [0, 1].
    let u = (f32(gid.x) + 0.5) / f32(params.size.x);
    let v = (f32(gid.y) + 0.5) / f32(params.size.y);
    let w = 0.5;

    if (params.output_mode == 0u) {
        let n = sample_noise(u, v, w, 0u);
        textureStore(out_tex, vec2<i32>(i32(gid.x), i32(gid.y)),
            vec4<f32>(n, 0.0, 0.0, 1.0));
    } else {
        let nl = sample_noise(u, v, w, 0u);
        let nc = sample_noise(u, v, w, 1u);
        let nh = sample_noise(u, v, w, 2u);
        var hue: f32;
        if (params.range == 1u) {
            hue = nh * 180.0;
        } else {
            hue = nh * 360.0;
        }
        textureStore(out_tex, vec2<i32>(i32(gid.x), i32(gid.y)),
            vec4<f32>(nl, nc * CHROMA_SCALE, hue, 1.0));
    }
}
