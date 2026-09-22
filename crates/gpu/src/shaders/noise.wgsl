// LayerKind::Noise. A function-for-function port of `core::noise`, which is
// the spec; change both together. Needs `sphere.wgsl` prepended.
//
// - Simplex: Gustavson's textureless simplex. Aperiodic.
// - Value: trilinear value noise, cell index modulo a per-axis period. The
//   hash is pure u32, so corners match the CPU bit for bit and a periodic
//   field tiles exactly.
//
// Range mapping and output modes match `eval_noise` in `core::eval`.

struct NoiseParams {
    size: vec2<u32>,
    dims: u32,            // 0=D1, 1=D2, 2=D3
    range: u32,           // 0=Unsigned, 1=Signed
    output_mode: u32,     // 0=Grayscale, 1=Color
    seed_base: u32,       // ctx.seed + seed_offset
    frequency: f32,
    w_coord: f32,         // third texture coordinate; 0.5 for flat bakes
    dom: vec4<f32>,       // bake domain (min_u, min_v, ext_u, ext_v)
    period: vec3<u32>,    // lattice period in cells; 0 = unbounded on that axis
    octaves: u32,         // 1..=MAX_OCTAVES
    lacunarity: f32,
    gain: f32,
    fractal_mode: u32,    // 0=Standard, 1=Turbulence, 2=Ridged
    normalize: u32,       // 0 = raw sum, 1 = divide by the amplitude sum
    kernel: u32,          // 0=Simplex, 1=Value
    face: u32,            // 0 = plane; k + 1 = cube face k. See sphere.wgsl.
    // Scalars, not a vec2: the Rust layout puts point_map at byte 96.
    _pad1: u32,
    _pad2: u32,
    point_map: array<vec4<f32>, 3>,
}

// Mirrors `core::noise::MAX_OCTAVES`. Bounds the loop so a bad uniform
// cannot hang the dispatch.
const MAX_OCTAVES: u32 = 8u;

// Mirrors `core::noise::OCTAVE_SEED_STRIDE`.
const OCTAVE_SEED_STRIDE: u32 = 0x9E3779B9u;

@group(0) @binding(0) var<uniform> params: NoiseParams;
@group(0) @binding(1) var out_tex: texture_storage_2d<rgba32float, write>;

fn permute3(x: vec3<f32>) -> vec3<f32> {
    return (((x * 34.0) + 1.0) * x) % vec3<f32>(289.0);
}

fn permute4(x: vec4<f32>) -> vec4<f32> {
    return (((x * 34.0) + 1.0) * x) % vec4<f32>(289.0);
}

fn taylor_inv_sqrt4(r: vec4<f32>) -> vec4<f32> {
    return 1.79284291400159 - 0.85373472095314 * r;
}

// Roughly [-1, 1].

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

// Roughly [-1, 1].

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
    // Top 24 bits, so the result is exact in f32 and below 1.
    return f32(x >> 8u) * (1.0 / 16777216.0);
}

// Simplex seeds by translating the input; twin of `seed_offsets`. Takes the
// seed because each octave has its own.
fn seeded_uv_for(u: f32, v: f32, s: u32) -> vec2<f32> {
    let ox = hash_to_float01(wang_hash(s ^ 0xA1B2C3D4u)) * 256.0;
    let oy = hash_to_float01(wang_hash(s ^ 0x51F0E7A9u)) * 256.0;
    return vec2<f32>(u + ox, v + oy);
}

fn seeded_uvw_for(u: f32, v: f32, w: f32, s: u32) -> vec3<f32> {
    let ox = hash_to_float01(wang_hash(s ^ 0xA1B2C3D4u)) * 256.0;
    let oy = hash_to_float01(wang_hash(s ^ 0x51F0E7A9u)) * 256.0;
    let oz = hash_to_float01(wang_hash(s ^ 0x0BADC0DEu)) * 256.0;
    return vec3<f32>(u + ox, v + oy, w + oz);
}

// Twin of `lattice_hash` in core::noise.
fn lattice_hash(cell: vec3<i32>, seed: u32) -> u32 {
    var h = (u32(cell.x) * 1597334677u)
          ^ (u32(cell.y) * 3812015801u)
          ^ (u32(cell.z) * 2654435761u)
          ^ wang_hash(seed);
    h = h ^ (h >> 15u);
    h = h * 2246822519u;
    h = h ^ (h >> 13u);
    h = h * 3266489917u;
    h = h ^ (h >> 16u);
    return h;
}

// Twin of `wrap_cell`. WGSL's `%` on i32 truncates toward zero, as Rust's
// does, so the double-mod is the same expression on both sides.
fn wrap_cell(i: i32, period: u32) -> i32 {
    if (period == 0u) { return i; }
    let p = i32(period);
    return ((i % p) + p) % p;
}

fn smooth_weight(t: f32) -> f32 {
    return t * t * (3.0 - 2.0 * t);
}

fn lerp1(a: f32, b: f32, t: f32) -> f32 {
    return a + (b - a) * t;
}

fn lattice_corner(base: vec3<i32>, d: vec3<i32>, period: vec3<u32>, seed: u32) -> f32 {
    let cell = vec3<i32>(
        wrap_cell(base.x + d.x, period.x),
        wrap_cell(base.y + d.y, period.y),
        wrap_cell(base.z + d.z, period.z),
    );
    return hash_to_float01(lattice_hash(cell, seed)) * 2.0 - 1.0;
}

// Trilinear value noise at `p`, roughly [-1, 1]; twin of `value_noise`.
fn value_noise(p: vec3<f32>, period: vec3<u32>, seed: u32) -> f32 {
    let i = floor(p);
    let base = vec3<i32>(i32(i.x), i32(i.y), i32(i.z));
    let w = vec3<f32>(
        smooth_weight(p.x - i.x),
        smooth_weight(p.y - i.y),
        smooth_weight(p.z - i.z),
    );
    let c00 = lerp1(
        lattice_corner(base, vec3<i32>(0, 0, 0), period, seed),
        lattice_corner(base, vec3<i32>(1, 0, 0), period, seed), w.x);
    let c10 = lerp1(
        lattice_corner(base, vec3<i32>(0, 1, 0), period, seed),
        lattice_corner(base, vec3<i32>(1, 1, 0), period, seed), w.x);
    let c01 = lerp1(
        lattice_corner(base, vec3<i32>(0, 0, 1), period, seed),
        lattice_corner(base, vec3<i32>(1, 0, 1), period, seed), w.x);
    let c11 = lerp1(
        lattice_corner(base, vec3<i32>(0, 1, 1), period, seed),
        lattice_corner(base, vec3<i32>(1, 1, 1), period, seed), w.x);
    let c0 = lerp1(c00, c10, w.y);
    let c1 = lerp1(c01, c11, w.y);
    return lerp1(c0, c1, w.z);
}

// Twin of `octave`. The value kernel seeds through the hash, not by
// translating, so the lattice stays on integers and the period holds.
fn noise_octave(
    u: f32, v: f32, w: f32,
    seed: u32,
    f: f32,
    period: vec3<u32>,
) -> f32 {
    if (params.kernel == 1u) {
        switch params.dims {
            case 0u: {
                return value_noise(vec3<f32>(u * f, 0.0, 0.0),
                    vec3<u32>(period.x, 0u, 0u), seed);
            }
            case 2u: {
                return value_noise(vec3<f32>(u * f, v * f, w * f), period, seed);
            }
            default: {
                return value_noise(vec3<f32>(u * f, v * f, 0.0),
                    vec3<u32>(period.x, period.y, 0u), seed);
            }
        }
    }
    switch params.dims {
        case 0u: {
            let sh = seeded_uv_for(u, 0.0, seed);
            return snoise2(vec2<f32>(sh.x * f, sh.y));
        }
        case 2u: {
            return snoise3(seeded_uvw_for(u, v, w, seed) * f);
        }
        default: {
            return snoise2(seeded_uv_for(u, v, seed) * f);
        }
    }
}

// Twin of `shape`.
fn shape_octave(r: f32) -> f32 {
    switch params.fractal_mode {
        case 1u: { return abs(r); }
        case 2u: {
            let t = 1.0 - abs(r);
            return t * t;
        }
        default: { return r; }
    }
}

// Twin of `octave_period`. `floor(x + 0.5)` rather than `round`, which
// breaks ties to even here and away from zero in Rust.
fn octave_period(base: vec3<u32>, scale: f32) -> vec3<u32> {
    var out = vec3<u32>(0u, 0u, 0u);
    if (base.x != 0u) { out.x = u32(floor(f32(base.x) * scale + 0.5)); }
    if (base.y != 0u) { out.y = u32(floor(f32(base.y) * scale + 0.5)); }
    if (base.z != 0u) { out.z = u32(floor(f32(base.z) * scale + 0.5)); }
    return out;
}

// Twin of `core::noise::sample`. `extra_seed_offset` is the output
// channel (0 for grayscale, 0/1/2 for L/C/hue).
fn sample_noise(u: f32, v: f32, w: f32, extra_seed_offset: u32) -> f32 {
    let seed = params.seed_base + extra_seed_offset;
    let octaves = clamp(params.octaves, 1u, MAX_OCTAVES);
    var frequency = params.frequency;
    var period_scale = 1.0;
    var amplitude = 1.0;
    var sum = 0.0;
    var amplitude_sum = 0.0;

    for (var i: u32 = 0u; i < octaves; i = i + 1u) {
        let r = noise_octave(
            u, v, w,
            seed + i * OCTAVE_SEED_STRIDE,
            frequency,
            octave_period(params.period, period_scale),
        );
        sum = sum + amplitude * shape_octave(r);
        amplitude_sum = amplitude_sum + amplitude;
        frequency = frequency * params.lacunarity;
        period_scale = period_scale * params.lacunarity;
        amplitude = amplitude * params.gain;
    }

    var acc = sum;
    if (params.normalize == 1u && amplitude_sum > 0.0) {
        acc = sum / amplitude_sum;
    }
    // Turbulence and Ridged are in [0, 1]; move them to [-1, 1] first.
    var signed_value = acc;
    if (params.fractal_mode != 0u) {
        signed_value = acc * 2.0 - 1.0;
    }
    if (params.range == 0u) {
        return signed_value * 0.5 + 0.5;
    }
    return signed_value;
}

const CHROMA_SCALE: f32 = 0.15;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.size.x || gid.y >= params.size.y) { return; }
    let p = map_point(params.point_map, sample_point(params.face, params.dom, gid.xy, params.size, params.w_coord));
    let u = p.x;
    let v = p.y;
    let w = p.z;

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
