//! Noise kernels — the *specification* both backends implement.
//!
//! The WGSL twin is `crates/gpu/src/shaders/noise.wgsl`, function for
//! function. Any change here is a spec change and must land on both sides in
//! the same commit, or a graph will look one way in a GPU bake and another in
//! the CPU fallback.
//!
//! Two kernels live here:
//!
//! - [`Kernel::Simplex`] — Gustavson's textureless simplex, described below.
//!   Aperiodic: there is no lattice to wrap, so it cannot tile.
//! - [`Kernel::Value`] — trilinear value noise on an integer lattice, whose
//!   cell index can be taken modulo a per-axis period. That is what makes a
//!   baked texture seamless, and what reproduces a consumer's own periodic
//!   value noise.
//!
//! Both feed [`sample`], which also runs the [`Fractal`] octave loop.
//!
//! Stefan Gustavson's textureless simplex noise: a permutation-polynomial
//! gradient hash, entirely `f32` polynomial arithmetic. An `f64` kernel off a
//! different permutation table would produce visibly different noise from the
//! GPU, so the same graph would render one way with a working wgpu backend
//! and another without.
//!
//! # Why the seed translates the coordinates
//!
//! Gustavson's kernel has no permutation table to reseed: the hash is baked
//! into the polynomial. The seed is hashed to a fixed offset and added to the
//! sample position instead.
//!
//! The value kernel is seeded differently, and has to be: a fractional
//! translation would move the lattice off the integers and destroy the
//! tiling the kernel exists for. Its seed goes into the lattice *hash*
//! instead, so the cell grid stays put however the field is reseeded.
//!
//! # Op-order contract
//!
//! Only `+ - *`, comparisons, `floor`, `abs`, `min`/`max` and float remainder
//! appear here — nothing transcendental, nothing whose rounding could differ
//! between backends. `fract` is written out as `x - floor(x)` because Rust's
//! `f32::fract` truncates toward zero and WGSL's does not, which would differ
//! on every negative coordinate.
//!
//! The value kernel's hash is wholly `u32` arithmetic, so the two backends
//! agree on lattice corner values bit for bit; only the trilinear blend is
//! float, and it is six multiply-adds a sample.
//!
//! Every float literal is written with the *same decimal string* as the
//! shader, so both sides round one number to `f32` rather than two different
//! shortenings of it. That is why this module turns off
//! `clippy::excessive_precision`, which would helpfully truncate them and
//! quietly reintroduce the divergence.
//!
//! # What parity is held
//!
//! `cpu_and_gpu_noise_agree` in `texture-graph-gpu` holds the two backends to
//! **within one sRGB step** across D1/D2/D3 and both ranges, most samples
//! identical.
//!
//! Not bit-exactness: that would need a float read-back path, so the
//! comparison is on the field rather than 8-bit pixels, and every
//! multiply-add pinned into an explicitly fused form, since shader compilers
//! contract `a*b + c` unasked. The residual ±1 also covers the Oklch→sRGB
//! stage, which is its own pair of implementations.

#![allow(clippy::excessive_precision)]

/// Dimensionality of the noise field, mirroring [`crate::NoiseDims`].
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Dims {
    D1,
    D2,
    D3,
}

/// Which kernel generates the field, mirroring [`crate::NoiseKernel`].
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Kernel {
    Simplex,
    Value,
}

/// How an octave's raw sample is shaped before it is summed, mirroring
/// [`crate::FractalMode`].
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum FractalMode {
    Standard,
    Turbulence,
    Ridged,
}

/// Octave stack, mirroring [`crate::Fractal`].
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Fractal {
    pub octaves: u32,
    pub lacunarity: f32,
    pub gain: f32,
    pub mode: FractalMode,
    pub normalize: bool,
}

/// Upper bound on [`Fractal::octaves`]. The shader unrolls nothing, but a
/// bound keeps a typo from costing a thousand samples a pixel.
pub const MAX_OCTAVES: u32 = 8;

/// Everything about a noise field except where it is sampled and what seed
/// it runs under.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Spec {
    pub dims: Dims,
    pub kernel: Kernel,
    /// Cycles per unit sample-space, at the first octave.
    pub frequency: f32,
    /// Lattice period in cells, per axis; `0` = unbounded. [`Kernel::Value`]
    /// only.
    pub period: [u32; 3],
    pub fractal: Fractal,
    /// `true` = `[-1, 1]`, `false` = `[0, 1]`.
    pub signed: bool,
}

// ---- Permutation helpers -----------------------------------------------

/// `(((x * 34) + 1) * x) mod 289` — the permutation polynomial.
fn permute(x: f32) -> f32 {
    (((x * 34.0) + 1.0) * x) % 289.0
}

fn taylor_inv_sqrt(r: f32) -> f32 {
    1.79284291400159 - 0.85373472095314 * r
}

/// WGSL `fract`: `x - floor(x)`, which for negative `x` is *not* Rust's
/// `f32::fract`.
fn fract(x: f32) -> f32 {
    x - x.floor()
}

/// WGSL `step(edge, x)`: 1 when `x >= edge`, else 0.
fn step(edge: f32, x: f32) -> f32 {
    if x >= edge { 1.0 } else { 0.0 }
}

fn dot2(a: [f32; 2], b: [f32; 2]) -> f32 {
    a[0] * b[0] + a[1] * b[1]
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn dot4(a: [f32; 4], b: [f32; 4]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
}

// ---- 2D ----------------------------------------------------------------

/// 2D simplex noise, roughly `[-1, 1]`.
pub fn snoise2(v: [f32; 2]) -> f32 {
    const CX: f32 = 0.211324865405187;
    const CY: f32 = 0.366025403784439;
    const CZ: f32 = -0.577350269189626;
    const CW: f32 = 0.024390243902439;

    let s = dot2(v, [CY, CY]);
    let mut i = [(v[0] + s).floor(), (v[1] + s).floor()];
    let t = dot2(i, [CX, CX]);
    let x0 = [v[0] - i[0] + t, v[1] - i[1] + t];

    let i1 = if x0[0] > x0[1] { [1.0, 0.0] } else { [0.0, 1.0] };

    // x12 = (x0.xy + C.xx, x0.xy + C.zz), then the first pair shifted by i1.
    let x12 = [
        x0[0] + CX - i1[0],
        x0[1] + CX - i1[1],
        x0[0] + CZ,
        x0[1] + CZ,
    ];

    i = [i[0] % 289.0, i[1] % 289.0];
    let p = [
        permute(permute(i[1]) + i[0]),
        permute(permute(i[1] + i1[1]) + i[0] + i1[0]),
        permute(permute(i[1] + 1.0) + i[0] + 1.0),
    ];

    let mut m = [
        (0.5 - dot2(x0, x0)).max(0.0),
        (0.5 - dot2([x12[0], x12[1]], [x12[0], x12[1]])).max(0.0),
        (0.5 - dot2([x12[2], x12[3]], [x12[2], x12[3]])).max(0.0),
    ];
    for e in &mut m {
        *e = *e * *e;
        *e = *e * *e;
    }

    let mut g = [0.0f32; 3];
    for k in 0..3 {
        let x = 2.0 * fract(p[k] * CW) - 1.0;
        let h = x.abs() - 0.5;
        let ox = (x + 0.5).floor();
        let a0 = x - ox;
        m[k] *= 1.79284291400159 - 0.85373472095314 * (a0 * a0 + h * h);
        g[k] = match k {
            0 => a0 * x0[0] + h * x0[1],
            1 => a0 * x12[0] + h * x12[1],
            _ => a0 * x12[2] + h * x12[3],
        };
    }

    130.0 * dot3(m, g)
}

// ---- 3D ----------------------------------------------------------------

/// 3D simplex noise, roughly `[-1, 1]`.
///
/// The index loops walk four lanes of several arrays at once, which is what
/// the shader's `vec4` lanes are. Zipping them would read better in
/// isolation and worse against the twin, and the twin is the point.
#[allow(clippy::needless_range_loop)]
pub fn snoise3(v: [f32; 3]) -> f32 {
    const CX: f32 = 1.0 / 6.0;
    const CY: f32 = 1.0 / 3.0;

    let s = dot3(v, [CY, CY, CY]);
    let mut i = [(v[0] + s).floor(), (v[1] + s).floor(), (v[2] + s).floor()];
    let t = dot3(i, [CX, CX, CX]);
    let x0 = [v[0] - i[0] + t, v[1] - i[1] + t, v[2] - i[2] + t];

    // g = step(x0.yzx, x0.xyz), l = 1 - g.
    let g = [step(x0[1], x0[0]), step(x0[2], x0[1]), step(x0[0], x0[2])];
    let l = [1.0 - g[0], 1.0 - g[1], 1.0 - g[2]];
    // i1 = min(g.xyz, l.zxy), i2 = max(g.xyz, l.zxy).
    let lzxy = [l[2], l[0], l[1]];
    let i1 = [g[0].min(lzxy[0]), g[1].min(lzxy[1]), g[2].min(lzxy[2])];
    let i2 = [g[0].max(lzxy[0]), g[1].max(lzxy[1]), g[2].max(lzxy[2])];

    let x1 = [x0[0] - i1[0] + CX, x0[1] - i1[1] + CX, x0[2] - i1[2] + CX];
    let x2 = [x0[0] - i2[0] + CY, x0[1] - i2[1] + CY, x0[2] - i2[2] + CY];
    let x3 = [x0[0] - 0.5, x0[1] - 0.5, x0[2] - 0.5];

    i = [i[0] % 289.0, i[1] % 289.0, i[2] % 289.0];
    let off = |k: usize| -> [f32; 4] { [0.0, i1[k], i2[k], 1.0] };
    let mut p = [0.0f32; 4];
    for n in 0..4 {
        let a = permute(i[2] + off(2)[n]);
        let b = permute(a + i[1] + off(1)[n]);
        p[n] = permute(b + i[0] + off(0)[n]);
    }

    // ns = 1/7 * (2, 0.5, 1) - (0, 1, 0)
    const NS_X: f32 = 0.142857142857 * 2.0;
    const NS_Y: f32 = 0.142857142857 * 0.5 - 1.0;
    const NS_Z: f32 = 0.142857142857;

    let mut x = [0.0f32; 4];
    let mut y = [0.0f32; 4];
    let mut h = [0.0f32; 4];
    for n in 0..4 {
        let j = p[n] - 49.0 * (p[n] * NS_Z * NS_Z).floor();
        let x_ = (j * NS_Z).floor();
        let y_ = (j - 7.0 * x_).floor();
        x[n] = x_ * NS_X + NS_Y;
        y[n] = y_ * NS_X + NS_Y;
        h[n] = 1.0 - x[n].abs() - y[n].abs();
    }

    let b0 = [x[0], x[1], y[0], y[1]];
    let b1 = [x[2], x[3], y[2], y[3]];
    let s0 = [
        b0[0].floor() * 2.0 + 1.0,
        b0[1].floor() * 2.0 + 1.0,
        b0[2].floor() * 2.0 + 1.0,
        b0[3].floor() * 2.0 + 1.0,
    ];
    let s1 = [
        b1[0].floor() * 2.0 + 1.0,
        b1[1].floor() * 2.0 + 1.0,
        b1[2].floor() * 2.0 + 1.0,
        b1[3].floor() * 2.0 + 1.0,
    ];
    // sh = -step(h, 0)
    let sh = [
        -step(h[0], 0.0),
        -step(h[1], 0.0),
        -step(h[2], 0.0),
        -step(h[3], 0.0),
    ];

    let a0 = [
        b0[0] + s0[0] * sh[0],
        b0[2] + s0[2] * sh[0],
        b0[1] + s0[1] * sh[1],
        b0[3] + s0[3] * sh[1],
    ];
    let a1 = [
        b1[0] + s1[0] * sh[2],
        b1[2] + s1[2] * sh[2],
        b1[1] + s1[1] * sh[3],
        b1[3] + s1[3] * sh[3],
    ];

    let mut p0 = [a0[0], a0[1], h[0]];
    let mut p1 = [a0[2], a0[3], h[1]];
    let mut p2 = [a1[0], a1[1], h[2]];
    let mut p3 = [a1[2], a1[3], h[3]];

    let norm = [
        taylor_inv_sqrt(dot3(p0, p0)),
        taylor_inv_sqrt(dot3(p1, p1)),
        taylor_inv_sqrt(dot3(p2, p2)),
        taylor_inv_sqrt(dot3(p3, p3)),
    ];
    for k in 0..3 {
        p0[k] *= norm[0];
        p1[k] *= norm[1];
        p2[k] *= norm[2];
        p3[k] *= norm[3];
    }

    let mut m = [
        (0.6 - dot3(x0, x0)).max(0.0),
        (0.6 - dot3(x1, x1)).max(0.0),
        (0.6 - dot3(x2, x2)).max(0.0),
        (0.6 - dot3(x3, x3)).max(0.0),
    ];
    for e in &mut m {
        *e = *e * *e;
    }
    let m2 = [m[0] * m[0], m[1] * m[1], m[2] * m[2], m[3] * m[3]];

    42.0 * dot4(
        m2,
        [dot3(p0, x0), dot3(p1, x1), dot3(p2, x2), dot3(p3, x3)],
    )
}

// ---- Value noise -------------------------------------------------------

/// Hash of one lattice corner. Integer end to end, so the two backends
/// produce the *same bits* for a corner rather than two roundings of one
/// polynomial — which is what lets a periodic field match itself exactly
/// across a tile boundary.
///
/// Mixing constants are the consumer's own plume hash (§1 of
/// `documentation/game-consumer-features.md`), finished with the usual
/// xor-shift-multiply avalanche.
fn lattice_hash(cell: [i32; 3], seed: u32) -> u32 {
    let mut h = (cell[0] as u32).wrapping_mul(1_597_334_677)
        ^ (cell[1] as u32).wrapping_mul(3_812_015_801)
        ^ (cell[2] as u32).wrapping_mul(2_654_435_761)
        ^ wang_hash(seed);
    h ^= h >> 15;
    h = h.wrapping_mul(2_246_822_519);
    h ^= h >> 13;
    h = h.wrapping_mul(3_266_489_917);
    h ^= h >> 16;
    h
}

/// Wrap a lattice cell index into `[0, period)`. `period == 0` is
/// "unbounded on this axis" and passes the index through. Written the way
/// WGSL has to write it (`%` truncates toward zero on both sides), not as
/// `rem_euclid`, so the twin is a transcription.
fn wrap_cell(i: i32, period: u32) -> i32 {
    if period == 0 {
        return i;
    }
    let p = period as i32;
    ((i % p) + p) % p
}

/// Smoothstep weight `t²(3 - 2t)`. Gives C¹ continuity across cell walls;
/// plain linear weights would show the lattice as a grid of creases.
fn smooth_weight(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Trilinear value noise at `p`, roughly `[-1, 1]`.
///
/// The field repeats every `period` cells on each axis where `period != 0`.
fn value_noise(p: [f32; 3], period: [u32; 3], seed: u32) -> f32 {
    let i = [p[0].floor(), p[1].floor(), p[2].floor()];
    let base = [i[0] as i32, i[1] as i32, i[2] as i32];
    let w = [
        smooth_weight(p[0] - i[0]),
        smooth_weight(p[1] - i[1]),
        smooth_weight(p[2] - i[2]),
    ];
    let corner = |dx: i32, dy: i32, dz: i32| -> f32 {
        let cell = [
            wrap_cell(base[0] + dx, period[0]),
            wrap_cell(base[1] + dy, period[1]),
            wrap_cell(base[2] + dz, period[2]),
        ];
        hash_to_float01(lattice_hash(cell, seed)) * 2.0 - 1.0
    };
    let c00 = lerp(corner(0, 0, 0), corner(1, 0, 0), w[0]);
    let c10 = lerp(corner(0, 1, 0), corner(1, 1, 0), w[0]);
    let c01 = lerp(corner(0, 0, 1), corner(1, 0, 1), w[0]);
    let c11 = lerp(corner(0, 1, 1), corner(1, 1, 1), w[0]);
    let c0 = lerp(c00, c10, w[1]);
    let c1 = lerp(c01, c11, w[1]);
    lerp(c0, c1, w[2])
}

// ---- Seeding -----------------------------------------------------------

/// Wang hash. Not cryptographic — it only has to make different
/// `(seed, seed_offset)` pairs produce visually distinct noise.
pub fn wang_hash(seed: u32) -> u32 {
    let mut x = seed;
    x = (x ^ 61) ^ (x >> 16);
    x = x.wrapping_add(x << 3);
    x ^= x >> 4;
    x = x.wrapping_mul(0x27d4_eb2d);
    x ^= x >> 15;
    x
}

/// Top 24 bits of `x` as a float in `[0, 1)`.
pub fn hash_to_float01(x: u32) -> f32 {
    (x >> 8) as f32 * (1.0 / 16_777_216.0)
}

/// The translation this seed applies to the sample position, per axis. The
/// XOR constants are arbitrary and must match the shader's.
fn seed_offsets(seed: u32) -> [f32; 3] {
    [
        hash_to_float01(wang_hash(seed ^ 0xA1B2_C3D4)) * 256.0,
        hash_to_float01(wang_hash(seed ^ 0x51F0_E7A9)) * 256.0,
        hash_to_float01(wang_hash(seed ^ 0x0BAD_C0DE)) * 256.0,
    ]
}

/// Stride between octave seeds. Large and odd so octave `i` of one output
/// channel never lands on octave `j` of the next — channels are only one
/// apart, and a collision there would correlate L with C visibly.
const OCTAVE_SEED_STRIDE: u32 = 0x9E37_79B9;

/// One raw octave, in the kernel's own roughly-`[-1, 1]` convention.
fn octave(
    dims: Dims,
    kernel: Kernel,
    seed: u32,
    frequency: f32,
    period: [u32; 3],
    uvw: [f32; 3],
) -> f32 {
    let f = frequency;
    match kernel {
        Kernel::Simplex => {
            let o = seed_offsets(seed);
            match dims {
                // D1 samples the 2D kernel with y pinned — and, as on the
                // GPU, only the x axis is scaled by frequency.
                Dims::D1 => snoise2([(uvw[0] + o[0]) * f, o[1]]),
                Dims::D2 => snoise2([(uvw[0] + o[0]) * f, (uvw[1] + o[1]) * f]),
                Dims::D3 => snoise3([
                    (uvw[0] + o[0]) * f,
                    (uvw[1] + o[1]) * f,
                    (uvw[2] + o[2]) * f,
                ]),
            }
        }
        // Unused axes are pinned to exactly 0, which lands on a lattice
        // plane rather than somewhere arbitrary inside a cell. The seed
        // stays out of the coordinates — see the module docs.
        Kernel::Value => match dims {
            Dims::D1 => value_noise([uvw[0] * f, 0.0, 0.0], [period[0], 0, 0], seed),
            Dims::D2 => value_noise(
                [uvw[0] * f, uvw[1] * f, 0.0],
                [period[0], period[1], 0],
                seed,
            ),
            Dims::D3 => value_noise([uvw[0] * f, uvw[1] * f, uvw[2] * f], period, seed),
        },
    }
}

/// Shape one octave before it is summed. Expressed on the signed sample
/// `r ∈ [-1, 1]`, so `1 - |2n - 1|` over an unsigned `n` is just `1 - |r|`.
fn shape(r: f32, mode: FractalMode) -> f32 {
    match mode {
        FractalMode::Standard => r,
        FractalMode::Turbulence => r.abs(),
        FractalMode::Ridged => {
            let t = 1.0 - r.abs();
            t * t
        }
    }
}

/// This octave's lattice period: the base period scaled alongside the
/// frequency, so a tiling field stays tiling as octaves are added.
///
/// `floor(x + 0.5)` rather than `round`, because WGSL's `round` breaks ties
/// to even and Rust's breaks them away from zero. Only an integral
/// `lacunarity` keeps this exact — at 2.13 the scaled period and the scaled
/// frequency drift apart and the tile seams.
fn octave_period(base: [u32; 3], scale: f32) -> [u32; 3] {
    let axis = |p: u32| -> u32 {
        if p == 0 {
            0
        } else {
            (p as f32 * scale + 0.5).floor() as u32
        }
    };
    [axis(base[0]), axis(base[1]), axis(base[2])]
}

/// One noise sample, in the same `[-1, 1]` (signed) or `[0, 1]` (unsigned)
/// convention the shader uses.
///
/// `seed` is already `ctx.seed + seed_offset + channel`. The twin is
/// `sample_noise` in noise.wgsl.
///
/// The octave loop always produces a signed-convention value, and
/// [`Spec::signed`] maps it: `Turbulence` and `Ridged` land in `[0, 1]`
/// naturally, so they are stretched to `[-1, 1]` and — for the unsigned
/// case — folded straight back. One rule, rather than a range that depends
/// on the mode.
///
/// A one-octave `Standard` fractal reduces to the bare kernel exactly,
/// which is what a file written before fractals existed loads as.
pub fn sample(spec: &Spec, seed: u32, uvw: [f32; 3]) -> f32 {
    let octaves = spec.fractal.octaves.clamp(1, MAX_OCTAVES);
    let mut frequency = spec.frequency;
    let mut period_scale = 1.0f32;
    let mut amplitude = 1.0f32;
    let mut sum = 0.0f32;
    let mut amplitude_sum = 0.0f32;

    for i in 0..octaves {
        let r = octave(
            spec.dims,
            spec.kernel,
            seed.wrapping_add(i.wrapping_mul(OCTAVE_SEED_STRIDE)),
            frequency,
            octave_period(spec.period, period_scale),
            uvw,
        );
        sum += amplitude * shape(r, spec.fractal.mode);
        amplitude_sum += amplitude;
        frequency *= spec.fractal.lacunarity;
        period_scale *= spec.fractal.lacunarity;
        amplitude *= spec.fractal.gain;
    }

    let acc = if spec.fractal.normalize && amplitude_sum > 0.0 {
        sum / amplitude_sum
    } else {
        sum
    };
    let signed = match spec.fractal.mode {
        FractalMode::Standard => acc,
        FractalMode::Turbulence | FractalMode::Ridged => acc * 2.0 - 1.0,
    };
    if spec.signed { signed } else { signed * 0.5 + 0.5 }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The kernel is meant to land in roughly `[-1, 1]` and to actually
    /// vary. A constant or an out-of-range field would sail past every
    /// parity check — both backends would agree on nonsense.
    #[test]
    fn the_kernels_vary_and_stay_in_range() {
        let mut lo2 = f32::INFINITY;
        let mut hi2 = f32::NEG_INFINITY;
        let mut lo3 = f32::INFINITY;
        let mut hi3 = f32::NEG_INFINITY;
        for a in 0..40 {
            for b in 0..40 {
                let (x, y) = (a as f32 * 0.37 - 7.0, b as f32 * 0.41 - 7.0);
                let n2 = snoise2([x, y]);
                let n3 = snoise3([x, y, x * 0.5 - y * 0.25]);
                lo2 = lo2.min(n2);
                hi2 = hi2.max(n2);
                lo3 = lo3.min(n3);
                hi3 = hi3.max(n3);
            }
        }
        assert!(lo2 >= -1.05 && hi2 <= 1.05, "2D out of range: {lo2}..{hi2}");
        assert!(lo3 >= -1.05 && hi3 <= 1.05, "3D out of range: {lo3}..{hi3}");
        assert!(hi2 - lo2 > 1.0, "2D barely varies: {lo2}..{hi2}");
        assert!(hi3 - lo3 > 1.0, "3D barely varies: {lo3}..{hi3}");
    }

    /// Negative coordinates are the case Rust's own `fract` would get wrong,
    /// and they are ordinary here: the seed offset is positive but a
    /// transform can push a sample anywhere.
    #[test]
    fn negative_coordinates_are_continuous_across_zero() {
        for k in 1..20 {
            let e = k as f32 * 1e-3;
            let a = snoise2([-e, 0.25]);
            let b = snoise2([e, 0.25]);
            assert!(
                (a - b).abs() < 0.2,
                "discontinuity across x=0 at ±{e}: {a} vs {b}"
            );
        }
    }

    /// One octave of `Standard`, the shape a file without fractal fields
    /// loads as.
    fn plain(dims: Dims, kernel: Kernel, frequency: f32, signed: bool) -> Spec {
        Spec {
            dims,
            kernel,
            frequency,
            period: [0; 3],
            fractal: Fractal {
                octaves: 1,
                lacunarity: 2.0,
                gain: 0.5,
                mode: FractalMode::Standard,
                normalize: true,
            },
            signed,
        }
    }

    /// Different seeds have to give different fields, or `seed_offset` is
    /// decorative. Both kernels: the value kernel seeds through its hash
    /// rather than the coordinates, so this is a separate claim for it.
    #[test]
    fn the_seed_moves_the_field() {
        for kernel in [Kernel::Simplex, Kernel::Value] {
            let spec = plain(Dims::D2, kernel, 4.0, true);
            let at = |seed| sample(&spec, seed, [0.3, 0.7, 0.5]);
            assert_ne!(at(0), at(1), "{kernel:?}");
            assert_ne!(at(1), at(2), "{kernel:?}");
        }
    }

    /// Unsigned is the signed field mapped onto `[0, 1]`, which is what the
    /// shader's `raw * 0.5 + 0.5` says.
    #[test]
    fn unsigned_is_the_signed_field_remapped() {
        for k in 0..10 {
            let uvw = [k as f32 * 0.1, 0.4, 0.6];
            let signed = sample(&plain(Dims::D2, Kernel::Simplex, 3.0, true), 7, uvw);
            let unsigned = sample(&plain(Dims::D2, Kernel::Simplex, 3.0, false), 7, uvw);
            assert_eq!(unsigned, signed * 0.5 + 0.5);
        }
    }

    /// The value kernel has to satisfy the same range-and-varies claim the
    /// simplex one does, or a tiling bake would tile nothing usefully.
    #[test]
    fn the_value_kernel_varies_and_stays_in_range() {
        let spec = plain(Dims::D3, Kernel::Value, 5.0, true);
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for a in 0..40 {
            for b in 0..40 {
                let (u, v) = (a as f32 / 40.0, b as f32 / 40.0);
                let n = sample(&spec, 11, [u, v, u * 0.5 + v * 0.25]);
                lo = lo.min(n);
                hi = hi.max(n);
            }
        }
        assert!(lo >= -1.0 && hi <= 1.0, "out of range: {lo}..{hi}");
        assert!(hi - lo > 1.0, "barely varies: {lo}..{hi}");
    }

    /// The point of the whole kernel: `f(x) == f(x + period / frequency)`,
    /// bit for bit, not nearly.
    ///
    /// Coordinates are chosen so the shifted sample is exact in f32 — at
    /// `period == frequency` the shift is one whole unit, and a `k/64` grid
    /// survives both the add and the frequency multiply.
    #[test]
    fn the_value_kernel_tiles_exactly() {
        for (dims, period) in [
            (Dims::D1, [8, 0, 0]),
            (Dims::D2, [8, 8, 0]),
            (Dims::D3, [8, 8, 8]),
        ] {
            let spec = Spec { period, ..plain(dims, Kernel::Value, 8.0, true) };
            for k in 0..64 {
                let uvw = [k as f32 / 64.0, (k as f32 * 3.0) / 64.0, k as f32 / 64.0];
                let here = sample(&spec, 5, uvw);
                for axis in 0..3 {
                    if period[axis] == 0 {
                        continue;
                    }
                    let mut shifted = uvw;
                    // period / frequency = 1.0 unit of sample space.
                    shifted[axis] += 1.0;
                    assert_eq!(
                        here,
                        sample(&spec, 5, shifted),
                        "{dims:?} axis {axis} at {uvw:?}"
                    );
                }
            }
        }
    }

    /// An unbounded axis must *not* repeat, or `period: 0` is silently
    /// doing something.
    #[test]
    fn an_unbounded_axis_does_not_repeat() {
        // Dyadic coordinates, so `+ 1.0` is exact in f32 and any difference
        // is the kernel's rather than the literal's.
        let spec = Spec { period: [8, 0, 0], ..plain(Dims::D2, Kernel::Value, 8.0, true) };
        let here = sample(&spec, 5, [0.3125, 0.3125, 0.0]);
        assert_ne!(here, sample(&spec, 5, [0.3125, 1.3125, 0.0]));
        assert_eq!(here, sample(&spec, 5, [1.3125, 0.3125, 0.0]));
    }

    /// A tiling field stays tiling as octaves are added: the period scales
    /// with the frequency, so every octave wraps on the same boundary.
    #[test]
    fn a_fractal_of_a_tiling_field_still_tiles() {
        let spec = Spec {
            period: [4, 4, 0],
            fractal: Fractal {
                octaves: 4,
                lacunarity: 2.0,
                gain: 0.5,
                mode: FractalMode::Ridged,
                normalize: true,
            },
            ..plain(Dims::D2, Kernel::Value, 4.0, false)
        };
        for k in 0..32 {
            let uvw = [k as f32 / 32.0, (k as f32 * 5.0) / 32.0, 0.0];
            let shifted = [uvw[0] + 1.0, uvw[1], 0.0];
            assert_eq!(sample(&spec, 2, uvw), sample(&spec, 2, shifted), "at {uvw:?}");
        }
    }

    /// One `Standard` octave is the bare kernel — the compatibility claim
    /// that lets `Fractal::default()` be the value an old file loads as.
    #[test]
    fn one_standard_octave_is_the_bare_kernel() {
        for kernel in [Kernel::Simplex, Kernel::Value] {
            let spec = plain(Dims::D2, kernel, 6.0, true);
            for k in 0..20 {
                let uvw = [k as f32 * 0.05, 0.41, 0.0];
                let bare = octave(Dims::D2, kernel, 9, 6.0, [0; 3], uvw);
                assert_eq!(sample(&spec, 9, uvw), bare, "{kernel:?} at {uvw:?}");
            }
        }
    }

    /// Normalized fbm stays inside the range its single octave had. An
    /// un-normalized one is allowed out, and a graph that wants the extra
    /// headroom asks for it.
    #[test]
    fn normalized_octaves_stay_in_range() {
        for mode in [FractalMode::Standard, FractalMode::Turbulence, FractalMode::Ridged] {
            let spec = Spec {
                fractal: Fractal {
                    octaves: 6,
                    lacunarity: 2.0,
                    gain: 0.5,
                    mode,
                    normalize: true,
                },
                ..plain(Dims::D2, Kernel::Value, 3.0, true)
            };
            for k in 0..50 {
                let uvw = [k as f32 * 0.02, k as f32 * 0.013, 0.0];
                let n = sample(&spec, 4, uvw);
                assert!((-1.0..=1.0).contains(&n), "{mode:?} gave {n} at {uvw:?}");
            }
        }
    }

    /// Octaves past the first have to actually contribute, or `octaves` is
    /// an expensive no-op.
    #[test]
    fn more_octaves_change_the_field() {
        let one = plain(Dims::D2, Kernel::Value, 3.0, true);
        let five = Spec { fractal: Fractal { octaves: 5, ..one.fractal }, ..one };
        assert_ne!(sample(&one, 1, [0.37, 0.61, 0.0]), sample(&five, 1, [0.37, 0.61, 0.0]));
    }
}
