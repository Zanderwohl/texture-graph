//! Noise kernels: the specification both backends implement.
//!
//! `crates/gpu/src/shaders/noise.wgsl` mirrors this file function for
//! function. Change both in the same commit, or GPU and CPU bakes differ.
//!
//! - [`Kernel::Simplex`]: Stefan Gustavson's textureless simplex, all `f32`
//!   polynomial arithmetic. Cannot tile. An `f64` kernel or a different
//!   permutation table would visibly differ from the GPU.
//! - [`Kernel::Value`]: trilinear value noise on an integer lattice, whose
//!   cell index can wrap at a per-axis period so bakes tile.
//!
//! [`sample`] runs either through the [`Fractal`] octave loop.
//!
//! # Seeding
//!
//! Simplex has no permutation table to reseed, so the seed is hashed to an
//! offset added to the sample position. The value kernel puts the seed into
//! the lattice hash instead: a fractional offset would move the lattice off
//! the integers and break tiling.
//!
//! # Op-order contract
//!
//! Only `+ - *`, comparisons, `floor`, `abs`, `min`/`max` and float remainder
//! appear here, so rounding cannot differ between backends. `fract` is
//! `x - floor(x)` because Rust's `f32::fract` truncates toward zero and
//! WGSL's does not.
//!
//! The value kernel's hash is all `u32`, so lattice corners match bit for
//! bit; only the trilinear blend is float.
//!
//! Float literals use the same decimal strings as the shader, so both round
//! the same number to `f32`. `clippy::excessive_precision` is off because it
//! would truncate them.
//!
//! # Parity
//!
//! `cpu_and_gpu_noise_agree` in `texture-graph-gpu` holds the backends to
//! within one sRGB step across D1/D2/D3 and both ranges. Bit-exactness
//! would need a float read-back and explicitly fused multiply-adds, since
//! shader compilers contract `a*b + c` on their own. The ±1 also covers the
//! two Oklch→sRGB implementations.

#![allow(clippy::excessive_precision)]

/// Mirrors [`crate::NoiseDims`].
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Dims {
    D1,
    D2,
    D3,
}

/// Mirrors [`crate::NoiseKernel`].
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Kernel {
    Simplex,
    Value,
}

/// Mirrors [`crate::FractalMode`].
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum FractalMode {
    Standard,
    Turbulence,
    Ridged,
}

/// Mirrors [`crate::Fractal`].
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Fractal {
    pub octaves: u32,
    pub lacunarity: f32,
    pub gain: f32,
    pub mode: FractalMode,
    pub normalize: bool,
}

/// Upper bound on [`Fractal::octaves`], so a mistyped count cannot cost
/// thousands of samples a pixel.
pub const MAX_OCTAVES: u32 = 8;

/// A noise field, apart from sample position and seed.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Spec {
    pub dims: Dims,
    pub kernel: Kernel,
    /// Cycles per unit of sample space, at the first octave.
    pub frequency: f32,
    /// Lattice period in cells per axis; `0` is unbounded. [`Kernel::Value`]
    /// only.
    pub period: [u32; 3],
    pub fractal: Fractal,
    /// `true` = `[-1, 1]`, `false` = `[0, 1]`.
    pub signed: bool,
}

fn permute(x: f32) -> f32 {
    (((x * 34.0) + 1.0) * x) % 289.0
}

fn taylor_inv_sqrt(r: f32) -> f32 {
    1.79284291400159 - 0.85373472095314 * r
}

/// WGSL `fract`. Differs from Rust's `f32::fract` for negative `x`.
fn fract(x: f32) -> f32 {
    x - x.floor()
}

/// WGSL `step`.
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

/// 3D simplex noise, roughly `[-1, 1]`.
///
/// The index loops mirror the shader's `vec4` lanes, so they stay loops
/// rather than zips to keep the two easy to compare.
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

/// Integer-only, so both backends produce the same bits for a corner and
/// a periodic field matches itself exactly across a tile boundary.
///
/// Mixing constants are the consumer's plume hash (section 1 of
/// `documentation/game-consumer-features.md`), then an xor-shift-multiply
/// avalanche.
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

/// Wrap into `[0, period)`; `period == 0` passes through. Not `rem_euclid`,
/// so it matches the WGSL, where `%` truncates toward zero.
fn wrap_cell(i: i32, period: u32) -> i32 {
    if period == 0 {
        return i;
    }
    let p = period as i32;
    ((i % p) + p) % p
}

/// C¹ across cell walls; linear weights would show the lattice as creases.
fn smooth_weight(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Roughly `[-1, 1]`.
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

/// Wang hash. Not cryptographic; it only has to make different seeds look
/// distinct.
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

/// The XOR constants are arbitrary and must match the shader's.
fn seed_offsets(seed: u32) -> [f32; 3] {
    [
        hash_to_float01(wang_hash(seed ^ 0xA1B2_C3D4)) * 256.0,
        hash_to_float01(wang_hash(seed ^ 0x51F0_E7A9)) * 256.0,
        hash_to_float01(wang_hash(seed ^ 0x0BAD_C0DE)) * 256.0,
    ]
}

/// Large and odd so octave seeds of one channel never collide with the
/// next channel's, which is only one seed apart. A collision would visibly
/// correlate L with C.
const OCTAVE_SEED_STRIDE: u32 = 0x9E37_79B9;

/// Roughly `[-1, 1]`.
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
                // As on the GPU, only x is scaled by frequency.
                Dims::D1 => snoise2([(uvw[0] + o[0]) * f, o[1]]),
                Dims::D2 => snoise2([(uvw[0] + o[0]) * f, (uvw[1] + o[1]) * f]),
                Dims::D3 => snoise3([
                    (uvw[0] + o[0]) * f,
                    (uvw[1] + o[1]) * f,
                    (uvw[2] + o[2]) * f,
                ]),
            }
        }
        // Unused axes are 0, on a lattice plane rather than inside a cell.
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

/// On the signed sample `r`, where the documented `1 - |2n - 1|` over
/// unsigned `n` is `1 - |r|`.
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

/// Scales the period with the frequency so every octave tiles. Exact only
/// for integral lacunarity.
///
/// `floor(x + 0.5)`, not `round`: WGSL's `round` breaks ties to even and
/// Rust's away from zero.
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

/// One sample in `[-1, 1]` if [`Spec::signed`], else `[0, 1]`. `seed` is
/// the full `ctx.seed + seed_offset + channel`. Mirrors `sample_noise` in
/// noise.wgsl.
///
/// `Turbulence` and `Ridged` sum into `[0, 1]`, so they are stretched to
/// `[-1, 1]` first; every mode then maps to unsigned the same way.
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

    /// Parity checks cannot catch a constant or out-of-range field, since
    /// both backends would agree on it.
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

    /// Rust's `fract` would break here; a transform can make any coordinate
    /// negative.
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

    /// Checked for both kernels, since they seed differently.
    #[test]
    fn the_seed_moves_the_field() {
        for kernel in [Kernel::Simplex, Kernel::Value] {
            let spec = plain(Dims::D2, kernel, 4.0, true);
            let at = |seed| sample(&spec, seed, [0.3, 0.7, 0.5]);
            assert_ne!(at(0), at(1), "{kernel:?}");
            assert_ne!(at(1), at(2), "{kernel:?}");
        }
    }

    #[test]
    fn unsigned_is_the_signed_field_remapped() {
        for k in 0..10 {
            let uvw = [k as f32 * 0.1, 0.4, 0.6];
            let signed = sample(&plain(Dims::D2, Kernel::Simplex, 3.0, true), 7, uvw);
            let unsigned = sample(&plain(Dims::D2, Kernel::Simplex, 3.0, false), 7, uvw);
            assert_eq!(unsigned, signed * 0.5 + 0.5);
        }
    }

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

    /// `f(x) == f(x + period / frequency)` bit for bit. The `k/64` grid
    /// and a shift of 1.0 keep every coordinate exact in f32.
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

    #[test]
    fn an_unbounded_axis_does_not_repeat() {
        // Dyadic coordinates, so `+ 1.0` is exact in f32.
        let spec = Spec { period: [8, 0, 0], ..plain(Dims::D2, Kernel::Value, 8.0, true) };
        let here = sample(&spec, 5, [0.3125, 0.3125, 0.0]);
        assert_ne!(here, sample(&spec, 5, [0.3125, 1.3125, 0.0]));
        assert_eq!(here, sample(&spec, 5, [1.3125, 0.3125, 0.0]));
    }

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

    /// Files without fractal fields load as `Fractal::default()`, which
    /// relies on this.
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

    #[test]
    fn more_octaves_change_the_field() {
        let one = plain(Dims::D2, Kernel::Value, 3.0, true);
        let five = Spec { fractal: Fractal { octaves: 5, ..one.fractal }, ..one };
        assert_ne!(sample(&one, 1, [0.37, 0.61, 0.0]), sample(&five, 1, [0.37, 0.61, 0.0]));
    }
}
