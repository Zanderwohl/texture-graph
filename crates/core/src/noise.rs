//! Simplex noise — the *specification* both backends implement.
//!
//! The WGSL twin is `crates/gpu/src/shaders/noise.wgsl`, function for
//! function. Any change here is a spec change and must land on both sides in
//! the same commit, or a graph will look one way in a GPU bake and another in
//! the CPU fallback.
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
//! # Op-order contract
//!
//! Only `+ - *`, comparisons, `floor`, `abs`, `min`/`max` and float remainder
//! appear here — nothing transcendental, nothing whose rounding could differ
//! between backends. `fract` is written out as `x - floor(x)` because Rust's
//! `f32::fract` truncates toward zero and WGSL's does not, which would differ
//! on every negative coordinate.
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

/// One noise sample, in the same `[-1, 1]` (signed) or `[0, 1]` (unsigned)
/// convention the shader uses.
///
/// `seed` is already `ctx.seed + seed_offset + channel`; `signed` selects the
/// range. The twin is `sample_noise` in noise.wgsl.
pub fn sample(dims: Dims, seed: u32, frequency: f32, uvw: [f32; 3], signed: bool) -> f32 {
    let o = seed_offsets(seed);
    let f = frequency;
    let raw = match dims {
        // D1 samples the 2D kernel with y pinned — and, as on the GPU, only
        // the x axis is scaled by frequency.
        Dims::D1 => snoise2([(uvw[0] + o[0]) * f, o[1]]),
        Dims::D2 => snoise2([(uvw[0] + o[0]) * f, (uvw[1] + o[1]) * f]),
        Dims::D3 => snoise3([
            (uvw[0] + o[0]) * f,
            (uvw[1] + o[1]) * f,
            (uvw[2] + o[2]) * f,
        ]),
    };
    if signed { raw } else { raw * 0.5 + 0.5 }
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

    /// Different seeds have to give different fields, or `seed_offset` is
    /// decorative.
    #[test]
    fn the_seed_moves_the_field() {
        let at = |seed| sample(Dims::D2, seed, 4.0, [0.3, 0.7, 0.5], true);
        assert_ne!(at(0), at(1));
        assert_ne!(at(1), at(2));
    }

    /// Unsigned is the signed field mapped onto `[0, 1]`, which is what the
    /// shader's `raw * 0.5 + 0.5` says.
    #[test]
    fn unsigned_is_the_signed_field_remapped() {
        for k in 0..10 {
            let uvw = [k as f32 * 0.1, 0.4, 0.6];
            let signed = sample(Dims::D2, 7, 3.0, uvw, true);
            let unsigned = sample(Dims::D2, 7, 3.0, uvw, false);
            assert_eq!(unsigned, signed * 0.5 + 0.5);
        }
    }
}
