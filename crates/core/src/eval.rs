use std::collections::HashMap;

use crate::color::{Color, blend, normal_to_color, scalar_of};
use crate::graph::{Graph, Layer};
use crate::id::LayerId;
use crate::kind::{
    Axis, BlendMode, ColorInput, ColorRamp, CoordMode, Criterion, EXTEND_LIMIT, EdgeMode,
    FractalMode, HeightToNormal, LayerKind, Map, MinMax, MinMaxMode, Mix, Noise, NoiseDims,
    NoiseKernel, NoiseOutput, NoiseRange, RadialDim, ScalarInput, Transform, Wave,
    WaveShape,
};

/// A sample point in the graph's canonical unit cube. Consumers of the
/// core map their own coordinates (mesh UVs, world position, planar grid)
/// into this space.
#[derive(Copy, Clone, Debug)]
pub struct Sample {
    pub u: f32,
    pub v: f32,
    pub w: f32,
}

/// The `w` a *flat* bake samples a 3D field at — the middle of the unit
/// cube, not its floor.
///
/// Both backends must agree, or a 3D graph is two different slices of the
/// same volume depending on where it was rendered.
pub const FLAT_W: f32 = 0.5;

impl Sample {
    pub const fn new(u: f32, v: f32, w: f32) -> Self {
        Self { u, v, w }
    }
    /// A sample at `w = 0`. For a *flat bake* of a graph that may contain
    /// 3D fields, use [`Sample::flat`] instead — see [`FLAT_W`].
    pub const fn uv(u: f32, v: f32) -> Self {
        Self { u, v, w: 0.0 }
    }
    /// The sample a flat bake takes at `(u, v)`, on the [`FLAT_W`] slice.
    pub const fn flat(u: f32, v: f32) -> Self {
        Self { u, v, w: FLAT_W }
    }
}

/// Ambient parameters that stay constant across a whole bake — the global
/// noise seed, and finite-difference step used by `HeightToNormal`.
#[derive(Copy, Clone, Debug)]
pub struct EvalCtx {
    pub seed: u32,
    pub normal_epsilon: f32,
}

impl Default for EvalCtx {
    fn default() -> Self {
        Self { seed: 0xC0DED00D, normal_epsilon: 1.0 / 512.0 }
    }
}

/// Material sample from the graph's Output at `s`.
#[derive(Copy, Clone, Debug)]
pub struct Material {
    pub color: Color,
    pub roughness: f32,
    pub metallic: f32,
    pub normal: Color,
}

/// Evaluate the graph's `Output` at `s` and return every material channel.
pub fn evaluate_material(g: &Graph, s: Sample, ctx: &EvalCtx) -> Material {
    let by_id: HashMap<LayerId, &Layer> = g.layers.iter().map(|l| (l.id, l)).collect();
    let out = &g.output;
    let color = eval_opt(out.color, s, &by_id, ctx);
    let roughness = eval_scalar(&out.roughness, s, &by_id, ctx);
    let metallic = eval_scalar(&out.metallic, s, &by_id, ctx);
    let normal = match out.normal {
        Some(id) => eval_layer(id, s, &by_id, ctx),
        None => normal_to_color([0.0, 0.0, 1.0]),
    };
    Material { color, roughness, metallic, normal }
}

/// Evaluate any single layer's color at `s` — useful for per-layer previews.
pub fn evaluate(g: &Graph, id: LayerId, s: Sample, ctx: &EvalCtx) -> Color {
    let by_id: HashMap<LayerId, &Layer> = g.layers.iter().map(|l| (l.id, l)).collect();
    eval_layer(id, s, &by_id, ctx)
}

// ---- Core dispatch ------------------------------------------------------

fn eval_layer(id: LayerId, s: Sample, by_id: &HashMap<LayerId, &Layer>, ctx: &EvalCtx) -> Color {
    let Some(layer) = by_id.get(&id) else {
        return Color::new(0.0, 0.0, 0.0, 1.0);
    };
    match &layer.kind {
        LayerKind::Color(c) => *c,
        LayerKind::Noise(n) => eval_noise(n, s, ctx),
        LayerKind::ColorRamp(r) => eval_ramp(r, s, by_id, ctx),
        LayerKind::Transform(t) => eval_transform(t, s, by_id, ctx),
        LayerKind::Mix(m) => eval_mix(m, s, by_id, ctx),
        LayerKind::Map(m) => eval_map(m, s, by_id, ctx),
        LayerKind::MinMax(mm) => eval_min_max(mm, s, by_id, ctx),
        LayerKind::HeightToNormal(h) => eval_h2n(h, s, by_id, ctx),
        LayerKind::Wave(w) => eval_wave(w, s, by_id, ctx),
    }
}

/// The "missing texture" grid shown for unconnected (`None`) layer
/// inputs: a magenta/black checkerboard, 16 cells per unit in u, v, AND w
/// so it stays a solid 3D checker in volume bakes. Mirrored by
/// `missing.wgsl` on the GPU — keep the cell count and colors in sync.
/// Magenta constants are Oklch of sRGB (1, 0, 1).
pub fn missing_texture(s: Sample) -> Color {
    const CELLS: f32 = 16.0;
    let cell = |x: f32| (x * CELLS).floor() as i64;
    let parity = (cell(s.u) + cell(s.v) + cell(s.w)).rem_euclid(2);
    if parity == 0 {
        Color::new(0.7017, 0.3223, 328.36, 1.0)
    } else {
        Color::new(0.0, 0.0, 0.0, 1.0)
    }
}

/// Evaluate an optional layer input; `None` samples the missing grid.
fn eval_opt(
    id: Option<LayerId>,
    s: Sample,
    by_id: &HashMap<LayerId, &Layer>,
    ctx: &EvalCtx,
) -> Color {
    match id {
        Some(id) => eval_layer(id, s, by_id, ctx),
        None => missing_texture(s),
    }
}

fn eval_color_input(ci: &ColorInput, s: Sample, by_id: &HashMap<LayerId, &Layer>, ctx: &EvalCtx) -> Color {
    match ci {
        ColorInput::Const(c) => *c,
        ColorInput::Layer(id) => eval_layer(*id, s, by_id, ctx),
    }
}

fn eval_scalar(si: &ScalarInput, s: Sample, by_id: &HashMap<LayerId, &Layer>, ctx: &EvalCtx) -> f32 {
    match si {
        ScalarInput::Const(v) => *v,
        ScalarInput::Layer(id) => scalar_of(eval_layer(*id, s, by_id, ctx)),
    }
}

// ---- Node implementations ----------------------------------------------

/// Translate the serialized node into the kernel's own vocabulary. The two
/// sets of enums stay separate so `crate::noise` — the spec the shader is
/// transcribed from — does not depend on the file format.
fn noise_spec(n: &Noise) -> crate::noise::Spec {
    crate::noise::Spec {
        dims: match n.dims {
            NoiseDims::D1 => crate::noise::Dims::D1,
            NoiseDims::D2 => crate::noise::Dims::D2,
            NoiseDims::D3 => crate::noise::Dims::D3,
        },
        kernel: match n.kernel {
            NoiseKernel::Simplex => crate::noise::Kernel::Simplex,
            NoiseKernel::Value => crate::noise::Kernel::Value,
        },
        frequency: n.frequency,
        period: n.period,
        fractal: crate::noise::Fractal {
            octaves: n.fractal.octaves,
            lacunarity: n.fractal.lacunarity,
            gain: n.fractal.gain,
            mode: match n.fractal.mode {
                FractalMode::Standard => crate::noise::FractalMode::Standard,
                FractalMode::Turbulence => crate::noise::FractalMode::Turbulence,
                FractalMode::Ridged => crate::noise::FractalMode::Ridged,
            },
            normalize: n.fractal.normalize,
        },
        signed: matches!(n.range, NoiseRange::Signed),
    }
}

fn eval_noise(n: &Noise, s: Sample, ctx: &EvalCtx) -> Color {
    // Straight through to the shared kernel; `crate::noise` has the why.
    let spec = noise_spec(n);
    let sample = |off: u32| -> f32 {
        crate::noise::sample(
            &spec,
            ctx.seed.wrapping_add(n.seed_offset).wrapping_add(off),
            [s.u, s.v, s.w],
        )
    };

    match n.output {
        NoiseOutput::Grayscale => {
            let v = sample(0);
            Color::new(v, 0.0, 0.0, 1.0)
        }
        NoiseOutput::Color => {
            // Three independent noises → L, C, hue directly in Oklch.
            // Chroma is scaled to a comfortable in-gamut band; hue is a
            // full turn for unsigned, half-turn magnitude for signed.
            const CHROMA_SCALE: f32 = 0.15;
            let nl = sample(0);
            let nc = sample(1);
            let nh = sample(2);
            let hue = match n.range {
                NoiseRange::Signed => nh * 180.0,
                NoiseRange::Unsigned => nh * 360.0,
            };
            Color::new(nl, nc * CHROMA_SCALE, hue, 1.0)
        }
    }
}

fn eval_ramp(r: &ColorRamp, s: Sample, by_id: &HashMap<LayerId, &Layer>, ctx: &EvalCtx) -> Color {
    // Domain: `s.u`. Off either end the ramp holds the outermost stop's
    // color (t clamped to [0, 1]) — a first stop at 0.3 paints [0, 0.3]
    // with its own color, and likewise past the last stop.
    debug_assert!(r.stops.len() >= 2, "ramp validation should reject <2 stops");
    // Find the segment containing s.u.
    let mut lo_idx = 0usize;
    let mut hi_idx = r.stops.len() - 1;
    for i in 0..r.stops.len() - 1 {
        let a = r.stops[i].t;
        let b = r.stops[i + 1].t;
        if s.u >= a && s.u <= b {
            lo_idx = i;
            hi_idx = i + 1;
            break;
        }
        if s.u < a && i == 0 {
            lo_idx = 0;
            hi_idx = 1;
            break;
        }
        if s.u > b && i == r.stops.len() - 2 {
            lo_idx = r.stops.len() - 2;
            hi_idx = r.stops.len() - 1;
            break;
        }
    }
    let a = &r.stops[lo_idx];
    let b = &r.stops[hi_idx];
    let span = b.t - a.t;
    let t = if span.abs() < f32::EPSILON {
        0.0
    } else {
        ((s.u - a.t) / span).clamp(0.0, 1.0)
    };
    let ca = eval_color_input(&a.color, s, by_id, ctx);
    let cb = eval_color_input(&b.color, s, by_id, ctx);
    blend(ca, cb, t, r.space)
}

/// Transform, then apply the edge policy to the resulting U/V:
/// `Clamp` pins them to the [0, 1] square (matching a baked texture's
/// edge clamp); `Extend` samples the source at the true coordinates.
/// Affine mappings extend without limit (the GPU bakes sources over the
/// exact requested region); radial mappings render the missing grid
/// beyond the [`EXTEND_LIMIT`] box, matching the GPU's conservative cap.
fn eval_transform(
    t: &Transform,
    s: Sample,
    by_id: &HashMap<LayerId, &Layer>,
    ctx: &EvalCtx,
) -> Color {
    let ts = apply_transform(t, s);
    match t.edge_mode {
        EdgeMode::Clamp => {
            let clamped = Sample::new(ts.u.clamp(0.0, 1.0), ts.v.clamp(0.0, 1.0), ts.w);
            eval_opt(t.source, clamped, by_id, ctx)
        }
        EdgeMode::Extend => {
            let lo = 0.5 - EXTEND_LIMIT;
            let hi = 0.5 + EXTEND_LIMIT;
            let capped = matches!(t.coord_mode, CoordMode::Radial { .. });
            if capped && (ts.u < lo || ts.u > hi || ts.v < lo || ts.v > hi) {
                missing_texture(ts)
            } else {
                eval_opt(t.source, ts, by_id, ctx)
            }
        }
    }
}

fn apply_transform(t: &Transform, s: Sample) -> Sample {
    // 1. Recenter.
    let mut u = s.u - t.offset[0];
    let mut v = s.v - t.offset[1];
    let w = s.w - t.offset[2];
    // 2. Rotate in UV plane.
    if t.rotate_uv != 0.0 {
        let (sin, cos) = t.rotate_uv.sin_cos();
        let ru = u * cos - v * sin;
        let rv = u * sin + v * cos;
        u = ru;
        v = rv;
    }
    // 3. Per-axis scale.
    let u = u * t.scale[0];
    let v = v * t.scale[1];
    let w = w * t.scale[2];

    match t.coord_mode {
        CoordMode::Passthrough => Sample::new(u, v, w),
        CoordMode::Permute(axes) => {
            let coord = |a: Axis| match a {
                Axis::U => u,
                Axis::V => v,
                Axis::W => w,
            };
            Sample::new(coord(axes[0]), coord(axes[1]), coord(axes[2]))
        }
        CoordMode::Radial { dim, into } => {
            let r = match dim {
                RadialDim::D2 => (u * u + v * v).sqrt(),
                RadialDim::D3 => (u * u + v * v + w * w).sqrt(),
            };
            match into {
                Axis::U => Sample::new(r, 0.0, 0.0),
                Axis::V => Sample::new(0.0, r, 0.0),
                Axis::W => Sample::new(0.0, 0.0, r),
            }
        }
    }
}

fn eval_mix(m: &Mix, s: Sample, by_id: &HashMap<LayerId, &Layer>, ctx: &EvalCtx) -> Color {
    use palette::{IntoColor, LinSrgb, Oklab, Oklch, WithAlpha};
    let a = eval_opt(m.a, s, by_id, ctx);
    let b = eval_opt(m.b, s, by_id, ctx);
    match m.mode {
        // Add/Sub compose in Oklab where chroma is a Cartesian (a, b) vector.
        // Grayscale-noise + Add is the fractal-noise use case; L just sums.
        BlendMode::Add | BlendMode::Subtract => {
            let sign = if matches!(m.mode, BlendMode::Add) { 1.0 } else { -1.0 };
            let al: Oklab = a.color.into_color();
            let bl: Oklab = b.color.into_color();
            let combined = Oklab::new(al.l + sign * bl.l, al.a + sign * bl.a, al.b + sign * bl.b);
            let back: Oklch = combined.into_color();
            back.with_alpha(a.alpha)
        }
        // Multiply as optical darkening: componentwise in linear sRGB.
        BlendMode::Multiply => {
            let la: LinSrgb = a.color.into_color();
            let lb: LinSrgb = b.color.into_color();
            let prod = LinSrgb::new(la.red * lb.red, la.green * lb.green, la.blue * lb.blue);
            let back: Oklch = prod.into_color();
            back.with_alpha(a.alpha * b.alpha)
        }
        BlendMode::Blend => {
            let t = eval_scalar(&m.factor, s, by_id, ctx);
            blend(a, b, t, m.space)
        }
    }
}

fn eval_map(m: &Map, s: Sample, by_id: &HashMap<LayerId, &Layer>, ctx: &EvalCtx) -> Color {
    let value = eval_opt(m.value, s, by_id, ctx);
    let t = scalar_of(value);
    // Look up palette at (t, 0, 0).
    eval_opt(m.palette, Sample::new(t, 0.0, 0.0), by_id, ctx)
}

fn eval_min_max(m: &MinMax, s: Sample, by_id: &HashMap<LayerId, &Layer>, ctx: &EvalCtx) -> Color {
    let a = eval_opt(m.a, s, by_id, ctx);
    let b = eval_opt(m.b, s, by_id, ctx);
    let va = criterion_of(a, m.criterion);
    let vb = criterion_of(b, m.criterion);
    let a_wins = match m.mode {
        MinMaxMode::Min => va <= vb,
        MinMaxMode::Max => va >= vb,
    };
    if a_wins { a } else { b }
}

/// Read the chosen criterion off `c`. Whole pixel — including alpha and
/// hue — flows through the winner, so this only picks the number for the
/// compare.
fn criterion_of(c: Color, crit: Criterion) -> f32 {
    use palette::{Hsv, IntoColor, Srgb};
    match crit {
        Criterion::Alpha => return c.alpha,
        Criterion::Chroma => return c.chroma,
        _ => {}
    }
    // Everything else needs a trip through gamma-encoded sRGB.
    let srgb: Srgb = c.color.into_color();
    match crit {
        Criterion::Red => srgb.red,
        Criterion::Green => srgb.green,
        Criterion::Blue => srgb.blue,
        Criterion::Luma => 0.2126 * srgb.red + 0.7152 * srgb.green + 0.0722 * srgb.blue,
        Criterion::Saturation => {
            let hsv: Hsv = srgb.into_color();
            hsv.saturation
        }
        Criterion::Value => {
            let hsv: Hsv = srgb.into_color();
            hsv.value
        }
        // Handled above.
        Criterion::Alpha | Criterion::Chroma => unreachable!(),
    }
}

/// One cycle of `shape` at phase `t`, in `[-1, 1]`.
///
/// `t` is reduced to `[0, 1)` first — `fract` written as `x - floor(x)`,
/// because Rust's `f32::fract` truncates toward zero and WGSL's does not,
/// which would differ on every negative input. The same reason
/// `crate::noise` writes it out.
pub fn wave_cycle(t: f32, shape: WaveShape) -> f32 {
    let frac = |x: f32| x - x.floor();
    let t = frac(t);
    match shape {
        WaveShape::Sine => (std::f32::consts::TAU * t).sin(),
        // Phase-aligned with sine: 0 at t = 0, +1 at t = 0.25.
        WaveShape::Triangle => 1.0 - 4.0 * (frac(t + 0.25) - 0.5).abs(),
        WaveShape::Square => {
            if t < 0.5 {
                1.0
            } else {
                -1.0
            }
        }
        WaveShape::Sawtooth => 2.0 * t - 1.0,
    }
}

fn eval_wave(w: &Wave, s: Sample, by_id: &HashMap<LayerId, &Layer>, ctx: &EvalCtx) -> Color {
    let x = eval_scalar(&w.input, s, by_id, ctx);
    let v = wave_cycle(x * w.frequency + w.phase, w.shape);
    let l = match w.range {
        NoiseRange::Signed => v,
        NoiseRange::Unsigned => v * 0.5 + 0.5,
    };
    Color::new(l, 0.0, 0.0, 1.0)
}

fn eval_h2n(h: &HeightToNormal, s: Sample, by_id: &HashMap<LayerId, &Layer>, ctx: &EvalCtx) -> Color {
    let eps = ctx.normal_epsilon.max(f32::EPSILON);
    let sample_l = |ds: Sample| scalar_of(eval_opt(h.source, ds, by_id, ctx));
    let l_px = sample_l(Sample::new(s.u + eps, s.v, s.w));
    let l_nx = sample_l(Sample::new(s.u - eps, s.v, s.w));
    let l_py = sample_l(Sample::new(s.u, s.v + eps, s.w));
    let l_ny = sample_l(Sample::new(s.u, s.v - eps, s.w));
    // Central-difference gradient of the height field.
    let dhdx = (l_px - l_nx) / (2.0 * eps);
    let dhdy = (l_py - l_ny) / (2.0 * eps);
    // Normal to a heightfield z = h(x,y) is (-dh/dx, -dh/dy, 1), scaled by strength.
    let mut n = [-dhdx * h.strength, -dhdy * h.strength, 1.0];
    let mag = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(f32::EPSILON);
    n[0] /= mag;
    n[1] /= mag;
    n[2] /= mag;
    normal_to_color(n)
}

#[cfg(test)]
mod wave_tests {
    use super::*;
    use crate::kind::{Wave, WaveShape};

    /// Every shape is described over the phase within a cycle, so it has
    /// to actually be periodic in it.
    #[test]
    fn every_shape_repeats_once_a_cycle() {
        for shape in [
            WaveShape::Sine,
            WaveShape::Triangle,
            WaveShape::Square,
            WaveShape::Sawtooth,
        ] {
            for k in 0..16 {
                let t = k as f32 / 16.0;
                let here = wave_cycle(t, shape);
                for turns in [-3.0, -1.0, 1.0, 4.0] {
                    let there = wave_cycle(t + turns, shape);
                    assert!(
                        (here - there).abs() < 1e-5,
                        "{shape:?} at {t} vs {} turns away: {here} vs {there}",
                        turns
                    );
                }
            }
        }
    }

    /// Sine, triangle and square share a phase: zero-crossings and sign in
    /// the same places. A sawtooth deliberately does not — it ramps.
    #[test]
    fn shapes_stay_inside_range_and_share_sines_phase() {
        for k in 0..64 {
            let t = k as f32 / 64.0;
            for shape in [
                WaveShape::Sine,
                WaveShape::Triangle,
                WaveShape::Square,
                WaveShape::Sawtooth,
            ] {
                let v = wave_cycle(t, shape);
                assert!((-1.0..=1.0).contains(&v), "{shape:?} at {t} gave {v}");
            }
            // Away from the crossings at 0 and 0.5, the three agree on sign.
            if (t - 0.0).abs() > 1e-3 && (t - 0.5).abs() > 1e-3 {
                let sine = wave_cycle(t, WaveShape::Sine);
                for shape in [WaveShape::Triangle, WaveShape::Square] {
                    let v = wave_cycle(t, shape);
                    assert_eq!(
                        sine > 0.0,
                        v > 0.0,
                        "{shape:?} disagrees with sine on sign at {t}"
                    );
                }
            }
        }
        // The landmarks, exactly.
        assert!((wave_cycle(0.25, WaveShape::Triangle) - 1.0).abs() < 1e-6);
        assert!((wave_cycle(0.75, WaveShape::Triangle) + 1.0).abs() < 1e-6);
        assert_eq!(wave_cycle(0.0, WaveShape::Sawtooth), -1.0);
        assert!((wave_cycle(0.999, WaveShape::Sawtooth) - 1.0).abs() < 0.01);
    }

    /// `frequency` counts cycles, `phase` shifts by cycles, and `range`
    /// maps the result the same way grayscale noise does.
    #[test]
    fn frequency_and_phase_are_counted_in_cycles() {
        let mut g = Graph::new();
        let id = g.output.color.unwrap();
        let wave = |frequency, phase, range| Wave {
            input: ScalarInput::Const(0.125),
            shape: WaveShape::Sine,
            frequency,
            phase,
            range,
        };
        let at = |g: &Graph, id| scalar_of(evaluate(g, id, Sample::uv(0.0, 0.0), &EvalCtx::default()));

        // Const input 0.125 at frequency 2 is phase 0.25 — sine's peak.
        g.set_kind(id, LayerKind::Wave(wave(2.0, 0.0, NoiseRange::Signed))).unwrap();
        assert!((at(&g, id) - 1.0).abs() < 1e-5);
        // A full turn of phase changes nothing.
        g.set_kind(id, LayerKind::Wave(wave(2.0, 1.0, NoiseRange::Signed))).unwrap();
        assert!((at(&g, id) - 1.0).abs() < 1e-5);
        // Half a turn inverts it.
        g.set_kind(id, LayerKind::Wave(wave(2.0, 0.5, NoiseRange::Signed))).unwrap();
        assert!((at(&g, id) + 1.0).abs() < 1e-5);
        // Unsigned is the signed field on [0, 1].
        g.set_kind(id, LayerKind::Wave(wave(2.0, 0.0, NoiseRange::Unsigned))).unwrap();
        assert!((at(&g, id) - 1.0).abs() < 1e-5);
        g.set_kind(id, LayerKind::Wave(wave(2.0, 0.5, NoiseRange::Unsigned))).unwrap();
        assert!(at(&g, id).abs() < 1e-5);
    }

    /// The wave reads its input as a scalar, so wiring a layer in makes
    /// the output vary with that layer rather than sit flat.
    #[test]
    fn a_layer_input_makes_the_wave_vary() {
        let mut g = Graph::new();
        let ramp = g.output.color.unwrap();
        g.set_kind(
            ramp,
            LayerKind::ColorRamp(crate::kind::ColorRamp {
                stops: vec![
                    crate::kind::ColorStop {
                        t: 0.0,
                        color: ColorInput::Const(Color::new(0.0, 0.0, 0.0, 1.0)),
                    },
                    crate::kind::ColorStop {
                        t: 1.0,
                        color: ColorInput::Const(Color::new(1.0, 0.0, 0.0, 1.0)),
                    },
                ],
                space: crate::color::BlendSpace::Oklch,
            }),
        )
        .unwrap();
        let w = g
            .add_layer(
                "bands",
                LayerKind::Wave(Wave {
                    input: ScalarInput::Layer(ramp),
                    shape: WaveShape::Sine,
                    frequency: 4.0,
                    phase: 0.0,
                    range: NoiseRange::Unsigned,
                }),
            )
            .unwrap();
        let ctx = EvalCtx::default();
        let at = |u| scalar_of(evaluate(&g, w, Sample::uv(u, 0.5), &ctx));
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for k in 0..64 {
            let v = at(k as f32 / 64.0);
            lo = lo.min(v);
            hi = hi.max(v);
        }
        assert!(hi - lo > 0.9, "bands barely vary: {lo}..{hi}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::{BlendSpace, oklcha};
    use crate::kind::{ColorStop, LayerKind};

    #[test]
    fn evaluate_default_material_matches_placeholder_color() {
        let g = Graph::new();
        let m = evaluate_material(&g, Sample::uv(0.5, 0.5), &EvalCtx::default());
        assert!((m.color.l - 0.5).abs() < 1e-6);
        assert_eq!(m.roughness, 0.5);
        assert_eq!(m.metallic, 0.0);
    }

    #[test]
    fn color_ramp_endpoints() {
        let mut g = Graph::new();
        let ramp_id = g
            .add_layer(
                "ramp",
                LayerKind::ColorRamp(ColorRamp {
                    stops: vec![
                        ColorStop { t: 0.0, color: ColorInput::Const(oklcha(0.0, 0.0, 0.0, 1.0)) },
                        ColorStop { t: 1.0, color: ColorInput::Const(oklcha(1.0, 0.0, 0.0, 1.0)) },
                    ],
                    space: BlendSpace::Oklch,
                }),
            )
            .unwrap();
        let ctx = EvalCtx::default();
        let by_id: HashMap<LayerId, &Layer> = g.layers.iter().map(|l| (l.id, l)).collect();
        let start = eval_layer(ramp_id, Sample::uv(0.0, 0.0), &by_id, &ctx);
        let end = eval_layer(ramp_id, Sample::uv(1.0, 0.0), &by_id, &ctx);
        assert!(start.l.abs() < 1e-4);
        assert!((end.l - 1.0).abs() < 1e-4);
    }

    #[test]
    fn cycle_rejected() {
        let mut g = Graph::new();
        let a = g.add_layer("a", LayerKind::Color(oklcha(0.1, 0.0, 0.0, 1.0))).unwrap();
        let b = g
            .add_layer(
                "b",
                LayerKind::Mix(Mix {
                    a: Some(a),
                    b: Some(a),
                    mode: BlendMode::Add,
                    factor: ScalarInput::Const(0.5),
                    space: BlendSpace::Oklch,
                }),
            )
            .unwrap();
        // Rewriting `a` to reference `b` would create a cycle.
        let err = g.set_kind(
            a,
            LayerKind::Mix(Mix {
                a: Some(b),
                b: Some(b),
                mode: BlendMode::Add,
                factor: ScalarInput::Const(0.5),
                space: BlendSpace::Oklch,
            }),
        );
        assert!(matches!(err, Err(crate::graph::GraphError::Cycle(_))));
    }

    /// A u-gradient ramp under a scaling Transform, for edge-mode tests.
    fn ramp_under_transform(edge_mode: EdgeMode, scale_u: f32) -> (Graph, LayerId) {
        let mut g = Graph::new();
        let ramp = g
            .add_layer(
                "ramp",
                LayerKind::ColorRamp(ColorRamp {
                    stops: vec![
                        ColorStop { t: 0.0, color: ColorInput::Const(oklcha(0.0, 0.0, 0.0, 1.0)) },
                        ColorStop { t: 1.0, color: ColorInput::Const(oklcha(1.0, 0.0, 0.0, 1.0)) },
                    ],
                    space: BlendSpace::Oklch,
                }),
            )
            .unwrap();
        let t = g
            .add_layer(
                "xform",
                LayerKind::Transform(Transform {
                    source: Some(ramp),
                    offset: [0.0; 3],
                    rotate_uv: 0.0,
                    scale: [scale_u, 1.0, 1.0],
                    coord_mode: CoordMode::Passthrough,
                    edge_mode,
                }),
            )
            .unwrap();
        (g, t)
    }

    #[test]
    fn transform_clamp_pins_uv_to_unit_square() {
        let (g, t) = ramp_under_transform(EdgeMode::Clamp, 4.0);
        let by_id: HashMap<LayerId, &Layer> = g.layers.iter().map(|l| (l.id, l)).collect();
        let ctx = EvalCtx::default();
        // u=0.5 transforms to 2.0 -> clamps to 1.0 -> ramp's white end.
        let c = eval_layer(t, Sample::uv(0.5, 0.5), &by_id, &ctx);
        assert!((c.l - 1.0).abs() < 1e-4, "expected clamped end color, got L={}", c.l);
    }

    #[test]
    fn transform_extend_samples_beyond_unit_square() {
        let (g, t) = ramp_under_transform(EdgeMode::Extend, 4.0);
        let by_id: HashMap<LayerId, &Layer> = g.layers.iter().map(|l| (l.id, l)).collect();
        let ctx = EvalCtx::default();
        // u=0.5 -> 2.0: the ramp holds its end color past t=1, and extend
        // actually reaches it (not the missing grid).
        let c = eval_layer(t, Sample::uv(0.5, 0.5), &by_id, &ctx);
        assert!((c.l - 1.0).abs() < 1e-4);
        assert!(c.chroma.abs() < 1e-4, "should be the ramp color, not the magenta grid");
    }

    #[test]
    fn transform_extend_affine_is_unbounded() {
        let (g, t) = ramp_under_transform(EdgeMode::Extend, 40.0);
        let by_id: HashMap<LayerId, &Layer> = g.layers.iter().map(|l| (l.id, l)).collect();
        let ctx = EvalCtx::default();
        // u=0.9 -> 36.0, far past EXTEND_LIMIT — an affine extend still
        // samples the source (which holds its end color), never the grid.
        let c = eval_layer(t, Sample::uv(0.9, 0.5), &by_id, &ctx);
        assert!((c.l - 1.0).abs() < 1e-4, "expected held end color, got L={}", c.l);
        assert!(c.chroma.abs() < 1e-4, "should be the ramp color, not the magenta grid");
    }

    #[test]
    fn transform_extend_radial_caps_at_limit() {
        let (mut g, t) = ramp_under_transform(EdgeMode::Extend, 40.0);
        let radial = Transform {
            source: g.layers.iter().find(|l| l.name == "ramp").map(|l| l.id),
            offset: [0.0; 3],
            rotate_uv: 0.0,
            scale: [40.0, 1.0, 1.0],
            coord_mode: CoordMode::Radial { dim: crate::RadialDim::D2, into: Axis::U },
            edge_mode: EdgeMode::Extend,
        };
        g.set_kind(t, LayerKind::Transform(radial)).unwrap();
        let by_id: HashMap<LayerId, &Layer> = g.layers.iter().map(|l| (l.id, l)).collect();
        let ctx = EvalCtx::default();
        // u=0.9 -> radius ~36, past EXTEND_LIMIT: radial extends cap, so
        // the missing checker shows.
        let ts = apply_transform(&radial, Sample::uv(0.9, 0.5));
        assert!(ts.u > 0.5 + EXTEND_LIMIT);
        let expected = missing_texture(ts);
        let c = eval_layer(t, Sample::uv(0.9, 0.5), &by_id, &ctx);
        assert!((c.l - expected.l).abs() < 1e-5 && (c.chroma - expected.chroma).abs() < 1e-5);
    }
}
