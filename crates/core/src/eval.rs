use std::collections::{BTreeMap, HashMap};

use crate::color::{Color, blend, normal_to_color, oklcha, scalar_of};
use crate::graph::{Graph, Layer};
use crate::id::LayerId;
use crate::param::ParamValue;
use crate::kind::{
    Axis, BlendMode, ColorInput, ColorRamp, CoordMode, Criterion, EXTEND_LIMIT, EdgeMode,
    FractalMode, HeightToNormal, LayerKind, Map, MinMax, MinMaxMode, Mix, Noise, NoiseDims,
    NoiseKernel, NoiseOutput, NoiseRange, RadialDim, ScalarInput, Transform, Warp, WarpMode,
    Wave, WaveShape,
};

/// A point in the graph's unit cube. Callers map their own coordinates into
/// it.
#[derive(Copy, Clone, Debug)]
pub struct Sample {
    pub u: f32,
    pub v: f32,
    pub w: f32,
}

/// The `w` a flat bake samples a 3D field at. CPU and GPU must agree, or
/// they render different slices of the same volume.
pub const FLAT_W: f32 = 0.5;

impl Sample {
    pub const fn new(u: f32, v: f32, w: f32) -> Self {
        Self { u, v, w }
    }
    /// A sample at `w = 0`. A flat bake should use [`Sample::flat`].
    pub const fn uv(u: f32, v: f32) -> Self {
        Self { u, v, w: 0.0 }
    }
    /// A sample on the [`FLAT_W`] slice.
    pub const fn flat(u: f32, v: f32) -> Self {
        Self { u, v, w: FLAT_W }
    }
}

/// Settings constant across a whole bake.
#[derive(Clone, Debug)]
pub struct EvalCtx {
    pub seed: u32,
    /// Finite-difference step for `HeightToNormal`, in UV units.
    pub normal_epsilon: f32,
    /// Bindings for [`Graph::params`]. A parameter left out takes its
    /// declared default.
    pub params: BTreeMap<String, ParamValue>,
}

impl EvalCtx {
    /// The constant this socket reads, or `None` when it reads a layer.
    /// `self` must already be resolved with [`Graph::resolve_params`].
    pub fn scalar_const(&self, si: &ScalarInput) -> Option<f32> {
        match si {
            ScalarInput::Const(v) => Some(*v),
            ScalarInput::Layer(_) => None,
            ScalarInput::Param(name) => {
                Some(self.params.get(name.as_str()).and_then(ParamValue::as_scalar).unwrap_or(UNBOUND_SCALAR))
            }
        }
    }

    /// See [`EvalCtx::scalar_const`].
    pub fn color_const(&self, ci: &ColorInput) -> Option<Color> {
        match ci {
            ColorInput::Const(c) => Some(*c),
            ColorInput::Layer(_) => None,
            ColorInput::Param(name) => Some(
                self.params
                    .get(name.as_str())
                    .and_then(ParamValue::as_color)
                    .unwrap_or_else(unbound_color),
            ),
        }
    }

    pub fn with_param(mut self, name: impl Into<String>, value: ParamValue) -> Self {
        self.params.insert(name.into(), value);
        self
    }
}

impl Default for EvalCtx {
    fn default() -> Self {
        Self {
            seed: 0xC0DED00D,
            normal_epsilon: 1.0 / 512.0,
            params: BTreeMap::new(),
        }
    }
}

/// Every channel of the graph's Output at one sample.
#[derive(Copy, Clone, Debug)]
pub struct Material {
    pub color: Color,
    pub roughness: f32,
    pub metallic: f32,
    pub normal: Color,
}

/// What an undeclared parameter reads as. Only a hand-assembled `Graph`
/// can reach this; the mutators reject undeclared names.
const UNBOUND_SCALAR: f32 = 0.0;

fn unbound_color() -> Color {
    Color::new(0.0, 0.0, 0.0, 1.0)
}

pub fn evaluate_material(g: &Graph, s: Sample, ctx: &EvalCtx) -> Material {
    let ctx = &g.resolve_params(ctx);
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

/// Evaluate one layer's color, as for a per-layer preview.
pub fn evaluate(g: &Graph, id: LayerId, s: Sample, ctx: &EvalCtx) -> Color {
    let ctx = &g.resolve_params(ctx);
    let by_id: HashMap<LayerId, &Layer> = g.layers.iter().map(|l| (l.id, l)).collect();
    eval_layer(id, s, &by_id, ctx)
}

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
        LayerKind::Warp(w) => eval_warp(w, s, by_id, ctx),
        LayerKind::Coordinate(c) => {
            let at = match c.axis {
                Axis::U => s.u,
                Axis::V => s.v,
                Axis::W => s.w,
            };
            oklcha(at, 0.0, 0.0, 1.0)
        }
    }
}

/// The magenta/black checker shown for an unconnected input. Checkered in
/// `w` too, so it stays a 3D checker in volume bakes. Keep the cell count
/// and colors in sync with `missing.wgsl`. Magenta is Oklch of sRGB (1, 0, 1).
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
        // `ctx` is resolved, so a miss means a hand-built graph.
        ColorInput::Param(name) => ctx
            .params
            .get(name.as_str())
            .and_then(ParamValue::as_color)
            .unwrap_or_else(unbound_color),
    }
}

fn eval_scalar(si: &ScalarInput, s: Sample, by_id: &HashMap<LayerId, &Layer>, ctx: &EvalCtx) -> f32 {
    match si {
        ScalarInput::Const(v) => *v,
        ScalarInput::Layer(id) => scalar_of(eval_layer(*id, s, by_id, ctx)),
        ScalarInput::Param(name) => ctx
            .params
            .get(name.as_str())
            .and_then(ParamValue::as_scalar)
            .unwrap_or(UNBOUND_SCALAR),
    }
}

/// The enums stay separate so `crate::noise`, which the shader is
/// transcribed from, does not depend on the file format.
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
            // Chroma scale keeps most colors in gamut.
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
    // Samples along `s.u`. Past either end stop, the ramp holds that stop's color.
    debug_assert!(r.stops.len() >= 2, "ramp validation should reject <2 stops");
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

/// `Extend` is unbounded for affine mappings, since the GPU bakes sources
/// over the exact requested region. Radial mappings show the missing grid
/// beyond the [`EXTEND_LIMIT`] box, matching the GPU's cap.
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
    let mut u = s.u - t.offset[0];
    let mut v = s.v - t.offset[1];
    let w = s.w - t.offset[2];
    if t.rotate_uv != 0.0 {
        let (sin, cos) = t.rotate_uv.sin_cos();
        let ru = u * cos - v * sin;
        let rv = u * sin + v * cos;
        u = ru;
        v = rv;
    }
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
        // Oklab, where chroma is a Cartesian (a, b) vector that sums.
        BlendMode::Add | BlendMode::Subtract => {
            let sign = if matches!(m.mode, BlendMode::Add) { 1.0 } else { -1.0 };
            let al: Oklab = a.color.into_color();
            let bl: Oklab = b.color.into_color();
            let combined = Oklab::new(al.l + sign * bl.l, al.a + sign * bl.a, al.b + sign * bl.b);
            let back: Oklch = combined.into_color();
            back.with_alpha(a.alpha)
        }
        // Linear sRGB, for optical darkening.
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

fn criterion_of(c: Color, crit: Criterion) -> f32 {
    use palette::{Hsv, IntoColor, Srgb};
    match crit {
        Criterion::Alpha => return c.alpha,
        Criterion::Chroma => return c.chroma,
        _ => {}
    }
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
        Criterion::Alpha | Criterion::Chroma => unreachable!(),
    }
}

/// Per-axis displacement before `amount` scales it. Must match `warp.wgsl`.
pub fn warp_displacement(by: Color, mode: WarpMode) -> [f32; 3] {
    match mode {
        WarpMode::Scalar => {
            let l = scalar_of(by);
            [l, l, l]
        }
        // Hue in turns, not degrees, so it is on a similar scale to L.
        WarpMode::Vector => [by.l, by.chroma, by.hue.into_degrees() / 360.0],
    }
}

/// The region a warp may sample: the unit square grown by `|amount|`
/// (assuming a driver in `[-1, 1]`), capped at the [`EXTEND_LIMIT`] box.
/// Must match the region `schedule.rs` bakes the source over.
fn warp_bounds(amount: [f32; 3]) -> ([f32; 2], [f32; 2]) {
    let lo = |a: f32| (0.0 - a.abs()).max(0.5 - EXTEND_LIMIT);
    let hi = |a: f32| (1.0 + a.abs()).min(0.5 + EXTEND_LIMIT);
    ([lo(amount[0]), lo(amount[1])], [hi(amount[0]), hi(amount[1])])
}

fn eval_warp(w: &Warp, s: Sample, by_id: &HashMap<LayerId, &Layer>, ctx: &EvalCtx) -> Color {
    let d = warp_displacement(eval_opt(w.by, s, by_id, ctx), w.mode);
    let moved = Sample::new(
        s.u + d[0] * w.amount[0],
        s.v + d[1] * w.amount[1],
        s.w + d[2] * w.amount[2],
    );
    let (lo, hi) = warp_bounds(w.amount);
    if moved.u < lo[0] || moved.u > hi[0] || moved.v < lo[1] || moved.v > hi[1] {
        return missing_texture(moved);
    }
    eval_opt(w.source, moved, by_id, ctx)
}

/// One cycle of `shape` at phase `t` (in cycles), in `[-1, 1]`.
///
/// Uses `x - floor(x)`, not `f32::fract`, which truncates toward zero and
/// so differs from WGSL's `fract` on negative input.
pub fn wave_cycle(t: f32, shape: WaveShape) -> f32 {
    let frac = |x: f32| x - x.floor();
    let t = frac(t);
    match shape {
        WaveShape::Sine => (std::f32::consts::TAU * t).sin(),
        // In phase with sine: 0 at t = 0, +1 at t = 0.25.
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
    let dhdx = (l_px - l_nx) / (2.0 * eps);
    let dhdy = (l_py - l_ny) / (2.0 * eps);
    let mut n = [-dhdx * h.strength, -dhdy * h.strength, 1.0];
    let mag = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(f32::EPSILON);
    n[0] /= mag;
    n[1] /= mag;
    n[2] /= mag;
    normal_to_color(n)
}

#[cfg(test)]
mod param_tests {
    use super::*;
    use crate::color::oklcha;
    use crate::kind::{BlendMode, ColorRamp, ColorStop, Mix};
    use crate::param::{ParamDecl, ParamValue};

    /// A Mix whose factor is the scalar parameter `contrast`.
    fn parameterized() -> (Graph, LayerId) {
        let mut g = Graph::new();
        g.declare_param(ParamDecl::scalar("contrast", 0.0, 1.0, 0.25)).unwrap();
        let dark = g.output.color.unwrap();
        g.set_kind(dark, LayerKind::Color(oklcha(0.0, 0.0, 0.0, 1.0))).unwrap();
        let light = g
            .add_layer("light", LayerKind::Color(oklcha(1.0, 0.0, 0.0, 1.0)))
            .unwrap();
        let mix = g
            .add_layer(
                "mix",
                LayerKind::Mix(Mix {
                    a: Some(dark),
                    b: Some(light),
                    mode: BlendMode::Blend,
                    factor: ScalarInput::Param("contrast".into()),
                    space: crate::color::BlendSpace::Oklch,
                }),
            )
            .unwrap();
        (g, mix)
    }

    #[test]
    fn one_graph_reads_differently_under_different_bindings() {
        let (g, mix) = parameterized();
        let at = |ctx: &EvalCtx| scalar_of(evaluate(&g, mix, Sample::uv(0.5, 0.5), ctx));

        let default = at(&EvalCtx::default());
        assert!((default - 0.25).abs() < 1e-5, "default gave {default}");

        for want in [0.0f32, 0.5, 1.0] {
            let ctx = EvalCtx::default().with_param("contrast", ParamValue::Scalar(want));
            let got = at(&ctx);
            assert!((got - want).abs() < 1e-5, "bound {want} read as {got}");
        }
    }

    #[test]
    fn a_color_parameter_drives_a_ramp_stop() {
        let mut g = Graph::new();
        g.declare_param(ParamDecl::color("tint", oklcha(0.2, 0.0, 0.0, 1.0))).unwrap();
        let ramp = g.output.color.unwrap();
        g.set_kind(
            ramp,
            LayerKind::ColorRamp(ColorRamp {
                stops: vec![
                    ColorStop { t: 0.0, color: ColorInput::Param("tint".into()) },
                    ColorStop { t: 1.0, color: ColorInput::Param("tint".into()) },
                ],
                space: crate::color::BlendSpace::Oklch,
            }),
        )
        .unwrap();
        let at = |ctx: &EvalCtx| evaluate(&g, ramp, Sample::uv(0.5, 0.5), ctx);

        assert!((at(&EvalCtx::default()).l - 0.2).abs() < 1e-5);
        let bound = EvalCtx::default()
            .with_param("tint", ParamValue::Color(oklcha(0.8, 0.1, 210.0, 1.0)));
        let got = at(&bound);
        assert!((got.l - 0.8).abs() < 1e-5, "L was {}", got.l);
        assert!((got.chroma - 0.1).abs() < 1e-5, "C was {}", got.chroma);
    }

    #[test]
    fn the_caller_does_not_have_to_resolve_first() {
        let (g, mix) = parameterized();
        let raw = EvalCtx::default().with_param("contrast", ParamValue::Scalar(0.9));
        let pre_resolved = g.resolve_params(&raw);
        assert_eq!(
            scalar_of(evaluate(&g, mix, Sample::uv(0.5, 0.5), &raw)),
            scalar_of(evaluate(&g, mix, Sample::uv(0.5, 0.5), &pre_resolved)),
        );
    }
}

#[cfg(test)]
mod warp_tests {
    use super::*;
    use crate::color::oklcha;
    use crate::kind::{Warp, WarpMode};

    /// A U ramp warped by a flat driver of lightness `driver_l`.
    fn warped(driver_l: f32, amount: [f32; 3], mode: WarpMode) -> (Graph, LayerId) {
        let mut g = Graph::new();
        let ramp = g.output.color.unwrap();
        g.set_kind(
            ramp,
            LayerKind::ColorRamp(crate::kind::ColorRamp {
                stops: vec![
                    crate::kind::ColorStop {
                        t: 0.0,
                        color: ColorInput::Const(oklcha(0.0, 0.0, 0.0, 1.0)),
                    },
                    crate::kind::ColorStop {
                        t: 1.0,
                        color: ColorInput::Const(oklcha(1.0, 0.0, 0.0, 1.0)),
                    },
                ],
                space: crate::color::BlendSpace::Oklch,
            }),
        )
        .unwrap();
        let driver = g
            .add_layer("driver", LayerKind::Color(oklcha(driver_l, 0.0, 0.0, 1.0)))
            .unwrap();
        let w = g
            .add_layer(
                "warp",
                LayerKind::Warp(Warp {
                    source: Some(ramp),
                    by: Some(driver),
                    mode,
                    amount,
                }),
            )
            .unwrap();
        (g, w)
    }

    /// The driver is not recenterd: a 1.0 driver with amount 0.25 shifts by 0.25.
    #[test]
    fn the_displacement_is_the_drivers_value_times_amount() {
        let (g, w) = warped(1.0, [0.25, 0.0, 0.0], WarpMode::Scalar);
        let ctx = EvalCtx::default();
        let ramp = g.output.color.unwrap();
        for k in 0..8 {
            let u = k as f32 / 16.0;
            let warped_here = scalar_of(evaluate(&g, w, Sample::uv(u, 0.5), &ctx));
            let source_there = scalar_of(evaluate(&g, ramp, Sample::uv(u + 0.25, 0.5), &ctx));
            assert!(
                (warped_here - source_there).abs() < 1e-5,
                "at u={u}: warp gave {warped_here}, source at u+0.25 gave {source_there}"
            );
        }
    }

    #[test]
    fn a_zero_driver_is_a_no_op() {
        let (g, w) = warped(0.0, [0.5, 0.5, 0.0], WarpMode::Scalar);
        let ctx = EvalCtx::default();
        let ramp = g.output.color.unwrap();
        for k in 0..16 {
            let u = k as f32 / 16.0;
            assert_eq!(
                scalar_of(evaluate(&g, w, Sample::uv(u, 0.5), &ctx)),
                scalar_of(evaluate(&g, ramp, Sample::uv(u, 0.5), &ctx)),
            );
        }
    }

    /// A driver in `[-1, 1]` lands on data; beyond it the missing grid
    /// shows, as on the GPU when a fetch leaves the baked domain.
    #[test]
    fn a_driver_past_one_falls_off_the_baked_domain() {
        let ctx = EvalCtx::default();
        let (inside, w_in) = warped(1.0, [0.1, 0.0, 0.0], WarpMode::Scalar);
        let at_edge = evaluate(&inside, w_in, Sample::uv(1.0, 0.5), &ctx);
        assert_ne!(at_edge, missing_texture(Sample::uv(1.1, 0.5)));

        let (outside, w_out) = warped(4.0, [0.1, 0.0, 0.0], WarpMode::Scalar);
        let far = evaluate(&outside, w_out, Sample::uv(1.0, 0.5), &ctx);
        assert_eq!(far, missing_texture(Sample::new(1.4, 0.5, 0.0)));
    }

    #[test]
    fn vector_mode_reads_three_channels() {
        let c = oklcha(0.8, 0.1, 180.0, 1.0);
        assert_eq!(warp_displacement(c, WarpMode::Scalar), [0.8, 0.8, 0.8]);
        let v = warp_displacement(c, WarpMode::Vector);
        assert!((v[0] - 0.8).abs() < 1e-6);
        assert!((v[1] - 0.1).abs() < 1e-6);
        assert!((v[2] - 0.5).abs() < 1e-6, "hue should be half a turn, got {}", v[2]);
    }

    #[test]
    fn a_varying_driver_distorts_rather_than_shifts() {
        let mut g = Graph::new();
        let ramp = g.output.color.unwrap();
        g.set_kind(
            ramp,
            LayerKind::ColorRamp(crate::kind::ColorRamp {
                stops: vec![
                    crate::kind::ColorStop {
                        t: 0.0,
                        color: ColorInput::Const(oklcha(0.0, 0.0, 0.0, 1.0)),
                    },
                    crate::kind::ColorStop {
                        t: 1.0,
                        color: ColorInput::Const(oklcha(1.0, 0.0, 0.0, 1.0)),
                    },
                ],
                space: crate::color::BlendSpace::Oklch,
            }),
        )
        .unwrap();
        let driver = g
            .add_layer(
                "driver",
                LayerKind::Noise(crate::kind::Noise {
                    range: NoiseRange::Signed,
                    kernel: crate::kind::NoiseKernel::Value,
                    frequency: 6.0,
                    ..crate::kind::Noise::default()
                }),
            )
            .unwrap();
        let w = g
            .add_layer(
                "warp",
                LayerKind::Warp(Warp {
                    source: Some(ramp),
                    by: Some(driver),
                    mode: WarpMode::Scalar,
                    amount: [0.2, 0.0, 0.0],
                }),
            )
            .unwrap();
        let ctx = EvalCtx::default();
        // The ramp is constant down a column; the warp must not be.
        let column: Vec<f32> = (0..16)
            .map(|k| scalar_of(evaluate(&g, w, Sample::uv(0.5, k as f32 / 16.0), &ctx)))
            .collect();
        let first = column[0];
        assert!(
            column.iter().any(|v| (v - first).abs() > 0.01),
            "the warp shifted uniformly instead of distorting: {column:?}"
        );
    }
}

#[cfg(test)]
mod wave_tests {
    use super::*;
    use crate::kind::{Wave, WaveShape};

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

    /// Sine, triangle and square agree on sign; sawtooth is exempt.
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
        assert!((wave_cycle(0.25, WaveShape::Triangle) - 1.0).abs() < 1e-6);
        assert!((wave_cycle(0.75, WaveShape::Triangle) + 1.0).abs() < 1e-6);
        assert_eq!(wave_cycle(0.0, WaveShape::Sawtooth), -1.0);
        assert!((wave_cycle(0.999, WaveShape::Sawtooth) - 1.0).abs() < 0.01);
    }

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

        // 0.125 at frequency 2 is phase 0.25, sine's peak.
        g.set_kind(id, LayerKind::Wave(wave(2.0, 0.0, NoiseRange::Signed))).unwrap();
        assert!((at(&g, id) - 1.0).abs() < 1e-5);
        g.set_kind(id, LayerKind::Wave(wave(2.0, 1.0, NoiseRange::Signed))).unwrap();
        assert!((at(&g, id) - 1.0).abs() < 1e-5);
        g.set_kind(id, LayerKind::Wave(wave(2.0, 0.5, NoiseRange::Signed))).unwrap();
        assert!((at(&g, id) + 1.0).abs() < 1e-5);
        g.set_kind(id, LayerKind::Wave(wave(2.0, 0.0, NoiseRange::Unsigned))).unwrap();
        assert!((at(&g, id) - 1.0).abs() < 1e-5);
        g.set_kind(id, LayerKind::Wave(wave(2.0, 0.5, NoiseRange::Unsigned))).unwrap();
        assert!(at(&g, id).abs() < 1e-5);
    }

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
        // u=0.5 scales to 2.0 and clamps to 1.0.
        let c = eval_layer(t, Sample::uv(0.5, 0.5), &by_id, &ctx);
        assert!((c.l - 1.0).abs() < 1e-4, "expected clamped end color, got L={}", c.l);
    }

    #[test]
    fn transform_extend_samples_beyond_unit_square() {
        let (g, t) = ramp_under_transform(EdgeMode::Extend, 4.0);
        let by_id: HashMap<LayerId, &Layer> = g.layers.iter().map(|l| (l.id, l)).collect();
        let ctx = EvalCtx::default();
        let c = eval_layer(t, Sample::uv(0.5, 0.5), &by_id, &ctx);
        assert!((c.l - 1.0).abs() < 1e-4);
        assert!(c.chroma.abs() < 1e-4, "should be the ramp color, not the magenta grid");
    }

    #[test]
    fn transform_extend_affine_is_unbounded() {
        let (g, t) = ramp_under_transform(EdgeMode::Extend, 40.0);
        let by_id: HashMap<LayerId, &Layer> = g.layers.iter().map(|l| (l.id, l)).collect();
        let ctx = EvalCtx::default();
        // u=0.9 scales to 36.0, far past EXTEND_LIMIT.
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
        let ts = apply_transform(&radial, Sample::uv(0.9, 0.5));
        assert!(ts.u > 0.5 + EXTEND_LIMIT);
        let expected = missing_texture(ts);
        let c = eval_layer(t, Sample::uv(0.9, 0.5), &by_id, &ctx);
        assert!((c.l - expected.l).abs() < 1e-5 && (c.chroma - expected.chroma).abs() < 1e-5);
    }
}
