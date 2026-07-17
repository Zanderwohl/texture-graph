use std::collections::HashMap;

use noise::{NoiseFn, Simplex};

use crate::color::{Color, blend, normal_to_color, scalar_of};
use crate::graph::{Graph, Layer};
use crate::id::LayerId;
use crate::kind::{
    Axis, BlendMode, ColorInput, ColorRamp, CoordMode, HeightToNormal, LayerKind, Map, Mix,
    Noise, NoiseDims, NoiseOutput, NoiseRange, RadialDim, ScalarInput, Transform,
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

impl Sample {
    pub const fn new(u: f32, v: f32, w: f32) -> Self {
        Self { u, v, w }
    }
    pub const fn uv(u: f32, v: f32) -> Self {
        Self { u, v, w: 0.0 }
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
    let color = eval_layer(out.color, s, &by_id, ctx);
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
        LayerKind::Transform(t) => eval_layer(t.source, apply_transform(t, s), by_id, ctx),
        LayerKind::Mix(m) => eval_mix(m, s, by_id, ctx),
        LayerKind::Map(m) => eval_map(m, s, by_id, ctx),
        LayerKind::HeightToNormal(h) => eval_h2n(h, s, by_id, ctx),
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

fn eval_noise(n: &Noise, s: Sample, ctx: &EvalCtx) -> Color {
    let sample = |off: u32| -> f32 {
        let simplex = Simplex::new(ctx.seed.wrapping_add(n.seed_offset).wrapping_add(off));
        let f = n.frequency as f64;
        let raw = match n.dims {
            NoiseDims::D1 => simplex.get([s.u as f64 * f, 0.0]),
            NoiseDims::D2 => simplex.get([s.u as f64 * f, s.v as f64 * f]),
            NoiseDims::D3 => simplex.get([s.u as f64 * f, s.v as f64 * f, s.w as f64 * f]),
        } as f32;
        match n.range {
            NoiseRange::Signed => raw,
            NoiseRange::Unsigned => raw * 0.5 + 0.5,
        }
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
    // Domain: `s.u`, unclamped. Extrapolation off either end uses the
    // outermost segment's slope (equivalent to using the first/last stop as
    // the neighbor).
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
    let t = if span.abs() < f32::EPSILON { 0.0 } else { (s.u - a.t) / span };
    let ca = eval_color_input(&a.color, s, by_id, ctx);
    let cb = eval_color_input(&b.color, s, by_id, ctx);
    blend(ca, cb, t, r.space)
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
    let a = eval_layer(m.a, s, by_id, ctx);
    let b = eval_layer(m.b, s, by_id, ctx);
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
    let value = eval_layer(m.value, s, by_id, ctx);
    let t = scalar_of(value);
    // Look up palette at (t, 0, 0).
    eval_layer(m.palette, Sample::new(t, 0.0, 0.0), by_id, ctx)
}

fn eval_h2n(h: &HeightToNormal, s: Sample, by_id: &HashMap<LayerId, &Layer>, ctx: &EvalCtx) -> Color {
    let eps = ctx.normal_epsilon.max(f32::EPSILON);
    let sample_l = |ds: Sample| scalar_of(eval_layer(h.source, ds, by_id, ctx));
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
                    a,
                    b: a,
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
                a: b,
                b,
                mode: BlendMode::Add,
                factor: ScalarInput::Const(0.5),
                space: BlendSpace::Oklch,
            }),
        );
        assert!(matches!(err, Err(crate::graph::GraphError::Cycle(_))));
    }
}
