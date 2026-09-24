use serde::{Deserialize, Serialize};

use crate::color::{BlendSpace, Color};
use crate::id::LayerId;
use crate::param::ParamUse;

/// A node's type and settings.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum LayerKind {
    Color(Color),
    Noise(Noise),
    ColorRamp(ColorRamp),
    Transform(Transform),
    Mix(Mix),
    Map(Map),
    MinMax(MinMax),
    HeightToNormal(HeightToNormal),
    Wave(Wave),
    Warp(Warp),
    Coordinate(Coordinate),
    Craters(Craters),
}

impl LayerKind {
    pub fn category_label(&self) -> &'static str {
        match self {
            LayerKind::Color(_) => "Color",
            LayerKind::Noise(_) => "Noise",
            LayerKind::ColorRamp(_) => "ColorRamp",
            LayerKind::Transform(_) => "Transform",
            LayerKind::Mix(_) => "Mix",
            LayerKind::Map(_) => "Map",
            LayerKind::MinMax(_) => "MinMax",
            LayerKind::HeightToNormal(_) => "HeightToNormal",
            LayerKind::Wave(_) => "Wave",
            LayerKind::Warp(_) => "Warp",
            LayerKind::Coordinate(_) => "Coordinate",
            LayerKind::Craters(_) => "Craters",
        }
    }

    /// Every layer this node depends on. Must agree with
    /// [`LayerKind::input_sockets`] on both ids and order.
    pub fn inputs(&self) -> Vec<LayerId> {
        let mut out = Vec::new();
        let push_color = |out: &mut Vec<LayerId>, ci: &ColorInput| {
            if let ColorInput::Layer(id) = ci {
                out.push(*id);
            }
        };
        let push_scalar = |out: &mut Vec<LayerId>, si: &ScalarInput| {
            if let ScalarInput::Layer(id) = si {
                out.push(*id);
            }
        };
        match self {
            LayerKind::Color(_) | LayerKind::Noise(_) | LayerKind::Coordinate(_) => {}
            LayerKind::ColorRamp(r) => {
                for s in &r.stops {
                    push_color(&mut out, &s.color);
                }
            }
            LayerKind::Transform(t) => out.extend(t.source),
            LayerKind::Mix(m) => {
                out.extend(m.a);
                out.extend(m.b);
                push_scalar(&mut out, &m.factor);
            }
            LayerKind::Map(m) => {
                out.extend(m.value);
                out.extend(m.palette);
            }
            LayerKind::MinMax(mm) => {
                out.extend(mm.a);
                out.extend(mm.b);
            }
            LayerKind::HeightToNormal(h) => out.extend(h.source),
            LayerKind::Wave(w) => push_scalar(&mut out, &w.input),
            LayerKind::Warp(w) => {
                out.extend(w.source);
                out.extend(w.by);
            }
            LayerKind::Craters(c) => {
                push_scalar(&mut out, &c.under);
                push_scalar(&mut out, &c.density);
            }
        }
        out
    }

    /// Every parameter this node reads, with the sort of socket reading it.
    pub fn param_refs(&self) -> Vec<(&str, ParamUse)> {
        let mut out = Vec::new();
        match self {
            LayerKind::Color(_) | LayerKind::Noise(_) => {}
            LayerKind::ColorRamp(r) => {
                for s in &r.stops {
                    if let ColorInput::Param(name) = &s.color {
                        out.push((name.as_str(), ParamUse::Color));
                    }
                }
            }
            LayerKind::Mix(m) => {
                if let ScalarInput::Param(name) = &m.factor {
                    out.push((name.as_str(), ParamUse::Scalar));
                }
            }
            LayerKind::Wave(w) => {
                if let ScalarInput::Param(name) = &w.input {
                    out.push((name.as_str(), ParamUse::Scalar));
                }
            }
            LayerKind::Craters(c) => {
                for si in [&c.under, &c.density] {
                    if let ScalarInput::Param(name) = si {
                        out.push((name.as_str(), ParamUse::Scalar));
                    }
                }
            }
            LayerKind::Transform(_)
            | LayerKind::Map(_)
            | LayerKind::MinMax(_)
            | LayerKind::HeightToNormal(_)
            | LayerKind::Warp(_)
            | LayerKind::Coordinate(_) => {}
        }
        out
    }
}

/// A color-valued input.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ColorInput {
    Const(Color),
    Layer(LayerId),
    /// A parameter from [`crate::Graph::params`], bound per bake by
    /// [`crate::EvalCtx::params`]. Must be declared as a
    /// [`crate::ParamKind::Color`]. On the GPU it becomes the same uniform a
    /// `Const` would.
    Param(String),
}

/// A scalar-valued input. A layer is read as its Oklch L.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ScalarInput {
    Const(f32),
    Layer(LayerId),
    /// As [`ColorInput::Param`], declared as [`crate::ParamKind::Scalar`].
    Param(String),
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct Noise {
    pub dims: NoiseDims,
    pub seed_offset: u32,
    /// Cycles per unit of sample space.
    pub frequency: f32,
    pub range: NoiseRange,
    pub output: NoiseOutput,
    #[serde(default)]
    pub kernel: NoiseKernel,
    /// Lattice period in cells per axis; `0` is unbounded. The field
    /// repeats every `period / frequency` units, so it tiles the unit cube
    /// when `period == frequency` and both are integral.
    ///
    /// [`NoiseKernel::Value`] only; a nonzero period on `Simplex` is
    /// [`crate::GraphError::PeriodicSimplex`]. Tiled 3D simplex needs a 6D
    /// kernel, which this crate lacks.
    #[serde(default)]
    pub period: [u32; 3],
    #[serde(default)]
    pub fractal: Fractal,
}

impl Default for Noise {
    fn default() -> Self {
        Self {
            dims: NoiseDims::D2,
            seed_offset: 0,
            frequency: 4.0,
            range: NoiseRange::Unsigned,
            output: NoiseOutput::Grayscale,
            kernel: NoiseKernel::Simplex,
            period: [0; 3],
            fractal: Fractal::default(),
        }
    }
}

/// Implemented in [`crate::noise`] and `noise.wgsl`.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum NoiseKernel {
    /// Gustavson simplex. Cannot tile.
    #[default]
    Simplex,
    /// Trilinear value noise on an integer lattice, smoothstep weights.
    /// Can tile.
    Value,
}

/// Octave stack. Part of [`Noise`] rather than a separate node because the
/// baker evaluates each layer once per pixel over a fixed domain, so a
/// downstream node cannot re-sample its input at another scale.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Fractal {
    /// `1..=`[`crate::noise::MAX_OCTAVES`]. 1 is exactly the bare kernel.
    pub octaves: u32,
    /// Frequency multiplier per octave. Only an integral value keeps a
    /// [`Noise::period`] tiling exactly.
    pub lacunarity: f32,
    /// Amplitude multiplier per octave.
    pub gain: f32,
    pub mode: FractalMode,
    /// Divide by the sum of amplitudes so the result stays in range.
    pub normalize: bool,
}

impl Default for Fractal {
    /// One octave, the bare kernel.
    fn default() -> Self {
        Self {
            octaves: 1,
            lacunarity: 2.0,
            gain: 0.5,
            mode: FractalMode::Standard,
            normalize: true,
        }
    }
}

/// How each octave is shaped before it is summed. Written over the signed
/// sample `r ∈ [-1, 1]`; `n` below is the unsigned `r * 0.5 + 0.5`.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum FractalMode {
    /// `Σ aᵢ · n(fᵢx)`, ordinary fbm.
    #[default]
    Standard,
    /// `Σ aᵢ · |n|`.
    Turbulence,
    /// `Σ aᵢ · (1 - |2n - 1|)²`.
    Ridged,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum NoiseDims { D1, D2, D3 }

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum NoiseRange {
    /// `[0, 1]`.
    Unsigned,
    /// `[-1, 1]`.
    Signed,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum NoiseOutput {
    /// One sample drives L; C = 0.
    Grayscale,
    /// Three independent samples drive L, C and hue.
    Color,
}

/// Color ramp sampled along U. Evaluation assumes `stops` sorted by `t`;
/// [`crate::Graph`] does not enforce it.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ColorRamp {
    pub stops: Vec<ColorStop>,
    pub space: BlendSpace,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ColorStop {
    pub t: f32,
    pub color: ColorInput,
}

/// Moves the sample point before evaluating `source`: subtract `offset`
/// (the center for rotation and radial modes), rotate by `rotate_uv`,
/// multiply by `scale`, then apply `coord_mode`.
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct Transform {
    /// `None` renders as the missing-texture grid.
    pub source: Option<LayerId>,
    pub offset: [f32; 3],
    /// Radians, in the UV plane.
    pub rotate_uv: f32,
    pub scale: [f32; 3],
    pub coord_mode: CoordMode,
    #[serde(default)]
    pub edge_mode: EdgeMode,
}

/// What a [`Transform`] does when U or V leaves `[0, 1]`.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum EdgeMode {
    /// Clamp U and V to `[0, 1]`.
    #[default]
    Clamp,
    /// Sample beyond `[0, 1]`. Unbounded for affine modes; radial modes
    /// show the missing-texture grid past [`EXTEND_LIMIT`].
    Extend,
}

/// How far a radial [`EdgeMode::Extend`] transform may sample from 0.5 on
/// U and V. Outside `0.5 ± EXTEND_LIMIT` the missing-texture grid shows.
pub const EXTEND_LIMIT: f32 = 8.0;

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub enum CoordMode {
    Passthrough,
    /// `[V, U, W]` swaps u and v.
    Permute([Axis; 3]),
    /// Distance from the origin, fed into axis `into`; the others are 0.
    Radial { dim: RadialDim, into: Axis },
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum Axis {
    #[default]
    U,
    V,
    W,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum RadialDim {
    /// `sqrt(u² + v²)`.
    D2,
    /// `sqrt(u² + v² + w²)`.
    D3,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Mix {
    /// `None` renders as the missing-texture grid.
    pub a: Option<LayerId>,
    pub b: Option<LayerId>,
    pub mode: BlendMode,
    /// Only used by `BlendMode::Blend`.
    pub factor: ScalarInput,
    /// Only used by `BlendMode::Blend`.
    pub space: BlendSpace,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum BlendMode {
    Add,
    Subtract,
    Multiply,
    Blend,
}

/// Gradient map: samples `palette` at `(L, 0, 0)`, where L is `value`'s.
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct Map {
    /// `None` renders as the missing-texture grid.
    pub value: Option<LayerId>,
    pub palette: Option<LayerId>,
}

/// Tangent-space normal map from `source`'s L as height. `strength` scales
/// the slope.
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct HeightToNormal {
    /// `None` renders as the missing-texture grid.
    pub source: Option<LayerId>,
    pub strength: f32,
}

/// Domain warp: displace the sample point by `by`'s value times `amount`,
/// then read `source` there. The value is not recenterd, so an unsigned
/// driver pushes in one direction only.
///
/// The driver is assumed to lie in `[-1, 1]`. The baker grows `source`'s
/// domain by `|amount|` per axis, capped at [`EXTEND_LIMIT`]; a displacement
/// beyond that shows the missing-texture grid on both backends.
///
/// `amount[2]` works on the CPU only. The GPU bakes each slice's inputs at
/// that slice's `w` and cannot re-sample them at another, so leave it at
/// zero unless the CPU evaluator is the only consumer.
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct Warp {
    /// `None` renders as the missing-texture grid.
    pub source: Option<LayerId>,
    /// `None` displaces by the missing-texture grid's values, so the
    /// mistake is visible.
    pub by: Option<LayerId>,
    pub mode: WarpMode,
    /// Per-axis scale on the displacement, in sample-space units.
    pub amount: [f32; 3],
}

impl Default for Warp {
    fn default() -> Self {
        Self {
            source: None,
            by: None,
            mode: WarpMode::Scalar,
            amount: [0.1, 0.1, 0.0],
        }
    }
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum WarpMode {
    /// The driver's L displaces all three axes.
    #[default]
    Scalar,
    /// L, C and hue / 360 drive x, y and z. They differ in scale (Color
    /// noise puts chroma in `[0, 0.15]`), which is why `amount` is per-axis.
    Vector,
}

/// A periodic waveform of `input * frequency + phase`, output as gray.
///
/// Not bound by the noise op-order contract: [`WaveShape::Sine`] is
/// transcendental, so CPU and GPU agree to within one sRGB step rather than
/// bit-exactly.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Wave {
    pub input: ScalarInput,
    pub shape: WaveShape,
    /// Cycles per unit of `input`. A shader's `sin(x * k)` is
    /// `frequency: k / 2π`.
    pub frequency: f32,
    /// Offset in cycles, not radians.
    pub phase: f32,
    pub range: NoiseRange,
}

impl Default for Wave {
    fn default() -> Self {
        Self {
            input: ScalarInput::Const(0.5),
            shape: WaveShape::Sine,
            frequency: 4.0,
            phase: 0.0,
            range: NoiseRange::Unsigned,
        }
    }
}

/// One axis of the sample point as a gray: L is the coordinate, unclamped.
/// In a sphere bake `V` is the point's height, not a row of the face.
#[derive(Copy, Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Coordinate {
    pub axis: Axis,
}

/// Defined over the phase `t ∈ [0, 1)`, with output in `[-1, 1]`.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum WaveShape {
    /// `sin(2πt)`.
    #[default]
    Sine,
    /// In phase with `Sine`: 0 at `t = 0`, peak at `t = 0.25`.
    Triangle,
    /// The sign of `Sine`.
    Square,
    /// Rises from `-1` to `+1` across the cycle. Not in phase with `Sine`.
    Sawtooth,
}

/// Per pixel, outputs whichever input wins on `criterion`, all channels
/// included.
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct MinMax {
    /// `None` renders as the missing-texture grid.
    pub a: Option<LayerId>,
    pub b: Option<LayerId>,
    pub mode: MinMaxMode,
    pub criterion: Criterion,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum MinMaxMode {
    Min,
    Max,
}

/// RGB, Saturation, Value and Luma are read in gamma-encoded sRGB, so they
/// match what a monitor shows. Alpha and Chroma come from Oklcha.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum Criterion {
    Red,
    Green,
    Blue,
    Saturation,
    Value,
    /// Rec.709 weights: `0.2126R + 0.7152G + 0.0722B`.
    Luma,
    Alpha,
    Chroma,
}

/// A scatter of impact craters over `under`, as height or as the brightness
/// of their ejecta. See [`crate::crater`] for the field and how craters
/// overprint what they land on; chain nodes through `under` for successive
/// series of impacts.
///
/// Height is `under` plus relief in sample-space units times `relief`, about
/// [`crate::crater::DATUM`]. Two nodes that differ only in `output` place the
/// same craters.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Craters {
    pub under: ScalarInput,
    /// Share of the largest class's cells holding a crater, `[0, 1]`. A layer
    /// is read where each pixel is, so it fades craters across its gradients
    /// rather than deciding them whole.
    pub density: ScalarInput,
    pub seed_offset: u32,
    /// Cells per unit of sample space in the largest class. A rim is at most a
    /// fifth of a cell in radius.
    pub frequency: f32,
    /// Size classes, each half the last, `1..=`[`crate::crater::MAX_CLASSES`].
    pub classes: u32,
    /// Occupancy per class relative to the class before: 1 is a `D^-2`
    /// cumulative count, more favors small craters.
    pub gain: f32,
    /// Bowl depth over rim radius, fresh and simple. The Moon's is about 0.4.
    pub depth: f32,
    /// `[0, 1]`: shallower bowls, lower rims and darker ejecta.
    pub age: f32,
    /// `[0, 1]`: how far a crater wipes out what was there before.
    pub erase: f32,
    /// Central peak in bowl depths, on the largest craters.
    pub peak: f32,
    /// `[0, 1]`: how much of the ejecta's brightness is in rays.
    pub rays: f32,
    pub relief: f32,
    pub surface: CraterSurface,
    pub output: CraterOutput,
}

impl Default for Craters {
    fn default() -> Self {
        Self {
            under: ScalarInput::Const(crate::crater::DATUM),
            density: ScalarInput::Const(0.6),
            seed_offset: 0,
            frequency: 4.0,
            classes: 4,
            gain: 1.0,
            depth: 0.4,
            age: 0.0,
            erase: 1.0,
            peak: 0.5,
            rays: 0.5,
            relief: 10.0,
            surface: CraterSurface::Plane,
            output: CraterOutput::Height,
        }
    }
}

/// Which way is up, so a crater is round on the surface being baked.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum CraterSurface {
    /// `+w`, for a flat bake.
    #[default]
    Plane,
    /// Away from the unit cube's center, for a sphere bake.
    Sphere,
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum CraterOutput {
    #[default]
    Height,
    /// Fresh ejecta's brightness in `[0, 1]`, rays included, over `under`.
    Ejecta,
}
