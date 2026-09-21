use serde::{Deserialize, Serialize};

use crate::color::{BlendSpace, Color};
use crate::id::LayerId;

/// One of the discriminated node types in the graph.
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
        }
    }

    /// Every layer this node depends on, for cycle detection and
    /// invalidation. Must agree with [`LayerKind::input_sockets`] on both
    /// ids and order.
    pub fn inputs(&self) -> Vec<LayerId> {
        let mut out = Vec::new();
        let push_color = |out: &mut Vec<LayerId>, ci: ColorInput| {
            if let ColorInput::Layer(id) = ci {
                out.push(id);
            }
        };
        let push_scalar = |out: &mut Vec<LayerId>, si: ScalarInput| {
            if let ScalarInput::Layer(id) = si {
                out.push(id);
            }
        };
        match self {
            LayerKind::Color(_) | LayerKind::Noise(_) => {}
            LayerKind::ColorRamp(r) => {
                for s in &r.stops {
                    push_color(&mut out, s.color);
                }
            }
            LayerKind::Transform(t) => out.extend(t.source),
            LayerKind::Mix(m) => {
                out.extend(m.a);
                out.extend(m.b);
                push_scalar(&mut out, m.factor);
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
            LayerKind::Wave(w) => push_scalar(&mut out, w.input),
        }
        out
    }
}

// ---- Inputs -------------------------------------------------------------

/// A color-valued input: either a constant chosen in the UI or another
/// layer's output.
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub enum ColorInput {
    Const(Color),
    Layer(LayerId),
}

/// A scalar-valued input. When a layer is referenced, its color's Oklch L
/// (perceptual lightness) is used as the scalar.
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub enum ScalarInput {
    Const(f32),
    Layer(LayerId),
}

// ---- Node payloads ------------------------------------------------------

/// Noise at a chosen dimensionality, with a chosen numeric range and
/// either grayscale or LCh-colorful output.
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct Noise {
    pub dims: NoiseDims,
    pub seed_offset: u32,
    /// Cycles per unit sample-space; `1.0` = one feature per full UVW cube.
    pub frequency: f32,
    pub range: NoiseRange,
    pub output: NoiseOutput,
    /// Which kernel generates the field. A file without the field loads as
    /// [`NoiseKernel::Simplex`], which is what every graph written before
    /// this field existed contains.
    #[serde(default)]
    pub kernel: NoiseKernel,
    /// Lattice period in cells, per axis; `0` = unbounded on that axis.
    ///
    /// With `frequency = f` and `period = p`, the lattice cell index is
    /// taken `mod p`, so the field repeats every `p / f` units of sample
    /// space. Seamless across the unit cube is exactly the case `p == f`
    /// with `f` integral — anything else tiles, just not on the cube.
    ///
    /// Per-axis rather than one flag because periodicity is often wanted
    /// on one axis alone: an animation that scrolls through w forever
    /// wants `[0, 0, 64]` and nothing more.
    ///
    /// [`NoiseKernel::Value`] only — a nonzero period on `Simplex` is
    /// rejected as [`crate::GraphError::PeriodicSimplex`] rather than
    /// silently ignored, because tiled simplex needs a 6D kernel for 3D
    /// and this crate does not have one.
    #[serde(default)]
    pub period: [u32; 3],
    /// Octave stack. The default is one octave, which is the plain kernel.
    #[serde(default)]
    pub fractal: Fractal,
}

impl Default for Noise {
    /// The node a fresh Noise layer starts as: one octave of aperiodic
    /// simplex, grayscale, `[0, 1]`. Every other field is `..Default`'s
    /// job at a call site that only cares about one of them.
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

/// Which kernel generates a [`Noise`] field. The twin implementations are
/// [`crate::noise`] (CPU) and `noise.wgsl` (GPU).
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum NoiseKernel {
    /// Gustavson simplex. Aperiodic — it has no lattice to wrap — and the
    /// default a file without the field loads as.
    #[default]
    Simplex,
    /// Trilinear value noise on an integer lattice, smoothstep weights.
    /// The only kernel that can tile, and the one a hand-written shader
    /// most likely already uses.
    Value,
}

/// Octave stack applied inside the kernel dispatch.
///
/// This lives on [`Noise`] rather than in an `Fbm { source }` node on
/// purpose. The baker evaluates each layer once per pixel over a fixed
/// domain, so a downstream node cannot re-sample its input at a different
/// scale — the same constraint already documented on `bake_volume`.
/// Octaves have to be where the coordinates are still live.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Fractal {
    /// `1..=`[`crate::noise::MAX_OCTAVES`]. 1 is a plain single-octave
    /// sample, and is exactly the bare kernel.
    pub octaves: u32,
    /// Frequency multiplier per octave. A float, not a shift: 2.13 and 2.4
    /// are as common as 2.0 in hand-tuned fbm. Only an integral value
    /// keeps a `period` tiling exactly — see [`Noise::period`].
    pub lacunarity: f32,
    /// Amplitude multiplier per octave.
    pub gain: f32,
    pub mode: FractalMode,
    /// Divide by the sum of amplitudes so the result stays in range.
    pub normalize: bool,
}

impl Default for Fractal {
    /// One octave: the plain kernel, and what a file written before
    /// fractals existed means.
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
    /// `Σ aᵢ · n(fᵢx)` — ordinary fbm.
    #[default]
    Standard,
    /// `Σ aᵢ · |n|` — creased at every zero crossing.
    Turbulence,
    /// `Σ aᵢ · (1 - |2n - 1|)²` — filaments, as a star corona wants.
    Ridged,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum NoiseDims { D1, D2, D3 }

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum NoiseRange {
    /// `[0, 1]` grayscale intent.
    Unsigned,
    /// `[-1, 1]` — composes into fractal noise via `Mix::Add`.
    Signed,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum NoiseOutput {
    /// Single noise sample drives L; C=0.
    Grayscale,
    /// Three independent noise samples drive L, C, hue directly in Oklch.
    Color,
}

/// Multi-stop 1D color ramp, sampled along U; a [`Transform`] runs it along
/// V or radially. The graph keeps stops sorted by `t` on mutation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ColorRamp {
    pub stops: Vec<ColorStop>,
    pub space: BlendSpace,
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct ColorStop {
    pub t: f32,
    pub color: ColorInput,
}

/// Pre-transforms the sample point before evaluating `source`. Order of
/// operations: subtract `offset` (so `offset` acts as a center for rotate /
/// radial), rotate in UV by `rotate_uv`, multiply by `scale`, then apply
/// `coord_mode`.
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct Transform {
    /// `None` renders as the missing-texture grid.
    pub source: Option<LayerId>,
    pub offset: [f32; 3],
    /// Radians, in the UV plane, about the origin (after `offset` subtracts).
    pub rotate_uv: f32,
    pub scale: [f32; 3],
    pub coord_mode: CoordMode,
    /// What happens when the transformed sample leaves the UV square.
    /// Defaults to `Clamp` where a file doesn't say.
    #[serde(default)]
    pub edge_mode: EdgeMode,
}

/// Out-of-[0,1] sampling policy for [`Transform`], on the U/V axes.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum EdgeMode {
    /// Transformed U/V clamp to the [0, 1] edge.
    #[default]
    Clamp,
    /// Sample beyond [0, 1]: procedural sources (noise etc.) extend
    /// naturally. Affine mappings (passthrough/permute) extend without
    /// limit — the sampled region is known exactly, so sources are baked
    /// over it. Radial mappings are bounded by
    /// [`EXTEND_LIMIT`](crate::EXTEND_LIMIT); beyond that — or where a
    /// baked source can't cover the request — the missing-texture grid
    /// shows.
    Extend,
}

/// How far a *radial* [`EdgeMode::Extend`] transform may sample from the
/// unit square's center (0.5) on each of U and V: coordinates within
/// `[0.5 - EXTEND_LIMIT, 0.5 + EXTEND_LIMIT]` are valid, anything outside
/// renders as the missing-texture grid. Affine (passthrough/permute)
/// extends are exact and unbounded.
pub const EXTEND_LIMIT: f32 = 8.0;

/// Final axis remapping applied after affine transform.
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub enum CoordMode {
    /// Pass `(u, v, w)` to `source` as-is.
    Passthrough,
    /// Feed source with axes reordered. `[V, U, W]` swaps u and v.
    Permute([Axis; 3]),
    /// Reduce the sample to a scalar radius and feed it into `into`; the
    /// other two axes are set to 0.
    Radial { dim: RadialDim, into: Axis },
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum Axis { U, V, W }

/// Which coordinates contribute to the radial distance.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum RadialDim {
    /// `sqrt(u² + v²)` — planar distance from origin.
    D2,
    /// `sqrt(u² + v² + w²)` — spatial distance from origin.
    D3,
}

/// Blend two color layers.
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct Mix {
    /// `None` renders as the missing-texture grid.
    pub a: Option<LayerId>,
    pub b: Option<LayerId>,
    pub mode: BlendMode,
    /// Only consulted for `BlendMode::Blend`.
    pub factor: ScalarInput,
    /// Space in which `Blend` interpolates; ignored by the arithmetic modes.
    pub space: BlendSpace,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum BlendMode {
    Add,
    Subtract,
    Multiply,
    Blend,
}

/// Gradient-map: take L from `value` at the current sample, then look up
/// `palette` at `(t, 0, 0)`.
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct Map {
    /// `None` renders as the missing-texture grid.
    pub value: Option<LayerId>,
    pub palette: Option<LayerId>,
}

/// Convert a heightfield (from `source`'s L channel) into a tangent-space
/// normal map. `strength` scales the derivative before normalization; larger
/// values produce steeper apparent slopes.
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct HeightToNormal {
    /// `None` renders as the missing-texture grid.
    pub source: Option<LayerId>,
    pub strength: f32,
}

/// A periodic waveform over a scalar input — what `sin(...)` is, and what
/// a ColorRamp with enough stops to fake one is not.
///
/// `input` is read as a scalar (a layer's Oklch L), scaled by `frequency`,
/// shifted by `phase`, and shaped. The output drives L with C = 0, the
/// same as grayscale [`Noise`].
///
/// **Outside the noise op-order contract.** [`WaveShape::Sine`] is
/// transcendental, so the CPU and GPU implementations agree to within one
/// sRGB step rather than bit-exactly, and the parity test asserts the
/// looser bound rather than pretending otherwise.
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct Wave {
    /// Scalar to run the wave over. A `Const` makes the whole layer one
    /// flat value, which is valid and rarely what anybody wants.
    pub input: ScalarInput,
    pub shape: WaveShape,
    /// Cycles per unit of `input`, matching [`Noise::frequency`]'s unit.
    /// A hand-written shader's `sin(x * k)` counts radians, so it ports as
    /// `k / 2π` cycles — `sin(x * 18)` is `frequency: 2.8648`.
    pub frequency: f32,
    /// Offset in cycles: `0.25` is a quarter turn, `1.0` is no shift at
    /// all. In cycles rather than radians so the UI never shows a π.
    pub phase: f32,
    /// `Signed` is `[-1, 1]`, which composes with `Mix::Add`; `Unsigned`
    /// is `[0, 1]`, which is what bands want.
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

/// Waveform shape. Every one is described over the phase `t ∈ [0, 1)`
/// within a cycle and lands in `[-1, 1]` before [`Wave::range`] maps it.
///
/// Sine alone covers the known need; the other three came nearly free
/// once the node existed.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum WaveShape {
    /// `sin(2πt)`.
    #[default]
    Sine,
    /// Phase-aligned with `Sine`: 0 at `t = 0`, peaking at `t = 0.25`.
    Triangle,
    /// `+1` for the first half of the cycle, `-1` for the second — the
    /// sign of `Sine`.
    Square,
    /// A rising ramp from `-1` to `+1` across the cycle, resetting at each
    /// period. Not phase-aligned with `Sine`; a sawtooth that started
    /// mid-ramp would be the surprising one.
    Sawtooth,
}

/// Per-pixel winner-take-all between two color layers. Whichever pixel
/// wins on the chosen `criterion` under `mode`, its full Oklcha value
/// (all four channels) flows to the output — useful for salvaging a
/// specific attribute from noise while keeping the rest of that pixel
/// coherent.
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

/// Which attribute of each pixel to compare. RGB / Saturation / Value /
/// Luma are computed in gamma-encoded sRGB (what a monitor shows), so
/// "brightest red" matches naïve intuition. Alpha and Chroma are read
/// straight off the Oklcha value.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum Criterion {
    Red,
    Green,
    Blue,
    Saturation,
    Value,
    /// Rec.709 luma in gamma sRGB: `0.2126R + 0.7152G + 0.0722B`.
    Luma,
    Alpha,
    /// Oklch chroma (colorfulness). Perceptually uniform.
    Chroma,
}
