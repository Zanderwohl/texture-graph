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
        }
    }

    /// Every layer this node depends on, for cycle detection and
    /// invalidation.
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
            LayerKind::Transform(t) => out.push(t.source),
            LayerKind::Mix(m) => {
                out.push(m.a);
                out.push(m.b);
                push_scalar(&mut out, m.factor);
            }
            LayerKind::Map(m) => {
                out.push(m.value);
                out.push(m.palette);
            }
            LayerKind::MinMax(mm) => {
                out.push(mm.a);
                out.push(mm.b);
            }
            LayerKind::HeightToNormal(h) => out.push(h.source),
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

/// Simplex noise at a chosen dimensionality, with a chosen numeric range
/// and either grayscale or LCh-colorful output.
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct Noise {
    pub dims: NoiseDims,
    pub seed_offset: u32,
    /// Cycles per unit sample-space; `1.0` = one feature per full UVW cube.
    pub frequency: f32,
    pub range: NoiseRange,
    pub output: NoiseOutput,
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

/// Multi-stop 1D color ramp. Sampled along the U axis; feed through a
/// [`Transform`] to run it along V or radially. Stops SHOULD be sorted by
/// `t` ascending; the graph enforces this on mutation.
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
    pub source: LayerId,
    pub offset: [f32; 3],
    /// Radians, in the UV plane, about the origin (after `offset` subtracts).
    pub rotate_uv: f32,
    pub scale: [f32; 3],
    pub coord_mode: CoordMode,
}

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
    pub a: LayerId,
    pub b: LayerId,
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
    pub value: LayerId,
    pub palette: LayerId,
}

/// Convert a heightfield (from `source`'s L channel) into a tangent-space
/// normal map. `strength` scales the derivative before normalization; larger
/// values produce steeper apparent slopes.
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct HeightToNormal {
    pub source: LayerId,
    pub strength: f32,
}

/// Per-pixel winner-take-all between two color layers. Whichever pixel
/// wins on the chosen `criterion` under `mode`, its full Oklcha value
/// (all four channels) flows to the output — useful for salvaging a
/// specific attribute from noise while keeping the rest of that pixel
/// coherent.
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct MinMax {
    pub a: LayerId,
    pub b: LayerId,
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
