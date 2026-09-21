//! Named parameters: the knobs a graph exposes to whoever bakes it.
//!
//! Six planet-surface classes that differ only by palette and contrast are
//! six near-identical graphs, or one graph and a small block of values.
//! A [`ParamDecl`] travels with the graph, a [`ParamValue`] is supplied per
//! bake, and any `ScalarInput` or `ColorInput` can read one by name.
//!
//! This is also what makes a graph worth transmitting: the wire carries
//! the graph once and per-instance variation is a handful of numbers.
//!
//! # What binds, and what does not
//!
//! Only `ScalarInput` and `ColorInput` gained a `Param` variant, so the
//! reachable sockets are `Mix::factor`, `Wave::input`, the Output's
//! roughness and metallic, and a ColorRamp's stops. A `Color` *layer*
//! holds a plain `Color` rather than a `ColorInput` and cannot take a
//! param — changing it would break every graph already on disk, since RON
//! writes the variant name. A palette parameterised through ramp stops is
//! the intended shape anyway.
//!
//! # Where a name is checked
//!
//! At edit time, like a `LayerId`: [`crate::Graph`] rejects a kind that
//! reads an undeclared name, or reads a Color param where a scalar
//! belongs. The evaluator and the baker therefore never have to decide
//! what an unknown name means.

use serde::{Deserialize, Serialize};

use crate::color::Color;

/// One parameter a graph exposes. The `name` is also the map key in
/// [`crate::Graph::params`]; it is stored here too so a decl is
/// self-describing when it travels alone (a host building a UI, say).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParamDecl {
    pub name: String,
    pub kind: ParamKind,
    /// Used for any bake that does not bind this parameter.
    pub default: ParamValue,
    /// Free-form, for a host that surfaces the parameter to a person.
    pub description: Option<String>,
}

impl ParamDecl {
    /// A scalar parameter over `min..=max`, defaulting to `default`.
    pub fn scalar(name: impl Into<String>, min: f32, max: f32, default: f32) -> Self {
        Self {
            name: name.into(),
            kind: ParamKind::Scalar { min, max },
            default: ParamValue::Scalar(default),
            description: None,
        }
    }

    /// A color parameter.
    pub fn color(name: impl Into<String>, default: Color) -> Self {
        Self {
            name: name.into(),
            kind: ParamKind::Color,
            default: ParamValue::Color(default),
            description: None,
        }
    }
}

/// What kind of value a parameter carries, and — for a scalar — the range
/// a UI should offer. The range is advisory: nothing clamps to it, the
/// same way `Noise::frequency`'s slider bounds do not clamp the field.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ParamKind {
    Scalar { min: f32, max: f32 },
    Color,
}

impl ParamKind {
    pub fn label(&self) -> &'static str {
        match self {
            ParamKind::Scalar { .. } => "scalar",
            ParamKind::Color => "color",
        }
    }
}

/// A bound parameter value.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ParamValue {
    Scalar(f32),
    Color(Color),
}

impl ParamValue {
    /// True when this value is the sort `kind` describes. A decl whose
    /// default disagrees with its own kind is rejected at declaration, so
    /// nothing downstream has to cope with one.
    pub fn matches(&self, kind: &ParamKind) -> bool {
        matches!(
            (self, kind),
            (ParamValue::Scalar(_), ParamKind::Scalar { .. })
                | (ParamValue::Color(_), ParamKind::Color)
        )
    }

    /// The scalar this value carries, or `None` if it is a color.
    pub fn as_scalar(&self) -> Option<f32> {
        match self {
            ParamValue::Scalar(v) => Some(*v),
            ParamValue::Color(_) => None,
        }
    }

    /// The color this value carries, or `None` if it is a scalar.
    pub fn as_color(&self) -> Option<Color> {
        match self {
            ParamValue::Color(c) => Some(*c),
            ParamValue::Scalar(_) => None,
        }
    }
}

/// Which sort of socket read a parameter — what
/// [`crate::LayerKind::param_refs`] reports, so the graph can check a name
/// against its declared kind.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ParamUse {
    Scalar,
    Color,
}

impl ParamUse {
    /// True when a parameter of `kind` can be read by this sort of socket.
    pub fn accepts(self, kind: &ParamKind) -> bool {
        matches!(
            (self, kind),
            (ParamUse::Scalar, ParamKind::Scalar { .. }) | (ParamUse::Color, ParamKind::Color)
        )
    }

    pub fn label(self) -> &'static str {
        match self {
            ParamUse::Scalar => "scalar",
            ParamUse::Color => "color",
        }
    }
}
