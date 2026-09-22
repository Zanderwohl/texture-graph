//! Named parameters a graph exposes to whoever bakes it.
//!
//! A [`ParamDecl`] travels with the graph, a [`ParamValue`] is supplied per
//! bake, and any `ScalarInput` or `ColorInput` can read one by name. See
//! `documentation/game-consumer-features.md`, section 6.
//!
//! Only `ScalarInput` and `ColorInput` have a `Param` variant, so the
//! bindable sockets are `Mix::factor`, `Wave::input`, the Output's roughness
//! and metallic, and ColorRamp stops. A `Color` layer holds a plain `Color`
//! and cannot take a param: changing its type would break saved graphs,
//! since RON writes the variant name.
//!
//! [`crate::Graph`] checks names at edit time, rejecting an undeclared name
//! or a param of the wrong kind, so the evaluator never sees one.

use serde::{Deserialize, Serialize};

use crate::color::Color;

/// One parameter a graph exposes. `name` duplicates the map key in
/// [`crate::Graph::params`] so a decl is self-describing on its own.
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
    /// `min..=max` is advisory; see [`ParamKind`].
    pub fn scalar(name: impl Into<String>, min: f32, max: f32, default: f32) -> Self {
        Self {
            name: name.into(),
            kind: ParamKind::Scalar { min, max },
            default: ParamValue::Scalar(default),
            description: None,
        }
    }

    pub fn color(name: impl Into<String>, default: Color) -> Self {
        Self {
            name: name.into(),
            kind: ParamKind::Color,
            default: ParamValue::Color(default),
            description: None,
        }
    }
}

/// For a scalar, the range a UI should offer. Nothing clamps to it.
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
    /// A decl whose default fails this is rejected at declaration.
    pub fn matches(&self, kind: &ParamKind) -> bool {
        matches!(
            (self, kind),
            (ParamValue::Scalar(_), ParamKind::Scalar { .. })
                | (ParamValue::Color(_), ParamKind::Color)
        )
    }

    pub fn as_scalar(&self) -> Option<f32> {
        match self {
            ParamValue::Scalar(v) => Some(*v),
            ParamValue::Color(_) => None,
        }
    }

    pub fn as_color(&self) -> Option<Color> {
        match self {
            ParamValue::Color(c) => Some(*c),
            ParamValue::Scalar(_) => None,
        }
    }
}

/// Which sort of socket reads a parameter, as reported by
/// [`crate::LayerKind::param_refs`].
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ParamUse {
    Scalar,
    Color,
}

impl ParamUse {
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
