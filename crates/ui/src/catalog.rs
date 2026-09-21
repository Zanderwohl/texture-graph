//! The node catalog the editor offers: what the add-node menu and the
//! variant switcher list, in what order, and what a new node starts as.
//!
//! Separate from [`texture_graph_core::LayerKind`] on purpose. That type
//! says what a node *is*; this one says what the menu calls it and what it
//! looks like untouched, which is an editor question — a Noise node at
//! frequency 4.0 is a nicer first impression than one at 0.0, and neither
//! is more correct.

use texture_graph_core::color::oklcha;
use texture_graph_core::{
    BlendMode, BlendSpace, ColorInput, ColorRamp, ColorStop, CoordMode, Criterion, EdgeMode,
    Graph, HeightToNormal, LayerKind, Map, MinMax, MinMaxMode, Mix, Noise, ScalarInput,
    Transform, Wave,
};

/// One entry in the add-node menu and the variant switcher.
pub struct Variant {
    /// Menu text.
    pub label: &'static str,
    /// Matched instead of the label, so editing menu text can't change what
    /// an entry adds.
    pub kind: Kind,
}

/// The node an entry adds, independent of what the menu calls it.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Kind {
    Color,
    Noise,
    ColorRamp,
    Transform,
    Mix,
    Map,
    MinMax,
    HeightToNormal,
    Wave,
}

impl Kind {
    /// Exhaustive over [`LayerKind`], so a new kind in core won't compile
    /// without an answer here, and
    /// [`tests::every_kind_is_offered_in_the_menu`] then forces it into
    /// [`VARIANTS`].
    pub fn of(kind: &LayerKind) -> Kind {
        match kind {
            LayerKind::Color(_) => Kind::Color,
            LayerKind::Noise(_) => Kind::Noise,
            LayerKind::ColorRamp(_) => Kind::ColorRamp,
            LayerKind::Transform(_) => Kind::Transform,
            LayerKind::Mix(_) => Kind::Mix,
            LayerKind::Map(_) => Kind::Map,
            LayerKind::MinMax(_) => Kind::MinMax,
            LayerKind::HeightToNormal(_) => Kind::HeightToNormal,
            LayerKind::Wave(_) => Kind::Wave,
        }
    }
}

/// Every node the editor offers, in menu order.
///
/// Flat rather than grouped: nine entries still fit in one menu, and four
/// submenus of two would cost a hover and a pointer trip to reach any of
/// them. Group them when the catalog outgrows a single list.
pub const VARIANTS: &[Variant] = &[
    Variant { label: "Color", kind: Kind::Color },
    Variant { label: "Noise", kind: Kind::Noise },
    Variant { label: "ColorRamp", kind: Kind::ColorRamp },
    Variant { label: "Transform", kind: Kind::Transform },
    Variant { label: "Mix", kind: Kind::Mix },
    Variant { label: "Map", kind: Kind::Map },
    Variant { label: "MinMax", kind: Kind::MinMax },
    Variant { label: "HeightToNormal", kind: Kind::HeightToNormal },
    Variant { label: "Wave", kind: Kind::Wave },
];

/// What a node of this variant looks like the moment it is added.
///
/// Layer inputs start unconnected (`None`) — they render as the
/// missing-texture grid until the user picks a source, and can never trip
/// the cycle check on creation.
pub fn default_kind(kind: Kind) -> LayerKind {
    match kind {
        Kind::Color => LayerKind::Color(oklcha(0.5, 0.0, 0.0, 1.0)),
        Kind::Noise => LayerKind::Noise(Noise::default()),
        Kind::ColorRamp => LayerKind::ColorRamp(ColorRamp {
            stops: vec![
                ColorStop { t: 0.0, color: ColorInput::Const(oklcha(0.0, 0.0, 0.0, 1.0)) },
                ColorStop { t: 1.0, color: ColorInput::Const(oklcha(1.0, 0.0, 0.0, 1.0)) },
            ],
            space: BlendSpace::Oklch,
        }),
        Kind::Transform => LayerKind::Transform(Transform {
            source: None,
            offset: [0.5, 0.5, 0.5],
            rotate_uv: 0.0,
            scale: [1.0; 3],
            coord_mode: CoordMode::Passthrough,
            edge_mode: EdgeMode::default(),
        }),
        Kind::Mix => LayerKind::Mix(Mix {
            a: None,
            b: None,
            mode: BlendMode::Blend,
            factor: ScalarInput::Const(0.5),
            space: BlendSpace::Oklch,
        }),
        Kind::Map => LayerKind::Map(Map { value: None, palette: None }),
        Kind::MinMax => LayerKind::MinMax(MinMax {
            a: None,
            b: None,
            mode: MinMaxMode::Max,
            criterion: Criterion::Alpha,
        }),
        Kind::HeightToNormal => {
            LayerKind::HeightToNormal(HeightToNormal { source: None, strength: 1.0 })
        }
        Kind::Wave => LayerKind::Wave(Wave::default()),
    }
}

/// First name in `base`, `base 1`, `base 2`, … not taken by any layer.
pub fn unique_name(graph: &Graph, base: &str) -> String {
    if !graph.layers.iter().any(|l| l.name == base) {
        return base.to_string();
    }
    for n in 1..u32::MAX {
        let candidate = format!("{base} {n}");
        if !graph.layers.iter().any(|l| l.name == candidate) {
            return candidate;
        }
    }
    base.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A default built for the wrong variant is invisible: the menu entry
    /// says one thing and adds another, and the inspector then shows the
    /// switcher pointing somewhere the user did not click.
    #[test]
    fn every_default_is_the_variant_it_was_asked_for() {
        for v in VARIANTS {
            assert_eq!(
                Kind::of(&default_kind(v.kind)),
                v.kind,
                "{} builds the wrong node",
                v.label
            );
        }
    }

    /// `Kind::of` is exhaustive over `LayerKind`, so the compiler catches a
    /// new node kind that has no `Kind`. Nothing catches one that has a
    /// `Kind` but never made it into the menu — it would simply be
    /// unaddable, with no error anywhere.
    #[test]
    fn every_kind_is_offered_in_the_menu() {
        const ALL: &[Kind] = &[
            Kind::Color,
            Kind::Noise,
            Kind::ColorRamp,
            Kind::Transform,
            Kind::Mix,
            Kind::Map,
            Kind::MinMax,
            Kind::HeightToNormal,
            Kind::Wave,
        ];
        for kind in ALL {
            assert!(
                VARIANTS.iter().any(|v| v.kind == *kind),
                "{kind:?} exists but no menu entry adds it"
            );
        }
        assert_eq!(VARIANTS.len(), ALL.len(), "a variant is listed twice");
    }

    /// Every default has to be addable, or the menu offers nodes the graph
    /// refuses.
    #[test]
    fn every_default_can_actually_be_added() {
        let mut graph = Graph::new();
        for v in VARIANTS {
            let name = unique_name(&graph, v.label);
            graph
                .add_layer(name, default_kind(v.kind))
                .unwrap_or_else(|e| panic!("{} could not be added: {e}", v.label));
        }
    }

    #[test]
    fn unique_name_walks_past_what_is_taken() {
        let mut graph = Graph::new();
        assert_eq!(unique_name(&graph, "noise"), "noise");
        graph.add_layer("noise", default_kind(Kind::Noise)).unwrap();
        assert_eq!(unique_name(&graph, "noise"), "noise 1");
        graph.add_layer("noise 1", default_kind(Kind::Noise)).unwrap();
        assert_eq!(unique_name(&graph, "noise"), "noise 2");
    }
}
