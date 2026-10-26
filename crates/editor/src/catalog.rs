//! The node catalog: what the add-node menu and the variant switcher list,
//! in what order, and what a new node starts as.
//!
//! Kept apart from [`texture_graph_core::LayerKind`] because menu labels and
//! starting values are editor choices, not properties of the node.

use texture_graph_core::color::oklcha;
use texture_graph_core::{
    BlendMode, BlendSpace, ColorInput, ColorRamp, ColorStop, CoordMode, Coordinate, Craters,
    Criterion,
    EdgeMode,
    Graph, HeightToNormal, LayerKind, Map, MinMax, MinMaxMode, Mix, Noise, ScalarInput,
    Transform, Warp, Wave,
};

/// One entry in the add-node menu and the variant switcher.
pub struct Variant {
    pub label: &'static str,
    /// Matched instead of the label, so editing menu text can't change what
    /// an entry adds.
    pub kind: Kind,
}

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
    Warp,
    Coordinate,
    Craters,
}

impl Kind {
    /// Exhaustive over [`LayerKind`] so a new kind in core must be added
    /// here; [`tests::every_kind_is_offered_in_the_menu`] then forces it
    /// into [`VARIANTS`].
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
            LayerKind::Warp(_) => Kind::Warp,
            LayerKind::Coordinate(_) => Kind::Coordinate,
            LayerKind::Craters(_) => Kind::Craters,
        }
    }
}

/// Menu order. Flat rather than grouped while the list fits in one menu,
/// since submenus cost an extra hover per entry.
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
    Variant { label: "Warp", kind: Kind::Warp },
    Variant { label: "Coordinate", kind: Kind::Coordinate },
    Variant { label: "Craters", kind: Kind::Craters },
];

/// Layer inputs start unconnected so creation can never trip the cycle
/// check.
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
        Kind::Warp => LayerKind::Warp(Warp::default()),
        Kind::Coordinate => LayerKind::Coordinate(Coordinate::default()),
        Kind::Craters => LayerKind::Craters(Craters::default()),
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

    /// Otherwise a menu entry silently adds a different node than it names.
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

    /// A `Kind` missing from `VARIANTS` would be unaddable, with no error.
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
            Kind::Warp,
            Kind::Coordinate,
            Kind::Craters,
        ];
        for kind in ALL {
            assert!(
                VARIANTS.iter().any(|v| v.kind == *kind),
                "{kind:?} exists but no menu entry adds it"
            );
        }
        assert_eq!(VARIANTS.len(), ALL.len(), "a variant is listed twice");
    }

    /// Otherwise the menu offers nodes the graph refuses.
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
