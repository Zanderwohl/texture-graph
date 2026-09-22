//! Where each layer of a sphere bake is evaluated.
//!
//! On a cube face a texel's (u, v) is not a position in the field, so a
//! layer that re-samples its input at another (u, v) cannot read the face
//! its input baked. Two such kinds still have a meaning on a sphere, and are
//! planned around rather than refused:
//!
//! - A [`Map`]'s palette is looked up by value, not by position, so it bakes
//!   on a plane beside the faces: a [`ColorRamp`] there is the ramp it is in
//!   a flat bake.
//! - A passthrough [`Transform`] moves the sample point by an affine map, so
//!   the map is pushed down to the generators beneath it, which evaluate at
//!   the moved point. The Transform itself becomes a copy.
//!
//! [`Map`]: texture_graph_core::Map
//! [`ColorRamp`]: texture_graph_core::ColorRamp
//! [`Transform`]: texture_graph_core::Transform

use std::collections::HashMap;

use texture_graph_core::{CoordMode, EdgeMode, Graph, LayerId, LayerKind, Transform};

use crate::BakeError;

/// Rows of an affine map of the sample point: `p' = [row · (p, 1)]`.
pub(crate) type PointMap = [[f32; 4]; 3];

pub(crate) const IDENTITY: PointMap = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
];

#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) enum Placement {
    /// On the cube face, at the sample point moved by the map.
    Sphere(PointMap),
    /// On the plane a flat bake uses.
    Plane,
}

/// Every layer `root` depends on, and where it bakes.
pub(crate) fn plan(graph: &Graph, root: LayerId) -> Result<HashMap<LayerId, Placement>, BakeError> {
    let mut placed = HashMap::new();
    let mut stack = vec![(root, Placement::Sphere(IDENTITY))];
    while let Some((id, at)) = stack.pop() {
        if let Some(&before) = placed.get(&id) {
            if before != at {
                return Err(BakeError::Unsupported(
                    "a layer a sphere bake needs at two placements",
                ));
            }
            continue;
        }
        placed.insert(id, at);
        let Some(layer) = graph.get(id) else { continue };
        let Placement::Sphere(map) = at else {
            stack.extend(layer.kind.inputs().into_iter().map(|i| (i, Placement::Plane)));
            continue;
        };
        match &layer.kind {
            LayerKind::Color(_)
            | LayerKind::Noise(_)
            | LayerKind::Coordinate(_)
            | LayerKind::Mix(_)
            | LayerKind::MinMax(_)
            | LayerKind::Wave(_) => {
                stack.extend(layer.kind.inputs().into_iter().map(|i| (i, at)));
            }
            LayerKind::Map(m) => {
                stack.extend(m.value.map(|i| (i, at)));
                stack.extend(m.palette.map(|i| (i, Placement::Plane)));
            }
            LayerKind::Transform(t) => {
                if !matches!(t.coord_mode, CoordMode::Passthrough) {
                    return Err(BakeError::Unsupported(
                        "a Transform other than passthrough in a sphere bake",
                    ));
                }
                if t.edge_mode != EdgeMode::Extend {
                    return Err(BakeError::Unsupported("a clamping Transform in a sphere bake"));
                }
                let moved = Placement::Sphere(compose(&transform_map(t), &map));
                stack.extend(t.source.map(|i| (i, moved)));
            }
            LayerKind::ColorRamp(_) => {
                return Err(BakeError::Unsupported(
                    "ColorRamp in a sphere bake, other than as a Map's palette",
                ));
            }
            LayerKind::HeightToNormal(_) => {
                return Err(BakeError::Unsupported("HeightToNormal in a sphere bake"));
            }
            LayerKind::Warp(_) => return Err(BakeError::Unsupported("Warp in a sphere bake")),
        }
    }
    Ok(placed)
}

/// `apply_transform` in `core::eval`, for [`CoordMode::Passthrough`]:
/// subtract the offset, rotate in (u, v), scale.
fn transform_map(t: &Transform) -> PointMap {
    let (s, c) = t.rotate_uv.sin_cos();
    let [ou, ov, ow] = t.offset;
    let [su, sv, sw] = t.scale;
    [
        [su * c, -su * s, 0.0, -su * (c * ou - s * ov)],
        [sv * s, sv * c, 0.0, -sv * (s * ou + c * ov)],
        [0.0, 0.0, sw, -sw * ow],
    ]
}

/// `outer` after `inner`.
fn compose(outer: &PointMap, inner: &PointMap) -> PointMap {
    let mut out = [[0.0; 4]; 3];
    for (r, row) in out.iter_mut().enumerate() {
        for (k, cell) in row.iter_mut().enumerate() {
            *cell = (0..3).map(|j| outer[r][j] * inner[j][k]).sum::<f32>();
        }
        row[3] += outer[r][3];
    }
    out
}

#[cfg(test)]
mod tests {
    use texture_graph_core::{Axis, Coordinate, EvalCtx, Sample, eval};

    use super::*;

    fn apply(m: &PointMap, p: [f32; 3]) -> [f32; 3] {
        m.map(|r| r[0] * p[0] + r[1] * p[1] + r[2] * p[2] + r[3])
    }

    fn transform(offset: [f32; 3], rotate_uv: f32, scale: [f32; 3]) -> Transform {
        Transform {
            source: None,
            offset,
            rotate_uv,
            scale,
            coord_mode: CoordMode::Passthrough,
            edge_mode: EdgeMode::Extend,
        }
    }

    /// Read through a Coordinate, which is the sample point itself.
    #[test]
    fn the_map_is_the_cpus_transform() {
        let t = transform([0.2, 0.5, -0.1], 0.7, [1.3, 1.8, 0.6]);
        let p = [0.31, 0.77, 0.12];
        let got = apply(&transform_map(&t), p);
        for (axis, got) in [Axis::U, Axis::V, Axis::W].into_iter().zip(got) {
            let mut g = Graph::new();
            let c = g.add_layer("c", LayerKind::Coordinate(Coordinate { axis })).unwrap();
            let moved = g
                .add_layer("t", LayerKind::Transform(Transform { source: Some(c), ..t }))
                .unwrap();
            let want = eval::evaluate(&g, moved, Sample::new(p[0], p[1], p[2]), &EvalCtx::default()).l;
            assert!((got - want).abs() < 1e-5, "{axis:?}: {got} against {want}");
        }
    }

    /// The inner transform is the one nearer the root, so it moves the point
    /// first.
    #[test]
    fn composition_applies_the_inner_map_first() {
        let inner = transform_map(&transform([0.5, 0.0, 0.0], 0.0, [2.0, 1.0, 1.0]));
        let outer = transform_map(&transform([0.0, 0.0, 0.0], 0.3, [1.0, 1.0, 1.0]));
        let p = [0.9, 0.4, 0.2];
        let got = apply(&compose(&outer, &inner), p);
        let want = apply(&outer, apply(&inner, p));
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < 1e-6, "{got:?} against {want:?}");
        }
    }
}
