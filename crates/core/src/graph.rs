use std::collections::{BTreeMap, HashMap, HashSet};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::color::Color;
use crate::id::LayerId;
use crate::kind::{ColorRamp, LayerKind, NoiseKernel, ScalarInput};

/// Named node in the graph. Canvas position lives in [`Graph::canvases`], so
/// identity, display and layout stay independent.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Layer {
    pub id: LayerId,
    pub name: String,
    pub kind: LayerKind,
}

/// The graph's root, producing PBR-material channels for the renderer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Output {
    /// Layer whose color is the material's base color. `None` renders as
    /// the missing-texture grid.
    pub color: Option<LayerId>,
    /// 0 = smooth, 1 = rough.
    pub roughness: ScalarInput,
    /// 0 = dielectric, 1 = metal.
    pub metallic: ScalarInput,
    /// Tangent-space normal, XYZ encoded via
    /// [`crate::color::normal_to_color`]. `None` = flat normal.
    pub normal: Option<LayerId>,
}

impl Output {
    fn referenced(&self) -> Vec<LayerId> {
        let mut out = Vec::new();
        out.extend(self.color);
        if let ScalarInput::Layer(id) = self.roughness {
            out.push(id);
        }
        if let ScalarInput::Layer(id) = self.metallic {
            out.push(id);
        }
        if let Some(id) = self.normal {
            out.push(id);
        }
        out
    }
}

/// One named workspace canvas: a scatter of layer positions. Edges come from
/// [`LayerKind::inputs`] at render time and are not stored.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Canvas {
    /// Layers absent from the map are unplaced, and the UI picks a spot.
    pub positions: BTreeMap<LayerId, [f32; 2]>,
    /// The Output pseudo-node. `None` is unplaced, and also what a file
    /// without the field loads as.
    #[serde(default)]
    pub output_pos: Option<[f32; 2]>,
}

/// The full graph. Three orthogonal shapes travel with the data:
///
/// - **Identity** — [`Layer::id`], never reused. Layers stay ID-ascending so
///   serialization is stable however the UI reorders them.
/// - **List order** — [`Graph::list_order`], for the linear panel view.
/// - **Canvas positions** — [`Graph::canvases`], one scatter per workspace.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Graph {
    /// ID-ascending. Do not reorder for display; use `list_order` instead.
    pub layers: Vec<Layer>,
    /// Permutation of layer IDs for the linear list view.
    pub list_order: Vec<LayerId>,
    /// Named workspace canvases. Empty by default.
    pub canvases: BTreeMap<String, Canvas>,
    pub output: Output,
    next_id: u64,
}

#[derive(Debug, Error)]
pub enum GraphError {
    #[error("unknown layer id {0}")]
    UnknownId(LayerId),
    #[error("duplicate layer name: {0}")]
    DuplicateName(String),
    #[error("setting this input would create a cycle at {0}")]
    Cycle(LayerId),
    #[error("color ramp must have at least two stops")]
    RampTooFewStops,
    #[error("a periodic lattice needs the Value kernel; simplex has none to wrap")]
    PeriodicSimplex,
    #[error("canvas {0:?} does not exist")]
    UnknownCanvas(String),
    #[error("canvas {0:?} already exists")]
    DuplicateCanvas(String),
}

impl Graph {
    /// Empty graph with a placeholder Color layer as the output.
    pub fn new() -> Self {
        let mut g = Self {
            layers: Vec::new(),
            list_order: Vec::new(),
            canvases: BTreeMap::new(),
            output: Output {
                color: None,
                roughness: ScalarInput::Const(0.5),
                metallic: ScalarInput::Const(0.0),
                normal: None,
            },
            next_id: 1,
        };
        let id = g
            .add_layer("base color", LayerKind::Color(Color::new(0.5, 0.0, 0.0, 1.0)))
            .expect("first add cannot fail");
        g.output.color = Some(id);
        g
    }

    // ---- Lookup ---------------------------------------------------------

    pub fn get(&self, id: LayerId) -> Option<&Layer> {
        // Layers are ID-sorted, so binary search.
        self.layers
            .binary_search_by_key(&id, |l| l.id)
            .ok()
            .map(|i| &self.layers[i])
    }

    pub fn get_mut(&mut self, id: LayerId) -> Option<&mut Layer> {
        self.layers
            .binary_search_by_key(&id, |l| l.id)
            .ok()
            .map(|i| &mut self.layers[i])
    }

    pub fn contains(&self, id: LayerId) -> bool {
        self.layers.binary_search_by_key(&id, |l| l.id).is_ok()
    }

    /// True if `id` is reachable from the graph's Output (i.e. actually
    /// contributes to the final material).
    pub fn is_reachable(&self, id: LayerId) -> bool {
        let mut stack: Vec<LayerId> = self.output.referenced();
        let mut seen: HashSet<LayerId> = HashSet::new();
        while let Some(cur) = stack.pop() {
            if cur == id {
                return true;
            }
            if !seen.insert(cur) {
                continue;
            }
            if let Some(l) = self.get(cur) {
                stack.extend(l.kind.inputs());
            }
        }
        false
    }

    /// True when any layer reachable from the Output actually varies along
    /// the third (w) texture coordinate — i.e. the graph describes a solid
    /// 3D texture rather than a flat image. The 3D preview uses this to
    /// switch from UV mapping to volume ("solid") sampling.
    ///
    /// Detected sources of w-variation:
    /// - `Noise` with `dims == D3`
    /// - `Transform` that routes w into the sampled plane
    ///   (a `Permute` involving `Axis::W`, or a 3D `Radial`)
    pub fn output_is_3d(&self) -> bool {
        use crate::kind::{Axis, CoordMode, NoiseDims, RadialDim};
        let mut stack: Vec<LayerId> = self.output.referenced();
        let mut seen: HashSet<LayerId> = HashSet::new();
        while let Some(cur) = stack.pop() {
            if !seen.insert(cur) {
                continue;
            }
            let Some(l) = self.get(cur) else { continue };
            let is_3d = match &l.kind {
                LayerKind::Noise(n) => matches!(n.dims, NoiseDims::D3),
                LayerKind::Transform(t) => match &t.coord_mode {
                    CoordMode::Permute(axes) => {
                        axes.iter().any(|a| matches!(a, Axis::W))
                    }
                    CoordMode::Radial { dim: RadialDim::D3, .. } => true,
                    _ => false,
                },
                _ => false,
            };
            if is_3d {
                return true;
            }
            stack.extend(l.kind.inputs());
        }
        false
    }

    // ---- Layer mutation -------------------------------------------------

    /// Add a new layer. New IDs are always the current max + 1, so pushing
    /// keeps `layers` sorted. Name must be unique. New layer is appended
    /// to `list_order`; canvas positions are left unset (UI decides).
    pub fn add_layer(
        &mut self,
        name: impl Into<String>,
        kind: LayerKind,
    ) -> Result<LayerId, GraphError> {
        let name = name.into();
        if self.layers.iter().any(|l| l.name == name) {
            return Err(GraphError::DuplicateName(name));
        }
        for input in kind.inputs() {
            if !self.contains(input) {
                return Err(GraphError::UnknownId(input));
            }
        }
        validate_kind(&kind)?;
        let id = LayerId(self.next_id);
        self.next_id += 1;
        self.layers.push(Layer { id, name, kind });
        self.list_order.push(id);
        Ok(id)
    }

    /// Remove `id`. Every input that referenced it — in other layers or in
    /// the output — is disconnected first (falling back to `None` / const
    /// defaults, which render as the missing-texture grid). Cleans up list
    /// order and every canvas's position map.
    pub fn remove(&mut self, id: LayerId) -> Result<(), GraphError> {
        if !self.contains(id) {
            return Err(GraphError::UnknownId(id));
        }
        for l in &mut self.layers {
            let referencing: Vec<_> = l
                .kind
                .input_sockets()
                .iter()
                .filter(|s| s.value.connected_to() == Some(id))
                .map(|s| s.key)
                .collect();
            for key in referencing {
                let _ = l.kind.set_input(key, None);
            }
        }
        let referencing: Vec<_> = self
            .output
            .input_sockets()
            .iter()
            .filter(|s| s.value.connected_to() == Some(id))
            .map(|s| s.key)
            .collect();
        for key in referencing {
            let _ = self.output.set_input(key, None);
        }
        self.layers.retain(|l| l.id != id);
        self.list_order.retain(|x| *x != id);
        for canvas in self.canvases.values_mut() {
            canvas.positions.remove(&id);
        }
        Ok(())
    }

    /// Rename a layer. Name must remain unique.
    pub fn rename(
        &mut self,
        id: LayerId,
        new_name: impl Into<String>,
    ) -> Result<(), GraphError> {
        let new_name = new_name.into();
        if self.layers.iter().any(|l| l.id != id && l.name == new_name) {
            return Err(GraphError::DuplicateName(new_name));
        }
        let l = self.get_mut(id).ok_or(GraphError::UnknownId(id))?;
        l.name = new_name;
        Ok(())
    }

    /// Replace a layer's kind, rejecting cycles and unknown references.
    pub fn set_kind(&mut self, id: LayerId, kind: LayerKind) -> Result<(), GraphError> {
        if !self.contains(id) {
            return Err(GraphError::UnknownId(id));
        }
        for input in kind.inputs() {
            if !self.contains(input) {
                return Err(GraphError::UnknownId(input));
            }
        }
        validate_kind(&kind)?;
        let old = std::mem::replace(&mut self.get_mut(id).unwrap().kind, kind);
        if self.has_cycle_from(id) {
            self.get_mut(id).unwrap().kind = old;
            return Err(GraphError::Cycle(id));
        }
        Ok(())
    }

    /// Replace the output binding, rejecting unknown references.
    pub fn set_output(&mut self, output: Output) -> Result<(), GraphError> {
        for id in output.referenced() {
            if !self.contains(id) {
                return Err(GraphError::UnknownId(id));
            }
        }
        self.output = output;
        Ok(())
    }

    // ---- List order -----------------------------------------------------

    /// Move `id` to slot `to` in `list_order` (clamped to len − 1).
    pub fn set_list_position(&mut self, id: LayerId, to: usize) -> Result<(), GraphError> {
        let from = self
            .list_order
            .iter()
            .position(|x| *x == id)
            .ok_or(GraphError::UnknownId(id))?;
        let to = to.min(self.list_order.len().saturating_sub(1));
        if from == to {
            return Ok(());
        }
        let v = self.list_order.remove(from);
        self.list_order.insert(to, v);
        Ok(())
    }

    // ---- Canvases -------------------------------------------------------

    pub fn add_canvas(&mut self, name: impl Into<String>) -> Result<(), GraphError> {
        let name = name.into();
        if self.canvases.contains_key(&name) {
            return Err(GraphError::DuplicateCanvas(name));
        }
        self.canvases.insert(name, Canvas::default());
        Ok(())
    }

    pub fn remove_canvas(&mut self, name: &str) -> Result<(), GraphError> {
        self.canvases
            .remove(name)
            .map(|_| ())
            .ok_or_else(|| GraphError::UnknownCanvas(name.to_string()))
    }

    pub fn set_position(
        &mut self,
        canvas: &str,
        id: LayerId,
        pos: [f32; 2],
    ) -> Result<(), GraphError> {
        if !self.contains(id) {
            return Err(GraphError::UnknownId(id));
        }
        let c = self
            .canvases
            .get_mut(canvas)
            .ok_or_else(|| GraphError::UnknownCanvas(canvas.to_string()))?;
        c.positions.insert(id, pos);
        Ok(())
    }

    /// Store the Output pseudo-node's position on a canvas.
    pub fn set_output_position(
        &mut self,
        canvas: &str,
        pos: [f32; 2],
    ) -> Result<(), GraphError> {
        let c = self
            .canvases
            .get_mut(canvas)
            .ok_or_else(|| GraphError::UnknownCanvas(canvas.to_string()))?;
        c.output_pos = Some(pos);
        Ok(())
    }

    // ---- Cycle detection ------------------------------------------------

    /// True if wiring `candidate_input` into an input of `node` would
    /// create a cycle — i.e. `candidate_input` is `node` itself or
    /// (transitively) depends on it. Exact even when the new wire replaces
    /// an existing input: any offending path runs `candidate → … → node`
    /// through input edges and cannot pass through the edge being
    /// replaced. Intended for live drop-eligibility feedback; `set_kind`
    /// remains the authoritative validator.
    pub fn would_cycle(&self, node: LayerId, candidate_input: LayerId) -> bool {
        if candidate_input == node {
            return true;
        }
        let mut stack = vec![candidate_input];
        let mut seen: HashSet<LayerId> = HashSet::new();
        while let Some(cur) = stack.pop() {
            if cur == node {
                return true;
            }
            if !seen.insert(cur) {
                continue;
            }
            if let Some(l) = self.get(cur) {
                stack.extend(l.kind.inputs());
            }
        }
        false
    }

    /// DFS from `start` — returns true if any path revisits a node currently
    /// on the recursion stack.
    fn has_cycle_from(&self, start: LayerId) -> bool {
        let by_id: HashMap<LayerId, &Layer> = self.layers.iter().map(|l| (l.id, l)).collect();
        let mut visiting: HashSet<LayerId> = HashSet::new();
        let mut done: HashSet<LayerId> = HashSet::new();
        visit(start, &by_id, &mut visiting, &mut done)
    }
}

fn visit(
    node: LayerId,
    by_id: &HashMap<LayerId, &Layer>,
    visiting: &mut HashSet<LayerId>,
    done: &mut HashSet<LayerId>,
) -> bool {
    if done.contains(&node) {
        return false;
    }
    if !visiting.insert(node) {
        return true;
    }
    if let Some(l) = by_id.get(&node) {
        for child in l.kind.inputs() {
            if visit(child, by_id, visiting, done) {
                return true;
            }
        }
    }
    visiting.remove(&node);
    done.insert(node);
    false
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

fn validate_ramp(r: &ColorRamp) -> Result<(), GraphError> {
    if r.stops.len() < 2 {
        return Err(GraphError::RampTooFewStops);
    }
    Ok(())
}

/// Per-variant invariants the evaluator and the baker are allowed to
/// assume. Checked by every path that installs a kind, so neither backend
/// has to carry a fallback for a graph that cannot exist.
fn validate_kind(kind: &LayerKind) -> Result<(), GraphError> {
    match kind {
        LayerKind::ColorRamp(r) => validate_ramp(r),
        LayerKind::Noise(n) => {
            // Silently dropping the period would make a graph look tiled in
            // the editor and seam in the consumer — the one failure this
            // feature exists to prevent.
            if n.kernel == NoiseKernel::Simplex && n.period != [0; 3] {
                return Err(GraphError::PeriodicSimplex);
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kind::{Map, Mix, BlendMode};
    use crate::color::BlendSpace;

    /// a depends on b, b depends on c: a → b → c.
    fn chain() -> (Graph, LayerId, LayerId, LayerId) {
        let mut g = Graph::new();
        let c = g.layers[0].id;
        let b = g
            .add_layer("b", LayerKind::Map(Map { value: Some(c), palette: None }))
            .unwrap();
        let a = g
            .add_layer("a", LayerKind::Map(Map { value: Some(b), palette: None }))
            .unwrap();
        (g, a, b, c)
    }

    #[test]
    fn would_cycle_matches_dependency_direction() {
        let (g, a, b, c) = chain();
        // c's inputs may not include anything that depends on c.
        assert!(g.would_cycle(c, a));
        assert!(g.would_cycle(c, b));
        assert!(g.would_cycle(b, a));
        // Downstream nodes can take upstream ones as inputs.
        assert!(!g.would_cycle(a, c));
        assert!(!g.would_cycle(a, b));
        // Self-connection is a cycle.
        assert!(g.would_cycle(a, a));
    }

    #[test]
    fn would_cycle_is_exact_under_replacement() {
        let (mut g, a, _b, c) = chain();
        // An independent layer is fine as a's replacement input even
        // though a already has one.
        let x = g
            .add_layer("x", LayerKind::Map(Map { value: Some(c), palette: None }))
            .unwrap();
        assert!(!g.would_cycle(a, x));
    }

    #[test]
    fn would_cycle_agrees_with_set_kind() {
        let (mut g, a, b, c) = chain();
        for (node, input) in [(c, a), (b, a), (a, a), (a, c)] {
            let predicted = g.would_cycle(node, input);
            let result = g.set_kind(
                node,
                LayerKind::Mix(Mix {
                    a: Some(input),
                    b: None,
                    mode: BlendMode::Add,
                    factor: ScalarInput::Const(0.5),
                    space: BlendSpace::Oklch,
                }),
            );
            assert_eq!(
                predicted,
                matches!(result, Err(GraphError::Cycle(_))),
                "disagreement wiring {input:?} into {node:?}"
            );
            // Restore the chain for the next case.
            g = chain().0;
            // chain() rebuilds ids deterministically, so a/b/c stay valid.
        }
    }

    #[test]
    fn remove_disconnects_every_consumer() {
        let mut g = Graph::new();
        let victim = g.output.color.unwrap();
        let consumer = g
            .add_layer(
                "mix",
                LayerKind::Mix(Mix {
                    a: Some(victim),
                    b: None,
                    mode: BlendMode::Blend,
                    factor: ScalarInput::Layer(victim),
                    space: BlendSpace::Oklch,
                }),
            )
            .unwrap();
        g.output.normal = Some(victim);
        g.output.roughness = ScalarInput::Layer(victim);

        g.remove(victim).expect("referenced layers are removable");

        assert!(!g.contains(victim));
        let LayerKind::Mix(m) = &g.get(consumer).unwrap().kind else { panic!() };
        assert_eq!(m.a, None);
        assert!(matches!(m.factor, ScalarInput::Const(_)));
        assert_eq!(g.output.color, None);
        assert_eq!(g.output.normal, None);
        assert!(matches!(g.output.roughness, ScalarInput::Const(_)));
    }

    fn value_noise(period: [u32; 3]) -> LayerKind {
        LayerKind::Noise(crate::kind::Noise {
            dims: crate::kind::NoiseDims::D3,
            seed_offset: 0,
            frequency: 8.0,
            range: crate::kind::NoiseRange::Unsigned,
            output: crate::kind::NoiseOutput::Grayscale,
            kernel: NoiseKernel::Value,
            period,
            fractal: crate::kind::Fractal::default(),
        })
    }

    /// A period on the simplex kernel is rejected, not dropped: a silent
    /// no-op would look tiled here and seam in whatever samples the bake.
    #[test]
    fn a_periodic_simplex_is_rejected_by_every_path() {
        let mut periodic_simplex = value_noise([8, 8, 8]);
        let LayerKind::Noise(n) = &mut periodic_simplex else { panic!() };
        n.kernel = NoiseKernel::Simplex;

        let mut g = Graph::new();
        assert!(matches!(
            g.add_layer("bad", periodic_simplex.clone()),
            Err(GraphError::PeriodicSimplex)
        ));
        let id = g.output.color.unwrap();
        assert!(matches!(
            g.set_kind(id, periodic_simplex),
            Err(GraphError::PeriodicSimplex)
        ));
        // An unbounded period on simplex, and any period on value, are fine.
        let mut aperiodic = value_noise([0; 3]);
        let LayerKind::Noise(n) = &mut aperiodic else { panic!() };
        n.kernel = NoiseKernel::Simplex;
        assert!(g.add_layer("plain simplex", aperiodic).is_ok());
        assert!(g.add_layer("tiling value", value_noise([8, 8, 8])).is_ok());
    }

    /// The rejected kind must not be left installed — `set_kind` restores
    /// what was there on every other failure, and this one is no different.
    #[test]
    fn a_rejected_kind_leaves_the_layer_alone() {
        let mut g = Graph::new();
        let id = g.output.color.unwrap();
        let mut periodic_simplex = value_noise([4, 0, 0]);
        let LayerKind::Noise(n) = &mut periodic_simplex else { panic!() };
        n.kernel = NoiseKernel::Simplex;
        let _ = g.set_kind(id, periodic_simplex);
        assert!(matches!(g.get(id).unwrap().kind, LayerKind::Color(_)));
    }

    #[test]
    fn output_position_round_trip_and_unknown_canvas() {
        let mut g = Graph::new();
        assert!(matches!(
            g.set_output_position("main", [1.0, 2.0]),
            Err(GraphError::UnknownCanvas(_))
        ));
        g.add_canvas("main").unwrap();
        g.set_output_position("main", [1.0, 2.0]).unwrap();
        assert_eq!(g.canvases["main"].output_pos, Some([1.0, 2.0]));
    }

    #[test]
    fn canvas_without_output_pos_deserializes() {
        // A canvas serialized before `output_pos` existed.
        let old: Canvas = ron::from_str("(positions: {})").unwrap();
        assert_eq!(old.output_pos, None);
    }

    #[test]
    fn output_with_bare_color_id_deserializes() {
        // Files from before `Output::color` became optional store a bare
        // id; `implicit_some` (as used by file::load_from_str) wraps it.
        let options = ron::Options::default()
            .with_default_extension(ron::extensions::Extensions::IMPLICIT_SOME);
        let old: Output = options
            .from_str("(color: 3, roughness: Const(0.5), metallic: Const(0.0), normal: None)")
            .unwrap();
        assert_eq!(old.color, Some(LayerId(3)));
        assert_eq!(old.normal, None);
    }
}
