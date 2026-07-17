use std::collections::{BTreeMap, HashMap, HashSet};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::color::Color;
use crate::id::LayerId;
use crate::kind::{ColorRamp, LayerKind, ScalarInput};

/// Named node in the graph. Position on any workspace canvas is stored
/// separately in [`Graph::canvases`], so identity/display/layout stay
/// independent.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Layer {
    pub id: LayerId,
    pub name: String,
    pub kind: LayerKind,
}

/// The graph's root, producing PBR-material channels for the renderer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Output {
    /// Layer whose color is the material's base color.
    pub color: LayerId,
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
        let mut out = vec![self.color];
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

/// One named workspace canvas — a scatter of layer positions plus any
/// per-canvas view state we grow later. Edges are derived from
/// [`LayerKind::inputs`] at render time and not stored.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Canvas {
    /// Position of each layer within this canvas. Layers absent from the
    /// map should be treated as unplaced (UI decides where to drop them).
    pub positions: BTreeMap<LayerId, [f32; 2]>,
}

/// The full graph. Three orthogonal shapes travel with the data:
///
/// - **Identity** — [`Layer::id`], monotonically assigned, never reused.
///   Layers are kept in ID-ascending order in [`Graph::layers`] so
///   serialization is stable regardless of any UI reordering.
/// - **List order** — [`Graph::list_order`], a permutation of layer IDs
///   for the linear layer-panel view.
/// - **Canvas positions** — [`Graph::canvases`], zero or more named
///   node-graph workspaces each with its own scatter of positions.
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
    #[error("layer {0} is still referenced by another layer or the output")]
    StillReferenced(LayerId),
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
                color: LayerId(0),
                roughness: ScalarInput::Const(0.5),
                metallic: ScalarInput::Const(0.0),
                normal: None,
            },
            next_id: 1,
        };
        let id = g
            .add_layer("base color", LayerKind::Color(Color::new(0.5, 0.0, 0.0, 1.0)))
            .expect("first add cannot fail");
        g.output.color = id;
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
        if let LayerKind::ColorRamp(r) = &kind {
            validate_ramp(r)?;
        }
        let id = LayerId(self.next_id);
        self.next_id += 1;
        self.layers.push(Layer { id, name, kind });
        self.list_order.push(id);
        Ok(id)
    }

    /// Remove `id`. Fails if it is referenced by any other layer or by
    /// `output`. Cleans up list order and every canvas's position map.
    pub fn remove(&mut self, id: LayerId) -> Result<(), GraphError> {
        if !self.contains(id) {
            return Err(GraphError::UnknownId(id));
        }
        if self.output.referenced().contains(&id) {
            return Err(GraphError::StillReferenced(id));
        }
        for l in &self.layers {
            if l.kind.inputs().contains(&id) {
                return Err(GraphError::StillReferenced(id));
            }
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
        if let LayerKind::ColorRamp(r) = &kind {
            validate_ramp(r)?;
        }
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

    // ---- Cycle detection ------------------------------------------------

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
