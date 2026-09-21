use std::collections::{BTreeMap, HashMap, HashSet};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::color::Color;
use crate::id::LayerId;
use crate::eval::EvalCtx;
use crate::kind::{ColorInput, ColorRamp, LayerKind, NoiseKernel, ScalarInput};
use crate::param::{ParamDecl, ParamUse, ParamValue};

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
    /// Every parameter the output's scalar channels read. The twin of
    /// [`LayerKind::param_refs`].
    pub fn param_refs(&self) -> Vec<(&str, ParamUse)> {
        let mut out = Vec::new();
        for si in [&self.roughness, &self.metallic] {
            if let ScalarInput::Param(name) = si {
                out.push((name.as_str(), ParamUse::Scalar));
            }
        }
        out
    }

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
    /// Named parameters this graph exposes, keyed by
    /// [`ParamDecl::name`]. A file without the field loads as empty, which
    /// is what every graph written before parameters existed means.
    #[serde(default)]
    pub params: BTreeMap<String, ParamDecl>,
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
    #[error("no parameter named {0:?} is declared")]
    UnknownParam(String),
    #[error("parameter {name:?} is a {declared}, but a {wanted} socket reads it")]
    ParamTypeMismatch { name: String, declared: &'static str, wanted: &'static str },
    #[error("duplicate parameter name: {0}")]
    DuplicateParam(String),
    #[error("parameter {name:?} defaults to a value its own kind ({declared}) does not describe")]
    ParamDefaultMismatch { name: String, declared: &'static str },
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
            params: BTreeMap::new(),
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
    /// - `Coordinate` on `Axis::W`
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
                LayerKind::Coordinate(c) => matches!(c.axis, Axis::W),
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
        self.validate_param_refs(kind.param_refs())?;
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
        self.validate_param_refs(kind.param_refs())?;
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
        self.validate_param_refs(output.param_refs())?;
        self.output = output;
        Ok(())
    }

    /// Every parameter a kind reads must be declared, and declared as the
    /// sort of thing the socket reading it can use. Checked here, at edit
    /// time, for the same reason a `LayerId` is: the evaluator and the
    /// baker then never have to decide what a stray name means.
    fn validate_param_refs(&self, refs: Vec<(&str, ParamUse)>) -> Result<(), GraphError> {
        for (name, use_) in refs {
            let Some(decl) = self.params.get(name) else {
                return Err(GraphError::UnknownParam(name.to_string()));
            };
            if !use_.accepts(&decl.kind) {
                return Err(GraphError::ParamTypeMismatch {
                    name: name.to_string(),
                    declared: decl.kind.label(),
                    wanted: use_.label(),
                });
            }
        }
        Ok(())
    }

    // ---- Parameters -----------------------------------------------------

    /// Declare a new parameter. The name must be unused, and the default
    /// must be the kind the declaration says it is.
    pub fn declare_param(&mut self, decl: ParamDecl) -> Result<(), GraphError> {
        if self.params.contains_key(&decl.name) {
            return Err(GraphError::DuplicateParam(decl.name));
        }
        validate_decl(&decl)?;
        self.params.insert(decl.name.clone(), decl);
        Ok(())
    }

    /// Replace an existing declaration, keeping its name.
    ///
    /// Changing the *kind* is rejected while anything reads it — a socket
    /// bound to a scalar cannot survive it becoming a color, and silently
    /// unbinding every reader would lose work the user cannot see from
    /// here. Rebind or remove the readers first.
    pub fn set_param_decl(&mut self, name: &str, decl: ParamDecl) -> Result<(), GraphError> {
        let Some(old) = self.params.get(name) else {
            return Err(GraphError::UnknownParam(name.to_string()));
        };
        if decl.name != name && self.params.contains_key(&decl.name) {
            return Err(GraphError::DuplicateParam(decl.name));
        }
        validate_decl(&decl)?;
        let kind_changed = std::mem::discriminant(&old.kind) != std::mem::discriminant(&decl.kind);
        if (kind_changed || decl.name != name) && self.param_readers(name).next().is_some() {
            // A rename is the same problem: every reader holds the old
            // string. `rename_param` exists to do it properly.
            return Err(GraphError::ParamTypeMismatch {
                name: name.to_string(),
                declared: old.kind.label(),
                wanted: decl.kind.label(),
            });
        }
        self.params.remove(name);
        self.params.insert(decl.name.clone(), decl);
        Ok(())
    }

    /// Rename a parameter, rewriting every socket that reads it.
    pub fn rename_param(&mut self, name: &str, new_name: &str) -> Result<(), GraphError> {
        if !self.params.contains_key(name) {
            return Err(GraphError::UnknownParam(name.to_string()));
        }
        if name == new_name {
            return Ok(());
        }
        if self.params.contains_key(new_name) {
            return Err(GraphError::DuplicateParam(new_name.to_string()));
        }
        let mut decl = self.params.remove(name).unwrap();
        decl.name = new_name.to_string();
        self.params.insert(new_name.to_string(), decl);
        for l in &mut self.layers {
            rewrite_param(&mut l.kind, name, Some(new_name));
        }
        rewrite_output_param(&mut self.output, name, Some(new_name));
        Ok(())
    }

    /// Remove a parameter. Every socket reading it falls back to the
    /// declaration's default *as a constant*, so the graph keeps looking
    /// the way it did — the same courtesy [`Graph::remove`] does not get
    /// to offer a layer, because there is no constant that stands in for
    /// one.
    pub fn remove_param(&mut self, name: &str) -> Result<(), GraphError> {
        let Some(decl) = self.params.remove(name) else {
            return Err(GraphError::UnknownParam(name.to_string()));
        };
        let fallback = decl.default;
        for l in &mut self.layers {
            freeze_param(&mut l.kind, name, fallback);
        }
        freeze_output_param(&mut self.output, name, fallback);
        Ok(())
    }

    /// Every layer with a socket reading `name`. The Output is not a
    /// layer and is checked separately by the callers that care.
    pub fn param_readers<'a>(&'a self, name: &'a str) -> impl Iterator<Item = LayerId> + 'a {
        self.layers
            .iter()
            .filter(move |l| l.kind.param_refs().iter().any(|(n, _)| *n == name))
            .map(|l| l.id)
            .chain(
                self.output
                    .param_refs()
                    .iter()
                    .any(|(n, _)| *n == name)
                    // The Output has no id; report the color root so a
                    // caller has something to point at. `None` when it is
                    // unconnected, which is fine — the iterator is only
                    // ever asked whether it is empty.
                    .then_some(self.output.color)
                    .flatten(),
            )
    }

    /// The value a parameter carries under `ctx`: the binding it holds,
    /// or the declaration's default where it holds none or one of the
    /// wrong kind. `None` when the name is not declared at all.
    ///
    /// Unlike [`EvalCtx::scalar_const`] this works on an *unresolved*
    /// context, so a UI can ask about one parameter without building the
    /// whole map.
    pub fn param_value(&self, name: &str, ctx: &EvalCtx) -> Option<ParamValue> {
        let decl = self.params.get(name)?;
        Some(
            ctx.params
                .get(name)
                .copied()
                .filter(|v| v.matches(&decl.kind))
                .unwrap_or(decl.default),
        )
    }

    /// `ctx` with every declared parameter present: the binding it
    /// carries, or the declaration's default where it carries none or
    /// carries one of the wrong kind.
    ///
    /// Every entry point runs this before evaluating, so the readers
    /// downstream are a map lookup and nothing else.
    pub fn resolve_params(&self, ctx: &EvalCtx) -> EvalCtx {
        let mut out = ctx.clone();
        for (name, decl) in &self.params {
            let bound = out.params.get(name).filter(|v| v.matches(&decl.kind));
            if bound.is_none() {
                out.params.insert(name.clone(), decl.default);
            }
        }
        out
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

fn validate_decl(decl: &ParamDecl) -> Result<(), GraphError> {
    if !decl.default.matches(&decl.kind) {
        return Err(GraphError::ParamDefaultMismatch {
            name: decl.name.clone(),
            declared: decl.kind.label(),
        });
    }
    Ok(())
}

/// Point every socket reading `name` at `to`, or — with `None` — leave
/// them alone. Used by `rename_param`.
fn rewrite_param(kind: &mut LayerKind, name: &str, to: Option<&str>) {
    let Some(to) = to else { return };
    let scalar = |si: &mut ScalarInput| {
        if matches!(si, ScalarInput::Param(n) if n == name) {
            *si = ScalarInput::Param(to.to_string());
        }
    };
    match kind {
        LayerKind::Mix(m) => scalar(&mut m.factor),
        LayerKind::Wave(w) => scalar(&mut w.input),
        LayerKind::ColorRamp(r) => {
            for stop in &mut r.stops {
                if matches!(&stop.color, ColorInput::Param(n) if n == name) {
                    stop.color = ColorInput::Param(to.to_string());
                }
            }
        }
        _ => {}
    }
}

fn rewrite_output_param(output: &mut Output, name: &str, to: Option<&str>) {
    let Some(to) = to else { return };
    for si in [&mut output.roughness, &mut output.metallic] {
        if matches!(si, ScalarInput::Param(n) if n == name) {
            *si = ScalarInput::Param(to.to_string());
        }
    }
}

/// Replace every read of `name` with `value` as a constant, so removing a
/// parameter does not change what the graph renders.
fn freeze_param(kind: &mut LayerKind, name: &str, value: ParamValue) {
    let scalar = |si: &mut ScalarInput| {
        if matches!(si, ScalarInput::Param(n) if n == name) {
            *si = ScalarInput::Const(value.as_scalar().unwrap_or(0.0));
        }
    };
    match kind {
        LayerKind::Mix(m) => scalar(&mut m.factor),
        LayerKind::Wave(w) => scalar(&mut w.input),
        LayerKind::ColorRamp(r) => {
            for stop in &mut r.stops {
                if matches!(&stop.color, ColorInput::Param(n) if n == name) {
                    stop.color = ColorInput::Const(
                        value.as_color().unwrap_or(crate::color::oklcha(0.5, 0.0, 0.0, 1.0)),
                    );
                }
            }
        }
        _ => {}
    }
}

fn freeze_output_param(output: &mut Output, name: &str, value: ParamValue) {
    for si in [&mut output.roughness, &mut output.metallic] {
        if matches!(si, ScalarInput::Param(n) if n == name) {
            *si = ScalarInput::Const(value.as_scalar().unwrap_or(0.0));
        }
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
    use crate::param::{ParamDecl, ParamKind, ParamValue};
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

    // ---- Parameters ----------------------------------------------------

    fn mix_with_factor(a: LayerId, factor: ScalarInput) -> LayerKind {
        LayerKind::Mix(crate::kind::Mix {
            a: Some(a),
            b: Some(a),
            mode: crate::kind::BlendMode::Blend,
            factor,
            space: crate::color::BlendSpace::Oklch,
        })
    }

    fn ramp_with_stop(color: ColorInput) -> LayerKind {
        LayerKind::ColorRamp(ColorRamp {
            stops: vec![
                crate::kind::ColorStop {
                    t: 0.0,
                    color: ColorInput::Const(crate::color::oklcha(0.0, 0.0, 0.0, 1.0)),
                },
                crate::kind::ColorStop { t: 1.0, color },
            ],
            space: crate::color::BlendSpace::Oklch,
        })
    }

    /// A name is checked at edit time exactly the way a `LayerId` is, so
    /// neither backend ever has to decide what a stray one means.
    #[test]
    fn an_undeclared_parameter_is_rejected_like_an_unknown_layer() {
        let mut g = Graph::new();
        let base = g.output.color.unwrap();
        assert!(matches!(
            g.add_layer("m", mix_with_factor(base, ScalarInput::Param("nope".into()))),
            Err(GraphError::UnknownParam(n)) if n == "nope"
        ));

        g.declare_param(ParamDecl::scalar("contrast", 0.0, 2.0, 1.0)).unwrap();
        assert!(g
            .add_layer("m", mix_with_factor(base, ScalarInput::Param("contrast".into())))
            .is_ok());
    }

    /// A scalar socket cannot read a color parameter, and the error says
    /// which way round it went wrong.
    #[test]
    fn a_socket_cannot_read_the_wrong_kind_of_parameter() {
        let mut g = Graph::new();
        let base = g.output.color.unwrap();
        g.declare_param(ParamDecl::color("tint", crate::color::oklcha(0.5, 0.1, 30.0, 1.0)))
            .unwrap();
        let err = g
            .add_layer("m", mix_with_factor(base, ScalarInput::Param("tint".into())))
            .unwrap_err();
        assert!(
            matches!(&err, GraphError::ParamTypeMismatch { name, declared, wanted }
                if name == "tint" && *declared == "color" && *wanted == "scalar"),
            "unexpected error: {err}"
        );
        // The same name in a color socket is fine.
        assert!(g.add_layer("r", ramp_with_stop(ColorInput::Param("tint".into()))).is_ok());
    }

    /// The Output's scalar channels are validated on the same path.
    #[test]
    fn the_output_validates_its_parameters_too() {
        let mut g = Graph::new();
        let color = g.output.color;
        let bad = Output {
            color,
            roughness: ScalarInput::Param("nope".into()),
            metallic: ScalarInput::Const(0.0),
            normal: None,
        };
        assert!(matches!(g.set_output(bad), Err(GraphError::UnknownParam(_))));
    }

    #[test]
    fn a_declarations_default_must_match_its_own_kind() {
        let mut g = Graph::new();
        let bogus = ParamDecl {
            name: "x".into(),
            kind: ParamKind::Scalar { min: 0.0, max: 1.0 },
            default: ParamValue::Color(crate::color::oklcha(0.5, 0.0, 0.0, 1.0)),
            description: None,
        };
        assert!(matches!(
            g.declare_param(bogus),
            Err(GraphError::ParamDefaultMismatch { .. })
        ));
        g.declare_param(ParamDecl::scalar("x", 0.0, 1.0, 0.5)).unwrap();
        assert!(matches!(
            g.declare_param(ParamDecl::scalar("x", 0.0, 1.0, 0.5)),
            Err(GraphError::DuplicateParam(_))
        ));
    }

    /// Removing a parameter freezes its readers at the declared default,
    /// so the graph goes on rendering what it rendered.
    #[test]
    fn removing_a_parameter_freezes_its_readers() {
        let mut g = Graph::new();
        let base = g.output.color.unwrap();
        g.declare_param(ParamDecl::scalar("contrast", 0.0, 2.0, 0.75)).unwrap();
        g.declare_param(ParamDecl::color("tint", crate::color::oklcha(0.4, 0.2, 90.0, 1.0)))
            .unwrap();
        let m = g
            .add_layer("m", mix_with_factor(base, ScalarInput::Param("contrast".into())))
            .unwrap();
        let r = g.add_layer("r", ramp_with_stop(ColorInput::Param("tint".into()))).unwrap();
        g.set_output(Output {
            color: g.output.color,
            roughness: ScalarInput::Param("contrast".into()),
            metallic: ScalarInput::Const(0.0),
            normal: None,
        })
        .unwrap();

        g.remove_param("contrast").unwrap();
        let LayerKind::Mix(mix) = &g.get(m).unwrap().kind else { panic!() };
        assert!(matches!(mix.factor, ScalarInput::Const(v) if v == 0.75));
        assert!(matches!(g.output.roughness, ScalarInput::Const(v) if v == 0.75));
        // The other parameter is untouched.
        let LayerKind::ColorRamp(ramp) = &g.get(r).unwrap().kind else { panic!() };
        assert!(matches!(ramp.stops[1].color, ColorInput::Param(ref n) if n == "tint"));

        g.remove_param("tint").unwrap();
        let LayerKind::ColorRamp(ramp) = &g.get(r).unwrap().kind else { panic!() };
        let ColorInput::Const(c) = ramp.stops[1].color else { panic!("expected a const") };
        assert!((c.l - 0.4).abs() < 1e-6);

        assert!(matches!(g.remove_param("tint"), Err(GraphError::UnknownParam(_))));
    }

    /// A rename has to rewrite every reader, or the graph stops validating
    /// against itself.
    #[test]
    fn renaming_a_parameter_rewrites_its_readers() {
        let mut g = Graph::new();
        let base = g.output.color.unwrap();
        g.declare_param(ParamDecl::scalar("contrast", 0.0, 2.0, 1.0)).unwrap();
        let m = g
            .add_layer("m", mix_with_factor(base, ScalarInput::Param("contrast".into())))
            .unwrap();
        g.set_output(Output {
            color: g.output.color,
            roughness: ScalarInput::Param("contrast".into()),
            metallic: ScalarInput::Const(0.0),
            normal: None,
        })
        .unwrap();

        g.rename_param("contrast", "punch").unwrap();
        assert!(g.params.contains_key("punch"));
        assert!(!g.params.contains_key("contrast"));
        assert_eq!(g.params["punch"].name, "punch");
        let LayerKind::Mix(mix) = &g.get(m).unwrap().kind else { panic!() };
        assert!(matches!(mix.factor, ScalarInput::Param(ref n) if n == "punch"));
        assert!(matches!(g.output.roughness, ScalarInput::Param(ref n) if n == "punch"));

        // And the graph still validates: re-setting the same kind passes.
        let kind = g.get(m).unwrap().kind.clone();
        assert!(g.set_kind(m, kind).is_ok());
    }

    /// Changing a declared kind under a live reader would strand it, so it
    /// is refused rather than silently unbinding.
    #[test]
    fn a_kind_change_is_refused_while_something_reads_it() {
        let mut g = Graph::new();
        let base = g.output.color.unwrap();
        g.declare_param(ParamDecl::scalar("contrast", 0.0, 2.0, 1.0)).unwrap();
        g.add_layer("m", mix_with_factor(base, ScalarInput::Param("contrast".into())))
            .unwrap();
        let to_color =
            ParamDecl::color("contrast", crate::color::oklcha(0.5, 0.0, 0.0, 1.0));
        assert!(matches!(
            g.set_param_decl("contrast", to_color.clone()),
            Err(GraphError::ParamTypeMismatch { .. })
        ));
        // Editing the range and default in place is fine.
        assert!(g
            .set_param_decl("contrast", ParamDecl::scalar("contrast", -1.0, 3.0, 2.0))
            .is_ok());
        assert_eq!(g.params["contrast"].default, ParamValue::Scalar(2.0));
    }

    /// Resolution is "the binding, else the default" — including when the
    /// binding is the wrong kind, which a host supplying values by name
    /// can easily get wrong.
    #[test]
    fn resolution_prefers_the_binding_and_falls_back_to_the_default() {
        let mut g = Graph::new();
        g.declare_param(ParamDecl::scalar("contrast", 0.0, 2.0, 0.75)).unwrap();

        let bare = g.resolve_params(&crate::eval::EvalCtx::default());
        assert_eq!(bare.params["contrast"], ParamValue::Scalar(0.75));

        let bound = crate::eval::EvalCtx::default()
            .with_param("contrast", ParamValue::Scalar(1.5));
        assert_eq!(g.resolve_params(&bound).params["contrast"], ParamValue::Scalar(1.5));

        // Wrong kind: ignored in favour of the declared default.
        let wrong = crate::eval::EvalCtx::default().with_param(
            "contrast",
            ParamValue::Color(crate::color::oklcha(0.5, 0.0, 0.0, 1.0)),
        );
        assert_eq!(g.resolve_params(&wrong).params["contrast"], ParamValue::Scalar(0.75));
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
