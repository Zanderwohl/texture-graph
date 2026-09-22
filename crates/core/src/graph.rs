use std::collections::{BTreeMap, HashMap, HashSet};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::color::Color;
use crate::id::LayerId;
use crate::eval::EvalCtx;
use crate::kind::{ColorInput, ColorRamp, LayerKind, NoiseKernel, ScalarInput};
use crate::param::{ParamDecl, ParamUse, ParamValue};

/// A node in the graph. Its canvas position lives in [`Graph::canvases`].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Layer {
    pub id: LayerId,
    pub name: String,
    pub kind: LayerKind,
}

/// The graph's root, producing PBR-material channels for the renderer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Output {
    /// Base color. `None` renders as the missing-texture grid.
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
    /// See [`LayerKind::param_refs`].
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

/// Layer positions on one named canvas. Edges are not stored; they come from
/// [`LayerKind::inputs`].
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Canvas {
    /// Layers absent from the map are unplaced.
    pub positions: BTreeMap<LayerId, [f32; 2]>,
    /// The Output pseudo-node. `None` is unplaced.
    #[serde(default)]
    pub output_pos: Option<[f32; 2]>,
}

/// Identity ([`Layer::id`], never reused), list order and canvas positions
/// are stored separately. `layers` stays ID-ascending so serialization does
/// not change when the UI reorders layers.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Graph {
    /// ID-ascending. Do not reorder for display; use `list_order` instead.
    pub layers: Vec<Layer>,
    /// Permutation of layer IDs for the list view.
    pub list_order: Vec<LayerId>,
    pub canvases: BTreeMap<String, Canvas>,
    pub output: Output,
    /// Keyed by [`ParamDecl::name`].
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

    pub fn get(&self, id: LayerId) -> Option<&Layer> {
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

    /// True if `id` contributes to the Output.
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

    /// True when any layer reachable from the Output varies along `w`, so
    /// the graph is a solid texture rather than a flat image.
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

    /// Name must be unique. The layer is appended to `list_order` and left
    /// unplaced on every canvas.
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

    /// Every input that referenced `id`, in layers or the output, is
    /// disconnected first.
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

    /// Name must remain unique.
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

    /// Rejects cycles and unknown references, leaving the layer unchanged.
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

    /// Rejects unknown references.
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

    /// Checked at edit time so the evaluator and baker never meet an
    /// undeclared or mistyped name.
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

    /// The name must be unused and the default must match the declared kind.
    pub fn declare_param(&mut self, decl: ParamDecl) -> Result<(), GraphError> {
        if self.params.contains_key(&decl.name) {
            return Err(GraphError::DuplicateParam(decl.name));
        }
        validate_decl(&decl)?;
        self.params.insert(decl.name.clone(), decl);
        Ok(())
    }

    /// Changing the kind or name is rejected while anything reads the
    /// parameter, rather than silently unbinding the readers. Use
    /// [`Graph::rename_param`] to rename.
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

    /// Rewrites every socket that reads it.
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

    /// Every socket reading it gets the declared default as a constant, so
    /// the graph renders the same.
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

    /// Every layer with a socket reading `name`. If the Output reads it,
    /// the Output's color layer is also yielded (if connected), so only
    /// emptiness is reliable for the Output.
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
                    // The Output has no id. Callers only test for emptiness.
                    .then_some(self.output.color)
                    .flatten(),
            )
    }

    /// The binding in `ctx`, or the declared default if unbound or of the
    /// wrong kind. `None` if undeclared. Works on an unresolved context.
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

    /// `ctx` with every declared parameter present, as in
    /// [`Graph::param_value`]. Evaluation entry points call this first.
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

    /// True if wiring `candidate_input` into `node` would create a cycle.
    /// Exact even when the wire replaces an existing input, since any cycle
    /// path runs `candidate → … → node` and cannot use the replaced edge.
    /// For live UI feedback; `set_kind` is the authoritative check.
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

/// `None` leaves the sockets alone.
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

/// Per-variant invariants the evaluator and baker assume. Every path that
/// installs a kind must call this.
fn validate_kind(kind: &LayerKind) -> Result<(), GraphError> {
    match kind {
        LayerKind::ColorRamp(r) => validate_ramp(r),
        LayerKind::Noise(n) => {
            // Ignoring the period would look tiled here and seam in the consumer.
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
        assert!(g.would_cycle(c, a));
        assert!(g.would_cycle(c, b));
        assert!(g.would_cycle(b, a));
        assert!(!g.would_cycle(a, c));
        assert!(!g.would_cycle(a, b));
        assert!(g.would_cycle(a, a));
    }

    #[test]
    fn would_cycle_is_exact_under_replacement() {
        let (mut g, a, _b, c) = chain();
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
            // chain() assigns the same ids each time.
            g = chain().0;
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
        let mut aperiodic = value_noise([0; 3]);
        let LayerKind::Noise(n) = &mut aperiodic else { panic!() };
        n.kernel = NoiseKernel::Simplex;
        assert!(g.add_layer("plain simplex", aperiodic).is_ok());
        assert!(g.add_layer("tiling value", value_noise([8, 8, 8])).is_ok());
    }

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

    /// The error names the declared and wanted kinds the right way round.
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
        assert!(g.add_layer("r", ramp_with_stop(ColorInput::Param("tint".into()))).is_ok());
    }

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
        let LayerKind::ColorRamp(ramp) = &g.get(r).unwrap().kind else { panic!() };
        assert!(matches!(ramp.stops[1].color, ColorInput::Param(ref n) if n == "tint"));

        g.remove_param("tint").unwrap();
        let LayerKind::ColorRamp(ramp) = &g.get(r).unwrap().kind else { panic!() };
        let ColorInput::Const(c) = ramp.stops[1].color else { panic!("expected a const") };
        assert!((c.l - 0.4).abs() < 1e-6);

        assert!(matches!(g.remove_param("tint"), Err(GraphError::UnknownParam(_))));
    }

    /// A reader left on the old name would make the graph fail validation.
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

        let kind = g.get(m).unwrap().kind.clone();
        assert!(g.set_kind(m, kind).is_ok());
    }

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
        assert!(g
            .set_param_decl("contrast", ParamDecl::scalar("contrast", -1.0, 3.0, 2.0))
            .is_ok());
        assert_eq!(g.params["contrast"].default, ParamValue::Scalar(2.0));
    }

    /// A binding of the wrong kind falls back to the default.
    #[test]
    fn resolution_prefers_the_binding_and_falls_back_to_the_default() {
        let mut g = Graph::new();
        g.declare_param(ParamDecl::scalar("contrast", 0.0, 2.0, 0.75)).unwrap();

        let bare = g.resolve_params(&crate::eval::EvalCtx::default());
        assert_eq!(bare.params["contrast"], ParamValue::Scalar(0.75));

        let bound = crate::eval::EvalCtx::default()
            .with_param("contrast", ParamValue::Scalar(1.5));
        assert_eq!(g.resolve_params(&bound).params["contrast"], ParamValue::Scalar(1.5));

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
        let old: Canvas = ron::from_str("(positions: {})").unwrap();
        assert_eq!(old.output_pos, None);
    }

    #[test]
    fn output_with_bare_color_id_deserializes() {
        // Older files store a bare id; `implicit_some`, as in
        // `file::load_from_str`, wraps it.
        let options = ron::Options::default()
            .with_default_extension(ron::extensions::Extensions::IMPLICIT_SOME);
        let old: Output = options
            .from_str("(color: 3, roughness: Const(0.5), metallic: Const(0.0), normal: None)")
            .unwrap();
        assert_eq!(old.color, Some(LayerId(3)));
        assert_eq!(old.normal, None);
    }
}
