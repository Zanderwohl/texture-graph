//! UI-side state and the `EditCmd` command queue.
//!
//! Panels render against a shared `&Graph`, so edits become `EditCmd`s on
//! [`UiState::pending`] rather than mutations. The app drains them once every
//! panel has drawn, so error handling is in one place.

use std::collections::{HashMap, HashSet};

use texture_graph_core::{ConstValue, Graph, InputKey, LayerId, LayerKind, Output, ParamDecl};

/// UI-only state (not persisted with the graph).
#[derive(Debug)]
pub struct UiState {
    pub selected: Option<LayerId>,
    pub renaming: Option<Renaming>,
    /// Prefills the Save dialog. Just the filename on wasm.
    pub last_loaded_name: Option<String>,
    /// Read by the preview panel to decide whether to re-bake.
    pub dirty: bool,
    /// Bumped by edits that change evaluation. Caches of graph output store
    /// the revision they were computed at.
    pub revision: u64,
    /// Layers whose thumbnails the last frame's edits could have changed,
    /// consumers folded in. `None` means all of them.
    pub dirty_previews: Option<HashSet<LayerId>>,
    pub pending: Vec<EditCmd>,
    pub last_error: Option<String>,
    /// Set by the menu bar, consumed by `file_io` on the next frame.
    pub wants_save: bool,
    pub wants_open: bool,
    /// World space starts at `canvas_rect.min + canvas_pan`.
    pub canvas_pan: egui::Vec2,
    /// World units to screen pixels.
    pub canvas_zoom: f32,
    /// Holds the node at `mouse - grab_offset / zoom` every frame, so it
    /// doesn't lag a frame behind the mutation queue.
    pub drag: Option<NodeDrag>,
    /// Exclusive with `drag`: a press lands on a socket or a body, not both.
    pub wire_drag: Option<WireDrag>,
    pub ramp_drag: Option<RampDrag>,
    /// One menu serves all of a bar's indicators, so the stop the
    /// right-click chose has to outlive the click.
    pub ramp_menu: Option<RampMenu>,
    /// `Const` values displaced by a wire, restored on disconnect. Never
    /// persisted.
    pub saved_consts: HashMap<(NodeRef, InputKey), ConstValue>,
    /// Captured when the canvas menu opens, so a new node lands under the
    /// right-click.
    pub ctx_menu_world: Option<[f32; 2]>,
    /// `None` previews the full PBR material, `Some(id)` that layer's color
    /// as albedo over defaults.
    pub preview_target: Option<LayerId>,
    pub active_canvas: String,
}

/// A layer, or the material Output pseudo-node.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum NodeRef {
    Layer(LayerId),
    Output,
}

#[derive(Copy, Clone, Debug)]
pub struct NodeDrag {
    pub node: NodeRef,
    /// `node_top_left - pointer`, in screen pixels. Fixed for the drag.
    pub grab_offset: egui::Vec2,
}

/// An indicator drag on a ColorRamp's gradient bar.
#[derive(Copy, Clone, Debug)]
pub struct RampDrag {
    pub node: LayerId,
    /// Follows a crossing swap, which reorders the stops vec.
    pub stop: usize,
    /// `stop_t - pointer_t` at press, so an indicator grabbed off-center
    /// does not snap to the cursor.
    pub grab_dt: f32,
}

/// The stop a gradient bar's open menu is about. Bounds-checked against the
/// live stop count before it is acted on.
#[derive(Copy, Clone, Debug)]
pub struct RampMenu {
    pub node: LayerId,
    pub stop: usize,
}

/// The end grabbed decides only what the drag looks for; the connection it
/// makes is the same either way.
#[derive(Copy, Clone, Debug)]
pub enum WireDrag {
    FromOutput {
        src: LayerId,
        /// Set when an existing wire was pulled off an input. The edge is
        /// hidden while dragging, and no `EditCmd` fires until drop, so a
        /// canceled drag causes no rebake.
        detached_from: Option<(NodeRef, InputKey)>,
    },
    /// Only from an unconnected input. Dragging a connected one detaches
    /// its wire, which is [`WireDrag::FromOutput`].
    FromInput { node: NodeRef, key: InputKey },
}

impl WireDrag {
    pub fn detached_from(self) -> Option<(NodeRef, InputKey)> {
        match self {
            WireDrag::FromOutput { detached_from, .. } => detached_from,
            WireDrag::FromInput { .. } => None,
        }
    }
}

/// A node title being typed over. Layer names must be unique, so a
/// half-typed one needs somewhere to sit outside the graph.
#[derive(Clone, Debug)]
pub struct Renaming {
    pub node: LayerId,
    pub text: String,
    /// Only the first frame asks for focus: asking every frame makes the
    /// field impossible to blur, and blurring is what commits.
    pub focused: bool,
}

impl Default for UiState {
    fn default() -> Self {
        Self {
            selected: None,
            renaming: None,
            last_loaded_name: None,
            dirty: false,
            revision: 0,
            dirty_previews: None,
            pending: Vec::new(),
            last_error: None,
            wants_save: false,
            wants_open: false,
            canvas_pan: egui::vec2(40.0, 40.0),
            canvas_zoom: 1.0,
            drag: None,
            wire_drag: None,
            ramp_drag: None,
            ramp_menu: None,
            saved_consts: HashMap::new(),
            ctx_menu_world: None,
            preview_target: None,
            active_canvas: "main".to_string(),
        }
    }
}

impl UiState {
    pub fn push(&mut self, cmd: EditCmd) {
        self.pending.push(cmd);
    }

    /// Move `RampStop` keys through `map` (old index → new, or `None` to
    /// drop). Those keys are positional, so every reorder, insert and remove
    /// has to remap them or a later disconnect restores the wrong stop.
    pub fn remap_ramp_consts(&mut self, node: NodeRef, map: impl Fn(usize) -> Option<usize>) {
        let keys: Vec<usize> = self
            .saved_consts
            .keys()
            .filter_map(|(n, k)| match k {
                InputKey::RampStop(i) if *n == node => Some(*i),
                _ => None,
            })
            .collect();
        let affected: Vec<(usize, ConstValue)> = keys
            .into_iter()
            .filter_map(|i| {
                self.saved_consts
                    .remove(&(node, InputKey::RampStop(i)))
                    .map(|v| (i, v))
            })
            .collect();
        for (i, v) in affected {
            if let Some(j) = map(i) {
                self.saved_consts.insert((node, InputKey::RampStop(j)), v);
            }
        }
    }

    /// Apply every queued command in order, recording failures and carrying
    /// on: one refused connection must not swallow the node drag queued
    /// behind it. Returns whether anything landed.
    pub fn drain_into(&mut self, graph: &mut Graph) -> bool {
        let mut changed = false;
        let mut refused = false;
        let mut evaluated = false;
        for cmd in std::mem::take(&mut self.pending) {
            let affects_eval = cmd.affects_evaluation();
            let roots = affects_eval.then(|| dirty_roots(&cmd)).flatten();
            // Consumers are collected before and after: a Remove takes the
            // references that would have found them. Over-invalidating costs a
            // bake; under-invalidating shows a stale picture.
            let mut dirty = HashSet::new();
            if let Some(roots) = &roots {
                downstream(graph, roots.iter().copied(), &mut dirty);
            }
            let localized = !affects_eval || roots.is_some();
            match self.apply(cmd, graph) {
                Ok(()) => {
                    changed = true;
                    evaluated |= affects_eval;
                    if affects_eval {
                        self.dirty = true;
                    }
                    if let Some(roots) = &roots {
                        downstream(graph, roots.iter().copied(), &mut dirty);
                    }
                    if localized {
                        // `None` already means "all of them"; nothing to add.
                        if let Some(known) = &mut self.dirty_previews {
                            known.extend(dirty);
                        }
                    } else {
                        self.dirty_previews = None;
                    }
                }
                Err(e) => {
                    refused = true;
                    self.last_error = Some(e);
                }
            }
        }
        if evaluated {
            self.revision += 1;
        }
        // A refusal stays shown until an edit lands cleanly.
        if changed && !refused {
            self.last_error = None;
        }
        changed
    }

    /// Everything a freshly loaded graph must forget: ids mean different
    /// layers now, so anything holding one points at the wrong node.
    pub fn reset_for_new_graph(&mut self) {
        self.selected = None;
        self.renaming = None;
        self.last_error = None;
        self.drag = None;
        self.wire_drag = None;
        self.ramp_drag = None;
        self.ramp_menu = None;
        self.saved_consts.clear();
        self.ctx_menu_world = None;
        self.preview_target = None;
        self.dirty_previews = None;
    }

    /// Apply one command. Here rather than on [`EditCmd`] because of the
    /// UI-side bookkeeping: a removed layer must stop being the selection,
    /// the preview target and a `saved_consts` key.
    fn apply(&mut self, cmd: EditCmd, graph: &mut Graph) -> Result<(), String> {
        match cmd {
            EditCmd::AddLayer { name, kind, pos } => {
                let id = graph.add_layer(name, kind).map_err(|e| e.to_string())?;
                self.selected = Some(id);
                if let Some((canvas, pos)) = pos {
                    // The layer exists either way; a vanished canvas just
                    // leaves it unplaced.
                    let _ = graph.set_position(&canvas, id, pos);
                }
                Ok(())
            }
            EditCmd::Remove(id) => {
                if self.selected == Some(id) {
                    self.selected = None;
                }
                if self.preview_target == Some(id) {
                    self.preview_target = None;
                }
                if self.renaming.as_ref().is_some_and(|r| r.node == id) {
                    self.renaming = None;
                }
                self.saved_consts.retain(|(n, _), _| *n != NodeRef::Layer(id));
                graph.remove(id).map_err(|e| e.to_string())
            }
            EditCmd::Rename(id, new) => graph.rename(id, new).map_err(|e| e.to_string()),
            EditCmd::SetKind(id, k) => graph.set_kind(id, k).map_err(|e| e.to_string()),
            EditCmd::SetOutput(o) => graph.set_output(o).map_err(|e| e.to_string()),
            EditCmd::SetListPos(id, to) => {
                graph.set_list_position(id, to).map_err(|e| e.to_string())
            }
            EditCmd::SetPos { canvas, id, pos } => {
                graph.set_position(&canvas, id, pos).map_err(|e| e.to_string())
            }
            EditCmd::SetOutputPos { canvas, pos } => graph
                .set_output_position(&canvas, pos)
                .map_err(|e| e.to_string()),
            EditCmd::AddCanvas(name) => graph.add_canvas(name).map_err(|e| e.to_string()),
            EditCmd::RemoveCanvas(name) => {
                graph.remove_canvas(&name).map_err(|e| e.to_string())
            }
            EditCmd::Replace(new) => {
                *graph = new;
                self.reset_for_new_graph();
                Ok(())
            }
            EditCmd::DeclareParam(decl) => {
                graph.declare_param(decl).map_err(|e| e.to_string())
            }
            EditCmd::SetParamDecl(name, decl) => {
                graph.set_param_decl(&name, decl).map_err(|e| e.to_string())
            }
            EditCmd::RenameParam { from, to } => {
                graph.rename_param(&from, &to).map_err(|e| e.to_string())
            }
            EditCmd::RemoveParam(name) => {
                graph.remove_param(&name).map_err(|e| e.to_string())
            }
        }
    }
}

/// The layers an edit changes the value *of*, before consumers are folded
/// in. `None` when every thumbnail has to be assumed stale.
///
/// A new layer is nobody's input yet and the Output is nobody's input at
/// all, so both dirty nothing.
fn dirty_roots(cmd: &EditCmd) -> Option<Vec<LayerId>> {
    match cmd {
        EditCmd::AddLayer { .. } => Some(Vec::new()),
        EditCmd::Remove(id) | EditCmd::SetKind(id, _) => Some(vec![*id]),
        EditCmd::SetOutput(_) => Some(Vec::new()),
        // Layout only; `affects_evaluation` means these never get here, but
        // a total match forces a new command to state its answer.
        EditCmd::Rename(_, _)
        | EditCmd::SetListPos(_, _)
        | EditCmd::SetPos { .. }
        | EditCmd::SetOutputPos { .. }
        | EditCmd::AddCanvas(_)
        | EditCmd::RemoveCanvas(_) => Some(Vec::new()),
        // The ids either side of this aren't about the same layers.
        EditCmd::Replace(_) => None,
        // A parameter can be read anywhere, and `remove_param` rewrites
        // sockets across the graph; rebaking all is cheaper than tracking.
        EditCmd::DeclareParam(_)
        | EditCmd::SetParamDecl(_, _)
        | EditCmd::RenameParam { .. }
        | EditCmd::RemoveParam(_) => None,
    }
}

/// `roots` and everything that transitively reads them.
///
/// Scans every layer, because edges are stored input→node with no reverse
/// index. Graphs are tens of layers and this runs per landed edit.
fn downstream(
    graph: &Graph,
    roots: impl IntoIterator<Item = LayerId>,
    out: &mut HashSet<LayerId>,
) {
    let mut stack: Vec<LayerId> = roots.into_iter().collect();
    while let Some(cur) = stack.pop() {
        if !out.insert(cur) {
            continue;
        }
        for layer in &graph.layers {
            if layer.kind.inputs().contains(&cur) {
                stack.push(layer.id);
            }
        }
    }
}

/// One user-driven mutation, each variant mirroring a `Graph::*` method.
#[derive(Debug, Clone)]
pub enum EditCmd {
    /// `pos` places the layer right after the add: the `LayerId` doesn't
    /// exist at push time, so a panel can't build a separate `SetPos`.
    AddLayer { name: String, kind: LayerKind, pos: Option<(String, [f32; 2])> },
    Remove(LayerId),
    Rename(LayerId, String),
    SetKind(LayerId, LayerKind),
    SetOutput(Output),
    SetListPos(LayerId, usize),
    SetPos { canvas: String, id: LayerId, pos: [f32; 2] },
    SetOutputPos { canvas: String, pos: [f32; 2] },
    AddCanvas(String),
    RemoveCanvas(String),
    /// File > New and File > Open.
    Replace(Graph),
    DeclareParam(ParamDecl),
    /// Replace a declaration in place, keeping its name.
    SetParamDecl(String, ParamDecl),
    RenameParam { from: String, to: String },
    RemoveParam(String),
}

impl EditCmd {
    /// Whether this can change what the baker produces. Layout-only edits
    /// must not flip `dirty`: dragging a node emits `SetPos` every frame,
    /// and rebaking on each would make thumbnails flash.
    fn affects_evaluation(&self) -> bool {
        match self {
            EditCmd::AddLayer { .. }
            | EditCmd::Remove(_)
            | EditCmd::SetKind(_, _)
            | EditCmd::SetOutput(_)
            | EditCmd::Replace(_)
            // A declaration's default is what an unbound parameter reads,
            // so any of these can move the picture.
            | EditCmd::DeclareParam(_)
            | EditCmd::SetParamDecl(_, _)
            | EditCmd::RenameParam { .. }
            | EditCmd::RemoveParam(_) => true,
            EditCmd::Rename(_, _)
            | EditCmd::SetListPos(_, _)
            | EditCmd::SetPos { .. }
            | EditCmd::SetOutputPos { .. }
            | EditCmd::AddCanvas(_)
            | EditCmd::RemoveCanvas(_) => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use texture_graph_core::color::oklcha;
    use texture_graph_core::{CoordMode, EdgeMode, Transform};

    fn color(l: f32) -> LayerKind {
        LayerKind::Color(oklcha(l, 0.0, 0.0, 1.0))
    }

    fn reading(src: LayerId) -> LayerKind {
        LayerKind::Transform(Transform {
            source: Some(src),
            offset: [0.0; 3],
            rotate_uv: 0.0,
            scale: [1.0; 3],
            coord_mode: CoordMode::Passthrough,
            edge_mode: EdgeMode::Clamp,
        })
    }

    /// An edit retires the layer it touched and everything reading it, and
    /// nothing else. Too little shows a stale thumbnail; too much rebakes the
    /// whole canvas on every edit.
    #[test]
    fn an_edit_dirties_what_reads_it_and_leaves_the_rest_alone() {
        let mut graph = Graph::new();
        let a = graph.add_layer("a", color(0.2)).unwrap();
        let reader = graph.add_layer("reader", reading(a)).unwrap();
        let indirect = graph.add_layer("indirect", reading(reader)).unwrap();
        let unrelated = graph.add_layer("unrelated", color(0.8)).unwrap();

        // An empty set: nothing retired yet.
        let mut ui = UiState { dirty_previews: Some(HashSet::new()), ..Default::default() };

        ui.push(EditCmd::SetKind(a, color(0.9)));
        assert!(ui.drain_into(&mut graph));

        let dirty = ui.dirty_previews.expect("a SetKind is localizable");
        assert!(dirty.contains(&a), "the edited layer kept its picture");
        assert!(dirty.contains(&reader), "a layer reading the edited one kept its picture");
        assert!(dirty.contains(&indirect), "the closure stopped at one hop");
        assert!(!dirty.contains(&unrelated), "an unrelated layer was retired");
    }

    /// A removed layer's consumers must be found before the remove, while
    /// there are still edges pointing at it.
    #[test]
    fn removing_a_layer_dirties_what_used_to_read_it() {
        let mut graph = Graph::new();
        let a = graph.add_layer("a", color(0.2)).unwrap();
        let reader = graph.add_layer("reader", reading(a)).unwrap();

        let mut ui = UiState { dirty_previews: Some(HashSet::new()), ..Default::default() };
        ui.push(EditCmd::Remove(a));
        assert!(ui.drain_into(&mut graph), "{:?}", ui.last_error);

        let dirty = ui.dirty_previews.expect("a Remove is localizable");
        assert!(dirty.contains(&reader), "the layer left pointing at nothing kept its picture");
    }

    /// Dragging emits a position command every frame; counting it as an
    /// evaluation change would rebake thumbnails for the whole drag.
    #[test]
    fn moving_a_node_is_not_an_evaluation_change() {
        let mut graph = Graph::new();
        let a = graph.add_layer("a", color(0.2)).unwrap();

        let mut ui = UiState { dirty_previews: Some(HashSet::new()), ..Default::default() };
        let before = ui.revision;
        ui.push(EditCmd::SetPos {
            canvas: "main".to_string(),
            id: a,
            pos: [10.0, 10.0],
        });
        ui.push(EditCmd::Rename(a, "renamed".to_string()));
        assert!(ui.drain_into(&mut graph), "{:?}", ui.last_error);

        assert_eq!(ui.revision, before, "a layout edit bumped the revision");
        assert!(ui.dirty_previews.is_some_and(|d| d.is_empty()));
        assert!(!ui.dirty, "a layout edit marked the graph dirty");
    }
}
