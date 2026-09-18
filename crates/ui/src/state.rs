//! UI-side state and the `EditCmd` command queue.
//!
//! Panels render against a shared `&Graph` (immutable during draw). Any
//! change the user makes becomes an `EditCmd` pushed to
//! [`UiState::pending`]; the app drains that vec after all panels have
//! rendered and applies each command to the graph. This keeps the borrow
//! checker happy and localizes error handling.

use std::collections::HashMap;

use texture_graph_core::{ConstValue, Graph, InputKey, LayerId, LayerKind, Output};

/// UI-only state (not persisted with the graph).
#[derive(Debug)]
pub struct UiState {
    /// Layer selected in the list / inspector.
    pub selected: Option<LayerId>,
    /// Path or filename of the last file we loaded from — used to prefill
    /// the Save dialog. On wasm this is just the filename.
    pub last_loaded_name: Option<String>,
    /// Set by any applied `EditCmd`; consumed by the preview cache /
    /// output baker to decide whether to re-render.
    pub dirty: bool,
    /// Commands emitted by panels this frame. Drained at the end of the
    /// frame by [`UiState::drain_into`].
    pub pending: Vec<EditCmd>,
    /// Any error surfaced by a recent edit. Displayed as a status line.
    pub last_error: Option<String>,
    /// Set by the menu bar; consumed by `file_io` (once wired) to open a
    /// save dialog on the next frame.
    pub wants_save: bool,
    /// Same, for open/upload.
    pub wants_open: bool,
    /// Graph canvas pan, in screen pixels — the origin of world space is
    /// `canvas_rect.min + canvas_pan`.
    pub canvas_pan: egui::Vec2,
    /// Graph canvas zoom (world units → screen pixels).
    pub canvas_zoom: f32,
    /// Node currently being dragged in the graph canvas. Locks the node's
    /// world position at `mouse - grab_offset_screen / zoom` every frame,
    /// so there's no lag from the mutation queue's one-frame round trip.
    pub drag: Option<NodeDrag>,
    /// In-progress wire drag on the graph canvas. Mutually exclusive with
    /// `drag` — a press lands on either a socket or a node body.
    pub wire_drag: Option<WireDrag>,
    /// In-progress stop-indicator drag on a ColorRamp node's gradient bar.
    pub ramp_drag: Option<RampDrag>,
    /// Session-only memory of `Const` values displaced when a wire was
    /// connected over them, restored on disconnect. Never persisted.
    pub saved_consts: HashMap<(NodeRef, InputKey), ConstValue>,
    /// World position captured when the canvas context menu opened, so a
    /// newly added node lands under the right-click.
    pub ctx_menu_world: Option<[f32; 2]>,
    /// Node the preview panel renders. `None` = the Output node (the full
    /// PBR material); `Some(id)` = just that layer's color as albedo, with
    /// default roughness/metallic/normal. Set from the node context menu.
    pub preview_target: Option<LayerId>,
    /// Which canvas we're editing on the graph panel. Defaults to "main".
    pub active_canvas: String,
}

/// A node on the graph canvas: either a layer or the material Output
/// pseudo-node.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum NodeRef {
    Layer(LayerId),
    Output,
}

/// In-progress node drag on the graph canvas.
#[derive(Copy, Clone, Debug)]
pub struct NodeDrag {
    pub node: NodeRef,
    /// Where inside the node the pointer grabbed, in screen pixels
    /// (`node_screen_top_left - pointer`). Stays fixed for the whole drag.
    pub grab_offset: egui::Vec2,
}

/// In-progress indicator drag on a ColorRamp node's gradient bar.
#[derive(Copy, Clone, Debug)]
pub struct RampDrag {
    pub node: LayerId,
    /// Current index of the dragged stop — updated live when a crossing
    /// swap reorders the stops vec.
    pub stop: usize,
}

/// In-progress wire drag. The payload is always "the output of `src`".
#[derive(Copy, Clone, Debug)]
pub struct WireDrag {
    pub src: LayerId,
    /// Set when the drag started by pulling an existing wire off an input.
    /// That edge is hidden while dragging; no `EditCmd` fires until drop,
    /// so a cancelled drag causes zero rebakes.
    pub detached_from: Option<(NodeRef, InputKey)>,
}

impl Default for UiState {
    fn default() -> Self {
        Self {
            selected: None,
            last_loaded_name: None,
            dirty: false,
            pending: Vec::new(),
            last_error: None,
            wants_save: false,
            wants_open: false,
            canvas_pan: egui::vec2(40.0, 40.0),
            canvas_zoom: 1.0,
            drag: None,
            wire_drag: None,
            ramp_drag: None,
            saved_consts: HashMap::new(),
            ctx_menu_world: None,
            preview_target: None,
            active_canvas: "main".to_string(),
        }
    }
}

impl UiState {
    /// Push a command onto the queue.
    pub fn push(&mut self, cmd: EditCmd) {
        self.pending.push(cmd);
    }

    /// Rebuild `saved_consts` entries keyed `(node, RampStop(i))` through
    /// `map`: old index → `Some(new index)` or `None` to drop. `RampStop`
    /// keys are purely positional, so any reorder / insert / remove of a
    /// ramp's stops must remap them or later disconnects restore the wrong
    /// stop's const.
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

    /// Apply every queued command to `graph`. Errors from individual
    /// commands are recorded in `last_error`; remaining commands continue
    /// to apply.
    pub fn drain_into(&mut self, graph: &mut Graph) {
        for cmd in std::mem::take(&mut self.pending) {
            let affects_eval = cmd.affects_evaluation();
            match cmd.apply(graph) {
                Ok(()) => {
                    if affects_eval {
                        self.dirty = true;
                    }
                }
                Err(e) => self.last_error = Some(e),
            }
        }
    }
}

/// One user-driven mutation, ready to be applied to a `Graph`.
///
/// Every variant corresponds to a `Graph::*` mutation method.
#[derive(Debug, Clone)]
pub enum EditCmd {
    /// `pos` places the new layer on a canvas right after the add — needed
    /// because the new `LayerId` doesn't exist at push time, so a separate
    /// `SetPos` command can't be constructed by the panel.
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
    /// Replace the graph wholesale — used for File > New and File > Open.
    Replace(Graph),
}

impl EditCmd {
    /// Whether applying this command can change what the evaluator/baker
    /// produces. Layout-only edits (canvas positions, list order, canvas
    /// add/remove, layer rename) don't touch the DAG the baker walks and
    /// must NOT flip `dirty` — otherwise dragging a node in the canvas
    /// (which emits `SetPos` every frame) would rebake every preview each
    /// frame and make thumbnails flash out of existence.
    fn affects_evaluation(&self) -> bool {
        match self {
            EditCmd::AddLayer { .. }
            | EditCmd::Remove(_)
            | EditCmd::SetKind(_, _)
            | EditCmd::SetOutput(_)
            | EditCmd::Replace(_) => true,
            EditCmd::Rename(_, _)
            | EditCmd::SetListPos(_, _)
            | EditCmd::SetPos { .. }
            | EditCmd::SetOutputPos { .. }
            | EditCmd::AddCanvas(_)
            | EditCmd::RemoveCanvas(_) => false,
        }
    }

    fn apply(self, graph: &mut Graph) -> Result<(), String> {
        match self {
            EditCmd::AddLayer { name, kind, pos } => {
                let id = graph.add_layer(name, kind).map_err(|e| e.to_string())?;
                if let Some((canvas, pos)) = pos {
                    // Best-effort: the layer exists either way; a vanished
                    // canvas just leaves it unplaced.
                    let _ = graph.set_position(&canvas, id, pos);
                }
                Ok(())
            }
            EditCmd::Remove(id) => graph.remove(id).map_err(|e| e.to_string()),
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
                Ok(())
            }
        }
    }
}
