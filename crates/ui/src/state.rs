//! UI-side state and the `EditCmd` command queue.
//!
//! Panels render against a shared `&Graph` (immutable during draw). Any
//! change the user makes becomes an `EditCmd` pushed to
//! [`UiState::pending`]; the app drains that vec after all panels have
//! rendered and applies each command to the graph. This keeps the borrow
//! checker happy and localizes error handling.

use texture_graph_core::{Graph, LayerId, LayerKind, Output};

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
    /// Which canvas we're editing on the graph panel. Defaults to "main".
    pub active_canvas: String,
}

/// In-progress node drag on the graph canvas.
#[derive(Copy, Clone, Debug)]
pub struct NodeDrag {
    pub id: LayerId,
    /// Where inside the node the pointer grabbed, in screen pixels
    /// (`node_screen_top_left - pointer`). Stays fixed for the whole drag.
    pub grab_offset: egui::Vec2,
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
            active_canvas: "main".to_string(),
        }
    }
}

impl UiState {
    /// Push a command onto the queue.
    pub fn push(&mut self, cmd: EditCmd) {
        self.pending.push(cmd);
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
    AddLayer { name: String, kind: LayerKind },
    Remove(LayerId),
    Rename(LayerId, String),
    SetKind(LayerId, LayerKind),
    SetOutput(Output),
    SetListPos(LayerId, usize),
    SetPos { canvas: String, id: LayerId, pos: [f32; 2] },
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
            | EditCmd::AddCanvas(_)
            | EditCmd::RemoveCanvas(_) => false,
        }
    }

    fn apply(self, graph: &mut Graph) -> Result<(), String> {
        match self {
            EditCmd::AddLayer { name, kind } => graph
                .add_layer(name, kind)
                .map(|_| ())
                .map_err(|e| e.to_string()),
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
