//! Center panel: Blender-style node editor with pan/zoom.
//!
//! - **Scroll wheel** zooms about the pointer.
//! - **Drag on empty space** pans; **drag on a node body** moves the node.
//! - **Drag from a socket** starts a wire, in either direction: an output
//!   onto an input connects, replacing whatever was there, and an
//!   *unconnected* input onto an output does the same thing from the other
//!   end. Pulling a *connected* input off detaches it instead, and dropping
//!   on empty space disconnects.
//! - **Click a node's title** to rename it: Enter or clicking away commits,
//!   Escape discards.
//! - **Right-click on empty space** adds a node at the pointer;
//!   right-click on a node offers rename, preview, duplicate and delete.
//!
//! Node positions live at `Canvas.positions` (the Output pseudo-node at
//! `Canvas.output_pos`) for `state.active_canvas`; edges are derived from
//! socket values each frame. Layers without a stored position get
//! grid-placed for the frame only.

mod layout;
mod nodes;
mod ramp;
mod wires;

use std::collections::HashMap;

use texture_graph_core::{EvalCtx, Graph, LayerId};

use crate::app::GpuBits;
use crate::panels::inspector;
use crate::previews::PreviewCache;
use crate::state::{EditCmd, NodeRef, UiState};

// World-unit metrics, multiplied by `canvas_zoom` for screen space.
const NODE_WIDTH: f32 = 180.0;
const HEADER_H: f32 = 34.0;
const THUMB_H: f32 = 76.0;
const ROW_H: f32 = 22.0;
/// Height of the ColorRamp gradient-bar row: gradient strip on top, arrow
/// indicator strip along the bottom.
const RAMP_BAR_H: f32 = 40.0;
const BOTTOM_PAD: f32 = 6.0;
const SOCKET_R: f32 = 4.5;
/// Below this zoom, inline widgets are skipped (rows keep their height so
/// wires don't jump).
const ZOOM_WIDGETS_MIN: f32 = 0.4;
/// Below this zoom, row labels are skipped too — header only.
const ZOOM_LABELS_MIN: f32 = 0.25;

const GRID_STEP_X: f32 = NODE_WIDTH + 60.0;
const GRID_STEP_Y: f32 = 280.0;
const GRID_COLUMNS: usize = 5;
const ZOOM_MIN: f32 = 0.15;
const ZOOM_MAX: f32 = 4.0;
/// Scroll wheel sensitivity for zoom. Small — 1 wheel tick ≈ 15% zoom.
const ZOOM_PER_SCROLL: f32 = 0.0015;

/// Screen-space grab radius around a socket — clamped so sockets stay
/// grabbable when zoomed far out.
fn socket_hit_radius(zoom: f32) -> f32 {
    (SOCKET_R * zoom).max(5.0) + 4.0
}

pub fn show(
    ui: &mut egui::Ui,
    graph: &Graph,
    state: &mut UiState,
    previews: &mut PreviewCache,
    eval_ctx: &EvalCtx,
    gpu: Option<&mut GpuBits>,
) {
    ensure_active_canvas(graph, state);

    // Reserve the whole panel as a click+drag surface for pan.
    let (canvas_rect, canvas_resp) =
        ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());
    let painter = ui.painter_at(canvas_rect);
    painter.rect_filled(canvas_rect, 0.0, egui::Color32::from_gray(28));

    handle_zoom(ui, state, &canvas_resp, canvas_rect);

    // If there's an in-progress node drag, keep the node locked under the
    // cursor before we do anything else this frame.
    let drag_still_active = advance_active_drag(ui, state, canvas_rect);
    release_stale_ramp_drag(ui, graph, state);

    let (positions, output_pos) = resolve_positions(graph, state);
    let layouts = layout::compute_layouts(graph, state, &positions, output_pos, canvas_rect.min);
    let out_sockets: HashMap<LayerId, egui::Pos2> = layouts
        .iter()
        .filter_map(|l| match (l.node, l.output_socket) {
            (NodeRef::Layer(id), Some(p)) => Some((id, p)),
            _ => None,
        })
        .collect();

    wires::draw_edges(&painter, &layouts, &out_sockets, state);
    let node_hit = nodes::draw_and_interact_nodes(
        ui,
        &painter,
        graph,
        state,
        &layouts,
        canvas_rect,
        previews,
        eval_ctx,
        gpu,
    );
    canvas_context_menu(graph, state, &canvas_resp, canvas_rect);

    // Live wire last, so it paints on top of everything.
    let wire_active =
        wires::advance_wire_drag(ui, graph, state, &layouts, &out_sockets, &painter);

    // Pan only when nothing else claimed the drag.
    if !drag_still_active
        && !wire_active
        && !node_hit
        && state.ramp_drag.is_none()
        && canvas_resp.dragged()
    {
        state.canvas_pan += canvas_resp.drag_delta();
    }
}

// ---- Coordinate transforms ---------------------------------------------

fn world_to_screen(world: egui::Pos2, origin: egui::Pos2, state: &UiState) -> egui::Pos2 {
    origin + state.canvas_pan + world.to_vec2() * state.canvas_zoom
}

fn screen_to_world(screen: egui::Pos2, origin: egui::Pos2, state: &UiState) -> egui::Pos2 {
    ((screen - origin - state.canvas_pan) / state.canvas_zoom).to_pos2()
}

// ---- Zoom ---------------------------------------------------------------

fn handle_zoom(
    ui: &egui::Ui,
    state: &mut UiState,
    canvas_resp: &egui::Response,
    canvas_rect: egui::Rect,
) {
    if !canvas_resp.hovered() {
        return;
    }
    let scroll_y = ui.ctx().input(|i| i.smooth_scroll_delta.y);
    if scroll_y.abs() < 0.5 {
        return;
    }
    let ptr = ui
        .ctx()
        .input(|i| i.pointer.hover_pos())
        .unwrap_or_else(|| canvas_rect.center());
    let world_at_ptr = screen_to_world(ptr, canvas_rect.min, state);
    let factor = (1.0 + scroll_y * ZOOM_PER_SCROLL).clamp(0.5, 2.0);
    let new_zoom = (state.canvas_zoom * factor).clamp(ZOOM_MIN, ZOOM_MAX);
    state.canvas_zoom = new_zoom;
    // Keep world_at_ptr under ptr.
    let new_screen = world_to_screen(world_at_ptr, canvas_rect.min, state);
    state.canvas_pan += ptr - new_screen;
}

// ---- Layout ------------------------------------------------------------

fn ensure_active_canvas(graph: &Graph, state: &mut UiState) {
    if graph.canvases.contains_key(&state.active_canvas) {
        return;
    }
    let already_pending = state
        .pending
        .iter()
        .any(|c| matches!(c, EditCmd::AddCanvas(n) if *n == state.active_canvas));
    if !already_pending {
        state.push(EditCmd::AddCanvas(state.active_canvas.clone()));
    }
}

/// Stored positions plus this-frame grid placement for unplaced layers,
/// and the Output pseudo-node's position (stored, or a default spot to the
/// right of everything).
fn resolve_positions(
    graph: &Graph,
    state: &UiState,
) -> (HashMap<LayerId, egui::Pos2>, egui::Pos2) {
    let canvas = graph.canvases.get(&state.active_canvas);
    let stored = canvas.map(|c| &c.positions);
    let mut out = HashMap::with_capacity(graph.layers.len());
    let mut placement = 0usize;
    for l in &graph.layers {
        let pos = stored
            .and_then(|m| m.get(&l.id))
            .copied()
            .map(|[x, y]| egui::pos2(x, y))
            .unwrap_or_else(|| {
                let col = placement % GRID_COLUMNS;
                let row = placement / GRID_COLUMNS;
                placement += 1;
                egui::pos2(col as f32 * GRID_STEP_X, row as f32 * GRID_STEP_Y)
            });
        out.insert(l.id, pos);
    }

    let output_pos = canvas
        .and_then(|c| c.output_pos)
        .map(|[x, y]| egui::pos2(x, y))
        .unwrap_or_else(|| {
            let max_x = out.values().map(|p| p.x).fold(f32::NEG_INFINITY, f32::max);
            let mean_y = if out.is_empty() {
                0.0
            } else {
                out.values().map(|p| p.y).sum::<f32>() / out.len() as f32
            };
            if max_x.is_finite() {
                egui::pos2(max_x + NODE_WIDTH + 60.0, mean_y)
            } else {
                egui::pos2(0.0, 0.0)
            }
        });
    (out, output_pos)
}

// ---- Drag lock ---------------------------------------------------------

/// If there's a live node drag, compute the node's new world position from
/// the current pointer position and emit the position command. Returns
/// whether the drag is still active this frame.
fn advance_active_drag(ui: &egui::Ui, state: &mut UiState, canvas_rect: egui::Rect) -> bool {
    let Some(drag) = state.drag else {
        return false;
    };
    let (primary_down, ptr) = ui
        .ctx()
        .input(|i| (i.pointer.primary_down(), i.pointer.hover_pos()));
    if !primary_down {
        state.drag = None;
        return false;
    }
    // If the cursor leaves the canvas viewport, release. Prevents wandering
    // nodes when the pointer sneaks under a sibling panel.
    let Some(ptr) = ptr else {
        state.drag = None;
        return false;
    };
    if !canvas_rect.contains(ptr) {
        state.drag = None;
        return false;
    }
    let new_screen = ptr + drag.grab_offset;
    let new_world = screen_to_world(new_screen, canvas_rect.min, state);
    match drag.node {
        NodeRef::Layer(id) => state.push(EditCmd::SetPos {
            canvas: state.active_canvas.clone(),
            id,
            pos: [new_world.x, new_world.y],
        }),
        NodeRef::Output => state.push(EditCmd::SetOutputPos {
            canvas: state.active_canvas.clone(),
            pos: [new_world.x, new_world.y],
        }),
    }
    true
}

/// Ramp-indicator drags normally end via the bar's `drag_stopped`, but the
/// bar's interact vanishes if zoom drops below the widget threshold, and
/// the node/kind/stop can disappear mid-drag (graph replaced, stop removed)
/// — release explicitly in those cases so the drag can't go stale.
fn release_stale_ramp_drag(ui: &egui::Ui, graph: &Graph, state: &mut UiState) {
    let Some(d) = state.ramp_drag else { return };
    let alive = ui.ctx().input(|i| i.pointer.primary_down())
        && state.canvas_zoom >= ZOOM_WIDGETS_MIN
        && matches!(
            graph.get(d.node).map(|l| &l.kind),
            Some(texture_graph_core::LayerKind::ColorRamp(r)) if d.stop < r.stops.len()
        );
    if !alive {
        state.ramp_drag = None;
    }
}

// ---- Context menu -------------------------------------------------------

const VARIANTS: &[&str] = &[
    "Color",
    "Noise",
    "ColorRamp",
    "Transform",
    "Mix",
    "Map",
    "MinMax",
    "HeightToNormal",
];

/// Right-click on empty canvas: add a node of any type at the pointer.
/// (Right-clicks on nodes are claimed by the node's own interact, so this
/// only fires on the background.)
fn canvas_context_menu(
    graph: &Graph,
    state: &mut UiState,
    canvas_resp: &egui::Response,
    canvas_rect: egui::Rect,
) {
    if canvas_resp.secondary_clicked() {
        if let Some(p) = canvas_resp.interact_pointer_pos() {
            let w = screen_to_world(p, canvas_rect.min, state);
            state.ctx_menu_world = Some([w.x, w.y]);
        }
    }
    canvas_resp.context_menu(|ui| {
        for &variant in VARIANTS {
            if ui.button(variant).clicked() {
                let name = crate::util::unique_name(graph, &variant.to_ascii_lowercase());
                let kind = inspector::default_kind(variant, graph);
                let pos = state.ctx_menu_world.take().unwrap_or([0.0, 0.0]);
                state.push(EditCmd::AddLayer {
                    name,
                    kind,
                    pos: Some((state.active_canvas.clone(), pos)),
                });
                ui.close();
            }
        }
    });
}
