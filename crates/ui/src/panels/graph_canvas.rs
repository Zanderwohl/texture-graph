//! Center panel: 2D node graph with manual pan/zoom.
//!
//! - **Scroll wheel** zooms about the pointer.
//! - **Drag on empty space** pans.
//! - **Drag on a node** locks the node under the cursor (grab-offset) so
//!   the position tracks the cursor 1:1 even though `EditCmd::SetPos` is
//!   applied at end-of-frame.
//!
//! Nodes live at `Canvas.positions` for `state.active_canvas`; edges are
//! derived from `LayerKind::inputs()`. Layers without a stored position get
//! grid-placed for this frame only.

use std::collections::HashMap;

use texture_graph_core::{Graph, LayerId};

use crate::state::{EditCmd, NodeDrag, UiState};

const NODE_WIDTH: f32 = 140.0;
const NODE_HEIGHT: f32 = 44.0;
const GRID_STEP_X: f32 = NODE_WIDTH + 40.0;
const GRID_STEP_Y: f32 = NODE_HEIGHT + 20.0;
const GRID_COLUMNS: usize = 6;
const ZOOM_MIN: f32 = 0.15;
const ZOOM_MAX: f32 = 4.0;
/// Scroll wheel sensitivity for zoom. Small — 1 wheel tick ≈ 15% zoom.
const ZOOM_PER_SCROLL: f32 = 0.0015;

pub fn show(ui: &mut egui::Ui, graph: &Graph, state: &mut UiState) {
    ensure_active_canvas(graph, state);

    // Reserve the whole panel as a click+drag surface for pan.
    let (canvas_rect, canvas_resp) =
        ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());
    let painter = ui.painter_at(canvas_rect);

    // Zoom on scroll wheel (about pointer).
    handle_zoom(ui, state, &canvas_resp, canvas_rect);

    // If there's an in-progress node drag, keep the node locked under the
    // cursor before we do anything else this frame.
    let drag_still_active = advance_active_drag(ui, state, canvas_rect);

    let positions = resolve_positions(graph, state);
    draw_edges(&painter, graph, &positions, canvas_rect.min, state);
    let node_hit = draw_and_interact_nodes(ui, &painter, graph, state, &positions, canvas_rect);

    // Pan only when no node grabbed this drag and no node consumed the click.
    if !drag_still_active && !node_hit && canvas_resp.dragged() {
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

fn resolve_positions(graph: &Graph, state: &UiState) -> HashMap<LayerId, egui::Pos2> {
    let stored = graph
        .canvases
        .get(&state.active_canvas)
        .map(|c| &c.positions);
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
                egui::pos2(
                    col as f32 * GRID_STEP_X,
                    row as f32 * GRID_STEP_Y,
                )
            });
        out.insert(l.id, pos);
    }
    out
}

// ---- Drag lock ---------------------------------------------------------

/// If there's a live drag, compute the node's new world position from the
/// current pointer position and emit `EditCmd::SetPos`. Returns whether
/// the drag is still active this frame.
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
    state.push(EditCmd::SetPos {
        canvas: state.active_canvas.clone(),
        id: drag.id,
        pos: [new_world.x, new_world.y],
    });
    true
}

// ---- Drawing + hit-testing --------------------------------------------

fn draw_edges(
    painter: &egui::Painter,
    graph: &Graph,
    positions: &HashMap<LayerId, egui::Pos2>,
    origin: egui::Pos2,
    state: &UiState,
) {
    let stroke = egui::Stroke::new(1.5, egui::Color32::from_gray(160));
    for layer in &graph.layers {
        let dst_world = match positions.get(&layer.id) {
            Some(p) => *p,
            None => continue,
        };
        for input in layer.kind.inputs() {
            let Some(src_world) = positions.get(&input).copied() else {
                continue;
            };
            let src_screen = world_to_screen(src_world, origin, state);
            let dst_screen = world_to_screen(dst_world, origin, state);
            let a = src_screen + egui::vec2(NODE_WIDTH, NODE_HEIGHT * 0.5) * state.canvas_zoom;
            let b = dst_screen + egui::vec2(0.0, NODE_HEIGHT * 0.5) * state.canvas_zoom;
            let mid = ((b.x - a.x).abs() * 0.5).max(20.0 * state.canvas_zoom);
            let c1 = egui::pos2(a.x + mid, a.y);
            let c2 = egui::pos2(b.x - mid, b.y);
            painter.add(egui::Shape::CubicBezier(egui::epaint::CubicBezierShape {
                points: [a, c1, c2, b],
                closed: false,
                fill: egui::Color32::TRANSPARENT,
                stroke: stroke.into(),
            }));
        }
    }
}

/// Returns `true` if any node was clicked or drag-started this frame — so
/// the caller knows to suppress background pan.
fn draw_and_interact_nodes(
    ui: &mut egui::Ui,
    painter: &egui::Painter,
    graph: &Graph,
    state: &mut UiState,
    positions: &HashMap<LayerId, egui::Pos2>,
    canvas_rect: egui::Rect,
) -> bool {
    let z = state.canvas_zoom;
    let node_size_screen = egui::vec2(NODE_WIDTH, NODE_HEIGHT) * z;
    let mut hit = false;
    for layer in &graph.layers {
        let world = match positions.get(&layer.id) {
            Some(p) => *p,
            None => continue,
        };
        let screen_top_left = world_to_screen(world, canvas_rect.min, state);
        let rect = egui::Rect::from_min_size(screen_top_left, node_size_screen);

        let interact_id = egui::Id::new(("graph-node", layer.id.0));
        let resp = ui.interact(rect, interact_id, egui::Sense::click_and_drag());

        if resp.clicked() {
            state.selected = Some(layer.id);
            hit = true;
        }
        if resp.drag_started() {
            let ptr = ui
                .ctx()
                .input(|i| i.pointer.hover_pos())
                .unwrap_or(rect.center());
            state.drag = Some(NodeDrag {
                id: layer.id,
                grab_offset: screen_top_left - ptr,
            });
            state.selected = Some(layer.id);
            hit = true;
        }
        if resp.dragged() {
            hit = true;
        }

        draw_node(painter, layer, rect, state.selected == Some(layer.id), z);
    }
    hit
}

fn draw_node(
    painter: &egui::Painter,
    layer: &texture_graph_core::Layer,
    rect: egui::Rect,
    selected: bool,
    zoom: f32,
) {
    let bg = if selected {
        egui::Color32::from_rgb(60, 90, 120)
    } else {
        egui::Color32::from_gray(45)
    };
    let stroke = egui::Stroke::new(
        if selected { 2.0 } else { 1.0 },
        egui::Color32::from_gray(if selected { 240 } else { 180 }),
    );
    painter.rect(rect, 6.0 * zoom, bg, stroke, egui::StrokeKind::Middle);
    painter.text(
        rect.min + egui::vec2(10.0, 6.0) * zoom,
        egui::Align2::LEFT_TOP,
        &layer.name,
        egui::FontId::proportional(13.0 * zoom),
        egui::Color32::WHITE,
    );
    painter.text(
        rect.min + egui::vec2(10.0, 24.0) * zoom,
        egui::Align2::LEFT_TOP,
        layer.kind.category_label(),
        egui::FontId::proportional(11.0 * zoom),
        egui::Color32::from_gray(190),
    );
}
