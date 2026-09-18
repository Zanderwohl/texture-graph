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
use crate::catalog;
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
        for variant in catalog::VARIANTS {
            if ui.button(variant.label).clicked() {
                let name = catalog::unique_name(graph, &variant.label.to_ascii_lowercase());
                let pos = state.ctx_menu_world.take().unwrap_or([0.0, 0.0]);
                state.push(EditCmd::AddLayer {
                    name,
                    kind: catalog::default_kind(variant.kind),
                    pos: Some((state.active_canvas.clone(), pos)),
                });
                ui.close();
            }
        }
    });
}

#[cfg(test)]
mod canvas_tests {
    use super::*;
    use crate::catalog::{self, Kind};
    use crate::previews::PreviewCache;
    use texture_graph_core::{BlendMode, EvalCtx, LayerKind};

    const SCREEN: egui::Vec2 = egui::vec2(900.0, 600.0);

    /// Title of the layer at world (0, 0). Screen = canvas.min + pan +
    /// world, `title_rect` insets by (8, 2), and pan defaults to (40, 40).
    const BASE_TITLE: egui::Pos2 = egui::pos2(100.0, 50.0);
    /// Empty canvas, well clear of every node.
    const EMPTY: egui::Pos2 = egui::pos2(840.0, 560.0);

    /// Drives the real canvas through a real `egui::Context`, headlessly.
    struct Harness {
        ctx: egui::Context,
        graph: Graph,
        state: UiState,
        previews: PreviewCache,
        eval: EvalCtx,
    }

    impl Harness {
        fn new() -> Self {
            let mut graph = Graph::new();
            let noise = graph.add_layer("alpha", catalog::default_kind(Kind::Noise)).unwrap();
            let mix = graph.add_layer("beta", catalog::default_kind(Kind::Mix)).unwrap();
            let base = graph.layers[0].id;
            graph.add_canvas("main").unwrap();
            graph.set_position("main", base, [0.0, 0.0]).unwrap();
            graph.set_position("main", noise, [0.0, 320.0]).unwrap();
            graph.set_position("main", mix, [260.0, 0.0]).unwrap();
            graph.set_output_position("main", [520.0, 0.0]).unwrap();
            Self {
                ctx: egui::Context::default(),
                graph,
                state: UiState::default(),
                previews: PreviewCache::default(),
                eval: EvalCtx::default(),
            }
        }

        /// Run one frame, returning how many shapes egui painted in its
        /// error colour. Those are its debug overlays — `warn_on_id_clash`
        /// and `warn_if_rect_changes_id`, both on in debug builds — and any
        /// of them means the canvas moved a widget id around.
        ///
        /// The canvas paints nothing red of its own except an ineligible
        /// socket mid-wire-drag, which no test here performs.
        fn frame(&mut self, events: Vec<egui::Event>) -> usize {
            let input = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, SCREEN)),
                events,
                focused: true,
                ..Default::default()
            };
            let red = self.ctx.global_style().visuals.error_fg_color;
            let graph = &self.graph;
            let state = &mut self.state;
            let previews = &mut self.previews;
            let eval = &self.eval;
            let out = self.ctx.run_ui(input, |ctx| {
                egui::CentralPanel::default()
                    .frame(egui::Frame::NONE)
                    .show(ctx, |ui| {
                        super::show(ui, graph, state, previews, eval, None);
                    });
            });
            self.state.drain_into(&mut self.graph);
            let mut n = 0;
            for clipped in &out.shapes {
                count_red(&clipped.shape, red, &mut n);
            }
            n
        }

        /// Run `count` frames with no input, returning the total red count.
        fn settle(&mut self, count: usize) -> usize {
            (0..count).map(|_| self.frame(vec![])).sum()
        }

        fn click(&mut self, pos: egui::Pos2) -> usize {
            let down = vec![
                egui::Event::PointerMoved(pos),
                egui::Event::PointerButton {
                    pos,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::default(),
                },
            ];
            let up = vec![egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::default(),
            }];
            self.frame(down) + self.frame(up)
        }

        fn name_of(&self, i: usize) -> &str {
            &self.graph.layers[i].name
        }
    }

    fn count_red(shape: &egui::Shape, red: egui::Color32, n: &mut usize) {
        match shape {
            egui::Shape::Rect(r) if r.stroke.color == red || r.fill == red => *n += 1,
            egui::Shape::Text(t) if t.galley.text().contains('🔥') => *n += 1,
            egui::Shape::Vec(v) => {
                for s in v {
                    count_red(s, red, n);
                }
            }
            _ => {}
        }
    }

    /// Every inline widget in the canvas lives in a child `Ui`, and a child
    /// keyed by *salt* takes its auto-id seed from the parent's running
    /// counter. Showing the rename field adds one child before all the
    /// rows, so every slider and drag-value after it used to shift one slot
    /// along: egui flagged the whole canvas for a frame, and any in-flight
    /// interaction would have been handed to its neighbour. The ids are
    /// explicit now, and this is what holds them that way.
    #[test]
    fn starting_and_ending_a_rename_does_not_shuffle_widget_ids() {
        let mut h = Harness::new();
        assert_eq!(h.settle(3), 0, "the idle canvas already paints warnings");

        assert_eq!(h.click(BASE_TITLE), 0, "arming the rename shuffled ids");
        assert!(h.state.renaming.is_some(), "clicking the title did not start a rename");
        assert_eq!(h.settle(3), 0, "showing the rename field shuffled ids");

        // Clicking away blurs the field, which commits and removes it.
        assert_eq!(h.click(EMPTY), 0, "blurring the rename field shuffled ids");
        assert!(h.state.renaming.is_none(), "clicking away did not end the rename");
        assert_eq!(h.settle(3), 0, "removing the rename field shuffled ids");
    }

    /// The same hazard from the other direction, and the one that reaches
    /// the *rows*: a node's row count depends on its parameters, so
    /// switching a Mix off Blend removes two rows — and two child `Ui`s —
    /// from the middle of the canvas. Every widget laid out after it then
    /// shifts a slot along while staying exactly where it was, which is
    /// precisely the case egui's `warn_if_rect_changes_id` catches.
    ///
    /// The Output node is downstream of the Mix here and keeps two sliders,
    /// so there is something left to shift onto.
    #[test]
    fn changing_a_nodes_row_count_does_not_shuffle_later_widget_ids() {
        let mut h = Harness::new();
        assert_eq!(h.settle(3), 0, "the idle canvas already paints warnings");

        let mix = h.graph.layers[2].id;
        let LayerKind::Mix(mut m) = h.graph.get(mix).unwrap().kind.clone() else {
            panic!("layer 2 is the Mix")
        };
        assert_eq!(
            rows_before_and_after(&h.graph, mix, BlendMode::Add),
            (5, 3),
            "this test needs the row count to actually change"
        );
        m.mode = BlendMode::Add;
        h.graph.set_kind(mix, LayerKind::Mix(m)).unwrap();

        assert_eq!(h.settle(3), 0, "dropping two rows shuffled later ids");
    }

    /// How many rows the Mix has now, and how many it would have in `mode`.
    fn rows_before_and_after(graph: &Graph, id: LayerId, mode: BlendMode) -> (usize, usize) {
        let kind = graph.get(id).unwrap().kind.clone();
        let before = layout::rows_for(&kind).len();
        let LayerKind::Mix(mut m) = kind else { panic!() };
        m.mode = mode;
        (before, layout::rows_for(&LayerKind::Mix(m)).len())
    }

    /// Panning is the other way child `Ui` counts change, because
    /// off-screen nodes are skipped whole. This does not exercise the id
    /// hazard the way the row-count test does — every rect moves with the
    /// pan, so egui has no same-rect pair to compare — but it does prove
    /// the cull path itself paints nothing and does not panic.
    #[test]
    fn panning_nodes_out_of_view_is_quiet() {
        let mut h = Harness::new();
        assert_eq!(h.settle(3), 0);
        for pan in [-200.0, -400.0, -800.0, -1200.0] {
            h.state.canvas_pan = egui::vec2(pan, pan);
            assert_eq!(h.settle(2), 0, "panning to {pan} painted a warning");
        }
    }

    /// The rename itself, end to end through the real widget: click the
    /// title, type, press Enter, and the layer is renamed.
    #[test]
    fn clicking_a_title_renames_the_layer() {
        let mut h = Harness::new();
        h.settle(3);
        assert_eq!(h.name_of(0), "base color");

        h.click(BASE_TITLE);
        h.frame(vec![]);
        h.frame(vec![egui::Event::Text("!".to_string())]);
        h.frame(vec![egui::Event::Key {
            key: egui::Key::Enter,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::default(),
        }]);
        h.settle(2);

        assert_eq!(h.name_of(0), "base color!", "Enter did not commit the rename");
        assert!(h.state.renaming.is_none(), "the field outlived the commit");
    }

    /// Escape throws the edit away. Committing on blur is the other path,
    /// and the two must not be confused: Escape also surrenders focus, so
    /// an implementation that checked for a blur first would read a cancel
    /// as a commit.
    #[test]
    fn escape_abandons_a_rename() {
        let mut h = Harness::new();
        h.settle(3);
        h.click(BASE_TITLE);
        h.frame(vec![]);
        h.frame(vec![egui::Event::Text("zzz".to_string())]);
        h.frame(vec![egui::Event::Key {
            key: egui::Key::Escape,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::default(),
        }]);
        h.settle(2);

        assert_eq!(h.name_of(0), "base color", "Escape committed the edit anyway");
        assert!(h.state.renaming.is_none());
    }
}
