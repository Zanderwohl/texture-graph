//! Center panel: Blender-style node editor with pan/zoom.
//!
//! - **Scroll wheel** zooms about the pointer.
//! - **Drag on empty space** pans; **drag on a node body** moves the node.
//! - **Drag from a socket** starts a wire, either direction; dropping on
//!   empty space disconnects. Pulling a *connected* input off detaches it.
//! - **Click a node's title** to rename it: Enter or clicking away commits,
//!   Escape discards.
//! - **Right-click on empty space** adds a node at the pointer;
//!   right-click on a node offers rename, preview, duplicate and delete.
//!
//! Positions live on `Canvas` for `state.active_canvas`; edges are derived
//! from socket values each frame, and an unplaced layer is grid-placed for
//! that frame only.

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
/// ColorRamp bar row: gradient strip over an indicator strip.
const RAMP_BAR_H: f32 = 40.0;
const BOTTOM_PAD: f32 = 6.0;
const SOCKET_R: f32 = 4.5;
/// Below this, widgets are skipped; rows keep their height so wires stay put.
const ZOOM_WIDGETS_MIN: f32 = 0.4;
/// Below this, labels go too, leaving the header.
const ZOOM_LABELS_MIN: f32 = 0.25;

const GRID_STEP_X: f32 = NODE_WIDTH + 60.0;
const GRID_STEP_Y: f32 = 280.0;
const GRID_COLUMNS: usize = 5;
const ZOOM_MIN: f32 = 0.15;
const ZOOM_MAX: f32 = 4.0;
/// One wheel tick ≈ 15% zoom.
const ZOOM_PER_SCROLL: f32 = 0.0015;

/// Grab radius around a socket, floored so it survives zooming out.
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

    // The whole panel is a pan surface.
    let (canvas_rect, canvas_resp) =
        ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());
    let painter = ui.painter_at(canvas_rect);
    painter.rect_filled(canvas_rect, 0.0, egui::Color32::from_gray(28));

    handle_zoom(ui, state, &canvas_resp, canvas_rect);

    // Lock a dragged node under the cursor before anything else runs.
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

/// Stored positions, grid placement for unplaced layers, and the Output's
/// position — stored, or defaulted to the right of everything.
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

/// Resolve a live node drag against the pointer and emit its position.
/// Returns whether the drag is still active.
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
    // Release on leaving the viewport, or a node wanders off under a
    // sibling panel.
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

/// The bar's interact vanishes below the widget zoom threshold, and its
/// node or stop can disappear mid-drag, so a drag that outlives either has
/// to be released here.
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

/// Right-click on empty canvas adds a node at the pointer. A node's own
/// interact claims right-clicks on it, so this only fires on background.
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

    /// Title of the layer at world (0, 0): pan (40, 40) plus a (8, 2) inset.
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
        /// Last frame's paint list. A popup lays itself out at the pointer,
        /// so a menu item can only be found by the text it drew.
        shapes: Vec<egui::epaint::ClippedShape>,
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
                shapes: Vec::new(),
            }
        }

        /// Run one frame, returning how many shapes egui painted in its error
        /// colour. Those are `warn_on_id_clash` and `warn_if_rect_changes_id`,
        /// so any of them means a widget id moved. The canvas paints nothing
        /// red itself but an ineligible socket mid-wire-drag.
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
            self.shapes = out.shapes;
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

        /// A ColorRamp with stops at `ts`, clear of the other nodes and `EMPTY`.
        fn add_ramp(&mut self, ts: &[f32]) -> LayerId {
            let kind = LayerKind::ColorRamp(texture_graph_core::ColorRamp {
                stops: ts
                    .iter()
                    .map(|t| texture_graph_core::ColorStop {
                        t: *t,
                        color: texture_graph_core::ColorInput::Const(
                            texture_graph_core::color::oklcha(*t, 0.0, 0.0, 1.0),
                        ),
                    })
                    .collect(),
                space: texture_graph_core::BlendSpace::Oklch,
            });
            let id = self.graph.add_layer("ramp", kind).unwrap();
            self.graph.set_position("main", id, [260.0, 320.0]).unwrap();
            id
        }

        /// `id`'s gradient strip in screen space, and a y in the indicators.
        fn ramp_bar(&self, id: LayerId) -> (egui::Rect, f32) {
            let origin = egui::Pos2::ZERO;
            let (positions, output_pos) = resolve_positions(&self.graph, &self.state);
            let layouts =
                layout::compute_layouts(&self.graph, &self.state, &positions, output_pos, origin);
            let l = layouts
                .iter()
                .find(|l| l.node == NodeRef::Layer(id))
                .expect("the ramp is laid out");
            let row = l
                .rows
                .iter()
                .position(|r| matches!(r, layout::Row::Param(layout::ParamRow::RampBar)))
                .expect("a ramp has a bar row");
            let (grad, arrows) = nodes::ramp_bar_rects(l.row_rects[row], self.state.canvas_zoom);
            (grad, arrows.center().y)
        }

        /// Screen x of the indicator for stop `i` of `id`.
        fn indicator_x(&self, id: LayerId, i: usize) -> f32 {
            let (grad, _) = self.ramp_bar(id);
            grad.left() + self.stop_t(id, i).clamp(0.0, 1.0) * grad.width()
        }

        fn stop_t(&self, id: LayerId, i: usize) -> f32 {
            self.stops(id)[i].t
        }

        fn stops(&self, id: LayerId) -> Vec<texture_graph_core::ColorStop> {
            match &self.graph.get(id).expect("the ramp is still there").kind {
                LayerKind::ColorRamp(r) => r.stops.clone(),
                _ => panic!("not a ramp"),
            }
        }

        /// Press, move, release: three frames, the way a real drag arrives.
        fn drag(&mut self, from: egui::Pos2, to: egui::Pos2) -> usize {
            let button = |pos, pressed| egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed,
                modifiers: egui::Modifiers::default(),
            };
            self.frame(vec![egui::Event::PointerMoved(from), button(from, true)])
                + self.frame(vec![egui::Event::PointerMoved(to)])
                + self.frame(vec![button(to, false)])
        }

        fn right_click(&mut self, pos: egui::Pos2) -> usize {
            let button = |pressed| egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Secondary,
                pressed,
                modifiers: egui::Modifiers::default(),
            };
            self.frame(vec![egui::Event::PointerMoved(pos), button(true)])
                + self.frame(vec![button(false)])
        }

        fn a_menu_is_open(&self) -> bool {
            egui::Popup::is_any_open(&self.ctx)
        }

        /// Middle of the first text reading `label`. Whole-screen, so only
        /// for text unique on it — several nodes show a "0.50". Use
        /// [`Self::find_text_in`] otherwise.
        fn find_text(&self, label: &str) -> Option<egui::Pos2> {
            self.find_text_in(egui::Rect::EVERYTHING, label)
        }

        /// The same, restricted to text drawn inside `area`.
        fn find_text_in(&self, area: egui::Rect, label: &str) -> Option<egui::Pos2> {
            fn walk(
                shape: &egui::Shape,
                area: egui::Rect,
                label: &str,
                out: &mut Option<egui::Pos2>,
            ) {
                match shape {
                    egui::Shape::Text(t) if out.is_none() && t.galley.text() == label => {
                        let mid = t.pos + t.galley.size() / 2.0;
                        if area.contains(mid) {
                            *out = Some(mid);
                        }
                    }
                    egui::Shape::Vec(v) => {
                        for s in v {
                            walk(s, area, label, out);
                        }
                    }
                    _ => {}
                }
            }
            let mut out = None;
            for clipped in &self.shapes {
                walk(&clipped.shape, area, label, &mut out);
            }
            out
        }

        /// Screen rect of `id`'s node, to scope a text search to it.
        fn node_rect(&self, id: LayerId) -> egui::Rect {
            let (positions, output_pos) = resolve_positions(&self.graph, &self.state);
            let layouts = layout::compute_layouts(
                &self.graph,
                &self.state,
                &positions,
                output_pos,
                egui::Pos2::ZERO,
            );
            layouts
                .iter()
                .find(|l| l.node == NodeRef::Layer(id))
                .expect("the node is laid out")
                .rect
        }

        /// Right-click `pos`, then click the menu item reading `label`.
        fn menu_click(&mut self, pos: egui::Pos2, label: &str) -> usize {
            let mut red = self.right_click(pos) + self.settle(1);
            let item = self
                .find_text(label)
                .unwrap_or_else(|| panic!("no menu item reading {label:?}"));
            red += self.click(item);
            red + self.settle(1)
        }
    }

    /// An indicator grabbed off centre travels with the pointer instead of
    /// snapping under it, starting on the press rather than once egui has
    /// decided the press was a drag.
    #[test]
    fn a_stop_indicator_travels_with_the_pointer_that_grabbed_it() {
        let mut h = Harness::new();
        let ramp = h.add_ramp(&[0.0, 0.5, 1.0]);
        assert_eq!(h.settle(3), 0, "the idle canvas already paints warnings");

        let (grad, y) = h.ramp_bar(ramp);
        assert!(
            grad.width() > 100.0 && grad.right() < SCREEN.x,
            "the bar is on screen and big enough to drag across: {grad:?}"
        );
        // Inside the grab radius, far enough off that a snap would show.
        let from = egui::pos2(h.indicator_x(ramp, 1) + 4.0, y);
        let to = egui::pos2(from.x + 40.0, y);
        assert_eq!(h.drag(from, to), 0, "dragging a stop shuffled ids");

        let expected = 0.5 + 40.0 / grad.width();
        assert!(
            (h.stop_t(ramp, 1) - expected).abs() < 1e-3,
            "stop went to {} rather than {expected}",
            h.stop_t(ramp, 1)
        );
        assert!(h.state.ramp_drag.is_none(), "the release did not end the drag");
    }

    /// The grab radius is generous but not the whole bar: a press in open
    /// gradient must not drag the nearest stop over to it.
    #[test]
    fn a_press_in_open_gradient_grabs_nothing() {
        let mut h = Harness::new();
        let ramp = h.add_ramp(&[0.0, 0.5, 1.0]);
        assert_eq!(h.settle(3), 0, "the idle canvas already paints warnings");

        let (_, y) = h.ramp_bar(ramp);
        let from = egui::pos2(h.indicator_x(ramp, 1) + 30.0, y);
        assert_eq!(h.drag(from, egui::pos2(from.x + 40.0, y)), 0, "shuffled ids");
        assert_eq!(h.stops(ramp).iter().map(|s| s.t).collect::<Vec<_>>(), vec![0.0, 0.5, 1.0]);
    }

    /// Right-clicking an indicator opens its menu without arming a drag the
    /// release would then have to clean up.
    #[test]
    fn right_clicking_an_indicator_opens_a_menu_without_arming_a_drag() {
        let mut h = Harness::new();
        let ramp = h.add_ramp(&[0.0, 0.5, 1.0]);
        assert_eq!(h.settle(3), 0, "the idle canvas already paints warnings");

        let (_, y) = h.ramp_bar(ramp);
        let on = egui::pos2(h.indicator_x(ramp, 1), y);
        assert_eq!(h.right_click(on), 0, "opening the menu shuffled ids");
        assert!(h.a_menu_is_open(), "right-clicking an indicator opened no menu");
        assert!(h.state.ramp_drag.is_none(), "the right-click armed a drag");
        assert_eq!(h.stops(ramp).len(), 3, "the right-click changed the ramp");
    }

    /// Duplicate end to end: the copy lands a nudge right, and stays sorted.
    #[test]
    fn duplicating_a_stop_puts_the_copy_a_nudge_away() {
        let mut h = Harness::new();
        let ramp = h.add_ramp(&[0.0, 0.5, 1.0]);
        assert_eq!(h.settle(3), 0, "the idle canvas already paints warnings");

        let (_, y) = h.ramp_bar(ramp);
        let on = egui::pos2(h.indicator_x(ramp, 1), y);
        assert_eq!(h.menu_click(on, "Duplicate"), 0, "duplicating shuffled ids");

        let ts: Vec<f32> = h.stops(ramp).iter().map(|s| s.t).collect();
        assert_eq!(ts.len(), 4, "no stop was added: {ts:?}");
        assert!((ts[2] - 0.55).abs() < 1e-5, "copy landed wrong: {ts:?}");
        assert!(ts.windows(2).all(|w| w[0] <= w[1]), "ramp came out unsorted: {ts:?}");
    }

    /// Delete end to end, with no confirmation.
    #[test]
    fn deleting_a_stop_removes_it_outright() {
        let mut h = Harness::new();
        let ramp = h.add_ramp(&[0.0, 0.5, 1.0]);
        assert_eq!(h.settle(3), 0, "the idle canvas already paints warnings");

        let (_, y) = h.ramp_bar(ramp);
        let on = egui::pos2(h.indicator_x(ramp, 1), y);
        assert_eq!(h.menu_click(on, "Delete"), 0, "deleting a stop shuffled ids");
        assert_eq!(h.stops(ramp).iter().map(|s| s.t).collect::<Vec<_>>(), vec![0.0, 1.0]);
    }

    /// A ramp needs two stops to interpolate between. Delete stays listed at
    /// that floor so the menu keeps its shape, and must do nothing.
    #[test]
    fn the_last_two_stops_cannot_be_deleted() {
        let mut h = Harness::new();
        let ramp = h.add_ramp(&[0.0, 1.0]);
        assert_eq!(h.settle(3), 0, "the idle canvas already paints warnings");

        let (_, y) = h.ramp_bar(ramp);
        let on = egui::pos2(h.indicator_x(ramp, 1), y);
        assert_eq!(h.menu_click(on, "Delete"), 0, "the floor menu shuffled ids");
        assert_eq!(h.stops(ramp).len(), 2, "a ramp was left with too few stops");
    }

    /// The row's number field carries the same menu. egui doesn't bubble a
    /// secondary click from a child to its parent, so this is what breaks if
    /// the row relies on one interact behind its widgets.
    #[test]
    fn a_stop_row_opens_the_same_menu_as_its_indicator() {
        let mut h = Harness::new();
        let ramp = h.add_ramp(&[0.0, 0.5, 1.0]);
        assert_eq!(h.settle(3), 0, "the idle canvas already paints warnings");

        // Covers the two-decimal display and the row menu at once. Scoped to
        // the ramp, because other nodes draw a "0.50" too.
        let field = h
            .find_text_in(h.node_rect(ramp), "0.50")
            .expect("the middle stop shows its t as 0.50");
        assert_eq!(h.menu_click(field, "Duplicate"), 0, "the row menu shuffled ids");
        assert_eq!(h.stops(ramp).len(), 4, "the row menu did not duplicate");
    }

    /// Losing a stop from the middle leaves the indicators to its right on
    /// the same pixels while every index past the hole moves down. Anything
    /// keyed by index but placed by `t` then changes id without moving, which
    /// egui flags and which points an open menu at the wrong stop.
    ///
    /// Driven by `set_kind`, so it stays a test about removal itself.
    #[test]
    fn removing_a_stop_does_not_shuffle_widget_ids() {
        let mut h = Harness::new();
        let ramp = h.add_ramp(&[0.0, 0.5, 1.0]);
        assert_eq!(h.settle(3), 0, "the idle canvas already paints warnings");

        let mut stops = h.stops(ramp);
        stops.remove(1);
        h.graph
            .set_kind(
                ramp,
                LayerKind::ColorRamp(texture_graph_core::ColorRamp {
                    stops,
                    space: texture_graph_core::BlendSpace::Oklch,
                }),
            )
            .unwrap();
        assert_eq!(h.settle(3), 0, "dropping a stop shuffled ids");
    }

    /// The other way an index moves without its pixels: crossing swaps two
    /// stops in the vec while the one crossed never moves.
    ///
    /// egui doesn't flag this even with per-indicator ids — both ids survive
    /// the swap, so it reads the rect match as coincidence. The red count is
    /// a backstop; the crossing assertions are what earn their keep.
    #[test]
    fn dragging_a_stop_across_another_does_not_shuffle_widget_ids() {
        let mut h = Harness::new();
        let ramp = h.add_ramp(&[0.0, 0.4, 0.6, 1.0]);
        assert_eq!(h.settle(3), 0, "the idle canvas already paints warnings");

        let (_, y) = h.ramp_bar(ramp);
        let from = egui::pos2(h.indicator_x(ramp, 1), y);
        let to = egui::pos2(h.indicator_x(ramp, 2) + 8.0, y);
        assert_eq!(h.drag(from, to), 0, "crossing stops shuffled ids");

        let ts: Vec<f32> = h.stops(ramp).iter().map(|s| s.t).collect();
        assert!(ts.windows(2).all(|w| w[0] <= w[1]), "ramp came out unsorted: {ts:?}");
        assert!(ts[2] > 0.6, "the dragged stop did not cross: {ts:?}");
        assert_eq!(h.settle(3), 0, "the frames after the crossing shuffled ids");
    }

    /// One menu serves every indicator, so opening it depends on the
    /// right-click landing on one — and closing it has to be as explicit,
    /// since the close command rides along with a showing that won't happen.
    #[test]
    fn right_clicking_open_gradient_closes_an_open_stop_menu() {
        let mut h = Harness::new();
        let ramp = h.add_ramp(&[0.0, 0.5, 1.0]);
        assert_eq!(h.settle(3), 0, "the idle canvas already paints warnings");

        let (_, y) = h.ramp_bar(ramp);
        let on = h.indicator_x(ramp, 1);
        h.right_click(egui::pos2(on, y));
        h.settle(1);
        assert!(h.a_menu_is_open(), "the menu did not open to begin with");

        // Clear of every indicator, still on the bar.
        assert_eq!(h.right_click(egui::pos2(on + 30.0, y)), 0, "closing shuffled ids");
        h.settle(1);
        assert!(!h.a_menu_is_open(), "the menu outlived a click on open gradient");
        assert!(h.state.ramp_menu.is_none(), "the bar still names a stop");
        assert_eq!(h.stops(ramp).len(), 3, "the right-click changed the ramp");
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

    /// Inline widgets live in child `Ui`s, and one keyed by salt takes its
    /// auto-id seed from the parent's running counter. The rename field adds
    /// a child before all the rows, which would shift every slider and
    /// drag-value after it one slot along. Explicit ids prevent that, and
    /// this is what holds them.
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

    /// The same hazard reaching the rows: a node's row count follows its
    /// parameters, so switching a Mix off Blend drops two child `Ui`s from
    /// the middle of the canvas and everything after shifts a slot while
    /// staying put — exactly what `warn_if_rect_changes_id` catches.
    ///
    /// The Output is downstream here and keeps two sliders to shift onto.
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
