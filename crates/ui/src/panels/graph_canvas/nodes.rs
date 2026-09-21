//! Node pass: body interaction, chrome, thumbnails, widget rows, sockets.
//!
//! Registration order sets hit-test precedence, since egui's top-most wins.
//! Body first, then widgets, then sockets: sockets beat widgets beat drag.

use std::sync::Arc;

use texture_graph_core::{
    Axis, BlendMode, BlendSpace, Color, ColorInput, ColorRamp, ColorStop, CoordMode, Criterion,
    EvalCtx, Graph, InputKey, LayerId, LayerKind, MinMaxMode, NoiseKernel, RadialDim,
    ParamUse, ScalarInput, noise::MAX_OCTAVES,
};

use crate::app::GpuBits;
use crate::color_convert::{oklcha_to_srgba, srgba_to_oklcha};
use crate::previews::PreviewCache;
use crate::state::{
    EditCmd, NodeDrag, NodeRef, RampDrag, RampMenu, Renaming, UiState, WireDrag,
};
use crate::widgets::enum_combo::enum_combo;
use texture_graph_core::color::oklcha;

use crate::widgets::{node_labels, param_ref};

use super::layout::{socket_label, NodeLayout, ParamRow, Row};
use super::ramp;
use super::{socket_hit_radius, HEADER_H, SOCKET_R, ZOOM_LABELS_MIN, ZOOM_WIDGETS_MIN};

/// Whether anything claimed the pointer, which suppresses background pan.
pub fn draw_and_interact_nodes(
    ui: &mut egui::Ui,
    painter: &egui::Painter,
    graph: &Graph,
    state: &mut UiState,
    layouts: &[NodeLayout],
    canvas_rect: egui::Rect,
    previews: &mut PreviewCache,
    eval_ctx: &EvalCtx,
    mut gpu: Option<&mut GpuBits>,
) -> bool {
    let mut hit = false;
    let style = row_style(ui.style(), state.canvas_zoom);
    for layout in layouts {
        // Skipping off-screen nodes whole spends the frame's bake budget on
        // thumbnails somebody can see. The margin covers the sockets, which
        // sit on the edge and reach past the rect.
        let margin = socket_hit_radius(state.canvas_zoom);
        if !layout.rect.expand(margin).intersects(canvas_rect) && !held(state, layout.node) {
            continue;
        }
        hit |= body_interact(ui, graph, state, layout, canvas_rect);
        let renaming = matches!(layout.node, NodeRef::Layer(id)
            if state.renaming.as_ref().is_some_and(|r| r.node == id));
        draw_chrome(painter, graph, state, layout, renaming);
        // After the body, so the field beats the node drag.
        hit |= title(ui, graph, state, layout);
        if let (NodeRef::Layer(id), Some(thumb)) = (layout.node, layout.thumb_rect) {
            draw_thumbnail(ui, painter, graph, thumb, id, previews, eval_ctx, gpu.as_deref_mut());
        }
        match layout.node {
            NodeRef::Layer(id) => {
                layer_rows(ui, painter, graph, state, layout, canvas_rect, &style, id, eval_ctx)
            }
            NodeRef::Output => {
                output_rows(ui, painter, graph, state, layout, canvas_rect, &style)
            }
        }
        hit |= sockets(ui, painter, graph, state, layout);
    }
    hit
}

/// Whether this node must be drawn wherever it is, on screen or not.
///
/// A node being dragged, or whose ramp stop is, resolves that drag against
/// the pointer and would be dropped where it left the view. A node being
/// renamed would take the keyboard focus the rename commits by with it.
fn held(state: &UiState, node: NodeRef) -> bool {
    state.drag.is_some_and(|d| d.node == node)
        || matches!(node, NodeRef::Layer(id)
            if state.ramp_drag.is_some_and(|d| d.node == id)
                || state.renaming.as_ref().is_some_and(|r| r.node == id))
}

// ---- Body ---------------------------------------------------------------

/// Offsets a duplicate down and right, so the copy reads as a second node.
const DUPLICATE_OFFSET: f32 = 40.0;

fn body_interact(
    ui: &mut egui::Ui,
    graph: &Graph,
    state: &mut UiState,
    layout: &NodeLayout,
    canvas_rect: egui::Rect,
) -> bool {
    let resp = ui.interact(
        layout.rect,
        egui::Id::new(("graph-node", layout.node)),
        egui::Sense::click_and_drag(),
    );
    let mut hit = false;
    if resp.clicked() {
        if let NodeRef::Layer(id) = layout.node {
            state.selected = Some(id);
        }
        hit = true;
    }
    if resp.drag_started() {
        let ptr = ui
            .ctx()
            .input(|i| i.pointer.hover_pos())
            .unwrap_or(layout.rect.center());
        state.drag = Some(NodeDrag {
            node: layout.node,
            grab_offset: layout.rect.min - ptr,
        });
        if let NodeRef::Layer(id) = layout.node {
            state.selected = Some(id);
        }
        hit = true;
    }
    if resp.dragged() {
        hit = true;
    }
    match layout.node {
        NodeRef::Layer(id) => {
            let world = super::screen_to_world(layout.rect.min, canvas_rect.min, state);
            resp.context_menu(|ui| {
                // Arms the header field rather than typing here; a menu is
                // where people look for "rename".
                if ui.button("Rename").clicked() {
                    state.renaming = Some(Renaming {
                        node: id,
                        text: graph.get(id).map(|l| l.name.clone()).unwrap_or_default(),
                        focused: false,
                    });
                    ui.close();
                }
                // This layer's color alone, over default PBR channels.
                if ui.button("Preview").clicked() {
                    state.preview_target = Some(id);
                    ui.close();
                }
                if ui.button("Duplicate").clicked() {
                    if let Some(layer) = graph.get(id) {
                        let name = crate::catalog::unique_name(graph, &layer.name);
                        let pos = [world.x + DUPLICATE_OFFSET, world.y + DUPLICATE_OFFSET];
                        state.push(EditCmd::AddLayer {
                            name,
                            kind: layer.kind.clone(),
                            pos: Some((state.active_canvas.clone(), pos)),
                        });
                    }
                    ui.close();
                }
                if ui.button("Delete").clicked() {
                    state.push(EditCmd::Remove(id));
                    ui.close();
                }
            });
        }
        NodeRef::Output => {
            resp.context_menu(|ui| {
                if ui.button("Preview").clicked() {
                    state.preview_target = None;
                    ui.close();
                }
            });
        }
    }
    hit
}

fn draw_chrome(
    painter: &egui::Painter,
    graph: &Graph,
    state: &UiState,
    layout: &NodeLayout,
    renaming: bool,
) {
    let z = state.canvas_zoom;
    let selected = matches!(layout.node, NodeRef::Layer(id) if state.selected == Some(id));
    let bg = if selected {
        egui::Color32::from_rgb(60, 90, 120)
    } else {
        egui::Color32::from_gray(45)
    };
    let stroke = egui::Stroke::new(
        if selected { 2.0 } else { 1.0 },
        egui::Color32::from_gray(if selected { 240 } else { 180 }),
    );
    painter.rect(layout.rect, 6.0 * z, bg, stroke, egui::StrokeKind::Middle);

    let header_bg = match layout.node {
        NodeRef::Layer(_) => None,
        NodeRef::Output => Some(egui::Color32::from_rgb(110, 70, 30)),
    };
    if let Some(c) = header_bg {
        let header = egui::Rect::from_min_size(
            layout.rect.min,
            egui::vec2(layout.rect.width(), HEADER_H * z),
        );
        painter.rect_filled(header, 6.0 * z, c);
    }

    let (name, category) = match layout.node {
        NodeRef::Layer(id) => match graph.get(id) {
            Some(l) => (l.name.clone(), l.kind.category_label()),
            None => return,
        },
        NodeRef::Output => ("Material Output".to_string(), "Output"),
    };
    // While it is being typed over, the field is the title.
    if !renaming {
        painter.text(
            layout.rect.min + egui::vec2(10.0, 4.0) * z,
            egui::Align2::LEFT_TOP,
            name,
            egui::FontId::proportional(13.0 * z),
            egui::Color32::WHITE,
        );
    }
    painter.text(
        layout.rect.min + egui::vec2(10.0, 19.0) * z,
        egui::Align2::LEFT_TOP,
        category,
        egui::FontId::proportional(10.0 * z),
        egui::Color32::from_gray(190),
    );
}

// ---- Title --------------------------------------------------------------

/// The name's line in the header: click to rename, field appears in place.
fn title_rect(node: egui::Rect, z: f32) -> egui::Rect {
    egui::Rect::from_min_size(
        node.min + egui::vec2(8.0, 2.0) * z,
        egui::vec2(node.width() - 16.0 * z, 18.0 * z),
    )
}

/// Enter or clicking away commits, Escape discards.
///
/// The text lives in `UiState`, not the graph: names must be unique, and
/// half-typed text needs somewhere to sit while it is briefly a duplicate.
/// Committing queues a `Rename` like any edit, so a refusal lands in the
/// status row and the title stays put.
///
/// Returns whether the field claimed the pointer.
fn title(ui: &mut egui::Ui, graph: &Graph, state: &mut UiState, layout: &NodeLayout) -> bool {
    let NodeRef::Layer(id) = layout.node else { return false };
    let z = state.canvas_zoom;
    // Too far out for the header to be legible, let alone typed into.
    if z < ZOOM_WIDGETS_MIN {
        return false;
    }
    let rect = title_rect(layout.rect, z);
    let editing = state.renaming.as_ref().is_some_and(|r| r.node == id);

    // Both branches register one widget, at one rect, under one id. Changing
    // the shape of what is registered here is what `warn_if_rect_changes_id`
    // watches for, and it hands in-flight interactions to the wrong widget.
    let mut child = ui.new_child(
        egui::UiBuilder::new()
            .id(egui::Id::new(("graph-title", id.0)))
            .max_rect(rect),
    );

    if !editing {
        let resp = child.allocate_rect(rect, egui::Sense::click());
        if resp.clicked() {
            state.renaming = Some(Renaming {
                node: id,
                text: graph.get(id).map(|l| l.name.clone()).unwrap_or_default(),
                focused: false,
            });
            return true;
        }
        return false;
    }

    let mut text = state
        .renaming
        .as_ref()
        .map(|r| r.text.clone())
        .unwrap_or_default();
    let focused = state.renaming.as_ref().is_some_and(|r| r.focused);
    let resp = child.add_sized(
        rect.size(),
        egui::TextEdit::singleline(&mut text)
            .font(egui::FontId::proportional(13.0 * z))
            .margin(egui::Margin::symmetric((2.0 * z) as i8, 0)),
    );
    if !focused {
        // Asking every frame would make the field impossible to blur, and
        // blurring commits.
        resp.request_focus();
    }

    // Escape surrenders focus too, so it has to be read before the blur or a
    // cancelled rename commits.
    if ui.ctx().input(|i| i.key_pressed(egui::Key::Escape)) {
        state.renaming = None;
    } else if resp.lost_focus() {
        state.renaming = None;
        if let Some(cmd) = rename_command(graph, id, &text) {
            state.push(cmd);
        }
    } else {
        state.renaming = Some(Renaming { node: id, text, focused: true });
    }
    true
}

/// The edit a committed title amounts to, or `None` for a no-op. Reading a
/// title and clicking away must not mark the file unsaved; an emptied field
/// is the same case, and the model would refuse it anyway.
fn rename_command(graph: &Graph, id: LayerId, text: &str) -> Option<EditCmd> {
    let trimmed = text.trim();
    let current = &graph.get(id)?.name;
    (!trimmed.is_empty() && current != trimmed)
        .then(|| EditCmd::Rename(id, trimmed.to_string()))
}

#[allow(clippy::too_many_arguments)]
fn draw_thumbnail(
    ui: &egui::Ui,
    painter: &egui::Painter,
    graph: &Graph,
    thumb_row: egui::Rect,
    id: LayerId,
    previews: &mut PreviewCache,
    eval_ctx: &EvalCtx,
    gpu: Option<&mut GpuBits>,
) {
    let side = thumb_row.height() - 8.0;
    if side < 2.0 {
        return;
    }
    let rect = egui::Rect::from_center_size(thumb_row.center(), egui::Vec2::splat(side));
    match previews.get_or_build(ui.ctx(), graph, id, eval_ctx, gpu) {
        Some(tid) => {
            painter.image(
                tid,
                rect,
                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                egui::Color32::WHITE,
            );
        }
        None => {
            painter.rect_filled(rect, 2.0, egui::Color32::DARK_GRAY);
        }
    }
}

// ---- Inline widget rows -------------------------------------------------

/// Child Ui spanning one row, styled to the canvas zoom so fonts and
/// interact sizes track the node.
///
/// The id is explicit, not a salt. A salted child takes its id — and the
/// auto-ids of every widget inside it — from the parent's running counter,
/// so a row's widgets would depend on how many child Uis happened to come
/// before them: the rename field appearing, or a node scrolling out of view,
/// would swap ids under them and land an in-flight drag on the wrong one.
fn row_ui(
    ui: &mut egui::Ui,
    rect: egui::Rect,
    zoom: f32,
    canvas_rect: egui::Rect,
    style: &Arc<egui::Style>,
    salt: (NodeRef, usize),
) -> egui::Ui {
    let inner = rect.shrink2(egui::vec2(10.0 * zoom, 0.0));
    let mut child = ui.new_child(
        egui::UiBuilder::new()
            .id(egui::Id::new(("node-row", salt)))
            .max_rect(inner)
            .layout(egui::Layout::left_to_right(egui::Align::Center)),
    );
    child.set_clip_rect(canvas_rect.intersect(rect));
    child.set_style(style.clone());
    child
}

/// Depends on the canvas zoom and nothing else, so it is built once a frame
/// and shared: per-row it would be hundreds of `Style` clones.
fn row_style(base: &egui::Style, zoom: f32) -> Arc<egui::Style> {
    let mut style = base.clone();
    for font in style.text_styles.values_mut() {
        font.size *= zoom;
    }
    style.spacing.item_spacing = egui::vec2(4.0, 2.0) * zoom;
    style.spacing.interact_size = egui::vec2(32.0, 16.0) * zoom;
    style.spacing.slider_width = 58.0 * zoom;
    style.spacing.combo_width = 60.0 * zoom;
    style.spacing.button_padding = egui::vec2(3.0, 1.0) * zoom;
    style.spacing.icon_width *= zoom;
    style.spacing.icon_width_inner *= zoom;
    Arc::new(style)
}

/// Compact swatch, no LCh popover; nodes are tight and the inspector has one.
fn color_swatch(ui: &mut egui::Ui, color: &mut texture_graph_core::Color) -> egui::Response {
    let mut rgba = oklcha_to_srgba(*color);
    let resp = ui.color_edit_button_rgba_unmultiplied(&mut rgba);
    if resp.changed() {
        *color = srgba_to_oklcha(rgba);
    }
    resp
}

/// Painter-only socket label, for zoom too low to bother with widgets.
fn painter_row_label(painter: &egui::Painter, rect: egui::Rect, zoom: f32, label: &str) {
    painter.text(
        egui::pos2(rect.left() + 10.0 * zoom, rect.center().y),
        egui::Align2::LEFT_CENTER,
        label,
        egui::FontId::proportional(11.0 * zoom),
        egui::Color32::from_gray(200),
    );
}

#[allow(clippy::too_many_arguments)]
fn layer_rows(
    ui: &mut egui::Ui,
    painter: &egui::Painter,
    graph: &Graph,
    state: &mut UiState,
    layout: &NodeLayout,
    canvas_rect: egui::Rect,
    style: &Arc<egui::Style>,
    id: LayerId,
    eval_ctx: &EvalCtx,
) {
    let z = state.canvas_zoom;
    let Some(layer) = graph.get(id) else { return };

    if z < ZOOM_WIDGETS_MIN {
        if z >= ZOOM_LABELS_MIN {
            for (row, rect) in layout.rows.iter().zip(&layout.row_rects) {
                match row {
                    Row::Socket(key) => painter_row_label(painter, *rect, z, socket_label(*key)),
                    Row::Param(ParamRow::RampBar) => {
                        // Recognizable when zoomed out, but not interactive.
                        if let LayerKind::ColorRamp(r) = &layer.kind {
                            paint_ramp_bar(painter, graph, r, *rect, z, eval_ctx, None);
                        }
                    }
                    Row::Param(_) => {}
                }
            }
        }
        return;
    }

    let mut kind = layer.kind.clone();
    let mut changed = false;
    for (i, (row, rect)) in layout.rows.iter().zip(&layout.row_rects).enumerate() {
        match *row {
            Row::Param(ParamRow::RampBar) => {
                if let LayerKind::ColorRamp(r) = &mut kind {
                    changed |= ramp_bar(ui, painter, graph, state, id, r, *rect, eval_ctx);
                }
            }
            Row::Socket(key) => {
                let mut r = row_ui(ui, *rect, z, canvas_rect, style, (layout.node, i));
                changed |= socket_row(
                    &mut r, graph, eval_ctx, &mut kind, key, state, layout.node,
                );
            }
            Row::Param(p) => {
                let mut r = row_ui(ui, *rect, z, canvas_rect, style, (layout.node, i));
                changed |= param_row(&mut r, id, &mut kind, p);
            }
        }
    }
    if changed {
        state.push(EditCmd::SetKind(id, kind));
    }
}

// ---- Ramp stops ---------------------------------------------------------

/// What a stop's right-click menu asked for.
#[derive(Clone, Debug, Eq, PartialEq)]
enum StopAction {
    Duplicate,
    Delete,
    /// Read a declared color parameter instead of a constant.
    BindParam(String),
    /// Stop reading one, freezing the stop at what it currently shows.
    Unbind,
}

/// The items themselves, for whichever menu is showing them. Delete greys
/// out at the two-stop floor rather than vanishing, so the menu keeps its
/// shape.
fn stop_menu_items(
    ui: &mut egui::Ui,
    can_delete: bool,
    graph: &Graph,
    bound: bool,
) -> Option<StopAction> {
    let mut action = None;
    if ui.button("Duplicate").clicked() {
        action = Some(StopAction::Duplicate);
        ui.close();
    }
    if ui
        .add_enabled(can_delete, egui::Button::new("Delete"))
        .clicked()
    {
        action = Some(StopAction::Delete);
        ui.close();
    }
    // A stop is the one ColorInput in the editor, so this menu is where
    // a palette gets parameterised.
    if bound {
        if ui.button("Unbind parameter").clicked() {
            action = Some(StopAction::Unbind);
            ui.close();
        }
    } else {
        let mut any = false;
        for name in param_ref::usable(graph, ParamUse::Color) {
            any = true;
            if ui.button(format!("Read “{name}”")).clicked() {
                action = Some(StopAction::BindParam(name.to_string()));
                ui.close();
            }
        }
        if !any {
            ui.add_enabled(false, egui::Button::new("No color parameters"));
        }
    }
    action
}

/// The row's menu, hung on its number, its swatch or the bare stretch
/// beside them.
fn stop_menu(
    resp: &egui::Response,
    can_delete: bool,
    graph: &Graph,
    bound: bool,
) -> Option<StopAction> {
    let mut action = None;
    resp.context_menu(|ui| {
        action = stop_menu_items(ui, can_delete, graph, bound);
    });
    action
}

/// Carry out `action`. `saved_consts` keys are positional, so an insert or
/// remove has to drag them along; see [`UiState::remap_ramp_consts`].
fn apply_stop_action(
    r: &mut ColorRamp,
    i: usize,
    action: StopAction,
    state: &mut UiState,
    node: NodeRef,
    graph: &Graph,
    eval_ctx: &EvalCtx,
) -> bool {
    match action {
        StopAction::BindParam(name) => {
            let Some(stop) = r.stops.get_mut(i) else { return false };
            stop.color = ColorInput::Param(name);
            true
        }
        StopAction::Unbind => {
            let Some(stop) = r.stops.get_mut(i) else { return false };
            let ColorInput::Param(name) = &stop.color else { return false };
            // Freeze at what it shows, so unbinding is not also a repaint.
            let frozen = graph
                .param_value(name, eval_ctx)
                .and_then(|v| v.as_color())
                .unwrap_or_else(|| oklcha(0.5, 0.0, 0.0, 1.0));
            stop.color = ColorInput::Const(frozen);
            true
        }
        StopAction::Duplicate => {
            let Some(src) = r.stops.get(i).cloned() else {
                return false;
            };
            let t = ramp::duplicate_t(&r.stops, i);
            let j = ramp::insert_index(&r.stops, t);
            r.stops.insert(j, ColorStop { t, ..src });
            // The copy inherits no saved const: those belong to the stop a
            // wire displaced one on, and the remap moves keys, not copies.
            state.remap_ramp_consts(node, move |k| Some(if k >= j { k + 1 } else { k }));
            true
        }
        StopAction::Delete => {
            // The floor the data depends on; the menu greys out there too.
            if i >= r.stops.len() || r.stops.len() <= 2 {
                return false;
            }
            r.stops.remove(i);
            state.remap_ramp_consts(node, move |k| match k.cmp(&i) {
                std::cmp::Ordering::Less => Some(k),
                std::cmp::Ordering::Equal => None,
                std::cmp::Ordering::Greater => Some(k - 1),
            });
            true
        }
    }
}

// ---- Ramp bar -----------------------------------------------------------

/// Grab radius around an indicator: its half-width plus slop, floored so it
/// stays grabbable zoomed out. Generous, because a near miss drags nothing
/// at all and the row's height is reserved for this one gesture.
fn ramp_grab_radius(z: f32) -> f32 {
    (4.0 * z).max(4.0) + 5.0
}

/// Gradient strip and the arrow strip below it. Vertically, in world units:
/// 3 pad + 22 gradient + 12 arrows + 3 pad = RAMP_BAR_H.
pub(super) fn ramp_bar_rects(rect: egui::Rect, z: f32) -> (egui::Rect, egui::Rect) {
    let inner = rect.shrink2(egui::vec2(10.0 * z, 0.0));
    let grad = egui::Rect::from_min_size(
        inner.min + egui::vec2(0.0, 3.0 * z),
        egui::vec2(inner.width(), 22.0 * z),
    );
    let arrows = egui::Rect::from_min_size(
        egui::pos2(inner.left(), grad.bottom()),
        egui::vec2(inner.width(), 12.0 * z),
    );
    (grad, arrows)
}

/// Opaque display color for gradient/indicator painting.
fn ramp_color32(c: Color) -> egui::Color32 {
    let [r, g, b, _] = oklcha_to_srgba(c);
    egui::Color32::from_rgb(
        (r * 255.0 + 0.5) as u8,
        (g * 255.0 + 0.5) as u8,
        (b * 255.0 + 0.5) as u8,
    )
}

/// Gradient mesh and indicators; `highlight` outlines one white. Paint-only,
/// shared by the interactive widget and the low-zoom path.
fn paint_ramp_bar(
    painter: &egui::Painter,
    graph: &Graph,
    ramp: &ColorRamp,
    rect: egui::Rect,
    z: f32,
    eval_ctx: &EvalCtx,
    highlight: Option<usize>,
) {
    let (grad, arrows) = ramp_bar_rects(rect, z);
    if grad.width() < 1.0 || grad.height() < 1.0 {
        return;
    }
    let display: Vec<(f32, Color)> = ramp
        .stops
        .iter()
        .map(|s| (s.t, ramp::stop_display_color(graph, s, eval_ctx)))
        .collect();

    // Uniform coverage plus every interior stop, so a hard cusp lands on a
    // vertex instead of straddling a segment.
    const SEGMENTS: usize = 48;
    let mut xs: Vec<f32> = (0..=SEGMENTS).map(|i| i as f32 / SEGMENTS as f32).collect();
    xs.extend(display.iter().map(|(t, _)| *t).filter(|t| *t > 0.0 && *t < 1.0));
    xs.sort_by(f32::total_cmp);
    xs.dedup();

    let mut mesh = egui::Mesh::default();
    let at = |u: f32| grad.left() + u * grad.width();
    let mut prev = (at(xs[0]), ramp_color32(ramp::sample_display(&display, ramp.space, xs[0])));
    for &u in &xs[1..] {
        let cur = (at(u), ramp_color32(ramp::sample_display(&display, ramp.space, u)));
        let i0 = mesh.vertices.len() as u32;
        mesh.colored_vertex(egui::pos2(prev.0, grad.top()), prev.1);
        mesh.colored_vertex(egui::pos2(cur.0, grad.top()), cur.1);
        mesh.colored_vertex(egui::pos2(cur.0, grad.bottom()), cur.1);
        mesh.colored_vertex(egui::pos2(prev.0, grad.bottom()), prev.1);
        mesh.add_triangle(i0, i0 + 1, i0 + 2);
        mesh.add_triangle(i0, i0 + 2, i0 + 3);
        prev = cur;
    }
    painter.add(egui::Shape::mesh(mesh));
    painter.rect_stroke(
        grad,
        0.0,
        egui::Stroke::new(1.0, egui::Color32::from_gray(20)),
        egui::StrokeKind::Middle,
    );

    // In stop order, so a later stop paints on top at coincident t.
    let hw = 4.0 * z;
    let shoulder = arrows.top() + 5.0 * z;
    for (i, (t, c)) in display.iter().enumerate() {
        let x = at(t.clamp(0.0, 1.0));
        let points = vec![
            egui::pos2(x, arrows.top()),
            egui::pos2(x + hw, shoulder),
            egui::pos2(x + hw, arrows.bottom()),
            egui::pos2(x - hw, arrows.bottom()),
            egui::pos2(x - hw, shoulder),
        ];
        let stroke_color = if highlight == Some(i) {
            egui::Color32::WHITE
        } else {
            egui::Color32::BLACK
        };
        painter.add(egui::Shape::convex_polygon(
            points,
            ramp_color32(*c),
            egui::Stroke::new((1.0 * z).max(1.0), stroke_color),
        ));
    }
}

/// Interactive bar: drag indicators (crossing reorders), double-click the
/// gradient to add a stop.
#[allow(clippy::too_many_arguments)]
fn ramp_bar(
    ui: &mut egui::Ui,
    painter: &egui::Painter,
    graph: &Graph,
    state: &mut UiState,
    id: LayerId,
    ramp: &mut ColorRamp,
    rect: egui::Rect,
    eval_ctx: &EvalCtx,
) -> bool {
    let z = state.canvas_zoom;
    let (grad, _) = ramp_bar_rects(rect, z);
    if grad.width() < 1.0 {
        return false;
    }
    let grab = ramp_grab_radius(z);
    let ind_x = |t: f32| grad.left() + t.clamp(0.0, 1.0) * grad.width();
    // Unclamped, so a grab offset stays meaningful past either end.
    let pointer_t = |x: f32| (x - grad.left()) / grad.width();
    // `<=` so the topmost-painted stop wins a tie at coincident t.
    let indicator_at = |stops: &[ColorStop], x: f32| -> Option<usize> {
        let mut best: Option<(usize, f32)> = None;
        for (i, s) in stops.iter().enumerate() {
            let d = (ind_x(s.t) - x).abs();
            if d <= grab && best.is_none_or(|(_, bd)| d <= bd) {
                best = Some((i, d));
            }
        }
        best.map(|(i, _)| i)
    };

    // One interact for the whole bar: a stable id keeps drag ownership
    // through a crossing reorder, which moves indices.
    let resp = ui.interact(
        rect,
        egui::Id::new(("ramp-bar", id.0)),
        egui::Sense::click_and_drag(),
    );
    let mut changed = false;

    // Arm on the press, not `drag_started`: egui withholds that until the
    // pointer has travelled `max_click_dist`, and `interact_pointer_pos`
    // reports where it is now, so the hit test would run a grab radius from
    // where the user aimed. `is_pointer_button_down_on` is set on the press
    // frame and confirms the press is the bar's.
    if ui.ctx().input(|i| i.pointer.primary_pressed()) && resp.is_pointer_button_down_on() {
        if let Some(p) = resp.interact_pointer_pos() {
            if let Some(i) = indicator_at(&ramp.stops, p.x) {
                state.ramp_drag = Some(RampDrag {
                    node: id,
                    stop: i,
                    // Against where the indicator is drawn, which `ind_x`
                    // clamps: a `t` loaded out of range would otherwise set
                    // the offset to however far out it sat.
                    grab_dt: ramp.stops[i].t.clamp(0.0, 1.0) - pointer_t(p.x),
                });
            }
        }
    }
    // From global pointer state, not `resp.dragged()`: the drag stays stuck
    // to the indicator when the pointer overshoots the bar. Release is
    // button-up, in `release_stale_ramp_drag`.
    if let Some(drag) = state.ramp_drag.filter(|d| d.node == id) {
        // `interact_pos`, not `hover_pos`: the latter goes `None` when the
        // pointer leaves the window, stalling the indicator mid-drag.
        if let Some(p) = ui.ctx().input(|i| i.pointer.interact_pos()) {
            if drag.stop < ramp.stops.len() {
                // Carry the grab offset, so the indicator travels with the
                // cursor instead of snapping its centre under it.
                let t = (pointer_t(p.x) + drag.grab_dt).clamp(0.0, 1.0);
                // A press that hasn't moved is not an edit; pushing SetKind
                // anyway re-bakes every downstream thumbnail per frame.
                if ramp.stops[drag.stop].t != t {
                    let (new_i, swaps) = ramp::drag_stop_to(&mut ramp.stops, drag.stop, t);
                    for (a, b) in swaps {
                        state.remap_ramp_consts(NodeRef::Layer(id), move |j| {
                            Some(if j == a {
                                b
                            } else if j == b {
                                a
                            } else {
                                j
                            })
                        });
                    }
                    state.ramp_drag = Some(RampDrag { stop: new_i, ..drag });
                    changed = true;
                }
            }
        }
    }
    if resp.double_clicked() {
        if let Some(p) = resp.interact_pointer_pos() {
            // Near an indicator it's a mis-click.
            if indicator_at(&ramp.stops, p.x).is_none() {
                // The row is wider than its gradient; a press in the padding
                // adds a stop at the near end.
                let t = pointer_t(p.x).clamp(0.0, 1.0);
                let display: Vec<(f32, Color)> = ramp
                    .stops
                    .iter()
                    .map(|s| (s.t, ramp::stop_display_color(graph, s, eval_ctx)))
                    .collect();
                let color = ramp::sample_display(&display, ramp.space, t);
                let i = ramp::insert_index(&ramp.stops, t);
                ramp.stops.insert(i, ColorStop { t, color: ColorInput::Const(color) });
                state.remap_ramp_consts(NodeRef::Layer(id), move |j| {
                    Some(if j >= i { j + 1 } else { j })
                });
                changed = true;
            }
        }
    }

    // The indicators share the bar's response rather than each holding an
    // interact rect. A per-indicator id would be the stop's index while its
    // rect follows the stop's `t`, and removing a stop from the middle parts
    // those: the indicator to its right keeps its pixels under a new id,
    // which egui flags and which would point an open menu at the wrong stop.
    let menu_id = egui::Popup::default_response_id(&resp);
    let secondary = resp.secondary_clicked();
    let hit = secondary
        .then(|| resp.interact_pointer_pos())
        .flatten()
        .and_then(|p| indicator_at(&ramp.stops, p.x));
    if secondary {
        // Open gradient closes the menu rather than opening one about
        // whichever stop is nearest. It has to be said explicitly: only
        // `show` applies the command below, and there is nothing to show.
        state.ramp_menu = hit.map(|stop| RampMenu { node: id, stop });
        if hit.is_none() {
            egui::Popup::close_id(ui.ctx(), menu_id);
        }
    } else if state.ramp_menu.is_some_and(|m| m.node == id)
        && !egui::Popup::is_id_open(ui.ctx(), menu_id)
    {
        // It can also close without the bar hearing the click — inside the
        // popup, or Escape — so the stop must not outlive it.
        state.ramp_menu = None;
    }
    // `Response::context_menu`, minus opening on every secondary click.
    let open = if hit.is_some() {
        Some(egui::SetOpenCommand::Bool(true))
    } else if secondary || resp.clicked() {
        Some(egui::SetOpenCommand::Bool(false))
    } else {
        None
    };
    if let Some(menu) = state
        .ramp_menu
        .filter(|m| m.node == id && m.stop < ramp.stops.len())
    {
        let mut action = None;
        egui::Popup::menu(&resp)
            .open_memory(open)
            .at_pointer_fixed()
            .show(|ui| {
                action = stop_menu_items(
                    ui,
                    ramp.stops.len() > 2,
                    graph,
                    ramp.stops
                        .get(menu.stop)
                        .is_some_and(|s| matches!(s.color, ColorInput::Param(_))),
                )
            });
        if let Some(a) = action {
            changed |= apply_stop_action(
                ramp, menu.stop, a, state, NodeRef::Layer(id), graph, eval_ctx,
            );
            // Both actions move the indices, and the menu closed on the
            // click, so don't leave it naming whatever slid into that slot.
            state.ramp_menu = None;
        }
    }

    // Paint after interaction so indicators track this frame's drag.
    let dragging = state.ramp_drag.filter(|d| d.node == id).map(|d| d.stop);
    let hovering = resp
        .contains_pointer()
        .then(|| ui.ctx().input(|i| i.pointer.hover_pos()))
        .flatten()
        .and_then(|p| indicator_at(&ramp.stops, p.x));
    // The grab radius reaches past the outline, so the cursor is what tells
    // you you're inside it before committing to the press.
    if dragging.is_some() || hovering.is_some() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeHorizontal);
    }
    paint_ramp_bar(painter, graph, ramp, rect, z, eval_ctx, dragging.or(hovering));
    changed
}

/// Label plus the const widget, if unconnected and the type has one.
#[allow(clippy::too_many_arguments)]
fn socket_row(
    ui: &mut egui::Ui,
    graph: &Graph,
    eval_ctx: &EvalCtx,
    kind: &mut LayerKind,
    key: InputKey,
    state: &mut UiState,
    node: NodeRef,
) -> bool {
    let mut changed = false;
    match (&mut *kind, key) {
        (LayerKind::Mix(m), InputKey::MixFactor) => {
            ui.label(socket_label(key));
            changed |= param_ref::scalar_socket(
                ui,
                graph,
                eval_ctx,
                ("mix-factor", node),
                &mut m.factor,
                |ui, v| ui.add(egui::Slider::new(v, 0.0..=1.0)),
            );
        }
        (LayerKind::Wave(w), InputKey::WaveInput) => {
            ui.label(socket_label(key));
            changed |= param_ref::scalar_socket(
                ui,
                graph,
                eval_ctx,
                ("wave-input", node),
                &mut w.input,
                |ui, v| ui.add(egui::DragValue::new(v).speed(0.01)),
            );
        }
        (LayerKind::ColorRamp(r), InputKey::RampStop(i)) => {
            let can_delete = r.stops.len() > 2;
            let bound = r
                .stops
                .get(i)
                .is_some_and(|s| matches!(s.color, ColorInput::Param(_)));
            // egui doesn't bubble a secondary click from a child to its
            // parent, so the row's widgets carry the menu themselves and this
            // covers what they leave bare. Registered first, so they keep
            // first claim on the pointer for everything else.
            let mut action = stop_menu(
                &ui.interact(
                    ui.max_rect(),
                    egui::Id::new(("ramp-stop-row", node, i)),
                    egui::Sense::click(),
                ),
                can_delete,
                graph,
                bound,
            );
            if let Some(stop) = r.stops.get_mut(i) {
                let t = crate::widgets::ramp_stop::stop_t(ui, &mut stop.t);
                changed |= t.changed();
                action = action.or_else(|| stop_menu(&t, can_delete, graph, bound));
                match &mut stop.color {
                    ColorInput::Const(c) => {
                        let swatch = color_swatch(ui, c);
                        changed |= swatch.changed();
                        action = action.or_else(|| stop_menu(&swatch, can_delete, graph, bound));
                    }
                    ColorInput::Param(name) => {
                        // Read-only: the value belongs to the parameter,
                        // and the panel on the left is where it is edited.
                        let shown = graph
                            .param_value(name, eval_ctx)
                            .and_then(|v| v.as_color())
                            .unwrap_or_else(|| oklcha(0.7017, 0.3223, 328.36, 1.0));
                        let mut shown = shown;
                        let swatch = color_swatch(ui, &mut shown);
                        let label = ui.label(egui::RichText::new(name.as_str()).small());
                        action = action
                            .or_else(|| stop_menu(&swatch, can_delete, graph, bound))
                            .or_else(|| stop_menu(&label, can_delete, graph, bound));
                    }
                    ColorInput::Layer(_) => {}
                }
            }
            if let Some(a) = action {
                changed |= apply_stop_action(r, i, a, state, node, graph, eval_ctx);
            }
        }
        _ => {
            ui.label(socket_label(key));
        }
    }
    changed
}

fn param_row(ui: &mut egui::Ui, id: LayerId, kind: &mut LayerKind, p: ParamRow) -> bool {
    let salt = |field: &'static str| ("canvas", id.0, field);
    match (&mut *kind, p) {
        (LayerKind::Color(c), ParamRow::ColorValue) => {
            ui.label("color");
            color_swatch(ui, c).changed()
        }
        (LayerKind::Noise(n), ParamRow::NoiseKernel) => {
            noise_kernel_combo(ui, salt("kernel"), n)
        }
        (LayerKind::Noise(n), ParamRow::NoiseDims) => enum_combo(
            ui,
            salt("dims"),
            "dims",
            &mut n.dims,
            node_labels::DIMS,
            node_labels::dims,
        ),
        (LayerKind::Noise(n), ParamRow::NoiseOutput) => enum_combo(
            ui,
            salt("output"),
            "output",
            &mut n.output,
            node_labels::OUTPUTS,
            node_labels::output,
        ),
        (LayerKind::Noise(n), ParamRow::NoiseRange) => enum_combo(
            ui,
            salt("range"),
            "range",
            &mut n.range,
            node_labels::RANGES,
            node_labels::range,
        ),
        (LayerKind::Noise(n), ParamRow::NoiseFrequency) => {
            ui.label("freq");
            ui.add(egui::Slider::new(&mut n.frequency, 0.1..=64.0).logarithmic(true))
                .changed()
        }
        (LayerKind::Noise(n), ParamRow::NoiseSeed) => {
            ui.label("seed");
            ui.add(egui::DragValue::new(&mut n.seed_offset)).changed()
        }
        (LayerKind::Noise(n), ParamRow::NoisePeriod) => noise_period_row(ui, n),
        (LayerKind::Noise(n), ParamRow::NoiseOctaves) => {
            ui.label("octaves");
            ui.add(egui::Slider::new(&mut n.fractal.octaves, 1..=MAX_OCTAVES))
                .changed()
        }
        (LayerKind::Noise(n), ParamRow::NoiseFractalMode) => enum_combo(
            ui,
            salt("fractal-mode"),
            "fbm",
            &mut n.fractal.mode,
            node_labels::FRACTAL_MODES,
            node_labels::fractal_mode,
        ),
        (LayerKind::Noise(n), ParamRow::NoiseLacunarity) => {
            ui.label("lacunarity");
            ui.add(egui::Slider::new(&mut n.fractal.lacunarity, 1.0..=4.0))
                .changed()
        }
        (LayerKind::Noise(n), ParamRow::NoiseGain) => {
            ui.label("gain");
            ui.add(egui::Slider::new(&mut n.fractal.gain, 0.0..=1.0)).changed()
        }
        (LayerKind::Noise(n), ParamRow::NoiseNormalize) => {
            ui.label("normalize");
            ui.checkbox(&mut n.fractal.normalize, "").changed()
        }
        (LayerKind::Warp(w), ParamRow::WarpMode) => enum_combo(
            ui,
            salt("warp-mode"),
            "mode",
            &mut w.mode,
            node_labels::WARP_MODES,
            node_labels::warp_mode,
        ),
        (LayerKind::Warp(w), ParamRow::WarpAmount) => {
            vec3_row(ui, "amount", &mut w.amount, 0.005)
        }
        (LayerKind::Coordinate(c), ParamRow::CoordinateAxis) => enum_combo(
            ui,
            salt("coordinate-axis"),
            "axis",
            &mut c.axis,
            node_labels::AXES,
            node_labels::axis,
        ),
        (LayerKind::Wave(w), ParamRow::WaveShape) => enum_combo(
            ui,
            salt("wave-shape"),
            "shape",
            &mut w.shape,
            node_labels::SHAPES,
            node_labels::shape,
        ),
        (LayerKind::Wave(w), ParamRow::WaveFrequency) => {
            ui.label("freq");
            ui.add(egui::Slider::new(&mut w.frequency, 0.1..=64.0).logarithmic(true))
                .changed()
        }
        (LayerKind::Wave(w), ParamRow::WavePhase) => {
            ui.label("phase");
            ui.add(egui::Slider::new(&mut w.phase, 0.0..=1.0)).changed()
        }
        (LayerKind::Wave(w), ParamRow::WaveRange) => enum_combo(
            ui,
            salt("wave-range"),
            "range",
            &mut w.range,
            node_labels::RANGES,
            node_labels::range,
        ),
        (LayerKind::ColorRamp(r), ParamRow::RampSpace) => {
            blend_space_combo(ui, salt("space"), &mut r.space)
        }
        (LayerKind::Transform(t), ParamRow::TransformOffset) => {
            vec3_row(ui, "offset", &mut t.offset, 0.01)
        }
        (LayerKind::Transform(t), ParamRow::TransformRotate) => {
            ui.label("rotate");
            ui.add(egui::Slider::new(
                &mut t.rotate_uv,
                -std::f32::consts::PI..=std::f32::consts::PI,
            ))
            .changed()
        }
        (LayerKind::Transform(t), ParamRow::TransformScale) => {
            vec3_row(ui, "scale", &mut t.scale, 0.05)
        }
        (LayerKind::Transform(t), ParamRow::TransformCoordMode) => {
            let mut changed = false;
            let current = match t.coord_mode {
                CoordMode::Passthrough => "Passthrough",
                CoordMode::Permute(_) => "Permute",
                CoordMode::Radial { .. } => "Radial",
            };
            ui.label("coords");
            egui::ComboBox::from_id_salt(salt("coord-mode"))
                .selected_text(current)
                .show_ui(ui, |ui| {
                    for (key, next) in &[
                        ("Passthrough", CoordMode::Passthrough),
                        ("Permute", CoordMode::Permute([Axis::U, Axis::V, Axis::W])),
                        ("Radial", CoordMode::Radial { dim: RadialDim::D2, into: Axis::U }),
                    ] {
                        if ui.selectable_label(current == *key, *key).clicked() && current != *key
                        {
                            t.coord_mode = *next;
                            changed = true;
                        }
                    }
                });
            changed
        }
        (LayerKind::Transform(t), ParamRow::TransformEdgeMode) => enum_combo(
            ui,
            salt("edge-mode"),
            "edges",
            &mut t.edge_mode,
            &[texture_graph_core::EdgeMode::Clamp, texture_graph_core::EdgeMode::Extend],
            |e| match e {
                texture_graph_core::EdgeMode::Clamp => "clamp",
                texture_graph_core::EdgeMode::Extend => "extend",
            },
        ),
        (LayerKind::Transform(t), ParamRow::TransformPermute) => {
            let CoordMode::Permute(axes) = &mut t.coord_mode else { return false };
            let mut changed = false;
            for (i, axis) in axes.iter_mut().enumerate() {
                egui::ComboBox::from_id_salt(("canvas", id.0, "permute", i))
                    .selected_text(axis_label(*axis))
                    .show_ui(ui, |ui| {
                        for a in [Axis::U, Axis::V, Axis::W] {
                            if ui.selectable_label(*axis == a, axis_label(a)).clicked()
                                && *axis != a
                            {
                                *axis = a;
                                changed = true;
                            }
                        }
                    });
            }
            changed
        }
        (LayerKind::Transform(t), ParamRow::TransformRadialDim) => {
            let CoordMode::Radial { dim, .. } = &mut t.coord_mode else { return false };
            enum_combo(ui, salt("radial-dim"), "radial", dim, &[RadialDim::D2, RadialDim::D3], |d| {
                match d {
                    RadialDim::D2 => "2D",
                    RadialDim::D3 => "3D",
                }
            })
        }
        (LayerKind::Transform(t), ParamRow::TransformRadialInto) => {
            let CoordMode::Radial { into, .. } = &mut t.coord_mode else { return false };
            enum_combo(ui, salt("radial-into"), "into", into, &[Axis::U, Axis::V, Axis::W], axis_label)
        }
        (LayerKind::Mix(m), ParamRow::MixMode) => enum_combo(
            ui,
            salt("mode"),
            "mode",
            &mut m.mode,
            &[BlendMode::Add, BlendMode::Subtract, BlendMode::Multiply, BlendMode::Blend],
            |mode| match mode {
                BlendMode::Add => "Add",
                BlendMode::Subtract => "Subtract",
                BlendMode::Multiply => "Multiply",
                BlendMode::Blend => "Blend",
            },
        ),
        (LayerKind::Mix(m), ParamRow::MixSpace) => blend_space_combo(ui, salt("space"), &mut m.space),
        (LayerKind::MinMax(mm), ParamRow::MinMaxMode) => enum_combo(
            ui,
            salt("mode"),
            "mode",
            &mut mm.mode,
            &[MinMaxMode::Min, MinMaxMode::Max],
            |m| match m {
                MinMaxMode::Min => "Min",
                MinMaxMode::Max => "Max",
            },
        ),
        (LayerKind::MinMax(mm), ParamRow::MinMaxCriterion) => enum_combo(
            ui,
            salt("criterion"),
            "by",
            &mut mm.criterion,
            &[
                Criterion::Red,
                Criterion::Green,
                Criterion::Blue,
                Criterion::Saturation,
                Criterion::Value,
                Criterion::Luma,
                Criterion::Alpha,
                Criterion::Chroma,
            ],
            |c| match c {
                Criterion::Red => "Red",
                Criterion::Green => "Green",
                Criterion::Blue => "Blue",
                Criterion::Saturation => "Saturation",
                Criterion::Value => "Value",
                Criterion::Luma => "Luma",
                Criterion::Alpha => "Alpha",
                Criterion::Chroma => "Chroma",
            },
        ),
        (LayerKind::HeightToNormal(h), ParamRow::H2nStrength) => {
            ui.label("strength");
            ui.add(egui::Slider::new(&mut h.strength, 0.0..=8.0)).changed()
        }
        _ => false,
    }
}

fn blend_space_combo(
    ui: &mut egui::Ui,
    salt: impl std::hash::Hash + std::fmt::Debug,
    space: &mut BlendSpace,
) -> bool {
    enum_combo(
        ui,
        salt,
        "space",
        space,
        &[BlendSpace::Oklch, BlendSpace::LinearSrgb, BlendSpace::Hsv],
        |s| match s {
            BlendSpace::Oklch => "Oklch",
            BlendSpace::LinearSrgb => "LinearSrgb",
            BlendSpace::Hsv => "Hsv",
        },
    )
}

/// Kernel picker. Moving off the value kernel clears the period rather
/// than leaving one that `Graph::set_kind` would reject — the edit the user
/// made is the kernel, and an error toast about a field they cannot see on
/// simplex would be no help.
fn noise_kernel_combo(
    ui: &mut egui::Ui,
    salt: (&'static str, u64, &'static str),
    n: &mut texture_graph_core::Noise,
) -> bool {
    let changed = enum_combo(
        ui,
        salt,
        "kernel",
        &mut n.kernel,
        node_labels::KERNELS,
        node_labels::kernel,
    );
    if changed && n.kernel == NoiseKernel::Simplex {
        n.period = [0; 3];
    }
    changed
}

/// Per-axis lattice period, `0` meaning "does not repeat on this axis".
/// Hovering says what the current pair actually repeats at, because
/// `period / frequency` is the number that matters and neither field is it.
fn noise_period_row(ui: &mut egui::Ui, n: &mut texture_graph_core::Noise) -> bool {
    let mut changed = false;
    ui.label("period")
        .on_hover_text(node_labels::period_hint(n.frequency, n.period));
    for p in n.period.iter_mut() {
        changed |= ui.add(egui::DragValue::new(p).speed(0.25)).changed();
    }
    changed
}

fn vec3_row(ui: &mut egui::Ui, label: &str, v: &mut [f32; 3], speed: f32) -> bool {
    let mut changed = false;
    ui.label(label);
    for x in v.iter_mut() {
        changed |= ui.add(egui::DragValue::new(x).speed(speed as f64)).changed();
    }
    changed
}

fn axis_label(a: Axis) -> &'static str {
    match a {
        Axis::U => "U",
        Axis::V => "V",
        Axis::W => "W",
    }
}

#[allow(clippy::too_many_arguments)]
fn output_rows(
    ui: &mut egui::Ui,
    painter: &egui::Painter,
    graph: &Graph,
    state: &mut UiState,
    layout: &NodeLayout,
    canvas_rect: egui::Rect,
    style: &Arc<egui::Style>,
) {
    let z = state.canvas_zoom;
    if z < ZOOM_WIDGETS_MIN {
        if z >= ZOOM_LABELS_MIN {
            for (row, rect) in layout.rows.iter().zip(&layout.row_rects) {
                if let Row::Socket(key) = row {
                    painter_row_label(painter, *rect, z, socket_label(*key));
                }
            }
        }
        return;
    }

    let mut out = graph.output.clone();
    let mut changed = false;
    for (i, (row, rect)) in layout.rows.iter().zip(&layout.row_rects).enumerate() {
        let Row::Socket(key) = *row else { continue };
        let mut r = row_ui(ui, *rect, z, canvas_rect, style, (layout.node, i));
        r.label(socket_label(key));
        match key {
            InputKey::OutRoughness => {
                if let ScalarInput::Const(v) = &mut out.roughness {
                    changed |= r.add(egui::Slider::new(v, 0.0..=1.0)).changed();
                }
            }
            InputKey::OutMetallic => {
                if let ScalarInput::Const(v) = &mut out.metallic {
                    changed |= r.add(egui::Slider::new(v, 0.0..=1.0)).changed();
                }
            }
            InputKey::OutNormal => {
                if out.normal.is_none() {
                    r.weak("flat");
                }
            }
            _ => {}
        }
    }
    if changed {
        state.push(EditCmd::SetOutput(out));
    }
}

// ---- Sockets ------------------------------------------------------------

/// Interact with and paint every socket on this node. Registered after the
/// body and widgets so sockets win the pointer.
fn sockets(
    ui: &mut egui::Ui,
    painter: &egui::Painter,
    graph: &Graph,
    state: &mut UiState,
    layout: &NodeLayout,
) -> bool {
    let z = state.canvas_zoom;
    let hit_r = socket_hit_radius(z);
    let ptr = ui.ctx().input(|i| i.pointer.hover_pos());
    let mut hit = false;

    for sock in &layout.inputs {
        let resp = ui.interact(
            egui::Rect::from_center_size(sock.center, egui::Vec2::splat(2.0 * hit_r)),
            egui::Id::new(("socket", layout.node, sock.key)),
            egui::Sense::click_and_drag(),
        );
        if resp.drag_started() {
            // A connected input hands its wire over, far end still anchored.
            // An unconnected one starts a wire going the other way.
            state.wire_drag = Some(match sock.value.connected_to() {
                Some(src) => WireDrag::FromOutput {
                    src,
                    detached_from: Some((layout.node, sock.key)),
                },
                None => WireDrag::FromInput { node: layout.node, key: sock.key },
            });
            hit = true;
        }
        if resp.dragged() || resp.clicked() {
            hit = true;
        }

        // Raw distance: `Response::hovered` is unreliable while another
        // widget owns the drag.
        let hovered = ptr.is_some_and(|p| sock.center.distance(p) <= hit_r);
        let (radius, stroke_color) = socket_style(graph, state, layout.node, hovered, z);
        painter.circle(
            sock.center,
            radius,
            egui::Color32::WHITE,
            egui::Stroke::new((1.2 * z).max(1.0), stroke_color),
        );
    }

    if let (Some(center), NodeRef::Layer(id)) = (layout.output_socket, layout.node) {
        let resp = ui.interact(
            egui::Rect::from_center_size(center, egui::Vec2::splat(2.0 * hit_r)),
            egui::Id::new(("socket-out", id.0)),
            egui::Sense::click_and_drag(),
        );
        if resp.drag_started() {
            state.wire_drag = Some(WireDrag::FromOutput { src: id, detached_from: None });
            hit = true;
        }
        if resp.dragged() || resp.clicked() {
            hit = true;
        }
        let hovered = ptr.is_some_and(|p| center.distance(p) <= hit_r);
        let (radius, stroke_color) = output_socket_style(graph, state, id, hovered, z);
        painter.circle(
            center,
            radius,
            egui::Color32::WHITE,
            egui::Stroke::new((1.2 * z).max(1.0), stroke_color),
        );
    }
    hit
}

/// How much a hovered socket, or one a live wire could land on, grows by.
const SOCKET_HOVER: f32 = 1.35;
/// Smallest a socket ever draws, so it survives zooming out. Larger for one
/// a wire could land on, which has an answer to give.
const SOCKET_DRAW_MIN: f32 = 2.0;
const SOCKET_CANDIDATE_MIN: f32 = 3.0;

fn plain(hovered: bool, z: f32) -> (f32, egui::Color32) {
    let radius = if hovered { SOCKET_R * z * SOCKET_HOVER } else { SOCKET_R * z };
    (radius.max(SOCKET_DRAW_MIN), egui::Color32::BLACK)
}

fn eligibility(eligible: bool, z: f32) -> (f32, egui::Color32) {
    let big = (SOCKET_R * z * SOCKET_HOVER).max(SOCKET_CANDIDATE_MIN);
    (big, if eligible { egui::Color32::BLACK } else { egui::Color32::RED })
}

/// Radius and outline for an input socket. One that would refuse the wire in
/// the air rings red, while the drag can still be cancelled.
fn socket_style(
    graph: &Graph,
    state: &UiState,
    node: NodeRef,
    hovered: bool,
    z: f32,
) -> (f32, egui::Color32) {
    match state.wire_drag {
        // That wire hunts for an output, so no input is a candidate.
        Some(WireDrag::FromOutput { src, .. }) if hovered => {
            eligibility(super::wires::refusal(graph, node, src).is_none(), z)
        }
        _ => plain(hovered, z),
    }
}

/// The same for an output, while a wire is pulled backwards out of an input.
fn output_socket_style(
    graph: &Graph,
    state: &UiState,
    src: LayerId,
    hovered: bool,
    z: f32,
) -> (f32, egui::Color32) {
    match state.wire_drag {
        Some(WireDrag::FromInput { node, .. }) if hovered => {
            eligibility(super::wires::refusal(graph, node, src).is_none(), z)
        }
        _ => plain(hovered, z),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use texture_graph_core::color::oklcha;

    /// Reading a name and clicking away must not mark the graph unsaved.
    #[test]
    fn committing_an_unchanged_title_is_not_an_edit() {
        let mut graph = Graph::new();
        let id = graph
            .add_layer("continents", LayerKind::Color(oklcha(0.5, 0.0, 0.0, 1.0)))
            .unwrap();

        assert!(rename_command(&graph, id, "continents").is_none());
        assert!(rename_command(&graph, id, "  continents  ").is_none(), "trimmed first");
        assert!(rename_command(&graph, id, "").is_none());
        assert!(rename_command(&graph, id, "   ").is_none());
        assert!(matches!(
            rename_command(&graph, id, " oceans "),
            Some(EditCmd::Rename(_, name)) if name == "oceans"
        ));
    }
}
