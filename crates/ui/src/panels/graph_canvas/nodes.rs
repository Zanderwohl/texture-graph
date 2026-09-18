//! Node pass: body interaction, chrome, thumbnails, inline widget rows,
//! and socket interaction/painting.
//!
//! Registration order inside a node gives egui's top-most-wins hit testing
//! the right precedence automatically: body interact first, then inline
//! widgets, then sockets — so sockets beat widgets beat node-drag.

use std::sync::Arc;

use texture_graph_core::{
    Axis, BlendMode, BlendSpace, Color, ColorInput, ColorRamp, ColorStop, CoordMode, Criterion,
    EvalCtx, Graph, InputKey, LayerId, LayerKind, MinMaxMode, NoiseDims, NoiseOutput,
    NoiseRange, RadialDim, ScalarInput,
};

use crate::app::GpuBits;
use crate::color_convert::{oklcha_to_srgba, srgba_to_oklcha};
use crate::previews::PreviewCache;
use crate::state::{EditCmd, NodeDrag, NodeRef, RampDrag, Renaming, UiState, WireDrag};
use crate::widgets::enum_combo::enum_combo;

use super::layout::{socket_label, NodeLayout, ParamRow, Row};
use super::ramp;
use super::{socket_hit_radius, HEADER_H, SOCKET_R, ZOOM_LABELS_MIN, ZOOM_WIDGETS_MIN};

/// Returns `true` if any node body, widget, or socket claimed the pointer
/// this frame — the caller suppresses background pan.
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
        // Off-screen nodes are skipped whole: no interaction, no painting,
        // and no `get_or_build`, so the frame's bake budget goes to
        // thumbnails somebody can see. The margin covers the sockets, which
        // sit on the node's edge and so reach slightly past its rect.
        let margin = socket_hit_radius(state.canvas_zoom);
        if !layout.rect.expand(margin).intersects(canvas_rect) && !held(state, layout.node) {
            continue;
        }
        hit |= body_interact(ui, graph, state, layout, canvas_rect);
        let renaming = matches!(layout.node, NodeRef::Layer(id)
            if state.renaming.as_ref().is_some_and(|r| r.node == id));
        draw_chrome(painter, graph, state, layout, renaming);
        // After the body, so the field wins the pointer over the node drag.
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

/// Whether this node has to be drawn and interacted with wherever it is,
/// off screen or not.
///
/// Two do: the one the pointer is holding, because the drag is resolved
/// against the pointer and a node that stopped interacting on leaving the
/// view would be dropped there; and the one being renamed, because a text
/// field that stops existing takes the keyboard focus the rename is
/// committed by with it.
fn held(state: &UiState, node: NodeRef) -> bool {
    state.drag.is_some_and(|d| d.node == node)
        || matches!(node, NodeRef::Layer(id)
            if state.renaming.as_ref().is_some_and(|r| r.node == id))
}

// ---- Body ---------------------------------------------------------------

/// World-space offset of a duplicated node from its original — down and to
/// the right so the copy is visibly a separate, nearby node.
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
            // World position for a duplicate — offset from this node.
            let world = super::screen_to_world(layout.rect.min, canvas_rect.min, state);
            resp.context_menu(|ui| {
                // The menu entry only arms the header field; the typing
                // happens up there, where the name is. Kept alongside the
                // click-the-title path because a context menu is where
                // people look for "rename".
                if ui.button("Rename").clicked() {
                    state.renaming = Some(Renaming {
                        node: id,
                        text: graph.get(id).map(|l| l.name.clone()).unwrap_or_default(),
                        focused: false,
                    });
                    ui.close();
                }
                // Preview this node's color alone (default roughness/
                // metallic/normal) in the preview panel.
                if ui.button("Preview").clicked() {
                    state.preview_target = Some(id);
                    ui.close();
                }
                // Duplicate: copy every param (and input wiring) into a new
                // node placed down-and-right of this one.
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
                // Back to the full PBR material.
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
    // While the title is being typed over, the text field *is* the title.
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

/// The name's line in the header — what you click to rename, and where the
/// field appears when you do.
fn title_rect(node: egui::Rect, z: f32) -> egui::Rect {
    egui::Rect::from_min_size(
        node.min + egui::vec2(8.0, 2.0) * z,
        egui::vec2(node.width() - 16.0 * z, 18.0 * z),
    )
}

/// Click the title to edit it; **Enter** or clicking away commits,
/// **Escape** discards. Both paths put the painted label back.
///
/// The text lives in `UiState` rather than in the graph because names must
/// be unique: a rename is refused like any other edit, and half-typed text
/// needs somewhere to sit while it is briefly a duplicate of something.
/// Committing queues a `Rename` command like everything else, so a refusal
/// lands in the status row and the title simply stays what it was.
///
/// Returns whether the field claimed the pointer, which suppresses the node
/// drag underneath it.
fn title(ui: &mut egui::Ui, graph: &Graph, state: &mut UiState, layout: &NodeLayout) -> bool {
    let NodeRef::Layer(id) = layout.node else { return false };
    let z = state.canvas_zoom;
    // Zoomed out far enough that the header is a smudge, there is nothing
    // worth typing into; the label is painted and that is all.
    if z < ZOOM_WIDGETS_MIN {
        return false;
    }
    let rect = title_rect(layout.rect, z);
    let editing = state.renaming.as_ref().is_some_and(|r| r.node == id);

    // Both branches go through the same child Ui, at the same rect, with the
    // same explicit id — so each registers exactly one widget, at one rect,
    // under one auto-id, whether the title is a label you can click or a
    // field you can type in. Swapping the *shape* of what is registered
    // here is what egui's `warn_if_rect_changes_id` is watching for: the
    // widget at this rect would have one id on the frame before a rename
    // and another on the frame after, which paints a red outline and, worse,
    // hands any in-flight interaction to the wrong widget.
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
        // Only on the first frame. Asking every frame would make the field
        // impossible to blur, and blurring is one of the two ways to commit.
        resp.request_focus();
    }

    // Escape is checked before the field's own handling matters: it also
    // surrenders focus, so checking the discard case first is what keeps a
    // cancelled rename from being read as a blur and committed.
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

/// The edit a committed title amounts to, or `None` when it amounts to
/// nothing. Clicking a title, reading it and clicking away is not a change,
/// and queueing a no-op rename would mark the file unsaved for having been
/// read. An emptied field is the same case: a nameless layer isn't what the
/// user meant, and the model would refuse it anyway.
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

/// Child Ui spanning one row, with the app style scaled to the canvas zoom
/// so fonts, spacing, and interact sizes track the node visually.
///
/// The id is set with [`egui::UiBuilder::id`] — *explicit*, not a salt.
/// A salted child derives its `unique_id`, and with it the auto-ids of
/// every widget inside, from the parent's running `next_auto_id_salt`.
/// That makes each inline slider and drag-value depend on how many child
/// Uis happened to be created before it this frame — which changes when
/// the rename field appears or disappears, and when a node scrolls out of
/// view and stops registering. The widgets then swap ids under each other
/// for a frame: egui paints its `warn_if_rect_changes_id` outlines across
/// the whole canvas, and an in-progress drag lands on the wrong widget. An
/// explicit id is independent of the parent, so a row's widgets keep their
/// ids no matter what else is on screen.
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

/// The row style, which depends on the canvas zoom and on nothing else.
/// Built once per frame and shared by every row of every node: it was a
/// full `Style` clone per row per node per frame, which on a twenty-node
/// graph is hundreds of them.
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

/// Compact color swatch (no LCh popover — nodes are tight; the inspector
/// still has the full editor).
fn color_swatch(ui: &mut egui::Ui, color: &mut texture_graph_core::Color) -> bool {
    let mut rgba = oklcha_to_srgba(*color);
    if ui.color_edit_button_rgba_unmultiplied(&mut rgba).changed() {
        *color = srgba_to_oklcha(rgba);
        return true;
    }
    false
}

/// Painter-only fallback label for a socket row when zoom is too low for
/// real widgets.
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
                        // Keep the gradient recognizable when zoomed out —
                        // paint only, no interaction.
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
                changed |= socket_row(&mut r, &mut kind, key, state, layout.node);
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

// ---- Ramp bar -----------------------------------------------------------

/// Screen-space grab radius around a stop indicator, clamped so indicators
/// stay grabbable when zoomed out.
fn ramp_grab_radius(z: f32) -> f32 {
    (6.0 * z).max(6.0)
}

/// Split a RampBar row rect into the gradient strip and the arrow strip
/// below it. Vertical budget (world units × z): 3 pad + 22 gradient +
/// 12 arrows + 3 pad = RAMP_BAR_H.
fn ramp_bar_rects(rect: egui::Rect, z: f32) -> (egui::Rect, egui::Rect) {
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

/// Gradient mesh + stop indicators. `highlight` outlines that stop white
/// (hovered or dragged). Paint-only — shared by the interactive widget and
/// the low-zoom path.
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

    // Sample positions: uniform coverage plus every interior stop t, so
    // hard cusps land exactly on a vertex instead of straddling a segment.
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

    // Indicators: house-shaped pentagons pointing up at the stop's t, in
    // stop order so a later stop paints on top at coincident t.
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

/// Interactive gradient bar: drag indicators (crossing reorders), double-
/// click empty bar to add a stop. Returns whether `ramp` changed.
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
    let pointer_t = |x: f32| ((x - grad.left()) / grad.width()).clamp(0.0, 1.0);
    // Nearest indicator within the grab radius; `<=` so the later (topmost-
    // painted) stop wins a tie at coincident t.
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

    // One interact for the whole bar: a single stable id keeps egui drag
    // ownership through crossing reorders (per-indicator ids would change
    // index mid-drag).
    let resp = ui.interact(
        rect,
        egui::Id::new(("ramp-bar", id.0)),
        egui::Sense::click_and_drag(),
    );
    let mut changed = false;

    if resp.drag_started() {
        if let Some(p) = resp.interact_pointer_pos() {
            if let Some(i) = indicator_at(&ramp.stops, p.x) {
                state.ramp_drag = Some(RampDrag { node: id, stop: i });
            }
        }
    }
    // Advance from global pointer state, not `resp.dragged()` — the drag
    // must stay stuck to the indicator even when the pointer overshoots the
    // bar rect (t just clamps). Release is button-up, handled up front by
    // `release_stale_ramp_drag`.
    if let Some(drag) = state.ramp_drag.filter(|d| d.node == id) {
        if let Some(p) = ui.ctx().input(|i| i.pointer.hover_pos()) {
            if drag.stop < ramp.stops.len() {
                let (new_i, swaps) =
                    ramp::drag_stop_to(&mut ramp.stops, drag.stop, pointer_t(p.x));
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
                state.ramp_drag = Some(RampDrag { node: id, stop: new_i });
                changed = true;
            }
        }
    }
    if resp.double_clicked() {
        if let Some(p) = resp.interact_pointer_pos() {
            // Only on empty bar — near an indicator it's a mis-click.
            if indicator_at(&ramp.stops, p.x).is_none() {
                let t = pointer_t(p.x);
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

    // Paint after interaction so indicators track this frame's drag.
    let highlight = state
        .ramp_drag
        .filter(|d| d.node == id)
        .map(|d| d.stop)
        .or_else(|| {
            let p = ui.ctx().input(|i| i.pointer.hover_pos())?;
            rect.contains(p)
                .then(|| indicator_at(&ramp.stops, p.x))
                .flatten()
        });
    paint_ramp_bar(painter, graph, ramp, rect, z, eval_ctx, highlight);
    changed
}

/// Label plus the unconnected-const widget, if this socket type has one.
/// Connected sockets show the label only.
fn socket_row(
    ui: &mut egui::Ui,
    kind: &mut LayerKind,
    key: InputKey,
    state: &mut UiState,
    node: NodeRef,
) -> bool {
    let mut changed = false;
    match (&mut *kind, key) {
        (LayerKind::Mix(m), InputKey::MixFactor) => {
            ui.label(socket_label(key));
            if let ScalarInput::Const(v) = &mut m.factor {
                changed |= ui.add(egui::Slider::new(v, 0.0..=1.0)).changed();
            }
        }
        (LayerKind::ColorRamp(r), InputKey::RampStop(i)) => {
            let count = r.stops.len();
            if let Some(stop) = r.stops.get_mut(i) {
                changed |= ui
                    .add(egui::DragValue::new(&mut stop.t).speed(0.01).range(0.0..=1.0))
                    .changed();
                if let ColorInput::Const(c) = &mut stop.color {
                    changed |= color_swatch(ui, c);
                }
                // A ramp needs at least two stops; hide remove at the floor.
                if count > 2 && ui.small_button("x").clicked() {
                    r.stops.remove(i);
                    // RampStop keys are positional — shift saved consts down
                    // past the removed index, drop the removed stop's own.
                    state.remap_ramp_consts(node, move |j| match j.cmp(&i) {
                        std::cmp::Ordering::Less => Some(j),
                        std::cmp::Ordering::Equal => None,
                        std::cmp::Ordering::Greater => Some(j - 1),
                    });
                    changed = true;
                }
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
            color_swatch(ui, c)
        }
        (LayerKind::Noise(n), ParamRow::NoiseDims) => enum_combo(
            ui,
            salt("dims"),
            "dims",
            &mut n.dims,
            &[NoiseDims::D1, NoiseDims::D2, NoiseDims::D3],
            |d| match d {
                NoiseDims::D1 => "1D",
                NoiseDims::D2 => "2D",
                NoiseDims::D3 => "3D",
            },
        ),
        (LayerKind::Noise(n), ParamRow::NoiseOutput) => enum_combo(
            ui,
            salt("output"),
            "output",
            &mut n.output,
            &[NoiseOutput::Grayscale, NoiseOutput::Color],
            |o| match o {
                NoiseOutput::Grayscale => "grayscale",
                NoiseOutput::Color => "color (LCh)",
            },
        ),
        (LayerKind::Noise(n), ParamRow::NoiseRange) => enum_combo(
            ui,
            salt("range"),
            "range",
            &mut n.range,
            &[NoiseRange::Unsigned, NoiseRange::Signed],
            |r| match r {
                NoiseRange::Unsigned => "[0, 1]",
                NoiseRange::Signed => "[-1, 1]",
            },
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
            // A connected input hands its wire over — the far end stays put
            // and the loose end is looking for a new home. An unconnected
            // one starts a wire of its own, going the other way.
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

        // Hover from raw pointer distance — Response::hovered is unreliable
        // while another widget owns the drag.
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
/// Smallest a socket ever draws, so it stays visible when zoomed far out —
/// larger for one a wire could land on, which has an answer to give.
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

/// Radius and outline for an input socket. A socket that would refuse the
/// wire currently in the air rings red — the answer arrives while the wire
/// is still cancellable, rather than as a message after the drop.
fn socket_style(
    graph: &Graph,
    state: &UiState,
    node: NodeRef,
    hovered: bool,
    z: f32,
) -> (f32, egui::Color32) {
    match state.wire_drag {
        // A wire pulled out of an input is hunting for an *output*; no
        // input socket is a candidate for it, so none of them react.
        Some(WireDrag::FromOutput { src, .. }) if hovered => {
            eligibility(super::wires::refusal(graph, node, src).is_none(), z)
        }
        _ => plain(hovered, z),
    }
}

/// The same, for an output socket while a wire is being pulled backwards
/// out of an input.
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

    /// Committing a title that was not actually changed must not queue an
    /// edit. Otherwise clicking a name to read it and clicking away marks
    /// the graph unsaved, and the user is asked to save a file they did not
    /// touch.
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
