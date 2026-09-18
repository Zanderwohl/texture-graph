//! Connection curves: the edge pass, the live wire that follows the mouse,
//! and drop resolution into socket edits.

use std::collections::HashMap;

use texture_graph_core::{ConstValue, Graph, InputKey, LayerId, SocketValue};

use crate::state::{EditCmd, NodeRef, UiState};

use super::layout::NodeLayout;
use super::socket_hit_radius;

/// Horizontal-handle cubic bezier between an output and an input socket.
pub fn wire_shape(a: egui::Pos2, b: egui::Pos2, zoom: f32, stroke: egui::Stroke) -> egui::Shape {
    let mid = ((b.x - a.x).abs() * 0.5).max(20.0 * zoom);
    let c1 = egui::pos2(a.x + mid, a.y);
    let c2 = egui::pos2(b.x - mid, b.y);
    egui::Shape::CubicBezier(egui::epaint::CubicBezierShape {
        points: [a, c1, c2, b],
        closed: false,
        fill: egui::Color32::TRANSPARENT,
        stroke: stroke.into(),
    })
}

/// Draw every stored connection at exact socket geometry. The edge being
/// detach-dragged is hidden — no `EditCmd` has fired for it yet, so a
/// cancelled drag makes it simply reappear.
pub fn draw_edges(
    painter: &egui::Painter,
    layouts: &[NodeLayout],
    out_sockets: &HashMap<LayerId, egui::Pos2>,
    state: &UiState,
) {
    let stroke = egui::Stroke::new(
        (1.5 * state.canvas_zoom).max(1.0),
        egui::Color32::from_gray(160),
    );
    let detached = state.wire_drag.and_then(|w| w.detached_from);
    for layout in layouts {
        for sock in &layout.inputs {
            if detached == Some((layout.node, sock.key)) {
                continue;
            }
            let Some(src) = sock.value.connected_to() else { continue };
            let Some(a) = out_sockets.get(&src).copied() else { continue };
            painter.add(wire_shape(a, sock.center, state.canvas_zoom, stroke));
        }
    }
}

/// The input socket currently under the pointer, if any.
fn hovered_input<'a>(
    layouts: &'a [NodeLayout],
    ptr: egui::Pos2,
    zoom: f32,
) -> Option<(NodeRef, &'a super::layout::SocketLayout)> {
    let r = socket_hit_radius(zoom);
    let mut best: Option<(f32, NodeRef, &super::layout::SocketLayout)> = None;
    for layout in layouts {
        for sock in &layout.inputs {
            let d = sock.center.distance(ptr);
            if d <= r && best.map_or(true, |(bd, _, _)| d < bd) {
                best = Some((d, layout.node, sock));
            }
        }
    }
    best.map(|(_, node, sock)| (node, sock))
}

/// Advance the wire-drag state machine: draw the live wire while the
/// button is held, resolve the drop on release. Returns whether a wire
/// drag is active this frame (suppresses pan).
pub fn advance_wire_drag(
    ui: &egui::Ui,
    graph: &Graph,
    state: &mut UiState,
    layouts: &[NodeLayout],
    out_sockets: &HashMap<LayerId, egui::Pos2>,
    painter: &egui::Painter,
) -> bool {
    let Some(wire) = state.wire_drag else { return false };

    if ui.ctx().input(|i| i.key_pressed(egui::Key::Escape)) {
        state.wire_drag = None;
        return false;
    }
    let (primary_down, ptr) = ui
        .ctx()
        .input(|i| (i.pointer.primary_down(), i.pointer.hover_pos()));
    let Some(ptr) = ptr else {
        // Pointer left the window entirely — treat as a cancel.
        state.wire_drag = None;
        return false;
    };
    let Some(src_pos) = out_sockets.get(&wire.src).copied() else {
        state.wire_drag = None;
        return false;
    };

    let target = hovered_input(layouts, ptr, state.canvas_zoom);
    let eligibility = target.map(|(node, sock)| {
        let ok = match node {
            // Nothing depends on the Output node, so it can never cycle.
            NodeRef::Output => true,
            NodeRef::Layer(dst) => !graph.would_cycle(dst, wire.src),
        };
        (node, sock.key, sock.center, ok)
    });

    if primary_down {
        let (end, color) = match eligibility {
            Some((_, _, center, true)) => (center, egui::Color32::from_gray(230)),
            Some((_, _, center, false)) => (center, egui::Color32::LIGHT_RED),
            None => (ptr, egui::Color32::from_gray(230)),
        };
        let stroke = egui::Stroke::new((2.0 * state.canvas_zoom).max(1.2), color);
        painter.add(wire_shape(src_pos, end, state.canvas_zoom, stroke));
        return true;
    }

    // Released — resolve the drop.
    state.wire_drag = None;
    match eligibility {
        Some((node, key, _, eligible)) => {
            if wire.detached_from == Some((node, key)) {
                // Dropped back where it came from.
            } else if !eligible {
                state.last_error =
                    Some("connection would create a cycle".to_string());
            } else {
                let mut edits: Vec<(NodeRef, InputKey, Option<LayerId>)> = Vec::new();
                if let Some((n0, k0)) = wire.detached_from {
                    edits.push((n0, k0, None));
                }
                edits.push((node, key, Some(wire.src)));
                apply_socket_edits(graph, state, &edits);
            }
        }
        None => {
            if let Some((n0, k0)) = wire.detached_from {
                apply_socket_edits(graph, state, &[(n0, k0, None)]);
            }
        }
    }
    false
}

/// Apply a batch of socket edits, one `EditCmd` per touched node, stashing
/// and restoring displaced `Const` values along the way. Grouping matters:
/// a detach-and-reconnect on the same node must land as ONE `SetKind` so
/// the intermediate state never exists.
pub fn apply_socket_edits(
    graph: &Graph,
    state: &mut UiState,
    edits: &[(NodeRef, InputKey, Option<LayerId>)],
) {
    let mut groups: Vec<(NodeRef, Vec<(InputKey, Option<LayerId>)>)> = Vec::new();
    for &(node, key, target) in edits {
        match groups.iter_mut().find(|(n, _)| *n == node) {
            Some((_, list)) => list.push((key, target)),
            None => groups.push((node, vec![(key, target)])),
        }
    }

    for (node, list) in groups {
        match node {
            NodeRef::Layer(id) => {
                let Some(layer) = graph.get(id) else { continue };
                let mut kind = layer.kind.clone();
                let mut ok = true;
                for (key, target) in list {
                    match kind.set_input(key, target) {
                        Ok(replaced) => {
                            remember_consts(state, node, key, target, replaced, |cv| {
                                let _ = kind.set_const(key, cv);
                            });
                        }
                        Err(e) => {
                            state.last_error = Some(e.to_string());
                            ok = false;
                        }
                    }
                }
                if ok {
                    state.push(EditCmd::SetKind(id, kind));
                }
            }
            NodeRef::Output => {
                let mut out = graph.output.clone();
                let mut ok = true;
                for (key, target) in list {
                    match out.set_input(key, target) {
                        Ok(replaced) => {
                            remember_consts(state, node, key, target, replaced, |cv| {
                                let _ = out.set_const(key, cv);
                            });
                        }
                        Err(e) => {
                            state.last_error = Some(e.to_string());
                            ok = false;
                        }
                    }
                }
                if ok {
                    state.push(EditCmd::SetOutput(out));
                }
            }
        }
    }
}

/// On connect: stash the `Const` the wire displaced. On disconnect: hand a
/// previously stashed `Const` to `restore` so the socket comes back with
/// its pre-connection value instead of the default.
fn remember_consts(
    state: &mut UiState,
    node: NodeRef,
    key: InputKey,
    target: Option<LayerId>,
    replaced: SocketValue,
    restore: impl FnOnce(ConstValue),
) {
    use texture_graph_core::{ColorInput, ScalarInput};
    if target.is_some() {
        match replaced {
            SocketValue::Color(ColorInput::Const(c)) => {
                state.saved_consts.insert((node, key), ConstValue::Color(c));
            }
            SocketValue::Scalar(ScalarInput::Const(v)) => {
                state.saved_consts.insert((node, key), ConstValue::Scalar(v));
            }
            _ => {}
        }
    } else if let Some(cv) = state.saved_consts.remove(&(node, key)) {
        restore(cv);
    }
}
