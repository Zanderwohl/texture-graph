//! Connection curves: the edge pass, the live wire that follows the mouse,
//! and drop resolution into socket edits.

use std::collections::HashMap;

use texture_graph_core::{ConstValue, Graph, InputKey, LayerId, SocketValue};

use crate::state::{EditCmd, NodeRef, UiState, WireDrag};

use super::layout::{NodeLayout, SocketLayout};
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
    let detached = state.wire_drag.and_then(WireDrag::detached_from);
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

/// The nearest of `sockets` to the pointer, within the grab radius. Both
/// socket sides hunt the same way; only the set differs.
fn nearest<T>(
    sockets: impl Iterator<Item = (egui::Pos2, T)>,
    ptr: egui::Pos2,
    zoom: f32,
) -> Option<T> {
    let r = socket_hit_radius(zoom);
    let mut best: Option<(f32, T)> = None;
    for (center, item) in sockets {
        let d = center.distance(ptr);
        if d <= r && best.as_ref().is_none_or(|(bd, _)| d < *bd) {
            best = Some((d, item));
        }
    }
    best.map(|(_, item)| item)
}

/// The input socket currently under the pointer, if any.
fn hovered_input(
    layouts: &[NodeLayout],
    ptr: egui::Pos2,
    zoom: f32,
) -> Option<(NodeRef, &SocketLayout)> {
    let sockets = layouts
        .iter()
        .flat_map(|l| l.inputs.iter().map(move |s| (s.center, (l.node, s))));
    nearest(sockets, ptr, zoom)
}

/// The output socket under the pointer, if any — what a wire pulled
/// backwards out of an input is hunting for. The Output pseudo-node has
/// none, which `NodeLayout::output_socket` being `None` there already says.
pub fn hovered_output(
    layouts: &[NodeLayout],
    ptr: egui::Pos2,
    zoom: f32,
) -> Option<(LayerId, egui::Pos2)> {
    let sockets = layouts.iter().filter_map(|l| match (l.node, l.output_socket) {
        (NodeRef::Layer(id), Some(center)) => Some((center, (id, center))),
        _ => None,
    });
    nearest(sockets, ptr, zoom)
}

/// The connection a drop would make, whichever end the drag started from.
struct Candidate {
    node: NodeRef,
    key: InputKey,
    src: LayerId,
    /// Where the live wire should end while this is the target.
    at: egui::Pos2,
}

/// Where the wire is anchored: the socket the drag started from, which
/// stays put for the whole drag.
fn anchor(
    wire: WireDrag,
    layouts: &[NodeLayout],
    out_sockets: &HashMap<LayerId, egui::Pos2>,
) -> Option<egui::Pos2> {
    match wire {
        WireDrag::FromOutput { src, .. } => out_sockets.get(&src).copied(),
        WireDrag::FromInput { node, key } => layouts
            .iter()
            .find(|l| l.node == node)?
            .inputs
            .iter()
            .find(|s| s.key == key)
            .map(|s| s.center),
    }
}

/// Whether a connection would be admitted, decided while the wire is still
/// in the air so the socket can turn red before the user commits.
///
/// Only cycles for now — every layer produces a `Color`, so there is no
/// type mismatch to catch. The graph stays the authority: a drop always
/// goes through `set_input`, and anything this misses is refused there.
pub fn eligible(graph: &Graph, node: NodeRef, src: LayerId) -> bool {
    match node {
        // Nothing depends on the Output node, so it can never cycle.
        NodeRef::Output => true,
        NodeRef::Layer(dst) => !graph.would_cycle(dst, src),
    }
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
    // The node the drag started from was deleted mid-drag.
    let Some(anchor_pos) = anchor(wire, layouts, out_sockets) else {
        state.wire_drag = None;
        return false;
    };

    // The two directions differ only in what they are hunting for; from
    // here down a candidate connection is a candidate connection.
    let candidate = match wire {
        WireDrag::FromOutput { src, .. } => hovered_input(layouts, ptr, state.canvas_zoom)
            .map(|(node, sock)| Candidate { node, key: sock.key, src, at: sock.center }),
        WireDrag::FromInput { node, key } => hovered_output(layouts, ptr, state.canvas_zoom)
            .map(|(src, at)| Candidate { node, key, src, at }),
    };
    let admitted = candidate
        .as_ref()
        .map(|c| eligible(graph, c.node, c.src));

    if primary_down {
        let (end, color) = match (&candidate, admitted) {
            (Some(c), Some(true)) => (c.at, egui::Color32::from_gray(230)),
            (Some(c), _) => (c.at, egui::Color32::LIGHT_RED),
            (None, _) => (ptr, egui::Color32::from_gray(230)),
        };
        let stroke = egui::Stroke::new((2.0 * state.canvas_zoom).max(1.2), color);
        painter.add(wire_shape(anchor_pos, end, state.canvas_zoom, stroke));
        return true;
    }

    // Released — resolve the drop.
    let detached = wire.detached_from();
    state.wire_drag = None;
    match candidate {
        Some(c) => {
            if detached == Some((c.node, c.key)) {
                // Dropped back where it came from: nothing happened.
            } else if admitted != Some(true) {
                // Refused before it can disturb anything — in particular
                // the wire it was detached from stays where it is.
                state.last_error = Some("connection would create a cycle".to_string());
            } else {
                let mut edits: Vec<(NodeRef, InputKey, Option<LayerId>)> = Vec::new();
                if let Some((n0, k0)) = detached {
                    edits.push((n0, k0, None));
                }
                edits.push((c.node, c.key, Some(c.src)));
                apply_socket_edits(graph, state, &edits);
            }
        }
        // Dropped on empty space: a detached wire is gone, a fresh one
        // never existed.
        None => {
            if let Some((n0, k0)) = detached {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn layout_at(node: NodeRef, min: egui::Pos2, output: Option<egui::Pos2>) -> NodeLayout {
        NodeLayout {
            node,
            rect: egui::Rect::from_min_size(min, egui::vec2(100.0, 40.0)),
            thumb_rect: None,
            rows: Vec::new(),
            row_rects: Vec::new(),
            inputs: Vec::new(),
            output_socket: output,
        }
    }

    /// A wire dragged backwards out of an input is hunting for an output,
    /// and the Output pseudo-node does not have one — dropping on it would
    /// be asking the material's result to feed a node.
    #[test]
    fn a_backwards_wire_finds_outputs_and_never_the_material_output() {
        let layouts = vec![
            layout_at(
                NodeRef::Layer(LayerId(1)),
                egui::pos2(0.0, 0.0),
                Some(egui::pos2(100.0, 10.0)),
            ),
            layout_at(
                NodeRef::Layer(LayerId(2)),
                egui::pos2(0.0, 80.0),
                Some(egui::pos2(100.0, 90.0)),
            ),
            layout_at(NodeRef::Output, egui::pos2(200.0, 0.0), None),
        ];
        assert_eq!(
            hovered_output(&layouts, egui::pos2(101.0, 11.0), 1.0),
            Some((LayerId(1), egui::pos2(100.0, 10.0)))
        );
        // Nearest wins when two are within reach.
        assert_eq!(
            hovered_output(&layouts, egui::pos2(100.0, 88.0), 1.0).map(|(id, _)| id),
            Some(LayerId(2))
        );
        // Nowhere near a socket, and on top of the Output node, are both
        // "no target".
        assert_eq!(hovered_output(&layouts, egui::pos2(50.0, 40.0), 1.0), None);
        assert_eq!(hovered_output(&layouts, egui::pos2(250.0, 20.0), 1.0), None);
    }
}
