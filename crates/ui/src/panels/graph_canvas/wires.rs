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

/// Every stored connection at exact socket geometry. A detach-dragged edge
/// is hidden but unedited, so cancelling makes it reappear.
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

/// Nearest of `sockets` within the grab radius. Both sides hunt alike.
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

/// The output socket under the pointer: what a wire pulled backwards out of
/// an input hunts for. The Output pseudo-node has none.
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

/// The socket the drag started from, anchored for its duration.
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

/// Why a drop would be refused, decided while the wire is in the air so the
/// socket can ring red before the user commits, and reused as the message if
/// they commit anyway — one function, so colour and sentence agree.
///
/// Only cycles: every layer produces a `Color` and every socket takes one,
/// so there is no type mismatch to catch. The graph stays the authority; a
/// drop goes through `set_kind`, which refuses anything this misses.
pub fn refusal(graph: &Graph, node: NodeRef, src: LayerId) -> Option<String> {
    // Nothing reads the material output, so it can't be in a loop.
    let NodeRef::Layer(dst) = node else { return None };
    if !graph.would_cycle(dst, src) {
        return None;
    }
    let name = |id: LayerId| {
        graph
            .get(id)
            .map(|l| l.name.clone())
            .unwrap_or_else(|| format!("layer {}", id.0))
    };
    Some(if src == dst {
        format!("{} can't read itself", name(dst))
    } else {
        // Naming both ends is the point: on a dozen nodes, "that would make
        // a loop" leaves the user hunting for the other half.
        format!("{} already reads {} — that would make a loop", name(src), name(dst))
    })
}

/// Draw the live wire while the button is held, resolve the drop on release.
/// Returns whether a drag is active, which suppresses pan.
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
        // Pointer left the window: cancel.
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
    let refused = candidate.as_ref().and_then(|c| refusal(graph, c.node, c.src));

    if primary_down {
        let (end, color) = match (&candidate, &refused) {
            (Some(c), None) => (c.at, egui::Color32::from_gray(230)),
            (Some(c), Some(_)) => (c.at, egui::Color32::LIGHT_RED),
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
            } else if let Some(why) = refused {
                // Refused before it can disturb anything — in particular
                // the wire it was detached from stays where it is.
                state.last_error = Some(why);
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
/// stashed `Const` to `restore`, so the socket comes back with
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
    use crate::catalog::{self, Kind};
    use texture_graph_core::LayerKind;

    /// A chain albedo → warp → crackle. `crackle` reads `warp` reads
    /// `albedo`. The names are distinctive on purpose: "a" and "c" occur
    /// inside the refusal's own wording, so asserting on them would pass
    /// whether or not the message named anything.
    fn chain() -> (Graph, LayerId, LayerId, LayerId) {
        let mut g = Graph::new();
        let albedo = g.add_layer("albedo", catalog::default_kind(Kind::Color)).unwrap();
        let warp = g.add_layer("warp", reading(albedo)).unwrap();
        let crackle = g.add_layer("crackle", reading(warp)).unwrap();
        (g, albedo, warp, crackle)
    }

    fn reading(src: LayerId) -> LayerKind {
        let LayerKind::Transform(mut t) = catalog::default_kind(Kind::Transform) else {
            unreachable!("Transform's default is a Transform")
        };
        t.source = Some(src);
        LayerKind::Transform(t)
    }

    /// The red-socket rule. This is what the user sees *before* committing,
    /// so it has to agree with what the graph would do — a socket that
    /// stays black and then refuses the drop is worse than one that never
    /// looked inviting.
    #[test]
    fn a_socket_that_would_refuse_the_drop_says_so_in_advance() {
        let (mut g, albedo, _warp, crackle) = chain();

        // crackle already reads albedo transitively, so wiring crackle
        // back into albedo closes the loop.
        let why = refusal(&g, NodeRef::Layer(albedo), crackle)
            .expect("albedo → warp → crackle → albedo is a loop");
        assert!(why.contains("loop"), "unhelpful: {why}");
        assert!(
            why.contains("albedo") && why.contains("crackle"),
            "the message names neither end of the loop: {why}"
        );

        // And the graph agrees: what the socket predicted, `set_kind` does.
        assert!(g.set_kind(albedo, reading(crackle)).is_err());
    }

    /// The other half of the same agreement: a socket that rings black has
    /// to actually accept the drop, or the wire bounces with an error the
    /// user was given no warning about.
    #[test]
    fn a_socket_that_admits_the_drop_really_takes_it() {
        let (mut g, albedo, _warp, crackle) = chain();
        // albedo into crackle runs with the flow, not against it.
        assert_eq!(refusal(&g, NodeRef::Layer(crackle), albedo), None);
        assert!(g.set_kind(crackle, reading(albedo)).is_ok());
    }

    #[test]
    fn a_layer_cannot_read_itself() {
        let (g, albedo, _warp, _crackle) = chain();
        let why =
            refusal(&g, NodeRef::Layer(albedo), albedo).expect("a self-loop is a loop");
        assert!(why.contains("itself"), "unhelpful: {why}");
        assert!(why.contains("albedo"), "the message names no layer: {why}");
    }

    /// Nothing reads the material output, so no wire into it can close a
    /// loop — including one from a layer that reads everything else.
    #[test]
    fn the_material_output_never_refuses_a_wire() {
        let (g, _albedo, _warp, crackle) = chain();
        assert_eq!(refusal(&g, NodeRef::Output, crackle), None);
    }

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
