//! Deterministic node layout: per-kind row tables → rects and socket
//! geometry, computed before any widget runs so edges, sockets, and
//! interaction all agree on positions with no measure-then-place flicker.

use std::collections::HashMap;

use texture_graph_core::{Graph, InputKey, LayerId, LayerKind, SocketValue};

use crate::state::{NodeRef, UiState};

use super::{world_to_screen, BOTTOM_PAD, HEADER_H, NODE_WIDTH, RAMP_BAR_H, ROW_H, THUMB_H};

/// One visual row in a node body, below the header (and thumbnail).
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Row {
    /// Socket circle on the left edge; label + inline widget when unconnected.
    Socket(InputKey),
    /// No socket — a literal parameter row.
    Param(ParamRow),
}

/// Fixed-height literal parameter rows, per kind.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum ParamRow {
    ColorValue,
    NoiseDims,
    NoiseOutput,
    NoiseRange,
    NoiseFrequency,
    NoiseSeed,
    RampSpace,
    /// Gradient preview bar with draggable stop indicators — taller than a
    /// standard row (see `row_height`).
    RampBar,
    TransformOffset,
    TransformRotate,
    TransformScale,
    TransformCoordMode,
    TransformEdgeMode,
    TransformPermute,
    TransformRadialDim,
    TransformRadialInto,
    MixMode,
    MixSpace,
    MinMaxMode,
    MinMaxCriterion,
    H2nStrength,
}

/// Row table for a layer node. Height is fully determined by the current
/// kind (including conditional rows), never by widget measurement.
pub fn rows_for(kind: &LayerKind) -> Vec<Row> {
    use texture_graph_core::{BlendMode, CoordMode};
    let mut rows = Vec::new();
    match kind {
        LayerKind::Color(_) => rows.push(Row::Param(ParamRow::ColorValue)),
        LayerKind::Noise(_) => rows.extend([
            Row::Param(ParamRow::NoiseDims),
            Row::Param(ParamRow::NoiseOutput),
            Row::Param(ParamRow::NoiseRange),
            Row::Param(ParamRow::NoiseFrequency),
            Row::Param(ParamRow::NoiseSeed),
        ]),
        LayerKind::ColorRamp(r) => {
            rows.push(Row::Param(ParamRow::RampSpace));
            rows.push(Row::Param(ParamRow::RampBar));
            for i in 0..r.stops.len() {
                rows.push(Row::Socket(InputKey::RampStop(i)));
            }
        }
        LayerKind::Transform(t) => {
            rows.extend([
                Row::Socket(InputKey::TransformSource),
                Row::Param(ParamRow::TransformOffset),
                Row::Param(ParamRow::TransformRotate),
                Row::Param(ParamRow::TransformScale),
                Row::Param(ParamRow::TransformCoordMode),
                Row::Param(ParamRow::TransformEdgeMode),
            ]);
            match t.coord_mode {
                CoordMode::Passthrough => {}
                CoordMode::Permute(_) => rows.push(Row::Param(ParamRow::TransformPermute)),
                CoordMode::Radial { .. } => rows.extend([
                    Row::Param(ParamRow::TransformRadialDim),
                    Row::Param(ParamRow::TransformRadialInto),
                ]),
            }
        }
        LayerKind::Mix(m) => {
            rows.extend([
                Row::Socket(InputKey::MixA),
                Row::Socket(InputKey::MixB),
                Row::Param(ParamRow::MixMode),
            ]);
            if matches!(m.mode, BlendMode::Blend) {
                rows.push(Row::Param(ParamRow::MixSpace));
                rows.push(Row::Socket(InputKey::MixFactor));
            }
        }
        LayerKind::Map(_) => rows.extend([
            Row::Socket(InputKey::MapValue),
            Row::Socket(InputKey::MapPalette),
        ]),
        LayerKind::MinMax(_) => rows.extend([
            Row::Socket(InputKey::MinMaxA),
            Row::Socket(InputKey::MinMaxB),
            Row::Param(ParamRow::MinMaxMode),
            Row::Param(ParamRow::MinMaxCriterion),
        ]),
        LayerKind::HeightToNormal(_) => rows.extend([
            Row::Socket(InputKey::H2nSource),
            Row::Param(ParamRow::H2nStrength),
        ]),
    }
    rows
}

/// Row table for the material Output pseudo-node.
pub fn rows_for_output() -> Vec<Row> {
    vec![
        Row::Socket(InputKey::OutColor),
        Row::Socket(InputKey::OutRoughness),
        Row::Socket(InputKey::OutMetallic),
        Row::Socket(InputKey::OutNormal),
    ]
}

/// Display label next to a socket circle.
pub fn socket_label(key: InputKey) -> &'static str {
    match key {
        InputKey::TransformSource | InputKey::H2nSource => "source",
        InputKey::MixA | InputKey::MinMaxA => "a",
        InputKey::MixB | InputKey::MinMaxB => "b",
        InputKey::MixFactor => "factor",
        InputKey::MapValue => "value",
        InputKey::MapPalette => "palette",
        InputKey::RampStop(_) => "stop",
        InputKey::OutColor => "color",
        InputKey::OutRoughness => "roughness",
        InputKey::OutMetallic => "metallic",
        InputKey::OutNormal => "normal",
    }
}

/// One input socket resolved to screen geometry.
pub struct SocketLayout {
    pub key: InputKey,
    /// Circle center, on the node's left edge, vertically centered in its
    /// row.
    pub center: egui::Pos2,
    pub value: SocketValue,
}

/// Everything the edge pass, node pass, and wire pass need to agree on for
/// one node, all in screen space for the current pan/zoom.
pub struct NodeLayout {
    pub node: NodeRef,
    pub rect: egui::Rect,
    pub thumb_rect: Option<egui::Rect>,
    pub rows: Vec<Row>,
    pub row_rects: Vec<egui::Rect>,
    pub inputs: Vec<SocketLayout>,
    /// Right-edge output circle; `None` for the Output pseudo-node.
    pub output_socket: Option<egui::Pos2>,
}

/// Height of one row in world units. Everything is `ROW_H` except the
/// ColorRamp gradient bar.
pub fn row_height(row: &Row) -> f32 {
    match row {
        Row::Param(ParamRow::RampBar) => RAMP_BAR_H,
        _ => ROW_H,
    }
}

/// Node height in world units for a given row table.
pub fn node_height(rows: &[Row], thumb: bool) -> f32 {
    HEADER_H
        + if thumb { THUMB_H } else { 0.0 }
        + rows.iter().map(row_height).sum::<f32>()
        + BOTTOM_PAD
}

/// Build layouts for every layer node plus the Output pseudo-node.
pub fn compute_layouts(
    graph: &Graph,
    state: &UiState,
    positions: &HashMap<LayerId, egui::Pos2>,
    output_pos: egui::Pos2,
    origin: egui::Pos2,
) -> Vec<NodeLayout> {
    let z = state.canvas_zoom;
    let mut out = Vec::with_capacity(graph.layers.len() + 1);

    for layer in &graph.layers {
        let Some(world) = positions.get(&layer.id).copied() else { continue };
        let rows = rows_for(&layer.kind);
        let sockets = layer.kind.input_sockets();
        out.push(build_layout(
            NodeRef::Layer(layer.id),
            world_to_screen(world, origin, state),
            rows,
            &sockets,
            true,
            true,
            z,
        ));
    }

    let out_sockets = graph.output.input_sockets();
    out.push(build_layout(
        NodeRef::Output,
        world_to_screen(output_pos, origin, state),
        rows_for_output(),
        &out_sockets,
        false,
        false,
        z,
    ));
    out
}

fn build_layout(
    node: NodeRef,
    top_left: egui::Pos2,
    rows: Vec<Row>,
    sockets: &[texture_graph_core::InputSocket],
    thumb: bool,
    has_output: bool,
    z: f32,
) -> NodeLayout {
    let height = node_height(&rows, thumb);
    let rect = egui::Rect::from_min_size(top_left, egui::vec2(NODE_WIDTH, height) * z);

    let thumb_rect = thumb.then(|| {
        egui::Rect::from_min_size(
            top_left + egui::vec2(0.0, HEADER_H) * z,
            egui::vec2(NODE_WIDTH, THUMB_H) * z,
        )
    });

    let rows_top = top_left.y + (HEADER_H + if thumb { THUMB_H } else { 0.0 }) * z;
    let mut y = rows_top;
    let row_rects: Vec<egui::Rect> = rows
        .iter()
        .map(|row| {
            let h = row_height(row) * z;
            let r = egui::Rect::from_min_size(
                egui::pos2(top_left.x, y),
                egui::vec2(NODE_WIDTH * z, h),
            );
            y += h;
            r
        })
        .collect();

    let inputs = rows
        .iter()
        .zip(&row_rects)
        .filter_map(|(row, rrect)| {
            let Row::Socket(key) = *row else { return None };
            let value = sockets.iter().find(|s| s.key == key)?.value;
            Some(SocketLayout {
                key,
                center: egui::pos2(rect.left(), rrect.center().y),
                value,
            })
        })
        .collect();

    let output_socket =
        has_output.then(|| egui::pos2(rect.right(), rect.top() + HEADER_H * 0.5 * z));

    NodeLayout {
        node,
        rect,
        thumb_rect,
        rows,
        row_rects,
        inputs,
        output_socket,
    }
}
