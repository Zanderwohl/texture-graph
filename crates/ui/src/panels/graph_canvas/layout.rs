//! Node layout: per-kind row tables to rects and socket geometry, resolved
//! before any widget runs so edges, sockets and interaction agree on
//! positions without measure-then-place flicker.

use std::collections::HashMap;

use texture_graph_core::{Graph, InputKey, LayerId, LayerKind, SocketValue};

use crate::state::{NodeRef, UiState};

use super::{world_to_screen, BOTTOM_PAD, HEADER_H, NODE_WIDTH, RAMP_BAR_H, ROW_H, THUMB_H};

/// One visual row in a node body, below the header (and thumbnail).
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Row {
    /// Socket on the left edge; label and widget when unconnected.
    Socket(InputKey),
    Param(ParamRow),
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum ParamRow {
    ColorValue,
    NoiseDims,
    NoiseOutput,
    NoiseRange,
    NoiseFrequency,
    NoiseSeed,
    NoiseKernel,
    /// Only shown for the value kernel — simplex has no lattice to wrap.
    NoisePeriod,
    NoiseOctaves,
    /// This and the next three rows only appear with more than one octave.
    NoiseFractalMode,
    NoiseLacunarity,
    NoiseGain,
    NoiseNormalize,
    RampSpace,
    /// Gradient bar with draggable indicators; taller than a standard row.
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
    WaveShape,
    WaveFrequency,
    WavePhase,
    WaveRange,
    WarpMode,
    WarpAmount,
    CoordinateAxis,
}

/// Row table for a layer node. Height follows the kind, conditional rows
/// included, never widget measurement.
pub fn rows_for(kind: &LayerKind) -> Vec<Row> {
    use texture_graph_core::{BlendMode, CoordMode, ScalarInput};
    let mut rows = Vec::new();
    match kind {
        LayerKind::Color(_) => rows.push(Row::Param(ParamRow::ColorValue)),
        LayerKind::Noise(n) => {
            rows.extend([
                Row::Param(ParamRow::NoiseKernel),
                Row::Param(ParamRow::NoiseDims),
                Row::Param(ParamRow::NoiseOutput),
                Row::Param(ParamRow::NoiseRange),
                Row::Param(ParamRow::NoiseFrequency),
                Row::Param(ParamRow::NoiseSeed),
            ]);
            if n.kernel == texture_graph_core::NoiseKernel::Value {
                rows.push(Row::Param(ParamRow::NoisePeriod));
            }
            rows.push(Row::Param(ParamRow::NoiseOctaves));
            if n.fractal.octaves > 1 {
                rows.extend([
                    Row::Param(ParamRow::NoiseFractalMode),
                    Row::Param(ParamRow::NoiseLacunarity),
                    Row::Param(ParamRow::NoiseGain),
                    Row::Param(ParamRow::NoiseNormalize),
                ]);
            }
        }
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
            }
            // Only Blend reads the factor, but a wired factor stays an edge in
            // every mode. Without a row its wire has no anchor, and the
            // inspector hides it too, so nothing could reach it.
            if matches!(m.mode, BlendMode::Blend)
                || matches!(m.factor, ScalarInput::Layer(_))
            {
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
        LayerKind::Warp(_) => rows.extend([
            Row::Socket(InputKey::WarpSource),
            Row::Socket(InputKey::WarpBy),
            Row::Param(ParamRow::WarpMode),
            Row::Param(ParamRow::WarpAmount),
        ]),
        LayerKind::Wave(_) => rows.extend([
            Row::Socket(InputKey::WaveInput),
            Row::Param(ParamRow::WaveShape),
            Row::Param(ParamRow::WaveFrequency),
            Row::Param(ParamRow::WavePhase),
            Row::Param(ParamRow::WaveRange),
        ]),
        LayerKind::Coordinate(_) => rows.push(Row::Param(ParamRow::CoordinateAxis)),
    }
    rows
}

pub fn rows_for_output() -> Vec<Row> {
    vec![
        Row::Socket(InputKey::OutColor),
        Row::Socket(InputKey::OutRoughness),
        Row::Socket(InputKey::OutMetallic),
        Row::Socket(InputKey::OutNormal),
    ]
}

pub fn socket_label(key: InputKey) -> &'static str {
    match key {
        InputKey::TransformSource | InputKey::H2nSource | InputKey::WarpSource => "source",
        InputKey::MixA | InputKey::MinMaxA => "a",
        InputKey::MixB | InputKey::MinMaxB => "b",
        InputKey::MixFactor => "factor",
        InputKey::WaveInput => "input",
        InputKey::WarpBy => "by",
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
    /// Circle center, on the left edge and centerd in its row.
    pub center: egui::Pos2,
    pub value: SocketValue,
}

/// What the edge, node and wire passes must agree on for one node, in screen
/// space at the current pan and zoom.
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

pub fn row_height(row: &Row) -> f32 {
    match row {
        Row::Param(ParamRow::RampBar) => RAMP_BAR_H,
        _ => ROW_H,
    }
}

/// World units.
pub fn node_height(rows: &[Row], thumb: bool) -> f32 {
    HEADER_H
        + if thumb { THUMB_H } else { 0.0 }
        + rows.iter().map(row_height).sum::<f32>()
        + BOTTOM_PAD
}

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
            let value = sockets.iter().find(|s| s.key == key)?.value.clone();
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::{self, Kind};
    use texture_graph_core::color::oklcha;
    use texture_graph_core::{
        Axis, BlendMode, ColorInput, ColorStop, CoordMode, Graph, RadialDim, ScalarInput,
    };

    fn output() -> texture_graph_core::Output {
        Graph::new().output
    }

    fn laid_out(kind: &LayerKind) -> Vec<InputKey> {
        rows_for(kind)
            .into_iter()
            .filter_map(|r| match r {
                Row::Socket(key) => Some(key),
                Row::Param(_) => None,
            })
            .collect()
    }

    fn modeled(kind: &LayerKind) -> Vec<InputKey> {
        kind.input_sockets().into_iter().map(|s| s.key).collect()
    }

    /// A socket drawn but not modeled loses its wire; one modeled but not
    /// drawn cannot be connected. Exact equality holds only for catalog
    /// defaults, since `Mix` hides an unwired factor outside Blend.
    #[test]
    fn every_node_lays_out_exactly_the_sockets_it_has() {
        for variant in catalog::VARIANTS {
            let kind = catalog::default_kind(variant.kind);
            assert_eq!(
                laid_out(&kind),
                modeled(&kind),
                "{} lays out the wrong sockets",
                variant.label
            );
        }
    }

    /// The Output pseudo-node isn't a catalog variant, so the sweep above
    /// never reaches it.
    #[test]
    fn the_output_lays_out_exactly_the_sockets_it_has() {
        let output = output();
        let laid_out: Vec<InputKey> = rows_for_output()
            .into_iter()
            .filter_map(|r| match r {
                Row::Socket(key) => Some(key),
                Row::Param(_) => None,
            })
            .collect();
        let modeled: Vec<InputKey> =
            output.input_sockets().into_iter().map(|s| s.key).collect();
        assert_eq!(laid_out, modeled);
    }

    /// Row labels are written out here, so they can drift from the model's.
    #[test]
    fn socket_labels_say_what_the_model_says() {
        for variant in catalog::VARIANTS {
            let kind = catalog::default_kind(variant.kind);
            for socket in kind.input_sockets() {
                assert_eq!(
                    socket_label(socket.key),
                    socket.label,
                    "{} socket {:?}",
                    variant.label,
                    socket.key
                );
            }
        }
        for socket in output().input_sockets() {
            assert_eq!(socket_label(socket.key), socket.label, "output {:?}", socket.key);
        }
    }

    /// A conditional row missing from `rows_for` would draw a widget outside
    /// the node body.
    #[test]
    fn conditional_rows_change_the_height_they_are_on() {
        // Mix: Blend adds a space row and a factor socket.
        let LayerKind::Mix(mut m) = catalog::default_kind(Kind::Mix) else { panic!() };
        m.mode = BlendMode::Add;
        let plain = node_height(&rows_for(&LayerKind::Mix(m.clone())), true);
        m.mode = BlendMode::Blend;
        assert!(
            node_height(&rows_for(&LayerKind::Mix(m)), true) > plain,
            "Blend's extra rows did not make the node taller"
        );

        // Transform: each coord mode carries its own parameter rows.
        let LayerKind::Transform(mut t) = catalog::default_kind(Kind::Transform) else {
            panic!()
        };
        t.coord_mode = CoordMode::Passthrough;
        let passthrough = node_height(&rows_for(&LayerKind::Transform(t)), true);
        t.coord_mode = CoordMode::Radial { dim: RadialDim::D2, into: Axis::U };
        assert!(
            node_height(&rows_for(&LayerKind::Transform(t)), true) > passthrough,
            "Radial's extra rows did not make the node taller"
        );
    }

    /// A stop with no row cannot be wired.
    #[test]
    fn a_ramp_lays_out_one_socket_per_stop() {
        let LayerKind::ColorRamp(mut r) = catalog::default_kind(Kind::ColorRamp) else {
            panic!()
        };
        assert_eq!(laid_out(&LayerKind::ColorRamp(r.clone())).len(), r.stops.len());
        r.stops.push(ColorStop {
            t: 0.5,
            color: ColorInput::Const(oklcha(0.5, 0.0, 0.0, 1.0)),
        });
        let kind = LayerKind::ColorRamp(r);
        assert_eq!(laid_out(&kind), modeled(&kind));
    }

    /// A factor wired in Blend and switched to Add is still an edge. Without
    /// a row it vanishes from the canvas and the inspector, with no way to
    /// reach it.
    #[test]
    fn a_connected_socket_always_has_a_row_to_hang_its_wire_on() {
        let modes = [
            BlendMode::Add,
            BlendMode::Subtract,
            BlendMode::Multiply,
            BlendMode::Blend,
        ];
        for mode in modes {
            let LayerKind::Mix(mut m) = catalog::default_kind(Kind::Mix) else { panic!() };
            m.mode = mode;
            m.factor = ScalarInput::Layer(LayerId(1));
            let kind = LayerKind::Mix(m);
            let rows = laid_out(&kind);
            for socket in kind.input_sockets() {
                if socket.value.connected_to().is_some() {
                    assert!(
                        rows.contains(&socket.key),
                        "{mode:?}: {:?} is wired but has no row",
                        socket.key
                    );
                }
            }
        }
    }
}
