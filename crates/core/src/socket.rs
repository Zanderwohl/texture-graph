//! Socket reflection over [`LayerKind`] and [`Output`], so a node editor can
//! enumerate and rewire inputs without matching on every variant.
//!
//! [`LayerKind::input_sockets`] must agree with [`LayerKind::inputs`] on both
//! ids and order; the test below holds them together.

use crate::color::Color;
use crate::graph::Output;
use crate::id::LayerId;
use crate::kind::{ColorInput, LayerKind, ScalarInput};

/// Identifies one input socket on a layer or on the graph [`Output`].
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum InputKey {
    TransformSource,
    MixA,
    MixB,
    MixFactor,
    MapValue,
    MapPalette,
    MinMaxA,
    MinMaxB,
    H2nSource,
    WaveInput,
    WarpSource,
    WarpBy,
    RampStop(usize),
    OutColor,
    OutRoughness,
    OutMetallic,
    OutNormal,
}

/// Current value of a socket, as read from the model.
#[derive(Copy, Clone, Debug)]
pub enum SocketValue {
    /// An `Option<LayerId>` field. `None` renders as the missing-texture
    /// grid.
    LayerOpt(Option<LayerId>),
    Color(ColorInput),
    Scalar(ScalarInput),
}

impl SocketValue {
    /// The layer this socket is wired to, if any.
    pub fn connected_to(&self) -> Option<LayerId> {
        match *self {
            SocketValue::LayerOpt(opt) => opt,
            SocketValue::Color(ColorInput::Layer(id)) => Some(id),
            SocketValue::Color(ColorInput::Const(_)) => None,
            SocketValue::Scalar(ScalarInput::Layer(id)) => Some(id),
            SocketValue::Scalar(ScalarInput::Const(_)) => None,
        }
    }
}

/// One connectable input on a node, in top-to-bottom display order.
#[derive(Copy, Clone, Debug)]
pub struct InputSocket {
    pub key: InputKey,
    pub label: &'static str,
    pub value: SocketValue,
}

#[derive(Debug, thiserror::Error)]
pub enum SocketError {
    #[error("no such socket on this node")]
    NoSuchSocket,
}

/// A constant writable into a `ColorInput`/`ScalarInput` socket, so an editor
/// can restore what a connection displaced.
#[derive(Copy, Clone, Debug)]
pub enum ConstValue {
    Color(Color),
    Scalar(f32),
}

/// Mid gray, matching the UI's defaults.
fn disconnect_color() -> Color {
    crate::color::oklcha(0.5, 0.0, 0.0, 1.0)
}
/// Const value a `ScalarInput` socket falls back to on disconnect.
const DISCONNECT_SCALAR: f32 = 0.5;

/// Write `target` into a socket slot, returning what was there before.
fn set_opt(slot: &mut Option<LayerId>, target: Option<LayerId>) -> SocketValue {
    SocketValue::LayerOpt(std::mem::replace(slot, target))
}

fn set_color(slot: &mut ColorInput, target: Option<LayerId>) -> SocketValue {
    let new = match target {
        Some(id) => ColorInput::Layer(id),
        None => ColorInput::Const(disconnect_color()),
    };
    SocketValue::Color(std::mem::replace(slot, new))
}

fn set_scalar(slot: &mut ScalarInput, target: Option<LayerId>) -> SocketValue {
    let new = match target {
        Some(id) => ScalarInput::Layer(id),
        None => ScalarInput::Const(DISCONNECT_SCALAR),
    };
    SocketValue::Scalar(std::mem::replace(slot, new))
}

impl LayerKind {
    /// Ordered list of connectable input sockets (top-to-bottom node order).
    pub fn input_sockets(&self) -> Vec<InputSocket> {
        let sock = |key, label, value| InputSocket { key, label, value };
        match self {
            LayerKind::Color(_) | LayerKind::Noise(_) => Vec::new(),
            LayerKind::ColorRamp(r) => r
                .stops
                .iter()
                .enumerate()
                .map(|(i, s)| {
                    sock(InputKey::RampStop(i), "stop", SocketValue::Color(s.color))
                })
                .collect(),
            LayerKind::Transform(t) => vec![sock(
                InputKey::TransformSource,
                "source",
                SocketValue::LayerOpt(t.source),
            )],
            LayerKind::Mix(m) => vec![
                sock(InputKey::MixA, "a", SocketValue::LayerOpt(m.a)),
                sock(InputKey::MixB, "b", SocketValue::LayerOpt(m.b)),
                sock(InputKey::MixFactor, "factor", SocketValue::Scalar(m.factor)),
            ],
            LayerKind::Map(m) => vec![
                sock(InputKey::MapValue, "value", SocketValue::LayerOpt(m.value)),
                sock(
                    InputKey::MapPalette,
                    "palette",
                    SocketValue::LayerOpt(m.palette),
                ),
            ],
            LayerKind::MinMax(mm) => vec![
                sock(InputKey::MinMaxA, "a", SocketValue::LayerOpt(mm.a)),
                sock(InputKey::MinMaxB, "b", SocketValue::LayerOpt(mm.b)),
            ],
            LayerKind::HeightToNormal(h) => vec![sock(
                InputKey::H2nSource,
                "source",
                SocketValue::LayerOpt(h.source),
            )],
            LayerKind::Wave(w) => {
                vec![sock(InputKey::WaveInput, "input", SocketValue::Scalar(w.input))]
            }
            LayerKind::Warp(w) => vec![
                sock(InputKey::WarpSource, "source", SocketValue::LayerOpt(w.source)),
                sock(InputKey::WarpBy, "by", SocketValue::LayerOpt(w.by)),
            ],
        }
    }

    /// Connect or disconnect one socket. A disconnected
    /// `ColorInput`/`ScalarInput` falls back to a `Const` default —
    /// callers that want to restore a previous const value can do so from
    /// the returned replaced [`SocketValue`].
    pub fn set_input(
        &mut self,
        key: InputKey,
        target: Option<LayerId>,
    ) -> Result<SocketValue, SocketError> {
        match (self, key) {
            (LayerKind::Transform(t), InputKey::TransformSource) => {
                Ok(set_opt(&mut t.source, target))
            }
            (LayerKind::Mix(m), InputKey::MixA) => Ok(set_opt(&mut m.a, target)),
            (LayerKind::Mix(m), InputKey::MixB) => Ok(set_opt(&mut m.b, target)),
            (LayerKind::Mix(m), InputKey::MixFactor) => Ok(set_scalar(&mut m.factor, target)),
            (LayerKind::Map(m), InputKey::MapValue) => Ok(set_opt(&mut m.value, target)),
            (LayerKind::Map(m), InputKey::MapPalette) => Ok(set_opt(&mut m.palette, target)),
            (LayerKind::MinMax(mm), InputKey::MinMaxA) => Ok(set_opt(&mut mm.a, target)),
            (LayerKind::MinMax(mm), InputKey::MinMaxB) => Ok(set_opt(&mut mm.b, target)),
            (LayerKind::HeightToNormal(h), InputKey::H2nSource) => {
                Ok(set_opt(&mut h.source, target))
            }
            (LayerKind::Wave(w), InputKey::WaveInput) => Ok(set_scalar(&mut w.input, target)),
            (LayerKind::Warp(w), InputKey::WarpSource) => Ok(set_opt(&mut w.source, target)),
            (LayerKind::Warp(w), InputKey::WarpBy) => Ok(set_opt(&mut w.by, target)),
            (LayerKind::ColorRamp(r), InputKey::RampStop(i)) => match r.stops.get_mut(i) {
                // A stop can vanish between drag start and drop (removed in
                // the inspector) — report it rather than panic.
                Some(stop) => Ok(set_color(&mut stop.color, target)),
                None => Err(SocketError::NoSuchSocket),
            },
            _ => Err(SocketError::NoSuchSocket),
        }
    }

    /// Write a constant into a `ColorInput`/`ScalarInput` socket. Errors on
    /// plain-layer sockets and type mismatches.
    pub fn set_const(&mut self, key: InputKey, value: ConstValue) -> Result<(), SocketError> {
        match (self, key, value) {
            (LayerKind::Mix(m), InputKey::MixFactor, ConstValue::Scalar(v)) => {
                m.factor = ScalarInput::Const(v);
                Ok(())
            }
            (LayerKind::Wave(w), InputKey::WaveInput, ConstValue::Scalar(v)) => {
                w.input = ScalarInput::Const(v);
                Ok(())
            }
            (LayerKind::ColorRamp(r), InputKey::RampStop(i), ConstValue::Color(c)) => {
                match r.stops.get_mut(i) {
                    Some(stop) => {
                        stop.color = ColorInput::Const(c);
                        Ok(())
                    }
                    None => Err(SocketError::NoSuchSocket),
                }
            }
            _ => Err(SocketError::NoSuchSocket),
        }
    }
}

impl Output {
    /// The material output's input sockets, top-to-bottom.
    pub fn input_sockets(&self) -> Vec<InputSocket> {
        vec![
            InputSocket {
                key: InputKey::OutColor,
                label: "color",
                value: SocketValue::LayerOpt(self.color),
            },
            InputSocket {
                key: InputKey::OutRoughness,
                label: "roughness",
                value: SocketValue::Scalar(self.roughness),
            },
            InputSocket {
                key: InputKey::OutMetallic,
                label: "metallic",
                value: SocketValue::Scalar(self.metallic),
            },
            InputSocket {
                key: InputKey::OutNormal,
                label: "normal",
                value: SocketValue::LayerOpt(self.normal),
            },
        ]
    }

    /// Connect or disconnect one output socket.
    pub fn set_input(
        &mut self,
        key: InputKey,
        target: Option<LayerId>,
    ) -> Result<SocketValue, SocketError> {
        match key {
            InputKey::OutColor => Ok(set_opt(&mut self.color, target)),
            InputKey::OutRoughness => Ok(set_scalar(&mut self.roughness, target)),
            InputKey::OutMetallic => Ok(set_scalar(&mut self.metallic, target)),
            InputKey::OutNormal => Ok(set_opt(&mut self.normal, target)),
            _ => Err(SocketError::NoSuchSocket),
        }
    }

    /// Write a constant into `roughness`/`metallic`. Errors elsewhere.
    pub fn set_const(&mut self, key: InputKey, value: ConstValue) -> Result<(), SocketError> {
        match (key, value) {
            (InputKey::OutRoughness, ConstValue::Scalar(v)) => {
                self.roughness = ScalarInput::Const(v);
                Ok(())
            }
            (InputKey::OutMetallic, ConstValue::Scalar(v)) => {
                self.metallic = ScalarInput::Const(v);
                Ok(())
            }
            _ => Err(SocketError::NoSuchSocket),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::{BlendSpace, oklcha};
    use crate::kind::{
        BlendMode, ColorRamp, ColorStop, CoordMode, Criterion, EdgeMode, HeightToNormal, Map, MinMax,
        MinMaxMode, Mix, Transform,
    };

    fn id(n: u64) -> LayerId {
        LayerId(n)
    }

    /// One fully-connected instance of every variant that has inputs.
    fn connected_samples() -> Vec<LayerKind> {
        vec![
            LayerKind::ColorRamp(ColorRamp {
                stops: vec![
                    ColorStop { t: 0.0, color: ColorInput::Layer(id(2)) },
                    ColorStop { t: 1.0, color: ColorInput::Layer(id(3)) },
                ],
                space: BlendSpace::Oklch,
            }),
            LayerKind::Transform(Transform {
                source: Some(id(4)),
                offset: [0.0; 3],
                rotate_uv: 0.0,
                scale: [1.0; 3],
                coord_mode: CoordMode::Passthrough,
                edge_mode: EdgeMode::default(),
            }),
            LayerKind::Mix(Mix {
                a: Some(id(5)),
                b: Some(id(6)),
                mode: BlendMode::Blend,
                factor: ScalarInput::Layer(id(7)),
                space: BlendSpace::Oklch,
            }),
            LayerKind::Map(Map { value: Some(id(8)), palette: Some(id(9)) }),
            LayerKind::MinMax(MinMax {
                a: Some(id(10)),
                b: Some(id(11)),
                mode: MinMaxMode::Min,
                criterion: Criterion::Luma,
            }),
            LayerKind::HeightToNormal(HeightToNormal { source: Some(id(12)), strength: 1.0 }),
            LayerKind::Wave(crate::kind::Wave {
                input: ScalarInput::Layer(id(13)),
                ..crate::kind::Wave::default()
            }),
            LayerKind::Warp(crate::kind::Warp {
                source: Some(id(14)),
                by: Some(id(15)),
                ..crate::kind::Warp::default()
            }),
        ]
    }

    #[test]
    fn input_sockets_match_inputs_for_connected_nodes() {
        for kind in connected_samples() {
            let from_sockets: Vec<LayerId> = kind
                .input_sockets()
                .iter()
                .filter_map(|s| s.value.connected_to())
                .collect();
            assert_eq!(from_sockets, kind.inputs(), "mismatch for {kind:?}");
        }
    }

    #[test]
    fn set_input_round_trips_every_socket() {
        for mut kind in connected_samples() {
            for sock in kind.clone().input_sockets() {
                // Rewire to a fresh id.
                let prev = kind.set_input(sock.key, Some(id(99))).unwrap();
                assert_eq!(prev.connected_to(), sock.value.connected_to());
                let read = kind
                    .input_sockets()
                    .into_iter()
                    .find(|s| s.key == sock.key)
                    .unwrap();
                assert_eq!(read.value.connected_to(), Some(id(99)));

                // Disconnect.
                kind.set_input(sock.key, None).unwrap();
                let read = kind
                    .input_sockets()
                    .into_iter()
                    .find(|s| s.key == sock.key)
                    .unwrap();
                assert_eq!(read.value.connected_to(), None);
                match read.value {
                    SocketValue::LayerOpt(v) => assert!(v.is_none()),
                    SocketValue::Color(ColorInput::Const(_)) => {}
                    SocketValue::Scalar(ScalarInput::Const(v)) => assert_eq!(v, 0.5),
                    other => panic!("unexpected disconnected value {other:?}"),
                }
            }
        }
    }

    #[test]
    fn wrong_key_is_no_such_socket() {
        let mut kind = LayerKind::Map(Map { value: None, palette: None });
        assert!(matches!(
            kind.set_input(InputKey::MixA, Some(id(1))),
            Err(SocketError::NoSuchSocket)
        ));
        let mut color = LayerKind::Color(oklcha(0.5, 0.0, 0.0, 1.0));
        assert!(matches!(
            color.set_input(InputKey::TransformSource, None),
            Err(SocketError::NoSuchSocket)
        ));
    }

    #[test]
    fn out_of_range_ramp_stop_is_no_such_socket() {
        let mut kind = LayerKind::ColorRamp(ColorRamp {
            stops: vec![
                ColorStop { t: 0.0, color: ColorInput::Const(oklcha(0.0, 0.0, 0.0, 1.0)) },
                ColorStop { t: 1.0, color: ColorInput::Const(oklcha(1.0, 0.0, 0.0, 1.0)) },
            ],
            space: BlendSpace::Oklch,
        });
        assert!(matches!(
            kind.set_input(InputKey::RampStop(2), Some(id(1))),
            Err(SocketError::NoSuchSocket)
        ));
    }

    #[test]
    fn output_sockets_round_trip() {
        let mut out = Output {
            color: Some(id(1)),
            roughness: ScalarInput::Const(0.5),
            metallic: ScalarInput::Const(0.0),
            normal: None,
        };
        // Color is nullable like any other input — `None` renders as the
        // missing-texture grid.
        out.set_input(InputKey::OutColor, None).unwrap();
        assert_eq!(out.color, None);
        out.set_input(InputKey::OutColor, Some(id(2))).unwrap();
        assert_eq!(out.color, Some(id(2)));

        out.set_input(InputKey::OutNormal, Some(id(3))).unwrap();
        assert_eq!(out.normal, Some(id(3)));
        out.set_input(InputKey::OutNormal, None).unwrap();
        assert_eq!(out.normal, None);

        assert!(matches!(
            out.set_input(InputKey::MixA, Some(id(2))),
            Err(SocketError::NoSuchSocket)
        ));
    }
}
