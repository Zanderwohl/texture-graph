//! Display strings for [`texture_graph_core::WaveShape`], shared by the
//! inspector and the node body for the reason
//! [`crate::widgets::noise_labels`] gives.

use texture_graph_core::WaveShape;

pub const SHAPES: &[WaveShape] = &[
    WaveShape::Sine,
    WaveShape::Triangle,
    WaveShape::Square,
    WaveShape::Sawtooth,
];

pub fn shape(s: WaveShape) -> &'static str {
    match s {
        WaveShape::Sine => "sine",
        WaveShape::Triangle => "triangle",
        WaveShape::Square => "square",
        WaveShape::Sawtooth => "sawtooth",
    }
}
