//! Round-trip conversions between our Oklcha storage and egui's
//! sRGB-space color widgets.
//!
//! We lose precision on out-of-sRGB-gamut Oklch values — that's accepted
//! for MVP; the LCh sliders in `widgets::color_edit` are the escape hatch.

use palette::{IntoColor, Oklch, Srgb};
use texture_graph_core::Color;

/// Oklch (with alpha) → gamma-encoded sRGB `[r, g, b, a]` in `[0, 1]`.
pub fn oklcha_to_srgba(c: Color) -> [f32; 4] {
    let srgb: Srgb = Oklch::new(c.l, c.chroma.max(0.0), c.hue).into_color();
    [
        srgb.red.clamp(0.0, 1.0),
        srgb.green.clamp(0.0, 1.0),
        srgb.blue.clamp(0.0, 1.0),
        c.alpha.clamp(0.0, 1.0),
    ]
}

/// Gamma-encoded sRGB `[r, g, b, a]` in `[0, 1]` → Oklch (with alpha).
pub fn srgba_to_oklcha(rgba: [f32; 4]) -> Color {
    let ok: Oklch = Srgb::new(rgba[0], rgba[1], rgba[2]).into_color();
    Color::new(ok.l, ok.chroma, ok.hue, rgba[3])
}
