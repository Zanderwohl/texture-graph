use palette::{Hsv, IntoColor, LinSrgb, Oklch, Srgb, WithAlpha};
use serde::{Deserialize, Serialize};

/// Working color type: Oklch + alpha, hue in degrees. Not clamped between
/// layers, so L and C may leave their nominal ranges (signed noise, additive
/// octaves). Clamping happens at [`Output`](crate::Output).
pub type Color = palette::Oklcha<f32>;

/// Space in which two colors are interpolated.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum BlendSpace {
    /// Perceptually uniform. Default.
    Oklch,
    /// Linear-light sRGB, physically correct for optical mixing.
    LinearSrgb,
    /// HSV of gamma-encoded sRGB.
    Hsv,
}

impl Default for BlendSpace {
    fn default() -> Self {
        BlendSpace::Oklch
    }
}

/// Interpolate `a` and `b` at `t ∈ [0, 1]`. `t` is not clamped, so this
/// extrapolates outside that range.
pub fn blend(a: Color, b: Color, t: f32, space: BlendSpace) -> Color {
    let alpha = lerp(a.alpha, b.alpha, t);
    match space {
        BlendSpace::Oklch => {
            let c = lerp_oklch(a.color, b.color, t);
            c.with_alpha(alpha)
        }
        BlendSpace::LinearSrgb => {
            let la: LinSrgb = a.color.into_color();
            let lb: LinSrgb = b.color.into_color();
            let mixed = LinSrgb::new(
                lerp(la.red, lb.red, t),
                lerp(la.green, lb.green, t),
                lerp(la.blue, lb.blue, t),
            );
            let back: Oklch = mixed.into_color();
            back.with_alpha(alpha)
        }
        BlendSpace::Hsv => {
            let sa: Srgb = a.color.into_color();
            let sb: Srgb = b.color.into_color();
            let ha: Hsv = sa.into_color();
            let hb: Hsv = sb.into_color();
            let mixed = Hsv::new(
                lerp_hue_deg(ha.hue.into_degrees(), hb.hue.into_degrees(), t),
                lerp(ha.saturation, hb.saturation, t),
                lerp(ha.value, hb.value, t),
            );
            let back_srgb: Srgb = mixed.into_color();
            let back: Oklch = back_srgb.into_color();
            back.with_alpha(alpha)
        }
    }
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn lerp_oklch(a: Oklch, b: Oklch, t: f32) -> Oklch {
    let ha = a.hue.into_degrees();
    let hb = b.hue.into_degrees();
    Oklch::new(
        lerp(a.l, b.l, t),
        lerp(a.chroma, b.chroma, t),
        lerp_hue_deg(ha, hb, t),
    )
}

/// Shortest-arc hue lerp in degrees.
fn lerp_hue_deg(a: f32, b: f32, t: f32) -> f32 {
    let mut d = (b - a) % 360.0;
    if d > 180.0 {
        d -= 360.0;
    } else if d < -180.0 {
        d += 360.0;
    }
    a + d * t
}

/// Oklch L, the scalar read from a color by
/// [`ScalarInput::Layer`](crate::ScalarInput::Layer).
pub fn scalar_of(c: Color) -> f32 {
    c.l
}

/// Convert to 8-bit sRGBA, clamping L, C and alpha to display range and
/// clipping out-of-gamut channels.
pub fn to_srgb8(c: Color) -> [u8; 4] {
    let clamped = Oklch::new(c.l.clamp(0.0, 1.0), c.chroma.max(0.0), c.hue);
    let srgb: Srgb = clamped.into_color();
    let r = (srgb.red.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    let g = (srgb.green.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    let b = (srgb.blue.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    let a = (c.alpha.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    [r, g, b, a]
}

/// Hue in degrees.
pub fn oklcha(l: f32, chroma: f32, hue_deg: f32, alpha: f32) -> Color {
    Color::new(l, chroma, hue_deg, alpha)
}

/// Encode a tangent-space normal (components in [-1, 1]) as sRGB
/// `n * 0.5 + 0.5`.
pub fn normal_to_color(n: [f32; 3]) -> Color {
    let s = Srgb::new(n[0] * 0.5 + 0.5, n[1] * 0.5 + 0.5, n[2] * 0.5 + 0.5);
    let ok: Oklch = s.into_color();
    ok.with_alpha(1.0)
}

