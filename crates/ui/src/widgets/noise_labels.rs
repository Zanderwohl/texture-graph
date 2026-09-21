//! Display strings for the [`texture_graph_core::Noise`] enums.
//!
//! Shared because the same field is edited in two places — the inspector
//! column and the node body on the canvas — and a control that reads
//! "ridged" in one and "Ridged" in the other is a bug report waiting to
//! happen.

use texture_graph_core::{FractalMode, NoiseDims, NoiseKernel, NoiseOutput, NoiseRange};

pub const DIMS: &[NoiseDims] = &[NoiseDims::D1, NoiseDims::D2, NoiseDims::D3];
pub const OUTPUTS: &[NoiseOutput] = &[NoiseOutput::Grayscale, NoiseOutput::Color];
pub const RANGES: &[NoiseRange] = &[NoiseRange::Unsigned, NoiseRange::Signed];
pub const KERNELS: &[NoiseKernel] = &[NoiseKernel::Simplex, NoiseKernel::Value];
pub const FRACTAL_MODES: &[FractalMode] =
    &[FractalMode::Standard, FractalMode::Turbulence, FractalMode::Ridged];

pub fn dims(d: NoiseDims) -> &'static str {
    match d {
        NoiseDims::D1 => "1D",
        NoiseDims::D2 => "2D",
        NoiseDims::D3 => "3D",
    }
}

pub fn output(o: NoiseOutput) -> &'static str {
    match o {
        NoiseOutput::Grayscale => "grayscale",
        NoiseOutput::Color => "color (LCh)",
    }
}

pub fn range(r: NoiseRange) -> &'static str {
    match r {
        NoiseRange::Unsigned => "[0, 1]",
        NoiseRange::Signed => "[-1, 1]",
    }
}

pub fn kernel(k: NoiseKernel) -> &'static str {
    match k {
        NoiseKernel::Simplex => "simplex",
        NoiseKernel::Value => "value (tileable)",
    }
}

pub fn fractal_mode(m: FractalMode) -> &'static str {
    match m {
        FractalMode::Standard => "standard",
        FractalMode::Turbulence => "turbulence",
        FractalMode::Ridged => "ridged",
    }
}

/// What the current `frequency`/`period` pair actually does, in words.
///
/// The repeat length is `period / frequency` sample-space units, and only
/// `period == frequency` is seamless across the unit cube. Saying so beats
/// rounding the number the user typed into one that happens to tile.
pub fn period_hint(frequency: f32, period: [u32; 3]) -> String {
    if period == [0; 3] {
        return "no period — the field never repeats".to_string();
    }
    if frequency <= 0.0 {
        return "frequency must be positive for a period to mean anything".to_string();
    }
    let axes = ["u", "v", "w"];
    let mut parts = Vec::new();
    for (i, p) in period.iter().enumerate() {
        if *p == 0 {
            continue;
        }
        let repeat = *p as f32 / frequency;
        let seamless = (repeat - 1.0).abs() < 1e-4;
        parts.push(format!(
            "{} every {repeat:.3}{}",
            axes[i],
            if seamless { " (seamless on the unit cube)" } else { "" }
        ));
    }
    format!("repeats: {}", parts.join(", "))
}
