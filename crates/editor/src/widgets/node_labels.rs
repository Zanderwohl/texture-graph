//! Display strings for the node enums, shared so the inspector and the
//! canvas node bodies label the same field the same way.

use texture_graph_core::{
    Axis, CraterOutput, CraterSurface, FractalMode, NoiseDims, NoiseKernel, NoiseOutput, NoiseRange, WarpMode, WaveShape,
};

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

/// The repeat length is `period / frequency` sample-space units, and only
/// `period == frequency` is seamless on the unit cube. The hint says so
/// instead of rounding the user's numbers to ones that tile.
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

pub const AXES: &[Axis] = &[Axis::U, Axis::V, Axis::W];

pub fn axis(a: Axis) -> &'static str {
    match a {
        Axis::U => "u",
        Axis::V => "v",
        Axis::W => "w",
    }
}

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

pub const WARP_MODES: &[WarpMode] = &[WarpMode::Scalar, WarpMode::Vector];

pub fn warp_mode(m: WarpMode) -> &'static str {
    match m {
        WarpMode::Scalar => "scalar (L)",
        WarpMode::Vector => "vector (L, C, hue)",
    }
}

pub const CRATER_SURFACES: &[CraterSurface] = &[CraterSurface::Plane, CraterSurface::Sphere];

pub fn crater_surface(s: CraterSurface) -> &'static str {
    match s {
        CraterSurface::Plane => "plane",
        CraterSurface::Sphere => "sphere",
    }
}

pub const CRATER_OUTPUTS: &[CraterOutput] = &[CraterOutput::Height, CraterOutput::Ejecta];

pub fn crater_output(o: CraterOutput) -> &'static str {
    match o {
        CraterOutput::Height => "height",
        CraterOutput::Ejecta => "ejecta",
    }
}
