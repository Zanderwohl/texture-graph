//! Bake a texture graph to a PPM image, headlessly.
//!
//! ```text
//! cargo run -p texture-graph-gpu --example bake -- out.ppm [graph.tg] [size]
//! ```
//!
//! The whole headless path, and what holds it open: `cargo test` compiles
//! examples, so this stops building if the crate ever grows a dependency on
//! the editor. `texture-graph-gpu` pulls in no egui, eframe or winit.
//!
//! PPM needs no dependency to write. `magick out.ppm out.png` to shrink it.

use std::path::PathBuf;

use texture_graph_core::{
    BlendSpace, ColorInput, ColorRamp, ColorStop, EvalCtx, Graph, LayerKind, Noise, NoiseDims,
    NoiseOutput, NoiseRange, Output, ScalarInput, color::oklcha, load_from_path,
};
use texture_graph_gpu::{Baker, DeviceCtx, readback};

fn main() {
    let mut args = std::env::args().skip(1);
    let out_path = PathBuf::from(args.next().unwrap_or_else(|| {
        eprintln!("usage: bake <out.ppm> [graph.tg] [size]");
        std::process::exit(2);
    }));
    let graph_path = args.next().map(PathBuf::from);
    let size: u32 = args
        .next()
        .map(|s| s.parse().expect("size must be a whole number of pixels"))
        .unwrap_or(512);

    let graph = match &graph_path {
        Some(path) => load_from_path(path).expect("load graph").graph,
        None => demo_graph(),
    };

    // No window, surface or event loop: just an adapter and a queue.
    // `pollster` only blocks on the init; a caller with a runtime awaits it.
    let ctx = pollster::block_on(DeviceCtx::request_headless()).expect("no usable GPU adapter");
    let mut baker = Baker::new(ctx.clone());
    let baked = baker
        .bake_output(&graph, (size, size), &EvalCtx::default(), false)
        .expect("bake");

    let image = readback::read_rgba8(&ctx, &baked.color, baked.size);
    std::fs::write(&out_path, image.to_ppm()).expect("write ppm");

    let what = graph_path
        .as_deref()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| "built-in demo graph".to_string());
    println!("baked {what} at {size}x{size} -> {}", out_path.display());
}

/// Noise through a two-stop ramp, so a run with no arguments still shows
/// whether the pipeline works.
fn demo_graph() -> Graph {
    let mut g = Graph::new();
    let noise = g
        .add_layer(
            "noise",
            LayerKind::Noise(Noise {
                dims: NoiseDims::D2,
                seed_offset: 0,
                frequency: 6.0,
                range: NoiseRange::Unsigned,
                output: NoiseOutput::Grayscale,
                ..Noise::default()
            }),
        )
        .unwrap();
    let ramp = g
        .add_layer(
            "ramp",
            LayerKind::ColorRamp(ColorRamp {
                stops: vec![
                    ColorStop { t: 0.0, color: ColorInput::Const(oklcha(0.15, 0.06, 250.0, 1.0)) },
                    ColorStop { t: 1.0, color: ColorInput::Const(oklcha(0.92, 0.10, 80.0, 1.0)) },
                ],
                space: BlendSpace::Oklch,
            }),
        )
        .unwrap();
    // `Map` turns the scalar noise into colour: value in, palette across.
    let mapped = g
        .add_layer(
            "mapped",
            LayerKind::Map(texture_graph_core::Map { value: Some(noise), palette: Some(ramp) }),
        )
        .unwrap();
    g.set_output(Output {
        color: Some(mapped),
        roughness: ScalarInput::Const(0.5),
        metallic: ScalarInput::Const(0.0),
        normal: None,
    })
    .unwrap();
    g
}
