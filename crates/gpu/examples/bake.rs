//! Bake a texture graph to a PPM image, headlessly.
//!
//! ```text
//! cargo run -p texture-graph-gpu --example bake -- out.ppm [graph.tg] [size]
//! ```
//!
//! This is the whole headless path, and it exists as much to *hold* that path
//! open as to be useful: an example is compiled by `cargo test`, so if the
//! crate ever grows a dependency on the editor — or if baking stops being
//! reachable without one — this stops building. `texture-graph-gpu` pulls in
//! no egui, eframe or winit, and nothing here is allowed to change that.
//!
//! PPM because it needs no dependency at all: a short ASCII header and raw
//! RGB bytes. `magick out.ppm out.png` if you want something smaller.

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

    // Nothing here has a window, a surface or an event loop — just an adapter
    // and a queue. `pollster` only turns the async init into a blocking call;
    // a caller with its own runtime would await it instead.
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

/// Something with visible structure, so a run with no arguments still shows
/// whether the pipeline is working: noise through a two-stop ramp.
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
    // The ramp reads the noise as its own domain via a Map: value in, palette
    // across. `Map` is what turns a scalar field into colour.
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
