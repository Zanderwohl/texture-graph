//! End-to-end bake tests. Every variant should have one here comparing the
//! GPU output against `core`'s CPU evaluator.

use std::collections::HashSet;

use texture_graph_core::{
    BlendMode, BlendSpace, Color, ColorInput, ColorRamp, ColorStop, CoordMode, Criterion,
    EdgeMode, EvalCtx, Fractal, FractalMode, Graph, HeightToNormal, LayerId, LayerKind, Map,
    MinMax, MinMaxMode, Mix, Noise, NoiseDims, NoiseKernel, NoiseOutput, NoiseRange, Output,
    ScalarInput, Transform, Wave, WaveShape, color::to_srgb8,
};

use crate::{Baker, DeviceCtx, ScalarFormat};

const SIZE: (u32, u32) = (8, 8);

fn add_mix(g: &mut Graph, name: &str, a: LayerId, b: LayerId) -> LayerId {
    g.add_layer(
        name,
        LayerKind::Mix(Mix {
            a: Some(a),
            b: Some(b),
            mode: BlendMode::Add,
            factor: ScalarInput::Const(0.5),
            space: BlendSpace::Oklch,
        }),
    )
    .unwrap()
}

fn add_transform(
    g: &mut Graph,
    name: &str,
    src: LayerId,
    scale: [f32; 3],
    edge_mode: EdgeMode,
) -> LayerId {
    g.add_layer(
        name,
        LayerKind::Transform(Transform {
            source: Some(src),
            offset: [0.0; 3],
            rotate_uv: 0.0,
            scale,
            coord_mode: CoordMode::Passthrough,
            edge_mode,
        }),
    )
    .unwrap()
}

/// Every pixel of an Rgba8Unorm texture, as `[r, g, b, a]`.
///
/// Wraps the public [`crate::readback::read_rgba8`], so these tests exercise
/// the path a headless caller takes and there is no second copy of the
/// row-padding arithmetic to get wrong.
fn readback_all_pixels(ctx: &DeviceCtx, tex: &wgpu::Texture, size: (u32, u32)) -> Vec<[u8; 4]> {
    crate::readback::read_rgba8(ctx, tex, size)
        .pixels
        .as_chunks::<4>()
        .0
        .iter()
        .map(|p| [p[0], p[1], p[2], p[3]])
        .collect()
}

fn readback_first_pixel(ctx: &DeviceCtx, tex: &wgpu::Texture) -> [u8; 4] {
    let bytes_per_row = 256u32;
    let readback = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (bytes_per_row * SIZE.1) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("readback-enc"),
    });
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(SIZE.1),
            },
        },
        wgpu::Extent3d { width: SIZE.0, height: SIZE.1, depth_or_array_layers: 1 },
    );
    ctx.queue.submit([enc.finish()]);

    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
    ctx.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    rx.recv().expect("chan").expect("map");
    let data = slice.get_mapped_range();
    let px = [data[0], data[1], data[2], data[3]];
    drop(data);
    readback.unmap();
    px
}

#[test]
fn color_layer_matches_cpu_to_srgb8() {
    let ctx = pollster::block_on(DeviceCtx::request_headless()).expect("headless");
    let mut baker = Baker::new(ctx.clone());

    let mut graph = Graph::new();
    // Overwrite the auto-created base color's kind with a specific value.
    let base = graph.output.color;
    graph
        .set_kind(base.unwrap(), LayerKind::Color(Color::new(0.6, 0.15, 40.0, 1.0)))
        .unwrap();
    graph
        .set_output(Output {
            color: base,
            roughness: ScalarInput::Const(0.5),
            metallic: ScalarInput::Const(0.0),
            normal: None,
        })
        .unwrap();

    let out = baker
        .bake_output(&graph, SIZE, &EvalCtx::default(), false)
        .expect("bake");

    let cpu = to_srgb8(Color::new(0.6, 0.15, 40.0, 1.0));
    let gpu = readback_first_pixel(&ctx, &out.color);
    for i in 0..4 {
        let d = (cpu[i] as i32 - gpu[i] as i32).abs();
        assert!(
            d <= 1,
            "channel {i}: cpu {} gpu {} (delta {d})",
            cpu[i], gpu[i],
        );
    }

    // Roughness const: uniform gray at sRGB(0.5).
    let rough_expected_lin = 0.5f32;
    let rough_expected_srgb = if rough_expected_lin <= 0.0031308 {
        12.92 * rough_expected_lin
    } else {
        1.055 * rough_expected_lin.powf(1.0 / 2.4) - 0.055
    };
    let rough_expected = (rough_expected_srgb * 255.0 + 0.5) as u8;
    let rough = readback_first_pixel(&ctx, &out.roughness);
    for i in 0..3 {
        let d = (rough_expected as i32 - rough[i] as i32).abs();
        assert!(d <= 1, "rough channel {i}: expect {rough_expected} got {} (delta {d})", rough[i]);
    }

    // Metallic const 0.0 → sRGB(0.0) = 0.
    let metal = readback_first_pixel(&ctx, &out.metallic);
    assert!(metal[0] <= 1 && metal[1] <= 1 && metal[2] <= 1);

    // Absent normal → flat blue [128, 128, 255, 255].
    let normal = readback_first_pixel(&ctx, &out.normal);
    for (i, want) in [128u8, 128, 255, 255].iter().enumerate() {
        let d = (*want as i32 - normal[i] as i32).abs();
        assert!(d <= 1, "normal channel {i}: expect {want} got {} (delta {d})", normal[i]);
    }
}

#[test]
fn noise_layer_has_variance_and_is_deterministic() {
    let ctx = pollster::block_on(DeviceCtx::request_headless()).expect("headless");
    let mut baker = Baker::new(ctx.clone());
    let size = (32u32, 32u32);

    let mut graph = Graph::new();
    let base = graph.output.color;
    graph
        .set_kind(
            base.unwrap(),
            LayerKind::Noise(Noise {
                dims: NoiseDims::D2,
                seed_offset: 0,
                frequency: 4.0,
                range: NoiseRange::Unsigned,
                output: NoiseOutput::Grayscale,
                ..Noise::default()
            }),
        )
        .unwrap();

    let out = baker
        .bake_output(&graph, size, &EvalCtx::default(), false)
        .expect("bake noise");
    let px = readback_all_pixels(&ctx, &out.color, size);

    let (mut min, mut max) = (255u8, 0u8);
    for p in &px {
        min = min.min(p[0]);
        max = max.max(p[0]);
    }
    assert!(max as i32 - min as i32 > 40, "noise output has too little variance (min={min} max={max})");

    // Determinism: baking again with an identical graph must produce identical pixels.
    let out2 = baker
        .bake_output(&graph, size, &EvalCtx::default(), false)
        .expect("bake noise 2");
    let px2 = readback_all_pixels(&ctx, &out2.color, size);
    assert_eq!(px, px2, "noise output changed across identical bakes");
}

#[test]
fn mix_add_of_two_colors_matches_cpu() {
    let ctx = pollster::block_on(DeviceCtx::request_headless()).expect("headless");
    let mut baker = Baker::new(ctx.clone());
    let mut graph = Graph::new();
    let a = graph.output.color;
    graph.set_kind(a.unwrap(), LayerKind::Color(Color::new(0.4, 0.0, 0.0, 1.0))).unwrap();
    let b = graph.add_layer("b", LayerKind::Color(Color::new(0.3, 0.05, 30.0, 1.0))).unwrap();
    let mix = graph
        .add_layer(
            "mix",
            LayerKind::Mix(Mix {
                a: Some(a.unwrap()),
                b: Some(b),
                mode: BlendMode::Add,
                factor: ScalarInput::Const(0.5),
                space: BlendSpace::Oklch,
            }),
        )
        .unwrap();
    graph.set_output(Output {
        color: Some(mix),
        roughness: ScalarInput::Const(0.5),
        metallic: ScalarInput::Const(0.0),
        normal: None,
    }).unwrap();

    let out = baker.bake_output(&graph, SIZE, &EvalCtx::default(), false).expect("bake");
    let gpu = readback_first_pixel(&ctx, &out.color);

    // CPU reference — evaluate the mix directly.
    let cpu_material = texture_graph_core::evaluate_material(
        &graph, texture_graph_core::Sample::uv(0.5, 0.5), &EvalCtx::default(),
    );
    let cpu = to_srgb8(cpu_material.color);
    for i in 0..4 {
        let d = (cpu[i] as i32 - gpu[i] as i32).abs();
        assert!(d <= 2, "mix add channel {i}: cpu {} gpu {} (delta {d})", cpu[i], gpu[i]);
    }
}

#[test]
fn ramp_black_to_white_at_midpoint_is_gray() {
    let ctx = pollster::block_on(DeviceCtx::request_headless()).expect("headless");
    let mut baker = Baker::new(ctx.clone());
    let mut graph = Graph::new();
    let ramp = graph
        .add_layer(
            "ramp",
            LayerKind::ColorRamp(ColorRamp {
                stops: vec![
                    ColorStop { t: 0.0, color: ColorInput::Const(Color::new(0.0, 0.0, 0.0, 1.0)) },
                    ColorStop { t: 1.0, color: ColorInput::Const(Color::new(1.0, 0.0, 0.0, 1.0)) },
                ],
                space: BlendSpace::Oklch,
            }),
        )
        .unwrap();
    graph.set_output(Output {
        color: Some(ramp),
        roughness: ScalarInput::Const(0.5),
        metallic: ScalarInput::Const(0.0),
        normal: None,
    }).unwrap();

    let out = baker.bake_output(&graph, SIZE, &EvalCtx::default(), false).expect("bake ramp");
    let px = readback_all_pixels(&ctx, &out.color, SIZE);
    // Middle column should be roughly the sRGB encoding of Oklch L=0.5 gray.
    // Pixel-center convention: u = (x + 0.5) / 8. The 4th column is u = 0.5625.
    let mid_col = 4;
    let mid_pixel = px[(SIZE.1 / 2) as usize * SIZE.0 as usize + mid_col];
    let expected = to_srgb8(Color::new(0.5625, 0.0, 0.0, 1.0));
    for i in 0..3 {
        let d = (expected[i] as i32 - mid_pixel[i] as i32).abs();
        assert!(d <= 3, "ramp mid pixel channel {i}: expect {} got {} (delta {d})", expected[i], mid_pixel[i]);
    }
    // Left edge dark, right edge light.
    let left = px[0][0];
    let right = px[SIZE.0 as usize - 1][0];
    assert!(left < right, "ramp left {left} not < right {right}");
}

#[test]
fn ramp_with_layer_ref_stops_reads_from_source_layers() {
    // Ramp with two stops, both layer-refs. Stop 0 points at "black" layer,
    // stop 1 points at "white" layer. Baked into a 16-wide texture — middle
    // column should read as gray L≈0.5.
    let ctx = pollster::block_on(DeviceCtx::request_headless()).expect("headless");
    let mut baker = Baker::new(ctx.clone());
    let mut graph = Graph::new();
    let black = graph.output.color;
    graph.set_kind(black.unwrap(), LayerKind::Color(Color::new(0.0, 0.0, 0.0, 1.0))).unwrap();
    let white = graph
        .add_layer("white", LayerKind::Color(Color::new(1.0, 0.0, 0.0, 1.0)))
        .unwrap();
    let ramp = graph
        .add_layer(
            "ramp",
            LayerKind::ColorRamp(ColorRamp {
                stops: vec![
                    ColorStop { t: 0.0, color: ColorInput::Layer(black.unwrap()) },
                    ColorStop { t: 1.0, color: ColorInput::Layer(white) },
                ],
                space: BlendSpace::Oklch,
            }),
        )
        .unwrap();
    graph.set_output(Output {
        color: Some(ramp),
        roughness: ScalarInput::Const(0.5),
        metallic: ScalarInput::Const(0.0),
        normal: None,
    }).unwrap();

    let out = baker.bake_output(&graph, (16, 16), &EvalCtx::default(), false).expect("bake");
    let px = readback_all_pixels(&ctx, &out.color, (16, 16));
    // Column 8 is u = 8.5/16 = 0.53125 → L ≈ 0.53125.
    let mid_pixel = px[16 * 8 + 8];
    let expected = to_srgb8(Color::new(0.53125, 0.0, 0.0, 1.0));
    for i in 0..3 {
        let d = (expected[i] as i32 - mid_pixel[i] as i32).abs();
        assert!(d <= 3, "layer-ref ramp mid channel {i}: expect {} got {} (delta {d})", expected[i], mid_pixel[i]);
    }
    // Left edge should read the black layer (~0), right edge the white (~255).
    let left = px[16 * 8][0];
    let right = px[16 * 8 + 15][0];
    assert!(left < 30, "left edge should read black layer, got {left}");
    assert!(right > 220, "right edge should read white layer, got {right}");
}

#[test]
fn ramp_rejects_more_than_eight_unique_layer_stops() {
    let ctx = pollster::block_on(DeviceCtx::request_headless()).expect("headless");
    let mut baker = Baker::new(ctx.clone());
    let mut graph = Graph::new();
    // Auto-created base color layer + 8 more colors = 9 distinct layers.
    let mut layer_ids = vec![graph.output.color];
    for i in 0..8u32 {
        layer_ids.push(
            Some(graph
                .add_layer(
                    &format!("c{i}"),
                    LayerKind::Color(Color::new(0.1 * i as f32, 0.0, 0.0, 1.0)),
                )
                .unwrap()),
        );
    }
    let stops: Vec<_> = layer_ids
        .iter()
        .enumerate()
        .map(|(i, &id)| ColorStop {
            t: i as f32 / 8.0,
            color: ColorInput::Layer(id.unwrap()),
        })
        .collect();
    let ramp = graph
        .add_layer(
            "ramp",
            LayerKind::ColorRamp(ColorRamp { stops, space: BlendSpace::Oklch }),
        )
        .unwrap();
    graph.set_output(Output {
        color: Some(ramp),
        roughness: ScalarInput::Const(0.5),
        metallic: ScalarInput::Const(0.0),
        normal: None,
    }).unwrap();
    match baker.bake_output(&graph, (8, 8), &EvalCtx::default(), false) {
        Ok(_) => panic!("expected 9-layer ramp to be rejected"),
        Err(crate::BakeError::Unsupported(msg)) => assert!(msg.contains(">8")),
        Err(other) => panic!("unexpected error: {other}"),
    }
}

#[test]
fn transform_passthrough_of_color_is_identity() {
    let ctx = pollster::block_on(DeviceCtx::request_headless()).expect("headless");
    let mut baker = Baker::new(ctx.clone());
    let mut graph = Graph::new();
    let c = graph.output.color;
    graph.set_kind(c.unwrap(), LayerKind::Color(Color::new(0.5, 0.1, 20.0, 1.0))).unwrap();
    let t = graph.add_layer(
        "t",
        LayerKind::Transform(Transform {
            source: Some(c.unwrap()),
            offset: [0.0, 0.0, 0.0],
            rotate_uv: 0.0,
            scale: [1.0, 1.0, 1.0],
            coord_mode: CoordMode::Passthrough,
                edge_mode: EdgeMode::default(),
        }),
    ).unwrap();
    graph.set_output(Output {
        color: Some(t),
        roughness: ScalarInput::Const(0.5),
        metallic: ScalarInput::Const(0.0),
        normal: None,
    }).unwrap();
    let out = baker.bake_output(&graph, SIZE, &EvalCtx::default(), false).expect("bake xform");
    let px = readback_first_pixel(&ctx, &out.color);
    let expected = to_srgb8(Color::new(0.5, 0.1, 20.0, 1.0));
    for i in 0..4 {
        let d = (expected[i] as i32 - px[i] as i32).abs();
        assert!(d <= 1, "xform identity channel {i}: expect {} got {} (delta {d})", expected[i], px[i]);
    }
}

#[test]
fn map_gray_value_through_bw_ramp_matches_ramp_lookup() {
    let ctx = pollster::block_on(DeviceCtx::request_headless()).expect("headless");
    let mut baker = Baker::new(ctx.clone());
    let mut graph = Graph::new();
    // value layer: uniform L=0.5.
    let value = graph.output.color;
    graph.set_kind(value.unwrap(), LayerKind::Color(Color::new(0.5, 0.0, 0.0, 1.0))).unwrap();
    // palette: black-to-white ramp.
    let palette = graph.add_layer(
        "palette",
        LayerKind::ColorRamp(ColorRamp {
            stops: vec![
                ColorStop { t: 0.0, color: ColorInput::Const(Color::new(0.0, 0.0, 0.0, 1.0)) },
                ColorStop { t: 1.0, color: ColorInput::Const(Color::new(1.0, 0.0, 0.0, 1.0)) },
            ],
            space: BlendSpace::Oklch,
        }),
    ).unwrap();
    let map = graph.add_layer(
        "map",
        LayerKind::Map(Map { value: Some(value.unwrap()), palette: Some(palette) }),
    ).unwrap();
    graph.set_output(Output {
        color: Some(map),
        roughness: ScalarInput::Const(0.5),
        metallic: ScalarInput::Const(0.0),
        normal: None,
    }).unwrap();
    let out = baker.bake_output(&graph, (16, 16), &EvalCtx::default(), false).expect("bake map");
    let px = readback_first_pixel(&ctx, &out.color);
    // Map samples palette at t=L(value)=0.5. The 16-wide palette baked with
    // pixel-center convention: nearest column is 8 → u=(8+0.5)/16=0.53125,
    // ramp gives L=0.53125.
    let expected = to_srgb8(Color::new(0.53125, 0.0, 0.0, 1.0));
    for i in 0..3 {
        let d = (expected[i] as i32 - px[i] as i32).abs();
        assert!(d <= 3, "map channel {i}: expect {} got {} (delta {d})", expected[i], px[i]);
    }
}

/// Timing observation, not an assertion: five 1024² bakes of a 15-octave
/// fractal stack, reporting median record-and-submit latency and the
/// scheduler's peak_slots. Measuring GPU wall time would need timestamp
/// queries.
///
/// `cargo test -p texture-graph-gpu perf_fractal -- --nocapture`
#[test]
fn perf_fractal_stack_1024() {
    let ctx = pollster::block_on(DeviceCtx::request_headless()).expect("headless");
    let mut baker = Baker::new(ctx.clone());

    let mut graph = Graph::new();
    let base = graph.output.color;
    graph
        .set_kind(
            base.unwrap(),
            LayerKind::Noise(Noise {
                dims: NoiseDims::D2,
                seed_offset: 0,
                frequency: 1.0,
                range: NoiseRange::Signed,
                output: NoiseOutput::Grayscale,
                ..Noise::default()
            }),
        )
        .unwrap();

    let mut cur = base;
    for octave in 1..15 {
        let freq = (1u32 << octave) as f32;
        let n = graph
            .add_layer(
                &format!("noise-{octave}"),
                LayerKind::Noise(Noise {
                    dims: NoiseDims::D2,
                    seed_offset: octave as u32,
                    frequency: freq,
                    range: NoiseRange::Signed,
                    output: NoiseOutput::Grayscale,
                    ..Noise::default()
                }),
            )
            .unwrap();
        cur = Some(graph
            .add_layer(
                &format!("sum-{octave}"),
                LayerKind::Mix(Mix {
                    a: Some(cur.unwrap()),
                    b: Some(n),
                    mode: BlendMode::Add,
                    factor: ScalarInput::Const(0.5),
                    space: BlendSpace::Oklch,
                }),
            )
            .unwrap());
    }
    graph
        .set_output(Output {
            color: cur,
            roughness: ScalarInput::Const(0.5),
            metallic: ScalarInput::Const(0.0),
            normal: None,
        })
        .unwrap();

    // Warmup — first bake compiles pipelines on some drivers.
    let _ = baker
        .bake_output(&graph, (1024, 1024), &EvalCtx::default(), false)
        .expect("warmup bake");

    let mut times: Vec<std::time::Duration> = Vec::new();
    for _ in 0..5 {
        let t0 = std::time::Instant::now();
        let _ = baker
            .bake_output(&graph, (1024, 1024), &EvalCtx::default(), false)
            .expect("bake");
        // Measure through GPU completion, not just command recording.
        ctx.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        times.push(t0.elapsed());
    }
    times.sort();
    let median = times[times.len() / 2];
    let layers = graph.layers.len();
    eprintln!("perf_fractal_stack_1024: layers={} median={:?}", layers, median);
    // Not asserted — human reads the number.
}

#[test]
fn min_max_by_luma_picks_brighter_pixel_whole() {
    // Two Color layers: (L=0.2, C=0, h=0) and (L=0.8, C=0.15, h=200).
    // Max by Luma should propagate the bright layer's pixel — including its
    // chroma and hue, not just its L.
    let ctx = pollster::block_on(DeviceCtx::request_headless()).expect("headless");
    let mut baker = Baker::new(ctx.clone());
    let mut graph = Graph::new();
    let dark = graph.output.color;
    graph.set_kind(dark.unwrap(), LayerKind::Color(Color::new(0.2, 0.0, 0.0, 1.0))).unwrap();
    let bright = graph
        .add_layer("bright", LayerKind::Color(Color::new(0.8, 0.15, 200.0, 1.0)))
        .unwrap();
    let mm = graph
        .add_layer(
            "mm",
            LayerKind::MinMax(MinMax {
                a: Some(dark.unwrap()),
                b: Some(bright),
                mode: MinMaxMode::Max,
                criterion: Criterion::Luma,
            }),
        )
        .unwrap();
    graph.set_output(Output {
        color: Some(mm),
        roughness: ScalarInput::Const(0.5),
        metallic: ScalarInput::Const(0.0),
        normal: None,
    }).unwrap();
    let out = baker.bake_output(&graph, SIZE, &EvalCtx::default(), false).expect("bake");
    let gpu = readback_first_pixel(&ctx, &out.color);
    // Winner is the bright layer — compare against its packed sRGB.
    let expect = to_srgb8(Color::new(0.8, 0.15, 200.0, 1.0));
    for i in 0..4 {
        let d = (expect[i] as i32 - gpu[i] as i32).abs();
        assert!(d <= 2, "MinMax(Max, Luma) channel {i}: expect {} got {} (delta {d})", expect[i], gpu[i]);
    }
}

#[test]
fn min_max_matches_cpu_over_all_criteria() {
    // GPU against CPU per criterion, on a fixed pair of opaque colors, so
    // drift in either `criterion_of` shows up. Alpha needs its own test:
    // non-1.0 alpha would trip the display-side gray checker compositing
    // and make packed-pixel equality meaningless.
    let ctx = pollster::block_on(DeviceCtx::request_headless()).expect("headless");
    let mut baker = Baker::new(ctx.clone());
    let a_col = Color::new(0.3, 0.2, 100.0, 1.0);
    let b_col = Color::new(0.7, 0.05, 20.0, 1.0);
    let criteria = [
        Criterion::Red,
        Criterion::Green,
        Criterion::Blue,
        Criterion::Saturation,
        Criterion::Value,
        Criterion::Luma,
        Criterion::Chroma,
    ];
    for &crit in &criteria {
        for &mode in &[MinMaxMode::Min, MinMaxMode::Max] {
            let mut graph = Graph::new();
            let a = graph.output.color;
            graph.set_kind(a.unwrap(), LayerKind::Color(a_col)).unwrap();
            let b = graph.add_layer("b", LayerKind::Color(b_col)).unwrap();
            let mm = graph
                .add_layer(
                    "mm",
                    LayerKind::MinMax(MinMax { a: Some(a.unwrap()), b: Some(b), mode, criterion: crit }),
                )
                .unwrap();
            graph.set_output(Output {
                color: Some(mm),
                roughness: ScalarInput::Const(0.5),
                metallic: ScalarInput::Const(0.0),
                normal: None,
            }).unwrap();
            let out = baker.bake_output(&graph, SIZE, &EvalCtx::default(), false).expect("bake");
            let gpu = readback_first_pixel(&ctx, &out.color);
            let cpu_material = texture_graph_core::evaluate_material(
                &graph,
                texture_graph_core::Sample::uv(0.0, 0.0),
                &EvalCtx::default(),
            );
            let cpu = to_srgb8(cpu_material.color);
            for i in 0..3 {
                let d = (cpu[i] as i32 - gpu[i] as i32).abs();
                assert!(
                    d <= 3,
                    "criterion={crit:?} mode={mode:?} channel {i}: cpu {} gpu {} (delta {d})",
                    cpu[i], gpu[i],
                );
            }
        }
    }
}

#[test]
fn min_max_by_alpha_picks_correct_layer() {
    // Distinct L, distinct alpha; both opaque enough that the alpha checker
    // barely nudges the display value. Test asserts the *winning L* — using
    // the higher-alpha layer's L when Max, lower-alpha layer's L when Min.
    let ctx = pollster::block_on(DeviceCtx::request_headless()).expect("headless");
    let mut baker = Baker::new(ctx.clone());
    let low = Color::new(0.2, 0.0, 0.0, 0.98);
    let high = Color::new(0.8, 0.0, 0.0, 1.0);
    for (mode, expected_l) in [(MinMaxMode::Max, 0.8), (MinMaxMode::Min, 0.2)] {
        let mut graph = Graph::new();
        let a = graph.output.color;
        graph.set_kind(a.unwrap(), LayerKind::Color(low)).unwrap();
        let b = graph.add_layer(&format!("b-{mode:?}"), LayerKind::Color(high)).unwrap();
        let mm = graph
            .add_layer(
                &format!("mm-{mode:?}"),
                LayerKind::MinMax(MinMax {
                    a: Some(a.unwrap()), b: Some(b), mode,
                    criterion: Criterion::Alpha,
                }),
            )
            .unwrap();
        graph.set_output(Output {
            color: Some(mm),
            roughness: ScalarInput::Const(0.5),
            metallic: ScalarInput::Const(0.0),
            normal: None,
        }).unwrap();
        let out = baker.bake_output(&graph, SIZE, &EvalCtx::default(), false).expect("bake");
        let gpu = readback_first_pixel(&ctx, &out.color);
        // Winning color's rough sRGB grayscale target (allow slack because
        // low.alpha=0.98 lets a hint of gray leak in).
        let expect = to_srgb8(Color::new(expected_l, 0.0, 0.0, 1.0));
        let d = (expect[0] as i32 - gpu[0] as i32).abs();
        assert!(d <= 5, "MinMax(Alpha, {mode:?}) expect~{} got {} (delta {d})", expect[0], gpu[0]);
    }
}

#[test]
fn alpha_lt_one_shows_gray_checker_backing() {
    // Color layer with alpha = 0.5. Every pixel should be a blend between
    // the color's sRGB and one of two gray checker cells — never the pure
    // solid color and never the magenta out-of-range checker.
    let ctx = pollster::block_on(DeviceCtx::request_headless()).expect("headless");
    let mut baker = Baker::new(ctx.clone());
    let mut graph = Graph::new();
    let base = graph.output.color;
    graph.set_kind(base.unwrap(), LayerKind::Color(Color::new(0.5, 0.0, 0.0, 0.5))).unwrap();
    graph.set_output(Output {
        color: base,
        roughness: ScalarInput::Const(0.5),
        metallic: ScalarInput::Const(0.0),
        normal: None,
    }).unwrap();
    let size = (24u32, 24u32);
    let out = baker.bake_output(&graph, size, &EvalCtx::default(), false).expect("bake");
    let px = readback_all_pixels(&ctx, &out.color, size);
    // Solid opaque sRGB of L=0.5 gray:
    let solid = to_srgb8(Color::new(0.5, 0.0, 0.0, 1.0))[0];
    // Never see the magenta out-of-range marker.
    for p in &px {
        assert!(
            !(p[0] == 255 && p[1] == 0 && p[2] == 255),
            "alpha < 1 should not trip the out-of-range checker: {:?}",
            p,
        );
        // Alpha channel always fully opaque after compositing.
        assert_eq!(p[3], 255, "output alpha should be 1 after alpha compositing");
    }
    // At least two distinct pixel values appear — the two checker cells
    // blend to different results.
    let unique: std::collections::HashSet<u8> = px.iter().map(|p| p[0]).collect();
    assert!(unique.len() >= 2, "expected multiple values from checker blend, got {unique:?}");
    // Every displayed value falls between the blended-with-lightgray and
    // blended-with-darkgray endpoints.
    let light_end = ((solid as f32) * 0.5 + 0.75 * 255.0 * 0.5).round() as i32;
    let dark_end  = ((solid as f32) * 0.5 + 0.55 * 255.0 * 0.5).round() as i32;
    let lo = light_end.min(dark_end) - 4;
    let hi = light_end.max(dark_end) + 4;
    for p in &px {
        let v = p[0] as i32;
        assert!(
            v >= lo && v <= hi,
            "pixel red channel {v} outside expected checker range [{lo}, {hi}]",
        );
    }
}

#[test]
fn out_of_range_l_paints_magenta_black_checker() {
    // Color layer with L = -0.5 (well outside [0,1]). Every pixel is out of
    // range, so every cell should be one of the two checker colors: bright
    // magenta (255, 0, 255) or black (0, 0, 0). No other colors allowed.
    let ctx = pollster::block_on(DeviceCtx::request_headless()).expect("headless");
    let mut baker = Baker::new(ctx.clone());
    let mut graph = Graph::new();
    let base = graph.output.color;
    graph
        .set_kind(base.unwrap(), LayerKind::Color(Color::new(-0.5, 0.0, 0.0, 1.0)))
        .unwrap();
    graph.set_output(Output {
        color: base,
        roughness: ScalarInput::Const(0.5),
        metallic: ScalarInput::Const(0.0),
        normal: None,
    }).unwrap();
    let size = (40u32, 40u32);
    let out = baker.bake_output(&graph, size, &EvalCtx::default(), false).expect("bake");
    let px = readback_all_pixels(&ctx, &out.color, size);
    // Every pixel is either magenta or black.
    let mut magentas = 0usize;
    let mut blacks = 0usize;
    for p in &px {
        let is_magenta = p[0] == 255 && p[1] == 0 && p[2] == 255;
        let is_black = p[0] == 0 && p[1] == 0 && p[2] == 0;
        assert!(
            is_magenta || is_black,
            "unexpected non-checker pixel {:?}",
            p,
        );
        if is_magenta { magentas += 1; }
        if is_black { blacks += 1; }
    }
    // Both colors should be present.
    assert!(magentas > 0, "no magenta cells");
    assert!(blacks > 0, "no black cells");
    // 10-px cells on a 40-px image → 4x4 = 16 cells; roughly balanced.
    assert!(magentas > 500, "too few magenta pixels: {magentas}");
    assert!(blacks > 500, "too few black pixels: {blacks}");
}

#[test]
fn in_range_color_does_not_trigger_checker() {
    // Regression: a normal in-range color must not accidentally show the
    // checker. Uses the same L=0.6 the color_layer test does.
    let ctx = pollster::block_on(DeviceCtx::request_headless()).expect("headless");
    let mut baker = Baker::new(ctx.clone());
    let mut graph = Graph::new();
    let base = graph.output.color;
    graph
        .set_kind(base.unwrap(), LayerKind::Color(Color::new(0.6, 0.15, 40.0, 1.0)))
        .unwrap();
    graph.set_output(Output {
        color: base,
        roughness: ScalarInput::Const(0.5),
        metallic: ScalarInput::Const(0.0),
        normal: None,
    }).unwrap();
    let out = baker.bake_output(&graph, (16, 16), &EvalCtx::default(), false).expect("bake");
    let px = readback_all_pixels(&ctx, &out.color, (16, 16));
    // No pixel should be exactly bright magenta.
    for p in &px {
        assert!(
            !(p[0] == 255 && p[1] == 0 && p[2] == 255),
            "in-range color unexpectedly hit the checker: {:?}",
            p,
        );
    }
}

#[test]
fn bake_previews_returns_one_texture_per_authored_layer() {
    // Includes unreachable layers — the no-reuse schedule adds every authored
    // layer, not just those wired to Output.
    let ctx = pollster::block_on(DeviceCtx::request_headless()).expect("headless");
    let mut baker = Baker::new(ctx.clone());
    let mut graph = Graph::new();
    let base = graph.output.color;
    graph.set_kind(base.unwrap(), LayerKind::Color(Color::new(0.5, 0.0, 0.0, 1.0))).unwrap();
    let _b = graph.add_layer("b", LayerKind::Color(Color::new(0.3, 0.1, 60.0, 1.0))).unwrap();
    let _unreachable = graph
        .add_layer("dead", LayerKind::Color(Color::new(0.9, 0.05, 200.0, 1.0)))
        .unwrap();
    let previews =
        baker.bake_previews(&graph, &EvalCtx::default(), None).expect("bake previews");
    assert_eq!(previews.len(), 3, "expected one preview per authored layer");
    // Sanity: each output is 128² Rgba8Unorm.
    for (_, tex) in &previews {
        assert_eq!(tex.width(), 128);
        assert_eq!(tex.height(), 128);
        assert_eq!(tex.format(), wgpu::TextureFormat::Rgba8Unorm);
    }
}

/// Asking for a subset must hand back exactly that subset. A layer that
/// came along only because something wanted reads it is dispatched, but it
/// is not packed and not returned — the caller is still showing the texture
/// it already had for that one, and handing it a second would leak the
/// first.
#[test]
fn bake_previews_returns_only_the_layers_that_were_asked_for() {
    let ctx = pollster::block_on(DeviceCtx::request_headless()).expect("headless");
    let mut baker = Baker::new(ctx.clone());
    let mut graph = Graph::new();
    let base = graph.output.color.unwrap();
    graph.set_kind(base, LayerKind::Color(Color::new(0.5, 0.0, 0.0, 1.0))).unwrap();
    let b = graph.add_layer("b", LayerKind::Color(Color::new(0.3, 0.1, 60.0, 1.0))).unwrap();
    // `mixed` reads both, so asking for it schedules them without packing
    // them.
    let mixed = add_mix(&mut graph, "mixed", base, b);
    let untouched = graph
        .add_layer("untouched", LayerKind::Color(Color::new(0.9, 0.05, 200.0, 1.0)))
        .unwrap();

    let wanted: HashSet<LayerId> = [mixed].into_iter().collect();
    let previews = baker
        .bake_previews(&graph, &EvalCtx::default(), Some(&wanted))
        .expect("bake previews");
    assert_eq!(previews.keys().copied().collect::<Vec<_>>(), vec![mixed]);
    assert!(!previews.contains_key(&untouched), "an unrelated layer was baked");
}

/// A thumbnail must be the same picture whichever way it was baked. The
/// trap is bake domains: they are decided by a layer's *consumers*, so a
/// subset schedule that only walked the subset would give a layer a
/// narrower domain whenever the consumer that widened it was clean, and
/// its thumbnail would come back at a different effective resolution
/// depending on what else happened to be stale.
#[test]
fn a_subset_bake_gives_the_same_picture_as_a_full_one() {
    use texture_graph_core::EdgeMode;
    let ctx = pollster::block_on(DeviceCtx::request_headless()).expect("headless");
    let mut baker = Baker::new(ctx.clone());
    let mut graph = Graph::new();
    let src = graph.output.color.unwrap();
    graph
        .set_kind(
            src,
            LayerKind::Noise(Noise {
                dims: NoiseDims::D2,
                seed_offset: 7,
                frequency: 8.0,
                range: NoiseRange::Unsigned,
                output: NoiseOutput::Grayscale,
                ..Noise::default()
            }),
        )
        .unwrap();
    // The extend transform is what widens `src`'s bake domain past the
    // unit square. It is not in `wanted` below.
    let _zoom = add_transform(&mut graph, "zoom", src, [0.25, 0.25, 1.0], EdgeMode::Extend);

    let full = baker.bake_previews(&graph, &EvalCtx::default(), None).expect("full");
    let wanted: HashSet<LayerId> = [src].into_iter().collect();
    let subset = baker
        .bake_previews(&graph, &EvalCtx::default(), Some(&wanted))
        .expect("subset");

    let a = readback_all_pixels(&ctx, &full[&src], (128, 128));
    let b = readback_all_pixels(&ctx, &subset[&src], (128, 128));
    assert_eq!(a, b, "the same layer baked two ways gave two different pictures");
}

#[test]
fn h2n_on_flat_source_gives_flat_normal() {
    // Uniform source → zero gradient → normal = (0, 0, 1) → sRGB (128, 128, 255).
    let ctx = pollster::block_on(DeviceCtx::request_headless()).expect("headless");
    let mut baker = Baker::new(ctx.clone());
    let mut graph = Graph::new();
    let source = graph.output.color;
    graph.set_kind(source.unwrap(), LayerKind::Color(Color::new(0.5, 0.0, 0.0, 1.0))).unwrap();
    let n = graph
        .add_layer("n", LayerKind::HeightToNormal(HeightToNormal { source: Some(source.unwrap()), strength: 1.0 }))
        .unwrap();
    graph.set_output(Output {
        color: graph.output.color,
        roughness: ScalarInput::Const(0.5),
        metallic: ScalarInput::Const(0.0),
        normal: Some(n),
    }).unwrap();
    let out = baker.bake_output(&graph, (16, 16), &EvalCtx::default(), false).expect("bake h2n");
    // Sample a pixel in the interior, not the edge — clamped neighbors at
    // the edge give a phantom gradient because the "outside" is a copy of
    // the edge and adjacent-inside pixels differ; on uniform input they
    // agree but the pixel-center convention still yields a clean interior.
    let px = readback_all_pixels(&ctx, &out.normal, (16, 16));
    let interior = px[8 * 16 + 8];
    for (i, want) in [128u8, 128, 255, 255].iter().enumerate() {
        let d = (*want as i32 - interior[i] as i32).abs();
        assert!(d <= 2, "flat-normal channel {i}: expect {want} got {} (delta {d})", interior[i]);
    }
}

#[test]
fn h2n_on_horizontal_ramp_tilts_normal_toward_negative_u() {
    // Horizontal ramp: L increases along u. Central-difference gradient is
    // positive in u, zero in v. Normal = (-slope, 0, 1)/norm, so x-channel
    // of the encoded sRGB should read *less than* 128 (nx < 0 → n*0.5+0.5 < 0.5).
    let ctx = pollster::block_on(DeviceCtx::request_headless()).expect("headless");
    let mut baker = Baker::new(ctx.clone());
    let mut graph = Graph::new();
    let ramp = graph
        .add_layer(
            "ramp",
            LayerKind::ColorRamp(ColorRamp {
                stops: vec![
                    ColorStop { t: 0.0, color: ColorInput::Const(Color::new(0.0, 0.0, 0.0, 1.0)) },
                    ColorStop { t: 1.0, color: ColorInput::Const(Color::new(1.0, 0.0, 0.0, 1.0)) },
                ],
                space: BlendSpace::Oklch,
            }),
        )
        .unwrap();
    let n = graph
        .add_layer("n", LayerKind::HeightToNormal(HeightToNormal { source: Some(ramp), strength: 1.0 }))
        .unwrap();
    graph.set_output(Output {
        color: graph.output.color,
        roughness: ScalarInput::Const(0.5),
        metallic: ScalarInput::Const(0.0),
        normal: Some(n),
    }).unwrap();
    let out = baker.bake_output(&graph, (32, 32), &EvalCtx::default(), false).expect("bake h2n ramp");
    let px = readback_all_pixels(&ctx, &out.normal, (32, 32));
    let interior = px[16 * 32 + 16];
    // Encoded (n*0.5 + 0.5). Slope is positive in u, so nx < 0 → r < 128.
    // v-gradient is 0 → g ≈ 128. Blue always ≥ 128 (nz always ≥ 0).
    assert!(interior[0] < 120, "expected r < 120 (tilted normal), got {}", interior[0]);
    assert!(interior[1] >= 125 && interior[1] <= 130, "g ≈ 128, got {}", interior[1]);
    assert!(interior[2] >= 128, "b >= 128, got {}", interior[2]);
}

#[test]
fn noise_signed_range_lifts_grays_around_50pct() {
    // Signed range maps [-1,1] straight into L. After Oklch->sRGB clamp+gamma
    // that maps roughly around mid-gray for the center of the distribution.
    let ctx = pollster::block_on(DeviceCtx::request_headless()).expect("headless");
    let mut baker = Baker::new(ctx.clone());
    let size = (32u32, 32u32);
    let mut graph = Graph::new();
    let base = graph.output.color;
    graph
        .set_kind(
            base.unwrap(),
            LayerKind::Noise(Noise {
                dims: NoiseDims::D2,
                seed_offset: 7,
                frequency: 2.0,
                range: NoiseRange::Signed,
                output: NoiseOutput::Grayscale,
                ..Noise::default()
            }),
        )
        .unwrap();
    let out = baker.bake_output(&graph, size, &EvalCtx::default(), false).expect("bake");
    let px = readback_all_pixels(&ctx, &out.color, size);
    // Signed noise produces L both negative and positive; pack clamps to
    // [0,1], so we expect at least one pixel darker than mid-gray and one
    // near or above mid-gray.
    let (mut min, mut max) = (255u8, 0u8);
    for p in &px { min = min.min(p[0]); max = max.max(p[0]); }
    assert!(min < 140 && max > 100, "range spread looks wrong (min={min} max={max})");
}


#[test]
fn null_inputs_render_missing_texture_grid() {
    // A multi-input kind on a single-layer graph must not default its inputs
    // to the only layer available — itself — and die on "would create a
    // cycle". They default to None and render as the missing grid.
    let ctx = pollster::block_on(DeviceCtx::request_headless()).expect("headless");
    let mut baker = Baker::new(ctx.clone());

    let mut graph = Graph::new();
    let base = graph.output.color;
    graph
        .set_kind(base.unwrap(), LayerKind::Map(Map { value: None, palette: None }))
        .expect("null-input Map must not trip the cycle check");
    baker
        .bake_output(&graph, (64, 64), &EvalCtx::default(), false)
        .expect("null-input Map bakes");

    // Passthrough Transform of a null source shows the grid verbatim:
    // 16 cells across [0,1]² → 4-px cells at 64². Every pixel is either
    // magenta or black, and adjacent cells alternate.
    graph
        .set_kind(
            base.unwrap(),
            LayerKind::Transform(Transform {
                source: None,
                offset: [0.0; 3],
                rotate_uv: 0.0,
                scale: [1.0; 3],
                coord_mode: CoordMode::Passthrough,
                edge_mode: EdgeMode::default(),
            }),
        )
        .unwrap();
    let out = baker
        .bake_output(&graph, (64, 64), &EvalCtx::default(), false)
        .expect("null-input Transform bakes");
    let px = readback_all_pixels(&ctx, &out.color, (64, 64));
    let is_magenta = |p: &[u8; 4]| p[0] > 235 && p[1] < 20 && p[3] == 255;
    let is_black = |p: &[u8; 4]| p[0] < 20 && p[1] < 20 && p[2] < 20 && p[3] == 255;
    let magenta = px.iter().filter(|p| is_magenta(p)).count();
    let black = px.iter().filter(|p| is_black(p)).count();
    assert_eq!(
        magenta + black,
        px.len(),
        "missing grid must be only magenta/black (magenta={magenta} black={black})"
    );
    assert!(magenta > 0 && black > 0, "grid should contain both colors");
    let a = px[1 * 64 + 1];
    let b = px[1 * 64 + 5];
    assert_ne!(is_magenta(&a), is_magenta(&b), "adjacent 4-px cells must alternate");

    // CPU evaluator agrees pixel-for-pixel on the grid's layout.
    let cpu = texture_graph_core::evaluate(
        &graph,
        base.unwrap(),
        texture_graph_core::Sample::new(1.5 / 64.0, 1.5 / 64.0, 0.5),
        &EvalCtx::default(),
    );
    let cpu_px = texture_graph_core::color::to_srgb8(cpu);
    assert_eq!(
        is_magenta(&a),
        cpu_px[0] > 235,
        "CPU and GPU disagree on cell color at (1,1): gpu={a:?} cpu={cpu_px:?}"
    );
}

#[test]
fn null_output_color_bakes_missing_grid() {
    // Disconnecting the Output's color socket is legal; the material's
    // base color becomes the missing-texture grid instead of an error.
    let ctx = pollster::block_on(DeviceCtx::request_headless()).expect("headless");
    let mut baker = Baker::new(ctx.clone());

    let mut graph = Graph::new();
    graph.output.color = None;
    let out = baker
        .bake_output(&graph, (64, 64), &EvalCtx::default(), false)
        .expect("null output color bakes");

    let px = readback_all_pixels(&ctx, &out.color, (64, 64));
    let is_magenta = |p: &[u8; 4]| p[0] > 235 && p[1] < 20 && p[3] == 255;
    let is_black = |p: &[u8; 4]| p[0] < 20 && p[1] < 20 && p[2] < 20 && p[3] == 255;
    let magenta = px.iter().filter(|p| is_magenta(p)).count();
    let black = px.iter().filter(|p| is_black(p)).count();
    assert_eq!(
        magenta + black,
        px.len(),
        "null output color must bake the magenta/black grid (magenta={magenta} black={black})"
    );
    assert!(magenta > 0 && black > 0, "grid should contain both colors");

    // CPU material evaluation agrees.
    let m = texture_graph_core::evaluate_material(
        &graph,
        texture_graph_core::Sample::new(1.5 / 64.0, 1.5 / 64.0, 0.5),
        &EvalCtx::default(),
    );
    let cpu_px = texture_graph_core::color::to_srgb8(m.color);
    assert_eq!(
        is_magenta(&px[1 * 64 + 1]),
        cpu_px[0] > 235,
        "CPU and GPU disagree on the null-output grid at (1,1)"
    );
}

/// Extend-mode transform: the source is baked over the transform's real
/// sampling domain, so out-of-[0,1] samples hit real data and match CPU
/// eval. The inner radial transform makes the field genuinely different
/// from what Clamp mode would produce.
#[test]
fn extend_transform_matches_cpu_beyond_unit_square() {
    let ctx = pollster::block_on(DeviceCtx::request_headless()).expect("headless");
    let mut baker = Baker::new(ctx.clone());
    let mut graph = Graph::new();
    let ramp = graph.output.color.unwrap();
    graph
        .set_kind(
            ramp,
            LayerKind::ColorRamp(ColorRamp {
                stops: vec![
                    ColorStop { t: 0.0, color: ColorInput::Const(Color::new(0.05, 0.0, 0.0, 1.0)) },
                    ColorStop { t: 1.0, color: ColorInput::Const(Color::new(0.95, 0.0, 0.0, 1.0)) },
                ],
                space: BlendSpace::Oklch,
            }),
        )
        .unwrap();
    let inner = graph
        .add_layer(
            "radial",
            LayerKind::Transform(Transform {
                source: Some(ramp),
                offset: [0.0; 3],
                rotate_uv: 0.0,
                scale: [0.35, 0.35, 1.0],
                coord_mode: CoordMode::Radial {
                    dim: texture_graph_core::RadialDim::D2,
                    into: texture_graph_core::Axis::U,
                },
                edge_mode: EdgeMode::Clamp,
            }),
        )
        .unwrap();
    let outer = graph
        .add_layer(
            "spread",
            LayerKind::Transform(Transform {
                source: Some(inner),
                offset: [0.0; 3],
                rotate_uv: 0.0,
                scale: [2.0, 2.0, 1.0],
                coord_mode: CoordMode::Passthrough,
                edge_mode: EdgeMode::Extend,
            }),
        )
        .unwrap();
    graph
        .set_output(Output {
            color: Some(outer),
            roughness: ScalarInput::Const(0.5),
            metallic: ScalarInput::Const(0.0),
            normal: None,
        })
        .unwrap();

    const RES: u32 = 64;
    let out = baker
        .bake_output(&graph, (RES, RES), &EvalCtx::default(), false)
        .expect("bake extend");
    let px = readback_all_pixels(&ctx, &out.color, (RES, RES));
    let ctx_eval = EvalCtx::default();
    let mut max_delta = 0i32;
    for y in 0..RES {
        for x in 0..RES {
            let u = (x as f32 + 0.5) / RES as f32;
            let v = (y as f32 + 0.5) / RES as f32;
            let m = texture_graph_core::evaluate_material(&graph, texture_graph_core::Sample::uv(u, v), &ctx_eval);
            let expected = to_srgb8(m.color);
            let got = px[(y * RES + x) as usize];
            for i in 0..3 {
                max_delta = max_delta.max((expected[i] as i32 - got[i] as i32).abs());
            }
        }
    }
    // Nearest-texel resampling across the widened domain quantizes the
    // smooth field slightly; anything small proves extend samples real
    // data (clamp-vs-extend differs by ~100+ sRGB steps mid-frame).
    assert!(max_delta <= 14, "extend parity max delta {max_delta}");

    // Sanity: switching the outer transform to Clamp changes the image.
    graph
        .set_kind(
            outer,
            LayerKind::Transform(Transform {
                source: Some(inner),
                offset: [0.0; 3],
                rotate_uv: 0.0,
                scale: [2.0, 2.0, 1.0],
                coord_mode: CoordMode::Passthrough,
                edge_mode: EdgeMode::Clamp,
            }),
        )
        .unwrap();
    let out_clamp = baker
        .bake_output(&graph, (RES, RES), &EvalCtx::default(), false)
        .expect("bake clamp");
    let px_clamp = readback_all_pixels(&ctx, &out_clamp.color, (RES, RES));
    let mid = ((RES / 2) * RES + RES - 4) as usize;
    let diff = (px[mid][0] as i32 - px_clamp[mid][0] as i32).abs();
    assert!(diff > 20, "extend and clamp should differ off the unit square (diff {diff})");
}

/// An affine extend is unbounded: scale 40 samples far past the old cap
/// and still matches CPU eval (the ramp holds its end color out there).
#[test]
fn affine_extend_is_unbounded_and_matches_cpu() {
    let ctx = pollster::block_on(DeviceCtx::request_headless()).expect("headless");
    let mut baker = Baker::new(ctx.clone());
    let mut graph = Graph::new();
    let ramp = graph.output.color.unwrap();
    graph
        .set_kind(
            ramp,
            LayerKind::ColorRamp(ColorRamp {
                stops: vec![
                    ColorStop { t: 0.0, color: ColorInput::Const(Color::new(0.05, 0.0, 0.0, 1.0)) },
                    ColorStop { t: 1.0, color: ColorInput::Const(Color::new(0.95, 0.0, 0.0, 1.0)) },
                ],
                space: BlendSpace::Oklch,
            }),
        )
        .unwrap();
    let t = graph
        .add_layer(
            "wide",
            LayerKind::Transform(Transform {
                source: Some(ramp),
                offset: [0.0; 3],
                rotate_uv: 0.0,
                scale: [40.0, 1.0, 1.0],
                coord_mode: CoordMode::Passthrough,
                edge_mode: EdgeMode::Extend,
            }),
        )
        .unwrap();
    graph
        .set_output(Output {
            color: Some(t),
            roughness: ScalarInput::Const(0.5),
            metallic: ScalarInput::Const(0.0),
            normal: None,
        })
        .unwrap();
    let out = baker.bake_output(&graph, SIZE, &EvalCtx::default(), false).expect("bake");
    let px = readback_all_pixels(&ctx, &out.color, SIZE);
    let ctx_eval = EvalCtx::default();
    // Skip the leftmost columns where the ramp's [0, 1] span falls between
    // texels of the 40-wide bake; from u=40*x/W > 1 on it's the held end
    // color, which must match CPU exactly — and must NOT be the grid.
    for y in 0..SIZE.1 {
        for x in 4..SIZE.0 {
            let u = (x as f32 + 0.5) / SIZE.0 as f32;
            let v = (y as f32 + 0.5) / SIZE.1 as f32;
            let m = texture_graph_core::evaluate_material(&graph, texture_graph_core::Sample::uv(u, v), &ctx_eval);
            let expected = to_srgb8(m.color);
            let got = px[(y * SIZE.0 + x) as usize];
            for i in 0..3 {
                let d = (expected[i] as i32 - got[i] as i32).abs();
                assert!(
                    d <= 2,
                    "affine extend parity at ({x},{y}) ch{i}: expect {} got {}",
                    expected[i], got[i]
                );
            }
        }
    }
}

/// Radial extend past core's EXTEND_LIMIT cap renders the missing grid,
/// exactly as CPU eval does.
#[test]
fn radial_extend_past_limit_shows_missing_grid_like_cpu() {
    let ctx = pollster::block_on(DeviceCtx::request_headless()).expect("headless");
    let mut baker = Baker::new(ctx.clone());
    let mut graph = Graph::new();
    let c = graph.output.color.unwrap();
    let t = graph
        .add_layer(
            "wild",
            LayerKind::Transform(Transform {
                source: Some(c),
                offset: [0.0; 3],
                rotate_uv: 0.0,
                scale: [40.0, 1.0, 1.0],
                coord_mode: CoordMode::Radial {
                    dim: texture_graph_core::RadialDim::D2,
                    into: texture_graph_core::Axis::U,
                },
                edge_mode: EdgeMode::Extend,
            }),
        )
        .unwrap();
    graph
        .set_output(Output {
            color: Some(t),
            roughness: ScalarInput::Const(0.5),
            metallic: ScalarInput::Const(0.0),
            normal: None,
        })
        .unwrap();
    let out = baker.bake_output(&graph, SIZE, &EvalCtx::default(), false).expect("bake");
    let px = readback_all_pixels(&ctx, &out.color, SIZE);
    let ctx_eval = EvalCtx::default();
    // Right half of the frame samples far past the cap on both CPU and
    // GPU — compare pixels there exactly (same checker formula).
    for y in 0..SIZE.1 {
        for x in (SIZE.0 / 2)..SIZE.0 {
            let u = (x as f32 + 0.5) / SIZE.0 as f32;
            let v = (y as f32 + 0.5) / SIZE.1 as f32;
            let m = texture_graph_core::evaluate_material(&graph, texture_graph_core::Sample::uv(u, v), &ctx_eval);
            let expected = to_srgb8(m.color);
            let got = px[(y * SIZE.0 + x) as usize];
            for i in 0..3 {
                let d = (expected[i] as i32 - got[i] as i32).abs();
                assert!(
                    d <= 2,
                    "missing-grid parity at ({x},{y}) ch{i}: expect {} got {}",
                    expected[i], got[i]
                );
            }
        }
    }
}

/// CPU/GPU noise parity — the claim `core::noise` exists to make good.
///
/// A histogram rather than a single max, because the shape of the
/// disagreement is the diagnostic: a few ±1 steps are rounding in the
/// Oklch→sRGB stage, while a long tail means the kernels have diverged.
///
/// Samples with L outside `[0, 1]` are skipped, not compared. Signed noise
/// spends about a sixth of its range there, and the backends differ on
/// purpose: `to_srgb8` clamps where `pack_srgb8` paints the out-of-range
/// checker. That is a display choice, and this is about the field under it.
///
/// Returns `(histogram, worst, compared, skipped)`.
fn noise_parity_histogram(n: Noise) -> (Vec<u32>, u32, u32, u32) {
    let ctx = pollster::block_on(DeviceCtx::request_headless()).expect("headless");
    let mut baker = Baker::new(ctx.clone());
    let mut graph = Graph::new();
    let id = graph.output.color.unwrap();
    graph.set_kind(id, LayerKind::Noise(n)).unwrap();

    const RES: u32 = 64;
    let eval = EvalCtx::default();
    let out = baker.bake_output(&graph, (RES, RES), &eval, false).expect("bake");
    let px = readback_all_pixels(&ctx, &out.color, (RES, RES));

    let mut hist = vec![0u32; 6];
    let (mut worst, mut compared, mut skipped) = (0u32, 0u32, 0u32);
    for y in 0..RES {
        for x in 0..RES {
            let u = (x as f32 + 0.5) / RES as f32;
            let v = (y as f32 + 0.5) / RES as f32;
            let m = texture_graph_core::evaluate_material(
                &graph,
                // w = 0.5, which is where a flat GPU bake slices a 3D
                // field (`dispatch_kind`'s `w` argument). `Sample::uv` says
                // 0.0, so a D3 graph compared that way is two different
                // slices of the same volume.
                texture_graph_core::Sample::new(u, v, 0.5),
                &eval,
            );
            if !(0.0..=1.0).contains(&m.color.l) {
                skipped += 1;
                continue;
            }
            compared += 1;
            let expected = to_srgb8(m.color);
            let got = px[(y * RES + x) as usize];
            for i in 0..3 {
                let d = (expected[i] as i32 - got[i] as i32).unsigned_abs();
                worst = worst.max(d);
                hist[(d as usize).min(5)] += 1;
            }
        }
    }
    (hist, worst, compared, skipped)
}

#[test]
fn cpu_and_gpu_noise_agree() {
    let plain = |dims, range, frequency| Noise {
        dims,
        seed_offset: 3,
        frequency,
        range,
        output: NoiseOutput::Grayscale,
        ..Noise::default()
    };
    let value = |dims, range, frequency, period| Noise {
        kernel: NoiseKernel::Value,
        period,
        ..plain(dims, range, frequency)
    };
    let cases: [(&str, Noise); 12] = [
        // Simplex, the pre-existing coverage.
        ("simplex D2 unsigned", plain(NoiseDims::D2, NoiseRange::Unsigned, 4.0)),
        ("simplex D2 signed", plain(NoiseDims::D2, NoiseRange::Signed, 9.0)),
        ("simplex D1 unsigned", plain(NoiseDims::D1, NoiseRange::Unsigned, 6.0)),
        ("simplex D3 unsigned", plain(NoiseDims::D3, NoiseRange::Unsigned, 5.0)),
        ("simplex D3 signed", plain(NoiseDims::D3, NoiseRange::Signed, 2.0)),
        // Value, aperiodic and periodic, every dimensionality.
        ("value D1 unsigned", value(NoiseDims::D1, NoiseRange::Unsigned, 6.0, [0; 3])),
        ("value D2 unsigned", value(NoiseDims::D2, NoiseRange::Unsigned, 4.0, [0; 3])),
        ("value D2 signed", value(NoiseDims::D2, NoiseRange::Signed, 9.0, [0; 3])),
        ("value D3 unsigned", value(NoiseDims::D3, NoiseRange::Unsigned, 5.0, [0; 3])),
        ("value D2 tiling", value(NoiseDims::D2, NoiseRange::Unsigned, 8.0, [8, 8, 0])),
        // Octaves, on both kernels and a shaping mode each.
        (
            "value ridged x4",
            Noise {
                fractal: Fractal {
                    octaves: 4,
                    lacunarity: 2.0,
                    gain: 0.5,
                    mode: FractalMode::Ridged,
                    normalize: true,
                },
                ..value(NoiseDims::D3, NoiseRange::Unsigned, 3.0, [0; 3])
            },
        ),
        (
            "simplex turbulence x5",
            Noise {
                fractal: Fractal {
                    octaves: 5,
                    lacunarity: 2.13,
                    gain: 0.55,
                    mode: FractalMode::Turbulence,
                    normalize: true,
                },
                ..plain(NoiseDims::D2, NoiseRange::Unsigned, 3.0)
            },
        ),
    ];
    for (label, n) in cases {
        let (hist, worst, compared, skipped) = noise_parity_histogram(n);
        println!(
            "{label}: worst={worst} hist={hist:?} compared={compared} skipped={skipped}"
        );
        // A test that skipped everything would pass on an empty
        // comparison. Signed noise spends a good third of its range below
        // zero, so the floor is a quarter of the image rather than half.
        assert!(compared * 4 > 64 * 64, "{label}: only {compared} samples were in range");
        assert!(worst <= 1, "{label}: worst sRGB delta {worst}, histogram {hist:?}");
    }
}

/// The claim §1 of the consumer doc is built on, checked where it matters:
/// on the GPU, in a real bake, not only in the CPU kernel.
///
/// `period / frequency` is the repeat length in sample space. At
/// `period = 4`, `frequency = 16` that is a quarter of a unit, so a bake of
/// the unit square is four identical tiles across and down. Exact pixel
/// equality is the right bar: same shader, same lattice cells, same integer
/// corner hashes.
const TILE_PERIOD: u32 = 4;
const TILES_ACROSS: u32 = 4;
const TILE_RES: u32 = 64;

fn tiling_noise(kernel: NoiseKernel, period: [u32; 3]) -> Noise {
    Noise {
        dims: NoiseDims::D2,
        seed_offset: 1,
        frequency: (TILE_PERIOD * TILES_ACROSS) as f32,
        range: NoiseRange::Unsigned,
        output: NoiseOutput::Grayscale,
        kernel,
        period,
        fractal: Fractal::default(),
    }
}

fn bake_noise_pixels(n: Noise) -> Vec<[u8; 4]> {
    let ctx = pollster::block_on(DeviceCtx::request_headless()).expect("headless");
    let mut baker = Baker::new(ctx.clone());
    let mut graph = Graph::new();
    let id = graph.output.color.unwrap();
    graph.set_kind(id, LayerKind::Noise(n)).unwrap();
    let size = (TILE_RES, TILE_RES);
    let out = baker.bake_output(&graph, size, &EvalCtx::default(), false).expect("bake");
    readback_all_pixels(&ctx, &out.color, size)
}

#[test]
fn a_periodic_value_bake_tiles_exactly_on_the_gpu() {
    let px = bake_noise_pixels(tiling_noise(
        NoiseKernel::Value,
        [TILE_PERIOD, TILE_PERIOD, 0],
    ));
    let tile = TILE_RES / TILES_ACROSS;
    let mut varied = false;
    for y in 0..tile {
        for x in 0..tile {
            let first = px[(y * TILE_RES + x) as usize];
            if first != px[0] {
                varied = true;
            }
            for ty in 0..TILES_ACROSS {
                for tx in 0..TILES_ACROSS {
                    let (sx, sy) = (x + tx * tile, y + ty * tile);
                    assert_eq!(
                        first,
                        px[(sy * TILE_RES + sx) as usize],
                        "tile ({tx},{ty}) differs at ({x},{y})"
                    );
                }
            }
        }
    }
    assert!(varied, "the field is constant, so tiling proves nothing");
}

/// The same bake without a period must *not* repeat — otherwise the field
/// tiles for some reason of its own and the test above proves nothing.
#[test]
fn an_aperiodic_bake_does_not_tile() {
    let tile = TILE_RES / TILES_ACROSS;
    let repeats = |px: &[[u8; 4]]| {
        (0..tile).all(|y| {
            (0..tile).all(|x| {
                px[(y * TILE_RES + x) as usize] == px[(y * TILE_RES + x + tile) as usize]
            })
        })
    };
    for kernel in [NoiseKernel::Simplex, NoiseKernel::Value] {
        let px = bake_noise_pixels(tiling_noise(kernel, [0; 3]));
        assert!(!repeats(&px), "{kernel:?} repeated across a tile boundary");
    }
}

// ---- Single-channel output (§5) ----------------------------------------

/// A graph whose Output is a plain gray, plus an unrelated noise layer.
/// The noise is deliberately *not* wired to the Output: `bake_scalar` is
/// supposed to bake the layer it is asked for, so a graph can carry
/// several fields side by side.
fn scalar_fixture(kernel: NoiseKernel) -> (Graph, LayerId) {
    let mut graph = Graph::new();
    let field = graph
        .add_layer(
            "field",
            LayerKind::Noise(Noise {
                dims: NoiseDims::D3,
                seed_offset: 7,
                frequency: 5.0,
                range: NoiseRange::Unsigned,
                output: NoiseOutput::Grayscale,
                kernel,
                period: [0; 3],
                fractal: Fractal { octaves: 3, ..Fractal::default() },
            }),
        )
        .unwrap();
    (graph, field)
}

fn cpu_scalar(graph: &Graph, layer: LayerId, u: f32, v: f32, w: f32) -> f32 {
    texture_graph_core::color::scalar_of(texture_graph_core::evaluate(
        graph,
        layer,
        texture_graph_core::Sample::new(u, v, w),
        &EvalCtx::default(),
    ))
}

/// Every format reproduces the CPU field to its own precision, and the
/// layer baked is the one asked for rather than whatever the Output
/// happens to point at.
#[test]
fn scalar_bake_matches_the_cpu_field_in_every_format() {
    const RES: u32 = 32;
    let (graph, field) = scalar_fixture(NoiseKernel::Value);
    let ctx = pollster::block_on(DeviceCtx::request_headless()).expect("headless");
    let mut baker = Baker::new(ctx.clone());

    for (format, tol) in [
        // 1/255 of quantization, plus the CPU/GPU kernel difference.
        (ScalarFormat::R8Unorm, 1.0 / 255.0 + 2e-3),
        // Half has ~3 decimal digits over [0, 1].
        (ScalarFormat::R16Float, 1e-3 + 2e-3),
        (ScalarFormat::R32Float, 2e-3),
    ] {
        let tex = baker
            .bake_scalar(&graph, field, (RES, RES), format, &EvalCtx::default())
            .expect("bake_scalar");
        let img = crate::readback::read_scalar(&ctx, &tex, (RES, RES), format);
        assert_eq!((img.width, img.height, img.depth), (RES, RES, 1));
        assert_eq!(
            img.bytes.len() as u32,
            RES * RES * format.bytes_per_texel(),
            "{format:?}: one texel per pixel, no row padding"
        );

        let mut worst = 0.0f32;
        for y in 0..RES {
            for x in 0..RES {
                let u = (x as f32 + 0.5) / RES as f32;
                let v = (y as f32 + 0.5) / RES as f32;
                let expected = cpu_scalar(&graph, field, u, v, texture_graph_core::FLAT_W);
                let got = img.value(x, y, 0).unwrap();
                worst = worst.max((expected - got).abs());
            }
        }
        assert!(worst <= tol, "{format:?}: worst scalar delta {worst} > {tol}");
    }
}

/// `R8Unorm` clamps as the format requires; the float formats keep a
/// signed field intact, which is the reason to pay for them.
#[test]
fn scalar_formats_differ_on_out_of_range_values() {
    const RES: u32 = 16;
    let mut graph = Graph::new();
    let signed = graph
        .add_layer(
            "signed",
            LayerKind::Noise(Noise {
                dims: NoiseDims::D2,
                seed_offset: 2,
                frequency: 6.0,
                range: NoiseRange::Signed,
                output: NoiseOutput::Grayscale,
                kernel: NoiseKernel::Value,
                ..Noise::default()
            }),
        )
        .unwrap();
    let ctx = pollster::block_on(DeviceCtx::request_headless()).expect("headless");
    let mut baker = Baker::new(ctx.clone());
    let read = |baker: &mut Baker, format| {
        let tex = baker
            .bake_scalar(&graph, signed, (RES, RES), format, &EvalCtx::default())
            .expect("bake_scalar");
        crate::readback::read_scalar(&ctx, &tex, (RES, RES), format)
    };

    let unorm = read(&mut baker, ScalarFormat::R8Unorm);
    let float = read(&mut baker, ScalarFormat::R32Float);
    let mut saw_negative = false;
    for y in 0..RES {
        for x in 0..RES {
            let f = float.value(x, y, 0).unwrap();
            if f < -1e-3 {
                saw_negative = true;
                assert_eq!(
                    unorm.value(x, y, 0).unwrap(),
                    0.0,
                    "R8Unorm should have clamped {f} at ({x},{y})"
                );
            }
        }
    }
    assert!(saw_negative, "signed noise produced nothing below zero to clamp");
}

/// The volume path: `res³` at slice centres, matching the CPU field and
/// actually varying along w.
#[test]
fn scalar_volume_matches_the_cpu_field_through_w() {
    const RES: u32 = 16;
    const DEPTH: u32 = 8;
    let (graph, field) = scalar_fixture(NoiseKernel::Value);
    let ctx = pollster::block_on(DeviceCtx::request_headless()).expect("headless");
    let mut baker = Baker::new(ctx.clone());
    let vol = baker
        .bake_scalar_volume(
            &graph, field, RES, DEPTH, ScalarFormat::R32Float, &EvalCtx::default(),
        )
        .expect("bake_scalar_volume");
    assert_eq!(vol.size, (RES, RES, DEPTH));
    let img = crate::readback::read_scalar_volume(&ctx, &vol.texture, vol.size, vol.format);

    let mut worst = 0.0f32;
    let mut varies_through_w = false;
    for z in 0..DEPTH {
        let w = (z as f32 + 0.5) / DEPTH as f32;
        for y in 0..RES {
            for x in 0..RES {
                let u = (x as f32 + 0.5) / RES as f32;
                let v = (y as f32 + 0.5) / RES as f32;
                let expected = cpu_scalar(&graph, field, u, v, w);
                let got = img.value(x, y, z).unwrap();
                worst = worst.max((expected - got).abs());
            }
        }
        if z > 0 && img.value(0, 0, z) != img.value(0, 0, 0) {
            varies_through_w = true;
        }
    }
    assert!(worst <= 2e-3, "worst scalar delta through the volume: {worst}");
    assert!(varies_through_w, "the volume is the same slice repeated");
}

/// `bake_scalar` schedules only what its layer reads. A graph with a big
/// unrelated Output branch must not pay for it.
#[test]
fn scalar_bake_ignores_layers_its_field_does_not_read() {
    let (mut graph, field) = scalar_fixture(NoiseKernel::Simplex);
    let mut chain = graph.output.color.unwrap();
    for i in 0..4 {
        chain = add_mix(&mut graph, &format!("unrelated-{i}"), chain, chain);
    }
    graph
        .set_output(Output {
            color: Some(chain),
            roughness: ScalarInput::Const(0.5),
            metallic: ScalarInput::Const(0.0),
            normal: None,
        })
        .unwrap();
    let sched = crate::schedule::schedule_layer(&graph, field).expect("schedule");
    assert_eq!(sched.order, vec![field], "only the field should be dispatched");
    // And the ordinary Output schedule still covers the branch, so the two
    // are genuinely different plans rather than one that lost layers.
    assert!(crate::schedule::schedule(&graph).unwrap().order.len() > 1);
}

/// The async readers have to produce the same bytes as the blocking ones —
/// they are the only path a wasm host has, so a difference would be a bug
/// nobody native ever hits.
#[test]
fn async_readback_agrees_with_the_blocking_one() {
    const RES: u32 = 16;
    let (graph, field) = scalar_fixture(NoiseKernel::Value);
    let ctx = pollster::block_on(DeviceCtx::request_headless()).expect("headless");
    let mut baker = Baker::new(ctx.clone());

    let tex = baker
        .bake_scalar(&graph, field, (RES, RES), ScalarFormat::R16Float, &EvalCtx::default())
        .expect("bake_scalar");
    let blocking =
        crate::readback::read_scalar(&ctx, &tex, (RES, RES), ScalarFormat::R16Float);
    let asynced = drive(
        &ctx,
        crate::readback::read_scalar_async(&ctx, &tex, (RES, RES), ScalarFormat::R16Float),
    );
    assert_eq!(blocking.bytes, asynced.bytes);

    let color = baker
        .bake_output(&graph, (RES, RES), &EvalCtx::default(), false)
        .expect("bake")
        .color;
    let blocking_rgba = crate::readback::read_rgba8(&ctx, &color, (RES, RES));
    let async_rgba = drive(&ctx, crate::readback::read_rgba8_async(&ctx, &color, (RES, RES)));
    assert_eq!(blocking_rgba.pixels, async_rgba.pixels);
}

/// Run a readback future to completion on native, the way a host with a
/// frame loop would: poll the future, then give the device a chance to
/// resolve the buffer map, and repeat.
///
/// `pollster::block_on` alone deadlocks here — it parks the only thread
/// that could call `Device::poll`. On wasm neither is needed; the browser
/// resolves the map on its own event loop, which is the whole reason the
/// async readers exist.
fn drive<F: std::future::Future>(ctx: &DeviceCtx, fut: F) -> F::Output {
    use std::sync::Arc;
    use std::task::{Context, Poll, Wake, Waker};

    struct Noop;
    impl Wake for Noop {
        fn wake(self: Arc<Self>) {}
    }
    let waker = Waker::from(Arc::new(Noop));
    let mut cx = Context::from_waker(&waker);
    let mut fut = Box::pin(fut);
    loop {
        if let Poll::Ready(v) = fut.as_mut().poll(&mut cx) {
            return v;
        }
        let _ = ctx.device.poll(wgpu::PollType::Poll);
    }
}

// ---- Wave (§4) ----------------------------------------------------------

/// CPU/GPU parity for the waveform node, at the looser bound its doc
/// comment promises.
///
/// `Sine` is transcendental, so this is explicitly *not* the noise
/// op-order contract: within one sRGB step, not bit-exact. Asserting the
/// bound honestly is the point — a test that demanded equality here would
/// either be disabled or force a lie in the doc comment.
#[test]
fn cpu_and_gpu_wave_agree_within_one_srgb_step() {
    const RES: u32 = 64;
    let ctx = pollster::block_on(DeviceCtx::request_headless()).expect("headless");
    let mut baker = Baker::new(ctx.clone());

    for shape in [WaveShape::Sine, WaveShape::Triangle, WaveShape::Square, WaveShape::Sawtooth] {
        for range in [NoiseRange::Unsigned, NoiseRange::Signed] {
            // A ramp along U gives the wave a coordinate to run over —
            // the shape the banded-planet surface needs.
            let mut graph = Graph::new();
            let ramp = graph.output.color.unwrap();
            graph
                .set_kind(
                    ramp,
                    LayerKind::ColorRamp(ColorRamp {
                        stops: vec![
                            ColorStop {
                                t: 0.0,
                                color: ColorInput::Const(Color::new(0.0, 0.0, 0.0, 1.0)),
                            },
                            ColorStop {
                                t: 1.0,
                                color: ColorInput::Const(Color::new(1.0, 0.0, 0.0, 1.0)),
                            },
                        ],
                        space: BlendSpace::Oklch,
                    }),
                )
                .unwrap();
            let wave = graph
                .add_layer(
                    "wave",
                    LayerKind::Wave(Wave {
                        input: ScalarInput::Layer(ramp),
                        shape,
                        frequency: 5.0,
                        phase: 0.125,
                        range,
                    }),
                )
                .unwrap();
            graph
                .set_output(Output {
                    color: Some(wave),
                    roughness: ScalarInput::Const(0.5),
                    metallic: ScalarInput::Const(0.0),
                    normal: None,
                })
                .unwrap();

            let eval = EvalCtx::default();
            let out = baker.bake_output(&graph, (RES, RES), &eval, false).expect("bake");
            let px = readback_all_pixels(&ctx, &out.color, (RES, RES));
            let (mut worst, mut compared) = (0u32, 0u32);
            for y in 0..RES {
                for x in 0..RES {
                    let u = (x as f32 + 0.5) / RES as f32;
                    let v = (y as f32 + 0.5) / RES as f32;
                    let m = texture_graph_core::evaluate_material(
                        &graph,
                        texture_graph_core::Sample::new(u, v, texture_graph_core::FLAT_W),
                        &eval,
                    );
                    // Signed output spends half its range below zero,
                    // where the two backends differ on purpose — `to_srgb8`
                    // clamps and `pack_srgb8` paints the checker.
                    if !(0.0..=1.0).contains(&m.color.l) {
                        continue;
                    }
                    compared += 1;
                    let expected = to_srgb8(m.color);
                    let got = px[(y * RES + x) as usize];
                    for i in 0..3 {
                        worst = worst
                            .max((expected[i] as i32 - got[i] as i32).unsigned_abs());
                    }
                }
            }
            assert!(compared * 4 > RES * RES, "{shape:?} {range:?}: only {compared} in range");
            assert!(worst <= 1, "{shape:?} {range:?}: worst sRGB delta {worst}");
        }
    }
}

/// A `Const` input never reads the bound texture — the shader's
/// placeholder path. The result must be one flat value, not whatever
/// happened to be in the slot.
#[test]
fn a_const_input_wave_bakes_flat() {
    const RES: u32 = 16;
    let ctx = pollster::block_on(DeviceCtx::request_headless()).expect("headless");
    let mut baker = Baker::new(ctx.clone());
    let mut graph = Graph::new();
    let id = graph.output.color.unwrap();
    graph
        .set_kind(
            id,
            LayerKind::Wave(Wave {
                input: ScalarInput::Const(0.125),
                shape: WaveShape::Sine,
                frequency: 2.0,
                phase: 0.0,
                range: NoiseRange::Unsigned,
            }),
        )
        .unwrap();
    let out = baker.bake_output(&graph, (RES, RES), &EvalCtx::default(), false).expect("bake");
    let px = readback_all_pixels(&ctx, &out.color, (RES, RES));
    assert!(px.iter().all(|p| *p == px[0]), "a constant input gave a varying field");
    // 0.125 × 2 cycles is sine's peak, so L = 1 — white.
    assert!(px[0][0] > 250, "expected the peak of the wave, got {:?}", px[0]);
}
