//! End-to-end bake tests. Each variant that lands should add one here
//! comparing the GPU output to `core`'s CPU evaluator.

use texture_graph_core::{
    Color, EvalCtx, Graph, LayerKind, Noise, NoiseDims, NoiseOutput, NoiseRange, Output,
    ScalarInput, color::to_srgb8,
};

use crate::{Baker, DeviceCtx};

const SIZE: (u32, u32) = (8, 8);

/// Read every pixel of an Rgba8Unorm texture into a flat Vec<[u8; 4]>.
fn readback_all_pixels(ctx: &DeviceCtx, tex: &wgpu::Texture, size: (u32, u32)) -> Vec<[u8; 4]> {
    // 256-byte-aligned row stride.
    let raw_bpr = size.0 * 4;
    let bytes_per_row = (raw_bpr + 255) & !255;
    let readback = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback-all"),
        size: (bytes_per_row * size.1) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("readback-all-enc"),
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
                rows_per_image: Some(size.1),
            },
        },
        wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
    );
    ctx.queue.submit([enc.finish()]);
    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
    ctx.device.poll(wgpu::PollType::Wait).expect("poll");
    rx.recv().expect("chan").expect("map");
    let data = slice.get_mapped_range();
    let mut out = Vec::with_capacity((size.0 * size.1) as usize);
    for y in 0..size.1 {
        let row = &data[(y * bytes_per_row) as usize..][..raw_bpr as usize];
        for x in 0..size.0 {
            let px = &row[(x * 4) as usize..][..4];
            out.push([px[0], px[1], px[2], px[3]]);
        }
    }
    drop(data);
    readback.unmap();
    out
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
    ctx.device.poll(wgpu::PollType::Wait).expect("poll");
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
        .set_kind(base, LayerKind::Color(Color::new(0.6, 0.15, 40.0, 1.0)))
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
        .bake_output(&graph, SIZE, &EvalCtx::default())
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
            base,
            LayerKind::Noise(Noise {
                dims: NoiseDims::D2,
                seed_offset: 0,
                frequency: 4.0,
                range: NoiseRange::Unsigned,
                output: NoiseOutput::Grayscale,
            }),
        )
        .unwrap();

    let out = baker
        .bake_output(&graph, size, &EvalCtx::default())
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
        .bake_output(&graph, size, &EvalCtx::default())
        .expect("bake noise 2");
    let px2 = readback_all_pixels(&ctx, &out2.color, size);
    assert_eq!(px, px2, "noise output changed across identical bakes");
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
            base,
            LayerKind::Noise(Noise {
                dims: NoiseDims::D2,
                seed_offset: 7,
                frequency: 2.0,
                range: NoiseRange::Signed,
                output: NoiseOutput::Grayscale,
            }),
        )
        .unwrap();
    let out = baker.bake_output(&graph, size, &EvalCtx::default()).expect("bake");
    let px = readback_all_pixels(&ctx, &out.color, size);
    // Signed noise produces L both negative and positive; pack clamps to
    // [0,1], so we expect at least one pixel darker than mid-gray and one
    // near or above mid-gray.
    let (mut min, mut max) = (255u8, 0u8);
    for p in &px { min = min.min(p[0]); max = max.max(p[0]); }
    assert!(min < 140 && max > 100, "range spread looks wrong (min={min} max={max})");
}

