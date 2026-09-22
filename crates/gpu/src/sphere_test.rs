//! The sphere bake: that it holds the CPU's field, that its faces are where
//! the GPU's cube sampler looks for them, and that it refuses unsupported
//! layers.

use texture_graph_core::{
    Axis, BlendMode, BlendSpace, Coordinate, EvalCtx, Fractal, Graph, LayerId,
    LayerKind, Mix, Noise, NoiseDims, NoiseKernel, NoiseRange, ScalarInput,
    Transform, CoordMode, EdgeMode, Wave, WaveShape, color::oklcha, cube_sample, eval,
};

use crate::{BakeError, Baker, DeviceCtx, ScalarFormat, read_scalar_volume};

fn blend(g: &mut Graph, name: &str, a: LayerId, b: LayerId, factor: f32) -> LayerId {
    g.add_layer(
        name,
        LayerKind::Mix(Mix {
            a: Some(a),
            b: Some(b),
            mode: BlendMode::Blend,
            factor: ScalarInput::Const(factor),
            space: BlendSpace::Oklch,
        }),
    )
    .unwrap()
}

/// Latitude bands bent by noise and stretched past `[0, 1]`, using every
/// kind a sphere bake allows.
fn banded() -> (Graph, LayerId) {
    let mut g = Graph::new();
    let height = g.add_layer("height", LayerKind::Coordinate(Coordinate { axis: Axis::V })).unwrap();
    let turbulence = g
        .add_layer(
            "turbulence",
            LayerKind::Noise(Noise {
                dims: NoiseDims::D3,
                kernel: NoiseKernel::Value,
                frequency: 6.2,
                fractal: Fractal { octaves: 4, lacunarity: 2.13, ..Default::default() },
                ..Default::default()
            }),
        )
        .unwrap();
    let bent = blend(&mut g, "bent", height, turbulence, 0.03);
    let bands = g
        .add_layer(
            "bands",
            LayerKind::Wave(Wave {
                input: ScalarInput::Layer(bent),
                shape: WaveShape::Sine,
                frequency: 5.9,
                phase: 0.1,
                range: NoiseRange::Unsigned,
            }),
        )
        .unwrap();
    let half = g.add_layer("half", LayerKind::Color(oklcha(0.5, 0.0, 0.0, 1.0))).unwrap();
    let out = blend(&mut g, "sharpened", half, bands, 1.24);
    (g, out)
}

fn cpu_on_sphere(g: &Graph, id: LayerId, face: u32, x: u32, y: u32, size: u32) -> f32 {
    let (u, v) = ((x as f32 + 0.5) / size as f32, (y as f32 + 0.5) / size as f32);
    eval::evaluate(g, id, cube_sample(face, u, v), &EvalCtx::default()).l
}

#[test]
fn a_sphere_bake_holds_the_cpu_field_on_every_face() {
    const FACE: u32 = 24;
    let (graph, field) = banded();
    let ctx = pollster::block_on(DeviceCtx::request_headless()).expect("headless");
    let cube = Baker::new(ctx.clone())
        .bake_scalar_cube(&graph, field, FACE, ScalarFormat::R32Float, &EvalCtx::default())
        .expect("bake_scalar_cube");
    let img = read_scalar_volume(&ctx, &cube.texture, (FACE, FACE, 6), cube.format);

    let mut worst = 0.0f32;
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for face in 0..6 {
        for y in 0..FACE {
            for x in 0..FACE {
                let got = img.value(x, y, face).unwrap();
                worst = worst.max((got - cpu_on_sphere(&graph, field, face, x, y, FACE)).abs());
                lo = lo.min(got);
                hi = hi.max(got);
            }
        }
    }
    // `sin` differs between backends.
    assert!(worst <= 2.0 / 255.0, "worst delta over the sphere: {worst}");
    // Proves the factor is not clamped on the way through.
    assert!(lo < -0.05 && hi > 1.05, "field spans {lo}..{hi}");
}

/// A face stored mirrored, rotated or in the wrong layer passes every
/// CPU-side test and fails here.
#[test]
fn faces_land_where_the_gpu_cube_sampler_looks_for_them() {
    const FACE: u32 = 32;
    let ctx = pollster::block_on(DeviceCtx::request_headless()).expect("headless");
    let mut baker = Baker::new(ctx.clone());
    let cubes: Vec<wgpu::TextureView> = [Axis::U, Axis::V, Axis::W]
        .into_iter()
        .map(|axis| {
            let mut g = Graph::new();
            let c = g.add_layer("c", LayerKind::Coordinate(Coordinate { axis })).unwrap();
            let cube = baker
                .bake_scalar_cube(&g, c, FACE, ScalarFormat::R32Float, &EvalCtx::default())
                .expect("bake");
            cube.texture.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::Cube),
                ..Default::default()
            })
        })
        .collect();

    let dirs: Vec<[f32; 4]> = vec![
        [1.0, 0.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.8, 0.5, -0.3, 0.0],
        [-0.4, 0.9, 0.6, 0.0],
        [0.2, -0.7, 0.9, 0.0],
        [-0.6, -0.2, -0.8, 0.0],
        [0.5, 0.3, 0.9, 0.0],
        [-0.9, 0.4, 0.1, 0.0],
    ];
    let got = sample_cubes(&ctx, &cubes, &dirs);
    for (d, g) in dirs.iter().zip(&got) {
        let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        let want = [d[0] / r * 0.5 + 0.5, d[1] / r * 0.5 + 0.5, d[2] / r * 0.5 + 0.5];
        for k in 0..3 {
            assert!(
                (g[k] - want[k]).abs() < 0.05,
                "direction {d:?}: sampled {g:?}, expected {want:?}",
            );
        }
    }
}

#[test]
fn a_sphere_bake_refuses_what_has_no_meaning_on_a_sphere() {
    let mut g = Graph::new();
    let n = g.add_layer("n", LayerKind::Noise(Noise::default())).unwrap();
    let t = g
        .add_layer(
            "t",
            LayerKind::Transform(Transform {
                source: Some(n),
                offset: [0.0; 3],
                rotate_uv: 0.0,
                scale: [2.0; 3],
                coord_mode: CoordMode::Passthrough,
                edge_mode: EdgeMode::Clamp,
            }),
        )
        .unwrap();
    let ctx = pollster::block_on(DeviceCtx::request_headless()).expect("headless");
    let refused = Baker::new(ctx)
        .bake_scalar_cube(&g, t, 8, ScalarFormat::R8Unorm, &EvalCtx::default())
        .err();
    assert!(matches!(refused, Some(BakeError::Unsupported(_))), "got {refused:?}");
}

#[test]
fn a_coordinate_on_a_plane_is_the_pixel() {
    const SIZE: u32 = 16;
    let ctx = pollster::block_on(DeviceCtx::request_headless()).expect("headless");
    let mut baker = Baker::new(ctx.clone());
    for axis in [Axis::U, Axis::V, Axis::W] {
        let mut g = Graph::new();
        let c = g.add_layer("c", LayerKind::Coordinate(Coordinate { axis })).unwrap();
        let tex = baker
            .bake_scalar(&g, c, (SIZE, SIZE), ScalarFormat::R32Float, &EvalCtx::default())
            .expect("bake_scalar");
        let img = crate::read_scalar(&ctx, &tex, (SIZE, SIZE), ScalarFormat::R32Float);
        for (x, y) in [(0, 0), (5, 11), (15, 15)] {
            let want = match axis {
                Axis::U => (x as f32 + 0.5) / SIZE as f32,
                Axis::V => (y as f32 + 0.5) / SIZE as f32,
                Axis::W => texture_graph_core::FLAT_W,
            };
            let got = img.value(x, y, 0).unwrap();
            assert!((got - want).abs() < 1e-6, "{axis:?} at ({x}, {y}): {got}, expected {want}");
        }
    }
}

/// One `[u, v, w]` per direction, nearest filtering.
fn sample_cubes(ctx: &DeviceCtx, cubes: &[wgpu::TextureView], dirs: &[[f32; 4]]) -> Vec<[f32; 3]> {
    const SHADER: &str = r#"
        @group(0) @binding(0) var cube_u: texture_cube<f32>;
        @group(0) @binding(1) var cube_v: texture_cube<f32>;
        @group(0) @binding(2) var cube_w: texture_cube<f32>;
        @group(0) @binding(3) var nearest: sampler;
        @group(0) @binding(4) var<storage, read> dirs: array<vec4<f32>>;
        @group(0) @binding(5) var<storage, read_write> out: array<vec4<f32>>;

        @compute @workgroup_size(1)
        fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
            let d = dirs[gid.x].xyz;
            out[gid.x] = vec4<f32>(
                textureSampleLevel(cube_u, nearest, d, 0.0).r,
                textureSampleLevel(cube_v, nearest, d, 0.0).r,
                textureSampleLevel(cube_w, nearest, d, 0.0).r,
                0.0,
            );
        }
    "#;
    use wgpu::util::DeviceExt;
    let device = &ctx.device;
    let cube_entry = |binding| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::Cube,
            multisampled: false,
        },
        count: None,
    };
    let buffer_entry = |binding, read_only| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("sphere-test-bgl"),
        entries: &[
            cube_entry(0),
            cube_entry(1),
            cube_entry(2),
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                count: None,
            },
            buffer_entry(4, true),
            buffer_entry(5, false),
        ],
    });
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor::default());
    let dir_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("sphere-test-dirs"),
        contents: bytemuck::cast_slice(dirs),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let bytes = (dirs.len() * 16) as u64;
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sphere-test-out"),
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let read_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sphere-test-read"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("sphere-test-bg"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&cubes[0]) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&cubes[1]) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&cubes[2]) },
            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&sampler) },
            wgpu::BindGroupEntry { binding: 4, resource: dir_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: out_buf.as_entire_binding() },
        ],
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("sphere-test-pl"),
        bind_group_layouts: &[Some(&bgl)],
        ..Default::default()
    });
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("sphere-test-shader"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("sphere-test-pipeline"),
        layout: Some(&pl),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let mut enc = device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(dirs.len() as u32, 1, 1);
    }
    enc.copy_buffer_to_buffer(&out_buf, 0, &read_buf, 0, bytes);
    ctx.queue.submit([enc.finish()]);
    read_buf.slice(..).map_async(wgpu::MapMode::Read, |r| r.expect("map"));
    device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let data = read_buf.slice(..).get_mapped_range();
    let values: &[[f32; 4]] = bytemuck::cast_slice(&data);
    values.iter().map(|v| [v[0], v[1], v[2]]).collect()
}

/// A volume whose Output normal is a HeightToNormal carries that node's
/// source as a height, for the solid preview to bump by.
#[test]
fn volume_bake_carries_the_height_under_a_height_to_normal() {
    use texture_graph_core::{Axis, Coordinate, HeightToNormal, Output, ScalarInput};
    let ctx = pollster::block_on(crate::DeviceCtx::request_headless()).expect("headless");
    let mut baker = crate::Baker::new(ctx.clone());
    let mut g = texture_graph_core::Graph::new();
    let w = g.add_layer("w", LayerKind::Coordinate(Coordinate { axis: Axis::W })).unwrap();
    let n = g
        .add_layer("n", LayerKind::HeightToNormal(HeightToNormal { source: Some(w), strength: 3.0 }))
        .unwrap();
    let color = g.output.color;
    g.set_output(Output { color, roughness: ScalarInput::Const(0.5), metallic: ScalarInput::Const(0.0), normal: Some(n) })
        .unwrap();

    const R: u32 = 16;
    let vol = baker.bake_volume(&g, R, R, &EvalCtx::default()).expect("bake");
    let bump = vol.bump.expect("a HeightToNormal output bakes a height");
    assert_eq!(bump.strength, 3.0);
    let h = crate::read_scalar_volume(&ctx, &bump.height, (R, R, R), crate::ScalarFormat::R16Float);
    for z in [0, 7, 15] {
        let want = (z as f32 + 0.5) / R as f32;
        let got = h.value(5, 9, z).unwrap();
        assert!((got - want).abs() < 2e-3, "z={z}: want {want}, got {got}");
    }

    g.set_output(Output { color, roughness: ScalarInput::Const(0.5), metallic: ScalarInput::Const(0.0), normal: None })
        .unwrap();
    assert!(baker.bake_volume(&g, R, R, &EvalCtx::default()).unwrap().bump.is_none());
}

/// A Map through a ColorRamp palette, over noise moved by a Transform: the
/// palette bakes on a plane and the Transform's map moves the noise's points.
#[test]
fn a_color_cube_holds_the_cpu_color_through_a_palette_and_a_transform() {
    use texture_graph_core::{ColorInput, ColorRamp, ColorStop, Map, color::to_srgb8};
    const FACE: u32 = 32;
    let mut g = Graph::new();
    let n = g
        .add_layer(
            "n",
            LayerKind::Noise(Noise {
                dims: NoiseDims::D3,
                frequency: 2.3,
                fractal: Fractal { octaves: 3, ..Default::default() },
                ..Default::default()
            }),
        )
        .unwrap();
    let moved = g
        .add_layer(
            "moved",
            LayerKind::Transform(Transform {
                source: Some(n),
                offset: [0.1, 0.5, 0.2],
                rotate_uv: 0.4,
                scale: [1.0, 1.8, 0.7],
                coord_mode: CoordMode::Passthrough,
                edge_mode: EdgeMode::Extend,
            }),
        )
        .unwrap();
    let stop = |t, l, c, h| ColorStop { t, color: ColorInput::Const(oklcha(l, c, h, 1.0)) };
    let ramp = g
        .add_layer(
            "ramp",
            LayerKind::ColorRamp(ColorRamp {
                stops: vec![stop(0.0, 0.3, 0.05, 250.0), stop(1.0, 0.8, 0.1, 60.0)],
                space: BlendSpace::Oklch,
            }),
        )
        .unwrap();
    let out = g
        .add_layer("out", LayerKind::Map(Map { value: Some(moved), palette: Some(ramp) }))
        .unwrap();
    g.output.color = Some(out);

    let ctx = pollster::block_on(DeviceCtx::request_headless()).expect("headless");
    let cube = Baker::new(ctx.clone())
        .bake_color_cube(&g, FACE, &EvalCtx::default())
        .expect("bake_color_cube");
    let img = crate::read_rgba8_layers(&ctx, &cube.texture, (FACE, FACE, 6));

    let mut worst = 0u8;
    for face in 0..6 {
        for y in 0..FACE {
            for x in 0..FACE {
                let (u, v) = ((x as f32 + 0.5) / FACE as f32, (y as f32 + 0.5) / FACE as f32);
                let want =
                    to_srgb8(eval::evaluate(&g, out, cube_sample(face, u, v), &EvalCtx::default()));
                let got = img.pixel(x, face * FACE + y).unwrap();
                for c in 0..3 {
                    worst = worst.max(got[c].abs_diff(want[c]));
                }
            }
        }
    }
    // The GPU reads the palette from texels and interpolates between them.
    assert!(worst <= 3, "worst channel delta over the sphere: {worst}");
}

/// Two Transforms reaching one noise would need it baked twice.
#[test]
fn a_sphere_bake_refuses_a_layer_moved_two_ways() {
    let mut g = Graph::new();
    let n = g.add_layer("n", LayerKind::Noise(Noise::default())).unwrap();
    let moved = |g: &mut Graph, name, s| {
        g.add_layer(
            name,
            LayerKind::Transform(Transform {
                source: Some(n),
                offset: [0.0; 3],
                rotate_uv: 0.0,
                scale: [s; 3],
                coord_mode: CoordMode::Passthrough,
                edge_mode: EdgeMode::Extend,
            }),
        )
        .unwrap()
    };
    let a = moved(&mut g, "a", 2.0);
    let b = moved(&mut g, "b", 3.0);
    let both = blend(&mut g, "both", a, b, 0.5);
    let ctx = pollster::block_on(DeviceCtx::request_headless()).expect("headless");
    let refused = Baker::new(ctx)
        .bake_scalar_cube(&g, both, 8, ScalarFormat::R8Unorm, &EvalCtx::default())
        .err();
    assert!(matches!(refused, Some(BakeError::Unsupported(_))), "got {refused:?}");
}

/// A constant reads no position, so feeding a moved and an unmoved branch
/// is one bake, not two. The earthlike clouds share their `zero` this way.
#[test]
fn a_constant_can_feed_moved_and_unmoved_branches() {
    let mut g = Graph::new();
    let zero = g.add_layer("zero", LayerKind::Color(oklcha(0.0, 0.0, 0.0, 1.0))).unwrap();
    let n = g.add_layer("n", LayerKind::Noise(Noise::default())).unwrap();
    let weather = blend(&mut g, "weather", zero, n, 0.5);
    let moved = g
        .add_layer(
            "moved",
            LayerKind::Transform(Transform {
                source: Some(weather),
                offset: [0.0, 0.5, 0.0],
                rotate_uv: 0.0,
                scale: [1.0, 1.8, 1.0],
                coord_mode: CoordMode::Passthrough,
                edge_mode: EdgeMode::Extend,
            }),
        )
        .unwrap();
    let out = blend(&mut g, "out", zero, moved, 0.5);
    let ctx = pollster::block_on(DeviceCtx::request_headless()).expect("headless");
    Baker::new(ctx)
        .bake_scalar_cube(&g, out, 8, ScalarFormat::R8Unorm, &EvalCtx::default())
        .expect("one placement for the constant");
}
