//! Readback tests for the 3D scene renderer's orientation and lighting.
//!
//! Shapes are textured red for u < 0.5 and green above. An inside-out mesh
//! mirrors the split on a cube or shows the wrong hemisphere on a sphere;
//! broken lighting drops lit faces to ambient.

use texture_graph_core::{
    Color, EvalCtx, Graph, LayerKind, Noise, NoiseDims, NoiseOutput, NoiseRange,
};

use crate::baker::{BakeOutput, VolumeOutput};
use crate::scene::{SceneCamera, SceneMaterial, SceneRenderer, SceneShape};
use crate::{Baker, DeviceCtx};

const SIZE: u32 = 128;

fn make_tex(ctx: &DeviceCtx, label: &str, f: impl Fn(u32, u32) -> [u8; 4]) -> wgpu::Texture {
    const W: u32 = 64;
    let tex = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width: W, height: W, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let mut data = Vec::with_capacity((W * W * 4) as usize);
    for y in 0..W {
        for x in 0..W {
            data.extend_from_slice(&f(x, y));
        }
    }
    ctx.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &data,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(W * 4),
            rows_per_image: Some(W),
        },
        wgpu::Extent3d { width: W, height: W, depth_or_array_layers: 1 },
    );
    tex
}

fn split_material(ctx: &DeviceCtx) -> BakeOutput {
    BakeOutput {
        color: make_tex(ctx, "test-color", |x, _| {
            if x < 32 { [255, 0, 0, 255] } else { [0, 255, 0, 255] }
        }),
        roughness: make_tex(ctx, "test-rough", |_, _| [128, 128, 128, 255]),
        metallic: make_tex(ctx, "test-metal", |_, _| [0, 0, 0, 255]),
        normal: make_tex(ctx, "test-normal", |_, _| [128, 128, 255, 255]),
        size: (64, 64),
    }
}

fn make_volume(
    ctx: &DeviceCtx,
    label: &str,
    res: u32,
    f: impl Fn(u32, u32, u32) -> [u8; 4],
) -> wgpu::Texture {
    let tex = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width: res, height: res, depth_or_array_layers: res },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let mut data = Vec::with_capacity((res * res * res * 4) as usize);
    for z in 0..res {
        for y in 0..res {
            for x in 0..res {
                data.extend_from_slice(&f(x, y, z));
            }
        }
    }
    ctx.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &data,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(res * 4),
            rows_per_image: Some(res),
        },
        wgpu::Extent3d { width: res, height: res, depth_or_array_layers: res },
    );
    tex
}

fn solid_split_material(ctx: &DeviceCtx) -> VolumeOutput {
    const R: u32 = 32;
    VolumeOutput {
        color: make_volume(ctx, "vol-color", R, |x, _, _| {
            if x < R / 2 { [255, 0, 0, 255] } else { [0, 255, 0, 255] }
        }),
        roughness: make_volume(ctx, "vol-rough", R, |_, _, _| [128, 128, 128, 255]),
        metallic: make_volume(ctx, "vol-metal", R, |_, _, _| [0, 0, 0, 255]),
        normal: make_volume(ctx, "vol-normal", R, |_, _, _| [128, 128, 255, 255]),
        size: (R, R, R),
    }
}

/// Camera fixed on +Z; `yaw` turns the model.
fn render_frame_with(
    ctx: &DeviceCtx,
    shape: SceneShape,
    yaw: f32,
    material: SceneMaterial<'_>,
) -> Vec<[u8; 4]> {
    let scene = SceneRenderer::new(&ctx.device);

    let color = scene.make_color_target(&ctx.device, (SIZE, SIZE));
    let depth = scene.make_depth_target(&ctx.device, (SIZE, SIZE));
    let color_view = color.create_view(&wgpu::TextureViewDescriptor::default());
    let depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());

    let camera = SceneCamera {
        orientation: glam::Quat::from_rotation_y(yaw),
        ..SceneCamera::default()
    };
    scene.render_into(ctx, material, shape, &color_view, &depth_view, (SIZE, SIZE), &camera);

    // 128 px * 4 B = 512 B rows, already 256-aligned.
    let bytes_per_row = SIZE * 4;
    let readback = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("scene-readback"),
        size: (bytes_per_row * SIZE) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("scene-read-enc") });
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &color,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(SIZE),
            },
        },
        wgpu::Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
    );
    ctx.queue.submit([enc.finish()]);

    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    ctx.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    rx.recv().expect("map channel").expect("map");
    let data = slice.get_mapped_range();
    let pixels: Vec<[u8; 4]> = data
        .chunks_exact(4)
        .map(|c| [c[0], c[1], c[2], c[3]])
        .collect();
    drop(data);
    readback.unmap();
    pixels
}

fn render_frame(ctx: &DeviceCtx, shape: SceneShape, yaw: f32) -> Vec<[u8; 4]> {
    let material = split_material(ctx);
    render_frame_with(ctx, shape, yaw, SceneMaterial::Uv(&material))
}

fn px(frame: &[[u8; 4]], x: u32, y: u32) -> [u8; 4] {
    frame[(y * SIZE + x) as usize]
}

#[test]
fn cube_front_face_is_lit_and_not_mirrored() {
    let ctx = pollster::block_on(DeviceCtx::request_headless()).expect("headless wgpu init");
    let frame = render_frame(&ctx, SceneShape::Cube, 0.0);

    let c = SIZE / 2;
    let left = px(&frame, SIZE / 2 - SIZE / 8, c);
    let right = px(&frame, SIZE / 2 + SIZE / 8, c);
    let center = px(&frame, c, c);
    eprintln!("cube: left={left:?} right={right:?} center={center:?}");

    // The camera is on +Z at yaw 0 and the +Z face has u increasing with +X,
    // so red (u < 0.5) belongs on the left. Inside-out shows the -Z face
    // through the cube and mirrors the split.
    assert!(
        left[0] > left[1] + 40,
        "left of cube face should be red-dominant, got {left:?} (mirrored ⇒ inside-out winding)"
    );
    assert!(
        right[1] > right[0] + 40,
        "right of cube face should be green-dominant, got {right:?} (mirrored ⇒ inside-out winding)"
    );

    // Key + fill give about 2.8× white on this face, well above 120 after
    // Reinhard and gamma. Ambient alone is below 60.
    let brightness = left[0].max(right[1]);
    assert!(
        brightness > 120,
        "front face should be brightly lit, peak channel {brightness} (dark ⇒ lighting/normals bug)"
    );
}

#[test]
fn facing_surface_stays_lit_through_full_turntable_spin() {
    // The lights are fixed to the camera, so whichever face turns toward
    // the viewer must stay lit.
    let ctx = pollster::block_on(DeviceCtx::request_headless()).expect("headless wgpu init");
    let c = SIZE / 2;
    for step in 0..8 {
        let yaw = step as f32 * std::f32::consts::TAU / 8.0;
        for shape in [SceneShape::Cube, SceneShape::Sphere] {
            let frame = render_frame(&ctx, shape, yaw);
            let p = px(&frame, c, c);
            let peak = p[0].max(p[1]).max(p[2]);
            eprintln!("{shape:?} yaw={yaw:.2}: center={p:?}");
            assert!(
                peak > 120,
                "{shape:?} at yaw {yaw:.2} should be lit, center {p:?} (dark ⇒ \
                 light rig not camera-relative)"
            );
        }
    }
}

#[test]
fn sphere_shows_near_hemisphere_with_smooth_shading() {
    let ctx = pollster::block_on(DeviceCtx::request_headless()).expect("headless wgpu init");
    let frame = render_frame(&ctx, SceneShape::Sphere, 0.0);

    let c = SIZE / 2;
    let center = px(&frame, c, c);
    eprintln!("sphere: center={center:?}");
    let scan: Vec<[u8; 4]> = (0..8).map(|i| px(&frame, SIZE / 8 * i + SIZE / 16, c)).collect();
    eprintln!("sphere scanline: {scan:?}");

    // theta = 0 is at +X, so from +Z the near hemisphere is u ∈ (0, 0.5),
    // all red. Green means inside-out winding.
    assert!(
        center[0] > center[1] + 40,
        "sphere center should be red (near hemisphere), got {center:?} (green ⇒ inside-out winding)"
    );

    assert!(
        center[0] > 100,
        "sphere center should be lit, got {center:?}"
    );

    // Conical shading artifacts show as a jump between adjacent samples.
    let vals: Vec<i32> = (40..88).map(|x| px(&frame, x, c)[0] as i32).collect();
    let max_jump = vals.windows(2).map(|w| (w[1] - w[0]).abs()).max().unwrap();
    eprintln!("sphere equator max adjacent-pixel jump: {max_jump}");
    assert!(
        max_jump < 25,
        "equator shading should be smooth, max adjacent jump {max_jump}"
    );
}

#[test]
fn solid_material_samples_by_object_position() {
    // Object-space -X (tex x < 0.5, red) is screen-left from +Z.
    let ctx = pollster::block_on(DeviceCtx::request_headless()).expect("headless wgpu init");
    let volume = solid_split_material(&ctx);
    let c = SIZE / 2;
    for shape in [SceneShape::Sphere, SceneShape::Cube] {
        let frame = render_frame_with(&ctx, shape, 0.0, SceneMaterial::Solid(&volume));
        let left = px(&frame, SIZE / 2 - SIZE / 8, c);
        let right = px(&frame, SIZE / 2 + SIZE / 8, c);
        eprintln!("solid {shape:?}: left={left:?} right={right:?}");
        assert!(
            left[0] > left[1] + 40,
            "solid {shape:?} left should be red (obj x < 0), got {left:?}"
        );
        assert!(
            right[1] > right[0] + 40,
            "solid {shape:?} right should be green (obj x > 0), got {right:?}"
        );
        assert!(
            left[0].max(right[1]) > 120,
            "solid {shape:?} should be lit, got left={left:?} right={right:?}"
        );
    }
}

/// Read one z-slice of a 3D Rgba8 texture. `res * 4` must be 256-aligned.
fn read_volume_slice(ctx: &DeviceCtx, vol: &wgpu::Texture, res: u32, z: u32) -> Vec<u8> {
    let bytes_per_row = res * 4;
    assert_eq!(bytes_per_row % 256, 0, "test resolution must keep rows 256-aligned");
    let buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vol-slice-readback"),
        size: (bytes_per_row * res) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("vol-read-enc") });
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: vol,
            mip_level: 0,
            origin: wgpu::Origin3d { x: 0, y: 0, z },
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buf,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(res),
            },
        },
        wgpu::Extent3d { width: res, height: res, depth_or_array_layers: 1 },
    );
    ctx.queue.submit([enc.finish()]);
    let slice = buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    ctx.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    rx.recv().expect("map channel").expect("map");
    let data = slice.get_mapped_range().to_vec();
    buf.unmap();
    data
}

fn noise_graph(dims: NoiseDims) -> Graph {
    let mut graph = Graph::new();
    let base = graph.output.color;
    graph
        .set_kind(
            base.unwrap(),
            LayerKind::Noise(Noise {
                dims,
                seed_offset: 0,
                frequency: 4.0,
                range: NoiseRange::Unsigned,
                output: NoiseOutput::Grayscale,
                ..Noise::default()
            }),
        )
        .unwrap();
    graph
}

#[test]
fn volume_bake_varies_along_w_only_for_3d_graphs() {
    let ctx = pollster::block_on(DeviceCtx::request_headless()).expect("headless wgpu init");
    let mut baker = Baker::new(ctx.clone());
    const RES: u32 = 64; // 64 * 4 B = 256 B rows, copy-aligned
    // Odd, so the middle slice is at w = (31 + 0.5) / 63 = 0.5, as the flat
    // bake below.
    const DEPTH: u32 = 63;
    const MID: u32 = DEPTH / 2;

    let g3 = noise_graph(NoiseDims::D3);
    assert!(g3.output_is_3d(), "D3-noise graph should report as 3D");
    let vol = baker.bake_volume(&g3, RES, DEPTH, &EvalCtx::default()).expect("bake 3d volume");
    let near = read_volume_slice(&ctx, &vol.color, RES, 0);
    let mid = read_volume_slice(&ctx, &vol.color, RES, MID);
    assert_ne!(near, mid, "3D noise slices at different w should differ");

    let g2 = noise_graph(NoiseDims::D2);
    assert!(!g2.output_is_3d(), "D2-noise graph should NOT report as 3D");
    let vol2 = baker.bake_volume(&g2, RES, DEPTH, &EvalCtx::default()).expect("bake 2d volume");
    let near2 = read_volume_slice(&ctx, &vol2.color, RES, 0);
    let far2 = read_volume_slice(&ctx, &vol2.color, RES, MID);
    assert_eq!(near2, far2, "2D noise slices should be identical at every w");

    let flat = baker.bake_output(&g3, (RES, RES), &EvalCtx::default(), false).expect("flat bake");
    let flat_px = read_texture_2d(&ctx, &flat.color, RES);
    assert_eq!(flat_px, mid, "volume center slice should match the flat (w=0.5) bake");
}

#[test]
fn background_is_flat_transparency_checker() {
    // 8-px cells, 0.75/0.55 sRGB → bytes 191/140.
    let ctx = pollster::block_on(DeviceCtx::request_headless()).expect("headless wgpu init");
    let frame = render_frame(&ctx, SceneShape::Sphere, 0.0);
    let a = px(&frame, 3, 3);
    let b = px(&frame, 11, 3);
    eprintln!("background cells: {a:?} {b:?}");
    for (name, p) in [("a", a), ("b", b)] {
        assert!(
            p[0] == p[1] && p[1] == p[2],
            "background cell {name} should be gray, got {p:?}"
        );
        assert!(
            (135..=195).contains(&p[0]),
            "background cell {name} should be checker gray, got {p:?}"
        );
    }
    assert!(
        a[0].abs_diff(b[0]) > 30,
        "adjacent 8-px cells should alternate light/dark, got {a:?} vs {b:?}"
    );
}

#[test]
fn object_alpha_switch_keeps_real_alpha_in_volume_bakes() {
    // The flat bake composites over the checker; the volume bake keeps alpha.
    let ctx = pollster::block_on(DeviceCtx::request_headless()).expect("headless wgpu init");
    let mut baker = Baker::new(ctx.clone());
    const RES: u32 = 64;

    let mut graph = Graph::new();
    let base = graph.output.color;
    graph
        .set_kind(base.unwrap(), LayerKind::Color(Color::new(0.6, 0.1, 40.0, 0.5)))
        .unwrap();

    let flat = baker
        .bake_output(&graph, (RES, RES), &EvalCtx::default(), false)
        .expect("flat bake");
    let flat_px = read_texture_2d(&ctx, &flat.color, RES);
    assert_eq!(flat_px[3], 255, "flat bake composites the checker to alpha=1");

    let vol = baker.bake_volume(&graph, RES, 8, &EvalCtx::default()).expect("volume bake");
    let vol_px = read_volume_slice(&ctx, &vol.color, RES, 0);
    let a = vol_px[3] as i32;
    assert!(
        (a - 128).abs() <= 2,
        "volume bake keeps real alpha (~128), got {a}"
    );
    // No checker mixed in: neighboring 8-px cells must match.
    assert_eq!(
        &vol_px[0..4],
        &vol_px[(8 * 4) as usize..(8 * 4 + 4) as usize],
        "no alpha-backing checker in object-alpha output"
    );
}

#[test]
fn out_of_range_checker_alternates_along_w() {
    // D2 noise is the same on every slice, so slices differ only by the
    // out-of-range checker's w axis: with 10-texel cells, slices 0 and 10
    // must differ and 0 and 20 must match.
    let ctx = pollster::block_on(DeviceCtx::request_headless()).expect("headless wgpu init");
    let mut baker = Baker::new(ctx.clone());
    const RES: u32 = 64;

    let mut graph = Graph::new();
    let base = graph.output.color;
    graph
        .set_kind(
            base.unwrap(),
            LayerKind::Noise(Noise {
                dims: NoiseDims::D2,
                seed_offset: 0,
                frequency: 4.0,
                range: NoiseRange::Signed,
                output: NoiseOutput::Grayscale,
                ..Noise::default()
            }),
        )
        .unwrap();

    let vol = baker.bake_volume(&graph, RES, 30, &EvalCtx::default()).expect("bake");
    let s0 = read_volume_slice(&ctx, &vol.color, RES, 0);
    let s10 = read_volume_slice(&ctx, &vol.color, RES, 10);
    let s20 = read_volume_slice(&ctx, &vol.color, RES, 20);
    assert_ne!(s0, s10, "checker parity should flip between z-cells");
    assert_eq!(s0, s20, "checker parity should restore two z-cells later");
}

/// Read a full 2D Rgba8 texture. `res * 4` must be 256-aligned.
fn read_texture_2d(ctx: &DeviceCtx, tex: &wgpu::Texture, res: u32) -> Vec<u8> {
    let bytes_per_row = res * 4;
    assert_eq!(bytes_per_row % 256, 0);
    let buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tex2d-readback"),
        size: (bytes_per_row * res) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("tex2d-read-enc") });
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buf,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(res),
            },
        },
        wgpu::Extent3d { width: res, height: res, depth_or_array_layers: 1 },
    );
    ctx.queue.submit([enc.finish()]);
    let slice = buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    ctx.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    rx.recv().expect("map channel").expect("map");
    let data = slice.get_mapped_range().to_vec();
    buf.unmap();
    data
}
