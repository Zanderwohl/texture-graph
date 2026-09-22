//! Checks wgpu init, a compute dispatch and readback: a constant Oklcha
//! value must round-trip exactly through an Rgba32Float texture.

use bytemuck::{Pod, Zeroable};

use crate::DeviceCtx;

const TEX_SIZE: u32 = 8;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Params {
    color: [f32; 4],
    size: [u32; 2],
    _pad: [u32; 2],
}

fn run_smoke() -> [f32; 4] {
    let ctx = pollster::block_on(DeviceCtx::request_headless())
        .expect("headless wgpu init");
    let expected = [0.5f32, 0.1, 30.0, 1.0];

    let tex = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("smoke-out"),
        size: wgpu::Extent3d { width: TEX_SIZE, height: TEX_SIZE, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let tex_view = tex.create_view(&wgpu::TextureViewDescriptor::default());

    let params = Params { color: expected, size: [TEX_SIZE, TEX_SIZE], _pad: [0, 0] };
    let uniform = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("smoke-params"),
        size: std::mem::size_of::<Params>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    ctx.queue.write_buffer(&uniform, 0, bytemuck::bytes_of(&params));

    let shader = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("smoke-shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/smoke.wgsl").into()),
    });

    let bgl = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("smoke-bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba32Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
        ],
    });
    let pl_layout = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("smoke-pl"),
        bind_group_layouts: &[Some(&bgl)],
        ..Default::default()
    });
    let pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("smoke-pipeline"),
        layout: Some(&pl_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("smoke-bg"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&tex_view) },
        ],
    });

    // Copies need 256-byte-aligned rows; 8 px * 16 B = 128 B.
    let bytes_per_row = 256u32;
    let readback = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("smoke-readback"),
        size: (bytes_per_row * TEX_SIZE) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("smoke-enc"),
    });
    {
        let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("smoke-cpass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bg, &[]);
        cpass.dispatch_workgroups(1, 1, 1);
    }
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(TEX_SIZE),
            },
        },
        wgpu::Extent3d { width: TEX_SIZE, height: TEX_SIZE, depth_or_array_layers: 1 },
    );
    ctx.queue.submit([enc.finish()]);

    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
    ctx.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    rx.recv().expect("map channel").expect("map");
    let data = slice.get_mapped_range();
    let pixel: [f32; 4] = bytemuck::from_bytes::<[f32; 4]>(&data[..16]).clone();
    drop(data);
    readback.unmap();
    pixel
}

#[test]
fn smoke_write_solid_oklcha_roundtrips() {
    let got = run_smoke();
    let want = [0.5f32, 0.1, 30.0, 1.0];
    for i in 0..4 {
        assert!(
            (got[i] - want[i]).abs() < 1e-6,
            "channel {i}: got {} want {}",
            got[i], want[i]
        );
    }
}
