//! GPU compute-shader baker.
//!
//! The Baker owns the wgpu device handles, a cached pool of `Rgba32Float`
//! intermediate textures sized to the current output, and one compute
//! pipeline per LayerKind variant. It walks the schedule from `schedule.rs`
//! and dispatches one shader per layer, then packs the four PBR channels
//! into `Rgba8Unorm` textures the UI can register with egui-wgpu.
//!
//! **Storage note.** wgpu does not permit storage bindings to sRGB view
//! formats, so packed outputs are `Rgba8Unorm` containing sRGB-encoded
//! values. `pack_srgb8.wgsl` applies the gamma manually to match
//! `core::color::to_srgb8`.

use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};
use texture_graph_core::{
    EvalCtx, Graph, LayerId, LayerKind, Noise, NoiseDims, NoiseOutput, NoiseRange,
};

use crate::device::DeviceCtx;
use crate::schedule::{OutputSlots, ScalarSlot, schedule};

/// Four sRGB-encoded 8-bit-per-channel textures ready for display.
pub struct BakeOutput {
    pub color: wgpu::Texture,
    pub roughness: wgpu::Texture,
    pub metallic: wgpu::Texture,
    pub normal: wgpu::Texture,
    pub size: (u32, u32),
}

#[derive(Debug)]
pub enum BakeError {
    Schedule(crate::schedule::ScheduleError),
    Unsupported(&'static str),
}

impl std::fmt::Display for BakeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BakeError::Schedule(e) => write!(f, "schedule: {e}"),
            BakeError::Unsupported(k) => write!(f, "gpu pipeline not implemented for {k}"),
        }
    }
}

impl std::error::Error for BakeError {}

impl From<crate::schedule::ScheduleError> for BakeError {
    fn from(e: crate::schedule::ScheduleError) -> Self {
        BakeError::Schedule(e)
    }
}

pub struct Baker {
    ctx: DeviceCtx,
    pool: Vec<wgpu::Texture>,
    pool_views: Vec<wgpu::TextureView>,
    pool_size: (u32, u32),

    color_pipeline: wgpu::ComputePipeline,
    color_bgl: wgpu::BindGroupLayout,
    noise_pipeline: wgpu::ComputePipeline,
    // noise reuses `color_bgl`: same binding shape (uniform + storage_texture).
    pack_pipeline: wgpu::ComputePipeline,
    pack_bgl: wgpu::BindGroupLayout,
    solid_pipeline: wgpu::ComputePipeline,
    solid_bgl: wgpu::BindGroupLayout,
}

impl Baker {
    pub fn new(ctx: DeviceCtx) -> Self {
        let (color_pipeline, color_bgl) = make_color_pipeline(&ctx.device);
        let noise_pipeline = make_noise_pipeline(&ctx.device, &color_bgl);
        let (pack_pipeline, pack_bgl) = make_pack_pipeline(&ctx.device);
        let (solid_pipeline, solid_bgl) = make_solid_pipeline(&ctx.device);
        Self {
            ctx,
            pool: Vec::new(),
            pool_views: Vec::new(),
            pool_size: (0, 0),
            color_pipeline,
            color_bgl,
            noise_pipeline,
            pack_pipeline,
            pack_bgl,
            solid_pipeline,
            solid_bgl,
        }
    }

    pub fn ctx(&self) -> &DeviceCtx {
        &self.ctx
    }

    fn ensure_pool(&mut self, size: (u32, u32), needed: u32) {
        let resize = self.pool_size != size;
        if resize {
            self.pool.clear();
            self.pool_views.clear();
            self.pool_size = size;
        }
        while self.pool.len() < needed as usize {
            let tex = self.ctx.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("tg-pool"),
                size: wgpu::Extent3d {
                    width: size.0,
                    height: size.1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba32Float,
                usage: wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            self.pool.push(tex);
            self.pool_views.push(view);
        }
    }

    /// Bake the graph's output at `size`. Only variants with pipelines
    /// implemented so far are supported; others return `Unsupported`.
    pub fn bake_output(
        &mut self,
        graph: &Graph,
        size: (u32, u32),
        _ctx: &EvalCtx,
    ) -> Result<BakeOutput, BakeError> {
        let sched = schedule(graph)?;
        self.ensure_pool(size, sched.peak_slots.max(1));

        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("tg-bake-output"),
            });

        // Dispatch each layer.
        for &id in &sched.order {
            let slot = *sched.slot_of.get(&id).unwrap() as usize;
            let layer = graph.get(id).unwrap();
            match &layer.kind {
                LayerKind::Color(c) => {
                    dispatch_color(
                        &self.ctx,
                        &mut encoder,
                        &self.color_pipeline,
                        &self.color_bgl,
                        &self.pool_views[slot],
                        [c.l, c.chroma, c.hue.into_degrees(), c.alpha],
                        size,
                    );
                }
                LayerKind::Noise(n) => {
                    dispatch_noise(
                        &self.ctx,
                        &mut encoder,
                        &self.noise_pipeline,
                        &self.color_bgl,
                        &self.pool_views[slot],
                        n,
                        _ctx.seed,
                        size,
                    );
                }
                other => {
                    return Err(BakeError::Unsupported(other.category_label()));
                }
            }
        }

        // Pack the four output channels.
        let color = make_output_texture(&self.ctx.device, size, "tg-color");
        let roughness = make_output_texture(&self.ctx.device, size, "tg-rough");
        let metallic = make_output_texture(&self.ctx.device, size, "tg-metal");
        let normal = make_output_texture(&self.ctx.device, size, "tg-normal");

        pack_channel(
            &self.ctx,
            &mut encoder,
            &self.pack_pipeline,
            &self.pack_bgl,
            &self.solid_pipeline,
            &self.solid_bgl,
            &self.pool_views,
            &color.create_view(&wgpu::TextureViewDescriptor::default()),
            size,
            OutputChannel::Color,
            &sched.output_slots,
        );
        pack_channel(
            &self.ctx,
            &mut encoder,
            &self.pack_pipeline,
            &self.pack_bgl,
            &self.solid_pipeline,
            &self.solid_bgl,
            &self.pool_views,
            &roughness.create_view(&wgpu::TextureViewDescriptor::default()),
            size,
            OutputChannel::Roughness,
            &sched.output_slots,
        );
        pack_channel(
            &self.ctx,
            &mut encoder,
            &self.pack_pipeline,
            &self.pack_bgl,
            &self.solid_pipeline,
            &self.solid_bgl,
            &self.pool_views,
            &metallic.create_view(&wgpu::TextureViewDescriptor::default()),
            size,
            OutputChannel::Metallic,
            &sched.output_slots,
        );
        pack_channel(
            &self.ctx,
            &mut encoder,
            &self.pack_pipeline,
            &self.pack_bgl,
            &self.solid_pipeline,
            &self.solid_bgl,
            &self.pool_views,
            &normal.create_view(&wgpu::TextureViewDescriptor::default()),
            size,
            OutputChannel::Normal,
            &sched.output_slots,
        );

        self.ctx.queue.submit([encoder.finish()]);
        Ok(BakeOutput { color, roughness, metallic, normal, size })
    }
}

// ---- Dispatch helpers --------------------------------------------------

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ColorParams {
    color: [f32; 4],
    size: [u32; 2],
    _pad: [u32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct NoiseParams {
    size: [u32; 2],
    dims: u32,
    range: u32,
    output_mode: u32,
    seed_base: u32,
    frequency: f32,
    _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct PackParams {
    size: [u32; 2],
    mode: u32,
    _pad: u32,
    const_value: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct SolidParams {
    color: [f32; 4],
    size: [u32; 2],
    _pad: [u32; 2],
}

fn dispatch_noise(
    ctx: &DeviceCtx,
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    bgl: &wgpu::BindGroupLayout,
    dst_view: &wgpu::TextureView,
    n: &Noise,
    ctx_seed: u32,
    size: (u32, u32),
) {
    let dims = match n.dims {
        NoiseDims::D1 => 0u32,
        NoiseDims::D2 => 1u32,
        NoiseDims::D3 => 2u32,
    };
    let range = match n.range {
        NoiseRange::Unsigned => 0u32,
        NoiseRange::Signed => 1u32,
    };
    let output_mode = match n.output {
        NoiseOutput::Grayscale => 0u32,
        NoiseOutput::Color => 1u32,
    };
    let params = NoiseParams {
        size: [size.0, size.1],
        dims,
        range,
        output_mode,
        seed_base: ctx_seed.wrapping_add(n.seed_offset),
        frequency: n.frequency,
        _pad: 0,
    };
    let ubo = create_uniform(&ctx.device, bytemuck::bytes_of(&params), "noise-params");
    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("noise-bg"),
        layout: bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: ubo.as_entire_binding() },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(dst_view),
            },
        ],
    });
    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("noise-cpass"),
        timestamp_writes: None,
    });
    cpass.set_pipeline(pipeline);
    cpass.set_bind_group(0, &bg, &[]);
    let (wg_x, wg_y) = workgroup_counts(size);
    cpass.dispatch_workgroups(wg_x, wg_y, 1);
}

fn dispatch_color(
    ctx: &DeviceCtx,
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    bgl: &wgpu::BindGroupLayout,
    dst_view: &wgpu::TextureView,
    color: [f32; 4],
    size: (u32, u32),
) {
    let params = ColorParams { color, size: [size.0, size.1], _pad: [0, 0] };
    let ubo = create_uniform(&ctx.device, bytemuck::bytes_of(&params), "color-params");
    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("color-bg"),
        layout: bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: ubo.as_entire_binding() },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(dst_view),
            },
        ],
    });
    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("color-cpass"),
        timestamp_writes: None,
    });
    cpass.set_pipeline(pipeline);
    cpass.set_bind_group(0, &bg, &[]);
    let (wg_x, wg_y) = workgroup_counts(size);
    cpass.dispatch_workgroups(wg_x, wg_y, 1);
}

#[derive(Copy, Clone)]
enum OutputChannel {
    Color,
    Roughness,
    Metallic,
    Normal,
}

fn pack_channel(
    ctx: &DeviceCtx,
    encoder: &mut wgpu::CommandEncoder,
    pack_pipeline: &wgpu::ComputePipeline,
    pack_bgl: &wgpu::BindGroupLayout,
    solid_pipeline: &wgpu::ComputePipeline,
    solid_bgl: &wgpu::BindGroupLayout,
    pool_views: &[wgpu::TextureView],
    dst_view: &wgpu::TextureView,
    size: (u32, u32),
    channel: OutputChannel,
    out: &OutputSlots,
) {
    match channel {
        OutputChannel::Color => dispatch_pack(
            ctx, encoder, pack_pipeline, pack_bgl,
            &pool_views[out.color as usize], dst_view, size, 0, [0.0; 4],
        ),
        OutputChannel::Roughness => match out.roughness {
            ScalarSlot::Const(v) => dispatch_solid(
                ctx, encoder, solid_pipeline, solid_bgl, dst_view, size,
                [srgb_of_linear_component(v), srgb_of_linear_component(v),
                 srgb_of_linear_component(v), 1.0],
            ),
            ScalarSlot::Slot(s) => dispatch_pack(
                ctx, encoder, pack_pipeline, pack_bgl,
                &pool_views[s as usize], dst_view, size, 2, [0.0; 4],
            ),
        },
        OutputChannel::Metallic => match out.metallic {
            ScalarSlot::Const(v) => dispatch_solid(
                ctx, encoder, solid_pipeline, solid_bgl, dst_view, size,
                [srgb_of_linear_component(v), srgb_of_linear_component(v),
                 srgb_of_linear_component(v), 1.0],
            ),
            ScalarSlot::Slot(s) => dispatch_pack(
                ctx, encoder, pack_pipeline, pack_bgl,
                &pool_views[s as usize], dst_view, size, 2, [0.0; 4],
            ),
        },
        OutputChannel::Normal => match out.normal {
            None => dispatch_solid(
                ctx, encoder, solid_pipeline, solid_bgl, dst_view, size,
                [0.5, 0.5, 1.0, 1.0],
            ),
            Some(s) => dispatch_pack(
                ctx, encoder, pack_pipeline, pack_bgl,
                &pool_views[s as usize], dst_view, size, 3, [0.0; 4],
            ),
        },
    }
}

fn dispatch_pack(
    ctx: &DeviceCtx,
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    bgl: &wgpu::BindGroupLayout,
    src_view: &wgpu::TextureView,
    dst_view: &wgpu::TextureView,
    size: (u32, u32),
    mode: u32,
    const_value: [f32; 4],
) {
    let params = PackParams {
        size: [size.0, size.1],
        mode,
        _pad: 0,
        const_value,
    };
    let ubo = create_uniform(&ctx.device, bytemuck::bytes_of(&params), "pack-params");
    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("pack-bg"),
        layout: bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: ubo.as_entire_binding() },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(src_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(dst_view),
            },
        ],
    });
    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("pack-cpass"),
        timestamp_writes: None,
    });
    cpass.set_pipeline(pipeline);
    cpass.set_bind_group(0, &bg, &[]);
    let (wg_x, wg_y) = workgroup_counts(size);
    cpass.dispatch_workgroups(wg_x, wg_y, 1);
}

fn dispatch_solid(
    ctx: &DeviceCtx,
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    bgl: &wgpu::BindGroupLayout,
    dst_view: &wgpu::TextureView,
    size: (u32, u32),
    color: [f32; 4],
) {
    let params = SolidParams { color, size: [size.0, size.1], _pad: [0, 0] };
    let ubo = create_uniform(&ctx.device, bytemuck::bytes_of(&params), "solid-params");
    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("solid-bg"),
        layout: bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: ubo.as_entire_binding() },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(dst_view),
            },
        ],
    });
    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("solid-cpass"),
        timestamp_writes: None,
    });
    cpass.set_pipeline(pipeline);
    cpass.set_bind_group(0, &bg, &[]);
    let (wg_x, wg_y) = workgroup_counts(size);
    cpass.dispatch_workgroups(wg_x, wg_y, 1);
}

fn create_uniform(device: &wgpu::Device, bytes: &[u8], label: &str) -> wgpu::Buffer {
    use wgpu::util::DeviceExt;
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytes,
        usage: wgpu::BufferUsages::UNIFORM,
    })
}

fn workgroup_counts(size: (u32, u32)) -> (u32, u32) {
    ((size.0 + 7) / 8, (size.1 + 7) / 8)
}

fn make_output_texture(device: &wgpu::Device, size: (u32, u32), label: &str) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: size.0,
            height: size.1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

/// CPU-side sRGB gamma of one 0..=1 linear-light component. Used to encode
/// constant scalar output channels so they visually match the shader path.
fn srgb_of_linear_component(x: f32) -> f32 {
    let clamped = x.clamp(0.0, 1.0);
    if clamped <= 0.0031308 {
        12.92 * clamped
    } else {
        1.055 * clamped.powf(1.0 / 2.4) - 0.055
    }
}

// ---- Pipeline factories ------------------------------------------------

fn make_color_pipeline(
    device: &wgpu::Device,
) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("color-bgl"),
        entries: &[
            uniform_bgle(0),
            storage_texture_bgle(1, wgpu::TextureFormat::Rgba32Float),
        ],
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("color-pl"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("color-shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/color.wgsl").into()),
    });
    let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("color-pipeline"),
        layout: Some(&pl),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    (pipe, bgl)
}

fn make_noise_pipeline(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
) -> wgpu::ComputePipeline {
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("noise-pl"),
        bind_group_layouts: &[bgl],
        push_constant_ranges: &[],
    });
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("noise-shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/noise.wgsl").into()),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("noise-pipeline"),
        layout: Some(&pl),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn make_pack_pipeline(
    device: &wgpu::Device,
) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("pack-bgl"),
        entries: &[
            uniform_bgle(0),
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            storage_texture_bgle(2, wgpu::TextureFormat::Rgba8Unorm),
        ],
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("pack-pl"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("pack-shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/pack_srgb8.wgsl").into()),
    });
    let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("pack-pipeline"),
        layout: Some(&pl),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    (pipe, bgl)
}

fn make_solid_pipeline(
    device: &wgpu::Device,
) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("solid-bgl"),
        entries: &[
            uniform_bgle(0),
            storage_texture_bgle(1, wgpu::TextureFormat::Rgba8Unorm),
        ],
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("solid-pl"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("solid-shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/solid.wgsl").into()),
    });
    let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("solid-pipeline"),
        layout: Some(&pl),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    (pipe, bgl)
}

fn uniform_bgle(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_texture_bgle(
    binding: u32,
    format: wgpu::TextureFormat,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    }
}

// Silence unused-import warning while other imports are pending future variants.
#[allow(dead_code)]
fn _sink(_: HashMap<LayerId, ()>) {}
