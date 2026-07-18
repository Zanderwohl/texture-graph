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
    Axis, BlendMode, BlendSpace, ColorInput, ColorRamp, CoordMode, Criterion, EvalCtx, Graph,
    HeightToNormal, LayerId, LayerKind, MinMax, MinMaxMode, Mix, Noise, NoiseDims, NoiseOutput,
    NoiseRange, RadialDim, ScalarInput, Transform,
};

use crate::device::DeviceCtx;
use crate::schedule::{OutputSlots, ScalarSlot, Schedule, schedule, schedule_no_reuse};

/// Max stops per ColorRamp supported by the GPU baker.
const MAX_RAMP_STOPS: usize = 16;
/// Max distinct layer-referenced stops per ColorRamp. Bindings must be
/// declared at pipeline-creation time, so this is a hard cap.
const MAX_RAMP_INPUTS: usize = 8;

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
    transform_pipeline: wgpu::ComputePipeline,
    transform_bgl: wgpu::BindGroupLayout,
    mix_pipeline: wgpu::ComputePipeline,
    mix_bgl: wgpu::BindGroupLayout,
    map_pipeline: wgpu::ComputePipeline,
    map_bgl: wgpu::BindGroupLayout,
    min_max_pipeline: wgpu::ComputePipeline,
    // min_max reuses `map_bgl` — same binding shape (uniform + storage_out + 2 inputs).
    ramp_pipeline: wgpu::ComputePipeline,
    ramp_bgl: wgpu::BindGroupLayout,
    /// 1×1 Rgba32Float sampled-only texture. Bound into unused ramp input
    /// slots so we never collide with an output storage binding. Kept alive
    /// by the Baker so `dummy_input_view` stays valid.
    #[allow(dead_code)]
    dummy_input: wgpu::Texture,
    dummy_input_view: wgpu::TextureView,
    h2n_pipeline: wgpu::ComputePipeline,
    // h2n reuses `transform_bgl` — same binding shape (uniform + storage_out + input_2d).
    pack_pipeline: wgpu::ComputePipeline,
    pack_bgl: wgpu::BindGroupLayout,
    solid_pipeline: wgpu::ComputePipeline,
    solid_bgl: wgpu::BindGroupLayout,
}

impl Baker {
    pub fn new(ctx: DeviceCtx) -> Self {
        let (color_pipeline, color_bgl) = make_color_pipeline(&ctx.device);
        let noise_pipeline = make_noise_pipeline(&ctx.device, &color_bgl);
        let (transform_pipeline, transform_bgl) = make_transform_pipeline(&ctx.device);
        let (mix_pipeline, mix_bgl) = make_mix_pipeline(&ctx.device);
        let (map_pipeline, map_bgl) = make_map_pipeline(&ctx.device);
        let min_max_pipeline = make_min_max_pipeline(&ctx.device, &map_bgl);
        let (ramp_pipeline, ramp_bgl) = make_ramp_pipeline(&ctx.device);
        let h2n_pipeline = make_h2n_pipeline(&ctx.device, &transform_bgl);
        let dummy_input = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("tg-dummy-input"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            // Sampled only — no STORAGE_BINDING so wgpu can't confuse this
            // with an output slot.
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let dummy_input_view = dummy_input.create_view(&wgpu::TextureViewDescriptor::default());
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
            transform_pipeline,
            transform_bgl,
            mix_pipeline,
            mix_bgl,
            map_pipeline,
            map_bgl,
            min_max_pipeline,
            ramp_pipeline,
            ramp_bgl,
            dummy_input,
            dummy_input_view,
            h2n_pipeline,
            pack_pipeline,
            pack_bgl,
            solid_pipeline,
            solid_bgl,
        }
    }

    pub fn ctx(&self) -> &DeviceCtx {
        &self.ctx
    }

    /// Bake per-layer thumbnails at 128². Every layer gets its own
    /// intermediate slot (no pebble reuse), then a `pack_srgb8` pass packs
    /// each to an `Rgba8Unorm` texture. Returns one texture per layer for
    /// the UI to register with egui-wgpu.
    pub fn bake_previews(
        &mut self,
        graph: &Graph,
        eval_ctx: &EvalCtx,
    ) -> Result<HashMap<LayerId, wgpu::Texture>, BakeError> {
        const PREVIEW_SIZE: (u32, u32) = (128, 128);
        let sched = schedule_no_reuse(graph)?;
        let size = PREVIEW_SIZE;
        let device = self.ctx.device.clone();

        // Fresh intermediate textures — one per layer, alive for the
        // duration of this bake. Not cached in `self.pool` because that
        // pool is sized to the output resolution.
        let mut inter_texs: Vec<wgpu::Texture> =
            Vec::with_capacity(sched.peak_slots as usize);
        let mut inter_views: Vec<wgpu::TextureView> =
            Vec::with_capacity(sched.peak_slots as usize);
        for _ in 0..sched.peak_slots {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("tg-preview-inter"),
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
            inter_texs.push(tex);
            inter_views.push(view);
        }

        // One packed Rgba8Unorm output per layer, kept and handed back.
        let mut outputs: HashMap<LayerId, wgpu::Texture> = HashMap::new();
        let mut output_views: HashMap<LayerId, wgpu::TextureView> = HashMap::new();
        for &id in &sched.order {
            let tex = make_output_texture(&device, size, "tg-preview-out");
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            output_views.insert(id, view);
            outputs.insert(id, tex);
        }

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("tg-bake-previews"),
            });

        // 1. Dispatch every layer into its own intermediate slot.
        for &id in &sched.order {
            let slot = *sched.slot_of.get(&id).unwrap() as usize;
            let layer = graph.get(id).unwrap();
            self.dispatch_kind(
                &mut encoder,
                layer,
                eval_ctx,
                &sched,
                &inter_views,
                size,
                slot,
            )?;
        }

        // 2. Pack every intermediate to its sRGB output.
        for &id in &sched.order {
            let slot = *sched.slot_of.get(&id).unwrap() as usize;
            let dst_view = output_views.get(&id).unwrap();
            dispatch_pack(
                &self.ctx,
                &mut encoder,
                &self.pack_pipeline,
                &self.pack_bgl,
                &inter_views[slot],
                dst_view,
                size,
                0,
                [0.0; 4],
            );
        }

        self.ctx.queue.submit([encoder.finish()]);
        Ok(outputs)
    }

    /// Emit compute dispatches for one layer, writing its output into
    /// `pool_views[dst_slot]`. Shared between `bake_output` and
    /// `bake_previews`.
    fn dispatch_kind(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        layer: &texture_graph_core::Layer,
        eval_ctx: &EvalCtx,
        sched: &Schedule,
        pool_views: &[wgpu::TextureView],
        size: (u32, u32),
        dst_slot: usize,
    ) -> Result<(), BakeError> {
        match &layer.kind {
            LayerKind::Color(c) => {
                dispatch_color(
                    &self.ctx,
                    encoder,
                    &self.color_pipeline,
                    &self.color_bgl,
                    &pool_views[dst_slot],
                    [c.l, c.chroma, c.hue.into_degrees(), c.alpha],
                    size,
                );
            }
            LayerKind::Noise(n) => {
                dispatch_noise(
                    &self.ctx,
                    encoder,
                    &self.noise_pipeline,
                    &self.color_bgl,
                    &pool_views[dst_slot],
                    n,
                    eval_ctx.seed,
                    size,
                );
            }
            LayerKind::Transform(t) => {
                let src_slot = slot_of(sched, t.source);
                dispatch_transform(
                    &self.ctx,
                    encoder,
                    &self.transform_pipeline,
                    &self.transform_bgl,
                    &pool_views[dst_slot],
                    &pool_views[src_slot as usize],
                    t,
                    size,
                );
            }
            LayerKind::Mix(m) => {
                let a_slot = slot_of(sched, m.a);
                let b_slot = slot_of(sched, m.b);
                let factor_slot = match m.factor {
                    ScalarInput::Layer(id) => slot_of(sched, id),
                    // Bind `a` as a placeholder for the factor texture; the
                    // shader only samples it when factor_is_layer == 1.
                    ScalarInput::Const(_) => a_slot,
                };
                dispatch_mix(
                    &self.ctx,
                    encoder,
                    &self.mix_pipeline,
                    &self.mix_bgl,
                    &pool_views[dst_slot],
                    &pool_views[a_slot as usize],
                    &pool_views[b_slot as usize],
                    &pool_views[factor_slot as usize],
                    m,
                    size,
                );
            }
            LayerKind::Map(m) => {
                let value_slot = slot_of(sched, m.value);
                let palette_slot = slot_of(sched, m.palette);
                dispatch_map(
                    &self.ctx,
                    encoder,
                    &self.map_pipeline,
                    &self.map_bgl,
                    &pool_views[dst_slot],
                    &pool_views[value_slot as usize],
                    &pool_views[palette_slot as usize],
                    size,
                );
            }
            LayerKind::MinMax(mm) => {
                let a_slot = slot_of(sched, mm.a);
                let b_slot = slot_of(sched, mm.b);
                dispatch_min_max(
                    &self.ctx,
                    encoder,
                    &self.min_max_pipeline,
                    &self.map_bgl,
                    &pool_views[dst_slot],
                    &pool_views[a_slot as usize],
                    &pool_views[b_slot as usize],
                    mm,
                    size,
                );
            }
            LayerKind::HeightToNormal(h) => {
                let src_slot = slot_of(sched, h.source);
                dispatch_h2n(
                    &self.ctx,
                    encoder,
                    &self.h2n_pipeline,
                    &self.transform_bgl,
                    &pool_views[dst_slot],
                    &pool_views[src_slot as usize],
                    h,
                    size,
                );
            }
            LayerKind::ColorRamp(r) => {
                if r.stops.len() > MAX_RAMP_STOPS {
                    return Err(BakeError::Unsupported("ColorRamp (>16 stops)"));
                }
                dispatch_ramp(
                    &self.ctx,
                    encoder,
                    &self.ramp_pipeline,
                    &self.ramp_bgl,
                    &pool_views[dst_slot],
                    r,
                    sched,
                    pool_views,
                    &self.dummy_input_view,
                    size,
                )?;
            }
        }
        Ok(())
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
        let t0 = std::time::Instant::now();
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
            self.dispatch_kind(
                &mut encoder,
                layer,
                _ctx,
                &sched,
                &self.pool_views,
                size,
                slot,
            )?;
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
        let elapsed = t0.elapsed();
        let inter_bytes = (sched.peak_slots as u64) * (size.0 as u64) * (size.1 as u64) * 16;
        log::debug!(
            "bake_output {}x{}: layers={} peak_slots={} record+submit={:?} intermediates≈{}MiB",
            size.0,
            size.1,
            sched.order.len(),
            sched.peak_slots,
            elapsed,
            inter_bytes / (1024 * 1024),
        );
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

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct TransformParams {
    size: [u32; 2],
    coord_mode: u32,   // 0=Passthrough, 1=Permute, 2=Radial
    rotate_uv: f32,
    offset: [f32; 4],  // (u, v, w, _)
    scale: [f32; 4],   // (u, v, w, _)
    permute: [u32; 4], // (a, b, c, _), each 0=U 1=V 2=W
    radial_dim: u32,   // 0=D2, 1=D3
    radial_into: u32,  // 0=U 1=V 2=W
    _pad: [u32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct MixParams {
    size: [u32; 2],
    mode: u32,             // 0=Add, 1=Subtract, 2=Multiply, 3=Blend
    space: u32,            // 0=Oklch, 1=LinearSrgb, 2=Hsv
    factor_const: f32,
    factor_is_layer: u32,  // 0=Const, 1=Layer
    _pad: [u32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct MapParams {
    size: [u32; 2],
    _pad: [u32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct MinMaxParams {
    size: [u32; 2],
    mode: u32,       // 0 = Min, 1 = Max
    criterion: u32,  // 0=R 1=G 2=B 3=Sat 4=Val 5=Luma 6=Alpha 7=Chroma
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct RampParams {
    size: [u32; 2],
    stop_count: u32,
    space: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct H2NParams {
    size: [u32; 2],
    strength: f32,
    _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct RampStopPacked {
    color: [f32; 4],   // Oklcha; used when kind == 0
    t: f32,
    kind: u32,         // 0 = const, 1 = layer
    input_index: u32,  // 0..7 into the ramp shader's input array
    _p0: f32,
}

fn slot_of(sched: &Schedule, id: LayerId) -> u32 {
    *sched
        .slot_of
        .get(&id)
        .expect("scheduler must include every referenced layer")
}

fn blend_space_code(b: BlendSpace) -> u32 {
    match b {
        BlendSpace::Oklch => 0,
        BlendSpace::LinearSrgb => 1,
        BlendSpace::Hsv => 2,
    }
}

fn axis_code(a: Axis) -> u32 {
    match a {
        Axis::U => 0,
        Axis::V => 1,
        Axis::W => 2,
    }
}

fn dispatch_h2n(
    ctx: &DeviceCtx,
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    bgl: &wgpu::BindGroupLayout,
    dst_view: &wgpu::TextureView,
    src_view: &wgpu::TextureView,
    h: &HeightToNormal,
    size: (u32, u32),
) {
    let params = H2NParams {
        size: [size.0, size.1],
        strength: h.strength,
        _pad: 0,
    };
    let ubo = create_uniform(&ctx.device, bytemuck::bytes_of(&params), "h2n-params");
    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("h2n-bg"),
        layout: bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: ubo.as_entire_binding() },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(dst_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(src_view),
            },
        ],
    });
    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("h2n-cpass"),
        timestamp_writes: None,
    });
    cpass.set_pipeline(pipeline);
    cpass.set_bind_group(0, &bg, &[]);
    let (wg_x, wg_y) = workgroup_counts(size);
    cpass.dispatch_workgroups(wg_x, wg_y, 1);
}

fn dispatch_transform(
    ctx: &DeviceCtx,
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    bgl: &wgpu::BindGroupLayout,
    dst_view: &wgpu::TextureView,
    src_view: &wgpu::TextureView,
    t: &Transform,
    size: (u32, u32),
) {
    let (coord_mode, permute, radial_dim, radial_into) = match t.coord_mode {
        CoordMode::Passthrough => (0u32, [0u32; 4], 0u32, 0u32),
        CoordMode::Permute(axes) => (
            1u32,
            [axis_code(axes[0]), axis_code(axes[1]), axis_code(axes[2]), 0],
            0,
            0,
        ),
        CoordMode::Radial { dim, into } => (
            2u32,
            [0u32; 4],
            match dim {
                RadialDim::D2 => 0,
                RadialDim::D3 => 1,
            },
            axis_code(into),
        ),
    };
    let params = TransformParams {
        size: [size.0, size.1],
        coord_mode,
        rotate_uv: t.rotate_uv,
        offset: [t.offset[0], t.offset[1], t.offset[2], 0.0],
        scale: [t.scale[0], t.scale[1], t.scale[2], 0.0],
        permute,
        radial_dim,
        radial_into,
        _pad: [0, 0],
    };
    let ubo = create_uniform(&ctx.device, bytemuck::bytes_of(&params), "transform-params");
    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("transform-bg"),
        layout: bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: ubo.as_entire_binding() },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(dst_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(src_view),
            },
        ],
    });
    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("transform-cpass"),
        timestamp_writes: None,
    });
    cpass.set_pipeline(pipeline);
    cpass.set_bind_group(0, &bg, &[]);
    let (wg_x, wg_y) = workgroup_counts(size);
    cpass.dispatch_workgroups(wg_x, wg_y, 1);
}

fn dispatch_mix(
    ctx: &DeviceCtx,
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    bgl: &wgpu::BindGroupLayout,
    dst_view: &wgpu::TextureView,
    a_view: &wgpu::TextureView,
    b_view: &wgpu::TextureView,
    factor_view: &wgpu::TextureView,
    m: &Mix,
    size: (u32, u32),
) {
    let mode = match m.mode {
        BlendMode::Add => 0u32,
        BlendMode::Subtract => 1,
        BlendMode::Multiply => 2,
        BlendMode::Blend => 3,
    };
    let (factor_const, factor_is_layer) = match m.factor {
        ScalarInput::Const(v) => (v, 0u32),
        ScalarInput::Layer(_) => (0.0, 1u32),
    };
    let params = MixParams {
        size: [size.0, size.1],
        mode,
        space: blend_space_code(m.space),
        factor_const,
        factor_is_layer,
        _pad: [0, 0],
    };
    let ubo = create_uniform(&ctx.device, bytemuck::bytes_of(&params), "mix-params");
    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("mix-bg"),
        layout: bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: ubo.as_entire_binding() },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(dst_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(a_view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(b_view),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(factor_view),
            },
        ],
    });
    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("mix-cpass"),
        timestamp_writes: None,
    });
    cpass.set_pipeline(pipeline);
    cpass.set_bind_group(0, &bg, &[]);
    let (wg_x, wg_y) = workgroup_counts(size);
    cpass.dispatch_workgroups(wg_x, wg_y, 1);
}

fn dispatch_min_max(
    ctx: &DeviceCtx,
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    bgl: &wgpu::BindGroupLayout,
    dst_view: &wgpu::TextureView,
    a_view: &wgpu::TextureView,
    b_view: &wgpu::TextureView,
    mm: &MinMax,
    size: (u32, u32),
) {
    let mode = match mm.mode {
        MinMaxMode::Min => 0u32,
        MinMaxMode::Max => 1u32,
    };
    let criterion = match mm.criterion {
        Criterion::Red => 0u32,
        Criterion::Green => 1,
        Criterion::Blue => 2,
        Criterion::Saturation => 3,
        Criterion::Value => 4,
        Criterion::Luma => 5,
        Criterion::Alpha => 6,
        Criterion::Chroma => 7,
    };
    let params = MinMaxParams { size: [size.0, size.1], mode, criterion };
    let ubo = create_uniform(&ctx.device, bytemuck::bytes_of(&params), "min-max-params");
    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("min-max-bg"),
        layout: bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: ubo.as_entire_binding() },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(dst_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(a_view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(b_view),
            },
        ],
    });
    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("min-max-cpass"),
        timestamp_writes: None,
    });
    cpass.set_pipeline(pipeline);
    cpass.set_bind_group(0, &bg, &[]);
    let (wg_x, wg_y) = workgroup_counts(size);
    cpass.dispatch_workgroups(wg_x, wg_y, 1);
}

fn dispatch_map(
    ctx: &DeviceCtx,
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    bgl: &wgpu::BindGroupLayout,
    dst_view: &wgpu::TextureView,
    value_view: &wgpu::TextureView,
    palette_view: &wgpu::TextureView,
    size: (u32, u32),
) {
    let params = MapParams { size: [size.0, size.1], _pad: [0, 0] };
    let ubo = create_uniform(&ctx.device, bytemuck::bytes_of(&params), "map-params");
    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("map-bg"),
        layout: bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: ubo.as_entire_binding() },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(dst_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(value_view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(palette_view),
            },
        ],
    });
    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("map-cpass"),
        timestamp_writes: None,
    });
    cpass.set_pipeline(pipeline);
    cpass.set_bind_group(0, &bg, &[]);
    let (wg_x, wg_y) = workgroup_counts(size);
    cpass.dispatch_workgroups(wg_x, wg_y, 1);
}

fn dispatch_ramp(
    ctx: &DeviceCtx,
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    bgl: &wgpu::BindGroupLayout,
    dst_view: &wgpu::TextureView,
    r: &ColorRamp,
    sched: &Schedule,
    pool_views: &[wgpu::TextureView],
    dummy_input_view: &wgpu::TextureView,
    size: (u32, u32),
) -> Result<(), BakeError> {
    use wgpu::util::DeviceExt;

    // Resolve unique layer-refs across stops → 0..N-1 input indices. Multiple
    // stops pointing at the same layer share one slot.
    let mut layer_to_input: HashMap<LayerId, u32> = HashMap::new();
    let mut input_pool_slots: Vec<u32> = Vec::new();
    for s in &r.stops {
        if let ColorInput::Layer(id) = s.color {
            if !layer_to_input.contains_key(&id) {
                if input_pool_slots.len() >= MAX_RAMP_INPUTS {
                    return Err(BakeError::Unsupported(
                        "ColorRamp (>8 unique layer-referenced stops)",
                    ));
                }
                layer_to_input.insert(id, input_pool_slots.len() as u32);
                input_pool_slots.push(slot_of(sched, id));
            }
        }
    }

    let params = RampParams {
        size: [size.0, size.1],
        stop_count: r.stops.len() as u32,
        space: blend_space_code(r.space),
    };
    let ubo = create_uniform(&ctx.device, bytemuck::bytes_of(&params), "ramp-params");

    let mut packed = Vec::with_capacity(r.stops.len());
    for s in &r.stops {
        match s.color {
            ColorInput::Const(c) => packed.push(RampStopPacked {
                color: [c.l, c.chroma, c.hue.into_degrees(), c.alpha],
                t: s.t,
                kind: 0,
                input_index: 0,
                _p0: 0.0,
            }),
            ColorInput::Layer(id) => {
                let input_index = *layer_to_input
                    .get(&id)
                    .expect("layer resolved into input map above");
                packed.push(RampStopPacked {
                    color: [0.0; 4],
                    t: s.t,
                    kind: 1,
                    input_index,
                    _p0: 0.0,
                });
            }
        }
    }
    let stops_ssbo = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ramp-stops"),
        contents: bytemuck::cast_slice(&packed),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // Bindings 3..11 are the input textures. Any unused slot binds the
    // Baker's persistent 1×1 dummy (sampled-only) so it never collides with
    // the storage output binding. The shader only samples slots referenced
    // by a Stop with kind == 1.
    let mut input_views: [&wgpu::TextureView; MAX_RAMP_INPUTS] =
        [dummy_input_view; MAX_RAMP_INPUTS];
    for (i, &slot) in input_pool_slots.iter().enumerate() {
        input_views[i] = &pool_views[slot as usize];
    }

    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("ramp-bg"),
        layout: bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: ubo.as_entire_binding() },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(dst_view),
            },
            wgpu::BindGroupEntry { binding: 2, resource: stops_ssbo.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(input_views[0]) },
            wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(input_views[1]) },
            wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(input_views[2]) },
            wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(input_views[3]) },
            wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(input_views[4]) },
            wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(input_views[5]) },
            wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(input_views[6]) },
            wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::TextureView(input_views[7]) },
        ],
    });
    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("ramp-cpass"),
        timestamp_writes: None,
    });
    cpass.set_pipeline(pipeline);
    cpass.set_bind_group(0, &bg, &[]);
    let (wg_x, wg_y) = workgroup_counts(size);
    cpass.dispatch_workgroups(wg_x, wg_y, 1);
    Ok(())
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
        bind_group_layouts: &[Some(&bgl)],
        ..Default::default()
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
        bind_group_layouts: &[Some(bgl)],
        ..Default::default()
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

fn input_texture_bgle(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn storage_buffer_bgle(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn make_h2n_pipeline(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
) -> wgpu::ComputePipeline {
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("h2n-pl"),
        bind_group_layouts: &[Some(bgl)],
        ..Default::default()
    });
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("h2n-shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/height_to_normal.wgsl").into()),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("h2n-pipeline"),
        layout: Some(&pl),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn make_transform_pipeline(
    device: &wgpu::Device,
) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("transform-bgl"),
        entries: &[
            uniform_bgle(0),
            storage_texture_bgle(1, wgpu::TextureFormat::Rgba32Float),
            input_texture_bgle(2),
        ],
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("transform-pl"),
        bind_group_layouts: &[Some(&bgl)],
        ..Default::default()
    });
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("transform-shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/transform.wgsl").into()),
    });
    let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("transform-pipeline"),
        layout: Some(&pl),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    (pipe, bgl)
}

fn make_mix_pipeline(
    device: &wgpu::Device,
) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("mix-bgl"),
        entries: &[
            uniform_bgle(0),
            storage_texture_bgle(1, wgpu::TextureFormat::Rgba32Float),
            input_texture_bgle(2),
            input_texture_bgle(3),
            input_texture_bgle(4),
        ],
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("mix-pl"),
        bind_group_layouts: &[Some(&bgl)],
        ..Default::default()
    });
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("mix-shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/mix.wgsl").into()),
    });
    let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("mix-pipeline"),
        layout: Some(&pl),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    (pipe, bgl)
}

fn make_map_pipeline(
    device: &wgpu::Device,
) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("map-bgl"),
        entries: &[
            uniform_bgle(0),
            storage_texture_bgle(1, wgpu::TextureFormat::Rgba32Float),
            input_texture_bgle(2),
            input_texture_bgle(3),
        ],
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("map-pl"),
        bind_group_layouts: &[Some(&bgl)],
        ..Default::default()
    });
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("map-shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/map.wgsl").into()),
    });
    let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("map-pipeline"),
        layout: Some(&pl),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    (pipe, bgl)
}

fn make_min_max_pipeline(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
) -> wgpu::ComputePipeline {
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("min-max-pl"),
        bind_group_layouts: &[Some(bgl)],
        ..Default::default()
    });
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("min-max-shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/min_max.wgsl").into()),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("min-max-pipeline"),
        layout: Some(&pl),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn make_ramp_pipeline(
    device: &wgpu::Device,
) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("ramp-bgl"),
        entries: &[
            uniform_bgle(0),
            storage_texture_bgle(1, wgpu::TextureFormat::Rgba32Float),
            storage_buffer_bgle(2),
            input_texture_bgle(3),
            input_texture_bgle(4),
            input_texture_bgle(5),
            input_texture_bgle(6),
            input_texture_bgle(7),
            input_texture_bgle(8),
            input_texture_bgle(9),
            input_texture_bgle(10),
        ],
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("ramp-pl"),
        bind_group_layouts: &[Some(&bgl)],
        ..Default::default()
    });
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("ramp-shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/color_ramp.wgsl").into()),
    });
    let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("ramp-pipeline"),
        layout: Some(&pl),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    (pipe, bgl)
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
        bind_group_layouts: &[Some(&bgl)],
        ..Default::default()
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
        bind_group_layouts: &[Some(&bgl)],
        ..Default::default()
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
