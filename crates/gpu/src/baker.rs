//! GPU compute-shader baker.
//!
//! Owns the device handles, a pool of `Rgba32Float` intermediates sized to
//! the current output, and one compute pipeline per `LayerKind`. Walks the
//! schedule dispatching a shader per layer, then packs the four PBR channels
//! into `Rgba8Unorm`.
//!
//! wgpu forbids storage bindings to sRGB view formats, so those outputs hold
//! sRGB-encoded values in a linear format and `pack_srgb8.wgsl` applies the
//! gamma by hand to match `core::color::to_srgb8`.

use std::collections::{HashMap, HashSet};

use bytemuck::{Pod, Zeroable};
use texture_graph_core::{
    Axis, BlendMode, BlendSpace, ColorInput, ColorRamp, CoordMode, Criterion, EvalCtx,
    FractalMode, Graph, HeightToNormal, LayerId, LayerKind, MinMax, MinMaxMode, Mix, Noise,
    NoiseDims, NoiseKernel, NoiseOutput, NoiseRange, RadialDim, ScalarInput, Transform,
    Wave, WaveShape,
};

use texture_graph_core::EdgeMode;

use crate::device::DeviceCtx;
use crate::schedule::{
    Domain, OutputSlots, ScalarSlot, Schedule, schedule, schedule_layer, schedule_previews,
};

/// Max stops per ColorRamp supported by the GPU baker.
const MAX_RAMP_STOPS: usize = 16;
/// Bindings are declared at pipeline creation, so this is a hard cap.
const MAX_RAMP_INPUTS: usize = 8;

/// Four sRGB-encoded 8-bit-per-channel textures ready for display.
pub struct BakeOutput {
    pub color: wgpu::Texture,
    pub roughness: wgpu::Texture,
    pub metallic: wgpu::Texture,
    pub normal: wgpu::Texture,
    pub size: (u32, u32),
}

/// The four PBR channels over a whole `res × res × depth` volume, `w` at each
/// slice centre. The 3D preview samples these at object-space position rather
/// than UV-mapping one flat slice.
pub struct VolumeOutput {
    pub color: wgpu::Texture,
    pub roughness: wgpu::Texture,
    pub metallic: wgpu::Texture,
    pub normal: wgpu::Texture,
    /// (width, height, depth) in texels.
    pub size: (u32, u32, u32),
}

/// Texel format for a single-channel bake.
///
/// All three are renderable in core WebGPU, which is why `bake_scalar`
/// goes through a render pass — `R8Unorm` is not a core *storage* format.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum ScalarFormat {
    /// One byte a texel, clamped to `[0, 1]` by the format. A quarter the
    /// memory of `R16Float` and the usual choice for a baked field.
    R8Unorm,
    /// Half a float a texel, unclamped, and filterable on WebGPU.
    R16Float,
    /// Full precision, unclamped — but **not filterable** on WebGPU
    /// without the `float32-filterable` feature, so a consumer sampling
    /// this with a linear sampler gets a validation error. Useful for
    /// tests and for a host that reads the values back rather than
    /// sampling them.
    R32Float,
}

impl ScalarFormat {
    pub fn texture_format(self) -> wgpu::TextureFormat {
        match self {
            ScalarFormat::R8Unorm => wgpu::TextureFormat::R8Unorm,
            ScalarFormat::R16Float => wgpu::TextureFormat::R16Float,
            ScalarFormat::R32Float => wgpu::TextureFormat::R32Float,
        }
    }

    pub fn bytes_per_texel(self) -> u32 {
        match self {
            ScalarFormat::R8Unorm => 1,
            ScalarFormat::R16Float => 2,
            ScalarFormat::R32Float => 4,
        }
    }
}

/// One scalar field over a whole `res × res × depth` volume, `w` at each
/// slice centre — the volume counterpart of [`Baker::bake_scalar`].
pub struct ScalarVolume {
    pub texture: wgpu::Texture,
    /// (width, height, depth) in texels.
    pub size: (u32, u32, u32),
    pub format: ScalarFormat,
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
    /// Bound into unused ramp input slots, so they never collide with an
    /// output storage binding. Held here to keep `dummy_input_view` valid.
    #[allow(dead_code)]
    dummy_input: wgpu::Texture,
    dummy_input_view: wgpu::TextureView,
    h2n_pipeline: wgpu::ComputePipeline,
    // wave reuses `transform_bgl` too — uniform + storage_out + input_2d.
    wave_pipeline: wgpu::ComputePipeline,
    // h2n reuses `transform_bgl` — same binding shape (uniform + storage_out + input_2d).
    // missing reuses `color_bgl`: same binding shape (uniform + storage_texture).
    missing_pipeline: wgpu::ComputePipeline,
    /// The magenta/black missing-texture grid, bound wherever a layer input
    /// is `None`. Refilled per bake, and per slice for volumes.
    missing_tex: Option<wgpu::Texture>,
    missing_view: Option<wgpu::TextureView>,
    pack_pipeline: wgpu::ComputePipeline,
    pack_bgl: wgpu::BindGroupLayout,
    solid_pipeline: wgpu::ComputePipeline,
    solid_bgl: wgpu::BindGroupLayout,
    /// One render pipeline per scalar format — a pipeline is bound to its
    /// colour-target format, so they cannot share. Built on first use
    /// rather than up front, because most sessions never bake a scalar.
    scalar_shader: wgpu::ShaderModule,
    scalar_bgl: wgpu::BindGroupLayout,
    scalar_pipelines: HashMap<wgpu::TextureFormat, wgpu::RenderPipeline>,
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
        let wave_pipeline = make_wave_pipeline(&ctx.device, &transform_bgl);
        let dummy_input = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("tg-dummy-input"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            // Sampled only: no STORAGE_BINDING, so wgpu can't confuse this
            // with an output slot.
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let dummy_input_view = dummy_input.create_view(&wgpu::TextureViewDescriptor::default());
        let missing_pipeline = make_missing_pipeline(&ctx.device, &color_bgl);
        let (pack_pipeline, pack_bgl) = make_pack_pipeline(&ctx.device);
        let (solid_pipeline, solid_bgl) = make_solid_pipeline(&ctx.device);
        let (scalar_shader, scalar_bgl) = make_scalar_pack_shader(&ctx.device);
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
            missing_pipeline,
            missing_tex: None,
            missing_view: None,
            ramp_pipeline,
            ramp_bgl,
            dummy_input,
            dummy_input_view,
            h2n_pipeline,
            wave_pipeline,
            pack_pipeline,
            pack_bgl,
            solid_pipeline,
            solid_bgl,
            scalar_shader,
            scalar_bgl,
            scalar_pipelines: HashMap::new(),
        }
    }

    pub fn ctx(&self) -> &DeviceCtx {
        &self.ctx
    }

    /// Bake per-layer thumbnails at 128². Every layer gets its own
    /// intermediate slot (no pebble reuse), then a `pack_srgb8` pass packs
    /// each to an `Rgba8Unorm` texture. Returns one texture per layer for
    /// the UI to register with egui-wgpu.
    /// Bake 128² thumbnails for `wanted` — or for every layer when it is
    /// `None`.
    ///
    /// Asking for a subset costs a subset: only `wanted` and the layers
    /// they transitively read are dispatched, and only `wanted` get an
    /// output texture and a pack pass. The returned map has exactly the
    /// layers that were asked for, so a caller holding textures for the
    /// rest keeps showing them.
    pub fn bake_previews(
        &mut self,
        graph: &Graph,
        eval_ctx: &EvalCtx,
        wanted: Option<&HashSet<LayerId>>,
    ) -> Result<HashMap<LayerId, wgpu::Texture>, BakeError> {
        const PREVIEW_SIZE: (u32, u32) = (128, 128);
        let sched = schedule_previews(graph, wanted)?;
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

        // One packed Rgba8Unorm output per *wanted* layer, kept and handed
        // back. The rest of `sched.order` is scheduled only because
        // something wanted reads it, and is never packed or returned.
        let packed: Vec<LayerId> = match wanted {
            Some(w) => sched.order.iter().copied().filter(|id| w.contains(id)).collect(),
            None => sched.order.clone(),
        };
        let mut outputs: HashMap<LayerId, wgpu::Texture> = HashMap::new();
        let mut output_views: HashMap<LayerId, wgpu::TextureView> = HashMap::new();
        for &id in &packed {
            let tex = make_output_texture(&device, size, "tg-preview-out");
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            output_views.insert(id, view);
            outputs.insert(id, tex);
        }

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("tg-bake-previews"),
            });

        // Preview-sized missing-texture grid (the pool-sized one on
        // `self` may be a different resolution).
        let (_missing_tex, missing_view) = make_missing_texture(&device, size);
        dispatch_missing(
            &self.ctx, &mut encoder, &self.missing_pipeline, &self.color_bgl,
            &missing_view, size, texture_graph_core::FLAT_W,
        );

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
                0.5,
                &missing_view,
            )?;
        }

        // 2. Pack every wanted intermediate to its sRGB output. Thumbnails
        // show the layer's own [0, 1] view even when its bake domain is
        // wider.
        for &id in &packed {
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
                false,
                0,
                domain_of(&sched, id),
            );
        }

        self.ctx.queue.submit([encoder.finish()]);
        Ok(outputs)
    }

    /// Emit compute dispatches for one layer, writing its output into
    /// `pool_views[dst_slot]`. Shared between `bake_output`,
    /// `bake_previews`, and `bake_volume`. `w` is the third texture
    /// coordinate for this pass — 0.5 for flat bakes, the slice center for
    /// volume bakes.
    fn dispatch_kind(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        layer: &texture_graph_core::Layer,
        eval_ctx: &EvalCtx,
        sched: &Schedule,
        pool_views: &[wgpu::TextureView],
        size: (u32, u32),
        dst_slot: usize,
        w: f32,
        missing_view: &wgpu::TextureView,
    ) -> Result<(), BakeError> {
        // A `None` input has no slot — it samples the missing-texture grid.
        let resolve = |opt: Option<texture_graph_core::LayerId>| -> &wgpu::TextureView {
            match opt {
                Some(id) => &pool_views[slot_of(sched, id) as usize],
                None => missing_view,
            }
        };
        // Bake domain of this layer and of each input (the missing grid is
        // always baked over the unit square).
        let own_dom = domain_of(sched, layer.id);
        let dom_opt = |opt: Option<texture_graph_core::LayerId>| -> [f32; 4] {
            match opt {
                Some(id) => domain_of(sched, id),
                None => Domain::UNIT.packed(),
            }
        };
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
                    w,
                    own_dom,
                );
            }
            LayerKind::Transform(t) => {
                dispatch_transform(
                    &self.ctx,
                    encoder,
                    &self.transform_pipeline,
                    &self.transform_bgl,
                    &pool_views[dst_slot],
                    resolve(t.source),
                    t,
                    size,
                    w,
                    own_dom,
                    dom_opt(t.source),
                );
            }
            LayerKind::Mix(m) => {
                let a_view = resolve(m.a);
                let (factor_view, dom_factor) = match m.factor {
                    ScalarInput::Layer(id) => (resolve(Some(id)), dom_opt(Some(id))),
                    // Bind `a` as a placeholder for the factor texture; the
                    // shader only samples it when factor_is_layer == 1.
                    ScalarInput::Const(_) => (a_view, dom_opt(m.a)),
                };
                dispatch_mix(
                    &self.ctx,
                    encoder,
                    &self.mix_pipeline,
                    &self.mix_bgl,
                    &pool_views[dst_slot],
                    a_view,
                    resolve(m.b),
                    factor_view,
                    m,
                    size,
                    own_dom,
                    dom_opt(m.a),
                    dom_opt(m.b),
                    dom_factor,
                );
            }
            LayerKind::Map(m) => {
                dispatch_map(
                    &self.ctx,
                    encoder,
                    &self.map_pipeline,
                    &self.map_bgl,
                    &pool_views[dst_slot],
                    resolve(m.value),
                    resolve(m.palette),
                    size,
                    own_dom,
                    dom_opt(m.value),
                    dom_opt(m.palette),
                );
            }
            LayerKind::MinMax(mm) => {
                dispatch_min_max(
                    &self.ctx,
                    encoder,
                    &self.min_max_pipeline,
                    &self.map_bgl,
                    &pool_views[dst_slot],
                    resolve(mm.a),
                    resolve(mm.b),
                    mm,
                    size,
                    own_dom,
                    dom_opt(mm.a),
                    dom_opt(mm.b),
                );
            }
            LayerKind::HeightToNormal(h) => {
                dispatch_h2n(
                    &self.ctx,
                    encoder,
                    &self.h2n_pipeline,
                    &self.transform_bgl,
                    &pool_views[dst_slot],
                    resolve(h.source),
                    h,
                    size,
                    own_dom,
                    dom_opt(h.source),
                );
            }
            LayerKind::Wave(wv) => {
                // A const input never reads the texture; bind the output's
                // own domain's placeholder rather than leave the slot
                // unbound, as Mix does for its factor.
                let (src_view, dom_input) = match wv.input {
                    ScalarInput::Layer(id) => (resolve(Some(id)), dom_opt(Some(id))),
                    ScalarInput::Const(_) => (missing_view, Domain::UNIT.packed()),
                };
                dispatch_wave(
                    &self.ctx,
                    encoder,
                    &self.wave_pipeline,
                    &self.transform_bgl,
                    &pool_views[dst_slot],
                    src_view,
                    wv,
                    size,
                    own_dom,
                    dom_input,
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
                    own_dom,
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
            self.missing_tex = None;
            self.missing_view = None;
        }
        if self.missing_tex.is_none() {
            let (tex, view) = make_missing_texture(&self.ctx.device, size);
            self.missing_tex = Some(tex);
            self.missing_view = Some(view);
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
    ///
    /// `object_alpha` switches the presentation of partial alpha: `false`
    /// composites the gray backing checker into the color channel (flat
    /// previews), `true` keeps the real alpha so the 3D preview can blend
    /// the object itself.
    pub fn bake_output(
        &mut self,
        graph: &Graph,
        size: (u32, u32),
        eval_ctx: &EvalCtx,
        object_alpha: bool,
    ) -> Result<BakeOutput, BakeError> {
        let t0 = BakeTimer::start();
        let sched = schedule(graph)?;
        self.ensure_pool(size, sched.peak_slots.max(1));

        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("tg-bake-output"),
            });

        let missing_view = self.missing_view.as_ref().unwrap();
        dispatch_missing(
            &self.ctx, &mut encoder, &self.missing_pipeline, &self.color_bgl,
            missing_view, size, texture_graph_core::FLAT_W,
        );

        // Dispatch each layer.
        for &id in &sched.order {
            let slot = *sched.slot_of.get(&id).unwrap() as usize;
            let layer = graph.get(id).unwrap();
            self.dispatch_kind(
                &mut encoder,
                layer,
                eval_ctx,
                &sched,
                &self.pool_views,
                size,
                slot,
                0.5,
                missing_view,
            )?;
        }

        // Pack the four output channels. Each pack maps display [0, 1] UV
        // into its source layer's bake domain.
        let chan_dom = |id: Option<LayerId>| -> [f32; 4] {
            match id {
                Some(id) => domain_of(&sched, id),
                None => Domain::UNIT.packed(),
            }
        };
        let color_dom = chan_dom(graph.output.color);
        let rough_dom = match graph.output.roughness {
            ScalarInput::Layer(id) => chan_dom(Some(id)),
            ScalarInput::Const(_) => Domain::UNIT.packed(),
        };
        let metal_dom = match graph.output.metallic {
            ScalarInput::Layer(id) => chan_dom(Some(id)),
            ScalarInput::Const(_) => Domain::UNIT.packed(),
        };
        let normal_dom = chan_dom(graph.output.normal);
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
            missing_view,
            &color.create_view(&wgpu::TextureViewDescriptor::default()),
            size,
            OutputChannel::Color,
            &sched.output_slots,
            object_alpha,
            0,
            color_dom,
        );
        pack_channel(
            &self.ctx,
            &mut encoder,
            &self.pack_pipeline,
            &self.pack_bgl,
            &self.solid_pipeline,
            &self.solid_bgl,
            &self.pool_views,
            missing_view,
            &roughness.create_view(&wgpu::TextureViewDescriptor::default()),
            size,
            OutputChannel::Roughness,
            &sched.output_slots,
            object_alpha,
            0,
            rough_dom,
        );
        pack_channel(
            &self.ctx,
            &mut encoder,
            &self.pack_pipeline,
            &self.pack_bgl,
            &self.solid_pipeline,
            &self.solid_bgl,
            &self.pool_views,
            missing_view,
            &metallic.create_view(&wgpu::TextureViewDescriptor::default()),
            size,
            OutputChannel::Metallic,
            &sched.output_slots,
            object_alpha,
            0,
            metal_dom,
        );
        pack_channel(
            &self.ctx,
            &mut encoder,
            &self.pack_pipeline,
            &self.pack_bgl,
            &self.solid_pipeline,
            &self.solid_bgl,
            &self.pool_views,
            missing_view,
            &normal.create_view(&wgpu::TextureViewDescriptor::default()),
            size,
            OutputChannel::Normal,
            &sched.output_slots,
            object_alpha,
            0,
            normal_dom,
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

    /// Bake one layer's scalar (its Oklch L) to a single-channel texture.
    ///
    /// The point is arithmetic, not taste: a consumer that wants one
    /// grayscale field out of [`Baker::bake_output`] pays for four
    /// `Rgba8Unorm` channels and four pack passes to throw three away.
    /// At 256³ that is the difference between 268 MB and 16 MB.
    ///
    /// Only `layer` and what it transitively reads are dispatched — the
    /// graph's Output is not consulted, so a graph can carry several
    /// fields side by side and a consumer can bake each one on its own.
    ///
    /// The texel is the raw scalar, not a display-encoded gray. `R8Unorm`
    /// clamps to `[0, 1]` as the format requires; the float formats keep
    /// the value unclamped, so a signed field survives.
    ///
    /// The returned texture carries `TEXTURE_BINDING | COPY_SRC`, so a
    /// consumer sharing this device can sample it directly and skip
    /// readback entirely — which is the fast path, and the reason this
    /// returns a texture rather than an image.
    pub fn bake_scalar(
        &mut self,
        graph: &Graph,
        layer: LayerId,
        size: (u32, u32),
        format: ScalarFormat,
        eval_ctx: &EvalCtx,
    ) -> Result<wgpu::Texture, BakeError> {
        let sched = schedule_layer(graph, layer)?;
        self.ensure_pool(size, sched.peak_slots.max(1));

        let device = self.ctx.device.clone();
        let dst = make_scalar_texture(&device, size, format, "tg-scalar");
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("tg-bake-scalar"),
        });
        self.record_scalar_slice(
            &mut encoder,
            graph,
            &sched,
            layer,
            size,
            texture_graph_core::FLAT_W,
            format,
            &dst.create_view(&wgpu::TextureViewDescriptor::default()),
            eval_ctx,
        )?;
        self.ctx.queue.submit([encoder.finish()]);
        Ok(dst)
    }

    /// [`Baker::bake_scalar`] over a volume: `res × res × depth` sampled at
    /// slice centres, exactly as [`Baker::bake_volume`] does, and with the
    /// same known limitation — a Transform's w offset/scale cannot
    /// re-sample its input at a different w, because each slice only has
    /// its inputs baked at that slice's w.
    ///
    /// One encoder and submit per slice, for the reason `bake_volume`
    /// gives: recording every slice into one encoder blows wgpu-metal's
    /// outstanding-command-buffer cap on a real graph.
    pub fn bake_scalar_volume(
        &mut self,
        graph: &Graph,
        layer: LayerId,
        res: u32,
        depth: u32,
        format: ScalarFormat,
        eval_ctx: &EvalCtx,
    ) -> Result<ScalarVolume, BakeError> {
        let t0 = BakeTimer::start();
        let sched = schedule_layer(graph, layer)?;
        let size = (res, res);
        self.ensure_pool(size, sched.peak_slots.max(1));

        let device = self.ctx.device.clone();
        // One reusable 2D slice target, copied into the volume per slice —
        // a 2D view of a 3D texture is not a thing WebGPU offers.
        let slice = make_scalar_texture(&device, size, format, "tg-scalar-vol-slice");
        let slice_view = slice.create_view(&wgpu::TextureViewDescriptor::default());
        let texture = make_scalar_volume_texture(&device, res, depth, format, "tg-scalar-vol");

        for z in 0..depth {
            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("tg-bake-scalar-volume"),
                });
            let w = (z as f32 + 0.5) / depth as f32;
            self.record_scalar_slice(
                &mut encoder, graph, &sched, layer, size, w, format, &slice_view, eval_ctx,
            )?;
            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &slice,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x: 0, y: 0, z },
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d { width: res, height: res, depth_or_array_layers: 1 },
            );
            self.ctx.queue.submit([encoder.finish()]);
        }

        log::debug!(
            "bake_scalar_volume {res}³ (depth={depth}, {format:?}): layers/slice={} \
             record+submit={:?}",
            sched.order.len(),
            t0.elapsed(),
        );
        Ok(ScalarVolume { texture, size: (res, res, depth), format })
    }

    /// Dispatch every layer for one `w` slice, then pack `layer`'s slot
    /// into `dst_view`. Shared by the flat and volume scalar bakes.
    #[allow(clippy::too_many_arguments)]
    fn record_scalar_slice(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        graph: &Graph,
        sched: &Schedule,
        layer: LayerId,
        size: (u32, u32),
        w: f32,
        format: ScalarFormat,
        dst_view: &wgpu::TextureView,
        eval_ctx: &EvalCtx,
    ) -> Result<(), BakeError> {
        let missing_view = self.missing_view.clone().expect("ensure_pool made one");
        dispatch_missing(
            &self.ctx, encoder, &self.missing_pipeline, &self.color_bgl,
            &missing_view, size, w,
        );
        for &id in &sched.order {
            let slot = *sched.slot_of.get(&id).unwrap() as usize;
            let l = graph.get(id).unwrap();
            self.dispatch_kind(
                encoder, l, eval_ctx, sched, &self.pool_views, size, slot, w, &missing_view,
            )?;
        }
        let src_slot = *sched
            .slot_of
            .get(&layer)
            .expect("schedule_layer always schedules its root") as usize;
        let pipeline = self.scalar_pipeline(format);
        draw_scalar_pack(
            &self.ctx,
            encoder,
            &pipeline,
            &self.scalar_bgl,
            &self.pool_views[src_slot],
            dst_view,
            size,
            domain_of(sched, layer),
        );
        Ok(())
    }

    /// The render pipeline for `format`, built on first use. Cloning a
    /// `RenderPipeline` is an `Arc` bump, which is what lets this hand one
    /// out without borrowing `self` for the rest of the call.
    fn scalar_pipeline(&mut self, format: ScalarFormat) -> wgpu::RenderPipeline {
        let tf = format.texture_format();
        self.scalar_pipelines
            .entry(tf)
            .or_insert_with(|| {
                make_scalar_pack_pipeline(
                    &self.ctx.device,
                    &self.scalar_shader,
                    &self.scalar_bgl,
                    tf,
                )
            })
            .clone()
    }

    /// Bake the graph as a solid 3D texture: run the whole per-slice 2D
    /// pipeline `depth` times with w advancing through the slice centers,
    /// packing each slice and copying it into layer `z` of four 3D
    /// textures. Reuses every existing 2D shader — the only per-slice
    /// difference is the `w` uniform fed to the coordinate-generating
    /// stages (noise, transform).
    ///
    /// Known limitation (same as the flat GPU path): a Transform's w
    /// *offset/scale* can't re-sample its input at a different w, because
    /// each slice only has its inputs baked at the same w.
    pub fn bake_volume(
        &mut self,
        graph: &Graph,
        res: u32,
        depth: u32,
        eval_ctx: &EvalCtx,
    ) -> Result<VolumeOutput, BakeError> {
        let t0 = BakeTimer::start();
        let sched = schedule(graph)?;
        let size = (res, res);
        self.ensure_pool(size, sched.peak_slots.max(1));

        // Reusable 2D slice targets (storage-written by pack, then copied
        // out) and the four 3D destination volumes.
        let slices: Vec<wgpu::Texture> = ["color", "rough", "metal", "normal"]
            .iter()
            .map(|n| make_output_texture(&self.ctx.device, size, &format!("tg-vol-slice-{n}")))
            .collect();
        let slice_views: Vec<wgpu::TextureView> = slices
            .iter()
            .map(|t| t.create_view(&wgpu::TextureViewDescriptor::default()))
            .collect();
        let volumes: Vec<wgpu::Texture> = ["color", "rough", "metal", "normal"]
            .iter()
            .map(|n| make_volume_texture(&self.ctx.device, res, depth, &format!("tg-vol-{n}")))
            .collect();

        const CHANNELS: [OutputChannel; 4] = [
            OutputChannel::Color,
            OutputChannel::Roughness,
            OutputChannel::Metallic,
            OutputChannel::Normal,
        ];
        let chan_dom = |id: Option<LayerId>| -> [f32; 4] {
            match id {
                Some(id) => domain_of(&sched, id),
                None => Domain::UNIT.packed(),
            }
        };
        let chan_doms: [[f32; 4]; 4] = [
            chan_dom(graph.output.color),
            match graph.output.roughness {
                ScalarInput::Layer(id) => chan_dom(Some(id)),
                ScalarInput::Const(_) => Domain::UNIT.packed(),
            },
            match graph.output.metallic {
                ScalarInput::Layer(id) => chan_dom(Some(id)),
                ScalarInput::Const(_) => Domain::UNIT.packed(),
            },
            chan_dom(graph.output.normal),
        ];
        let missing_view = self.missing_view.as_ref().unwrap();
        for z in 0..depth {
            // One encoder + submit PER SLICE. On Metal every compute pass
            // becomes its own command buffer that stays "outstanding" until
            // its encoder is submitted; recording all slices into one
            // encoder puts depth × layers passes in flight at once, which
            // blows wgpu-metal's 4096 outstanding-command-buffer cap on
            // real graphs (observed: ~30-layer graph × 64 slices → device
            // lost). Per-slice submits keep it bounded by one slice's
            // passes.
            let mut encoder = self
                .ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("tg-bake-volume"),
                });
            let w = (z as f32 + 0.5) / depth as f32;
            // Refill per slice — the grid alternates along w too.
            dispatch_missing(
                &self.ctx, &mut encoder, &self.missing_pipeline, &self.color_bgl,
                missing_view, size, w,
            );
            for &id in &sched.order {
                let slot = *sched.slot_of.get(&id).unwrap() as usize;
                let layer = graph.get(id).unwrap();
                self.dispatch_kind(
                    &mut encoder,
                    layer,
                    eval_ctx,
                    &sched,
                    &self.pool_views,
                    size,
                    slot,
                    w,
                    missing_view,
                )?;
            }
            for (i, channel) in CHANNELS.iter().enumerate() {
                pack_channel(
                    &self.ctx,
                    &mut encoder,
                    &self.pack_pipeline,
                    &self.pack_bgl,
                    &self.solid_pipeline,
                    &self.solid_bgl,
                    &self.pool_views,
                    missing_view,
                    &slice_views[i],
                    size,
                    *channel,
                    &sched.output_slots,
                    true,
                    z,
                    chan_doms[i],
                );
                encoder.copy_texture_to_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &slices[i],
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::TexelCopyTextureInfo {
                        texture: &volumes[i],
                        mip_level: 0,
                        origin: wgpu::Origin3d { x: 0, y: 0, z },
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::Extent3d {
                        width: res,
                        height: res,
                        depth_or_array_layers: 1,
                    },
                );
            }
            self.ctx.queue.submit([encoder.finish()]);
        }

        log::debug!(
            "bake_volume {res}³ (depth={depth}): layers/slice={} record+submit={:?}",
            sched.order.len(),
            t0.elapsed(),
        );
        let mut it = volumes.into_iter();
        Ok(VolumeOutput {
            color: it.next().unwrap(),
            roughness: it.next().unwrap(),
            metallic: it.next().unwrap(),
            normal: it.next().unwrap(),
            size: (res, res, depth),
        })
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
    w_coord: f32,
    /// Bake domain as (min_u, min_v, ext_u, ext_v).
    dom: [f32; 4],
    /// Lattice period in cells; 0 = unbounded on that axis. `vec3<u32>` in
    /// the shader, so it has to start 16-byte aligned — which offset 48
    /// is, and the trailing `_pad` keeps the struct a multiple of 16.
    period: [u32; 3],
    octaves: u32,
    lacunarity: f32,
    gain: f32,
    fractal_mode: u32,
    normalize: u32,
    kernel: u32,
    _pad: [u32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct PackParams {
    size: [u32; 2],
    mode: u32,
    /// 0 = composite the gray alpha checker (flat previews);
    /// 1 = keep real alpha (3D preview blends the object itself).
    alpha_object: u32,
    const_value: [f32; 4],
    /// Volume slice index in texels (0 for flat bakes) — third axis of
    /// the out-of-range 3D checkerboard.
    z_px: u32,
    _pad: [u32; 3],
    /// Source layer's bake domain (min_u, min_v, ext_u, ext_v).
    src_dom: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct MissingParams {
    size: [u32; 2],
    w_coord: f32,
    _pad: u32,
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
    w_coord: f32,      // third texture coordinate; 0.5 for flat bakes
    edge_mode: u32,    // 0=Clamp, 1=Extend
    /// Own bake domain (min_u, min_v, ext_u, ext_v).
    dom: [f32; 4],
    /// Source layer's bake domain.
    src_dom: [f32; 4],
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
    /// Own bake domain, then each input's (min_u, min_v, ext_u, ext_v).
    dom: [f32; 4],
    dom_a: [f32; 4],
    dom_b: [f32; 4],
    dom_factor: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct MapParams {
    size: [u32; 2],
    _pad: [u32; 2],
    /// Own bake domain, the value input's, and the palette's.
    dom: [f32; 4],
    dom_value: [f32; 4],
    dom_palette: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct MinMaxParams {
    size: [u32; 2],
    mode: u32,       // 0 = Min, 1 = Max
    criterion: u32,  // 0=R 1=G 2=B 3=Sat 4=Val 5=Luma 6=Alpha 7=Chroma
    /// Own bake domain, then each input's.
    dom: [f32; 4],
    dom_a: [f32; 4],
    dom_b: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct RampParams {
    size: [u32; 2],
    stop_count: u32,
    space: u32,
    /// Own bake domain.
    dom: [f32; 4],
    /// Bake domain of each of the 8 possible layer-stop inputs.
    input_doms: [[f32; 4]; 8],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct H2NParams {
    size: [u32; 2],
    strength: f32,
    _pad: u32,
    /// Own bake domain, then the source's.
    dom: [f32; 4],
    dom_src: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct WaveParams {
    size: [u32; 2],
    shape: u32,
    range: u32,
    frequency: f32,
    phase: f32,
    input_const: f32,
    input_is_layer: u32,
    /// Own bake domain, then the input layer's.
    dom: [f32; 4],
    dom_input: [f32; 4],
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

fn domain_of(sched: &Schedule, id: LayerId) -> [f32; 4] {
    sched
        .domain_of
        .get(&id)
        .copied()
        .unwrap_or(Domain::UNIT)
        .packed()
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
    dom: [f32; 4],
    dom_src: [f32; 4],
) {
    let params = H2NParams {
        size: [size.0, size.1],
        strength: h.strength,
        _pad: 0,
        dom,
        dom_src,
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

#[allow(clippy::too_many_arguments)]
fn dispatch_wave(
    ctx: &DeviceCtx,
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    bgl: &wgpu::BindGroupLayout,
    dst_view: &wgpu::TextureView,
    src_view: &wgpu::TextureView,
    w: &Wave,
    size: (u32, u32),
    dom: [f32; 4],
    dom_input: [f32; 4],
) {
    let shape = match w.shape {
        WaveShape::Sine => 0u32,
        WaveShape::Triangle => 1,
        WaveShape::Square => 2,
        WaveShape::Sawtooth => 3,
    };
    let (input_const, input_is_layer) = match w.input {
        ScalarInput::Const(v) => (v, 0u32),
        ScalarInput::Layer(_) => (0.0, 1u32),
    };
    let params = WaveParams {
        size: [size.0, size.1],
        shape,
        range: match w.range {
            NoiseRange::Unsigned => 0,
            NoiseRange::Signed => 1,
        },
        frequency: w.frequency,
        phase: w.phase,
        input_const,
        input_is_layer,
        dom,
        dom_input,
    };
    let ubo = create_uniform(&ctx.device, bytemuck::bytes_of(&params), "wave-params");
    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("wave-bg"),
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
        label: Some("wave-cpass"),
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
    w: f32,
    dom: [f32; 4],
    src_dom: [f32; 4],
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
        w_coord: w,
        edge_mode: match t.edge_mode {
            EdgeMode::Clamp => 0,
            EdgeMode::Extend => 1,
        },
        dom,
        src_dom,
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
    dom: [f32; 4],
    dom_a: [f32; 4],
    dom_b: [f32; 4],
    dom_factor: [f32; 4],
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
        dom,
        dom_a,
        dom_b,
        dom_factor,
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
    dom: [f32; 4],
    dom_a: [f32; 4],
    dom_b: [f32; 4],
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
    let params = MinMaxParams { size: [size.0, size.1], mode, criterion, dom, dom_a, dom_b };
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
    dom: [f32; 4],
    dom_value: [f32; 4],
    dom_palette: [f32; 4],
) {
    let params = MapParams { size: [size.0, size.1], _pad: [0, 0], dom, dom_value, dom_palette };
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
    dom: [f32; 4],
) -> Result<(), BakeError> {
    use wgpu::util::DeviceExt;

    // Resolve unique layer-refs across stops → 0..N-1 input indices. Multiple
    // stops pointing at the same layer share one slot.
    let mut layer_to_input: HashMap<LayerId, u32> = HashMap::new();
    let mut input_pool_slots: Vec<u32> = Vec::new();
    let mut input_doms = [Domain::UNIT.packed(); MAX_RAMP_INPUTS];
    for s in &r.stops {
        if let ColorInput::Layer(id) = s.color {
            if !layer_to_input.contains_key(&id) {
                if input_pool_slots.len() >= MAX_RAMP_INPUTS {
                    return Err(BakeError::Unsupported(
                        "ColorRamp (>8 unique layer-referenced stops)",
                    ));
                }
                input_doms[input_pool_slots.len()] = domain_of(sched, id);
                layer_to_input.insert(id, input_pool_slots.len() as u32);
                input_pool_slots.push(slot_of(sched, id));
            }
        }
    }

    let params = RampParams {
        size: [size.0, size.1],
        stop_count: r.stops.len() as u32,
        space: blend_space_code(r.space),
        dom,
        input_doms,
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
    w: f32,
    dom: [f32; 4],
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
    let kernel = match n.kernel {
        NoiseKernel::Simplex => 0u32,
        NoiseKernel::Value => 1u32,
    };
    let fractal_mode = match n.fractal.mode {
        FractalMode::Standard => 0u32,
        FractalMode::Turbulence => 1u32,
        FractalMode::Ridged => 2u32,
    };
    let params = NoiseParams {
        size: [size.0, size.1],
        dims,
        range,
        output_mode,
        seed_base: ctx_seed.wrapping_add(n.seed_offset),
        frequency: n.frequency,
        w_coord: w,
        dom,
        // `Graph` rejects a period on the simplex kernel, so the shader
        // never has to decide which of the two the caller meant.
        period: n.period,
        octaves: n.fractal.octaves.clamp(1, texture_graph_core::noise::MAX_OCTAVES),
        lacunarity: n.fractal.lacunarity,
        gain: n.fractal.gain,
        fractal_mode,
        normalize: n.fractal.normalize as u32,
        kernel,
        _pad: [0; 3],
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
    missing_view: &wgpu::TextureView,
    dst_view: &wgpu::TextureView,
    size: (u32, u32),
    channel: OutputChannel,
    out: &OutputSlots,
    alpha_object: bool,
    z_px: u32,
    src_dom: [f32; 4],
) {
    match channel {
        OutputChannel::Color => dispatch_pack(
            ctx, encoder, pack_pipeline, pack_bgl,
            match out.color {
                Some(slot) => &pool_views[slot as usize],
                // Unconnected output color — pack the missing-texture grid.
                None => missing_view,
            },
            dst_view, size, 0, [0.0; 4],
            alpha_object, z_px, src_dom,
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
                alpha_object, z_px, src_dom,
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
                alpha_object, z_px, src_dom,
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
                alpha_object, z_px, src_dom,
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
    alpha_object: bool,
    z_px: u32,
    src_dom: [f32; 4],
) {
    let params = PackParams {
        size: [size.0, size.1],
        mode,
        alpha_object: alpha_object as u32,
        const_value,
        z_px,
        _pad: [0; 3],
        src_dom,
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

/// Fill `dst_view` with the missing-texture grid at slice coordinate `w`.
fn dispatch_missing(
    ctx: &DeviceCtx,
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    bgl: &wgpu::BindGroupLayout,
    dst_view: &wgpu::TextureView,
    size: (u32, u32),
    w: f32,
) {
    let params = MissingParams { size: [size.0, size.1], w_coord: w, _pad: 0 };
    let ubo = create_uniform(&ctx.device, bytemuck::bytes_of(&params), "missing-params");
    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("missing-bg"),
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
        label: Some("missing-cpass"),
        timestamp_writes: None,
    });
    cpass.set_pipeline(pipeline);
    cpass.set_bind_group(0, &bg, &[]);
    let (wg_x, wg_y) = workgroup_counts(size);
    cpass.dispatch_workgroups(wg_x, wg_y, 1);
}

/// Pool-format texture for the missing-input grid: storage-written by
/// `dispatch_missing`, sampled by whatever layer has the unconnected input.
fn make_missing_texture(
    device: &wgpu::Device,
    size: (u32, u32),
) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("tg-missing"),
        size: wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
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

/// Wall-clock for the bake timing logs.
///
/// `std::time::Instant::now()` compiles for `wasm32-unknown-unknown` and
/// then panics — there is no clock behind it. A consumer baking in the
/// browser should not lose a texture to a `log::debug!` it never reads, so
/// the timer is simply absent there and the log says `None`.
#[derive(Copy, Clone)]
struct BakeTimer {
    #[cfg(not(target_arch = "wasm32"))]
    start: std::time::Instant,
}

impl BakeTimer {
    fn start() -> Self {
        Self {
            #[cfg(not(target_arch = "wasm32"))]
            start: std::time::Instant::now(),
        }
    }

    fn elapsed(self) -> Option<std::time::Duration> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            Some(self.start.elapsed())
        }
        #[cfg(target_arch = "wasm32")]
        {
            None
        }
    }
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

/// 3D `Rgba8Unorm` volume assembled slice-by-slice via
/// `copy_texture_to_texture` and sampled by the solid 3D preview.
fn make_volume_texture(device: &wgpu::Device, res: u32, depth: u32, label: &str) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: res,
            height: res,
            depth_or_array_layers: depth,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
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


// ---- Single-channel pack ----------------------------------------------

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ScalarPackParams {
    size: [u32; 2],
    _pad: [u32; 2],
    /// Source layer's bake domain (min_u, min_v, ext_u, ext_v).
    src_dom: [f32; 4],
}

/// Draw the fullscreen triangle that copies one intermediate's L into
/// `dst_view`. A render pass, not a dispatch — see `pack_scalar.wgsl`.
fn draw_scalar_pack(
    ctx: &DeviceCtx,
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::RenderPipeline,
    bgl: &wgpu::BindGroupLayout,
    src_view: &wgpu::TextureView,
    dst_view: &wgpu::TextureView,
    size: (u32, u32),
    src_dom: [f32; 4],
) {
    let params = ScalarPackParams { size: [size.0, size.1], _pad: [0; 2], src_dom };
    let ubo = create_uniform(&ctx.device, bytemuck::bytes_of(&params), "scalar-pack-params");
    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("scalar-pack-bg"),
        layout: bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: ubo.as_entire_binding() },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(src_view),
            },
        ],
    });
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("scalar-pack-pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: dst_view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                // The triangle covers every pixel, so the clear is only
                // there to satisfy the load op.
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &bg, &[]);
    pass.draw(0..3, 0..1);
}

fn make_scalar_texture(
    device: &wgpu::Device,
    size: (u32, u32),
    format: ScalarFormat,
    label: &str,
) -> wgpu::Texture {
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
        format: format.texture_format(),
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

fn make_scalar_volume_texture(
    device: &wgpu::Device,
    res: u32,
    depth: u32,
    format: ScalarFormat,
    label: &str,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: res,
            height: res,
            depth_or_array_layers: depth,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: format.texture_format(),
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

/// The shader module and bind-group layout every scalar-pack pipeline
/// shares; only the colour-target format differs between them.
fn make_scalar_pack_shader(
    device: &wgpu::Device,
) -> (wgpu::ShaderModule, wgpu::BindGroupLayout) {
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("scalar-pack-bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
        ],
    });
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("scalar-pack-shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/pack_scalar.wgsl").into()),
    });
    (shader, bgl)
}

fn make_scalar_pack_pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    bgl: &wgpu::BindGroupLayout,
    format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("scalar-pack-pl"),
        bind_group_layouts: &[Some(bgl)],
        ..Default::default()
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("scalar-pack-pipeline"),
        layout: Some(&pl),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    })
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

fn make_wave_pipeline(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
) -> wgpu::ComputePipeline {
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("wave-pl"),
        bind_group_layouts: &[Some(bgl)],
        ..Default::default()
    });
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("wave-shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/wave.wgsl").into()),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("wave-pipeline"),
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

fn make_missing_pipeline(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
) -> wgpu::ComputePipeline {
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("missing-pl"),
        bind_group_layouts: &[Some(bgl)],
        ..Default::default()
    });
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("missing-shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/missing.wgsl").into()),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("missing-pipeline"),
        layout: Some(&pl),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
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
