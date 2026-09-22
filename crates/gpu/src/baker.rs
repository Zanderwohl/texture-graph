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
    Axis, BlendMode, Coordinate, BlendSpace, ColorInput, ColorRamp, CoordMode, Criterion, EvalCtx,
    FractalMode, Graph, HeightToNormal, LayerId, LayerKind, MinMax, MinMaxMode, Mix, Noise,
    NoiseDims, NoiseKernel, NoiseOutput, NoiseRange, RadialDim, ScalarInput, Transform,
    Warp, WarpMode, Wave, WaveShape,
};

use texture_graph_core::EdgeMode;

use crate::device::DeviceCtx;
use crate::sphere::{self, Placement, PointMap};
use crate::schedule::{
    Domain, IdList, OutputSlots, ScalarSlot, Schedule, schedule, schedule_layer,
    schedule_previews,
};

const MAX_RAMP_STOPS: usize = 16;
/// Filterable on WebGPU, and unclamped, so a height past `[0, 1]` keeps its
/// slope.
const BUMP_FORMAT: ScalarFormat = ScalarFormat::R16Float;
const POOL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;
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
/// slice center. The 3D preview samples these at object-space position rather
/// than UV-mapping one flat slice.
pub struct VolumeOutput {
    pub color: wgpu::Texture,
    pub roughness: wgpu::Texture,
    pub metallic: wgpu::Texture,
    pub normal: wgpu::Texture,
    /// (width, height, depth) in texels.
    pub size: (u32, u32, u32),
    /// Set when the Output's normal is a [`HeightToNormal`]. `normal` is
    /// baked per slice and has no slope along w, so a solid preview bumps
    /// from this height instead.
    pub bump: Option<SolidBump>,
}

pub struct SolidBump {
    /// `R16Float`: the [`HeightToNormal`] source's L, unclamped.
    pub height: wgpu::Texture,
    /// [`HeightToNormal::strength`]: the slope scale per unit of sample space.
    pub strength: f32,
}

struct BumpJob {
    source: LayerId,
    strength: f32,
    slice: wgpu::Texture,
    slice_view: wgpu::TextureView,
    volume: wgpu::Texture,
}

/// A [`Baker::bake_volume`] in progress; see [`Baker::begin_volume`].
pub struct VolumeJob {
    graph: Graph,
    /// Already resolved against the graph's parameters.
    eval_ctx: EvalCtx,
    sched: Schedule,
    res: u32,
    depth: u32,
    next_z: u32,
    /// Behind `pool_views` and `missing_view`; held for as long as they are.
    _scratch: Vec<wgpu::Texture>,
    pool_views: Vec<wgpu::TextureView>,
    missing_view: wgpu::TextureView,
    slices: Vec<wgpu::Texture>,
    slice_views: Vec<wgpu::TextureView>,
    volumes: Vec<wgpu::Texture>,
    chan_doms: [[f32; 4]; 4],
    bump: Option<BumpJob>,
    started: BakeTimer,
}

impl VolumeJob {
    /// Width and height in texels.
    pub fn res(&self) -> u32 {
        self.res
    }

    pub fn is_done(&self) -> bool {
        self.next_z >= self.depth
    }

    /// Compute dispatches one slice records, for sizing a step to a budget.
    pub fn dispatches_per_slice(&self) -> u32 {
        // + the missing-texture refill and the four channel packs.
        self.sched.order.len() as u32 + 5
    }

    /// Slices not yet reached are empty until [`Self::is_done`].
    pub fn into_output(self) -> VolumeOutput {
        let mut it = self.volumes.into_iter();
        VolumeOutput {
            color: it.next().unwrap(),
            roughness: it.next().unwrap(),
            metallic: it.next().unwrap(),
            normal: it.next().unwrap(),
            size: (self.res, self.res, self.depth),
            bump: self.bump.map(|b| SolidBump { height: b.volume, strength: b.strength }),
        }
    }
}

/// Texel format for a single-channel bake.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum ScalarFormat {
    /// Clamped to `[0, 1]` by the format.
    R8Unorm,
    /// Unclamped, and filterable on WebGPU.
    R16Float,
    /// Unclamped. Not filterable on WebGPU without the `float32-filterable`
    /// feature: sampling it with a linear sampler is a validation error.
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

/// One scalar field over the six faces of a cube. See
/// [`texture_graph_core::sphere`].
///
/// Six array layers in `+X, -X, +Y, -Y, +Z, -Z` order, as a `Cube` view
/// expects. [`crate::read_scalar_volume`] reads it with
/// `size = (face, face, 6)`.
pub struct ScalarCube {
    pub texture: wgpu::Texture,
    /// Edge length of a face in texels.
    pub face: u32,
    pub format: ScalarFormat,
}

/// A graph's color output on a sphere; see [`Baker::bake_color_cube`].
pub struct ColorCube {
    pub texture: wgpu::Texture,
    /// Edge length of a face in texels.
    pub face: u32,
}

#[derive(Copy, Clone, Debug)]
enum Slice {
    Plane { w: f32 },
    /// Face `face`, sampled at the point `map` moves each texel's to.
    CubeFace { face: u32, map: PointMap },
}

impl Slice {
    /// The `w` a stage that only knows planes should use.
    fn w(self) -> f32 {
        match self {
            Slice::Plane { w } => w,
            Slice::CubeFace { .. } => texture_graph_core::FLAT_W,
        }
    }

    fn point_map(self) -> PointMap {
        match self {
            Slice::Plane { .. } => sphere::IDENTITY,
            Slice::CubeFace { map, .. } => map,
        }
    }

    /// The shaders' `face` uniform: 0 for a plane, `k + 1` for face `k`.
    fn face_code(self) -> u32 {
        match self {
            Slice::Plane { .. } => 0,
            Slice::CubeFace { face, .. } => face + 1,
        }
    }
}

/// One scalar field over a `res × res × depth` volume, `w` at each slice
/// center.
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
    coordinate_pipeline: wgpu::ComputePipeline,
    transform_pipeline: wgpu::ComputePipeline,
    transform_bgl: wgpu::BindGroupLayout,
    mix_pipeline: wgpu::ComputePipeline,
    mix_bgl: wgpu::BindGroupLayout,
    map_pipeline: wgpu::ComputePipeline,
    map_bgl: wgpu::BindGroupLayout,
    min_max_pipeline: wgpu::ComputePipeline,
    ramp_pipeline: wgpu::ComputePipeline,
    ramp_bgl: wgpu::BindGroupLayout,
    /// Bound into unused ramp input slots, so they never alias an output
    /// storage binding.
    #[allow(dead_code)]
    dummy_input: wgpu::Texture,
    dummy_input_view: wgpu::TextureView,
    h2n_pipeline: wgpu::ComputePipeline,
    wave_pipeline: wgpu::ComputePipeline,
    warp_pipeline: wgpu::ComputePipeline,
    missing_pipeline: wgpu::ComputePipeline,
    /// Bound wherever a layer input is `None`. Refilled per bake, and per
    /// slice for volumes.
    missing_tex: Option<wgpu::Texture>,
    missing_view: Option<wgpu::TextureView>,
    pack_pipeline: wgpu::ComputePipeline,
    pack_bgl: wgpu::BindGroupLayout,
    solid_pipeline: wgpu::ComputePipeline,
    solid_bgl: wgpu::BindGroupLayout,
    /// Scalar pipelines are built on first use, one per target format;
    /// most sessions never bake a scalar.
    scalar_shader: wgpu::ShaderModule,
    scalar_bgl: wgpu::BindGroupLayout,
    scalar_pipelines: HashMap<wgpu::TextureFormat, wgpu::RenderPipeline>,
}

impl Baker {
    pub fn new(ctx: DeviceCtx) -> Self {
        let (color_pipeline, color_bgl) = make_color_pipeline(&ctx.device);
        let noise_pipeline = make_noise_pipeline(&ctx.device, &color_bgl);
        let coordinate_pipeline = make_coordinate_pipeline(&ctx.device, &color_bgl);
        let (transform_pipeline, transform_bgl) = make_transform_pipeline(&ctx.device);
        let (mix_pipeline, mix_bgl) = make_mix_pipeline(&ctx.device);
        let (map_pipeline, map_bgl) = make_map_pipeline(&ctx.device);
        let min_max_pipeline = make_min_max_pipeline(&ctx.device, &map_bgl);
        let (ramp_pipeline, ramp_bgl) = make_ramp_pipeline(&ctx.device);
        let h2n_pipeline = make_h2n_pipeline(&ctx.device, &transform_bgl);
        let wave_pipeline = make_wave_pipeline(&ctx.device, &transform_bgl);
        let warp_pipeline = make_warp_pipeline(&ctx.device, &map_bgl);
        let dummy_input = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("tg-dummy-input"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            // No STORAGE_BINDING, so it cannot alias an output slot.
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let dummy_input_view = dummy_input.create_view(&wgpu::TextureViewDescriptor::default());
        let missing_pipeline = make_missing_pipeline(&ctx.device, &color_bgl);
        let (pack_pipeline, pack_bgl) = make_pack_pipeline(&ctx.device);
        let (solid_pipeline, solid_bgl) = make_solid_pipeline(&ctx.device);
        let (scalar_shader, scalar_bgl) = make_scalar_pack_shader(&ctx.device);
        log::debug!("baker created compute_pipelines=14 pool_format={:?}", POOL_FORMAT);
        Self {
            ctx,
            pool: Vec::new(),
            pool_views: Vec::new(),
            pool_size: (0, 0),
            color_pipeline,
            color_bgl,
            noise_pipeline,
            coordinate_pipeline,
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
            warp_pipeline,
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

    /// Bake 128² `Rgba8Unorm` thumbnails for `wanted`, or for every layer
    /// when it is `None`.
    ///
    /// Only `wanted` and what they transitively read are dispatched. The
    /// returned map holds exactly the layers asked for.
    pub fn bake_previews(
        &mut self,
        graph: &Graph,
        eval_ctx: &EvalCtx,
        wanted: Option<&HashSet<LayerId>>,
    ) -> Result<HashMap<LayerId, wgpu::Texture>, BakeError> {
        const PREVIEW_SIZE: (u32, u32) = (128, 128);
        let t0 = BakeTimer::start();
        log::debug!(
            "bake_previews start size={}x{} wanted={:?} graph_layers={}",
            PREVIEW_SIZE.0,
            PREVIEW_SIZE.1,
            wanted.map(|w| w.len()),
            graph.layers.len(),
        );
        let eval_ctx = &graph.resolve_params(eval_ctx);
        let sched = schedule_previews(graph, wanted, eval_ctx)?;
        let size = PREVIEW_SIZE;
        let device = self.ctx.device.clone();
        log_dispatches(graph, &sched, eval_ctx);

        // Not `self.pool`, which is sized to the output resolution.
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
        log::debug!(
            "bake_previews alloc inter={} format={:?} out={} size={}x{}",
            sched.peak_slots,
            POOL_FORMAT,
            packed.len(),
            size.0,
            size.1,
        );

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("tg-bake-previews"),
            });

        // `self.missing_view` is pool-sized.
        let (_missing_tex, missing_view) = make_missing_texture(&device, size);
        dispatch_missing(
            &self.ctx, &mut encoder, &self.missing_pipeline, &self.color_bgl,
            &missing_view, size, texture_graph_core::FLAT_W,
        );

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
                Slice::Plane { w: 0.5 },
                &missing_view,
            )?;
        }

        // Thumbnails show [0, 1] even when the bake domain is wider.
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
        log::debug!(
            "bake_previews done layers={} packed={} record+submit={:?}",
            sched.order.len(),
            packed.len(),
            t0.elapsed(),
        );
        Ok(outputs)
    }

    /// Writes one layer into `pool_views[dst_slot]`.
    fn dispatch_kind(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        layer: &texture_graph_core::Layer,
        eval_ctx: &EvalCtx,
        sched: &Schedule,
        pool_views: &[wgpu::TextureView],
        size: (u32, u32),
        dst_slot: usize,
        at: Slice,
        missing_view: &wgpu::TextureView,
    ) -> Result<(), BakeError> {
        let w = at.w();
        log::trace!(
            "dispatch layer id={} kind={} slot={dst_slot} w={w} face={}",
            layer.id,
            layer.kind.category_label(),
            at.face_code(),
        );
        let resolve = |opt: Option<texture_graph_core::LayerId>| -> &wgpu::TextureView {
            match opt {
                Some(id) => &pool_views[slot_of(sched, id) as usize],
                None => missing_view,
            }
        };
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
                    at.face_code(),
                    at.point_map(),
                    own_dom,
                );
            }
            LayerKind::Coordinate(c) => {
                dispatch_coordinate(
                    &self.ctx,
                    encoder,
                    &self.coordinate_pipeline,
                    &self.color_bgl,
                    &pool_views[dst_slot],
                    c,
                    size,
                    w,
                    at.face_code(),
                    at.point_map(),
                    own_dom,
                );
            }
            LayerKind::Transform(t) => {
                // On a face the map already moved the source's points, so
                // this is a copy.
                let copy;
                let t = match at {
                    Slice::Plane { .. } => t,
                    Slice::CubeFace { .. } => {
                        copy = Transform {
                            offset: [0.0; 3],
                            rotate_uv: 0.0,
                            scale: [1.0; 3],
                            coord_mode: CoordMode::Passthrough,
                            edge_mode: EdgeMode::Clamp,
                            ..*t
                        };
                        &copy
                    }
                };
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
                let (factor_view, dom_factor) = match &m.factor {
                    ScalarInput::Layer(id) => (resolve(Some(*id)), dom_opt(Some(*id))),
                    // `a` is a placeholder; the shader does not read it.
                    // Params are already resolved to numbers.
                    ScalarInput::Const(_) | ScalarInput::Param(_) => (a_view, dom_opt(m.a)),
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
                    eval_ctx,
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
            LayerKind::Warp(wp) => {
                dispatch_warp(
                    &self.ctx,
                    encoder,
                    &self.warp_pipeline,
                    &self.map_bgl,
                    &pool_views[dst_slot],
                    resolve(wp.source),
                    resolve(wp.by),
                    wp,
                    size,
                    own_dom,
                    dom_opt(wp.source),
                    dom_opt(wp.by),
                );
            }
            LayerKind::Wave(wv) => {
                // A const input is not read, but the slot must be bound.
                let (src_view, dom_input) = match &wv.input {
                    ScalarInput::Layer(id) => (resolve(Some(*id)), dom_opt(Some(*id))),
                    ScalarInput::Const(_) | ScalarInput::Param(_) => {
                        (missing_view, Domain::UNIT.packed())
                    }
                };
                dispatch_wave(
                    &self.ctx,
                    encoder,
                    &self.wave_pipeline,
                    &self.transform_bgl,
                    &pool_views[dst_slot],
                    src_view,
                    wv,
                    eval_ctx,
                    size,
                    own_dom,
                    dom_input,
                );
            }
            LayerKind::ColorRamp(r) => {
                if r.stops.len() > MAX_RAMP_STOPS {
                    log::debug!(
                        "dispatch unsupported id={} name={:?} stops={} max={MAX_RAMP_STOPS}",
                        layer.id,
                        layer.name,
                        r.stops.len(),
                    );
                    return Err(BakeError::Unsupported("ColorRamp (>16 stops)"));
                }
                dispatch_ramp(
                    &self.ctx,
                    encoder,
                    &self.ramp_pipeline,
                    &self.ramp_bgl,
                    &pool_views[dst_slot],
                    r,
                    eval_ctx,
                    sched,
                    pool_views,
                    &self.dummy_input_view,
                    size,
                    own_dom,
                )
                .inspect_err(|e| {
                    log::debug!(
                        "dispatch unsupported id={} name={:?} err={e}",
                        layer.id,
                        layer.name,
                    );
                })?;
            }
        }
        Ok(())
    }

    fn ensure_pool(&mut self, size: (u32, u32), needed: u32) {
        let resize = self.pool_size != size;
        if resize {
            log::debug!(
                "pool resize from={}x{} to={}x{} dropped={}",
                self.pool_size.0,
                self.pool_size.1,
                size.0,
                size.1,
                self.pool.len(),
            );
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
        if self.pool.len() < needed as usize {
            log::debug!(
                "pool grow from={} to={needed} size={}x{} format={:?} bytes≈{}MiB",
                self.pool.len(),
                size.0,
                size.1,
                POOL_FORMAT,
                (needed as u64) * (size.0 as u64) * (size.1 as u64) * 16 / (1024 * 1024),
            );
        }
        while self.pool.len() < needed as usize {
            let tex = make_pool_texture(&self.ctx.device, size);
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            self.pool.push(tex);
            self.pool_views.push(view);
        }
    }

    /// Bake the graph's output at `size`.
    ///
    /// `object_alpha = false` composites partial alpha over a gray checker,
    /// for flat previews; `true` keeps the alpha for the 3D preview to blend.
    pub fn bake_output(
        &mut self,
        graph: &Graph,
        size: (u32, u32),
        eval_ctx: &EvalCtx,
        object_alpha: bool,
    ) -> Result<BakeOutput, BakeError> {
        let t0 = BakeTimer::start();
        log::debug!(
            "bake_output start size={}x{} object_alpha={object_alpha} seed={} graph_layers={}",
            size.0,
            size.1,
            eval_ctx.seed,
            graph.layers.len(),
        );
        // After this, a `Param` socket is a constant.
        let eval_ctx = &graph.resolve_params(eval_ctx);
        let sched = schedule(graph, eval_ctx)?;
        self.ensure_pool(size, sched.peak_slots.max(1));
        log_dispatches(graph, &sched, eval_ctx);

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
                Slice::Plane { w: 0.5 },
                missing_view,
            )?;
        }

        let chan_dom = |id: Option<LayerId>| -> [f32; 4] {
            match id {
                Some(id) => domain_of(&sched, id),
                None => Domain::UNIT.packed(),
            }
        };
        let color_dom = chan_dom(graph.output.color);
        let scalar_chan_dom = |si: &ScalarInput| match si {
            ScalarInput::Layer(id) => chan_dom(Some(*id)),
            _ => Domain::UNIT.packed(),
        };
        let rough_dom = scalar_chan_dom(&graph.output.roughness);
        let metal_dom = scalar_chan_dom(&graph.output.metallic);
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
    /// At 256³ in `R8Unorm` that is 16 MB, against 268 MB for
    /// [`Baker::bake_output`].
    ///
    /// Only `layer` and what it transitively reads are dispatched; the
    /// graph's Output is not consulted.
    ///
    /// Texels are the raw scalar, not a display-encoded gray. `R8Unorm`
    /// clamps to `[0, 1]`; the float formats keep signed values.
    ///
    /// The texture has `TEXTURE_BINDING | COPY_SRC`, so a consumer on the
    /// same device can sample it without readback.
    pub fn bake_scalar(
        &mut self,
        graph: &Graph,
        layer: LayerId,
        size: (u32, u32),
        format: ScalarFormat,
        eval_ctx: &EvalCtx,
    ) -> Result<wgpu::Texture, BakeError> {
        let t0 = BakeTimer::start();
        log::debug!(
            "bake_scalar start layer={layer} size={}x{} format={format:?} seed={}",
            size.0,
            size.1,
            eval_ctx.seed,
        );
        let eval_ctx = &graph.resolve_params(eval_ctx);
        let sched = schedule_layer(graph, layer)?;
        self.ensure_pool(size, sched.peak_slots.max(1));
        log_dispatches(graph, &sched, eval_ctx);

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
            &|_| Slice::Plane { w: texture_graph_core::FLAT_W },
            format,
            &dst.create_view(&wgpu::TextureViewDescriptor::default()),
            eval_ctx,
        )?;
        self.ctx.queue.submit([encoder.finish()]);
        log::debug!(
            "bake_scalar {}x{} ({format:?}): layers={} record+submit={:?}",
            size.0,
            size.1,
            sched.order.len(),
            t0.elapsed(),
        );
        Ok(dst)
    }

    /// [`Baker::bake_scalar`] over a `res × res × depth` volume, sampled at
    /// slice centers as [`Baker::bake_volume`] does, with the same
    /// limitation on a Transform's w offset/scale.
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
        log::debug!(
            "bake_scalar_volume start layer={layer} res={res} depth={depth} format={format:?} \
             seed={}",
            eval_ctx.seed,
        );
        let eval_ctx = &graph.resolve_params(eval_ctx);
        let sched = schedule_layer(graph, layer)?;
        let size = (res, res);
        self.ensure_pool(size, sched.peak_slots.max(1));
        log_dispatches(graph, &sched, eval_ctx);

        let device = self.ctx.device.clone();
        // WebGPU has no 2D view of a 3D texture, so render to a 2D slice and
        // copy it in.
        let slice = make_scalar_texture(&device, size, format, "tg-scalar-vol-slice");
        let slice_view = slice.create_view(&wgpu::TextureViewDescriptor::default());
        let texture = make_scalar_volume_texture(&device, res, depth, format, "tg-scalar-vol");

        for z in 0..depth {
            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("tg-bake-scalar-volume"),
                });
            let w = (z as f32 + 0.5) / depth as f32;
            log::trace!("bake_scalar_volume slice z={z} w={w}");
            self.record_scalar_slice(
                &mut encoder, graph, &sched, layer, size, &|_| Slice::Plane { w }, format,
                &slice_view,
                eval_ctx,
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

    /// [`Baker::bake_scalar`] on a sphere: each texel of the six cube faces
    /// samples its direction's point on the sphere inscribed in the unit
    /// cube.
    ///
    /// Returns `Unsupported` if `layer` reads a ColorRamp, Transform, Map,
    /// HeightToNormal or Warp, which re-sample at other (u, v).
    pub fn bake_scalar_cube(
        &mut self,
        graph: &Graph,
        layer: LayerId,
        face: u32,
        format: ScalarFormat,
        eval_ctx: &EvalCtx,
    ) -> Result<ScalarCube, BakeError> {
        let t0 = BakeTimer::start();
        log::debug!(
            "bake_scalar_cube start layer={layer} face={face} format={format:?} seed={}",
            eval_ctx.seed,
        );
        let eval_ctx = &graph.resolve_params(eval_ctx);
        let (sched, placed) = Self::sphere_schedule(graph, layer).inspect_err(|e| {
            log::debug!("bake_scalar_cube unsupported err={e}");
        })?;
        let size = (face, face);
        self.ensure_pool(size, sched.peak_slots.max(1));
        log_dispatches(graph, &sched, eval_ctx);

        let device = self.ctx.device.clone();
        let slice = make_scalar_texture(&device, size, format, "tg-scalar-cube-face");
        let slice_view = slice.create_view(&wgpu::TextureViewDescriptor::default());
        let texture = make_scalar_cube_texture(&device, face, format, "tg-scalar-cube");

        for k in 0..texture_graph_core::CUBE_FACES {
            log::trace!("bake_scalar_cube face k={k}");
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("tg-bake-scalar-cube"),
            });
            let at = |id: LayerId| match placed.get(&id) {
                Some(Placement::Sphere(map)) => Slice::CubeFace { face: k, map: *map },
                _ => Slice::Plane { w: texture_graph_core::FLAT_W },
            };
            self.record_scalar_slice(
                &mut encoder, graph, &sched, layer, size, &at, format, &slice_view, eval_ctx,
            )?;
            copy_face(&mut encoder, &slice, &texture, face, k);
            self.ctx.queue.submit([encoder.finish()]);
        }

        log::debug!(
            "bake_scalar_cube 6×{face}² ({format:?}): layers/face={} record+submit={:?}",
            sched.order.len(),
            t0.elapsed(),
        );
        Ok(ScalarCube { texture, face, format })
    }

    /// Dispatch every layer for one slice, then pack `layer`'s slot into
    /// `dst_view`.
    #[allow(clippy::too_many_arguments)]
    fn record_scalar_slice(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        graph: &Graph,
        sched: &Schedule,
        layer: LayerId,
        size: (u32, u32),
        at: &dyn Fn(LayerId) -> Slice,
        format: ScalarFormat,
        dst_view: &wgpu::TextureView,
        eval_ctx: &EvalCtx,
    ) -> Result<(), BakeError> {
        self.record_layers(encoder, graph, sched, size, at, eval_ctx)?;
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

    /// Dispatch every scheduled layer, each at its own slice.
    fn record_layers(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        graph: &Graph,
        sched: &Schedule,
        size: (u32, u32),
        at: &dyn Fn(LayerId) -> Slice,
        eval_ctx: &EvalCtx,
    ) -> Result<(), BakeError> {
        let missing_view = self.missing_view.clone().expect("ensure_pool made one");
        let w = sched.order.first().map_or(texture_graph_core::FLAT_W, |&id| at(id).w());
        dispatch_missing(
            &self.ctx, encoder, &self.missing_pipeline, &self.color_bgl, &missing_view, size, w,
        );
        for &id in &sched.order {
            let slot = *sched.slot_of.get(&id).unwrap() as usize;
            let l = graph.get(id).unwrap();
            self.dispatch_kind(
                encoder, l, eval_ctx, sched, &self.pool_views, size, slot, at(id), &missing_view,
            )?;
        }
        Ok(())
    }

    /// The schedule for a sphere bake of `root`, and where each layer
    /// bakes. See [`crate::sphere`].
    fn sphere_schedule(
        graph: &Graph,
        root: LayerId,
    ) -> Result<(Schedule, HashMap<LayerId, Placement>), BakeError> {
        let mut sched = schedule_layer(graph, root)?;
        let placed = sphere::plan(graph, root)?;
        // A face covers exactly its own (u, v); an Extend transform's wider
        // domain is the plane's, and the point map has replaced it.
        for (&id, at) in &placed {
            if let Placement::Sphere(_) = at {
                sched.domain_of.insert(id, Domain::UNIT);
            }
        }
        Ok((sched, placed))
    }

    /// Bake the graph's color output on the sphere into a cubemap:
    /// `Rgba8Unorm`, sRGB-encoded and with straight alpha, six array layers
    /// in the order [`ScalarCube`] documents. Read it back with
    /// [`crate::read_rgba8_layers`].
    pub fn bake_color_cube(
        &mut self,
        graph: &Graph,
        face: u32,
        eval_ctx: &EvalCtx,
    ) -> Result<ColorCube, BakeError> {
        let t0 = BakeTimer::start();
        log::debug!("bake_color_cube start face={face} seed={}", eval_ctx.seed);
        let eval_ctx = &graph.resolve_params(eval_ctx);
        let root = graph
            .output
            .color
            .ok_or(BakeError::Unsupported("a color cube of a graph with no color output"))?;
        let (sched, placed) = Self::sphere_schedule(graph, root).inspect_err(|e| {
            log::debug!("bake_color_cube unsupported err={e}");
        })?;
        let size = (face, face);
        self.ensure_pool(size, sched.peak_slots.max(1));
        log_dispatches(graph, &sched, eval_ctx);

        let device = self.ctx.device.clone();
        let slice = make_output_texture(&device, size, "tg-color-cube-face");
        let slice_view = slice.create_view(&wgpu::TextureViewDescriptor::default());
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("tg-color-cube"),
            size: wgpu::Extent3d {
                width: face,
                height: face,
                depth_or_array_layers: texture_graph_core::CUBE_FACES,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let src_slot = *sched.slot_of.get(&root).expect("schedule_layer schedules its root");

        for k in 0..texture_graph_core::CUBE_FACES {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("tg-bake-color-cube"),
            });
            let at = |id: LayerId| match placed.get(&id) {
                Some(Placement::Sphere(map)) => Slice::CubeFace { face: k, map: *map },
                _ => Slice::Plane { w: texture_graph_core::FLAT_W },
            };
            self.record_layers(&mut encoder, graph, &sched, size, &at, eval_ctx)?;
            dispatch_pack(
                &self.ctx,
                &mut encoder,
                &self.pack_pipeline,
                &self.pack_bgl,
                &self.pool_views[src_slot as usize],
                &slice_view,
                size,
                0,
                [0.0; 4],
                true,
                0,
                Domain::UNIT.packed(),
            );
            copy_face(&mut encoder, &slice, &texture, face, k);
            self.ctx.queue.submit([encoder.finish()]);
        }

        log::debug!(
            "bake_color_cube 6×{face}²: layers/face={} record+submit={:?}",
            sched.order.len(),
            t0.elapsed(),
        );
        Ok(ColorCube { texture, face })
    }

    /// Returns a clone (an `Arc` bump) so `self` is not borrowed afterwards.
    fn scalar_pipeline(&mut self, format: ScalarFormat) -> wgpu::RenderPipeline {
        let tf = format.texture_format();
        if self.scalar_pipelines.contains_key(&tf) {
            log::trace!("scalar pipeline reused format={tf:?}");
        } else {
            log::debug!("scalar pipeline compiled format={tf:?} shader=pack_scalar.wgsl");
        }
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

    /// Bake the graph as four 3D textures by running the 2D pipeline once
    /// per slice, with `w` at the slice center.
    ///
    /// Can take a large part of a second of CPU at 256³; see
    /// [`Baker::begin_volume`] to spread it over frames.
    ///
    /// A Transform's w offset/scale has no effect, because each slice has
    /// its inputs baked only at its own w.
    pub fn bake_volume(
        &mut self,
        graph: &Graph,
        res: u32,
        depth: u32,
        eval_ctx: &EvalCtx,
    ) -> Result<VolumeOutput, BakeError> {
        let mut job = self.begin_volume(graph, res, depth, eval_ctx)?;
        self.step_volume(&mut job, u32::MAX)?;
        Ok(job.into_output())
    }

    /// Start a [`Baker::bake_volume`] that [`Baker::step_volume`] carries
    /// out a few slices at a time.
    ///
    /// The job snapshots the graph and owns its intermediates, so the graph
    /// can be edited and other bakes run between steps.
    pub fn begin_volume(
        &mut self,
        graph: &Graph,
        res: u32,
        depth: u32,
        eval_ctx: &EvalCtx,
    ) -> Result<VolumeJob, BakeError> {
        let started = BakeTimer::start();
        log::debug!(
            "begin_volume start res={res} depth={depth} seed={} graph_layers={}",
            eval_ctx.seed,
            graph.layers.len(),
        );
        let eval_ctx = graph.resolve_params(eval_ctx);
        let sched = schedule(graph, &eval_ctx)?;
        let size = (res, res);
        let device = &self.ctx.device;
        log_dispatches(graph, &sched, &eval_ctx);
        log::debug!(
            "begin_volume alloc pool={} format={:?} size={res}x{res} slices=4 volumes=4 \
             volume_format={:?} volume_bytes≈{}MiB",
            sched.peak_slots.max(1),
            POOL_FORMAT,
            wgpu::TextureFormat::Rgba8Unorm,
            4 * (res as u64) * (res as u64) * (depth as u64) * 4 / (1024 * 1024),
        );

        let pool: Vec<wgpu::Texture> =
            (0..sched.peak_slots.max(1)).map(|_| make_pool_texture(device, size)).collect();
        let pool_views: Vec<wgpu::TextureView> = pool
            .iter()
            .map(|t| t.create_view(&wgpu::TextureViewDescriptor::default()))
            .collect();
        let (missing_tex, missing_view) = make_missing_texture(device, size);
        let slices: Vec<wgpu::Texture> = ["color", "rough", "metal", "normal"]
            .iter()
            .map(|n| make_output_texture(device, size, &format!("tg-vol-slice-{n}")))
            .collect();
        let slice_views: Vec<wgpu::TextureView> = slices
            .iter()
            .map(|t| t.create_view(&wgpu::TextureViewDescriptor::default()))
            .collect();
        let volumes: Vec<wgpu::Texture> = ["color", "rough", "metal", "normal"]
            .iter()
            .map(|n| make_volume_texture(device, res, depth, &format!("tg-vol-{n}")))
            .collect();

        let chan_dom = |id: Option<LayerId>| -> [f32; 4] {
            match id {
                Some(id) => domain_of(&sched, id),
                None => Domain::UNIT.packed(),
            }
        };
        let chan_doms: [[f32; 4]; 4] = [
            chan_dom(graph.output.color),
            match &graph.output.roughness {
                ScalarInput::Layer(id) => chan_dom(Some(*id)),
                _ => Domain::UNIT.packed(),
            },
            match &graph.output.metallic {
                ScalarInput::Layer(id) => chan_dom(Some(*id)),
                _ => Domain::UNIT.packed(),
            },
            chan_dom(graph.output.normal),
        ];

        let bump = graph
            .output
            .normal
            .and_then(|id| match &graph.get(id)?.kind {
                LayerKind::HeightToNormal(HeightToNormal { source: Some(src), strength }) => {
                    Some((*src, *strength))
                }
                _ => None,
            })
            .map(|(source, strength)| {
                let slice = make_scalar_texture(device, size, BUMP_FORMAT, "tg-vol-slice-height");
                let slice_view = slice.create_view(&wgpu::TextureViewDescriptor::default());
                let volume =
                    make_scalar_volume_texture(device, res, depth, BUMP_FORMAT, "tg-vol-height");
                BumpJob { source, strength, slice, slice_view, volume }
            });

        Ok(VolumeJob {
            graph: graph.clone(),
            eval_ctx,
            sched,
            res,
            depth,
            next_z: 0,
            _scratch: pool.into_iter().chain([missing_tex]).collect(),
            pool_views,
            missing_view,
            slices,
            slice_views,
            volumes,
            chan_doms,
            bump,
            started,
        })
    }

    /// Record and submit up to `max_slices` more of `job`'s slices. Returns
    /// whether the job is finished.
    pub fn step_volume(&mut self, job: &mut VolumeJob, max_slices: u32) -> Result<bool, BakeError> {
        const CHANNELS: [OutputChannel; 4] = [
            OutputChannel::Color,
            OutputChannel::Roughness,
            OutputChannel::Metallic,
            OutputChannel::Normal,
        ];
        let t0 = BakeTimer::start();
        let first = job.next_z;
        let size = (job.res, job.res);
        let bump_pipeline = job.bump.as_ref().map(|_| self.scalar_pipeline(BUMP_FORMAT));
        let stop = job.next_z.saturating_add(max_slices).min(job.depth);
        for z in job.next_z..stop {
            // One submit per slice. On Metal each compute pass is a command
            // buffer outstanding until submit, and wgpu-metal caps those at
            // 4096; one encoder for all slices lost the device on a
            // ~30-layer graph × 64 slices.
            let mut encoder = self
                .ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("tg-bake-volume"),
                });
            let w = (z as f32 + 0.5) / job.depth as f32;
            // The grid alternates along w.
            dispatch_missing(
                &self.ctx, &mut encoder, &self.missing_pipeline, &self.color_bgl,
                &job.missing_view, size, w,
            );
            for &id in &job.sched.order {
                let slot = *job.sched.slot_of.get(&id).unwrap() as usize;
                let layer = job.graph.get(id).unwrap();
                self.dispatch_kind(
                    &mut encoder,
                    layer,
                    &job.eval_ctx,
                    &job.sched,
                    &job.pool_views,
                    size,
                    slot,
                    Slice::Plane { w },
                    &job.missing_view,
                )?;
                // A later layer may reuse the slot, so pack the height now.
                if let (Some(bump), Some(pipeline)) = (&job.bump, &bump_pipeline)
                    && bump.source == id
                {
                    draw_scalar_pack(
                        &self.ctx,
                        &mut encoder,
                        pipeline,
                        &self.scalar_bgl,
                        &job.pool_views[slot],
                        &bump.slice_view,
                        size,
                        domain_of(&job.sched, id),
                    );
                }
            }
            if let Some(bump) = &job.bump {
                encoder.copy_texture_to_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &bump.slice,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::TexelCopyTextureInfo {
                        texture: &bump.volume,
                        mip_level: 0,
                        origin: wgpu::Origin3d { x: 0, y: 0, z },
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::Extent3d { width: job.res, height: job.res, depth_or_array_layers: 1 },
                );
            }
            for (i, channel) in CHANNELS.iter().enumerate() {
                pack_channel(
                    &self.ctx,
                    &mut encoder,
                    &self.pack_pipeline,
                    &self.pack_bgl,
                    &self.solid_pipeline,
                    &self.solid_bgl,
                    &job.pool_views,
                    &job.missing_view,
                    &job.slice_views[i],
                    size,
                    *channel,
                    &job.sched.output_slots,
                    true,
                    z,
                    job.chan_doms[i],
                );
                encoder.copy_texture_to_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &job.slices[i],
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::TexelCopyTextureInfo {
                        texture: &job.volumes[i],
                        mip_level: 0,
                        origin: wgpu::Origin3d { x: 0, y: 0, z },
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::Extent3d {
                        width: job.res,
                        height: job.res,
                        depth_or_array_layers: 1,
                    },
                );
            }
            self.ctx.queue.submit([encoder.finish()]);
            job.next_z = z + 1;
        }

        log::trace!(
            "bake_volume {}³ slices {first}..{}: layers/slice={} record+submit={:?}",
            job.res,
            job.next_z,
            job.sched.order.len(),
            t0.elapsed(),
        );
        if job.is_done() && first < job.next_z {
            log::debug!(
                "bake_volume {}x{}x{} done: layers/slice={} since_begin={:?}",
                job.res,
                job.res,
                job.depth,
                job.sched.order.len(),
                job.started.elapsed(),
            );
        }
        Ok(job.is_done())
    }
}

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
    dom: [f32; 4],
    /// In cells; 0 = unbounded on that axis. `vec3<u32>` in the shader, so
    /// it must start 16-byte aligned (offset 48); `_pad` keeps the size a
    /// multiple of 16.
    period: [u32; 3],
    octaves: u32,
    lacunarity: f32,
    gain: f32,
    fractal_mode: u32,
    normalize: u32,
    kernel: u32,
    /// 0 for a plane; `k + 1` for cube face `k`.
    face: u32,
    _pad: [u32; 2],
    point_map: PointMap,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct PackParams {
    size: [u32; 2],
    mode: u32,
    /// 0 = composite over the gray checker, 1 = keep alpha.
    alpha_object: u32,
    const_value: [f32; 4],
    /// Volume slice index in texels; 0 for flat bakes.
    z_px: u32,
    _pad: [u32; 3],
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
    dom: [f32; 4],
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
    dom: [f32; 4],
    input_doms: [[f32; 4]; 8],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct H2NParams {
    size: [u32; 2],
    strength: f32,
    _pad: u32,
    dom: [f32; 4],
    dom_src: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct WarpParams {
    size: [u32; 2],
    mode: u32,
    _pad0: u32,
    amount: [f32; 4],
    dom: [f32; 4],
    /// Already grown by `|amount|` by the scheduler.
    dom_src: [f32; 4],
    dom_by: [f32; 4],
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
    dom: [f32; 4],
    dom_input: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct RampStopPacked {
    color: [f32; 4],   // Oklcha; used when kind == 0
    t: f32,
    kind: u32,         // 0 = const, 1 = layer
    input_index: u32,  // 0..7
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
fn dispatch_warp(
    ctx: &DeviceCtx,
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    bgl: &wgpu::BindGroupLayout,
    dst_view: &wgpu::TextureView,
    src_view: &wgpu::TextureView,
    by_view: &wgpu::TextureView,
    w: &Warp,
    size: (u32, u32),
    dom: [f32; 4],
    dom_src: [f32; 4],
    dom_by: [f32; 4],
) {
    let params = WarpParams {
        size: [size.0, size.1],
        mode: match w.mode {
            WarpMode::Scalar => 0,
            WarpMode::Vector => 1,
        },
        _pad0: 0,
        // The shader ignores amount.z; see warp.wgsl.
        amount: [w.amount[0], w.amount[1], w.amount[2], 0.0],
        dom,
        dom_src,
        dom_by,
    };
    let ubo = create_uniform(&ctx.device, bytemuck::bytes_of(&params), "warp-params");
    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("warp-bg"),
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
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(by_view),
            },
        ],
    });
    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("warp-cpass"),
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
    eval_ctx: &EvalCtx,
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
    let (input_const, input_is_layer) = match eval_ctx.scalar_const(&w.input) {
        Some(v) => (v, 0u32),
        None => (0.0, 1u32),
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
    eval_ctx: &EvalCtx,
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
    let (factor_const, factor_is_layer) = match eval_ctx.scalar_const(&m.factor) {
        Some(v) => (v, 0u32),
        None => (0.0, 1u32),
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
    eval_ctx: &EvalCtx,
    sched: &Schedule,
    pool_views: &[wgpu::TextureView],
    dummy_input_view: &wgpu::TextureView,
    size: (u32, u32),
    dom: [f32; 4],
) -> Result<(), BakeError> {
    use wgpu::util::DeviceExt;

    // Stops that read the same layer share one input binding.
    let mut layer_to_input: HashMap<LayerId, u32> = HashMap::new();
    let mut input_pool_slots: Vec<u32> = Vec::new();
    let mut input_doms = [Domain::UNIT.packed(); MAX_RAMP_INPUTS];
    for s in &r.stops {
        if let ColorInput::Layer(id) = &s.color {
            let id = *id;
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
        match eval_ctx.color_const(&s.color) {
            Some(c) => packed.push(RampStopPacked {
                color: [c.l, c.chroma, c.hue.into_degrees(), c.alpha],
                t: s.t,
                kind: 0,
                input_index: 0,
                _p0: 0.0,
            }),
            None => {
                let ColorInput::Layer(id) = &s.color else {
                    unreachable!("color_const covers everything but Layer")
                };
                let input_index = *layer_to_input
                    .get(id)
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

    // Unused input bindings get the 1×1 dummy, so they never alias the
    // storage output.
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

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct CoordinateParams {
    size: [u32; 2],
    axis: u32,
    face: u32,
    dom: [f32; 4],
    w_coord: f32,
    _pad: [u32; 3],
    point_map: PointMap,
}

#[allow(clippy::too_many_arguments)]
fn dispatch_coordinate(
    ctx: &DeviceCtx,
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    bgl: &wgpu::BindGroupLayout,
    dst_view: &wgpu::TextureView,
    c: &Coordinate,
    size: (u32, u32),
    w: f32,
    face: u32,
    point_map: PointMap,
    dom: [f32; 4],
) {
    let params = CoordinateParams {
        size: [size.0, size.1],
        axis: match c.axis {
            Axis::U => 0,
            Axis::V => 1,
            Axis::W => 2,
        },
        face,
        dom,
        w_coord: w,
        _pad: [0; 3],
        point_map,
    };
    let ubo = create_uniform(&ctx.device, bytemuck::bytes_of(&params), "coordinate-params");
    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("coordinate-bg"),
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
        label: Some("coordinate-cpass"),
        timestamp_writes: None,
    });
    cpass.set_pipeline(pipeline);
    cpass.set_bind_group(0, &bg, &[]);
    cpass.dispatch_workgroups(size.0.div_ceil(8), size.1.div_ceil(8), 1);
}

fn make_coordinate_pipeline(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
) -> wgpu::ComputePipeline {
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("coordinate-pl"),
        bind_group_layouts: &[Some(bgl)],
        ..Default::default()
    });
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("coordinate-shader"),
        source: wgpu::ShaderSource::Wgsl(
            concat!(include_str!("shaders/sphere.wgsl"), include_str!("shaders/coordinate.wgsl"))
                .into(),
        ),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("coordinate-pipeline"),
        layout: Some(&pl),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

#[allow(clippy::too_many_arguments)]
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
    face: u32,
    point_map: PointMap,
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
        // `Graph` rejects a period on the simplex kernel.
        period: n.period,
        octaves: n.fractal.octaves.clamp(1, texture_graph_core::noise::MAX_OCTAVES),
        lacunarity: n.fractal.lacunarity,
        gain: n.fractal.gain,
        fractal_mode,
        normalize: n.fractal.normalize as u32,
        kernel,
        face,
        _pad: [0; 2],
        point_map,
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

/// The shader a layer kind dispatches, for logs.
fn pipeline_name(kind: &LayerKind) -> &'static str {
    match kind {
        LayerKind::Color(_) => "color.wgsl",
        LayerKind::Noise(_) => "noise.wgsl",
        LayerKind::Coordinate(_) => "coordinate.wgsl",
        LayerKind::Transform(_) => "transform.wgsl",
        LayerKind::Mix(_) => "mix.wgsl",
        LayerKind::Map(_) => "map.wgsl",
        LayerKind::MinMax(_) => "min_max.wgsl",
        LayerKind::HeightToNormal(_) => "height_to_normal.wgsl",
        LayerKind::Warp(_) => "warp.wgsl",
        LayerKind::Wave(_) => "wave.wgsl",
        LayerKind::ColorRamp(_) => "color_ramp.wgsl",
    }
}

/// One line per layer a bake will dispatch, logged once per bake rather
/// than per slice.
fn log_dispatches(graph: &Graph, sched: &Schedule, eval_ctx: &EvalCtx) {
    if !log::log_enabled!(log::Level::Debug) {
        return;
    }
    for &id in &sched.order {
        let Some(layer) = graph.get(id) else { continue };
        let params: Vec<String> = layer
            .kind
            .param_refs()
            .iter()
            .map(|(name, _)| format!("{name}={:?}", eval_ctx.params.get(*name)))
            .collect();
        log::debug!(
            "dispatch layer id={id} name={:?} kind={} pipeline={} slot={} inputs={} params=[{}]",
            layer.name,
            layer.kind.category_label(),
            pipeline_name(&layer.kind),
            slot_of(sched, id),
            IdList(&layer.kind.inputs()),
            params.join(", "),
        );
    }
}

/// Wall-clock for the bake timing logs. Absent on wasm, where
/// `std::time::Instant::now()` compiles and then panics.
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

fn make_pool_texture(device: &wgpu::Device, size: (u32, u32)) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("tg-pool"),
        size: wgpu::Extent3d {
            width: size.0,
            height: size.1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: POOL_FORMAT,
        usage: wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING
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

/// Encodes constant scalar output channels to match the shader path.
fn srgb_of_linear_component(x: f32) -> f32 {
    let clamped = x.clamp(0.0, 1.0);
    if clamped <= 0.0031308 {
        12.92 * clamped
    } else {
        1.055 * clamped.powf(1.0 / 2.4) - 0.055
    }
}


#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ScalarPackParams {
    size: [u32; 2],
    _pad: [u32; 2],
    src_dom: [f32; 4],
}

/// A render pass, not a dispatch; see `pack_scalar.wgsl`.
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
                // The triangle covers every pixel; the clear is never seen.
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

fn copy_face(
    encoder: &mut wgpu::CommandEncoder,
    face_texture: &wgpu::Texture,
    cube: &wgpu::Texture,
    face: u32,
    k: u32,
) {
    encoder.copy_texture_to_texture(
        wgpu::TexelCopyTextureInfo {
            texture: face_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyTextureInfo {
            texture: cube,
            mip_level: 0,
            origin: wgpu::Origin3d { x: 0, y: 0, z: k },
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::Extent3d { width: face, height: face, depth_or_array_layers: 1 },
    );
}

fn make_scalar_cube_texture(
    device: &wgpu::Device,
    face: u32,
    format: ScalarFormat,
    label: &str,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: face,
            height: face,
            depth_or_array_layers: texture_graph_core::CUBE_FACES,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: format.texture_format(),
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST
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
        source: wgpu::ShaderSource::Wgsl(
            concat!(include_str!("shaders/sphere.wgsl"), include_str!("shaders/noise.wgsl")).into(),
        ),
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

fn make_warp_pipeline(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
) -> wgpu::ComputePipeline {
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("warp-pl"),
        bind_group_layouts: &[Some(bgl)],
        ..Default::default()
    });
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("warp-shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/warp.wgsl").into()),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("warp-pipeline"),
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

// Silences an unused-import warning.
#[allow(dead_code)]
fn _sink(_: HashMap<LayerId, ()>) {}
