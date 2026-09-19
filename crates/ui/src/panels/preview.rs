//! Right panel: preview of the graph's `Output`.
//!
//! **Flat** shows the four PBR channels as textures; **Quad**, **Sphere**
//! and **Cube** light them through the `SceneRenderer`.
//!
//! The Baker always produces the four textures. Flat mode registers them with
//! egui-wgpu directly; 3D modes feed the scene renderer, which draws into a
//! persistent colour target rebuilt only on a size or shape change, so
//! auto-spin doesn't churn allocations at 60 fps.

use egui::{Color32, ColorImage, TextureHandle, TextureOptions};
use texture_graph_core::color::to_srgb8;
use texture_graph_core::{
    EvalCtx, Graph, LayerId, Output, Sample, ScalarInput, evaluate_material,
};
use texture_graph_gpu::{SceneCamera, SceneMaterial, SceneShape, VolumeOutput};

use crate::app::GpuBits;
use crate::state::UiState;

/// Per-axis cap on the volume bake. Volume memory is cubic: 256³ Rgba8 is
/// 64 MiB a channel and 1024³ would be 4 GiB, so past the cap the size
/// buttons only sharpen the viewport.
const VOLUME_RES_CAP: u32 = 256;

pub struct PreviewPanelState {
    pub texture: Option<TextureHandle>,
    pub gpu_channels: Option<GpuChannels>,
    /// Baked only for a 3D graph shown on a 3D shape. Never registered with
    /// egui; only the scene pass samples it.
    pub volume: Option<VolumeOutput>,
    /// Persistent 3D render target + its egui id; rebuilt on size change.
    pub scene: Option<Scene3d>,
    pub size: u32,
    pub shape: PreviewShape,
    pub channel: PreviewChannel,
    pub gpu_error: Option<String>,
    pub yaw_rate_deg_per_sec: f32,
    /// Auto-spin advances this around world Y; drag-orbit composes trackball
    /// rotations onto it.
    pub orientation: glam::Quat,
    /// Runs until the user drag-orbits; the "Rotate" button restores it.
    pub auto_spin: bool,
    /// `state.dirty` is consumed once and fans out here, so the active mode
    /// rebakes now and the other when it is switched to.
    channels_stale: bool,
    volume_stale: bool,
    /// A change invalidates both bake products, so switching what a node
    /// previews rebakes at once.
    last_preview_target: Option<LayerId>,
}

pub struct GpuChannels {
    pub color_tex: wgpu::Texture,
    pub roughness_tex: wgpu::Texture,
    pub metallic_tex: wgpu::Texture,
    pub normal_tex: wgpu::Texture,
    pub color_id: egui::TextureId,
    pub roughness_id: egui::TextureId,
    pub metallic_id: egui::TextureId,
    pub normal_id: egui::TextureId,
    pub size: u32,
    /// Which alpha presentation this bake used: `false` = gray checker
    /// composited (flat preview), `true` = real alpha kept (3D preview
    /// blends the object). Mode switches rebake on mismatch.
    pub object_alpha: bool,
}

pub struct Scene3d {
    pub color_tex: wgpu::Texture,
    pub color_view: wgpu::TextureView,
    pub depth_tex: wgpu::Texture,
    pub depth_view: wgpu::TextureView,
    pub id: egui::TextureId,
    pub size: u32,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum PreviewChannel {
    Color,
    Roughness,
    Metallic,
    Normal,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum PreviewShape {
    Flat,
    /// 1×1 ground quad — the flat material shown lit in 3D.
    Quad,
    Sphere,
    Cube,
}

impl PreviewShape {
    /// Whether this shape is drawn by the 3D scene renderer (everything
    /// except `Flat`, which blits the baked channel textures directly).
    fn is_3d(self) -> bool {
        !matches!(self, PreviewShape::Flat)
    }
}

impl PreviewPanelState {
    pub fn new(size: u32) -> Self {
        Self {
            texture: None,
            gpu_channels: None,
            volume: None,
            scene: None,
            size,
            shape: PreviewShape::Flat,
            channel: PreviewChannel::Color,
            gpu_error: None,
            yaw_rate_deg_per_sec: 30.0,
            orientation: glam::Quat::IDENTITY,
            auto_spin: true,
            channels_stale: false,
            volume_stale: false,
            last_preview_target: None,
        }
    }
}

/// Build the graph the preview renders for a single-node "Preview": the
/// node's color as albedo, with default roughness/metallic/normal (a plain
/// lit look, not the graph's real PBR channels). The Output-node preview
/// uses the real graph unchanged.
fn preview_graph(graph: &Graph, id: LayerId) -> Graph {
    let mut g = graph.clone();
    g.output = Output {
        color: Some(id),
        roughness: ScalarInput::Const(0.5),
        metallic: ScalarInput::Const(0.0),
        normal: None,
    };
    g
}

pub fn show(
    ui: &mut egui::Ui,
    graph: &Graph,
    state: &mut UiState,
    preview: &mut PreviewPanelState,
    eval_ctx: &EvalCtx,
    mut gpu: Option<&mut GpuBits>,
) {
    // Resolve the preview target. A layer target shows just that node's
    // color; the Output node (`None`) shows the full material. A deleted
    // target (its layer removed) falls back to the full material.
    let preview_target = state.preview_target.filter(|id| graph.contains(*id));
    state.preview_target = preview_target;
    if preview.last_preview_target != preview_target {
        preview.last_preview_target = preview_target;
        preview.channels_stale = true;
        preview.volume_stale = true;
    }
    let effective_graph;
    let graph: &Graph = match preview_target {
        Some(id) => {
            effective_graph = preview_graph(graph, id);
            &effective_graph
        }
        None => graph,
    };

    ui.horizontal(|ui| {
        ui.heading("Preview");
        ui.add_space(8.0);
        // Shape selector (Flat / Sphere / Cube).
        let prev_shape = preview.shape;
        egui::ComboBox::from_id_salt("preview_shape")
            .selected_text(shape_label(preview.shape))
            .show_ui(ui, |ui| {
                for sh in [
                    PreviewShape::Flat,
                    PreviewShape::Quad,
                    PreviewShape::Sphere,
                    PreviewShape::Cube,
                ] {
                    if ui
                        .selectable_label(preview.shape == sh, shape_label(sh))
                        .clicked()
                    {
                        preview.shape = sh;
                    }
                }
            });
        if preview.shape != prev_shape {
            // Coming back to Flat, we can drop the 3D target; going into
            // 3D we lazily build it next frame.
            if preview.shape == PreviewShape::Flat {
                if let (Some(g), Some(scene)) = (gpu.as_ref(), preview.scene.take()) {
                    g.renderer.write().free_texture(&scene.id);
                }
            }
        }

        if preview.shape == PreviewShape::Flat {
            ui.add_space(8.0);
            egui::ComboBox::from_id_salt("preview_channel")
                .selected_text(channel_label(preview.channel))
                .show_ui(ui, |ui| {
                    for ch in [
                        PreviewChannel::Color,
                        PreviewChannel::Roughness,
                        PreviewChannel::Metallic,
                        PreviewChannel::Normal,
                    ] {
                        if ui
                            .selectable_label(preview.channel == ch, channel_label(ch))
                            .clicked()
                        {
                            preview.channel = ch;
                        }
                    }
                });
        }
    });
    ui.horizontal(|ui| {
        for &s in &[128u32, 256, 512, 1024] {
            let picked = preview.size == s;
            if ui.selectable_label(picked, format!("{s}")).clicked() && !picked {
                preview.size = s;
                preview.texture = None;
                if let Some(g) = gpu.as_ref() {
                    if let Some(old) = &preview.gpu_channels {
                        free_channels(&g.renderer, old);
                    }
                    if let Some(old) = preview.scene.take() {
                        g.renderer.write().free_texture(&old.id);
                    }
                }
                preview.gpu_channels = None;
            }
        }
    });
    ui.separator();

    // Consume the dirty flag once; both bake products go stale and each
    // rebakes when its mode needs it (the other lazily, on mode switch).
    if state.dirty {
        preview.channels_stale = true;
        preview.volume_stale = true;
        state.dirty = false;
    }

    // Solid 3D sampling: when a 3D shape displays a graph that actually
    // varies along w (`Graph::output_is_3d`), bake the graph as a volume
    // and sample it by object-space position instead of UV-wrapping a
    // flat slice. Falls back to the UV path if the volume bake fails.
    let want_solid = preview.shape.is_3d() && graph.output_is_3d();
    let mut solid_active = false;
    if want_solid {
        if let Some(gpu) = gpu.as_deref_mut() {
            let vol_res = preview.size.min(VOLUME_RES_CAP);
            // A size-button change invalidates the volume like it does the
            // flat channels: rebake when the resolution no longer matches.
            if preview.volume.as_ref().is_some_and(|v| v.size.0 != vol_res) {
                preview.volume = None;
            }
            if preview.volume_stale || preview.volume.is_none() {
                preview.gpu_error = None;
                match gpu.baker.bake_volume(graph, vol_res, vol_res, eval_ctx) {
                    Ok(v) => {
                        preview.volume = Some(v);
                        preview.volume_stale = false;
                    }
                    Err(e) => {
                        preview.gpu_error =
                            Some(format!("volume bake failed, using UV mapping: {e}"));
                        preview.volume = None;
                    }
                }
            }
            solid_active = preview.volume.is_some();
        }
    }

    // 1. Ensure the flat material textures are baked (Flat mode displays
    //    them; the UV-mapped 3D path samples them). Skipped while solid
    //    sampling covers the 3D view — it has its own product above.
    //    3D shapes want real alpha (the object blends); Flat wants the
    //    gray backing checker composited in.
    let want_object_alpha = preview.shape.is_3d();
    let needs_bake = !solid_active
        && (preview.channels_stale
            || (preview.gpu_channels.is_none() && preview.texture.is_none())
            || preview
                .gpu_channels
                .as_ref()
                .is_some_and(|c| c.size != preview.size || c.object_alpha != want_object_alpha));
    if needs_bake {
        preview.gpu_error = None;
        let baked_on_gpu = if let Some(gpu) = gpu.as_deref_mut() {
            match gpu.baker.bake_output(
                graph,
                (preview.size, preview.size),
                eval_ctx,
                want_object_alpha,
            ) {
                Ok(out) => {
                    if let Some(old) = preview.gpu_channels.take() {
                        free_channels(&gpu.renderer, &old);
                    }
                    preview.gpu_channels = Some(register_channels(
                        &gpu.renderer,
                        &gpu.baker,
                        out,
                        want_object_alpha,
                    ));
                    preview.texture = None;
                    true
                }
                Err(e) => {
                    preview.gpu_error = Some(format!("gpu bake fell back to CPU: {e}"));
                    false
                }
            }
        } else {
            false
        };
        if !baked_on_gpu {
            preview.texture =
                Some(bake_cpu(ui.ctx(), graph, preview.size, preview.channel, eval_ctx));
        }
        preview.channels_stale = false;
    }

    // 2. In 3D mode, keep a persistent scene target + rerender each frame.
    if preview.shape.is_3d()
        && gpu.is_some()
        && (solid_active || preview.gpu_channels.is_some())
    {
        // Rebuild the target first (borrows `preview` + `gpu` mutably).
        ensure_scene_target(preview, gpu.as_deref_mut().unwrap());
        // Then render into it. Split borrows: pull the pieces we need out
        // as separate references so the borrow checker sees no overlap.
        let gpu = gpu.as_deref_mut().unwrap();
        let scene = preview.scene.as_ref().unwrap();
        if preview.auto_spin {
            // Incremental so pausing/orbiting resumes from the current
            // orientation instead of snapping back to a time-derived angle.
            let dt = ui.ctx().input(|i| i.stable_dt).min(0.1);
            preview.orientation = glam::Quat::from_rotation_y(
                dt * preview.yaw_rate_deg_per_sec.to_radians(),
            ) * preview.orientation;
        }
        let mut camera = SceneCamera {
            orientation: preview.orientation,
            ..SceneCamera::default()
        };
        let shape = match preview.shape {
            PreviewShape::Sphere => SceneShape::Sphere,
            PreviewShape::Cube => SceneShape::Cube,
            PreviewShape::Quad => {
                // Look down at the ground plane from a raised 3/4 angle so
                // the whole face is visible; auto-spin turntables it in place.
                camera.pitch = 0.95;
                camera.distance = 2.6;
                SceneShape::Quad
            }
            PreviewShape::Flat => unreachable!(),
        };
        // Keep the UV material's texture clones alive past the match.
        let uv_material;
        let material = if solid_active {
            SceneMaterial::Solid(preview.volume.as_ref().unwrap())
        } else {
            let channels = preview.gpu_channels.as_ref().unwrap();
            uv_material = texture_graph_gpu::BakeOutput {
                color: channels.color_tex.clone(),
                roughness: channels.roughness_tex.clone(),
                metallic: channels.metallic_tex.clone(),
                normal: channels.normal_tex.clone(),
                size: (channels.size, channels.size),
            };
            SceneMaterial::Uv(&uv_material)
        };
        gpu.scene.render_into(
            gpu.baker.ctx(),
            material,
            shape,
            &scene.color_view,
            &scene.depth_view,
            (scene.size, scene.size),
            &camera,
        );
        if preview.auto_spin {
            ui.ctx().request_repaint();
        }
    }

    // 3. Display.
    let avail = ui.available_size();
    let side = avail.x.min(avail.y).max(64.0);
    match preview.shape {
        PreviewShape::Flat => {
            if let Some(g) = &preview.gpu_channels {
                let id = match preview.channel {
                    PreviewChannel::Color => g.color_id,
                    PreviewChannel::Roughness => g.roughness_id,
                    PreviewChannel::Metallic => g.metallic_id,
                    PreviewChannel::Normal => g.normal_id,
                };
                ui.add(
                    egui::Image::new((id, egui::vec2(side, side)))
                        .maintain_aspect_ratio(true)
                        .fit_to_exact_size(egui::vec2(side, side)),
                );
            } else if let Some(tex) = &preview.texture {
                ui.add(
                    egui::Image::new((tex.id(), egui::vec2(side, side)))
                        .maintain_aspect_ratio(true)
                        .fit_to_exact_size(egui::vec2(side, side)),
                );
            }
        }
        PreviewShape::Quad | PreviewShape::Sphere | PreviewShape::Cube => {
            if let Some(scene) = &preview.scene {
                let resp = ui.add(
                    egui::Image::new((scene.id, egui::vec2(side, side)))
                        .maintain_aspect_ratio(true)
                        .fit_to_exact_size(egui::vec2(side, side))
                        .sense(egui::Sense::drag()),
                );
                // Drag-orbit. egui only starts a drag when the press began
                // inside the widget, so a button already held down when the
                // pointer enters never grabs the model.
                if resp.drag_started_by(egui::PointerButton::Primary) {
                    preview.auto_spin = false;
                }
                if resp.dragged_by(egui::PointerButton::Primary) {
                    let d = resp.drag_delta();
                    if d != egui::Vec2::ZERO {
                        // Trackball on a large sphere: the grabbed surface
                        // point follows the pointer. Screen-space delta
                        // (right, down) maps to a world rotation axis
                        // (x=down-drag, y=right-drag); magnitude scales by
                        // the sphere radius (half the viewport).
                        let radius = (side * 0.5).max(1.0);
                        let axis = glam::Vec3::new(d.y, d.x, 0.0);
                        let angle = axis.length() / radius;
                        preview.orientation =
                            glam::Quat::from_axis_angle(axis.normalize(), angle)
                                * preview.orientation;
                    }
                }
                if ui.button("Rotate").clicked() {
                    preview.auto_spin = true;
                }
            } else {
                ui.weak("(3D preview needs the wgpu backend)");
            }
        }
    }
    if let Some(err) = &preview.gpu_error {
        ui.colored_label(egui::Color32::YELLOW, err);
    }
}

/// Create-or-reuse the persistent scene render target. Rebuilds when the
/// panel size changes; frees the old egui id first so we don't leak.
fn ensure_scene_target(preview: &mut PreviewPanelState, gpu: &mut GpuBits) {
    let needs_new = match &preview.scene {
        Some(s) => s.size != preview.size,
        None => true,
    };
    if !needs_new {
        return;
    }
    if let Some(old) = preview.scene.take() {
        gpu.renderer.write().free_texture(&old.id);
    }
    let device = &gpu.baker.ctx().device;
    let color_tex = gpu.scene.make_color_target(device, (preview.size, preview.size));
    let color_view = color_tex.create_view(&wgpu::TextureViewDescriptor::default());
    let depth_tex = gpu.scene.make_depth_target(device, (preview.size, preview.size));
    let depth_view = depth_tex.create_view(&wgpu::TextureViewDescriptor::default());
    let id = gpu.renderer.write().register_native_texture(
        device,
        &color_view,
        wgpu::FilterMode::Linear,
    );
    preview.scene = Some(Scene3d {
        color_tex,
        color_view,
        depth_tex,
        depth_view,
        id,
        size: preview.size,
    });
}

fn shape_label(sh: PreviewShape) -> &'static str {
    match sh {
        PreviewShape::Flat => "flat",
        PreviewShape::Quad => "quad",
        PreviewShape::Sphere => "sphere",
        PreviewShape::Cube => "cube",
    }
}

fn channel_label(ch: PreviewChannel) -> &'static str {
    match ch {
        PreviewChannel::Color => "color",
        PreviewChannel::Roughness => "roughness",
        PreviewChannel::Metallic => "metallic",
        PreviewChannel::Normal => "normal",
    }
}

fn register_channels(
    renderer: &egui::mutex::RwLock<egui_wgpu::Renderer>,
    baker: &texture_graph_gpu::Baker,
    out: texture_graph_gpu::BakeOutput,
    object_alpha: bool,
) -> GpuChannels {
    let device = &baker.ctx().device;
    let color_view = out.color.create_view(&wgpu::TextureViewDescriptor::default());
    let rough_view = out.roughness.create_view(&wgpu::TextureViewDescriptor::default());
    let metal_view = out.metallic.create_view(&wgpu::TextureViewDescriptor::default());
    let normal_view = out.normal.create_view(&wgpu::TextureViewDescriptor::default());
    let mut r = renderer.write();
    let color_id =
        r.register_native_texture(device, &color_view, wgpu::FilterMode::Linear);
    let roughness_id =
        r.register_native_texture(device, &rough_view, wgpu::FilterMode::Linear);
    let metallic_id =
        r.register_native_texture(device, &metal_view, wgpu::FilterMode::Linear);
    let normal_id =
        r.register_native_texture(device, &normal_view, wgpu::FilterMode::Linear);
    GpuChannels {
        color_tex: out.color,
        roughness_tex: out.roughness,
        metallic_tex: out.metallic,
        normal_tex: out.normal,
        color_id,
        roughness_id,
        metallic_id,
        normal_id,
        size: out.size.0,
        object_alpha,
    }
}

fn free_channels(
    renderer: &egui::mutex::RwLock<egui_wgpu::Renderer>,
    channels: &GpuChannels,
) {
    let mut r = renderer.write();
    r.free_texture(&channels.color_id);
    r.free_texture(&channels.roughness_id);
    r.free_texture(&channels.metallic_id);
    r.free_texture(&channels.normal_id);
}

fn bake_cpu(
    ctx: &egui::Context,
    graph: &Graph,
    size: u32,
    channel: PreviewChannel,
    eval_ctx: &EvalCtx,
) -> TextureHandle {
    let mut pixels = Vec::with_capacity((size * size) as usize);
    for y in 0..size {
        for x in 0..size {
            let u = x as f32 / (size - 1) as f32;
            let v = y as f32 / (size - 1) as f32;
            let m = evaluate_material(graph, Sample::flat(u, v), eval_ctx);
            let rgba = match channel {
                PreviewChannel::Color => to_srgb8(m.color),
                PreviewChannel::Normal => to_srgb8(m.normal),
                PreviewChannel::Roughness => scalar_to_rgba(m.roughness),
                PreviewChannel::Metallic => scalar_to_rgba(m.metallic),
            };
            pixels.push(Color32::from_rgba_unmultiplied(rgba[0], rgba[1], rgba[2], rgba[3]));
        }
    }
    let img = ColorImage {
        size: [size as usize, size as usize],
        source_size: egui::Vec2::new(size as f32, size as f32),
        pixels,
    };
    ctx.load_texture("flat-preview", img, TextureOptions::LINEAR)
}

fn scalar_to_rgba(v: f32) -> [u8; 4] {
    let g = (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    [g, g, g, 255]
}
