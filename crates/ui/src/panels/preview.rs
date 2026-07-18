//! Right panel: flat 2D bake of the graph's `Output`.
//!
//! GPU path (default when eframe's wgpu backend is live): bakes all four
//! PBR channels in one dispatch chain via `texture_graph_gpu::Baker`,
//! then registers each result with egui-wgpu so the panel just picks a
//! `TextureId` when the channel switches. CPU path (fallback) still runs
//! `evaluate_material` per pixel.

use egui::{Color32, ColorImage, TextureHandle, TextureOptions};
use texture_graph_core::color::to_srgb8;
use texture_graph_core::{EvalCtx, Graph, Sample, evaluate_material};

use crate::app::GpuBits;
use crate::state::UiState;

pub struct PreviewPanelState {
    /// CPU-baked flat texture, used when the GPU backend isn't available
    /// or the graph contains a variant the GPU baker doesn't cover.
    pub texture: Option<TextureHandle>,
    /// GPU-baked textures (color, roughness, metallic, normal), each
    /// registered with egui-wgpu as its own `TextureId`.
    pub gpu_channels: Option<GpuChannels>,
    /// Output pixel resolution — 256 on wasm, 512 on native by default.
    pub size: u32,
    pub channel: PreviewChannel,
    /// Last-seen error from a GPU bake; surfaced under the preview image.
    pub gpu_error: Option<String>,
}

/// Registered `TextureId`s for one bake, one per output channel. The
/// backing `wgpu::Texture`s are pinned here so their views (referenced by
/// the renderer) stay valid until we free the ids.
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
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum PreviewChannel {
    Color,
    Roughness,
    Metallic,
    Normal,
}

impl PreviewPanelState {
    pub fn new(size: u32) -> Self {
        Self {
            texture: None,
            gpu_channels: None,
            size,
            channel: PreviewChannel::Color,
            gpu_error: None,
        }
    }
}

pub fn show(
    ui: &mut egui::Ui,
    graph: &Graph,
    state: &mut UiState,
    preview: &mut PreviewPanelState,
    eval_ctx: &EvalCtx,
    gpu: Option<&mut GpuBits>,
) {
    ui.horizontal(|ui| {
        ui.heading("Preview");
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
    });
    ui.horizontal(|ui| {
        for &s in &[128u32, 256, 512, 1024] {
            let picked = preview.size == s;
            if ui.selectable_label(picked, format!("{s}")).clicked() && !picked {
                preview.size = s;
                // Force a rebake at the new size.
                preview.texture = None;
                if let Some(g) = &preview.gpu_channels {
                    if let Some(gpu) = gpu.as_ref() {
                        free_channels(&gpu.renderer, g);
                    }
                }
                preview.gpu_channels = None;
            }
        }
    });
    ui.separator();

    let needs_bake = state.dirty
        || (preview.gpu_channels.is_none() && preview.texture.is_none())
        || preview
            .gpu_channels
            .as_ref()
            .map_or(false, |c| c.size != preview.size);

    if needs_bake {
        preview.gpu_error = None;
        // Try GPU first; fall back to CPU on any error.
        let baked_on_gpu = if let Some(gpu) = gpu {
            match gpu.baker.bake_output(
                graph,
                (preview.size, preview.size),
                eval_ctx,
            ) {
                Ok(out) => {
                    if let Some(old) = preview.gpu_channels.take() {
                        free_channels(&gpu.renderer, &old);
                    }
                    preview.gpu_channels =
                        Some(register_channels(&gpu.renderer, &gpu.baker, out));
                    // Explicit reset: any earlier CPU texture is stale.
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
            if let Some(old) = preview.gpu_channels.take() {
                // No renderer handle here in the CPU-only branch; the
                // renderer already survives across frames, so leaking the
                // id would only leak on this specific transition. Defensive
                // reset to None keeps the state coherent.
                let _ = old;
            }
        }
        state.dirty = false;
    }

    let avail = ui.available_size();
    let side = avail.x.min(avail.y).max(64.0);
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
    if let Some(err) = &preview.gpu_error {
        ui.colored_label(egui::Color32::YELLOW, err);
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

// ---- CPU fallback ------------------------------------------------------

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
            let m = evaluate_material(graph, Sample::uv(u, v), eval_ctx);
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
