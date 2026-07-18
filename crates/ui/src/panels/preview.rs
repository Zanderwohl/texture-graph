//! Right panel: flat 2D bake of the graph's `Output` (color channel).
//!
//! We keep it dumb for MVP — just the color channel at a chosen
//! resolution. Roughness / metallic / normal previews land later.

use egui::{Color32, ColorImage, TextureHandle, TextureOptions};
use texture_graph_core::color::to_srgb8;
use texture_graph_core::{EvalCtx, Graph, Sample, evaluate_material};

use crate::state::UiState;

pub struct PreviewPanelState {
    /// Baked color-channel texture. `None` until the first bake or after
    /// invalidation.
    pub texture: Option<TextureHandle>,
    /// Output pixel resolution — 256 on wasm, 512 on native by default.
    /// (Set by `TextureGraphApp::new`.)
    pub size: u32,
    pub channel: PreviewChannel,
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
        Self { texture: None, size, channel: PreviewChannel::Color }
    }
}

pub fn show(
    ui: &mut egui::Ui,
    graph: &Graph,
    state: &mut UiState,
    preview: &mut PreviewPanelState,
    eval_ctx: &EvalCtx,
) {
    ui.horizontal(|ui| {
        ui.heading("Preview");
        ui.add_space(8.0);
        egui::ComboBox::from_id_salt("preview_channel")
            .selected_text(match preview.channel {
                PreviewChannel::Color => "color",
                PreviewChannel::Roughness => "roughness",
                PreviewChannel::Metallic => "metallic",
                PreviewChannel::Normal => "normal",
            })
            .show_ui(ui, |ui| {
                for ch in [
                    PreviewChannel::Color,
                    PreviewChannel::Roughness,
                    PreviewChannel::Metallic,
                    PreviewChannel::Normal,
                ] {
                    let label = match ch {
                        PreviewChannel::Color => "color",
                        PreviewChannel::Roughness => "roughness",
                        PreviewChannel::Metallic => "metallic",
                        PreviewChannel::Normal => "normal",
                    };
                    if ui.selectable_label(preview.channel == ch, label).clicked()
                        && preview.channel != ch
                    {
                        preview.channel = ch;
                        preview.texture = None;
                    }
                }
            });
    });
    ui.horizontal(|ui| {
        for &s in &[128u32, 256, 512, 1024] {
            let picked = preview.size == s;
            if ui.selectable_label(picked, format!("{s}")).clicked() && !picked {
                preview.size = s;
                preview.texture = None;
            }
        }
    });
    ui.separator();

    // Rebake when needed. `state.dirty` covers graph mutations; a `None`
    // texture covers channel/size changes and first frame.
    if state.dirty || preview.texture.is_none() {
        preview.texture = Some(bake(ui.ctx(), graph, preview.size, preview.channel, eval_ctx));
        state.dirty = false;
    }

    if let Some(tex) = &preview.texture {
        let avail = ui.available_size();
        let side = avail.x.min(avail.y).max(64.0);
        ui.add(
            egui::Image::new((tex.id(), egui::vec2(side, side)))
                .maintain_aspect_ratio(true)
                .fit_to_exact_size(egui::vec2(side, side)),
        );
    }
}

fn bake(
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
