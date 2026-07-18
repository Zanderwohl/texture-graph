//! Per-layer preview cache.
//!
//! GPU path (default when eframe's wgpu backend is live): the first request
//! after `invalidate` bulk-bakes previews for every layer at 128² via
//! `Baker::bake_previews`, registers each result with egui-wgpu, and hands
//! back `TextureId`s from a `HashMap`. Subsequent requests are cache hits.
//!
//! CPU path (fallback): per-request lazy bake via `evaluate`.
//!
//! Lifetime: on `invalidate`, GPU `TextureId`s are freed on the shared
//! `egui_wgpu::Renderer` before the map is cleared; the backing
//! `wgpu::Texture`s are then dropped.

use std::collections::HashMap;

use egui::{Color32, ColorImage, TextureHandle, TextureId, TextureOptions};
use texture_graph_core::color::to_srgb8;
use texture_graph_core::{EvalCtx, Graph, LayerId, Sample, evaluate};

use crate::app::GpuBits;

pub const PREVIEW_SIZE: u32 = 128;

/// One registered GPU thumbnail. Holds both the backing texture (to keep
/// its view alive on the renderer side) and the `TextureId` we display.
pub struct GpuThumb {
    #[allow(dead_code)]
    tex: wgpu::Texture,
    id: TextureId,
}

#[derive(Default)]
pub struct PreviewCache {
    gpu_entries: HashMap<LayerId, GpuThumb>,
    cpu_entries: HashMap<LayerId, TextureHandle>,
}

impl PreviewCache {
    /// Drop every cached preview. When the GPU path is active, first frees
    /// each registered `TextureId` on the shared renderer so we don't leak
    /// backing resources.
    pub fn invalidate(&mut self, gpu: Option<&GpuBits>) {
        if !self.gpu_entries.is_empty() {
            if let Some(gpu) = gpu {
                let mut r = gpu.renderer.write();
                for thumb in self.gpu_entries.values() {
                    r.free_texture(&thumb.id);
                }
            }
            self.gpu_entries.clear();
        }
        self.cpu_entries.clear();
    }

    /// Return (and cache) a 128×128 preview of `id`. Returns `None` if the
    /// layer doesn't exist. Uses GPU when available; falls back to CPU on
    /// any bake error.
    pub fn get_or_build(
        &mut self,
        egui_ctx: &egui::Context,
        graph: &Graph,
        id: LayerId,
        eval_ctx: &EvalCtx,
        gpu: Option<&mut GpuBits>,
    ) -> Option<TextureId> {
        graph.get(id)?;
        if let Some(thumb) = self.gpu_entries.get(&id) {
            return Some(thumb.id);
        }
        if let Some(h) = self.cpu_entries.get(&id) {
            return Some(h.id());
        }
        // Fresh entry — try GPU bulk bake first.
        if let Some(gpu) = gpu {
            if let Ok(textures) = gpu.baker.bake_previews(graph, eval_ctx) {
                let device = gpu.baker.ctx().device.clone();
                let mut r = gpu.renderer.write();
                for (lid, tex) in textures {
                    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
                    let tid = r.register_native_texture(
                        &device,
                        &view,
                        wgpu::FilterMode::Linear,
                    );
                    self.gpu_entries.insert(lid, GpuThumb { tex, id: tid });
                }
                return self.gpu_entries.get(&id).map(|t| t.id);
            }
            // GPU bake errored — fall through to CPU for this layer.
        }
        let handle = bake_cpu(egui_ctx, graph, id, eval_ctx);
        let tid = handle.id();
        self.cpu_entries.insert(id, handle);
        Some(tid)
    }
}

fn bake_cpu(
    ctx: &egui::Context,
    graph: &Graph,
    id: LayerId,
    eval_ctx: &EvalCtx,
) -> TextureHandle {
    let w = PREVIEW_SIZE;
    let h = PREVIEW_SIZE;
    let mut pixels = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        for x in 0..w {
            let u = x as f32 / (w - 1) as f32;
            let v = y as f32 / (h - 1) as f32;
            let color = evaluate(graph, id, Sample::uv(u, v), eval_ctx);
            let [r, g, b, a] = to_srgb8(color);
            pixels.push(Color32::from_rgba_unmultiplied(r, g, b, a));
        }
    }
    let img = ColorImage {
        size: [w as usize, h as usize],
        source_size: egui::Vec2::new(w as f32, h as f32),
        pixels,
    };
    ctx.load_texture(
        format!("layer-preview-{}", id.0),
        img,
        TextureOptions::LINEAR,
    )
}
