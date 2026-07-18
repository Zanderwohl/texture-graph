//! Per-layer preview cache.
//!
//! GPU path (default when eframe's wgpu backend is live): the first request
//! after `mark_stale` bulk-bakes previews for every layer at 128² via
//! `Baker::bake_previews`, registers each result with egui-wgpu, and swaps
//! the entries in atomically — old `TextureId`s stay displayable right up
//! until the new ones are ready, then get freed. This means an evaluation
//! change (e.g. tweaking noise frequency) doesn't visually blank the
//! thumbnails for a frame.
//!
//! CPU path (fallback): per-request lazy bake via `evaluate`.

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
    /// Set by the app after any evaluation-affecting edit. The next
    /// `get_or_build` will rebake and swap; entries stay displayable in
    /// the meantime.
    stale: bool,
}

impl PreviewCache {
    /// Mark the cache stale without dropping current entries. The next
    /// `get_or_build` rebuilds and only then swaps the new textures in.
    pub fn mark_stale(&mut self) {
        self.stale = true;
    }

    /// Return (and cache) a 128×128 preview of `id`. Returns `None` if the
    /// layer doesn't exist.
    ///
    /// When the cache is stale, the first call this frame rebuilds all
    /// entries via `Baker::bake_previews`, atomically swaps them in, then
    /// frees the old `TextureId`s. Subsequent same-frame calls are hits.
    pub fn get_or_build(
        &mut self,
        egui_ctx: &egui::Context,
        graph: &Graph,
        id: LayerId,
        eval_ctx: &EvalCtx,
        gpu: Option<&mut GpuBits>,
    ) -> Option<TextureId> {
        graph.get(id)?;

        if self.stale {
            self.rebuild(graph, eval_ctx, gpu);
            // Fall through to the normal lookup below regardless of
            // rebuild success — a partial success still hits at least
            // some entries; a total failure falls back to CPU per-layer.
        }

        if let Some(thumb) = self.gpu_entries.get(&id) {
            return Some(thumb.id);
        }
        if let Some(h) = self.cpu_entries.get(&id) {
            return Some(h.id());
        }
        // First-frame path (cold cache, no stale flag): if we're still
        // holding no entries at all, either the bulk bake failed above or
        // GPU isn't wired at all. Fall back to a single-layer CPU bake so
        // the row still shows something.
        let handle = bake_cpu(egui_ctx, graph, id, eval_ctx);
        let tid = handle.id();
        self.cpu_entries.insert(id, handle);
        Some(tid)
    }

    /// Bake a fresh set of GPU thumbnails, register them, and swap the
    /// cache atomically. Old entries stay displayable until we've fully
    /// registered the new ones, then get freed.
    fn rebuild(
        &mut self,
        graph: &Graph,
        eval_ctx: &EvalCtx,
        gpu: Option<&mut GpuBits>,
    ) {
        let Some(gpu) = gpu else {
            // No GPU — CPU entries are cheap enough to just drop; the
            // per-layer CPU fallback rebuilds them lazily.
            self.cpu_entries.clear();
            self.stale = false;
            return;
        };
        let bake_result = gpu.baker.bake_previews(graph, eval_ctx);
        let Ok(new_texs) = bake_result else {
            // Leave the current cache in place — the user still sees the
            // previous state instead of a blank grid — and reset the flag
            // so we don't re-attempt every frame.
            self.stale = false;
            return;
        };
        let device = gpu.baker.ctx().device.clone();
        let mut r = gpu.renderer.write();
        let mut new_map: HashMap<LayerId, GpuThumb> = HashMap::new();
        for (lid, tex) in new_texs {
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            let tid = r.register_native_texture(&device, &view, wgpu::FilterMode::Linear);
            new_map.insert(lid, GpuThumb { tex, id: tid });
        }
        // Only now free the outgoing ids — after the replacements are
        // registered so nothing displayed had a "no texture" gap.
        for old in self.gpu_entries.values() {
            r.free_texture(&old.id);
        }
        drop(r);
        self.gpu_entries = new_map;
        self.cpu_entries.clear();
        self.stale = false;
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
