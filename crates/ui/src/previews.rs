//! Per-layer preview cache. Each entry is a 128×128 egui texture built by
//! sampling the layer's output on a uv grid.
//!
//! Cache lifetime: entries stay valid until [`PreviewCache::invalidate`] is
//! called (which the app does on any applied mutation). Rebuild is lazy —
//! next request repopulates.

use std::collections::HashMap;

use egui::{Color32, ColorImage, TextureHandle, TextureOptions};
use texture_graph_core::color::to_srgb8;
use texture_graph_core::{EvalCtx, Graph, LayerId, Sample, evaluate};

pub const PREVIEW_SIZE: u32 = 128;

#[derive(Default)]
pub struct PreviewCache {
    entries: HashMap<LayerId, TextureHandle>,
}

impl PreviewCache {
    /// Drop every cached preview. Cheap.
    pub fn invalidate(&mut self) {
        self.entries.clear();
    }

    /// Return (and cache) a 128×128 preview of `id`. Returns `None` if the
    /// layer doesn't exist.
    pub fn get_or_build(
        &mut self,
        ctx: &egui::Context,
        graph: &Graph,
        id: LayerId,
        eval_ctx: &EvalCtx,
    ) -> Option<TextureHandle> {
        if let Some(h) = self.entries.get(&id) {
            return Some(h.clone());
        }
        graph.get(id)?;
        let handle = bake(ctx, graph, id, eval_ctx);
        self.entries.insert(id, handle.clone());
        Some(handle)
    }
}

fn bake(ctx: &egui::Context, graph: &Graph, id: LayerId, eval_ctx: &EvalCtx) -> TextureHandle {
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
