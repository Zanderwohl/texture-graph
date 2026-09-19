//! Per-layer preview cache.
//!
//! GPU path: `Baker::bake_previews` bakes thumbnails, registered with
//! egui-wgpu and kept per layer. CPU fallback bakes lazily via `evaluate`.
//!
//! # Staleness
//!
//! Textures are replaced in place and never cleared in bulk. An entry
//! records the graph revision it was baked from; an edit records, per layer
//! it can have changed, the revision at which that layer's picture stopped
//! being current. A layer whose entry is behind rebakes on the next request
//! and keeps showing its previous image until the replacement exists.
//!
//! Staleness is per layer because an edit usually reaches a handful of them
//! and a rebake costs two textures, a registration and a freed id each.
//!
//! Whoever supplies the dirty set owes the closure: a layer is safe to leave
//! alone only if nothing it reads changed either. [`crate::state`] computes
//! that in `dirty_roots` and `downstream`.

use std::collections::{HashMap, HashSet};

use egui::{Color32, ColorImage, TextureHandle, TextureId, TextureOptions};
use texture_graph_core::color::to_srgb8;
use texture_graph_core::{EvalCtx, Graph, LayerId, Sample, evaluate};

use crate::app::GpuBits;

pub const PREVIEW_SIZE: u32 = 128;

/// A registered GPU thumbnail. Keeps the texture, so the renderer's view
/// stays alive, alongside the id to display.
struct GpuThumb {
    #[allow(dead_code)]
    tex: wgpu::Texture,
    id: TextureId,
    /// The graph revision this was baked from, and the forced-retirement
    /// counter it was baked under.
    revision: u64,
    forced: u64,
}

struct CpuThumb {
    handle: TextureHandle,
    revision: u64,
    forced: u64,
}

#[derive(Default)]
pub struct PreviewCache {
    gpu_entries: HashMap<LayerId, GpuThumb>,
    cpu_entries: HashMap<LayerId, CpuThumb>,
    /// Per layer, the revision its picture stopped being current at. Missing
    /// means it never did, so any entry will do.
    stale_at: HashMap<LayerId, u64>,
    /// Bumped to retire every image for a reason the revision can't see, like
    /// a replaced graph or a failed device.
    forced: u64,
    /// The revision this frame is drawing at.
    revision: u64,
    /// At most one bulk bake a frame: the baker runs every stale layer in a
    /// single submit, so the first request rebuilds all of them.
    rebuilt_this_frame: bool,
}

impl PreviewCache {
    /// Start a frame, retiring the images `dirty` names — `None` for all.
    /// Images stay on screen until their replacements exist.
    ///
    /// Also drops entries for layers that no longer exist. That happens
    /// here rather than in `rebuild` because deleting a layer nothing reads
    /// leaves every *remaining* thumbnail current, so no rebuild is
    /// triggered and a texture registered with egui-wgpu would sit there
    /// until some later edit happened to need one.
    pub fn begin_frame(
        &mut self,
        graph: &Graph,
        revision: u64,
        dirty: Option<&HashSet<LayerId>>,
        gpu: Option<&GpuBits>,
    ) {
        self.revision = revision;
        self.rebuilt_this_frame = false;
        match dirty {
            None => self.forced += 1,
            Some(ids) => {
                for id in ids {
                    self.stale_at.insert(*id, revision);
                }
            }
        }
        self.forget_dead(graph, gpu);
    }

    /// Whether this layer's image is current: baked since the layer was
    /// last dirtied, and not under a retired forced generation. A layer
    /// with no entry is never fresh.
    fn is_fresh(&self, id: LayerId) -> bool {
        let since = self.stale_at.get(&id).copied().unwrap_or(0);
        let fresh = |rev: u64, forced: u64| forced == self.forced && rev >= since;
        self.gpu_entries
            .get(&id)
            .map(|e| fresh(e.revision, e.forced))
            .or_else(|| self.cpu_entries.get(&id).map(|e| fresh(e.revision, e.forced)))
            .unwrap_or(false)
    }

    /// Return (and cache) a 128×128 preview of `id`. `None` if the layer
    /// doesn't exist.
    pub fn get_or_build(
        &mut self,
        egui_ctx: &egui::Context,
        graph: &Graph,
        id: LayerId,
        eval_ctx: &EvalCtx,
        gpu: Option<&mut GpuBits>,
    ) -> Option<TextureId> {
        graph.get(id)?;

        if !self.is_fresh(id) && !self.rebuilt_this_frame {
            self.rebuilt_this_frame = true;
            self.rebuild(graph, eval_ctx, gpu);
        }

        if let Some(thumb) = self.gpu_entries.get(&id) {
            return Some(thumb.id);
        }
        if let Some(thumb) = self.cpu_entries.get(&id) {
            return Some(thumb.handle.id());
        }
        // Cold, and the bulk bake either failed or there's no GPU at all.
        // A single-layer CPU bake so the node still shows something.
        let handle = bake_cpu(egui_ctx, graph, id, eval_ctx);
        let tid = handle.id();
        self.cpu_entries.insert(
            id,
            CpuThumb { handle, revision: self.revision, forced: self.forced },
        );
        Some(tid)
    }

    /// The layers whose thumbnail is out of date and still exists.
    fn stale_layers(&self, graph: &Graph) -> HashSet<LayerId> {
        graph
            .layers
            .iter()
            .map(|l| l.id)
            .filter(|id| !self.is_fresh(*id))
            .collect()
    }

    /// Bake fresh thumbnails for every stale layer, register them, and swap
    /// those entries. A layer that wasn't stale is not baked, not
    /// registered, and keeps the texture it already had.
    fn rebuild(&mut self, graph: &Graph, eval_ctx: &EvalCtx, gpu: Option<&mut GpuBits>) {
        let wanted = self.stale_layers(graph);
        if wanted.is_empty() {
            return;
        }

        let Some(gpu) = gpu else {
            // No GPU — drop the stale CPU entries and let the per-layer
            // fallback in `get_or_build` rebuild them lazily.
            self.cpu_entries.retain(|id, _| !wanted.contains(id));
            return;
        };
        let Ok(new_texs) = gpu.baker.bake_previews(graph, eval_ctx, Some(&wanted)) else {
            // Keep the cache, so the user sees the last good state rather
            // than a blank grid. Entries stay stale and the next frame tries
            // again: a persistent failure costs an attempt per frame, but the
            // common one is transient and a thumbnail that never returns is
            // worse.
            return;
        };

        let device = gpu.baker.ctx().device.clone();
        let mut r = gpu.renderer.write();
        let mut replaced: Vec<TextureId> = Vec::new();
        for (lid, tex) in new_texs {
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            let tid = r.register_native_texture(&device, &view, wgpu::FilterMode::Linear);
            // Register the replacement before retiring what it replaces, so
            // nothing on screen has a frame with no texture behind it.
            let old = self.gpu_entries.insert(
                lid,
                GpuThumb { tex, id: tid, revision: self.revision, forced: self.forced },
            );
            if let Some(old) = old {
                replaced.push(old.id);
            }
            // A layer that just got a GPU thumbnail has no use for the CPU
            // one it may have been showing.
            self.cpu_entries.remove(&lid);
        }
        for id in replaced {
            r.free_texture(&id);
        }
    }

    /// Drop entries for layers that no longer exist, freeing what they had
    /// registered, so a long editing session doesn't accumulate textures
    /// for deleted nodes.
    fn forget_dead(&mut self, graph: &Graph, gpu: Option<&GpuBits>) {
        let dead: Vec<LayerId> = self
            .gpu_entries
            .keys()
            .chain(self.cpu_entries.keys())
            .copied()
            .filter(|id| !graph.contains(*id))
            .collect();
        if dead.is_empty() {
            self.stale_at.retain(|id, _| graph.contains(*id));
            return;
        }
        let mut freed: Vec<TextureId> = Vec::new();
        for id in dead {
            if let Some(thumb) = self.gpu_entries.remove(&id) {
                freed.push(thumb.id);
            }
            self.cpu_entries.remove(&id);
        }
        self.stale_at.retain(|id, _| graph.contains(*id));
        if let Some(gpu) = gpu {
            let mut r = gpu.renderer.write();
            for id in freed {
                r.free_texture(&id);
            }
        }
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
            let color = evaluate(graph, id, Sample::flat(u, v), eval_ctx);
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
