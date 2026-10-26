//! The texture-graph editor's panels, as plain egui.
//!
//! No window and no renderer: a host supplies both. [`Editor::show`] lays the
//! panels out the way the standalone app does; a host with its own docking
//! calls [`Editor::begin_frame`], the panels in [`panels`], and
//! [`Editor::end_frame`] instead. Opening and saving files is the host's too:
//! it reads [`UiState::wants_open`] and [`UiState::wants_save`].

pub mod catalog;
pub mod color_convert;
pub mod panels;
pub mod previews;
pub mod state;
pub mod throttle;
pub mod widgets;

use std::collections::HashSet;

use texture_graph_core::{EvalCtx, Graph};
use texture_graph_gpu::{Baker, SceneRenderer};

use panels::preview::PreviewPanelState;
use previews::PreviewCache;
pub use state::{EditCmd, UiState};

/// Turns a texture the baker made into something egui can draw. That is the
/// host renderer's job, so the editor never names one.
pub trait TextureRegistry {
    /// Sampled with linear filtering. The editor keeps `texture` alive until
    /// it frees the id.
    fn register(&mut self, texture: &wgpu::Texture, view: &wgpu::TextureView) -> egui::TextureId;
    fn free(&mut self, id: egui::TextureId);
}

/// The editor's own GPU state, on the device the host draws with. Absent when
/// the host has no GPU to share; previews then use the CPU path.
pub struct Gpu {
    pub baker: Baker,
    pub scene: SceneRenderer,
}

impl Gpu {
    /// `baker` must be on the device the host's [`TextureRegistry`] draws with.
    pub fn new(baker: Baker) -> Self {
        let scene = SceneRenderer::new(&baker.ctx().device);
        Self { baker, scene }
    }
}

/// One frame's GPU access: the editor's [`Gpu`] and the registry the host
/// lends for the frame. Lent rather than owned, because a host's registry
/// usually borrows host state (an asset store, a renderer) that it cannot
/// hand over for the editor's whole life.
pub struct GpuBits<'a> {
    pub baker: &'a mut Baker,
    pub scene: &'a mut SceneRenderer,
    pub textures: &'a mut dyn TextureRegistry,
}

/// Reborrows a lent registry for one call. Spelled out because `as_deref_mut`
/// cannot shorten a trait object's lifetime behind `&mut`.
fn reborrow<'s>(
    textures: &'s mut Option<&mut dyn TextureRegistry>,
) -> Option<&'s mut dyn TextureRegistry> {
    match textures {
        Some(textures) => Some(&mut **textures),
        None => None,
    }
}

fn lend<'a>(
    gpu: &'a mut Option<Gpu>,
    textures: Option<&'a mut dyn TextureRegistry>,
) -> Option<GpuBits<'a>> {
    let Gpu { baker, scene } = gpu.as_mut()?;
    Some(GpuBits { baker, scene, textures: textures? })
}

pub struct Editor {
    pub graph: Graph,
    pub ui: UiState,
    pub previews: PreviewCache,
    pub preview_panel: PreviewPanelState,
    pub eval_ctx: EvalCtx,
    pub gpu: Option<Gpu>,
}

impl Editor {
    pub fn new(gpu: Option<Gpu>, preview_size: u32) -> Self {
        Self {
            graph: Graph::new(),
            ui: UiState::default(),
            previews: PreviewCache::default(),
            preview_panel: PreviewPanelState::new(preview_size),
            eval_ctx: EvalCtx::default(),
            gpu,
        }
    }

    /// Before any panel draws: retires the thumbnails last frame's edits
    /// made stale.
    ///
    /// Here and below, `textures` is the host's registry for this frame;
    /// without one, or without a [`Gpu`], previews take the CPU path.
    pub fn begin_frame(&mut self, mut textures: Option<&mut dyn TextureRegistry>) {
        let dirty = self.ui.dirty_previews.replace(HashSet::new());
        self.previews.begin_frame(
            &self.graph,
            self.ui.revision,
            dirty.as_ref(),
            lend(&mut self.gpu, reborrow(&mut textures)).as_mut(),
        );
    }

    /// After every panel has drawn: applies the edits they queued. Returns
    /// whether the graph changed.
    pub fn end_frame(&mut self) -> bool {
        self.ui.drain_into(&mut self.graph)
    }

    /// Parameters on the left, the preview on the right, the graph between.
    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        mut textures: Option<&mut dyn TextureRegistry>,
    ) -> bool {
        self.begin_frame(reborrow(&mut textures));

        egui::Panel::right("texture_graph_preview")
            .resizable(true)
            .default_size(360.0)
            .show(ui, |ui| self.preview(ui, reborrow(&mut textures)));

        egui::Panel::left("texture_graph_params")
            .resizable(true)
            .default_size(220.0)
            .show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| self.params(ui));
            });

        egui::CentralPanel::default().show(ui, |ui| self.canvas(ui, textures));

        self.end_frame()
    }

    pub fn preview(&mut self, ui: &mut egui::Ui, mut textures: Option<&mut dyn TextureRegistry>) {
        panels::preview::show(
            ui,
            &self.graph,
            &mut self.ui,
            &mut self.preview_panel,
            &self.eval_ctx,
            lend(&mut self.gpu, reborrow(&mut textures)).as_mut(),
        );
    }

    pub fn params(&mut self, ui: &mut egui::Ui) {
        panels::params::show(ui, &self.graph, &mut self.ui, &mut self.eval_ctx);
    }

    pub fn canvas(&mut self, ui: &mut egui::Ui, mut textures: Option<&mut dyn TextureRegistry>) {
        if let Some(err) = &self.ui.last_error {
            ui.colored_label(egui::Color32::LIGHT_RED, err);
        }
        panels::graph_canvas::show(
            ui,
            &self.graph,
            &mut self.ui,
            &mut self.previews,
            &self.eval_ctx,
            lend(&mut self.gpu, reborrow(&mut textures)).as_mut(),
        );
    }
}
