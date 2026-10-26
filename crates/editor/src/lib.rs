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

/// Absent when the host has no GPU to share; previews then use the CPU path.
pub struct GpuBits {
    pub baker: Baker,
    pub scene: SceneRenderer,
    pub textures: Box<dyn TextureRegistry>,
}

impl GpuBits {
    /// `baker` must be on the device `textures` draws with.
    pub fn new(baker: Baker, textures: Box<dyn TextureRegistry>) -> Self {
        let scene = SceneRenderer::new(&baker.ctx().device);
        Self { baker, scene, textures }
    }
}

pub struct Editor {
    pub graph: Graph,
    pub ui: UiState,
    pub previews: PreviewCache,
    pub preview_panel: PreviewPanelState,
    pub eval_ctx: EvalCtx,
    pub gpu: Option<GpuBits>,
}

impl Editor {
    pub fn new(gpu: Option<GpuBits>, preview_size: u32) -> Self {
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
    pub fn begin_frame(&mut self) {
        let dirty = self.ui.dirty_previews.replace(HashSet::new());
        self.previews.begin_frame(
            &self.graph,
            self.ui.revision,
            dirty.as_ref(),
            self.gpu.as_mut(),
        );
    }

    /// After every panel has drawn: applies the edits they queued. Returns
    /// whether the graph changed.
    pub fn end_frame(&mut self) -> bool {
        self.ui.drain_into(&mut self.graph)
    }

    /// Parameters on the left, the preview on the right, the graph between.
    pub fn show(&mut self, ui: &mut egui::Ui) -> bool {
        self.begin_frame();

        egui::Panel::right("texture_graph_preview")
            .resizable(true)
            .default_size(360.0)
            .show(ui, |ui| self.preview(ui));

        egui::Panel::left("texture_graph_params")
            .resizable(true)
            .default_size(220.0)
            .show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| self.params(ui));
            });

        egui::CentralPanel::default().show(ui, |ui| self.canvas(ui));

        self.end_frame()
    }

    pub fn preview(&mut self, ui: &mut egui::Ui) {
        panels::preview::show(
            ui,
            &self.graph,
            &mut self.ui,
            &mut self.preview_panel,
            &self.eval_ctx,
            self.gpu.as_mut(),
        );
    }

    pub fn params(&mut self, ui: &mut egui::Ui) {
        panels::params::show(ui, &self.graph, &mut self.ui, &mut self.eval_ctx);
    }

    pub fn canvas(&mut self, ui: &mut egui::Ui) {
        if let Some(err) = &self.ui.last_error {
            ui.colored_label(egui::Color32::LIGHT_RED, err);
        }
        panels::graph_canvas::show(
            ui,
            &self.graph,
            &mut self.ui,
            &mut self.previews,
            &self.eval_ctx,
            self.gpu.as_mut(),
        );
    }
}
