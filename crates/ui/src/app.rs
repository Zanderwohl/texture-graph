use std::collections::HashSet;
use std::sync::Arc;

use eframe::CreationContext;
use egui::mutex::RwLock;
use texture_graph_core::{EvalCtx, Graph};
use texture_graph_gpu::{Baker, DeviceCtx, SceneRenderer};

use crate::file_io;
use crate::panels;
use crate::panels::preview::PreviewPanelState;
use crate::previews::PreviewCache;
use crate::state::UiState;

/// Baker plus the egui-wgpu renderer baked textures are registered against.
/// Absent if the wgpu backend failed to initialize.
pub struct GpuBits {
    pub baker: Baker,
    pub scene: SceneRenderer,
    pub renderer: Arc<RwLock<egui_wgpu::Renderer>>,
}

pub struct TextureGraphApp {
    pub graph: Graph,
    pub ui: UiState,
    pub previews: PreviewCache,
    pub preview_panel: PreviewPanelState,
    pub eval_ctx: EvalCtx,
    pub gpu: Option<GpuBits>,
}

impl TextureGraphApp {
    pub fn new(cc: &CreationContext<'_>) -> Self {
        // Smaller on web, where per-frame cost bites harder.
        #[cfg(target_arch = "wasm32")]
        let default_size = 256;
        #[cfg(not(target_arch = "wasm32"))]
        let default_size = 512;

        let gpu = cc.wgpu_render_state.as_ref().map(|rs| {
            let device_ctx = DeviceCtx::from_shared(
                Arc::new(rs.adapter.clone()),
                Arc::new(rs.device.clone()),
                Arc::new(rs.queue.clone()),
            );
            let scene = SceneRenderer::new(&device_ctx.device);
            GpuBits {
                baker: Baker::new(device_ctx),
                scene,
                renderer: rs.renderer.clone(),
            }
        });
        if gpu.is_none() {
            log::warn!("no wgpu render state on eframe — preview will use CPU path");
        }

        Self {
            graph: Graph::new(),
            ui: UiState::default(),
            previews: PreviewCache::default(),
            preview_panel: PreviewPanelState::new(default_size),
            eval_ctx: EvalCtx::default(),
            gpu,
        }
    }
}

impl eframe::App for TextureGraphApp {
    #[cfg(target_arch = "wasm32")]
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(&mut *self)
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        // Retire thumbnails invalidated last frame before anything asks for
        // one. `None` means all of them.
        let dirty = self.ui.dirty_previews.replace(HashSet::new());
        self.previews.begin_frame(
            &self.graph,
            self.ui.revision,
            dirty.as_ref(),
            self.gpu.as_ref(),
        );

        egui::Panel::top("menu_bar").show(ui, |ui| {
            panels::menu_bar::show(ui, &self.graph, &mut self.ui);
        });

        egui::Panel::right("preview")
            .resizable(true)
            .default_size(360.0)
            .show(ui, |ui| {
                panels::preview::show(
                    ui,
                    &self.graph,
                    &mut self.ui,
                    &mut self.preview_panel,
                    &self.eval_ctx,
                    self.gpu.as_mut(),
                );
            });

        // Left, because a parameter is an input to the whole graph rather
        // than a property of any node; the right panel is what the graph
        // produces.
        egui::Panel::left("params")
            .resizable(true)
            .default_size(220.0)
            .show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    panels::params::show(ui, &self.graph, &mut self.ui, &mut self.eval_ctx);
                });
            });

        egui::CentralPanel::default().show(ui, |ui| {
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
        });

        // Both platforms run the dialog off the UI thread and report back
        // through `poll_pending`: blocking on it deadlocks on macOS.
        file_io::handle_wants_save(&self.graph, &mut self.ui, ui.ctx());
        file_io::handle_wants_open(&mut self.ui, ui.ctx());
        file_io::poll_pending(&mut self.ui);

        // Apply queued mutations. `dirty_previews` is consumed at the top of
        // the next frame; `state.dirty` belongs to the preview panel, so it
        // stays set here.
        self.ui.drain_into(&mut self.graph);
    }
}
