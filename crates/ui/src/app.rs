use std::sync::Arc;

use eframe::CreationContext;
use egui::mutex::RwLock;
use texture_graph_editor::{Editor, GpuBits, TextureRegistry};
use texture_graph_gpu::{Baker, DeviceCtx};

use crate::file_io;
use crate::menu_bar;

/// eframe's renderer, which draws a wgpu view directly.
struct EguiWgpuTextures {
    device: wgpu::Device,
    renderer: Arc<RwLock<egui_wgpu::Renderer>>,
}

impl TextureRegistry for EguiWgpuTextures {
    fn register(&mut self, _texture: &wgpu::Texture, view: &wgpu::TextureView) -> egui::TextureId {
        self.renderer
            .write()
            .register_native_texture(&self.device, view, wgpu::FilterMode::Linear)
    }

    fn free(&mut self, id: egui::TextureId) {
        self.renderer.write().free_texture(&id);
    }
}

pub struct TextureGraphApp {
    pub editor: Editor,
}

impl TextureGraphApp {
    pub fn new(cc: &CreationContext<'_>) -> Self {
        // Smaller on web, where per-frame cost is higher.
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
            let textures = EguiWgpuTextures {
                device: rs.device.clone(),
                renderer: rs.renderer.clone(),
            };
            GpuBits::new(Baker::new(device_ctx), Box::new(textures))
        });
        if gpu.is_none() {
            log::warn!("no wgpu render state on eframe — preview will use CPU path");
        }
        log::info!(
            "startup platform={} render_state={} preview_size={}",
            if cfg!(target_arch = "wasm32") { "web" } else { "native" },
            gpu.is_some(),
            default_size,
        );

        Self { editor: Editor::new(gpu, default_size) }
    }
}

impl eframe::App for TextureGraphApp {
    #[cfg(target_arch = "wasm32")]
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(&mut *self)
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        egui::Panel::top("menu_bar").show(ui, |ui| {
            menu_bar::show(ui, &mut self.editor.ui);
        });

        file_io::handle_wants_save(&self.editor.graph, &mut self.editor.ui, ui.ctx());
        file_io::handle_wants_open(&mut self.editor.ui, ui.ctx());
        file_io::poll_pending(&mut self.editor.ui);

        self.editor.show(ui);
    }
}
