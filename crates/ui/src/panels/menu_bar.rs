use texture_graph_core::Graph;

use crate::state::{EditCmd, UiState};

/// Top menu bar. Native builds use "Save" / "Open"; wasm builds show
/// "Download" / "Upload". Actual dialog wiring is added in `file_io.rs`;
/// this file just emits `EditCmd`s and toggles a "wants save/open" flag
/// on the `UiState` (added later).
pub fn show(ui: &mut egui::Ui, _graph: &Graph, state: &mut UiState) {
    egui::MenuBar::new().ui(ui, |ui| {
        ui.menu_button("File", |ui| {
            if ui.button("New").clicked() {
                state.push(EditCmd::Replace(Graph::new()));
                state.last_loaded_name = None;
                ui.close();
            }
            ui.separator();
            #[cfg(target_arch = "wasm32")]
            {
                if ui.button("Upload…").clicked() {
                    state.wants_open = true;
                    ui.close();
                }
                if ui.button("Download").clicked() {
                    state.wants_save = true;
                    ui.close();
                }
            }
            #[cfg(not(target_arch = "wasm32"))]
            {
                if ui.button("Open…").clicked() {
                    state.wants_open = true;
                    ui.close();
                }
                if ui.button("Save…").clicked() {
                    state.wants_save = true;
                    ui.close();
                }
            }
        });
    });
}
