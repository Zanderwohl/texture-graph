#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result<()> {
    texture_graph_ui::run_native()
}

#[cfg(target_arch = "wasm32")]
fn main() {}
