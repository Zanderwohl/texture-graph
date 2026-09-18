//! Texture-graph editor UI. Same crate serves the native `texture-graph`
//! binary and the wasm build driven by Trunk (see `index.html`).

pub mod app;
pub mod color_convert;
pub mod file_io;
pub mod panels;
pub mod previews;
pub mod state;
pub mod util;
pub mod widgets;

pub use app::TextureGraphApp;

/// Native entry point. Called from `main.rs`.
#[cfg(not(target_arch = "wasm32"))]
pub fn run_native() -> eframe::Result<()> {
    env_logger::init();
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_min_inner_size([640.0, 400.0])
            .with_title("Texture Graph"),
        ..Default::default()
    };
    eframe::run_native(
        "Texture Graph",
        options,
        Box::new(|cc| Ok(Box::new(TextureGraphApp::new(cc)))),
    )
}

// Owns the `WebRunner` for the whole life of the wasm module. Dropping
// the runner uninstalls its event listeners and cancels its animation
// frame — so it must outlive the async setup that installed them.
#[cfg(target_arch = "wasm32")]
thread_local! {
    static RUNNER: std::cell::RefCell<Option<eframe::WebRunner>> =
        const { std::cell::RefCell::new(None) };
}

/// Web entry point. Wired up by Trunk via `#[wasm_bindgen(start)]`.
///
/// wasm-bindgen's `start` attribute wants a synchronous function; we
/// spawn the async eframe setup onto the browser event loop so errors
/// aren't silently swallowed by a fire-and-forgotten promise.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn start_web() {
    console_error_panic_hook::set_once();
    let _ = console_log::init_with_level(log::Level::Info);
    wasm_bindgen_futures::spawn_local(async move {
        if let Err(e) = run_web().await {
            web_sys::console::error_1(&e);
        }
    });
}

#[cfg(target_arch = "wasm32")]
async fn run_web() -> Result<(), wasm_bindgen::JsValue> {
    use eframe::wasm_bindgen::JsCast as _;
    let document = web_sys::window()
        .and_then(|w| w.document())
        .expect("no document");
    let canvas = document
        .get_element_by_id("texture-graph-canvas")
        .expect("missing #texture-graph-canvas")
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .expect("#texture-graph-canvas is not a <canvas>");
    let runner = eframe::WebRunner::new();
    runner
        .start(
            canvas,
            eframe::WebOptions::default(),
            Box::new(|cc| Ok(Box::new(TextureGraphApp::new(cc)))),
        )
        .await?;
    // Park the runner so its event handlers + animation-frame callbacks
    // outlive this async setup future.
    RUNNER.with(|cell| cell.borrow_mut().replace(runner));
    Ok(())
}
