//! File save/load. Both platforms run `rfd::AsyncFileDialog` off the UI
//! thread (native on a thread, web via `spawn_local`) and leave the result
//! for [`poll_pending`].
//!
//! Native must not block the UI thread on the dialog: macOS runs the panel on
//! the main queue, so the app hangs and no dialog appears.

use texture_graph_core::{FILE_EXTENSION, FileMetadata, Graph, TextureGraphFile};

use crate::state::{EditCmd, UiState};

pub fn handle_wants_save(graph: &Graph, state: &mut UiState, ctx: &egui::Context) {
    if !state.wants_save {
        return;
    }
    state.wants_save = false;
    let metadata = FileMetadata {
        name: state
            .last_loaded_name
            .clone()
            .unwrap_or_else(|| "untitled".to_string()),
        modified: String::new(),
        written_by: format!("texture-graph-ui {}", env!("CARGO_PKG_VERSION")),
        ..FileMetadata::default()
    };
    let file = TextureGraphFile::new(metadata, graph.clone());
    do_save(state, file, ctx);
}

pub fn handle_wants_open(state: &mut UiState, ctx: &egui::Context) {
    if !state.wants_open {
        return;
    }
    state.wants_open = false;
    do_open(state, ctx);
}

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use std::sync::Mutex;
    use std::sync::atomic::AtomicBool;

    /// One dialog at a time — a second request while one is up is dropped.
    pub static DIALOG_OPEN: AtomicBool = AtomicBool::new(false);
    pub static PENDING: Mutex<Option<Outcome>> = Mutex::new(None);

    pub enum Outcome {
        /// File text read from disk, ready to parse on the UI thread.
        Opened { stem: Option<String>, text: String },
        Saved { stem: Option<String> },
        Failed(String),
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn file_stem(path: &std::path::Path) -> Option<String> {
    path.file_stem().and_then(|s| s.to_str()).map(String::from)
}

#[cfg(not(target_arch = "wasm32"))]
fn finish_dialog(outcome: Option<native::Outcome>, ctx: &egui::Context) {
    *native::PENDING.lock().unwrap() = outcome;
    native::DIALOG_OPEN.store(false, std::sync::atomic::Ordering::SeqCst);
    // Wake the UI even with no input events arriving.
    ctx.request_repaint();
}

#[cfg(not(target_arch = "wasm32"))]
fn do_save(state: &mut UiState, file: TextureGraphFile, ctx: &egui::Context) {
    use std::sync::atomic::Ordering;
    if native::DIALOG_OPEN.swap(true, Ordering::SeqCst) {
        log::debug!("save ignored reason=dialog_open");
        return;
    }
    let text = match texture_graph_core::save_to_string(&file) {
        Ok(s) => s,
        Err(e) => {
            log::warn!("save failed stage=serialize error={e}");
            state.last_error = Some(format!("serialize failed: {e}"));
            native::DIALOG_OPEN.store(false, Ordering::SeqCst);
            return;
        }
    };
    let filename = format!("{}.{}", file.metadata.name, FILE_EXTENSION);
    let layers = file.graph.layers.len();
    let version = file.format_version;
    log::debug!("save dialog filename={:?}", filename);
    let ctx = ctx.clone();
    std::thread::spawn(move || {
        let outcome = pollster::block_on(
            rfd::AsyncFileDialog::new()
                .set_file_name(&filename)
                .add_filter("Texture Graph", &[FILE_EXTENSION])
                .save_file(),
        )
        .map(|handle| {
            let path = handle.path().to_path_buf();
            match std::fs::write(&path, text) {
                Ok(()) => {
                    log::info!(
                        "saved file path={:?} layers={} format_version={}",
                        path,
                        layers,
                        version
                    );
                    native::Outcome::Saved { stem: file_stem(&path) }
                }
                Err(e) => {
                    log::warn!("save failed stage=write path={:?} error={e}", path);
                    native::Outcome::Failed(format!("write failed: {e}"))
                }
            }
        });
        if outcome.is_none() {
            log::debug!("save canceled");
        }
        finish_dialog(outcome, &ctx);
    });
}

#[cfg(not(target_arch = "wasm32"))]
fn do_open(_state: &mut UiState, ctx: &egui::Context) {
    use std::sync::atomic::Ordering;
    if native::DIALOG_OPEN.swap(true, Ordering::SeqCst) {
        log::debug!("open ignored reason=dialog_open");
        return;
    }
    log::debug!("open dialog");
    let ctx = ctx.clone();
    std::thread::spawn(move || {
        let outcome = pollster::block_on(
            rfd::AsyncFileDialog::new()
                .add_filter("Texture Graph", &[FILE_EXTENSION])
                .pick_file(),
        )
        .map(|handle| {
            let path = handle.path().to_path_buf();
            match std::fs::read_to_string(&path) {
                Ok(text) => {
                    log::debug!("read file path={:?} bytes={}", path, text.len());
                    native::Outcome::Opened { stem: file_stem(&path), text }
                }
                Err(e) => {
                    log::warn!("open failed stage=read path={:?} error={e}", path);
                    native::Outcome::Failed(format!("read failed: {e}"))
                }
            }
        });
        if outcome.is_none() {
            log::debug!("open canceled");
        }
        finish_dialog(outcome, &ctx);
    });
}

#[cfg(target_arch = "wasm32")]
use std::cell::RefCell;

#[cfg(target_arch = "wasm32")]
thread_local! {
    /// Bytes from a finished open dialog, taken by [`poll_pending`].
    static PENDING_LOAD: RefCell<Option<Vec<u8>>> = const { RefCell::new(None) };
}

#[cfg(target_arch = "wasm32")]
fn do_save(_state: &mut UiState, file: TextureGraphFile, _ctx: &egui::Context) {
    let s = match texture_graph_core::save_to_string(&file) {
        Ok(s) => s,
        Err(e) => {
            log::warn!("save failed stage=serialize error={e}");
            return;
        }
    };
    let filename = format!("{}.{}", file.metadata.name, FILE_EXTENSION);
    let layers = file.graph.layers.len();
    let version = file.format_version;
    log::debug!("save dialog filename={:?}", filename);
    wasm_bindgen_futures::spawn_local(async move {
        if let Some(handle) = rfd::AsyncFileDialog::new()
            .set_file_name(&filename)
            .add_filter("Texture Graph", &[FILE_EXTENSION])
            .save_file()
            .await
        {
            match handle.write(s.as_bytes()).await {
                Ok(()) => log::info!(
                    "saved file name={:?} layers={} format_version={}",
                    handle.file_name(),
                    layers,
                    version
                ),
                Err(e) => log::warn!("save failed stage=write error={e}"),
            }
        } else {
            log::debug!("save canceled");
        }
    });
}

#[cfg(target_arch = "wasm32")]
fn do_open(_state: &mut UiState, ctx: &egui::Context) {
    log::debug!("open dialog");
    let ctx = ctx.clone();
    wasm_bindgen_futures::spawn_local(async move {
        if let Some(handle) = rfd::AsyncFileDialog::new()
            .add_filter("Texture Graph", &[FILE_EXTENSION])
            .pick_file()
            .await
        {
            let bytes = handle.read().await;
            log::debug!("read file name={:?} bytes={}", handle.file_name(), bytes.len());
            PENDING_LOAD.with(|cell| *cell.borrow_mut() = Some(bytes));
            ctx.request_repaint();
        } else {
            log::debug!("open canceled");
        }
    });
}

/// Called every frame.
pub fn poll_pending(state: &mut UiState) {
    #[cfg(target_arch = "wasm32")]
    {
        let bytes = PENDING_LOAD.with(|cell| cell.borrow_mut().take());
        if let Some(bytes) = bytes {
            match String::from_utf8(bytes) {
                Ok(s) => apply_loaded(state, None, &s),
                Err(_) => {
                    log::warn!("open failed stage=decode error=\"not UTF-8\"");
                    state.last_error = Some("load failed: not UTF-8".into());
                }
            }
        }
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let outcome = native::PENDING.lock().unwrap().take();
        match outcome {
            Some(native::Outcome::Opened { stem, text }) => apply_loaded(state, stem, &text),
            Some(native::Outcome::Saved { stem }) => {
                if stem.is_some() {
                    state.last_loaded_name = stem;
                }
                state.last_error = None;
            }
            Some(native::Outcome::Failed(e)) => state.last_error = Some(e),
            None => {}
        }
    }
}

fn apply_loaded(state: &mut UiState, stem: Option<String>, text: &str) {
    match texture_graph_core::load_from_str(text) {
        Ok(file) => {
            log::info!(
                "opened file name={:?} layers={} params={} format_version={}",
                stem.as_deref().unwrap_or(&file.metadata.name),
                file.graph.layers.len(),
                file.graph.params.len(),
                file.format_version,
            );
            if stem.is_some() {
                state.last_loaded_name = stem;
            }
            state.push(EditCmd::Replace(file.graph));
            state.last_error = None;
        }
        Err(e) => {
            log::warn!("open failed stage=parse error={e}");
            state.last_error = Some(format!("load failed: {e}"));
        }
    }
}
