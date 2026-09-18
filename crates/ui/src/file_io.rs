//! File save/load. Both native and web run `rfd::AsyncFileDialog` off the
//! UI: native on a background thread, web via `spawn_local`. Results land
//! in a pending slot polled once per frame by [`poll_pending`].
//!
//! Native MUST NOT block the UI thread on the dialog future: on macOS the
//! panel is dispatched to the main queue, so parking the main thread while
//! waiting deadlocks — the dialog never appears and the app hangs.
//!
//! Called from the app once per frame when `UiState::wants_save` or
//! `wants_open` is set; consumes the flag either way.

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

// ---- Native ------------------------------------------------------------

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use std::sync::Mutex;
    use std::sync::atomic::AtomicBool;

    /// One dialog at a time — a second request while one is up is dropped.
    pub static DIALOG_OPEN: AtomicBool = AtomicBool::new(false);
    /// Outcome of the last finished dialog, consumed by `poll_pending`.
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
    // Wake the UI even if the user isn't generating input events.
    ctx.request_repaint();
}

#[cfg(not(target_arch = "wasm32"))]
fn do_save(state: &mut UiState, file: TextureGraphFile, ctx: &egui::Context) {
    use std::sync::atomic::Ordering;
    if native::DIALOG_OPEN.swap(true, Ordering::SeqCst) {
        return;
    }
    let text = match texture_graph_core::save_to_string(&file) {
        Ok(s) => s,
        Err(e) => {
            state.last_error = Some(format!("serialize failed: {e}"));
            native::DIALOG_OPEN.store(false, Ordering::SeqCst);
            return;
        }
    };
    let filename = format!("{}.{}", file.metadata.name, FILE_EXTENSION);
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
                Ok(()) => native::Outcome::Saved { stem: file_stem(&path) },
                Err(e) => native::Outcome::Failed(format!("write failed: {e}")),
            }
        });
        finish_dialog(outcome, &ctx);
    });
}

#[cfg(not(target_arch = "wasm32"))]
fn do_open(_state: &mut UiState, ctx: &egui::Context) {
    use std::sync::atomic::Ordering;
    if native::DIALOG_OPEN.swap(true, Ordering::SeqCst) {
        return;
    }
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
                Ok(text) => native::Outcome::Opened { stem: file_stem(&path), text },
                Err(e) => native::Outcome::Failed(format!("read failed: {e}")),
            }
        });
        finish_dialog(outcome, &ctx);
    });
}

// ---- Web (wasm) --------------------------------------------------------

#[cfg(target_arch = "wasm32")]
use std::cell::RefCell;

#[cfg(target_arch = "wasm32")]
thread_local! {
    /// File bytes handed back from an in-flight open dialog. Polled every
    /// frame by [`poll_pending`] and turned into an `EditCmd::Replace`.
    static PENDING_LOAD: RefCell<Option<Vec<u8>>> = const { RefCell::new(None) };
}

#[cfg(target_arch = "wasm32")]
fn do_save(_state: &mut UiState, file: TextureGraphFile, _ctx: &egui::Context) {
    let s = match texture_graph_core::save_to_string(&file) {
        Ok(s) => s,
        Err(_) => return,
    };
    let filename = format!("{}.{}", file.metadata.name, FILE_EXTENSION);
    wasm_bindgen_futures::spawn_local(async move {
        if let Some(handle) = rfd::AsyncFileDialog::new()
            .set_file_name(&filename)
            .add_filter("Texture Graph", &[FILE_EXTENSION])
            .save_file()
            .await
        {
            let _ = handle.write(s.as_bytes()).await;
        }
    });
}

#[cfg(target_arch = "wasm32")]
fn do_open(_state: &mut UiState, ctx: &egui::Context) {
    let ctx = ctx.clone();
    wasm_bindgen_futures::spawn_local(async move {
        if let Some(handle) = rfd::AsyncFileDialog::new()
            .add_filter("Texture Graph", &[FILE_EXTENSION])
            .pick_file()
            .await
        {
            let bytes = handle.read().await;
            PENDING_LOAD.with(|cell| *cell.borrow_mut() = Some(bytes));
            ctx.request_repaint();
        }
    });
}

/// Called every frame; picks up the outcome of any finished dialog and
/// applies it to the UI state.
pub fn poll_pending(state: &mut UiState) {
    #[cfg(target_arch = "wasm32")]
    {
        let bytes = PENDING_LOAD.with(|cell| cell.borrow_mut().take());
        if let Some(bytes) = bytes {
            match String::from_utf8(bytes) {
                Ok(s) => apply_loaded(state, None, &s),
                Err(_) => state.last_error = Some("load failed: not UTF-8".into()),
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

/// Parse loaded file text and queue the graph replacement.
fn apply_loaded(state: &mut UiState, stem: Option<String>, text: &str) {
    match texture_graph_core::load_from_str(text) {
        Ok(file) => {
            if stem.is_some() {
                state.last_loaded_name = stem;
            }
            state.push(EditCmd::Replace(file.graph));
            state.last_error = None;
        }
        Err(e) => state.last_error = Some(format!("load failed: {e}")),
    }
}
