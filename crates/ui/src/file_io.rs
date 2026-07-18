//! File save/load. Native uses `rfd::AsyncFileDialog` via `pollster`.
//! Web (wasm) path is stubbed here and finished off in the web-build step.
//!
//! Called from the app once per frame when `UiState::wants_save` or
//! `wants_open` is set; consumes the flag either way.

use texture_graph_core::{FILE_EXTENSION, FileMetadata, Graph, TextureGraphFile};

use crate::state::{EditCmd, UiState};

pub fn handle_wants_save(graph: &Graph, state: &mut UiState) {
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
    do_save(state, file);
}

pub fn handle_wants_open(state: &mut UiState) {
    if !state.wants_open {
        return;
    }
    state.wants_open = false;
    do_open(state);
}

// ---- Native ------------------------------------------------------------

#[cfg(not(target_arch = "wasm32"))]
fn do_save(state: &mut UiState, file: TextureGraphFile) {
    let Some(handle) = pollster::block_on(
        rfd::AsyncFileDialog::new()
            .set_file_name(&format!("{}.{}", file.metadata.name, FILE_EXTENSION))
            .add_filter("Texture Graph", &[FILE_EXTENSION])
            .save_file(),
    ) else {
        return;
    };
    let path = handle.path().to_path_buf();
    match texture_graph_core::save_to_string(&file) {
        Ok(s) => {
            if let Err(e) = std::fs::write(&path, s) {
                state.last_error = Some(format!("write failed: {e}"));
            } else {
                state.last_loaded_name = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .map(|s| s.to_string());
                state.last_error = None;
            }
        }
        Err(e) => state.last_error = Some(format!("serialize failed: {e}")),
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn do_open(state: &mut UiState) {
    let Some(handle) = pollster::block_on(
        rfd::AsyncFileDialog::new()
            .add_filter("Texture Graph", &[FILE_EXTENSION])
            .pick_file(),
    ) else {
        return;
    };
    let path = handle.path().to_path_buf();
    match std::fs::read_to_string(&path) {
        Ok(s) => match texture_graph_core::load_from_str(&s) {
            Ok(file) => {
                state.last_loaded_name = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .map(|s| s.to_string());
                state.push(EditCmd::Replace(file.graph));
                state.last_error = None;
            }
            Err(e) => state.last_error = Some(format!("load failed: {e}")),
        },
        Err(e) => state.last_error = Some(format!("read failed: {e}")),
    }
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
fn do_save(_state: &mut UiState, file: TextureGraphFile) {
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
fn do_open(_state: &mut UiState) {
    wasm_bindgen_futures::spawn_local(async move {
        if let Some(handle) = rfd::AsyncFileDialog::new()
            .add_filter("Texture Graph", &[FILE_EXTENSION])
            .pick_file()
            .await
        {
            let bytes = handle.read().await;
            PENDING_LOAD.with(|cell| *cell.borrow_mut() = Some(bytes));
        }
    });
}

/// Called every frame on wasm; picks up any file the user just uploaded
/// and turns it into an `EditCmd::Replace`. No-op on native.
pub fn poll_pending(state: &mut UiState) {
    #[cfg(target_arch = "wasm32")]
    {
        let bytes = PENDING_LOAD.with(|cell| cell.borrow_mut().take());
        if let Some(bytes) = bytes {
            match String::from_utf8(bytes) {
                Ok(s) => match texture_graph_core::load_from_str(&s) {
                    Ok(file) => {
                        state.push(EditCmd::Replace(file.graph));
                        state.last_error = None;
                    }
                    Err(e) => state.last_error = Some(format!("load failed: {e}")),
                },
                Err(_) => state.last_error = Some("load failed: not UTF-8".into()),
            }
        }
    }
    #[cfg(not(target_arch = "wasm32"))]
    let _ = state;
}
