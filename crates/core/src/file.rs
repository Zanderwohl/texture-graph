//! On-disk format: RON, with a metadata block and the graph payload.
//!
//! A file must start with `TextureGraphFile(`, which doubles as the format's
//! magic; anything else is rejected before RON sees it. Extension `.tgraph`.
//!
//! [`load_from_str`] and [`save_to_string`] are unconditional; the path
//! helpers sit behind the default-on `std-fs` feature. `std::fs` compiles
//! for `wasm32-unknown-unknown` and then fails at runtime, so a wasm
//! consumer takes `default-features = false` and gets a compile error
//! instead of a mystery at load time.

#[cfg(feature = "std-fs")]
use std::fs;
#[cfg(feature = "std-fs")]
use std::path::Path;

use ron::ser::PrettyConfig;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::graph::Graph;

/// Not enforced on load, but `save_to_path` supplies it to an extensionless
/// path.
pub const FILE_EXTENSION: &str = "tgraph";

/// Current on-disk format version. Bump on any breaking layout change.
///
/// RON tags enum variants by name, so a `LayerKind` variant added here
/// keeps old files loading in new builds — it is the other direction that
/// needs the version, and that is what a reader older than a writer checks.
/// Bumped once for the batch in
/// `documentation/game-consumer-features.md`, not once per node.
pub const CURRENT_FORMAT_VERSION: u32 = 2;

/// Structural magic — the RON parser sees this literal as the top-level
/// struct name.
const MAGIC_PREFIX: &str = "TextureGraphFile(";

/// Root on-disk record.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TextureGraphFile {
    pub format_version: u32,
    pub metadata: FileMetadata,
    pub graph: Graph,
}

/// Human-authored header that travels with the graph. All fields are
/// optional in intent: empty strings and empty vectors are fine.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct FileMetadata {
    /// Human-readable name for the graph. May differ from the filename.
    pub name: String,
    /// Free-form description.
    pub description: Option<String>,
    pub authors: Vec<String>,
    /// ISO-8601 UTC. Caller-supplied, so this crate stays clock-free.
    pub modified: String,
    /// Version string of the application that wrote this file. Free-form.
    pub written_by: String,
}

impl TextureGraphFile {
    /// Package `graph` at the current format version with the given
    /// metadata.
    pub fn new(metadata: FileMetadata, graph: Graph) -> Self {
        Self {
            format_version: CURRENT_FORMAT_VERSION,
            metadata,
            graph,
        }
    }
}

#[derive(Debug, Error)]
pub enum SaveError {
    #[cfg(feature = "std-fs")]
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("serialize: {0}")]
    Serialize(#[from] ron::Error),
}

#[derive(Debug, Error)]
pub enum LoadError {
    #[cfg(feature = "std-fs")]
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("not a texture-graph file (missing `{MAGIC_PREFIX}` magic)")]
    NotATextureGraph,
    #[error("unsupported format version {0}; this build understands {CURRENT_FORMAT_VERSION}")]
    UnsupportedVersion(u32),
    #[error("deserialize: {0}")]
    Deserialize(#[from] ron::error::SpannedError),
}

/// Pretty RON. Struct names are on, so the top-level token is the magic
/// marker; field order is stable and diffs readably.
pub fn save_to_string(file: &TextureGraphFile) -> Result<String, ron::Error> {
    let cfg = PrettyConfig::new()
        .depth_limit(usize::MAX)
        .indentor("    ".to_string())
        .struct_names(true)
        .separate_tuple_members(false)
        .enumerate_arrays(false);
    ron::ser::to_string_pretty(file, cfg)
}

/// Parse a texture-graph file from RON text. Validates the magic marker
/// and the format version before deserializing.
pub fn load_from_str(s: &str) -> Result<TextureGraphFile, LoadError> {
    let trimmed = s.trim_start();
    if !trimmed.starts_with(MAGIC_PREFIX) {
        return Err(LoadError::NotATextureGraph);
    }
    // `implicit_some` accepts a bare `source: 3` for an `Option<LayerId>`.
    let options = ron::Options::default()
        .with_default_extension(ron::extensions::Extensions::IMPLICIT_SOME);
    let file: TextureGraphFile = options.from_str(s)?;
    if file.format_version > CURRENT_FORMAT_VERSION {
        return Err(LoadError::UnsupportedVersion(file.format_version));
    }
    Ok(file)
}

/// Write `file` to disk as pretty RON. If `path` has no extension, appends
/// [`FILE_EXTENSION`].
#[cfg(feature = "std-fs")]
pub fn save_to_path(file: &TextureGraphFile, path: impl AsRef<Path>) -> Result<(), SaveError> {
    let mut p = path.as_ref().to_path_buf();
    if p.extension().is_none() {
        p.set_extension(FILE_EXTENSION);
    }
    let s = save_to_string(file)?;
    fs::write(p, s)?;
    Ok(())
}

/// Read a texture-graph file from disk.
#[cfg(feature = "std-fs")]
pub fn load_from_path(path: impl AsRef<Path>) -> Result<TextureGraphFile, LoadError> {
    let s = fs::read_to_string(path)?;
    load_from_str(&s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::oklcha;
    use crate::kind::LayerKind;

    fn sample_metadata() -> FileMetadata {
        FileMetadata {
            name: "test".to_string(),
            description: Some("round trip".to_string()),
            authors: vec!["zandy".to_string()],
            modified: "2026-07-17T00:00:00Z".to_string(),
            written_by: "texture-graph-core tests".to_string(),
        }
    }

    #[test]
    fn round_trip_default_graph() {
        let file = TextureGraphFile::new(sample_metadata(), Graph::new());
        let s = save_to_string(&file).unwrap();
        assert!(s.trim_start().starts_with(MAGIC_PREFIX));
        let back = load_from_str(&s).unwrap();
        assert_eq!(back.format_version, CURRENT_FORMAT_VERSION);
        assert_eq!(back.metadata.name, "test");
        assert_eq!(back.graph.layers.len(), file.graph.layers.len());
    }

    /// A graph written before the value kernel, the period and the
    /// fractal stack existed still loads, and loads as the node it meant:
    /// one octave of aperiodic simplex.
    #[test]
    fn a_v1_noise_layer_loads_as_aperiodic_single_octave_simplex() {
        let v1 = r#"TextureGraphFile(
    format_version: 1,
    metadata: (name: "old", description: None, authors: [], modified: "", written_by: ""),
    graph: (
        layers: [(id: 1, name: "n", kind: Noise((
            dims: D2,
            seed_offset: 0,
            frequency: 4.0,
            range: Unsigned,
            output: Grayscale,
        )))],
        list_order: [1],
        canvases: {},
        output: (color: 1, roughness: Const(0.5), metallic: Const(0.0), normal: None),
        next_id: 2,
    ),
)"#;
        let file = load_from_str(v1).expect("a v1 file must still load");
        let crate::kind::LayerKind::Noise(n) = &file.graph.layers[0].kind else {
            panic!("expected a Noise layer")
        };
        assert_eq!(n.kernel, crate::kind::NoiseKernel::Simplex);
        assert_eq!(n.period, [0; 3]);
        assert_eq!(n.fractal, crate::kind::Fractal::default());
        assert_eq!(n.fractal.octaves, 1);
    }

    /// The other direction is what the version is for: a file from a
    /// newer build is refused rather than half-read.
    #[test]
    fn a_future_version_is_refused() {
        let s = save_to_string(&TextureGraphFile {
            format_version: CURRENT_FORMAT_VERSION + 1,
            metadata: sample_metadata(),
            graph: Graph::new(),
        })
        .unwrap();
        assert!(matches!(
            load_from_str(&s).unwrap_err(),
            LoadError::UnsupportedVersion(_)
        ));
    }

    #[test]
    fn rejects_non_texture_graph_file() {
        let err = load_from_str("SomeOtherFormat(foo: 1)").unwrap_err();
        assert!(matches!(err, LoadError::NotATextureGraph));
    }

    #[test]
    fn diff_stable_across_display_reorder() {
        let mut g = Graph::new();
        let a = g.add_layer("a", LayerKind::Color(oklcha(0.1, 0.0, 0.0, 1.0))).unwrap();
        let b = g.add_layer("b", LayerKind::Color(oklcha(0.9, 0.0, 0.0, 1.0))).unwrap();
        let before = save_to_string(&TextureGraphFile::new(sample_metadata(), g.clone())).unwrap();
        // Reversing the list_order changes UI display but must not touch
        // the serialized `layers` payload.
        g.set_list_position(a, 1).unwrap();
        g.set_list_position(b, 0).unwrap();
        let after = save_to_string(&TextureGraphFile::new(sample_metadata(), g)).unwrap();
        // The layers block is identical; only `list_order` differs.
        let extract_layers = |s: &str| {
            let start = s.find("layers:").unwrap();
            let end = s[start..].find("list_order:").unwrap();
            s[start..start + end].to_string()
        };
        assert_eq!(extract_layers(&before), extract_layers(&after));
    }
}
