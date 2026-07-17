//! Procedural texture graph — pure data model + evaluator.
//!
//! No rendering, no UI, no async. Consumers (WGPU preview, Tauri frontend,
//! headless bake) drive evaluation by sampling `(u, v, w) ∈ [0, 1]³` and
//! reading the resulting [`Color`].
//!
//! Contract:
//! - Every [`Layer`] produces a [`Color`] (Oklcha), unclamped.
//! - Clamping to display range happens only at the [`Output`] stage.
//! - The graph is a DAG; cycles are rejected at edit time, so the evaluator
//!   assumes acyclicity.

pub mod color;
pub mod eval;
pub mod file;
pub mod graph;
pub mod id;
pub mod kind;

pub use color::{BlendSpace, Color};
pub use eval::{EvalCtx, Sample, evaluate_material};
pub use file::{
    CURRENT_FORMAT_VERSION, FILE_EXTENSION, FileMetadata, LoadError, SaveError, TextureGraphFile,
    load_from_path, load_from_str, save_to_path, save_to_string,
};
pub use graph::{Canvas, Graph, GraphError, Layer, Output};
pub use id::LayerId;
pub use kind::{
    Axis, BlendMode, ColorInput, ColorRamp, ColorStop, CoordMode, HeightToNormal, LayerKind,
    Map, Mix, Noise, NoiseDims, NoiseOutput, NoiseRange, RadialDim, ScalarInput, Transform,
};
