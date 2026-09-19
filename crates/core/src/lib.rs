//! Procedural texture graph: data model and CPU evaluator.
//!
//! No rendering, no UI, no async. Sample at `(u, v, w) ∈ [0, 1]³` and read
//! back a [`Color`].
//!
//! - Every [`Layer`] produces unclamped Oklcha.
//! - Clamping to display range happens only at [`Output`].
//! - Cycles are rejected at edit time, so the evaluator assumes a DAG.

pub mod color;
pub mod eval;
pub mod file;
pub mod graph;
pub mod id;
pub mod kind;
pub mod noise;
pub mod socket;

pub use color::{BlendSpace, Color};
pub use eval::{EvalCtx, FLAT_W, Material, Sample, evaluate, evaluate_material};
pub use file::{
    CURRENT_FORMAT_VERSION, FILE_EXTENSION, FileMetadata, LoadError, SaveError, TextureGraphFile,
    load_from_path, load_from_str, save_to_path, save_to_string,
};
pub use graph::{Canvas, Graph, GraphError, Layer, Output};
pub use id::LayerId;
pub use kind::{
    Axis, BlendMode, ColorInput, ColorRamp, ColorStop, CoordMode, Criterion, EXTEND_LIMIT,
    EdgeMode, HeightToNormal, LayerKind, Map, MinMax, MinMaxMode, Mix, Noise, NoiseDims,
    NoiseOutput, NoiseRange, RadialDim, ScalarInput, Transform,
};
pub use socket::{ConstValue, InputKey, InputSocket, SocketError, SocketValue};
