//! Procedural texture graph: data model and CPU evaluator.
//!
//! No rendering, no UI, no async. Sample at `(u, v, w) ∈ [0, 1]³` and read
//! back a [`Color`].
//!
//! - Every [`Layer`] produces unclamped Oklcha.
//! - Clamping to display range happens only at [`Output`].
//! - Cycles are rejected at edit time, so the evaluator assumes a DAG.

pub mod color;
pub mod crater;
pub mod eval;
pub mod file;
pub mod graph;
pub mod id;
pub mod kind;
pub mod noise;
pub mod param;
pub mod socket;
pub mod sphere;

pub use color::{BlendSpace, Color};
pub use eval::{EvalCtx, FLAT_W, Material, Sample, evaluate, evaluate_material};
pub use file::{
    CURRENT_FORMAT_VERSION, FILE_EXTENSION, FileMetadata, LoadError, SaveError, TextureGraphFile,
    load_from_str, save_to_string,
};
#[cfg(feature = "std-fs")]
pub use file::{load_from_path, save_to_path};
pub use graph::{Canvas, Graph, GraphError, Layer, Output};
pub use id::LayerId;
pub use kind::{
    Axis, BlendMode, ColorInput, ColorRamp, ColorStop, CoordMode, Coordinate, CraterOutput,
    CraterSurface, Craters, Criterion, EXTEND_LIMIT,
    EdgeMode, Fractal, FractalMode, HeightToNormal, LayerKind, Map, MinMax, MinMaxMode, Mix,
    Noise, NoiseDims, NoiseKernel, NoiseOutput, NoiseRange, RadialDim, ScalarInput,
    Transform, Warp, WarpMode, Wave, WaveShape,
};
pub use param::{ParamDecl, ParamKind, ParamUse, ParamValue};
pub use socket::{ConstValue, InputKey, InputSocket, SocketError, SocketValue};
pub use sphere::{CUBE_FACES, cube_direction, cube_sample};
