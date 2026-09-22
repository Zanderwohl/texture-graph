//! GPU compute pipeline for `texture-graph`.
//!
//! Baker consumes a `Graph` and dispatches one compute shader per layer in
//! topological order, ping-ponging results through a small pool of
//! `Rgba32Float` textures assigned by the "pebble game" scheduler
//! (`schedule.rs`). Everything intermediate stays on the GPU as unclamped
//! Oklcha; only the four final PBR channels are packed to `Rgba8UnormSrgb`
//! and handed back for display.
//!
//! This crate does not depend on eframe/egui; the UI crate registers the
//! returned `wgpu::Texture`s with `egui_wgpu::Renderer::register_native_texture`.
//! `examples/bake.rs` compiles under `cargo test`, so an editor dependency
//! added here breaks the build.

pub mod baker;
pub mod device;
pub mod readback;
pub mod scene;
pub mod schedule;

pub use baker::{
    BakeError, BakeOutput, Baker, ScalarCube, ScalarFormat, ScalarVolume, SolidBump, VolumeJob,
    VolumeOutput,
};
pub use device::DeviceCtx;
pub use readback::{
    Image, ScalarImage, read_rgba8, read_rgba8_async, read_scalar, read_scalar_async,
    read_scalar_volume, read_scalar_volume_async,
};
pub use scene::{SceneBackground, SceneCamera, SceneLayer, SceneMaterial, SceneRenderer, SceneShape};
pub use schedule::{
    OutputSlots, ScalarSlot, Schedule, ScheduleError, schedule, schedule_layer,
    schedule_no_reuse,
};

#[cfg(test)]
mod bake_test;

#[cfg(test)]
mod smoke_test;

#[cfg(test)]
mod scene_test;

#[cfg(test)]
mod sphere_test;
