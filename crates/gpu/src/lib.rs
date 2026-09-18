//! GPU compute pipeline for `texture-graph`.
//!
//! Baker consumes a `Graph` and dispatches one compute shader per layer in
//! topological order, ping-ponging results through a small pool of
//! `Rgba32Float` textures assigned by the "pebble game" scheduler
//! (`schedule.rs`). Everything intermediate stays on the GPU as unclamped
//! Oklcha; only the four final PBR channels are packed to `Rgba8UnormSrgb`
//! and handed back for display.
//!
//! This crate stays independent of eframe/egui — the UI crate re-registers
//! the returned `wgpu::Texture`s via `egui_wgpu::Renderer::register_native_texture`.
//! `examples/bake.rs` is the headless path end to end, and is what keeps that
//! independence honest: it compiles under `cargo test`, so a stray editor
//! dependency would break the build rather than go unnoticed.

pub mod baker;
pub mod device;
pub mod readback;
pub mod scene;
pub mod schedule;

pub use baker::{BakeError, BakeOutput, Baker, VolumeOutput};
pub use device::DeviceCtx;
pub use readback::{Image, read_rgba8};
pub use scene::{SceneCamera, SceneMaterial, SceneRenderer, SceneShape};
pub use schedule::{OutputSlots, ScalarSlot, Schedule, ScheduleError, schedule, schedule_no_reuse};

#[cfg(test)]
mod bake_test;

#[cfg(test)]
mod smoke_test;

#[cfg(test)]
mod scene_test;
