//! wgpu device/queue initialization for native and wasm.
//!
//! The UI crate normally passes in a `Device`+`Queue` it already owns via
//! `eframe::CreationContext::wgpu_render_state`, so egui and compute share
//! one device. `DeviceCtx::request_headless` exists for tests and headless
//! bakes.

use std::sync::Arc;

/// Wrapper around the wgpu handles the baker needs. Cheap to clone —
/// wgpu resources are `Arc`-backed under the hood.
#[derive(Clone)]
pub struct DeviceCtx {
    pub adapter: Arc<wgpu::Adapter>,
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
}

impl DeviceCtx {
    /// Build a `DeviceCtx` from handles the caller already owns (e.g. the
    /// ones eframe hands out in `CreationContext::wgpu_render_state`).
    /// The wgpu `Instance` is not required — it's only used during initial
    /// adapter enumeration.
    pub fn from_shared(
        adapter: Arc<wgpu::Adapter>,
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
    ) -> Self {
        Self { adapter, device, queue }
    }

    /// Async headless init. Picks a HighPerformance adapter with no surface.
    /// Used by unit tests and any future CLI bake tool.
    pub async fn request_headless() -> Result<Self, DeviceInitError> {
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = wgpu::Backends::PRIMARY;
        let instance = wgpu::Instance::new(desc);
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .map_err(|_| DeviceInitError::NoAdapter)?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("texture-graph headless device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
                ..Default::default()
            })
            .await
            .map_err(DeviceInitError::RequestDevice)?;
        Ok(Self {
            adapter: Arc::new(adapter),
            device: Arc::new(device),
            queue: Arc::new(queue),
        })
    }
}

#[derive(Debug)]
pub enum DeviceInitError {
    NoAdapter,
    RequestDevice(wgpu::RequestDeviceError),
}

impl std::fmt::Display for DeviceInitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeviceInitError::NoAdapter => f.write_str("no suitable GPU adapter"),
            DeviceInitError::RequestDevice(e) => write!(f, "request_device: {e}"),
        }
    }
}

impl std::error::Error for DeviceInitError {}
