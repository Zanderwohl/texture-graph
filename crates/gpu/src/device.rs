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
    pub instance: Arc<wgpu::Instance>,
    pub adapter: Arc<wgpu::Adapter>,
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
}

impl DeviceCtx {
    /// Build a `DeviceCtx` from handles the caller already owns (e.g. the
    /// ones eframe hands out in `CreationContext::wgpu_render_state`).
    pub fn from_shared(
        instance: Arc<wgpu::Instance>,
        adapter: Arc<wgpu::Adapter>,
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
    ) -> Self {
        Self { instance, adapter, device, queue }
    }

    /// Async headless init. Picks a HighPerformance adapter with no surface.
    /// Used by unit tests and any future CLI bake tool.
    pub async fn request_headless() -> Result<Self, DeviceInitError> {
        let instance = Arc::new(wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        }));
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
            })
            .await
            .map_err(DeviceInitError::RequestDevice)?;
        Ok(Self {
            instance,
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
