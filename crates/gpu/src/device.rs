//! wgpu device/queue initialization for native and wasm.
//!
//! The UI crate normally passes in a `Device`+`Queue` it already owns via
//! `eframe::CreationContext::wgpu_render_state`, so egui and compute share
//! one device. `DeviceCtx::request_headless` exists for tests and headless
//! bakes.

use std::sync::Arc;

/// The wgpu handles the baker needs. Cheap to clone.
#[derive(Clone)]
pub struct DeviceCtx {
    pub adapter: Arc<wgpu::Adapter>,
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
}

impl DeviceCtx {
    /// Wrap handles the caller already owns, e.g. from eframe's
    /// `CreationContext::wgpu_render_state`. No `Instance` is needed.
    pub fn from_shared(
        adapter: Arc<wgpu::Adapter>,
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
    ) -> Self {
        log_device("shared", &adapter, &device);
        Self { adapter, device, queue }
    }

    /// Picks a high-performance adapter with no surface.
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
            .map_err(|e| {
                log::debug!("request_adapter failed err={e}");
                DeviceInitError::NoAdapter
            })?;
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
            .map_err(|e| {
                log::debug!("request_device failed err={e}");
                DeviceInitError::RequestDevice(e)
            })?;
        log_device("headless", &adapter, &device);
        Ok(Self {
            adapter: Arc::new(adapter),
            device: Arc::new(device),
            queue: Arc::new(queue),
        })
    }
}

fn log_device(source: &str, adapter: &wgpu::Adapter, device: &wgpu::Device) {
    let info = adapter.get_info();
    let limits = device.limits();
    log::info!(
        "gpu device source={source} adapter={:?} backend={:?} type={:?} driver={:?} \
         max_texture_2d={} max_texture_3d={} max_storage_textures_per_stage={} \
         max_buffer_size={}",
        info.name,
        info.backend,
        info.device_type,
        info.driver,
        limits.max_texture_dimension_2d,
        limits.max_texture_dimension_3d,
        limits.max_storage_textures_per_shader_stage,
        limits.max_buffer_size,
    );
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
