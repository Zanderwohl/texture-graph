//! Reading baked pixels back to the CPU. See `examples/bake.rs`.
//!
//! - Blocking ([`read_rgba8`], [`read_scalar`], [`read_scalar_volume`])
//!   waits with `Device::poll`, which does not work on
//!   `wasm32-unknown-unknown`.
//! - Async ([`read_rgba8_async`] and friends) awaits the map callback. On
//!   native, something must poll the device while it waits or the future
//!   never completes. On wasm nothing is required.
//!
//! A consumer that shares the host's device can sample the `bake_*`
//! texture directly instead.

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

use crate::baker::ScalarFormat;
use crate::device::DeviceCtx;

/// Row stride alignment wgpu requires of a texture-to-buffer copy.
const COPY_ALIGN: u32 = 256;

/// An 8-bit RGBA image read back from the GPU, tightly packed (no row
/// padding) and in row-major order from the top-left.
pub struct Image {
    pub width: u32,
    pub height: u32,
    /// `width * height * 4` bytes, RGBA.
    pub pixels: Vec<u8>,
}

impl Image {
    /// The pixel at `(x, y)` as `[r, g, b, a]`, or `None` if out of bounds.
    pub fn pixel(&self, x: u32, y: u32) -> Option<[u8; 4]> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let i = ((y * self.width + x) * 4) as usize;
        Some([
            self.pixels[i],
            self.pixels[i + 1],
            self.pixels[i + 2],
            self.pixels[i + 3],
        ])
    }

    /// Binary PPM (`P6`) — no dependency to write. PPM has no alpha channel,
    /// so alpha is dropped.
    pub fn to_ppm(&self) -> Vec<u8> {
        let mut out = format!("P6\n{} {}\n255\n", self.width, self.height).into_bytes();
        out.reserve(self.pixels.len() / 4 * 3);
        for px in self.pixels.as_chunks::<4>().0 {
            out.extend_from_slice(&px[..3]);
        }
        out
    }
}

/// A single-channel field read back from the GPU: raw texels in the
/// format it was baked in, tightly packed (no row padding), row-major from
/// the top-left and slice-major through `depth`. [`ScalarImage::value`]
/// decodes one texel.
pub struct ScalarImage {
    pub width: u32,
    pub height: u32,
    /// 1 for a flat bake.
    pub depth: u32,
    pub format: ScalarFormat,
    /// `width * height * depth * format.bytes_per_texel()` bytes.
    pub bytes: Vec<u8>,
}

impl ScalarImage {
    /// The texel at `(x, y, z)` as an `f32`, or `None` if out of bounds.
    ///
    /// `R8Unorm` decodes as `byte / 255`, matching how a sampler reads it.
    pub fn value(&self, x: u32, y: u32, z: u32) -> Option<f32> {
        if x >= self.width || y >= self.height || z >= self.depth {
            return None;
        }
        let stride = self.format.bytes_per_texel() as usize;
        let i = (((z * self.height + y) * self.width + x) as usize) * stride;
        Some(match self.format {
            ScalarFormat::R8Unorm => self.bytes[i] as f32 / 255.0,
            ScalarFormat::R16Float => {
                f16_to_f32(u16::from_le_bytes([self.bytes[i], self.bytes[i + 1]]))
            }
            ScalarFormat::R32Float => f32::from_le_bytes([
                self.bytes[i],
                self.bytes[i + 1],
                self.bytes[i + 2],
                self.bytes[i + 3],
            ]),
        })
    }
}

/// IEEE half to single. Hand-written to avoid a dependency for one use.
fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;
    let bits = match exp {
        // Zero or subnormal.
        0 => {
            if mant == 0 {
                sign
            } else {
                let v = mant as f32 * (1.0 / 16_777_216.0); // 2^-24
                return f32::from_bits(sign | v.to_bits());
            }
        }
        // Inf or NaN.
        0x1f => sign | 0x7f80_0000 | (mant << 13),
        _ => sign | ((exp + 112) << 23) | (mant << 13),
    };
    f32::from_bits(bits)
}

/// Returns the buffer and the padded row stride the caller has to strip.
fn copy_to_buffer(
    ctx: &DeviceCtx,
    tex: &wgpu::Texture,
    size: (u32, u32, u32),
    bytes_per_texel: u32,
) -> (wgpu::Buffer, u32) {
    let (width, height, depth) = size;
    let packed_bpr = width * bytes_per_texel;
    let padded_bpr = packed_bpr.div_ceil(COPY_ALIGN) * COPY_ALIGN;

    let readback = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tg-readback"),
        size: (padded_bpr * height * depth) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("tg-readback") });
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bpr),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d { width, height, depth_or_array_layers: depth },
    );
    ctx.queue.submit([enc.finish()]);
    (readback, padded_bpr)
}

fn unpad(buffer: &wgpu::Buffer, padded_bpr: u32, packed_bpr: u32, rows: u32) -> Vec<u8> {
    let data = buffer.slice(..).get_mapped_range();
    let mut out = Vec::with_capacity((packed_bpr * rows) as usize);
    for y in 0..rows {
        out.extend_from_slice(&data[(y * padded_bpr) as usize..][..packed_bpr as usize]);
    }
    drop(data);
    buffer.unmap();
    out
}

/// Not usable on wasm.
fn map_blocking(ctx: &DeviceCtx, buffer: &wgpu::Buffer) {
    let (tx, rx) = std::sync::mpsc::channel();
    buffer.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    ctx.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("device poll");
    rx.recv().expect("map channel").expect("map readback buffer");
}

#[derive(Default)]
struct MapSlot {
    result: Mutex<Option<Result<(), wgpu::BufferAsyncError>>>,
    waker: Mutex<Option<Waker>>,
}

/// Resolves from the map callback, so the crate needs no async runtime.
fn map_async_await(buffer: &wgpu::Buffer) -> impl Future<Output = ()> + use<> {
    let slot = Arc::new(MapSlot::default());
    let cb_slot = slot.clone();
    buffer.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        *cb_slot.result.lock().unwrap() = Some(r);
        if let Some(w) = cb_slot.waker.lock().unwrap().take() {
            w.wake();
        }
    });
    MapFuture { slot }
}

struct MapFuture {
    slot: Arc<MapSlot>,
}

impl Future for MapFuture {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        if let Some(r) = self.slot.result.lock().unwrap().take() {
            r.expect("map readback buffer");
            return Poll::Ready(());
        }
        // Re-check after storing the waker, so a callback that ran in
        // between is not missed.
        *self.slot.waker.lock().unwrap() = Some(cx.waker().clone());
        if let Some(r) = self.slot.result.lock().unwrap().take() {
            r.expect("map readback buffer");
            return Poll::Ready(());
        }
        Poll::Pending
    }
}

/// Copy an `Rgba8Unorm`-family texture back to the CPU.
///
/// A `size` other than the texture's own reads the wrong rectangle rather
/// than failing.
pub fn read_rgba8(ctx: &DeviceCtx, tex: &wgpu::Texture, size: (u32, u32)) -> Image {
    let (width, height) = size;
    let (buffer, padded_bpr) = copy_to_buffer(ctx, tex, (width, height, 1), 4);
    map_blocking(ctx, &buffer);
    let pixels = unpad(&buffer, padded_bpr, width * 4, height);
    Image { width, height, pixels }
}

/// [`read_rgba8`] without the device poll; see the module docs.
pub async fn read_rgba8_async(ctx: &DeviceCtx, tex: &wgpu::Texture, size: (u32, u32)) -> Image {
    let (width, height) = size;
    let (buffer, padded_bpr) = copy_to_buffer(ctx, tex, (width, height, 1), 4);
    map_async_await(&buffer).await;
    let pixels = unpad(&buffer, padded_bpr, width * 4, height);
    Image { width, height, pixels }
}

/// Copy a single-channel texture back to the CPU.
pub fn read_scalar(
    ctx: &DeviceCtx,
    tex: &wgpu::Texture,
    size: (u32, u32),
    format: ScalarFormat,
) -> ScalarImage {
    read_scalar_volume(ctx, tex, (size.0, size.1, 1), format)
}

/// [`read_scalar`] without the device poll; see the module docs.
pub async fn read_scalar_async(
    ctx: &DeviceCtx,
    tex: &wgpu::Texture,
    size: (u32, u32),
    format: ScalarFormat,
) -> ScalarImage {
    read_scalar_volume_async(ctx, tex, (size.0, size.1, 1), format).await
}

/// Copy a single-channel 3D texture back to the CPU, slice-major.
pub fn read_scalar_volume(
    ctx: &DeviceCtx,
    tex: &wgpu::Texture,
    size: (u32, u32, u32),
    format: ScalarFormat,
) -> ScalarImage {
    let bpt = format.bytes_per_texel();
    let (buffer, padded_bpr) = copy_to_buffer(ctx, tex, size, bpt);
    map_blocking(ctx, &buffer);
    scalar_image(&buffer, padded_bpr, size, format)
}

/// [`read_scalar_volume`] without the device poll; see the module docs.
pub async fn read_scalar_volume_async(
    ctx: &DeviceCtx,
    tex: &wgpu::Texture,
    size: (u32, u32, u32),
    format: ScalarFormat,
) -> ScalarImage {
    let bpt = format.bytes_per_texel();
    let (buffer, padded_bpr) = copy_to_buffer(ctx, tex, size, bpt);
    map_async_await(&buffer).await;
    scalar_image(&buffer, padded_bpr, size, format)
}

fn scalar_image(
    buffer: &wgpu::Buffer,
    padded_bpr: u32,
    size: (u32, u32, u32),
    format: ScalarFormat,
) -> ScalarImage {
    let (width, height, depth) = size;
    let bytes = unpad(buffer, padded_bpr, width * format.bytes_per_texel(), height * depth);
    ScalarImage { width, height, depth, format, bytes }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(width: u32, height: u32, fill: impl Fn(u32, u32) -> [u8; 4]) -> Image {
        let mut pixels = Vec::new();
        for y in 0..height {
            for x in 0..width {
                pixels.extend_from_slice(&fill(x, y));
            }
        }
        Image { width, height, pixels }
    }

    #[test]
    fn ppm_has_a_p6_header_and_three_bytes_per_pixel() {
        let img = image(3, 2, |x, y| [x as u8, y as u8, 7, 255]);
        let ppm = img.to_ppm();
        let header = b"P6\n3 2\n255\n";
        assert_eq!(&ppm[..header.len()], header);
        let body = &ppm[header.len()..];
        assert_eq!(body.len(), 3 * 3 * 2, "one RGB triple per pixel");
        assert_eq!(&body[body.len() - 3..], &[2, 1, 7]);
    }

    /// Alpha must be dropped, not composited into the color.
    #[test]
    fn ppm_drops_alpha_without_touching_color() {
        let img = image(2, 1, |x, _| [200, 100, 50, if x == 0 { 0 } else { 255 }]);
        let body = &img.to_ppm()[b"P6\n2 1\n255\n".len()..];
        assert_eq!(body, &[200, 100, 50, 200, 100, 50]);
    }

    #[test]
    fn pixel_indexes_row_major_and_bounds_checks() {
        let img = image(4, 3, |x, y| [x as u8, y as u8, 0, 255]);
        assert_eq!(img.pixel(0, 0), Some([0, 0, 0, 255]));
        assert_eq!(img.pixel(3, 2), Some([3, 2, 0, 255]));
        assert_eq!(img.pixel(4, 0), None);
        assert_eq!(img.pixel(0, 3), None);
    }
}
