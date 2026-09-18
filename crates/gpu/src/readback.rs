//! Getting baked pixels back off the GPU.
//!
//! The editor never needs this — it hands the baked `wgpu::Texture` straight
//! to egui and the pixels stay on the device. A headless consumer does: a
//! bake it cannot read is not a bake it can save. See `examples/bake.rs`.
//!
//! Blocking, because that is what a batch caller wants and the async
//! alternative would be an executor dependency in the public API. The future
//! this waits on is resolved by `Device::poll`, not by any runtime.

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

    /// Encode as binary PPM (`P6`), which every image tool reads and which
    /// costs no dependency to write. Alpha is dropped — PPM has no channel
    /// for it.
    pub fn to_ppm(&self) -> Vec<u8> {
        let mut out = format!("P6\n{} {}\n255\n", self.width, self.height).into_bytes();
        out.reserve(self.pixels.len() / 4 * 3);
        for px in self.pixels.as_chunks::<4>().0 {
            out.extend_from_slice(&px[..3]);
        }
        out
    }
}

/// Copy an `Rgba8Unorm`-family texture back to the CPU.
///
/// `size` is the texture's dimensions; passing anything else reads the wrong
/// rectangle rather than failing, because a `wgpu::Texture` carries its own
/// size and this asks for exactly what it is told to.
pub fn read_rgba8(ctx: &DeviceCtx, tex: &wgpu::Texture, size: (u32, u32)) -> Image {
    let (width, height) = size;
    // wgpu requires the copy's row stride to be 256-byte aligned, so the
    // buffer is padded and the padding is dropped on the way out.
    let packed_bpr = width * 4;
    let padded_bpr = packed_bpr.div_ceil(COPY_ALIGN) * COPY_ALIGN;

    let readback = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tg-readback"),
        size: (padded_bpr * height) as u64,
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
        wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
    );
    ctx.queue.submit([enc.finish()]);

    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    ctx.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("device poll");
    rx.recv().expect("map channel").expect("map readback buffer");

    let data = slice.get_mapped_range();
    let mut pixels = Vec::with_capacity((packed_bpr * height) as usize);
    for y in 0..height {
        let row = &data[(y * padded_bpr) as usize..][..packed_bpr as usize];
        pixels.extend_from_slice(row);
    }
    drop(data);
    readback.unmap();

    Image { width, height, pixels }
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

    /// The header is what tools parse, and the body must be exactly three
    /// bytes a pixel with no alpha and no row padding — the padding the
    /// readback strips is the whole reason this type exists.
    #[test]
    fn ppm_has_a_p6_header_and_three_bytes_per_pixel() {
        let img = image(3, 2, |x, y| [x as u8, y as u8, 7, 255]);
        let ppm = img.to_ppm();
        let header = b"P6\n3 2\n255\n";
        assert_eq!(&ppm[..header.len()], header);
        let body = &ppm[header.len()..];
        assert_eq!(body.len(), 3 * 3 * 2, "one RGB triple per pixel");
        // Row-major from the top-left: pixel (2, 1) is the last triple.
        assert_eq!(&body[body.len() - 3..], &[2, 1, 7]);
    }

    /// Alpha is dropped rather than composited — a bake with transparency
    /// must not come back with its colours silently multiplied.
    #[test]
    fn ppm_drops_alpha_without_touching_colour() {
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
