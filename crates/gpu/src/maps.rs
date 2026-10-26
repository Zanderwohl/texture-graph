//! A bake turned into the three maps a PBR renderer samples, each with a full
//! mip chain: colour, glTF-packed metallic-roughness, and a tangent-space
//! normal map.
//!
//! [`Baker::bake_output`](crate::Baker::bake_output) is shaped for display:
//! four separate images, roughness and metallic as display grays, one level
//! each. A game wants what glTF and Bevy's `StandardMaterial` read, and wants
//! mipmaps, or a tiled surface shimmers wherever a texel is smaller than a
//! pixel.

use crate::{BakeOutput, DeviceCtx, SRGB_VIEW};

/// All three are `Rgba8Unorm`, square, `mip_level_count` levels deep.
pub struct MaterialMaps {
    /// sRGB-encoded; sample it through [`SRGB_VIEW`], or decode by hand.
    pub color: wgpu::Texture,
    /// Linear. R occlusion (always 1), G roughness, B metallic: glTF's
    /// `metallicRoughnessTexture`.
    pub metallic_roughness: wgpu::Texture,
    /// Linear, `n * 0.5 + 0.5`, glTF's convention: +X right, +Y up the
    /// image, +Z out of the surface.
    pub normal: wgpu::Texture,
    pub size: u32,
    pub mip_level_count: u32,
}

/// Every level down to 1x1.
pub fn mip_level_count(size: u32) -> u32 {
    u32::BITS - size.max(1).leading_zeros()
}

pub struct MaterialMapper {
    pack: wgpu::RenderPipeline,
    mip: wgpu::RenderPipeline,
    mip_srgb: wgpu::RenderPipeline,
    mip_normal: wgpu::RenderPipeline,
    sampler: wgpu::Sampler,
}

impl MaterialMapper {
    pub fn new(device: &wgpu::Device) -> Self {
        let pack_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tg-maps-pack"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/maps_pack.wgsl").into()),
        });
        let mip_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tg-maps-mip"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/maps_mip.wgsl").into()),
        });
        let pipeline = |label, module, entry, format| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: None,
                vertex: wgpu::VertexState {
                    module,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let linear = wgpu::TextureFormat::Rgba8Unorm;
        Self {
            pack: pipeline("tg-maps-pack", &pack_shader, "fs", linear),
            mip: pipeline("tg-maps-mip", &mip_shader, "fs_average", linear),
            mip_srgb: pipeline("tg-maps-mip-srgb", &mip_shader, "fs_average", SRGB_VIEW[0]),
            mip_normal: pipeline("tg-maps-mip-normal", &mip_shader, "fs_normal", linear),
            sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("tg-maps-mip"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            }),
        }
    }

    /// `out` must be square, as a tiling material is. Submits and returns;
    /// nothing waits on the GPU.
    pub fn maps(&self, ctx: &DeviceCtx, out: &BakeOutput) -> MaterialMaps {
        let size = out.size.0;
        debug_assert_eq!(out.size.0, out.size.1, "a material tile is square");
        let levels = mip_level_count(size);
        let device = &ctx.device;
        let make = |label| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 1 },
                mip_level_count: levels,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: SRGB_VIEW,
            })
        };
        let maps = MaterialMaps {
            color: make("tg-maps-color"),
            metallic_roughness: make("tg-maps-metallic-roughness"),
            normal: make("tg-maps-normal"),
            size,
            mip_level_count: levels,
        };

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("tg-maps") });
        let whole = wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 1 };
        for (from, to) in [(&out.color, &maps.color), (&out.normal, &maps.normal)] {
            enc.copy_texture_to_texture(from.as_image_copy(), to.as_image_copy(), whole);
        }

        // Narrowed to sampling: the bake's outputs are storage textures too, and an sRGB format
        // cannot be a storage view.
        let srgb = |tex: &wgpu::Texture| {
            tex.create_view(&wgpu::TextureViewDescriptor {
                format: Some(SRGB_VIEW[0]),
                usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
                ..Default::default()
            })
        };
        let pack_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tg-maps-pack"),
            layout: &self.pack.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&srgb(&out.roughness)) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&srgb(&out.metallic)) },
            ],
        });
        draw(&mut enc, &self.pack, &pack_group, &level_view(&maps.metallic_roughness, 0, None));

        for level in 1..levels {
            for (tex, pipeline, format) in [
                (&maps.color, &self.mip_srgb, Some(SRGB_VIEW[0])),
                (&maps.metallic_roughness, &self.mip, None),
                (&maps.normal, &self.mip_normal, None),
            ] {
                let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("tg-maps-mip"),
                    layout: &pipeline.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&level_view(tex, level - 1, format)),
                        },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                    ],
                });
                draw(&mut enc, pipeline, &group, &level_view(tex, level, format));
            }
        }
        ctx.queue.submit([enc.finish()]);
        log::debug!("material maps size={size} levels={levels}");
        maps
    }
}

fn level_view(tex: &wgpu::Texture, level: u32, format: Option<wgpu::TextureFormat>) -> wgpu::TextureView {
    tex.create_view(&wgpu::TextureViewDescriptor {
        format,
        base_mip_level: level,
        mip_level_count: Some(1),
        ..Default::default()
    })
}

fn draw(enc: &mut wgpu::CommandEncoder, pipeline: &wgpu::RenderPipeline, group: &wgpu::BindGroup, target: &wgpu::TextureView) {
    let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("tg-maps"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: target,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, group, &[]);
    pass.draw(0..3, 0..1);
}
