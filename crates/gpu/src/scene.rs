//! 3D preview renderer: draws a `BakeOutput` onto a lit sphere, cube or quad
//! and into an `Rgba8Unorm` texture for egui-wgpu.

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Quat, Vec3};
use wgpu::util::DeviceExt;

use crate::baker::{BakeOutput, VolumeOutput};
use crate::device::DeviceCtx;

/// Which mesh to display.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum SceneShape {
    Sphere,
    Cube,
    /// A 1×1 quad flat on the XZ plane, seen from a raised angle.
    Quad,
}

/// What the mesh is textured with.
///
/// - `Uv`: 2D channels mapped by the mesh's UVs.
/// - `Solid`: 3D volume channels sampled at object-space position, with no
///   UV seams or pole pinching. With [`VolumeOutput::bump`] set, the
///   surface is bumped by that height's 3D gradient instead of by `normal`.
#[derive(Copy, Clone)]
pub enum SceneMaterial<'a> {
    Uv(&'a BakeOutput),
    Solid(&'a VolumeOutput),
}

#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub enum SceneBackground {
    /// The transparency checker, matching the flat preview's alpha backing.
    #[default]
    Checker,
    /// One color, in sRGB-encoded `[0, 1]`, as the target stores it.
    Solid([f32; 3]),
}

#[derive(Copy, Clone)]
pub struct SceneLayer<'a> {
    pub material: SceneMaterial<'a>,
    /// Uniform. A later layer scaled a little larger wraps the earlier ones,
    /// as a cloud deck does; its alpha shows them through.
    pub scale: f32,
}

#[derive(Copy, Clone, Debug)]
pub struct SceneCamera {
    /// Rotates the model; the camera and lights stay fixed.
    pub orientation: Quat,
    /// Radians around the model's local X axis, positive tipping toward the camera.
    pub pitch: f32,
    pub distance: f32,
    /// Vertical field of view, radians.
    pub fov_y: f32,
}

impl Default for SceneCamera {
    fn default() -> Self {
        Self {
            orientation: Quat::IDENTITY,
            pitch: 0.15,
            distance: 3.2,
            fov_y: std::f32::consts::FRAC_PI_4,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct SceneUniforms {
    view_proj: [[f32; 4]; 4],
    model:     [[f32; 4]; 4],
    camera_pos: [f32; 4],
    /// xyz = surface-to-light direction, w = intensity.
    lights: [[f32; 4]; 3],
    /// xyz = linear ambient tint. w = object-to-texture scale for the solid
    /// variant (`tex = obj_pos * w + 0.5`).
    ambient: [f32; 4],
    /// Solid variant only. x = bump strength, y = 1 when bumping from the
    /// height volume, z = one height texel in texture coordinates.
    bump: [f32; 4],
}

/// Offsets must stay (0, 12, 24, 40), as `vertex_attr_array!` computes them.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Vertex {
    pos:     [f32; 3],   // 0..12
    normal:  [f32; 3],   // 12..24
    tangent: [f32; 4],   // 24..40
    uv:      [f32; 2],   // 40..48
}

struct Mesh {
    vbuf: wgpu::Buffer,
    ibuf: wgpu::Buffer,
    index_count: u32,
}

pub struct SceneRenderer {
    pipeline_uv: wgpu::RenderPipeline,
    bgl_uv:      wgpu::BindGroupLayout,
    pipeline_solid: wgpu::RenderPipeline,
    bgl_solid:      wgpu::BindGroupLayout,
    pipeline_bg: wgpu::RenderPipeline,
    sampler:  wgpu::Sampler,
    /// Bound as the height of a solid material without one.
    flat_height: wgpu::TextureView,
    sphere:   Mesh,
    cube:     Mesh,
    quad:     Mesh,
}

impl SceneRenderer {
    pub fn new(device: &wgpu::Device) -> Self {
        let common = include_str!("shaders/scene_common.wgsl");
        let src_uv = format!("{common}{}", include_str!("shaders/scene_uv.wgsl"));
        let src_solid = format!("{common}{}", include_str!("shaders/scene_solid.wgsl"));

        let (pipeline_uv, bgl_uv) = make_scene_pipeline(
            device,
            &src_uv,
            wgpu::TextureViewDimension::D2,
            false,
            "scene-uv",
        );
        let (pipeline_solid, bgl_solid) = make_scene_pipeline(
            device,
            &src_solid,
            wgpu::TextureViewDimension::D3,
            true,
            "scene-solid",
        );

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("scene-sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let pipeline_bg = make_bg_pipeline(device);
        let flat_height = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("scene-flat-height"),
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D3,
                format: wgpu::TextureFormat::R16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor::default());

        let sphere = build_mesh(device, &sphere_verts_indices(128, 64), "sphere");
        let cube   = build_mesh(device, &cube_verts_indices(),        "cube");
        let quad   = build_mesh(device, &quad_verts_indices(),        "quad");
        log::debug!(
            "scene renderer created pipelines=3 sphere_indices={} cube_indices={} quad_indices={}",
            sphere.index_count,
            cube.index_count,
            quad.index_count,
        );

        Self {
            pipeline_uv,
            bgl_uv,
            pipeline_solid,
            bgl_solid,
            pipeline_bg,
            sampler,
            flat_height,
            sphere,
            cube,
            quad,
        }
    }

    /// The caller keeps it alive and re-registers it with egui-wgpu each time
    /// it is recreated.
    pub fn make_color_target(&self, device: &wgpu::Device, size: (u32, u32)) -> wgpu::Texture {
        log::debug!(
            "scene target create kind=color size={}x{} format={:?}",
            size.0,
            size.1,
            wgpu::TextureFormat::Rgba8Unorm,
        );
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("scene-color"),
            size: wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        })
    }

    pub fn make_depth_target(&self, device: &wgpu::Device, size: (u32, u32)) -> wgpu::Texture {
        log::debug!(
            "scene target create kind=depth size={}x{} format={:?}",
            size.0,
            size.1,
            wgpu::TextureFormat::Depth32Float,
        );
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("scene-depth"),
            size: wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
    }

    /// The caller owns the targets so they last across frames and egui-wgpu
    /// does not re-register a texture every frame.
    #[allow(clippy::too_many_arguments)]
    pub fn render_into(
        &self,
        ctx: &DeviceCtx,
        material: SceneMaterial<'_>,
        shape: SceneShape,
        color_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        size: (u32, u32),
        camera: &SceneCamera,
    ) {
        self.render_layers(
            ctx,
            &[SceneLayer { material, scale: 1.0 }],
            SceneBackground::Checker,
            shape,
            color_view,
            depth_view,
            size,
            camera,
        );
    }

    /// [`SceneRenderer::render_into`] with several meshes of the same shape,
    /// drawn in order.
    #[allow(clippy::too_many_arguments)]
    pub fn render_layers(
        &self,
        ctx: &DeviceCtx,
        layers: &[SceneLayer<'_>],
        background: SceneBackground,
        shape: SceneShape,
        color_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        size: (u32, u32),
        camera: &SceneCamera,
    ) {
        log::trace!(
            "scene render shape={shape:?} layers={} size={}x{} distance={} pitch={}",
            layers.len(),
            size.0,
            size.1,
            camera.distance,
            camera.pitch,
        );
        let mesh = match shape {
            SceneShape::Sphere => &self.sphere,
            SceneShape::Cube   => &self.cube,
            SceneShape::Quad   => &self.quad,
        };
        // Sphere spans [-1,1]; cube and quad span [-0.5,0.5].
        let obj_scale = match shape {
            SceneShape::Sphere => 0.5f32,
            SceneShape::Cube | SceneShape::Quad => 1.0f32,
        };

        let aspect  = size.0 as f32 / size.1 as f32;
        let proj    = Mat4::perspective_rh(camera.fov_y, aspect, 0.1, 20.0);
        // The camera is fixed and `orientation` rotates the model, so the
        // side facing the viewer is always lit. Orbiting the camera puts the
        // viewer on the unlit side for half a turn.
        let cam_pos = Vec3::new(
            0.0,
            camera.distance * camera.pitch.sin(),
            camera.distance * camera.pitch.cos(),
        );
        let view = Mat4::look_at_rh(cam_pos, Vec3::ZERO, Vec3::Y);
        let view_proj = proj * view;

        let bind_groups: Vec<(&wgpu::RenderPipeline, wgpu::BindGroup)> = layers
            .iter()
            .map(|layer| {
                let bump = match layer.material {
                    SceneMaterial::Solid(VolumeOutput { bump: Some(b), size, .. }) => {
                        [b.strength, 1.0, 1.0 / size.0.max(1) as f32, 0.0]
                    }
                    _ => [0.0; 4],
                };
                let model = Mat4::from_quat(camera.orientation)
                    * Mat4::from_scale(Vec3::splat(layer.scale));
                let uniforms = SceneUniforms {
                    view_proj:   view_proj.to_cols_array_2d(),
                    model:       model.to_cols_array_2d(),
                    camera_pos:  [cam_pos.x, cam_pos.y, cam_pos.z, 0.0],
                    lights:      three_point_rig(),
                    ambient:     [0.03, 0.03, 0.03, obj_scale],
                    bump,
                };
                let ubo = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("scene-uniforms"),
                    contents: bytemuck::bytes_of(&uniforms),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
                let view = |t: &wgpu::Texture| t.create_view(&wgpu::TextureViewDescriptor::default());
                let (pipeline, bgl, color_v, rough_v, metal_v, normal_v, height_v) =
                    match layer.material {
                        SceneMaterial::Uv(m) => (
                            &self.pipeline_uv,
                            &self.bgl_uv,
                            view(&m.color),
                            view(&m.roughness),
                            view(&m.metallic),
                            view(&m.normal),
                            None,
                        ),
                        SceneMaterial::Solid(v) => (
                            &self.pipeline_solid,
                            &self.bgl_solid,
                            view(&v.color),
                            view(&v.roughness),
                            view(&v.metallic),
                            view(&v.normal),
                            Some(
                                v.bump
                                    .as_ref()
                                    .map_or_else(|| self.flat_height.clone(), |b| view(&b.height)),
                            ),
                        ),
                    };
                let mut entries = vec![
                    wgpu::BindGroupEntry { binding: 0, resource: ubo.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&color_v)  },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&rough_v)  },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&metal_v)  },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&normal_v) },
                    wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                ];
                if let Some(h) = &height_v {
                    entries.push(wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::TextureView(h),
                    });
                }
                let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("scene-bg"),
                    layout: bgl,
                    entries: &entries,
                });
                (pipeline, bg)
            })
            .collect();

        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("scene-enc"),
        });
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scene-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(match background {
                            SceneBackground::Checker => wgpu::Color { r: 0.08, g: 0.08, b: 0.10, a: 1.0 },
                            SceneBackground::Solid([r, g, b]) => wgpu::Color {
                                r: r as f64, g: g as f64, b: b as f64, a: 1.0,
                            },
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if background == SceneBackground::Checker {
                pass.set_pipeline(&self.pipeline_bg);
                pass.draw(0..3, 0..1);
            }
            pass.set_vertex_buffer(0, mesh.vbuf.slice(..));
            pass.set_index_buffer(mesh.ibuf.slice(..), wgpu::IndexFormat::Uint32);
            for (pipeline, bg) in &bind_groups {
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, bg, &[]);
                pass.draw_indexed(0..mesh.index_count, 0, 0..1);
            }
        }
        ctx.queue.submit([enc.finish()]);
    }
}

/// White key, fill and rim lights. Each entry is `[dx, dy, dz, intensity]`,
/// with the world-space direction from the surface toward the light.
fn three_point_rig() -> [[f32; 4]; 3] {
    let normalize4 = |v: [f32; 3], i: f32| -> [f32; 4] {
        let mag = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-6);
        [v[0] / mag, v[1] / mag, v[2] / mag, i]
    };
    [
        // Key
        normalize4([ 1.0, 0.9,  1.2], 3.2),
        // Fill
        normalize4([-1.1, 0.3,  0.8], 1.2),
        // Rim
        normalize4([ 0.4, 1.0, -1.3], 2.2),
    ]
}

fn build_mesh(
    device: &wgpu::Device,
    src: &(Vec<Vertex>, Vec<u32>),
    label: &str,
) -> Mesh {
    let vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(&format!("{label}-vbuf")),
        contents: bytemuck::cast_slice(&src.0),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(&format!("{label}-ibuf")),
        contents: bytemuck::cast_slice(&src.1),
        usage: wgpu::BufferUsages::INDEX,
    });
    Mesh { vbuf, ibuf, index_count: src.1.len() as u32 }
}

/// Latitude/longitude sphere. The texture pinches at the poles and shows a
/// seam at theta = 0 unless it tiles.
fn sphere_verts_indices(sectors: u32, rings: u32) -> (Vec<Vertex>, Vec<u32>) {
    let mut verts = Vec::with_capacity(((sectors + 1) * (rings + 1)) as usize);
    for r in 0..=rings {
        let v = r as f32 / rings as f32;
        let phi = v * std::f32::consts::PI; // 0..π; 0 = +Y pole
        let (sin_phi, cos_phi) = phi.sin_cos();
        for s in 0..=sectors {
            let u = s as f32 / sectors as f32;
            let theta = u * std::f32::consts::TAU;
            let (sin_theta, cos_theta) = theta.sin_cos();
            let pos = [sin_phi * cos_theta, cos_phi, sin_phi * sin_theta];
            // ∂pos/∂theta, normalized.
            let tangent = [-sin_theta, 0.0, cos_theta];
            verts.push(Vertex {
                pos,
                normal: pos,
                tangent: [tangent[0], tangent[1], tangent[2], 1.0],
                uv: [u, v],
            });
        }
    }
    let stride = sectors + 1;
    let mut indices = Vec::with_capacity((sectors * rings * 6) as usize);
    for r in 0..rings {
        for s in 0..sectors {
            let a = r * stride + s;
            let b = a + 1;
            let c = a + stride;
            let d = c + 1;
            // CCW from outside: `a → b` is +theta (+X toward +Z) and
            // `a → c` is toward -Y. [a, c, b] renders the sphere inside-out.
            indices.extend_from_slice(&[a, b, c, b, d, c]);
        }
    }
    (verts, indices)
}

/// 4 vertices per face so each face has its own normal, tangent and UV.
fn cube_verts_indices() -> (Vec<Vertex>, Vec<u32>) {
    // ((normal, tangent), [(pos, uv); 4])
    let faces: [(([f32; 3], [f32; 3]), [([f32; 3], [f32; 2]); 4]); 6] = [
        // +X
        (
            ([ 1.0,  0.0,  0.0], [0.0, 0.0, -1.0]),
            [
                ([ 0.5, -0.5,  0.5], [0.0, 1.0]),
                ([ 0.5, -0.5, -0.5], [1.0, 1.0]),
                ([ 0.5,  0.5, -0.5], [1.0, 0.0]),
                ([ 0.5,  0.5,  0.5], [0.0, 0.0]),
            ],
        ),
        // -X
        (
            ([-1.0,  0.0,  0.0], [0.0, 0.0,  1.0]),
            [
                ([-0.5, -0.5, -0.5], [0.0, 1.0]),
                ([-0.5, -0.5,  0.5], [1.0, 1.0]),
                ([-0.5,  0.5,  0.5], [1.0, 0.0]),
                ([-0.5,  0.5, -0.5], [0.0, 0.0]),
            ],
        ),
        // +Y
        (
            ([ 0.0,  1.0,  0.0], [1.0, 0.0,  0.0]),
            [
                ([-0.5,  0.5,  0.5], [0.0, 1.0]),
                ([ 0.5,  0.5,  0.5], [1.0, 1.0]),
                ([ 0.5,  0.5, -0.5], [1.0, 0.0]),
                ([-0.5,  0.5, -0.5], [0.0, 0.0]),
            ],
        ),
        // -Y
        (
            ([ 0.0, -1.0,  0.0], [1.0, 0.0,  0.0]),
            [
                ([-0.5, -0.5, -0.5], [0.0, 1.0]),
                ([ 0.5, -0.5, -0.5], [1.0, 1.0]),
                ([ 0.5, -0.5,  0.5], [1.0, 0.0]),
                ([-0.5, -0.5,  0.5], [0.0, 0.0]),
            ],
        ),
        // +Z
        (
            ([ 0.0,  0.0,  1.0], [1.0, 0.0,  0.0]),
            [
                ([-0.5, -0.5,  0.5], [0.0, 1.0]),
                ([ 0.5, -0.5,  0.5], [1.0, 1.0]),
                ([ 0.5,  0.5,  0.5], [1.0, 0.0]),
                ([-0.5,  0.5,  0.5], [0.0, 0.0]),
            ],
        ),
        // -Z
        (
            ([ 0.0,  0.0, -1.0], [-1.0, 0.0, 0.0]),
            [
                ([ 0.5, -0.5, -0.5], [0.0, 1.0]),
                ([-0.5, -0.5, -0.5], [1.0, 1.0]),
                ([-0.5,  0.5, -0.5], [1.0, 0.0]),
                ([ 0.5,  0.5, -0.5], [0.0, 0.0]),
            ],
        ),
    ];
    let mut verts: Vec<Vertex> = Vec::with_capacity(24);
    let mut indices: Vec<u32> = Vec::with_capacity(36);
    for ((normal, tangent), quad) in &faces {
        let base = verts.len() as u32;
        for (pos, uv) in quad {
            verts.push(Vertex {
                pos: *pos,
                normal: *normal,
                tangent: [tangent[0], tangent[1], tangent[2], 1.0],
                uv: *uv,
            });
        }
        // Corners are BL, BR, TR, TL seen from outside, so this is CCW for
        // the pipeline's `front_face: Ccw` with back-face culling.
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    (verts, indices)
}

/// Same corners and winding as the cube's +Y face, at Y = 0, so its
/// normal-map handedness matches the cube's top.
fn quad_verts_indices() -> (Vec<Vertex>, Vec<u32>) {
    let normal = [0.0, 1.0, 0.0];
    let tangent = [1.0, 0.0, 0.0, 1.0];
    let corners: [([f32; 3], [f32; 2]); 4] = [
        ([-0.5, 0.0,  0.5], [0.0, 1.0]),
        ([ 0.5, 0.0,  0.5], [1.0, 1.0]),
        ([ 0.5, 0.0, -0.5], [1.0, 0.0]),
        ([-0.5, 0.0, -0.5], [0.0, 0.0]),
    ];
    let verts = corners
        .iter()
        .map(|(pos, uv)| Vertex { pos: *pos, normal, tangent, uv: *uv })
        .collect();
    (verts, vec![0, 1, 2, 0, 2, 3])
}

fn filterable_texture(
    binding: u32,
    dim: wgpu::TextureViewDimension,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: dim,
            multisampled: false,
        },
        count: None,
    }
}

/// Depth is neither tested nor written, so the mesh always draws over it.
fn make_bg_pipeline(device: &wgpu::Device) -> wgpu::RenderPipeline {
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("scene-bg-pl"),
        bind_group_layouts: &[],
        ..Default::default()
    });
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("scene-bg-shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/scene_bg.wgsl").into()),
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("scene-bg-pipeline"),
        layout: Some(&pl),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8Unorm,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Some(false),
            depth_compare: Some(wgpu::CompareFunction::Always),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

fn make_scene_pipeline(
    device: &wgpu::Device,
    shader_src: &str,
    tex_dim: wgpu::TextureViewDimension,
    with_height: bool,
    label: &str,
) -> (wgpu::RenderPipeline, wgpu::BindGroupLayout) {
    let mut entries = vec![
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        filterable_texture(1, tex_dim),
        filterable_texture(2, tex_dim),
        filterable_texture(3, tex_dim),
        filterable_texture(4, tex_dim),
        wgpu::BindGroupLayoutEntry {
            binding: 5,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        },
    ];
    if with_height {
        entries.push(filterable_texture(6, tex_dim));
    }
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(&format!("{label}-bgl")),
        entries: &entries,
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(&format!("{label}-pl")),
        bind_group_layouts: &[Some(&bgl)],
        ..Default::default()
    });
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(&format!("{label}-shader")),
        source: wgpu::ShaderSource::Wgsl(shader_src.into()),
    });
    let vertex_layout = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Vertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &wgpu::vertex_attr_array![
            0 => Float32x3,  // pos
            1 => Float32x3,  // normal
            2 => Float32x4,  // tangent
            3 => Float32x2,  // uv
        ],
    };
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(&format!("{label}-pipeline")),
        layout: Some(&pl),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[vertex_layout],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8Unorm,
                // Lets an object-alpha bake show the model translucent.
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back),
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Less),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });
    (pipeline, bgl)
}
