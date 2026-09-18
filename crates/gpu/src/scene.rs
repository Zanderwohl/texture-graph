//! 3D preview renderer. Takes a `BakeOutput` (color / roughness / metallic
//! / normal textures) and draws a lit mesh — sphere, cube, or quad — into a small
//! `Rgba8Unorm` texture ready for the UI to hand to egui-wgpu.
//!
//! One directional light, Cook-Torrance BRDF, tangent-space normal mapping,
//! Reinhard tonemap, gamma-correct sRGB output.

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
    /// A 1×1 quad lying flat on the ground (XZ plane, normal +Y) — a lit
    /// 3D view of the flat material, viewed from a raised angle.
    Quad,
}

/// What the mesh is textured with.
///
/// - `Uv`: flat 2D channels wrapped by the mesh's UVs (the classic path).
/// - `Solid`: 3D volume channels sampled at each fragment's object-space
///   position — the mesh looks carved out of the material; no UV seams,
///   no pole pinching. Produced by [`crate::Baker::bake_volume`].
pub enum SceneMaterial<'a> {
    Uv(&'a BakeOutput),
    Solid(&'a VolumeOutput),
}

/// Orbit camera + directional light. The renderer owns nothing here; the
/// caller drives it (e.g. an auto-spin timer in the UI panel).
#[derive(Copy, Clone, Debug)]
pub struct SceneCamera {
    /// Model orientation: the camera and light rig stay fixed while this
    /// spins/orbits the MODEL. Auto-spin advances it around world Y; user
    /// drag-orbit composes arbitrary trackball rotations onto it.
    pub orientation: Quat,
    /// Rotation in radians around the model's local X axis (positive =
    /// tip toward camera). Fixed for now; ready if we ever wire drag.
    pub pitch: f32,
    /// Distance from the model's origin to the camera.
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
    /// Three-point white studio rig. Each vec4 packs xyz = normalized
    /// direction FROM surface TO light, w = intensity. All lights are
    /// pure white in the shader.
    lights: [[f32; 4]; 3],
    /// xyz = linear ambient tint. w = object-space → texture-space scale
    /// used by the solid-material variant (`tex = obj_pos * w + 0.5`).
    ambient: [f32; 4],
}

/// 48-byte packed vertex. Field offsets MUST match the attribute offsets
/// `vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x4,
/// 3 => Float32x2]` computes (0, 12, 24, 40). Any padding between fields
/// desyncs Rust's Rgba32Float layout from wgpu's cumulative-offset layout
/// — the GPU reads normal / tangent / uv from the wrong bytes and the
/// mesh renders with garbage shading.
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
    /// UV-mapped material path (2D channel textures).
    pipeline_uv: wgpu::RenderPipeline,
    bgl_uv:      wgpu::BindGroupLayout,
    /// Solid material path (3D volume channels sampled by object pos).
    pipeline_solid: wgpu::RenderPipeline,
    bgl_solid:      wgpu::BindGroupLayout,
    /// Fullscreen transparency-checker background, drawn before the mesh.
    pipeline_bg: wgpu::RenderPipeline,
    sampler:  wgpu::Sampler,
    sphere:   Mesh,
    cube:     Mesh,
    quad:     Mesh,
}

impl SceneRenderer {
    pub fn new(device: &wgpu::Device) -> Self {
        // The two shader variants share scene_common.wgsl; each appends
        // its own material bindings + fs_main (2D vs 3D channel textures).
        let common = include_str!("shaders/scene_common.wgsl");
        let src_uv = format!("{common}{}", include_str!("shaders/scene_uv.wgsl"));
        let src_solid = format!("{common}{}", include_str!("shaders/scene_solid.wgsl"));

        let (pipeline_uv, bgl_uv) = make_scene_pipeline(
            device,
            &src_uv,
            wgpu::TextureViewDimension::D2,
            "scene-uv",
        );
        let (pipeline_solid, bgl_solid) = make_scene_pipeline(
            device,
            &src_solid,
            wgpu::TextureViewDimension::D3,
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

        let sphere = build_mesh(device, &sphere_verts_indices(48, 24), "sphere");
        let cube   = build_mesh(device, &cube_verts_indices(),        "cube");
        let quad   = build_mesh(device, &quad_verts_indices(),        "quad");

        Self {
            pipeline_uv,
            bgl_uv,
            pipeline_solid,
            bgl_solid,
            pipeline_bg,
            sampler,
            sphere,
            cube,
            quad,
        }
    }

    /// Allocate a scene color target (Rgba8Unorm, RENDER_ATTACHMENT +
    /// TEXTURE_BINDING). Caller keeps this alive and re-registers with
    /// egui-wgpu whenever it recreates.
    pub fn make_color_target(&self, device: &wgpu::Device, size: (u32, u32)) -> wgpu::Texture {
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

    /// Allocate a matching depth target. Not registered with egui — private
    /// to the scene pass.
    pub fn make_depth_target(&self, device: &wgpu::Device, size: (u32, u32)) -> wgpu::Texture {
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

    /// Render `material` onto `shape` into an existing color+depth pair.
    /// Caller owns both attachments so they can survive across frames
    /// (essential for smooth auto-spin — otherwise every frame re-registers
    /// a texture with egui-wgpu).
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
        let mesh = match shape {
            SceneShape::Sphere => &self.sphere,
            SceneShape::Cube   => &self.cube,
            SceneShape::Quad   => &self.quad,
        };
        // Object-space → [0,1]³ texture-space scale for solid sampling.
        // Sphere spans [-1,1] (radius 1); cube and quad span [-0.5,0.5].
        let obj_scale = match shape {
            SceneShape::Sphere => 0.5f32,
            SceneShape::Cube | SceneShape::Quad => 1.0f32,
        };

        let aspect  = size.0 as f32 / size.1 as f32;
        let proj    = Mat4::perspective_rh(camera.fov_y, aspect, 0.1, 20.0);
        // Turntable/orbit: the CAMERA stays fixed on the +Z side and
        // `orientation` spins the MODEL. The three-point rig is world-fixed
        // around the camera, so the surface facing the viewer is always the
        // lit one. (Orbiting the camera instead sends the viewer around to
        // the rig's shadow side for half of every revolution — the model
        // goes near-black.)
        let cam_pos = Vec3::new(
            0.0,
            camera.distance * camera.pitch.sin(),
            camera.distance * camera.pitch.cos(),
        );
        let view = Mat4::look_at_rh(cam_pos, Vec3::ZERO, Vec3::Y);
        let view_proj = proj * view;
        let uniforms = SceneUniforms {
            view_proj:   view_proj.to_cols_array_2d(),
            model:       Mat4::from_quat(camera.orientation).to_cols_array_2d(),
            camera_pos:  [cam_pos.x, cam_pos.y, cam_pos.z, 0.0],
            lights:      three_point_rig(),
            ambient:     [0.03, 0.03, 0.03, obj_scale],
        };
        let ubo = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("scene-uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let (pipeline, bgl, color_v, rough_v, metal_v, normal_v) = match material {
            SceneMaterial::Uv(m) => (
                &self.pipeline_uv,
                &self.bgl_uv,
                m.color.create_view(&wgpu::TextureViewDescriptor::default()),
                m.roughness.create_view(&wgpu::TextureViewDescriptor::default()),
                m.metallic.create_view(&wgpu::TextureViewDescriptor::default()),
                m.normal.create_view(&wgpu::TextureViewDescriptor::default()),
            ),
            SceneMaterial::Solid(v) => (
                &self.pipeline_solid,
                &self.bgl_solid,
                v.color.create_view(&wgpu::TextureViewDescriptor::default()),
                v.roughness.create_view(&wgpu::TextureViewDescriptor::default()),
                v.metallic.create_view(&wgpu::TextureViewDescriptor::default()),
                v.normal.create_view(&wgpu::TextureViewDescriptor::default()),
            ),
        };
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scene-bg"),
            layout: bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: ubo.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&color_v)  },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&rough_v)  },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&metal_v)  },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&normal_v) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::Sampler(&self.sampler) },
            ],
        });

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
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.08, g: 0.08, b: 0.10, a: 1.0,
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
            // Transparency checker first (ignores depth), then the mesh
            // blends over it.
            pass.set_pipeline(&self.pipeline_bg);
            pass.draw(0..3, 0..1);
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.set_vertex_buffer(0, mesh.vbuf.slice(..));
            pass.set_index_buffer(mesh.ibuf.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..mesh.index_count, 0, 0..1);
        }
        ctx.queue.submit([enc.finish()]);
    }
}

// ---- Lighting -----------------------------------------------------------

/// Classic three-point studio rig, all pure white with staggered
/// intensities — as if the same 5600K bulb were mounted on three C-stands.
///
/// - **Key**   — front-right, elevated ~35°, strongest.
/// - **Fill**  — front-left, gentle, roughly half the key's intensity to
///   fill shadows without erasing form.
/// - **Back**  — behind the subject, above, offset to the key's side; picks
///   out the rim without ever hitting the camera-facing surface directly.
///
/// Each entry: `[dx, dy, dz, intensity]` where `(dx,dy,dz)` is the
/// normalized world-space direction FROM the surface TOWARD the light.
fn three_point_rig() -> [[f32; 4]; 3] {
    let normalize4 = |v: [f32; 3], i: f32| -> [f32; 4] {
        let mag = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-6);
        [v[0] / mag, v[1] / mag, v[2] / mag, i]
    };
    [
        // Key — right, above, in front.
        normalize4([ 1.0, 0.9,  1.2], 3.2),
        // Fill — left, slightly above, in front.
        normalize4([-1.1, 0.3,  0.8], 1.2),
        // Back / rim — right-behind, high.
        normalize4([ 0.4, 1.0, -1.3], 2.2),
    ]
}

// ---- Mesh construction --------------------------------------------------

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

/// Latitude/longitude sphere. Poles collapse to a shared vertex per pole,
/// which pinches the texture there — acceptable for preview. Seam at
/// theta = 0 = 2π shows as a visible line if the texture doesn't tile.
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
            // Tangent along +u = ∂pos/∂theta, normalized.
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
            // CCW winding when viewed from outside. With this
            // parameterization (+u runs +X → +Z), walking `a → b` moves
            // along +theta and `a → c` moves down toward the -Y pole, so
            // the outside-CCW triangles are [a, b, c] and [b, d, c].
            // ([a, c, b] winds the other way — that renders the sphere
            // inside-out: near faces culled, camera sees the far
            // hemisphere. Caught by `scene_test`.)
            indices.extend_from_slice(&[a, b, c, b, d, c]);
        }
    }
    (verts, indices)
}

/// 24-vertex cube: 4 vertices per face so each face gets its own normal /
/// tangent / UV. Winding is CCW as viewed from outside each face.
fn cube_verts_indices() -> (Vec<Vertex>, Vec<u32>) {
    // face: (normal, tangent, quad_corners as (pos, uv))
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
        // CCW-from-outside triangulation of the quad. Each face's four
        // corners are laid out (0=BL, 1=BR, 2=TR, 3=TL) from the outside
        // viewer's POV, so `0 → 1 → 2` and `0 → 2 → 3` walk them
        // counterclockwise. Matches the pipeline's `front_face: Ccw` +
        // back-face culling. This is also the sphere's winding pattern.
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    (verts, indices)
}

/// A 1×1 quad lying flat on the ground: XZ plane at Y = 0, spanning
/// [-0.5, 0.5] on both axes, normal +Y, tangent +X. Corners and winding
/// mirror the cube's +Y (top) face, so it's CCW as viewed from above and
/// its normal-map handedness matches the cube's top.
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

/// Fullscreen transparency-checker background. No bind groups, no vertex
/// buffers; depth is neither tested nor written so the mesh always draws
/// over it.
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

/// Build one scene pipeline variant (UV or solid) from its concatenated
/// WGSL source. The two variants differ only in the material bindings'
/// view dimension (D2 vs D3) and their fs_main sampling code.
fn make_scene_pipeline(
    device: &wgpu::Device,
    shader_src: &str,
    tex_dim: wgpu::TextureViewDimension,
    label: &str,
) -> (wgpu::RenderPipeline, wgpu::BindGroupLayout) {
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(&format!("{label}-bgl")),
        entries: &[
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
        ],
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
                // Straight-alpha blend so an object-alpha bake shows the
                // model itself translucent over the scene background.
                // Opaque materials (alpha = 1) are unaffected.
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
