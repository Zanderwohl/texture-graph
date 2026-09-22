// Solid-texture material: 3D PBR volumes (`Baker::bake_volume`) sampled at
// the object-space position, which is before model rotation, so the
// texture turns with the mesh. Appended to scene_common.wgsl.
//
// tex = obj_pos * camera.ambient.w + 0.5 maps object space into [0,1]³
// (sphere radius 1 → 0.5; cube side 1 → 1).

@group(0) @binding(1) var color_tex:  texture_3d<f32>;
@group(0) @binding(2) var rough_tex:  texture_3d<f32>;
@group(0) @binding(3) var metal_tex:  texture_3d<f32>;
@group(0) @binding(4) var normal_tex: texture_3d<f32>;

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let p = in.obj_pos * camera.ambient.w + vec3<f32>(0.5);
    return shade(
        in,
        textureSample(color_tex,  samp, p),
        textureSample(rough_tex,  samp, p).r,
        textureSample(metal_tex,  samp, p).r,
        textureSample(normal_tex, samp, p).rgb,
    );
}
