// Solid-texture material variant: the four PBR channels are 3D volumes
// (`Baker::bake_volume`) sampled at each fragment's OBJECT-space position,
// as if the mesh were carved out of the material. No UV seams, no pole
// pinching — and the texture spins with the turntable because obj_pos is
// pre-model-rotation. Appended to scene_common.wgsl.
//
// `camera.ambient.w` maps object space into [0,1]³ texture space:
// tex = obj_pos * ambient.w + 0.5 (sphere radius 1 → 0.5; cube side 1 → 1).

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
