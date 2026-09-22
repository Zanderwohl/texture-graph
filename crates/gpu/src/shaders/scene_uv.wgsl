// UV-mapped material: 2D PBR textures sampled at vertex UVs. Appended to
// scene_common.wgsl.

@group(0) @binding(1) var color_tex:  texture_2d<f32>;
@group(0) @binding(2) var rough_tex:  texture_2d<f32>;
@group(0) @binding(3) var metal_tex:  texture_2d<f32>;
@group(0) @binding(4) var normal_tex: texture_2d<f32>;

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    return shade(
        in,
        textureSample(color_tex,  samp, in.uv),
        textureSample(rough_tex,  samp, in.uv).r,
        textureSample(metal_tex,  samp, in.uv).r,
        textureSample(normal_tex, samp, in.uv).rgb,
    );
}
