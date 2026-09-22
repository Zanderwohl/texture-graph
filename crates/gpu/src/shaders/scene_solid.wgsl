// Solid-texture material: 3D PBR volumes (`Baker::bake_volume`) sampled at
// the object-space position, which is before model rotation, so the
// texture turns with the mesh. Appended to scene_common.wgsl.
//
// tex = obj_pos * camera.ambient.w + 0.5 maps object space into [0,1]³
// (sphere radius 1 → 0.5; cube side 1 → 1).
//
// With a height volume bound (camera.bump.y = 1) the normal volume is
// ignored: baked a slice at a time, it has no slope along w. The height's 3D
// gradient, projected onto the tangent plane, bumps the surface instead.

@group(0) @binding(1) var color_tex:  texture_3d<f32>;
@group(0) @binding(2) var rough_tex:  texture_3d<f32>;
@group(0) @binding(3) var metal_tex:  texture_3d<f32>;
@group(0) @binding(4) var normal_tex: texture_3d<f32>;
@group(0) @binding(6) var height_tex: texture_3d<f32>;

fn height_at(p: vec3<f32>) -> f32 {
    return textureSampleLevel(height_tex, samp, p, 0.0).r;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let p = in.obj_pos * camera.ambient.w + vec3<f32>(0.5);
    if (camera.bump.y > 0.5) {
        let e = camera.bump.z;
        let ex = vec3<f32>(e, 0.0, 0.0);
        let ey = vec3<f32>(0.0, e, 0.0);
        let ez = vec3<f32>(0.0, 0.0, e);
        // Per unit of texture coordinate, which is the graph's sample space.
        let grad = vec3<f32>(
            height_at(p + ex) - height_at(p - ex),
            height_at(p + ey) - height_at(p - ey),
            height_at(p + ez) - height_at(p - ez),
        ) / (2.0 * e);
        let model3 = mat3x3<f32>(
            camera.model[0].xyz,
            camera.model[1].xyz,
            camera.model[2].xyz,
        );
        // Undo the model's uniform scale so a scaled shell bumps alike.
        let g = model3 * grad / length(model3[0]);
        let N_geo = normalize(in.world_normal);
        let g_surface = g - N_geo * dot(g, N_geo);
        let N = normalize(N_geo - camera.bump.x * g_surface);
        return shade_world(
            in,
            textureSample(color_tex, samp, p),
            textureSample(rough_tex, samp, p).r,
            textureSample(metal_tex, samp, p).r,
            N,
        );
    }
    return shade(
        in,
        textureSample(color_tex,  samp, p),
        textureSample(rough_tex,  samp, p).r,
        textureSample(metal_tex,  samp, p).r,
        textureSample(normal_tex, samp, p).rgb,
    );
}
