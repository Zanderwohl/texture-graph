// LayerKind::Coordinate: one axis of the sample point, as a gray. Must match
// the `Coordinate` arm of `core::eval`. Needs `sphere.wgsl` prepended.

struct CoordinateParams {
    size: vec2<u32>,
    axis: u32,            // 0=U, 1=V, 2=W
    face: u32,            // 0 = plane; k + 1 = cube face k
    dom: vec4<f32>,       // bake domain (min_u, min_v, ext_u, ext_v)
    w_coord: f32,
    point_map: array<vec4<f32>, 3>,
}

@group(0) @binding(0) var<uniform> params: CoordinateParams;
@group(0) @binding(1) var out_tex: texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.size.x || gid.y >= params.size.y) { return; }
    let p = map_point(params.point_map, sample_point(params.face, params.dom, gid.xy, params.size, params.w_coord));
    textureStore(out_tex, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(p[params.axis], 0.0, 0.0, 1.0));
}
