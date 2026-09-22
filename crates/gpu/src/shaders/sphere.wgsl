// Sample point for stages that generate coordinates. Prepended to their
// sources; must match `core::sphere`.
//
// `face` is 0 for a plane or a volume slice: the point is the pixel's (u, v)
// and the pass's w. For cube face `k` it is `k + 1`: the point is that face's
// direction on the sphere inscribed in the unit cube.

fn cube_direction(face: u32, u: f32, v: f32) -> vec3<f32> {
    let s = 2.0 * u - 1.0;
    let t = 2.0 * v - 1.0;
    switch face {
        case 0u: { return vec3<f32>(1.0, -t, -s); }
        case 1u: { return vec3<f32>(-1.0, -t, s); }
        case 2u: { return vec3<f32>(s, 1.0, t); }
        case 3u: { return vec3<f32>(s, -1.0, -t); }
        case 4u: { return vec3<f32>(s, -t, 1.0); }
        default: { return vec3<f32>(-s, -t, -1.0); }
    }
}

// Rows of the affine map a sphere bake pushes down from a Transform: the
// point that `p` moves to. The identity everywhere else.
fn map_point(m: array<vec4<f32>, 3>, p: vec3<f32>) -> vec3<f32> {
    let h = vec4<f32>(p, 1.0);
    return vec3<f32>(dot(m[0], h), dot(m[1], h), dot(m[2], h));
}

fn sample_point(face: u32, dom: vec4<f32>, gid: vec2<u32>, size: vec2<u32>, w: f32) -> vec3<f32> {
    let u = dom.x + (f32(gid.x) + 0.5) / f32(size.x) * dom.z;
    let v = dom.y + (f32(gid.y) + 0.5) / f32(size.y) * dom.w;
    if (face == 0u) {
        return vec3<f32>(u, v, w);
    }
    return normalize(cube_direction(face - 1u, u, v)) * 0.5 + vec3<f32>(0.5);
}
