// LayerKind::Transform. Read `source` at re-mapped (u, v) coords.
//
// The pre-transform is the same as `apply_transform` in `core::eval`:
// recenter → rotate in UV → per-axis scale → coord_mode (Passthrough,
// Permute, or Radial).
//
// GPU limitation: the source texture only holds data for [0, 1]² of the
// input domain (the resolution of the current bake). Sampling at
// out-of-range coords clamps to the texture edge — CPU eval would recurse
// into the source with the actual out-of-range sample, giving different
// results for e.g. Noise at coords > 1. Tileable inputs are unaffected by
// the clamp. Documented deviation from CPU.

struct TransformParams {
    size: vec2<u32>,
    coord_mode: u32,     // 0=Passthrough, 1=Permute, 2=Radial
    rotate_uv: f32,
    offset: vec4<f32>,   // (u, v, w, _)
    scale: vec4<f32>,    // (u, v, w, _)
    permute: vec4<u32>,  // (axis_a, axis_b, axis_c, _), each 0=U 1=V 2=W
    radial_dim: u32,     // 0=D2, 1=D3
    radial_into: u32,    // 0=U, 1=V, 2=W
    _pad: vec2<u32>,
}

@group(0) @binding(0) var<uniform> params: TransformParams;
@group(0) @binding(1) var out_tex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var src: texture_2d<f32>;

fn pick_axis(a: u32, uvw: vec3<f32>) -> f32 {
    switch a {
        case 0u: { return uvw.x; }
        case 1u: { return uvw.y; }
        default: { return uvw.z; }
    }
}

fn apply_transform(u_in: f32, v_in: f32, w_in: f32) -> vec3<f32> {
    var u = u_in - params.offset.x;
    var v = v_in - params.offset.y;
    let w = w_in - params.offset.z;
    if (params.rotate_uv != 0.0) {
        let s = sin(params.rotate_uv);
        let c = cos(params.rotate_uv);
        let ru = u * c - v * s;
        let rv = u * s + v * c;
        u = ru;
        v = rv;
    }
    let su = u * params.scale.x;
    let sv = v * params.scale.y;
    let sw = w * params.scale.z;

    switch params.coord_mode {
        case 1u: {
            let uvw = vec3<f32>(su, sv, sw);
            return vec3<f32>(
                pick_axis(params.permute.x, uvw),
                pick_axis(params.permute.y, uvw),
                pick_axis(params.permute.z, uvw),
            );
        }
        case 2u: {
            var r: f32;
            if (params.radial_dim == 0u) {
                r = sqrt(su * su + sv * sv);
            } else {
                r = sqrt(su * su + sv * sv + sw * sw);
            }
            switch params.radial_into {
                case 0u: { return vec3<f32>(r, 0.0, 0.0); }
                case 1u: { return vec3<f32>(0.0, r, 0.0); }
                default: { return vec3<f32>(0.0, 0.0, r); }
            }
        }
        default: {
            return vec3<f32>(su, sv, sw);
        }
    }
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.size.x || gid.y >= params.size.y) { return; }
    let coord = vec2<i32>(i32(gid.x), i32(gid.y));
    let u = (f32(gid.x) + 0.5) / f32(params.size.x);
    let v = (f32(gid.y) + 0.5) / f32(params.size.y);
    let w = 0.5;
    let t = apply_transform(u, v, w);
    // Sample source at t.uv; clamped to [0, size).
    let sx = clamp(t.x * f32(params.size.x), 0.0, f32(params.size.x) - 1.0);
    let sy = clamp(t.y * f32(params.size.y), 0.0, f32(params.size.y) - 1.0);
    let src_px = textureLoad(src, vec2<i32>(i32(sx), i32(sy)), 0);
    textureStore(out_tex, coord, src_px);
}
