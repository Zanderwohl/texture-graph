// LayerKind::Transform. Read `source` at re-mapped (u, v) coords.
//
// The pre-transform is the same as `apply_transform` in `core::eval`:
// recenter → rotate in UV → per-axis scale → coord_mode (Passthrough,
// Permute, or Radial).
//
// Edge modes (mirroring `eval_transform` in `core::eval`):
// - Clamp: transformed U/V pin to the [0, 1] square.
// - Extend: the scheduler bakes the source over the UV rectangle this
//   transform actually samples (capped at core's EXTEND_LIMIT box), so
//   out-of-[0,1] samples land on real data; anything outside the source's
//   baked domain shows the missing-texture grid.

struct TransformParams {
    size: vec2<u32>,
    coord_mode: u32,     // 0=Passthrough, 1=Permute, 2=Radial
    rotate_uv: f32,
    offset: vec4<f32>,   // (u, v, w, _)
    scale: vec4<f32>,    // (u, v, w, _)
    permute: vec4<u32>,  // (axis_a, axis_b, axis_c, _), each 0=U 1=V 2=W
    radial_dim: u32,     // 0=D2, 1=D3
    radial_into: u32,    // 0=U, 1=V, 2=W
    w_coord: f32,        // third texture coordinate; 0.5 for flat bakes
    edge_mode: u32,      // 0=Clamp, 1=Extend
    dom: vec4<f32>,      // own bake domain (min_u, min_v, ext_u, ext_v)
    src_dom: vec4<f32>,  // source's bake domain
}

@group(0) @binding(0) var<uniform> params: TransformParams;
@group(0) @binding(1) var out_tex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var src: texture_2d<f32>;

// Nearest texel of `uv` in the source baked over `dom`, edge-clamped.
fn dom_texel(dom: vec4<f32>, uv: vec2<f32>, size: vec2<u32>) -> vec2<i32> {
    let tx = (uv.x - dom.x) / dom.z * f32(size.x);
    let ty = (uv.y - dom.y) / dom.w * f32(size.y);
    return vec2<i32>(
        i32(clamp(tx, 0.0, f32(size.x) - 1.0)),
        i32(clamp(ty, 0.0, f32(size.y) - 1.0)),
    );
}

// The missing-texture grid at a sample point — mirrors `missing_texture`
// in core::eval and missing.wgsl (16 cells/unit, magenta/black, 3D).
const MISSING_CELLS: f32 = 16.0;

fn missing_color(p: vec3<f32>) -> vec4<f32> {
    let cell = vec3<i32>(
        i32(floor(p.x * MISSING_CELLS)),
        i32(floor(p.y * MISSING_CELLS)),
        i32(floor(p.z * MISSING_CELLS)),
    );
    if (((cell.x + cell.y + cell.z) % 2 + 2) % 2 == 0) {
        return vec4<f32>(0.7017, 0.3223, 328.36, 1.0);
    }
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}

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
    let u = params.dom.x + (f32(gid.x) + 0.5) / f32(params.size.x) * params.dom.z;
    let v = params.dom.y + (f32(gid.y) + 0.5) / f32(params.size.y) * params.dom.w;
    let w = params.w_coord;
    let t = apply_transform(u, v, w);
    var src_px: vec4<f32>;
    if (params.edge_mode == 0u) {
        // Clamp: pin to the unit square, then sample the source there.
        let cuv = vec2<f32>(clamp(t.x, 0.0, 1.0), clamp(t.y, 0.0, 1.0));
        src_px = textureLoad(src, dom_texel(params.src_dom, cuv, params.size), 0);
    } else if (t.x < params.src_dom.x || t.x > params.src_dom.x + params.src_dom.z ||
               t.y < params.src_dom.y || t.y > params.src_dom.y + params.src_dom.w) {
        // Extend, but past what the source's bake covers (the request was
        // capped): the missing grid, at the transformed sample point.
        src_px = missing_color(t);
    } else {
        src_px = textureLoad(src, dom_texel(params.src_dom, t.xy, params.size), 0);
    }
    textureStore(out_tex, coord, src_px);
}
