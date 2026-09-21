// LayerKind::Warp. Read `source` at a coordinate displaced by what `by`
// says there. Twin of `eval_warp` in `core::eval`.
//
// The fetch is the same machinery an `EdgeMode::Extend` Transform uses:
// the scheduler has already grown `source`'s bake domain by |amount| on
// each axis (capped at core's EXTEND_LIMIT box), so a displacement within
// the promised [-1, 1] always lands on real data. Anything further is
// outside baked territory and shows the missing-texture grid, exactly as
// an over-reaching extend does.
//
// Known limitation, shared with the flat GPU path in general: `amount.z`
// does nothing here. Each slice only has its inputs baked at that slice's
// w, so no downstream node can re-sample them at another one. The CPU
// evaluator does honour it — see `Warp`'s doc comment.

struct WarpParams {
    size: vec2<u32>,
    mode: u32,            // 0=Scalar, 1=Vector
    _pad0: u32,
    amount: vec4<f32>,    // (u, v, w, _)
    dom: vec4<f32>,       // own bake domain (min_u, min_v, ext_u, ext_v)
    dom_src: vec4<f32>,   // source's bake domain, already grown by |amount|
    dom_by: vec4<f32>,    // displacement field's bake domain
}

@group(0) @binding(0) var<uniform> params: WarpParams;
@group(0) @binding(1) var out_tex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var src: texture_2d<f32>;
@group(0) @binding(3) var by_tex: texture_2d<f32>;

fn dom_uv(dom: vec4<f32>, gid: vec2<u32>, size: vec2<u32>) -> vec2<f32> {
    return vec2<f32>(
        dom.x + (f32(gid.x) + 0.5) / f32(size.x) * dom.z,
        dom.y + (f32(gid.y) + 0.5) / f32(size.y) * dom.w,
    );
}

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

// Twin of `warp_displacement`. Oklcha is stored (l, chroma, hue_deg, a),
// so `Scalar` is the L channel and `Vector` reads the first three with
// hue put over a full turn.
fn displacement(by: vec4<f32>) -> vec3<f32> {
    if (params.mode == 1u) {
        return vec3<f32>(by.x, by.y, by.z / 360.0);
    }
    return vec3<f32>(by.x, by.x, by.x);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.size.x || gid.y >= params.size.y) { return; }
    let coord = vec2<i32>(i32(gid.x), i32(gid.y));
    let uv = dom_uv(params.dom, gid.xy, params.size);

    let by = textureLoad(by_tex, dom_texel(params.dom_by, uv, params.size), 0);
    let d = displacement(by);
    let moved = vec2<f32>(uv.x + d.x * params.amount.x, uv.y + d.y * params.amount.y);

    var out: vec4<f32>;
    if (moved.x < params.dom_src.x || moved.x > params.dom_src.x + params.dom_src.z ||
        moved.y < params.dom_src.y || moved.y > params.dom_src.y + params.dom_src.w) {
        out = missing_color(vec3<f32>(moved, 0.0));
    } else {
        out = textureLoad(src, dom_texel(params.dom_src, moved, params.size), 0);
    }
    textureStore(out_tex, coord, out);
}
