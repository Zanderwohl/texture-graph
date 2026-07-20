// Fills a pool texture with the "missing texture" grid used for
// unconnected (`None`) layer inputs: a magenta/black checkerboard with 16
// cells per unit in u, v, AND w so it stays a solid 3D checker in volume
// bakes. Mirrors `missing_texture` in core::eval — keep the cell count
// and Oklcha constants in sync. Magenta is Oklch of sRGB (1, 0, 1).

struct MissingParams {
    size: vec2<u32>,
    w_coord: f32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: MissingParams;
@group(0) @binding(1) var out_tex: texture_storage_2d<rgba32float, write>;

const CELLS: f32 = 16.0;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.size.x || gid.y >= params.size.y) { return; }
    let u = (f32(gid.x) + 0.5) / f32(params.size.x);
    let v = (f32(gid.y) + 0.5) / f32(params.size.y);
    let cell = vec3<i32>(
        i32(floor(u * CELLS)),
        i32(floor(v * CELLS)),
        i32(floor(params.w_coord * CELLS)),
    );
    var color: vec4<f32>;
    if (((cell.x + cell.y + cell.z) % 2 + 2) % 2 == 0) {
        color = vec4<f32>(0.7017, 0.3223, 328.36, 1.0);
    } else {
        color = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    textureStore(out_tex, vec2<i32>(i32(gid.x), i32(gid.y)), color);
}
