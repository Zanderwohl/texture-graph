// 3D scene background: the flat Photoshop-style transparency checker,
// drawn as a fullscreen triangle before the mesh. Screen-space 8-px cells
// with the same grays as pack_srgb8's alpha backing, so a translucent
// object blends over exactly the pattern the flat preview would show
// behind it.

const CELL: f32 = 8.0;
const LIGHT: vec3<f32> = vec3<f32>(0.75, 0.75, 0.75);
const DARK:  vec3<f32> = vec3<f32>(0.55, 0.55, 0.55);

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    // Single triangle covering the viewport: (-1,-1) (3,-1) (-1,3).
    let x = f32(i32(vi & 1u) * 4 - 1);
    let y = f32(i32(vi >> 1u) * 4 - 1);
    return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let cell = vec2<u32>(u32(pos.x / CELL), u32(pos.y / CELL));
    let parity = (cell.x + cell.y) & 1u;
    if (parity == 0u) {
        return vec4<f32>(LIGHT, 1.0);
    }
    return vec4<f32>(DARK, 1.0);
}
