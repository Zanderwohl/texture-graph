// Packs roughness and metallic into one texture the glTF way: occlusion in R
// (none: 1), roughness in G, metallic in B. Linear values: the sources are
// bound through their sRGB views, so loading them decodes the display gray.

@group(0) @binding(0) var roughness: texture_2d<f32>;
@group(0) @binding(1) var metallic: texture_2d<f32>;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let x = f32((i << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(i & 2u) * 2.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let at = vec2<i32>(pos.xy);
    let r = textureLoad(roughness, at, 0).x;
    let m = textureLoad(metallic, at, 0).x;
    return vec4<f32>(1.0, r, m, 1.0);
}
