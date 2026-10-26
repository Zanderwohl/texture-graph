// One mip level from the level above it. `src` is that one level's view; a
// bilinear sample at a destination pixel's center is the average of the 2x2
// source texels under it. Bound through an sRGB view, colour averages in
// linear light, as it has to.

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let x = f32((i << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(i & 2u) * 2.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

fn average(pos: vec4<f32>) -> vec4<f32> {
    let dst = max(textureDimensions(src) / 2u, vec2<u32>(1u));
    return textureSample(src, samp, pos.xy / vec2<f32>(dst));
}

@fragment
fn fs_average(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    return average(pos);
}

// An average of unit vectors is shorter than one, which reads as flatter;
// renormalized, a distant bumpy wall keeps its slopes.
@fragment
fn fs_normal(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let c = average(pos);
    let n = normalize(c.xyz * 2.0 - vec3<f32>(1.0));
    return vec4<f32>(n * 0.5 + vec3<f32>(0.5), c.w);
}
