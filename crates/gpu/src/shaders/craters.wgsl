// LayerKind::Craters. Twin of `core::crater`; needs `sphere.wgsl` prepended.
//
// When `*_is_layer == 0` the matching texture is a placeholder and is not read.

struct CratersParams {
    size: vec2<u32>,
    face: u32,             // 0 = plane; k + 1 = cube face k
    seed: u32,
    dom: vec4<f32>,        // own bake domain (min_u, min_v, ext_u, ext_v)
    dom_under: vec4<f32>,
    dom_density: vec4<f32>,
    point_map: array<vec4<f32>, 3>,
    w_coord: f32,
    frequency: f32,
    classes: u32,
    gain: f32,
    depth: f32,
    age: f32,
    erase: f32,
    peak: f32,
    rays: f32,
    relief: f32,
    sphere: u32,
    ejecta: u32,
    under_const: f32,
    under_is_layer: u32,
    density_const: f32,
    density_is_layer: u32,
}

@group(0) @binding(0) var<uniform> params: CratersParams;
@group(0) @binding(1) var out_tex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var tex_under: texture_2d<f32>;
@group(0) @binding(3) var tex_density: texture_2d<f32>;

const MAX_CLASSES: u32 = 6u;
const DATUM: f32 = 0.5;
const R_MAX: f32 = 0.2;
const EXTENT: f32 = 2.5;
const RAYS: u32 = 6u;

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

fn pcg(v: u32) -> u32 {
    let s = v * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}

fn unit(h: u32) -> f32 {
    return f32(h >> 8u) * (1.0 / 16777216.0);
}

fn cell_hash(seed: u32, k: u32, c: vec3<i32>) -> u32 {
    var h = pcg(seed ^ (k * 0x9E3779B9u));
    h = pcg(h + bitcast<u32>(c.x));
    h = pcg(h + bitcast<u32>(c.y));
    return pcg(h + bitcast<u32>(c.z));
}

fn ray_lobes(h_in: u32, dir: vec3<f32>) -> f32 {
    var h = h_in;
    var sum = 0.0;
    for (var i = 0u; i < RAYS; i++) {
        h = pcg(h);
        let a = unit(h) * 2.0 - 1.0;
        h = pcg(h);
        let b = unit(h) * 2.0 - 1.0;
        h = pcg(h);
        let c = unit(h) * 2.0 - 1.0;
        let len = max(sqrt(a * a + b * b + c * c), 1.0e-3);
        var l = saturate(dot(dir, vec3<f32>(a / len, b / len, c / len)));
        for (var k = 0; k < 5; k++) {
            l *= l;
        }
        sum += l;
    }
    return saturate(sum);
}

fn craters(p: vec3<f32>, under: f32, density: f32) -> f32 {
    var n = vec3<f32>(0.0, 0.0, 1.0);
    if (params.sphere == 1u) {
        let d = p - vec3<f32>(0.5);
        n = d / max(sqrt(dot(d, d)), 1.0e-6);
    }
    var out = under;
    var frequency = params.frequency;
    var occupancy = density;
    var shrink = 1.0;
    let classes = clamp(params.classes, 1u, MAX_CLASSES);
    for (var k = 0u; k < classes; k++) {
        let q = p * frequency;
        let base = vec3<i32>(floor(q));
        for (var dz = -1; dz <= 1; dz++) {
            for (var dy = -1; dy <= 1; dy++) {
                for (var dx = -1; dx <= 1; dx++) {
                    let c = base + vec3<i32>(dx, dy, dz);
                    var h = cell_hash(params.seed, k, c);
                    let present = saturate((occupancy - unit(h)) * 20.0);
                    if (present <= 0.0) { continue; }
                    h = pcg(h);
                    let cx = f32(c.x) + unit(h);
                    h = pcg(h);
                    let cy = f32(c.y) + unit(h);
                    h = pcg(h);
                    let cz = f32(c.z) + unit(h);
                    let v = q - vec3<f32>(cx, cy, cz);
                    let across = dot(v, n);
                    let slab = saturate((0.5 - abs(across)) * 8.0);
                    if (slab <= 0.0) { continue; }
                    h = pcg(h);
                    let size = 0.5 + 0.5 * unit(h);
                    let r = R_MAX * size;
                    let rho2 = max(dot(v, v) - across * across, 0.0);
                    let x = sqrt(rho2) / r;
                    if (x >= EXTENT) { continue; }
                    let weight = present * slab;
                    let t = (x - 1.0) / (EXTENT - 1.0);
                    let cover = select((1.0 - t) * (1.0 - t), 1.0, x < 1.0);
                    let big = saturate((size * shrink - 0.3) / 0.5);
                    if (params.ejecta == 1u) {
                        let tangent = v - across * n;
                        let dir = tangent / max(sqrt(dot(tangent, tangent)), 1.0e-6);
                        let lobes = ray_lobes(pcg(h), dir);
                        let rim = mix(1.0, 0.35 + 0.65 * lobes, params.rays);
                        let x2 = x * x;
                        let bright = select(rim, mix(0.55, rim, x2 * x2), x < 1.0);
                        out = mix(out, bright * (1.0 - params.age), weight * cover);
                    } else {
                        let d = params.depth * (1.0 - 0.5 * big) * (1.0 - 0.75 * params.age);
                        let rim_h = 0.3 * d * (1.0 - 0.4 * params.age);
                        var z: f32;
                        if (x < 1.0) {
                            let floor_z = rim_h - d * (1.0 - 0.6 * big);
                            let bowl = max(d * (x * x - 1.0) + rim_h, floor_z);
                            let s = saturate(1.0 - (x * x) * 16.0);
                            z = bowl + params.peak * big * d * s * s;
                        } else {
                            let inv = 1.0 / (x * x * x);
                            let tail = 1.0 / (EXTENT * EXTENT * EXTENT);
                            z = rim_h * (inv - tail) / (1.0 - tail);
                        }
                        let erased = mix(out, DATUM, params.erase * weight * cover);
                        out = erased + z * r / frequency * params.relief * weight;
                    }
                }
            }
        }
        frequency *= 2.0;
        occupancy *= params.gain;
        shrink *= 0.5;
    }
    return out;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.size.x || gid.y >= params.size.y) { return; }
    let uv = dom_uv(params.dom, gid.xy, params.size);
    var under = params.under_const;
    if (params.under_is_layer == 1u) {
        under = textureLoad(tex_under, dom_texel(params.dom_under, uv, params.size), 0).x;
    }
    var density = params.density_const;
    if (params.density_is_layer == 1u) {
        density = textureLoad(tex_density, dom_texel(params.dom_density, uv, params.size), 0).x;
    }
    let p = map_point(params.point_map, sample_point(params.face, params.dom, gid.xy, params.size, params.w_coord));
    textureStore(out_tex, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(craters(p, under, density), 0.0, 0.0, 1.0));
}
