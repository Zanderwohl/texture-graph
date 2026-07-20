// 3D preview scene — shared half. Concatenated (in Rust, at pipeline
// creation) with exactly one variant file that supplies the material
// bindings (1..4) and fs_main:
//
// - `scene_uv.wgsl`    — texture_2d channels sampled by mesh UV
// - `scene_solid.wgsl` — texture_3d channels sampled at the fragment's
//                        object-space position ("solid texturing")
//
// Cook-Torrance BRDF with a three-point white light rig. Tangent-space
// normal mapping via a per-vertex TBN. Roughness/metallic store their
// values sRGB-gamma-encoded (that's how pack_srgb8 writes scalars), so we
// decode them to linear before use.
//
// Output target is `Rgba8Unorm` — we apply the sRGB gamma manually in the
// fragment shader for consistency with `pack_srgb8` and to keep egui
// happy sampling it.

struct Camera {
    view_proj: mat4x4<f32>,
    model: mat4x4<f32>,          // turntable rotation (spins with time)
    camera_pos: vec4<f32>,
    // Three-point white studio rig: key + fill + back/rim. Each vec4
    // packs xyz = normalized direction FROM surface TO light, w = linear
    // intensity multiplier (pre-multiplied into pure white).
    lights: array<vec4<f32>, 3>,
    // xyz = linear ambient tint. w = object-space → texture-space scale
    // for solid sampling (`tex = obj_pos * w + 0.5`); unused by the UV
    // variant.
    ambient: vec4<f32>,
}

struct VsIn {
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tangent: vec4<f32>,   // w = bitangent handedness (+/-1)
    @location(3) uv: vec2<f32>,
}

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) world_tangent: vec3<f32>,
    @location(3) tangent_w: f32,
    @location(4) uv: vec2<f32>,
    // Object-space (pre-model-rotation) position: the solid variant's
    // sampling coordinate, so the texture spins WITH the turntable.
    @location(5) obj_pos: vec3<f32>,
}

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(5) var samp: sampler;

const PI: f32 = 3.14159265358979323846;

fn srgb_to_linear_c(x: f32) -> f32 {
    let cx = clamp(x, 0.0, 1.0);
    if (cx <= 0.04045) { return cx / 12.92; }
    return pow((cx + 0.055) / 1.055, 2.4);
}

fn linear_to_srgb_c(x: f32) -> f32 {
    let cx = clamp(x, 0.0, 1.0);
    if (cx <= 0.0031308) { return 12.92 * cx; }
    return 1.055 * pow(cx, 1.0 / 2.4) - 0.055;
}

fn srgb_to_linear3(s: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(srgb_to_linear_c(s.x), srgb_to_linear_c(s.y), srgb_to_linear_c(s.z));
}

fn linear_to_srgb3(l: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(linear_to_srgb_c(l.x), linear_to_srgb_c(l.y), linear_to_srgb_c(l.z));
}

@vertex
fn vs_main(v: VsIn) -> VsOut {
    var out: VsOut;
    let world = camera.model * vec4<f32>(v.pos, 1.0);
    out.world_pos = world.xyz;
    // Model has no non-uniform scale, so a plain rotation of normal/tangent
    // is correct.
    let model3 = mat3x3<f32>(
        camera.model[0].xyz,
        camera.model[1].xyz,
        camera.model[2].xyz,
    );
    out.world_normal  = model3 * v.normal;
    out.world_tangent = model3 * v.tangent.xyz;
    out.tangent_w     = v.tangent.w;
    out.uv            = v.uv;
    out.obj_pos       = v.pos;
    out.clip_pos      = camera.view_proj * world;
    return out;
}

// GGX / Smith terms —
fn d_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a  = roughness * roughness;
    let a2 = a * a;
    let denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / max(PI * denom * denom, 1e-7);
}

fn g_schlick_ggx(n_dot_x: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = r * r / 8.0;
    return n_dot_x / max(n_dot_x * (1.0 - k) + k, 1e-7);
}

fn g_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    return g_schlick_ggx(n_dot_v, roughness) * g_schlick_ggx(n_dot_l, roughness);
}

fn f_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    let x = clamp(1.0 - cos_theta, 0.0, 1.0);
    return f0 + (vec3<f32>(1.0) - f0) * pow(x, 5.0);
}

// Shared shading given the RAW channel samples the variant fetched.
// `base_srgba`: sRGB-encoded base color + straight alpha (alpha < 1 only
// when the bake ran with object-alpha presentation; the pipeline
// alpha-blends the shaded fragment over the scene background);
// `rough_enc`/`metal_enc`: sRGB-encoded scalars; `n_encoded`: raw
// normal-map value (`n*0.5+0.5`, NOT sRGB — do not decode; a "flat"
// (128,128,255) sample would get tilted ~40° off-axis and the whole
// surface goes dark or oddly conical).
fn shade(
    in: VsOut,
    base_srgba: vec4<f32>,
    rough_enc: f32,
    metal_enc: f32,
    n_encoded: vec3<f32>,
) -> vec4<f32> {
    let base      = srgb_to_linear3(base_srgba.rgb);
    let roughness = srgb_to_linear_c(rough_enc);
    let metallic  = srgb_to_linear_c(metal_enc);
    let n_tangent = normalize(n_encoded * 2.0 - vec3<f32>(1.0));

    // Build TBN. Handedness lets us flip the bitangent for mirrored UVs.
    let N_geo   = normalize(in.world_normal);
    let T       = normalize(in.world_tangent - N_geo * dot(N_geo, in.world_tangent));
    let B       = cross(N_geo, T) * in.tangent_w;
    let N       = normalize(mat3x3<f32>(T, B, N_geo) * n_tangent);

    let V       = normalize(camera.camera_pos.xyz - in.world_pos);
    let n_dot_v = max(dot(N, V), 1e-4);
    let f0      = mix(vec3<f32>(0.04), base, metallic);

    // Accumulate direct contributions from the three-point rig.
    var lo = vec3<f32>(0.0);
    for (var i: u32 = 0u; i < 3u; i = i + 1u) {
        let light  = camera.lights[i];
        let L      = normalize(light.xyz);
        let intens = light.w;
        let H      = normalize(V + L);
        let n_dot_l = max(dot(N, L), 0.0);
        let n_dot_h = max(dot(N, H), 0.0);
        let v_dot_h = max(dot(V, H), 0.0);

        let f         = f_schlick(v_dot_h, f0);
        let d         = d_ggx(n_dot_h, roughness);
        let g         = g_smith(n_dot_v, n_dot_l, roughness);
        let specular  = (d * g) * f / max(4.0 * n_dot_v * n_dot_l, 1e-7);
        let k_diffuse = (vec3<f32>(1.0) - f) * (1.0 - metallic);
        let diffuse   = k_diffuse * base / PI;

        // Pure white light times its intensity.
        lo = lo + (diffuse + specular) * vec3<f32>(intens) * n_dot_l;
    }

    let ambient = camera.ambient.xyz * base;
    var color   = ambient + lo;

    // Reinhard tone map — keeps hot metal specular from clipping into a
    // solid white blob.
    color = color / (color + vec3<f32>(1.0));

    return vec4<f32>(linear_to_srgb3(color), clamp(base_srgba.a, 0.0, 1.0));
}
