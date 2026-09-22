// 3D preview scene, shared part. Rust prepends it to one variant file,
// `scene_uv.wgsl` or `scene_solid.wgsl`, which supplies bindings 1..4 and
// fs_main.
//
// Roughness/metallic arrive sRGB-encoded, as pack_srgb8 writes scalars, and
// are decoded before use. The target is `Rgba8Unorm`, so the fragment shader
// applies sRGB gamma itself, matching `pack_srgb8`.

struct Camera {
    view_proj: mat4x4<f32>,
    model: mat4x4<f32>,          // turntable rotation
    camera_pos: vec4<f32>,
    // Key, fill, rim. xyz = direction from surface to light, w = linear
    // intensity of a white light.
    lights: array<vec4<f32>, 3>,
    // xyz = linear ambient tint. w = object-to-texture scale for the solid
    // variant (`tex = obj_pos * w + 0.5`).
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
    // Before model rotation, so a solid texture turns with the mesh.
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
    // Valid only while the model has no non-uniform scale.
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

// `base_srgba` is sRGB with straight alpha; `rough_enc`/`metal_enc` are
// sRGB-encoded. `n_encoded` is `n*0.5+0.5` and not sRGB: decoding it tilts a
// flat (128,128,255) sample about 40° off-axis.
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

    // tangent_w flips the bitangent for mirrored UVs.
    let N_geo   = normalize(in.world_normal);
    let T       = normalize(in.world_tangent - N_geo * dot(N_geo, in.world_tangent));
    let B       = cross(N_geo, T) * in.tangent_w;
    let N       = normalize(mat3x3<f32>(T, B, N_geo) * n_tangent);

    let V       = normalize(camera.camera_pos.xyz - in.world_pos);
    let n_dot_v = max(dot(N, V), 1e-4);
    let f0      = mix(vec3<f32>(0.04), base, metallic);

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

        lo = lo + (diffuse + specular) * vec3<f32>(intens) * n_dot_l;
    }

    let ambient = camera.ambient.xyz * base;
    var color   = ambient + lo;

    // Reinhard, so bright metal specular does not clip to white.
    color = color / (color + vec3<f32>(1.0));

    return vec4<f32>(linear_to_srgb3(color), clamp(base_srgba.a, 0.0, 1.0));
}
