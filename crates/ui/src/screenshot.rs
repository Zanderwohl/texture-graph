//! `texture-graph screenshot`: render one preview of a graph to a PNG and
//! exit, with no window.
//!
//! Makes the same choices as the preview panel: a 3D shape shows a graph
//! that varies along w as a solid (volume) texture, and anything else
//! UV-mapped; the quad is seen from the panel's raised angle.

use std::path::PathBuf;

use texture_graph_core::{EvalCtx, Graph, ParamValue, color::oklcha, load_from_path};
use texture_graph_gpu::{
    Baker, DeviceCtx, SceneBackground, SceneCamera, SceneLayer, SceneMaterial, SceneRenderer,
    SceneShape, VolumeOutput, read_rgba8,
};

pub const USAGE: &str = "\
usage: texture-graph screenshot <graph.tgraph> <out.png> [options]

  --shape flat|quad|sphere|cube   preview type (default sphere)
  --size N                        output width and height in pixels (default 768)
  --yaw DEG                       spin about the model's up axis (default 0)
  --pitch DEG                     camera elevation (default 8.6, the panel's)
  --tilt DEG                      lean the model's up axis toward the right (default 0)
  --distance D                    camera distance (default 3.2; quad 2.6)
  --volume N                      solid texture resolution per axis (default 256)
  --seed N                        EvalCtx seed (default the editor's)
  --param NAME=VALUE              bind a parameter: a number, or L,C,H for a color
  --shell GRAPH[@SCALE]           draw GRAPH on a second, larger mesh over the
                                  first, e.g. a cloud deck (scale default 1.015)
  --background checker|RRGGBB     what shows behind the mesh (default checker)
  --channel color|roughness|metallic|normal   flat only (default color)
  --ssaa N                        render N× larger and downsample (default 2)";

#[derive(Copy, Clone, Eq, PartialEq)]
enum Shape {
    Flat,
    Quad,
    Sphere,
    Cube,
}

#[derive(Copy, Clone)]
enum Channel {
    Color,
    Roughness,
    Metallic,
    Normal,
}

struct Args {
    graph: PathBuf,
    out: PathBuf,
    shape: Shape,
    size: u32,
    yaw: f32,
    pitch: Option<f32>,
    tilt: f32,
    distance: Option<f32>,
    volume: u32,
    eval_ctx: EvalCtx,
    shell: Option<(PathBuf, f32)>,
    background: SceneBackground,
    channel: Channel,
    ssaa: u32,
}

/// `args` excludes the program name and the `screenshot` word.
pub fn run(args: impl Iterator<Item = String>) -> Result<(), String> {
    let args = parse(args)?;
    let graph = load(&args.graph)?;
    let shell_graph = match &args.shell {
        Some((path, _)) => Some(load(path)?),
        None => None,
    };

    let ctx = pollster::block_on(DeviceCtx::request_headless())
        .map_err(|e| format!("no usable GPU adapter: {e}"))?;
    let mut baker = Baker::new(ctx.clone());

    let image = if args.shape == Shape::Flat {
        let out = baker
            .bake_output(&graph, (args.size, args.size), &args.eval_ctx, false)
            .map_err(|e| format!("bake: {e}"))?;
        let tex = match args.channel {
            Channel::Color => &out.color,
            Channel::Roughness => &out.roughness,
            Channel::Metallic => &out.metallic,
            Channel::Normal => &out.normal,
        };
        read_rgba8(&ctx, tex, out.size).pixels
    } else {
        render_3d(&ctx, &mut baker, &args, &graph, shell_graph.as_ref())?
    };

    write_png(&args.out, args.size, &image)?;
    println!("{} -> {}", args.graph.display(), args.out.display());
    Ok(())
}

fn render_3d(
    ctx: &DeviceCtx,
    baker: &mut Baker,
    args: &Args,
    graph: &Graph,
    shell: Option<&Graph>,
) -> Result<Vec<u8>, String> {
    let full = args.size * args.ssaa;
    let planet = bake_material(baker, graph, args)?;
    let shell_material = match shell {
        Some(g) => Some(bake_material(baker, g, args)?),
        None => None,
    };

    let mut layers = vec![SceneLayer { material: planet.as_scene(), scale: 1.0 }];
    if let (Some(m), Some((_, scale))) = (&shell_material, &args.shell) {
        layers.push(SceneLayer { material: m.as_scene(), scale: *scale });
    }

    let mut camera = SceneCamera {
        orientation: glam::Quat::from_rotation_z(-args.tilt.to_radians())
            * glam::Quat::from_rotation_y(args.yaw.to_radians()),
        ..SceneCamera::default()
    };
    let shape = match args.shape {
        Shape::Sphere => SceneShape::Sphere,
        Shape::Cube => SceneShape::Cube,
        Shape::Quad => {
            camera.pitch = 0.95;
            camera.distance = 2.6;
            SceneShape::Quad
        }
        Shape::Flat => unreachable!(),
    };
    if let Some(p) = args.pitch {
        camera.pitch = p.to_radians();
    }
    if let Some(d) = args.distance {
        camera.distance = d;
    }

    let scene = SceneRenderer::new(&ctx.device);
    let color = scene.make_color_target(&ctx.device, (full, full));
    let depth = scene.make_depth_target(&ctx.device, (full, full));
    scene.render_layers(
        ctx,
        &layers,
        args.background,
        shape,
        &color.create_view(&Default::default()),
        &depth.create_view(&Default::default()),
        (full, full),
        &camera,
    );
    let mut pixels = read_rgba8(ctx, &color, (full, full)).pixels;
    // Blending a translucent layer writes its alpha into the target too, but
    // what the panel shows is opaque.
    for px in pixels.as_chunks_mut::<4>().0 {
        px[3] = 255;
    }
    Ok(downsample(&pixels, full, args.ssaa))
}

enum Baked {
    Uv(texture_graph_gpu::BakeOutput),
    Solid(VolumeOutput),
}

impl Baked {
    fn as_scene(&self) -> SceneMaterial<'_> {
        match self {
            Baked::Uv(b) => SceneMaterial::Uv(b),
            Baked::Solid(v) => SceneMaterial::Solid(v),
        }
    }
}

fn bake_material(baker: &mut Baker, graph: &Graph, args: &Args) -> Result<Baked, String> {
    if graph.output_is_3d() {
        let v = baker
            .bake_volume(graph, args.volume, args.volume, &args.eval_ctx)
            .map_err(|e| format!("volume bake: {e}"))?;
        Ok(Baked::Solid(v))
    } else {
        let size = (args.size * args.ssaa).min(4096);
        let b = baker
            .bake_output(graph, (size, size), &args.eval_ctx, true)
            .map_err(|e| format!("bake: {e}"))?;
        Ok(Baked::Uv(b))
    }
}

fn load(path: &PathBuf) -> Result<Graph, String> {
    load_from_path(path)
        .map(|f| f.graph)
        .map_err(|e| format!("{}: {e}", path.display()))
}

fn parse(mut it: impl Iterator<Item = String>) -> Result<Args, String> {
    let mut positional = Vec::new();
    let mut args = Args {
        graph: PathBuf::new(),
        out: PathBuf::new(),
        shape: Shape::Sphere,
        size: 768,
        yaw: 0.0,
        pitch: None,
        tilt: 0.0,
        distance: None,
        volume: 256,
        eval_ctx: EvalCtx::default(),
        shell: None,
        background: SceneBackground::Checker,
        channel: Channel::Color,
        ssaa: 2,
    };
    while let Some(a) = it.next() {
        if !a.starts_with("--") {
            positional.push(a);
            continue;
        }
        let mut value = || it.next().ok_or_else(|| format!("{a} needs a value"));
        let num = |v: String| v.parse::<f32>().map_err(|_| format!("{a}: not a number: {v}"));
        let whole = |v: String| v.parse::<u32>().map_err(|_| format!("{a}: not a whole number: {v}"));
        match a.as_str() {
            "--shape" => {
                args.shape = match value()?.as_str() {
                    "flat" => Shape::Flat,
                    "quad" => Shape::Quad,
                    "sphere" => Shape::Sphere,
                    "cube" => Shape::Cube,
                    v => return Err(format!("--shape: unknown shape {v}")),
                }
            }
            "--size" => args.size = whole(value()?)?.max(1),
            "--yaw" => args.yaw = num(value()?)?,
            "--pitch" => args.pitch = Some(num(value()?)?),
            "--tilt" => args.tilt = num(value()?)?,
            "--distance" => args.distance = Some(num(value()?)?),
            "--volume" => args.volume = whole(value()?)?.max(2),
            "--seed" => args.eval_ctx.seed = whole(value()?)?,
            "--ssaa" => args.ssaa = whole(value()?)?.clamp(1, 4),
            "--param" => {
                let v = value()?;
                let (name, raw) =
                    v.split_once('=').ok_or_else(|| format!("--param: expected NAME=VALUE, got {v}"))?;
                args.eval_ctx.params.insert(name.to_string(), parse_param(raw)?);
            }
            "--shell" => {
                let v = value()?;
                args.shell = Some(match v.rsplit_once('@') {
                    Some((path, scale)) => (PathBuf::from(path), num(scale.to_string())?),
                    None => (PathBuf::from(v), 1.015),
                });
            }
            "--background" => {
                let v = value()?;
                args.background = if v == "checker" {
                    SceneBackground::Checker
                } else {
                    SceneBackground::Solid(parse_hex(&v)?)
                };
            }
            "--channel" => {
                args.channel = match value()?.as_str() {
                    "color" => Channel::Color,
                    "roughness" => Channel::Roughness,
                    "metallic" => Channel::Metallic,
                    "normal" => Channel::Normal,
                    v => return Err(format!("--channel: unknown channel {v}")),
                }
            }
            "--help" | "-h" => return Err(USAGE.to_string()),
            _ => return Err(format!("unknown option {a}\n\n{USAGE}")),
        }
    }
    let [graph, out]: [String; 2] = positional
        .try_into()
        .map_err(|_| format!("expected a graph and an output path\n\n{USAGE}"))?;
    args.graph = graph.into();
    args.out = out.into();
    Ok(args)
}

fn parse_param(raw: &str) -> Result<ParamValue, String> {
    let parts: Vec<&str> = raw.split(',').collect();
    let nums: Result<Vec<f32>, _> = parts.iter().map(|p| p.trim().parse::<f32>()).collect();
    match nums.map_err(|_| format!("--param: not a number or L,C,H: {raw}"))?.as_slice() {
        [x] => Ok(ParamValue::Scalar(*x)),
        [l, c, h] => Ok(ParamValue::Color(oklcha(*l, *c, *h, 1.0))),
        _ => Err(format!("--param: expected one number or L,C,H, got {raw}")),
    }
}

fn parse_hex(v: &str) -> Result<[f32; 3], String> {
    let v = v.trim_start_matches('#');
    let byte = |i: usize| {
        v.get(i..i + 2)
            .and_then(|s| u8::from_str_radix(s, 16).ok())
            .map(|b| b as f32 / 255.0)
    };
    match (v.len(), byte(0), byte(2), byte(4)) {
        (6, Some(r), Some(g), Some(b)) => Ok([r, g, b]),
        _ => Err(format!("--background: expected checker or RRGGBB, got {v}")),
    }
}

/// Box filter, averaged in linear light so edges do not darken.
fn downsample(pixels: &[u8], full: u32, factor: u32) -> Vec<u8> {
    if factor == 1 {
        return pixels.to_vec();
    }
    let to_lin = |c: u8| {
        let x = c as f32 / 255.0;
        if x <= 0.04045 { x / 12.92 } else { ((x + 0.055) / 1.055).powf(2.4) }
    };
    let to_srgb = |x: f32| {
        let s = if x <= 0.0031308 { 12.92 * x } else { 1.055 * x.powf(1.0 / 2.4) - 0.055 };
        (s.clamp(0.0, 1.0) * 255.0).round() as u8
    };
    let small = full / factor;
    let n = (factor * factor) as f32;
    let mut out = Vec::with_capacity((small * small * 4) as usize);
    for y in 0..small {
        for x in 0..small {
            let mut acc = [0.0f32; 4];
            for dy in 0..factor {
                for dx in 0..factor {
                    let i = (((y * factor + dy) * full + x * factor + dx) * 4) as usize;
                    for c in 0..3 {
                        acc[c] += to_lin(pixels[i + c]);
                    }
                    acc[3] += pixels[i + 3] as f32;
                }
            }
            out.extend_from_slice(&[
                to_srgb(acc[0] / n),
                to_srgb(acc[1] / n),
                to_srgb(acc[2] / n),
                (acc[3] / n).round() as u8,
            ]);
        }
    }
    out
}

fn write_png(path: &PathBuf, size: u32, rgba: &[u8]) -> Result<(), String> {
    let file = std::fs::File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), size, size);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    enc.set_source_srgb(png::SrgbRenderingIntent::Perceptual);
    enc.write_header()
        .and_then(|mut w| w.write_image_data(rgba))
        .map_err(|e| format!("{}: {e}", path.display()))
}
