//! Write the sample planet graphs to `samples/planets/`.
//!
//! ```text
//! cargo run -p texture-graph-core --example planets [out_dir]
//! ```
//!
//! Each graph is a class of planet rather than one planet: every noise is 3D
//! fbm on the sample point, so a sphere shows no seam or pole pinch, and
//! `EvalCtx::seed` picks the member of the class. Parameters shift the class
//! along its obvious axes (sea level, ice extent, and so on).
//!
//! Built here rather than by hand so the graphs stay reviewable as code; the
//! `.tgraph` files are the output.

use std::collections::HashMap;
use std::path::PathBuf;

use texture_graph_core::color::oklcha;
use texture_graph_core::*;

fn main() {
    let dir = PathBuf::from(std::env::args().nth(1).unwrap_or_else(|| "samples/planets".into()));
    std::fs::create_dir_all(&dir).expect("create output directory");
    for (file, name, description, graph) in [
        (
            "earthlike.tgraph",
            "Earthlike",
            "Oceans, continents with ridged ranges, climate-banded biomes and polar ice. \
             Normal map on land only; oceans are flat and glossy.",
            earthlike(),
        ),
        (
            "earthlike-clouds.tgraph",
            "Earthlike clouds",
            "Translucent cloud deck for a shell drawn just above earthlike.tgraph.",
            earthlike_clouds(),
        ),
        (
            "marslike.tgraph",
            "Marslike",
            "Dry rust world: highland/lowland dichotomy, canyon networks, dark albedo \
             provinces, small dusty polar caps.",
            marslike(),
        ),
        (
            "rocky.tgraph",
            "Rocky",
            "Any rocky world with air, from Mars to Earth to a snowball: sea level, ice, \
             life, oxidation and sand are parameters a host derives from what the world is.",
            rocky(),
        ),
        (
            "giant.tgraph",
            "Giant",
            "Any giant, from a hot Jupiter to Uranus: its band count, contrast, turbulence, \
             storms and polar haze are parameters, and its colors are what a host derives from \
             its chemistry.",
            giant(),
        ),
    ] {
        let meta = FileMetadata {
            name: name.into(),
            description: Some(description.into()),
            authors: vec![],
            modified: String::new(),
            written_by: "texture-graph-core examples/planets.rs".into(),
        };
        let path = dir.join(file);
        save_to_path(&TextureGraphFile::new(meta, graph), &path).expect("write graph");
        println!("wrote {}", path.display());
    }
}

/// Sea level on the continent field, before the `ocean` parameter moves it.
const EARTH_SEA: f32 = 0.545;

fn earthlike() -> Graph {
    let mut b = Builder::new();
    b.param("ocean", -0.15, 0.15, 0.0, "Raises the sea: positive drowns land, negative exposes shelf.");
    b.param("ice", -0.3, 0.3, 0.0, "Pushes polar ice toward the equator.");
    b.param("aridity", -0.3, 0.3, 0.0, "Widens the desert belts.");

    let continents = b.fbm("continents", 0, 1.3, 8, 0.52, FractalMode::Standard);
    let ocean = b.param_gray("ocean level", "ocean");
    let terrain_base = b.sub("terrain base", continents, ocean);
    let inland = b.remap("inland", terrain_base, EARTH_SEA + 0.01, EARTH_SEA + 0.12);
    let ridges = b.fbm("ridges", 1, 2.6, 7, 0.5, FractalMode::Ridged);
    let ranges = b.mask("ranges", ridges, inland);
    let ranges = b.scale("ranges scaled", ranges, 0.32);
    let terrain = b.add("terrain", terrain_base, ranges);

    // The ocean floor is clamped to sea level, so the normal map is flat there.
    let sea = b.gray("sea level", EARTH_SEA);
    let surface_height = b.max("surface height", terrain, sea);
    let normal = b.add_layer(
        "normal",
        LayerKind::HeightToNormal(HeightToNormal { source: Some(surface_height), strength: 0.12 }),
    );

    let land = b.remap("land", terrain, EARTH_SEA - 0.002, EARTH_SEA + 0.004);
    let elevation = b.remap("elevation", terrain, EARTH_SEA, EARTH_SEA + 0.42);
    let depth = b.remap("depth", terrain, EARTH_SEA - 0.2, EARTH_SEA);

    // Latitude: |y| on the sphere, 0 at the equator and 1 at the poles.
    let y = b.add_layer("y", LayerKind::Coordinate(Coordinate { axis: Axis::V }));
    let abs_lat = b.wave("abs latitude", y, WaveShape::Triangle, 1.0, 0.25);

    // Deserts sit around |y| = 0.42 (about 25°), wet belts at the equator
    // and in the temperate zone: minus a cosine of period 0.42 in v about
    // v = 0.5, which is 2.38 cycles with phase -0.5·2.38 - 0.25.
    let arid_belt = b.wave("arid belt", y, WaveShape::Sine, 2.38, -1.44);
    let moisture = b.fbm("moisture", 2, 2.2, 6, 0.55, FractalMode::Standard);
    let arid_param = b.param_gray("aridity level", "aridity");
    let aridity = b.weighted_sum("aridity", &[(arid_belt, 0.55), (moisture, 0.8), (arid_param, 1.0)]);
    let desert = b.remap("desert", aridity, 0.8, 0.92);

    // Deep ocean to cloud is kept to about 3.5 stops, the look of a photograph of
    // Earth rather than its true 6: a renderer with a narrow tone window clips a
    // darker ocean to black.
    let ocean_palette = b.ramp(
        "ocean palette",
        &[
            (0.0, oklcha(0.42, 0.08, 258.0, 1.0)),
            (0.7, oklcha(0.47, 0.09, 248.0, 1.0)),
            (0.93, oklcha(0.55, 0.10, 230.0, 1.0)),
            (1.0, oklcha(0.64, 0.09, 205.0, 1.0)),
        ],
    );
    let ocean_color = b.map("ocean color", depth, ocean_palette);
    let lush_palette = b.ramp(
        "lush palette",
        &[
            (0.0, oklcha(0.70, 0.06, 90.0, 1.0)),
            (0.02, oklcha(0.54, 0.10, 138.0, 1.0)),
            (0.25, oklcha(0.50, 0.09, 132.0, 1.0)),
            (0.5, oklcha(0.53, 0.06, 95.0, 1.0)),
            (0.72, oklcha(0.55, 0.04, 60.0, 1.0)),
            (0.86, oklcha(0.64, 0.015, 60.0, 1.0)),
            (0.93, oklcha(0.90, 0.005, 240.0, 1.0)),
            (1.0, oklcha(0.92, 0.0, 0.0, 1.0)),
        ],
    );
    let lush = b.map("lush", elevation, lush_palette);
    let arid_palette = b.ramp(
        "arid palette",
        &[
            (0.0, oklcha(0.78, 0.07, 85.0, 1.0)),
            (0.3, oklcha(0.72, 0.09, 72.0, 1.0)),
            (0.6, oklcha(0.58, 0.09, 50.0, 1.0)),
            (0.86, oklcha(0.55, 0.04, 50.0, 1.0)),
            (0.93, oklcha(0.90, 0.005, 240.0, 1.0)),
            (1.0, oklcha(0.92, 0.0, 0.0, 1.0)),
        ],
    );
    let arid = b.map("arid", elevation, arid_palette);
    let land_color = b.blend("land color", lush, arid, desert);
    let ground = b.blend("ground", ocean_color, land_color, land);

    let ice_noise = b.fbm("ice noise", 3, 4.0, 6, 0.5, FractalMode::Standard);
    let ice_param = b.param_gray("ice level", "ice");
    let ice_drive = b.weighted_sum(
        "ice drive",
        &[(abs_lat, 1.0), (ice_noise, 0.22), (elevation, 0.12), (ice_param, 1.0)],
    );
    let ice = b.remap("ice", ice_drive, 0.97, 1.0);
    let ice_color = b.gray_color("ice color", oklcha(0.91, 0.01, 230.0, 1.0));
    let color = b.blend("color", ground, ice_color, ice);

    let water_rough = b.gray("water roughness", 0.22);
    let land_rough = b.gray("land roughness", 0.85);
    let ice_rough = b.gray("ice roughness", 0.5);
    let ground_rough = b.blend("ground roughness", water_rough, land_rough, land);
    let roughness = b.blend("roughness", ground_rough, ice_rough, ice);

    b.finish(color, roughness, Some(normal))
}

fn earthlike_clouds() -> Graph {
    let mut b = Builder::new();
    b.param("cover", -0.2, 0.2, 0.0, "More cloud when positive.");

    let big = b.fbm("fronts", 20, 1.8, 7, 0.55, FractalMode::Standard);
    let wisps = b.fbm("wisps", 21, 5.0, 6, 0.55, FractalMode::Turbulence);
    // The desert belt's wave negated: cloudy at the equator and mid
    // latitudes, clear in the subtropics.
    let y = b.add_layer("y", LayerKind::Coordinate(Coordinate { axis: Axis::V }));
    let belts = b.wave("belts", y, WaveShape::Sine, 2.38, -0.94);
    let cover = b.param_gray("cover level", "cover");
    let weather = b.weighted_sum("weather", &[(big, 0.8), (wisps, 0.5)]);
    // Squash along y, about the equator, so systems stretch east-west as
    // winds shear them.
    let zonal = b.add_layer(
        "zonal",
        LayerKind::Transform(Transform {
            source: Some(weather),
            offset: [0.0, 0.5, 0.0],
            rotate_uv: 0.0,
            scale: [1.0, 1.8, 1.0],
            coord_mode: CoordMode::Passthrough,
            edge_mode: EdgeMode::Extend,
        }),
    );
    let drive = b.weighted_sum("drive", &[(zonal, 1.0), (belts, 0.12), (cover, 1.0)]);
    let density = b.remap("density", drive, 0.58, 0.8);
    let palette = b.ramp(
        "cloud palette",
        &[
            (0.0, oklcha(0.92, 0.0, 0.0, 0.0)),
            (0.4, oklcha(0.91, 0.0, 0.0, 0.45)),
            (1.0, oklcha(0.94, 0.0, 0.0, 0.92)),
        ],
    );
    let color = b.map("clouds", density, palette);
    let roughness = b.gray("roughness", 0.95);
    b.finish(color, roughness, None)
}

fn marslike() -> Graph {
    let mut b = Builder::new();
    b.param("ice", -0.3, 0.3, 0.0, "Pushes the polar caps toward the equator.");
    b.param("dark", -0.3, 0.3, 0.0, "More dark basaltic provinces when positive.");
    b.param_color("dust", oklcha(0.63, 0.13, 50.0, 1.0), "Midland dust color.");

    // Higher in the south, as on Mars.
    let y = b.add_layer("y", LayerKind::Coordinate(Coordinate { axis: Axis::V }));
    let base = b.fbm("base", 0, 1.1, 8, 0.55, FractalMode::Standard);
    let dichotomy = b.fbm("dichotomy", 1, 0.55, 3, 0.5, FractalMode::Standard);
    let south_up = b.scale("south up", y, -0.18);
    let heights = b.weighted_sum("heights", &[(base, 1.0), (dichotomy, 0.35), (south_up, 1.0)]);

    // The creases of ridged noise make canyon networks.
    let creases = b.fbm("creases", 4, 1.7, 6, 0.5, FractalMode::Ridged);
    let canyons = b.remap("canyons", creases, 0.72, 0.86);
    let volcanic = b.fbm("volcanic", 6, 3.2, 6, 0.5, FractalMode::Ridged);
    let terrain = b.weighted_sum(
        "terrain",
        &[(heights, 1.0), (canyons, -0.09), (volcanic, 0.07)],
    );
    let normal = b.add_layer(
        "normal",
        LayerKind::HeightToNormal(HeightToNormal { source: Some(terrain), strength: 0.14 }),
    );

    let elevation = b.remap("elevation", terrain, 0.48, 0.85);
    let palette = b.ramp_inputs(
        "rust palette",
        &[
            (0.0, ColorInput::Const(oklcha(0.46, 0.07, 38.0, 1.0))),
            (0.35, ColorInput::Const(oklcha(0.56, 0.12, 44.0, 1.0))),
            (0.6, ColorInput::Param("dust".into())),
            (1.0, ColorInput::Const(oklcha(0.74, 0.10, 64.0, 1.0))),
        ],
    );
    let rust = b.map("rust", elevation, palette);

    // Dark albedo provinces, kept off the high ground where dust settles.
    let albedo = b.fbm("albedo", 5, 1.9, 7, 0.55, FractalMode::Standard);
    let dark_param = b.param_gray("dark level", "dark");
    let dark_drive = b.weighted_sum("dark drive", &[(albedo, 1.0), (elevation, -0.25), (dark_param, 1.0)]);
    let dark = b.remap("dark", dark_drive, 0.45, 0.55);
    let basalt = b.gray_color("basalt", oklcha(0.43, 0.045, 38.0, 1.0));
    let dark_mix = b.scale("dark amount", dark, 0.65);
    let ground = b.blend("ground", rust, basalt, dark_mix);
    let canyon_shade = b.gray_color("canyon floor", oklcha(0.40, 0.07, 32.0, 1.0));
    let canyon_mix = b.scale("canyon amount", canyons, 0.6);
    let ground = b.blend("ground with canyons", ground, canyon_shade, canyon_mix);

    let abs_lat = b.wave("abs latitude", y, WaveShape::Triangle, 1.0, 0.25);
    let ice_noise = b.fbm("ice noise", 3, 3.5, 6, 0.55, FractalMode::Standard);
    let ice_param = b.param_gray("ice level", "ice");
    // Kept under 1: the GPU Map clamps its value to [0, 1] before the lookup.
    let ice_drive = b.weighted_sum("ice drive", &[(abs_lat, 0.8), (ice_noise, 0.24), (ice_param, 1.0)]);
    let ice = b.remap("ice", ice_drive, 0.85, 0.875);
    let ice_color = b.gray_color("ice color", oklcha(0.93, 0.025, 70.0, 1.0));
    let color = b.blend("color", ground, ice_color, ice);

    let dust_rough = b.gray("dust roughness", 0.92);
    let ice_rough = b.gray("ice roughness", 0.55);
    let roughness = b.blend("roughness", dust_rough, ice_rough, ice);

    b.finish(color, roughness, Some(normal))
}

/// The whole gamut of rocky worlds with air, in one graph. Earth and Mars are two settings of
/// it rather than two graphs: every parameter is a quantity a host derives from what the world
/// is made of -- how much water, how cold, whether anything lives there -- not a knob to taste.
fn rocky() -> Graph {
    let mut b = Builder::new();
    b.param("sea", 0.0, 1.0, 0.52, "Sea level on the height field. A host picks it from the share of the surface the ocean covers.");
    b.param("ice", -0.3, 1.5, 0.1, "Where the polar ice reaches: about the share of the surface under it, but a host measures that. Past one freezes the equator too.");
    b.param("life", 0.0, 1.0, 0.0, "How much of the wet land is alive.");
    b.param("rust", 0.0, 1.0, 0.5, "How oxidized bare ground is: gray basalt at 0, Mars at 1.");
    b.param("sand", 0.0, 1.0, 0.0, "How much of the dry land is pale sorted sand, which takes wind and water to make.");
    b.param("aridity", -0.3, 0.3, 0.0, "Widens the desert belts.");
    b.param("dark", -0.3, 0.3, 0.0, "More dark basaltic provinces when positive.");
    b.param_color("foliage", oklcha(0.40, 0.12, 145.0, 1.0), "Lowland growth. A host picks it from the star: gold under a hot one, dark under a red dwarf.");
    b.param_color("foliage high", oklcha(0.43, 0.10, 130.0, 1.0), "The sparser growth higher up.");

    let y = b.add_layer("y", LayerKind::Coordinate(Coordinate { axis: Axis::V }));
    let abs_lat = b.wave("abs latitude", y, WaveShape::Triangle, 1.0, 0.25);

    // Continents, plus Mars's hemispheric dichotomy at a lower frequency.
    let continents = b.fbm("continents", 0, 1.3, 8, 0.52, FractalMode::Standard);
    let dichotomy = b.fbm("dichotomy", 1, 0.55, 3, 0.5, FractalMode::Standard);
    let heights = b.weighted_sum("heights", &[(continents, 0.85), (dichotomy, 0.25)]);
    // Relief above the sea, centered on a half so a Map's clamp to [0, 1] has room each side.
    let sea = b.param_gray("sea level", "sea");
    let half = b.gray("half", 0.5);
    let below = b.sub("heights over sea", heights, sea);
    let relief = b.add("relief", below, half);

    let inland = b.remap("inland", relief, 0.51, 0.62);
    let ridges = b.fbm("ridges", 2, 2.6, 7, 0.5, FractalMode::Ridged);
    let ranges = b.mask("ranges", ridges, inland);
    // The creases of ridged noise make canyon networks.
    let creases = b.fbm("creases", 4, 1.7, 6, 0.5, FractalMode::Ridged);
    let canyons = b.remap("canyons", creases, 0.72, 0.86);
    let terrain = b.weighted_sum("terrain", &[(relief, 1.0), (ranges, 0.3), (canyons, -0.06)]);

    let floor = b.max("surface height", terrain, half);
    let normal = b.add_layer(
        "normal",
        LayerKind::HeightToNormal(HeightToNormal { source: Some(floor), strength: 0.12 }),
    );

    let land = b.remap("land", terrain, 0.498, 0.504);
    // Wide, so most land is lowland: a range is what climbs out of it.
    let elevation = b.remap("elevation", terrain, 0.5, 1.0);
    let depth = b.remap("depth", terrain, 0.3, 0.5);

    let ocean_palette = b.ramp(
        "ocean palette",
        &[
            (0.0, oklcha(0.42, 0.08, 258.0, 1.0)),
            (0.7, oklcha(0.47, 0.09, 248.0, 1.0)),
            (0.93, oklcha(0.55, 0.10, 230.0, 1.0)),
            (1.0, oklcha(0.64, 0.09, 205.0, 1.0)),
        ],
    );
    let ocean_color = b.map("ocean color", depth, ocean_palette);

    // Bare ground, between unweathered basalt and Mars.
    let basalt_palette = b.ramp(
        "basalt palette",
        &[
            (0.0, oklcha(0.42, 0.012, 60.0, 1.0)),
            (0.5, oklcha(0.51, 0.016, 65.0, 1.0)),
            (1.0, oklcha(0.62, 0.012, 70.0, 1.0)),
        ],
    );
    let rust_palette = b.ramp(
        "rust palette",
        &[
            (0.0, oklcha(0.46, 0.07, 38.0, 1.0)),
            (0.35, oklcha(0.56, 0.12, 44.0, 1.0)),
            (0.6, oklcha(0.63, 0.13, 50.0, 1.0)),
            (1.0, oklcha(0.74, 0.10, 64.0, 1.0)),
        ],
    );
    let gray_rock = b.map("gray rock", elevation, basalt_palette);
    let rusty_rock = b.map("rusty rock", elevation, rust_palette);
    let barren = b.mix("barren", gray_rock, rusty_rock, BlendMode::Blend, ScalarInput::Param("rust".into()));

    // Dark provinces, kept off the high ground where dust settles.
    let albedo = b.fbm("albedo", 5, 1.9, 7, 0.55, FractalMode::Standard);
    let dark_param = b.param_gray("dark level", "dark");
    let dark_drive = b.weighted_sum("dark drive", &[(albedo, 1.0), (elevation, -0.25), (dark_param, 1.0)]);
    let dark = b.remap("dark", dark_drive, 0.45, 0.55);
    let basalt = b.gray_color("basalt", oklcha(0.40, 0.03, 40.0, 1.0));
    let dark_mix = b.scale("dark amount", dark, 0.55);
    let rock = b.blend("rock", barren, basalt, dark_mix);
    let canyon_shade = b.gray_color("canyon floor", oklcha(0.40, 0.05, 36.0, 1.0));
    let canyon_mix = b.scale("canyon amount", canyons, 0.5);
    let rock = b.blend("rock with canyons", rock, canyon_shade, canyon_mix);

    // Deserts about |y| = 0.42 (25 degrees), wet belts at the equator and in the temperate zone.
    let arid_belt = b.wave("arid belt", y, WaveShape::Sine, 2.38, -1.44);
    let moisture = b.fbm("moisture", 6, 2.2, 6, 0.55, FractalMode::Standard);
    let arid_param = b.param_gray("aridity level", "aridity");
    let aridity = b.weighted_sum("aridity", &[(arid_belt, 0.55), (moisture, 0.8), (arid_param, 1.0)]);
    let desert = b.remap("desert", aridity, 0.8, 0.92);
    let white = b.gray("one", 1.0);
    let zero = b.black();
    let wet = b.blend("wet", white, zero, desert);

    let sand_palette = b.ramp(
        "sand palette",
        &[
            (0.0, oklcha(0.76, 0.07, 82.0, 1.0)),
            (0.4, oklcha(0.70, 0.085, 70.0, 1.0)),
            (0.8, oklcha(0.58, 0.07, 52.0, 1.0)),
            (1.0, oklcha(0.56, 0.04, 50.0, 1.0)),
        ],
    );
    let sand = b.map("sand", elevation, sand_palette);
    let sand_amount = b.mix("sand amount", zero, desert, BlendMode::Blend, ScalarInput::Param("sand".into()));
    let dry = b.blend("dry land", rock, sand, sand_amount);

    let lush_palette = b.ramp_inputs(
        "lush palette",
        &[
            (0.0, ColorInput::Const(oklcha(0.70, 0.06, 90.0, 1.0))),
            (0.02, ColorInput::Param("foliage".into())),
            (0.25, ColorInput::Param("foliage high".into())),
            (0.5, ColorInput::Const(oklcha(0.53, 0.06, 95.0, 1.0))),
            (0.72, ColorInput::Const(oklcha(0.55, 0.04, 60.0, 1.0))),
            (0.86, ColorInput::Const(oklcha(0.64, 0.015, 60.0, 1.0))),
            (0.93, ColorInput::Const(oklcha(0.90, 0.005, 240.0, 1.0))),
            (1.0, ColorInput::Const(oklcha(0.92, 0.0, 0.0, 1.0))),
        ],
    );
    let lush = b.map("lush", elevation, lush_palette);
    let green = b.mix("green", zero, wet, BlendMode::Blend, ScalarInput::Param("life".into()));
    let land_color = b.blend("land color", dry, lush, green);
    let ground = b.blend("ground", ocean_color, land_color, land);

    // Kept under 1 where it matters: a Map clamps its value to [0, 1] before the lookup, and
    // above the threshold a clamp changes nothing. The noise's mean is 0.15, so the edge sits
    // at |y| = 1 - ice; the coarse and fine octaves together make it ragged rather than a
    // circle of latitude, and high ground holds ice further from the pole.
    let ice_noise = b.fbm("ice noise", 3, 2.2, 7, 0.6, FractalMode::Standard);
    let ice_param = b.param_gray("ice level", "ice");
    let highland = b.mask("highland", elevation, land);
    let ice_drive = b.weighted_sum(
        "ice drive",
        &[(abs_lat, 0.75), (ice_noise, 0.3), (highland, 0.08), (ice_param, 0.75)],
    );
    let ice = b.remap("ice", ice_drive, 0.895, 0.915);
    let clean_ice = b.gray_color("clean ice", oklcha(0.92, 0.012, 230.0, 1.0));
    let dusty_ice = b.gray_color("dusty ice", oklcha(0.90, 0.03, 70.0, 1.0));
    let ice_color = b.mix("ice color", clean_ice, dusty_ice, BlendMode::Blend, ScalarInput::Param("rust".into()));
    let color = b.blend("color", ground, ice_color, ice);

    let water_rough = b.gray("water roughness", 0.22);
    let land_rough = b.gray("land roughness", 0.85);
    let ice_rough = b.gray("ice roughness", 0.5);
    let ground_rough = b.blend("ground roughness", water_rough, land_rough, land);
    let roughness = b.blend("roughness", ground_rough, ice_rough, ice);

    b.finish(color, roughness, Some(normal))
}

fn giant() -> Graph {
    let mut b = Builder::new();
    b.param("bands", 1.0, 12.0, 5.7, "Belt-zone pairs pole to pole. A host picks it from how fast and how big the giant is.");
    b.param("contrast", 0.0, 1.0, 0.9, "How far a belt departs from a zone.");
    b.param("turbulence", 0.0, 1.0, 0.8, "How far eddies wind the bands and draw the zones' white into the belts.");
    b.param("storms", 0.0, 1.0, 0.6, "How many ovals.");
    b.param("polar", 0.0, 0.5, 0.1, "About the share of the sphere under polar haze.");
    b.param("shift", 0.0, 1.0, 0.0, "Where the band pattern starts, in cycles.");
    b.param("spot", 0.0, 1.0, 1.0, "How much of a great spot the giant has.");
    b.param_color("zone", oklcha(0.88, 0.035, 80.0, 1.0), "The bright high deck.");
    b.param_color("belt", oklcha(0.68, 0.06, 55.0, 1.0), "A gap down to deeper, stained cloud.");
    b.param_color("tint", oklcha(0.58, 0.10, 45.0, 1.0), "A belt stained twice as deep: some belts, and the great spot.");
    b.param_color("storm", oklcha(0.92, 0.01, 90.0, 1.0), "The ovals' high tops.");
    b.param_color("polar color", oklcha(0.62, 0.03, 70.0, 1.0), "The haze over the poles.");

    let u = b.add_layer("x", LayerKind::Coordinate(Coordinate { axis: Axis::U }));
    let y = b.add_layer("y", LayerKind::Coordinate(Coordinate { axis: Axis::V }));
    let w = b.add_layer("z", LayerKind::Coordinate(Coordinate { axis: Axis::W }));
    let abs_lat = b.wave("abs latitude", y, WaveShape::Triangle, 1.0, 0.25);
    let half = b.gray("half", 0.5);
    let zero = b.black();
    let one = b.gray("one", 1.0);

    // The band phase in cycles, a twelfth at a time: a Wave's frequency is fixed, so the count
    // scales its input instead. A Multiply of two grays multiplies their L.
    let count = b.param_gray("band count", "bands");
    let twelfths = b.scale("band count over twelve", count, 1.0 / 12.0);
    let rows = b.multiply("rows", y, twelfths);
    let shift = b.param_gray("shift level", "shift");
    // Slow wander in longitude, and eddies at about a belt's width, both kept under a quarter
    // cycle: past that a warp stops winding bands and starts destroying them.
    let drift = b.fbm("drift", 1, 0.9, 3, 0.5, FractalMode::Standard);
    let eddies = b.fbm("eddies", 2, 5.0, 5, 0.55, FractalMode::Standard);
    let eddy = b.mix("eddy", half, eddies, BlendMode::Blend, ScalarInput::Param("turbulence".into()));
    let phase = b.weighted_sum(
        "phase",
        &[(rows, 1.0), (shift, 1.0 / 12.0), (drift, 0.4 / 12.0), (eddy, 0.55 / 12.0), (half, -0.95 / 12.0)],
    );
    // Everything finer than a band is a function of the phase, so it runs along the flow the
    // eddies wind rather than mottling it: the second harmonic makes no two belts the same
    // width, and the finer ones are the streaks inside them.
    let first = b.wave("first band", phase, WaveShape::Sine, 12.0, 0.0);
    let second = b.wave("second band", phase, WaveShape::Sine, 12.0 * 2.37, 0.3);
    let fine = b.wave("fine band", phase, WaveShape::Sine, 12.0 * 6.3, 0.1);
    let finer = b.wave("finer band", phase, WaveShape::Sine, 12.0 * 13.7, 0.6);
    let band = b.weighted_sum("band", &[(first, 0.56), (second, 0.2), (fine, 0.15), (finer, 0.09)]);
    let belted = b.remap("belted", band, 0.44, 0.62);
    // A belt fades and returns along its length, as Jupiter's southern one does.
    let fading = b.fbm("fading", 4, 1.3, 2, 0.5, FractalMode::Standard);
    let strength = b.remap("belt strength", fading, 0.25, 0.65);
    let strength = b.mix("belt presence", strength, one, BlendMode::Blend, ScalarInput::Const(0.45));
    let belt_raw = b.mask("belt raw", belted, strength);
    let contrasted = b.mix("contrasted", zero, belt_raw, BlendMode::Blend, ScalarInput::Param("contrast".into()));

    // Polar haze, from |sin latitude| = 1 - polar out, ragged; the bands give out before it.
    // Offset so its edge is inside [0, 1]: the GPU clamps a Map's value there and the CPU does
    // not, so a ramp stop past one bakes differently from how it evaluates.
    let polar = b.param_gray("polar level", "polar");
    let polar_noise = b.fbm("polar noise", 5, 3.0, 4, 0.5, FractalMode::Standard);
    let polar_drive = b.weighted_sum("polar drive", &[(abs_lat, 1.0), (polar, 1.0), (polar_noise, 0.1), (half, -0.4)]);
    let polar_edge = b.remap("polar edge", polar_drive, 0.83, 0.87);
    // None at all where the host says none, rather than a speck at each pole.
    let polar_on = b.remap("polar on", polar, 0.0, 0.02);
    let polar_mask = b.mask("polar", polar_edge, polar_on);
    let band_fade = b.remap("band fade", polar_drive, 0.72, 0.85);
    let open = b.blend("band field", one, zero, band_fade);

    // Festoons: zone drawn into the belts where the eddies curl, strung along the flow.
    let curls = b.fbm("curls", 6, 9.0, 4, 0.5, FractalMode::Ridged);
    let curl_band = b.weighted_sum("curl band", &[(curls, 0.7), (fine, 0.3)]);
    let wisp_drive = b.mix("wisp drive", zero, curl_band, BlendMode::Blend, ScalarInput::Param("turbulence".into()));
    let wisps = b.remap("wisps", wisp_drive, 0.5, 0.8);
    let taken = b.scale("wisps taken", wisps, 0.6);
    let kept = b.sub("belt kept", one, taken);
    let belt_open = b.mask("belt open", contrasted, open);
    let belt = b.mask("belt", belt_open, kept);

    // Ovals: blobs of a band-scale noise over a threshold the storm count lowers, fewest at
    // the poles.
    let ovals = b.fbm("ovals", 7, 14.0, 1, 0.5, FractalMode::Standard);
    let storms = b.param_gray("storm level", "storms");
    let oval_drive = b.weighted_sum("oval drive", &[(ovals, 1.0), (storms, 0.1), (abs_lat, -0.15)]);
    let small = b.remap("small ovals", oval_drive, 0.82, 0.87);
    let small = b.mask("small in the bands", small, open);
    // One great spot, 22 degrees south, three times as long as it is wide: an ellipsoid about a
    // point on the sphere, ragged by the eddies. Each axis is clamped to its reach before it is
    // squared, and the sum quartered, because the CPU caps a Multiply at one and the GPU does not.
    let spot_at = [22f32.to_radians().cos() * 0.5 + 0.5, 0.5 - 22f32.to_radians().sin() * 0.5, 0.5];
    let mut axes = Vec::new();
    for (name, coord, center, reach) in [("x", u, spot_at[0], 0.09), ("y", y, spot_at[1], 0.045), ("z", w, spot_at[2], 0.13)] {
        let at = b.gray(&format!("spot {name} center"), center);
        let ahead = b.sub(&format!("spot {name} ahead"), coord, at);
        let behind = b.sub(&format!("spot {name} behind"), at, coord);
        let off = b.max(&format!("spot {name} off"), ahead, behind);
        let scaled = b.remap(&format!("spot {name} reach"), off, 0.0, reach);
        axes.push(b.multiply(&format!("spot {name} squared"), scaled, scaled));
    }
    let reach = b.weighted_sum("spot reach", &[(axes[0], 0.25), (axes[1], 0.25), (axes[2], 0.25), (eddies, 0.12)]);
    let inside = b.remap("spot outside", reach, 0.28, 0.33);
    let great_raw = b.blend("spot inside", one, zero, inside);
    let great_spot = b.mix("great spot", zero, great_raw, BlendMode::Blend, ScalarInput::Param("spot".into()));
    let storm = b.max("storm", small, great_spot);

    // The color, in linear light, so it mixes as the shader's masks do.
    let zone_c = b.param_color_layer("zone color", "zone");
    let belt_c = b.param_color_layer("belt color", "belt");
    let tint_c = b.param_color_layer("tint color", "tint");
    let storm_c = b.param_color_layer("storm color", "storm");
    let polar_c = b.param_color_layer("polar haze color", "polar color");
    let staining = b.fbm("staining", 9, 2.2, 3, 0.5, FractalMode::Standard);
    let stained = b.remap("stained", staining, 0.5, 0.68);
    let tint_amount = b.mask("tint amount", stained, belt);
    let banded = b.linear_blend("banded", zone_c, belt_c, belt);
    let tinted = b.linear_blend("tinted", banded, tint_c, tint_amount);
    let ovals_c = b.linear_blend("oval color", storm_c, tint_c, great_spot);
    let stormed = b.linear_blend("stormed", tinted, ovals_c, storm);
    let color = b.linear_blend("color", stormed, polar_c, polar_mask);

    let roughness = b.gray("roughness", 0.8);
    b.finish(color, roughness, None)
}

/// Terse graph construction. Every scalar is a gray's Oklch L; `Blend` in
/// Oklch lerps L, so it doubles as the scalar arithmetic.
struct Builder {
    g: Graph,
    black: Option<LayerId>,
}

impl Builder {
    fn new() -> Self {
        let mut g = Graph::new();
        // `Graph::new` makes a placeholder output layer; nothing here uses it.
        let placeholder = g.output.color.unwrap();
        g.remove(placeholder).unwrap();
        Self { g, black: None }
    }

    fn add_layer(&mut self, name: &str, kind: LayerKind) -> LayerId {
        self.g.add_layer(name, kind).unwrap_or_else(|e| panic!("{name}: {e}"))
    }

    fn param(&mut self, name: &str, min: f32, max: f32, default: f32, description: &str) {
        let mut d = ParamDecl::scalar(name, min, max, default);
        d.description = Some(description.into());
        self.g.declare_param(d).unwrap();
    }

    fn param_color(&mut self, name: &str, default: Color, description: &str) {
        let mut d = ParamDecl::color(name, default);
        d.description = Some(description.into());
        self.g.declare_param(d).unwrap();
    }

    /// A gray whose L is the scalar parameter `param`.
    fn param_gray(&mut self, name: &str, param: &str) -> LayerId {
        let black = self.black();
        let white = self.gray(&format!("{name} white"), 1.0);
        self.add_layer(
            name,
            LayerKind::Mix(Mix {
                a: Some(black),
                b: Some(white),
                mode: BlendMode::Blend,
                factor: ScalarInput::Param(param.into()),
                space: BlendSpace::Oklch,
            }),
        )
    }

    fn black(&mut self) -> LayerId {
        if let Some(id) = self.black {
            return id;
        }
        let id = self.gray("zero", 0.0);
        self.black = Some(id);
        id
    }

    fn gray(&mut self, name: &str, l: f32) -> LayerId {
        self.gray_color(name, oklcha(l, 0.0, 0.0, 1.0))
    }

    fn gray_color(&mut self, name: &str, c: Color) -> LayerId {
        self.add_layer(name, LayerKind::Color(c))
    }

    /// Unsigned, normalized 3D simplex fbm.
    fn fbm(
        &mut self,
        name: &str,
        seed_offset: u32,
        frequency: f32,
        octaves: u32,
        gain: f32,
        mode: FractalMode,
    ) -> LayerId {
        self.add_layer(
            name,
            LayerKind::Noise(Noise {
                dims: NoiseDims::D3,
                seed_offset,
                frequency,
                range: NoiseRange::Unsigned,
                output: NoiseOutput::Grayscale,
                kernel: NoiseKernel::Simplex,
                period: [0; 3],
                fractal: Fractal { octaves, lacunarity: 2.0, gain, mode, normalize: true },
            }),
        )
    }

    fn wave(&mut self, name: &str, input: LayerId, shape: WaveShape, frequency: f32, phase: f32) -> LayerId {
        self.add_layer(
            name,
            LayerKind::Wave(Wave {
                input: ScalarInput::Layer(input),
                shape,
                frequency,
                phase,
                range: NoiseRange::Unsigned,
            }),
        )
    }

    fn mix(&mut self, name: &str, a: LayerId, b: LayerId, mode: BlendMode, factor: ScalarInput) -> LayerId {
        self.add_layer(
            name,
            LayerKind::Mix(Mix { a: Some(a), b: Some(b), mode, factor, space: BlendSpace::Oklch }),
        )
    }

    /// A layer that is the color parameter `param`: a palette of that one color, looked up.
    fn param_color_layer(&mut self, name: &str, param: &str) -> LayerId {
        let palette = self.ramp_inputs(
            &format!("{name} palette"),
            &[(0.0, ColorInput::Param(param.into())), (1.0, ColorInput::Param(param.into()))],
        );
        let half = self.gray(&format!("{name} lookup"), 0.5);
        self.map(name, half, palette)
    }

    /// `a * b`. On grays, in linear light, which multiplies their L.
    fn multiply(&mut self, name: &str, a: LayerId, b: LayerId) -> LayerId {
        self.mix(name, a, b, BlendMode::Multiply, ScalarInput::Const(0.0))
    }

    /// `blend` in linear light, as a renderer mixes albedos.
    fn linear_blend(&mut self, name: &str, a: LayerId, b: LayerId, factor: LayerId) -> LayerId {
        self.add_layer(
            name,
            LayerKind::Mix(Mix {
                a: Some(a),
                b: Some(b),
                mode: BlendMode::Blend,
                factor: ScalarInput::Layer(factor),
                space: BlendSpace::LinearSrgb,
            }),
        )
    }

    fn add(&mut self, name: &str, a: LayerId, b: LayerId) -> LayerId {
        self.mix(name, a, b, BlendMode::Add, ScalarInput::Const(0.0))
    }

    fn sub(&mut self, name: &str, a: LayerId, b: LayerId) -> LayerId {
        self.mix(name, a, b, BlendMode::Subtract, ScalarInput::Const(0.0))
    }

    fn blend(&mut self, name: &str, a: LayerId, b: LayerId, factor: LayerId) -> LayerId {
        self.mix(name, a, b, BlendMode::Blend, ScalarInput::Layer(factor))
    }

    /// `a * k`, as a lerp from zero.
    fn scale(&mut self, name: &str, a: LayerId, k: f32) -> LayerId {
        let black = self.black();
        self.mix(name, black, a, BlendMode::Blend, ScalarInput::Const(k))
    }

    /// `a * mask`.
    fn mask(&mut self, name: &str, a: LayerId, mask: LayerId) -> LayerId {
        let black = self.black();
        self.blend(name, black, a, mask)
    }

    /// `Σ kᵢ · aᵢ`, the intermediate terms named after `name`.
    fn weighted_sum(&mut self, name: &str, terms: &[(LayerId, f32)]) -> LayerId {
        let mut acc: Option<LayerId> = None;
        for (i, &(id, k)) in terms.iter().enumerate() {
            let last = i + 1 == terms.len();
            let term = if k == 1.0 { id } else { self.scale(&format!("{name} term {i}"), id, k) };
            acc = Some(match acc {
                None => term,
                Some(prev) => {
                    let label = if last { name.to_string() } else { format!("{name} partial {i}") };
                    self.add(&label, prev, term)
                }
            });
        }
        acc.expect("at least one term")
    }

    fn max(&mut self, name: &str, a: LayerId, b: LayerId) -> LayerId {
        self.add_layer(
            name,
            LayerKind::MinMax(MinMax {
                a: Some(a),
                b: Some(b),
                mode: MinMaxMode::Max,
                criterion: Criterion::Luma,
            }),
        )
    }

    fn ramp(&mut self, name: &str, stops: &[(f32, Color)]) -> LayerId {
        let stops: Vec<_> = stops.iter().map(|&(t, c)| (t, ColorInput::Const(c))).collect();
        self.ramp_inputs(name, &stops)
    }

    fn ramp_inputs(&mut self, name: &str, stops: &[(f32, ColorInput)]) -> LayerId {
        self.add_layer(
            name,
            LayerKind::ColorRamp(ColorRamp {
                stops: stops.iter().map(|(t, c)| ColorStop { t: *t, color: c.clone() }).collect(),
                space: BlendSpace::Oklch,
            }),
        )
    }

    fn map(&mut self, name: &str, value: LayerId, palette: LayerId) -> LayerId {
        self.add_layer(name, LayerKind::Map(Map { value: Some(value), palette: Some(palette) }))
    }

    /// `(x - lo) / (hi - lo)`, clamped to `[0, 1]`.
    fn remap(&mut self, name: &str, x: LayerId, lo: f32, hi: f32) -> LayerId {
        let palette = self.ramp(
            &format!("{name} ramp"),
            &[(lo, oklcha(0.0, 0.0, 0.0, 1.0)), (hi, oklcha(1.0, 0.0, 0.0, 1.0))],
        );
        self.map(name, x, palette)
    }

    /// Sets the output and lays every layer out left to right by depth.
    fn finish(mut self, color: LayerId, roughness: LayerId, normal: Option<LayerId>) -> Graph {
        self.g
            .set_output(Output {
                color: Some(color),
                roughness: ScalarInput::Layer(roughness),
                metallic: ScalarInput::Const(0.0),
                normal,
            })
            .unwrap();

        let mut depth: HashMap<LayerId, usize> = HashMap::new();
        // `layers` is id-ascending and inputs always predate their readers.
        for l in &self.g.layers {
            let d = l.kind.inputs().iter().map(|i| depth[i] + 1).max().unwrap_or(0);
            depth.insert(l.id, d);
        }
        let mut rows: HashMap<usize, usize> = HashMap::new();
        self.g.add_canvas("main").unwrap();
        let ids: Vec<LayerId> = self.g.layers.iter().map(|l| l.id).collect();
        for id in ids {
            let col = depth[&id];
            let row = rows.entry(col).or_default();
            self.g
                .set_position("main", id, [col as f32 * 260.0, *row as f32 * 150.0])
                .unwrap();
            *row += 1;
        }
        let last_col = depth.values().max().copied().unwrap_or(0) + 1;
        self.g.set_output_position("main", [last_col as f32 * 260.0, 0.0]).unwrap();
        self.g
    }
}
