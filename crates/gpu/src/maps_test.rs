//! `MaterialMapper` on a real device: what lands in which channel, and that mips average the way
//! each map needs.

use texture_graph_core::{
    Axis, Coordinate, EvalCtx, Graph, LayerKind, NoiseRange, Output, ScalarInput, Wave, WaveShape,
};

use crate::readback::read_rgba8_mip;
use crate::{Baker, DeviceCtx, MaterialMapper, mip_level_count};

const SIZE: u32 = 64;

fn maps(graph: &Graph) -> (DeviceCtx, crate::MaterialMaps) {
    let ctx = pollster::block_on(DeviceCtx::request_headless()).expect("headless");
    let mut baker = Baker::new(ctx.clone());
    let out = baker.bake_output(graph, (SIZE, SIZE), &EvalCtx::default(), false).expect("bakes");
    let maps = MaterialMapper::new(&ctx.device).maps(&ctx, &out);
    (ctx, maps)
}

fn close(a: u8, b: u8, tolerance: u8) -> bool {
    a.abs_diff(b) <= tolerance
}

#[test]
fn every_level_down_to_one_texel() {
    assert_eq!(mip_level_count(1), 1);
    assert_eq!(mip_level_count(64), 7);
    assert_eq!(mip_level_count(512), 10);
    assert_eq!(mip_level_count(100), 7);
}

/// Breaks if the packing moves a channel: Bevy reads roughness from G and metallic from B, and a
/// swap draws every rough wall as polished metal.
#[test]
fn roughness_and_metallic_land_where_gltf_reads_them_at_every_level() {
    let mut graph = Graph::new();
    graph
        .set_output(Output {
            color: graph.output.color,
            roughness: ScalarInput::Const(0.25),
            metallic: ScalarInput::Const(0.75),
            normal: None,
        })
        .unwrap();
    let (ctx, maps) = maps(&graph);
    for level in [0, maps.mip_level_count - 1] {
        let px = &read_rgba8_mip(&ctx, &maps.metallic_roughness, level).pixels[..4];
        assert_eq!(px[0], 255, "occlusion at level {level}");
        assert!(close(px[1], 64, 2), "roughness {} at level {level}", px[1]);
        assert!(close(px[2], 191, 2), "metallic {} at level {level}", px[2]);
    }
}

/// Black and white stripes average to half the light, which is 188 in sRGB. Averaging the stored
/// bytes instead gives 128: every distant wall drawn too dark.
#[test]
fn colour_mips_average_in_linear_light() {
    let mut graph = Graph::new();
    let u = graph.add_layer("u", LayerKind::Coordinate(Coordinate { axis: Axis::U })).unwrap();
    let stripes = graph
        .add_layer(
            "stripes",
            LayerKind::Wave(Wave {
                input: ScalarInput::Layer(u),
                shape: WaveShape::Square,
                frequency: 8.0,
                phase: 0.0,
                range: NoiseRange::Unsigned,
            }),
        )
        .unwrap();
    let mut output = graph.output.clone();
    output.color = Some(stripes);
    graph.set_output(output).unwrap();

    let (ctx, maps) = maps(&graph);
    let top = read_rgba8_mip(&ctx, &maps.color, 0).pixels;
    assert!(top.chunks(4).any(|p| p[0] < 10) && top.chunks(4).any(|p| p[0] > 245), "no stripes to average");
    let last = read_rgba8_mip(&ctx, &maps.color, maps.mip_level_count - 1).pixels;
    assert!(close(last[0], 188, 4), "the 1x1 level is {}, not half the light", last[0]);
}

#[test]
fn a_material_with_no_normal_is_flat_at_every_level() {
    let (ctx, maps) = maps(&Graph::new());
    for level in [0, maps.mip_level_count - 1] {
        let px = &read_rgba8_mip(&ctx, &maps.normal, level).pixels[..4];
        assert!(close(px[0], 128, 2) && close(px[1], 128, 2) && close(px[2], 255, 2), "{px:?} at level {level}");
    }
}
