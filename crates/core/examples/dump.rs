use texture_graph_core::*;
use texture_graph_core::kind::{ColorInput, ColorRamp, ColorStop, LayerKind};
use texture_graph_core::color::oklcha;

fn main() {
    let mut g = Graph::new();
    let a = g.add_layer("dark", LayerKind::Color(oklcha(0.2, 0.05, 30.0, 1.0))).unwrap();
    let b = g.add_layer("light", LayerKind::Color(oklcha(0.9, 0.05, 200.0, 1.0))).unwrap();
    let ramp = g.add_layer("ramp", LayerKind::ColorRamp(ColorRamp {
        stops: vec![
            ColorStop { t: 0.0, color: ColorInput::Layer(a) },
            ColorStop { t: 0.5, color: ColorInput::Const(oklcha(0.5, 0.1, 90.0, 1.0)) },
            ColorStop { t: 1.0, color: ColorInput::Layer(b) },
        ],
        space: BlendSpace::Oklch,
    })).unwrap();
    g.add_canvas("main").unwrap();
    g.set_position("main", a, [10.0, 20.0]).unwrap();
    g.set_position("main", b, [110.0, 20.0]).unwrap();
    g.set_position("main", ramp, [60.0, 120.0]).unwrap();
    let meta = FileMetadata {
        name: "demo".into(),
        description: Some("hand-rolled sample".into()),
        authors: vec!["zandy".into()],
        modified: "2026-07-17T00:00:00Z".into(),
        written_by: "dump".into(),
    };
    print!("{}", save_to_string(&TextureGraphFile::new(meta, g)).unwrap());
}
