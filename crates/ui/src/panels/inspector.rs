//! Per-layer inspector — variant-specific parameter widgets and a
//! variant-swap combobox.

use texture_graph_core::color::oklcha;
use texture_graph_core::{
    Axis, BlendMode, BlendSpace, ColorInput, ColorRamp, ColorStop, CoordMode, Criterion, Graph,
    HeightToNormal, LayerId, LayerKind, Map, MinMax, MinMaxMode, Mix, Noise, NoiseKernel,
    RadialDim, ScalarInput, Transform, noise::MAX_OCTAVES,
};

use crate::catalog::{self, Kind};
use crate::state::{EditCmd, UiState};
use crate::widgets::enum_combo::enum_combo;
use crate::widgets::noise_labels;
use crate::widgets::{color_edit, layer_ref};

/// Render the inspector for a single layer. Emits `EditCmd::SetKind` when
/// any control changes.
pub fn show(ui: &mut egui::Ui, graph: &Graph, state: &mut UiState, id: LayerId) {
    let Some(layer) = graph.get(id) else { return };
    let mut kind = layer.kind.clone();
    let mut changed = false;

    // Variant switcher — swaps the whole `LayerKind` to a fresh default of
    // the chosen variant.
    let current = Kind::of(&kind);
    let current_label = catalog::VARIANTS
        .iter()
        .find(|v| v.kind == current)
        .map_or("", |v| v.label);
    egui::ComboBox::from_id_salt(("variant", id.0))
        .selected_text(current_label)
        .show_ui(ui, |ui| {
            for v in catalog::VARIANTS {
                if ui.selectable_label(current == v.kind, v.label).clicked() && current != v.kind
                {
                    kind = catalog::default_kind(v.kind);
                    changed = true;
                }
            }
        });

    match &mut kind {
        LayerKind::Color(c) => {
            changed |= color_edit::oklcha_edit(ui, c);
        }
        LayerKind::Noise(n) => {
            changed |= noise_widgets(ui, id, n);
        }
        LayerKind::ColorRamp(r) => {
            changed |= ramp_widgets(ui, graph, id, r);
        }
        LayerKind::Transform(t) => {
            changed |= transform_widgets(ui, graph, id, t);
        }
        LayerKind::Mix(m) => {
            changed |= mix_widgets(ui, graph, id, m);
        }
        LayerKind::Map(m) => {
            changed |= map_widgets(ui, graph, id, m);
        }
        LayerKind::MinMax(mm) => {
            changed |= min_max_widgets(ui, graph, id, mm);
        }
        LayerKind::HeightToNormal(h) => {
            changed |= h2n_widgets(ui, graph, id, h);
        }
    }

    if changed {
        state.push(EditCmd::SetKind(id, kind));
    }
}

// ---- Variant widgets ----------------------------------------------------

fn noise_widgets(ui: &mut egui::Ui, id: LayerId, n: &mut Noise) -> bool {
    let mut changed = false;
    if enum_combo(
        ui,
        (id.0, "noise-kernel"),
        "kernel",
        &mut n.kernel,
        noise_labels::KERNELS,
        noise_labels::kernel,
    ) {
        changed = true;
        // Leaving a period behind on simplex would make `set_kind` reject
        // the very edit the user just made, over a field the simplex UI
        // does not even show.
        if n.kernel == NoiseKernel::Simplex {
            n.period = [0; 3];
        }
    }
    changed |= enum_combo(
        ui,
        (id.0, "noise-dims"),
        "dims",
        &mut n.dims,
        noise_labels::DIMS,
        noise_labels::dims,
    );
    changed |= enum_combo(
        ui,
        (id.0, "noise-output"),
        "output",
        &mut n.output,
        noise_labels::OUTPUTS,
        noise_labels::output,
    );
    changed |= enum_combo(
        ui,
        (id.0, "noise-range"),
        "range",
        &mut n.range,
        noise_labels::RANGES,
        noise_labels::range,
    );
    changed |= ui
        .add(
            egui::Slider::new(&mut n.frequency, 0.1..=64.0)
                .logarithmic(true)
                .text("frequency"),
        )
        .changed();
    changed |= ui
        .add(egui::DragValue::new(&mut n.seed_offset).prefix("seed+"))
        .changed();

    // Only the value kernel has a lattice to wrap; simplex would be
    // rejected outright, so the control stays hidden rather than armed.
    if n.kernel == NoiseKernel::Value {
        ui.horizontal(|ui| {
            ui.label("period");
            for p in n.period.iter_mut() {
                changed |= ui.add(egui::DragValue::new(p).speed(0.25)).changed();
            }
        });
        ui.weak(noise_labels::period_hint(n.frequency, n.period));
    }

    changed |= ui
        .add(egui::Slider::new(&mut n.fractal.octaves, 1..=MAX_OCTAVES).text("octaves"))
        .changed();
    // At one octave the rest of the stack describes nothing.
    if n.fractal.octaves > 1 {
        changed |= enum_combo(
            ui,
            (id.0, "noise-fractal-mode"),
            "fbm",
            &mut n.fractal.mode,
            noise_labels::FRACTAL_MODES,
            noise_labels::fractal_mode,
        );
        changed |= ui
            .add(egui::Slider::new(&mut n.fractal.lacunarity, 1.0..=4.0).text("lacunarity"))
            .changed();
        if n.period != [0; 3] && n.fractal.lacunarity.fract() != 0.0 {
            ui.weak(
                "a fractional lacunarity scales the period off the lattice — \
                 the field stops tiling exactly",
            );
        }
        changed |= ui
            .add(egui::Slider::new(&mut n.fractal.gain, 0.0..=1.0).text("gain"))
            .changed();
        changed |= ui.checkbox(&mut n.fractal.normalize, "normalize").changed();
    }
    changed
}

fn ramp_widgets(ui: &mut egui::Ui, graph: &Graph, id: LayerId, r: &mut ColorRamp) -> bool {
    let mut changed = false;
    changed |= enum_combo(
        ui,
        (id.0, "ramp-space"),
        "space",
        &mut r.space,
        &[BlendSpace::Oklch, BlendSpace::LinearSrgb, BlendSpace::Hsv],
        |s| match s {
            BlendSpace::Oklch => "Oklch",
            BlendSpace::LinearSrgb => "LinearSrgb",
            BlendSpace::Hsv => "Hsv",
        },
    );

    let stop_count = r.stops.len();
    let mut remove_at: Option<usize> = None;
    for (i, stop) in r.stops.iter_mut().enumerate() {
        ui.horizontal(|ui| {
            changed |= crate::widgets::ramp_stop::stop_t(ui, &mut stop.t).changed();
            changed |= crate::widgets::color_input::color_input_widget(
                ui,
                graph,
                (id.0, "ramp-stop", i),
                "stop",
                &mut stop.color,
                Some(id),
            );
            if stop_count > 2 && ui.small_button("remove").clicked() {
                remove_at = Some(i);
            }
        });
    }
    if let Some(i) = remove_at {
        r.stops.remove(i);
        changed = true;
    }
    if ui.small_button("+ stop").clicked() {
        let new_t = r.stops.last().map(|s| s.t).unwrap_or(1.0);
        r.stops.push(ColorStop {
            t: new_t,
            color: ColorInput::Const(oklcha(0.5, 0.0, 0.0, 1.0)),
        });
        changed = true;
    }
    changed
}

fn transform_widgets(ui: &mut egui::Ui, graph: &Graph, id: LayerId, t: &mut Transform) -> bool {
    let mut changed = false;
    if let Some(new) = layer_ref::layer_ref_opt(ui, ("tform-src", id.0), "source", t.source, graph, Some(id)) {
        t.source = new;
        changed = true;
    }
    changed |= vec3_drag(ui, "offset", &mut t.offset, 0.01);
    changed |= ui
        .add(
            egui::Slider::new(&mut t.rotate_uv, -std::f32::consts::PI..=std::f32::consts::PI)
                .text("rotate_uv (rad)"),
        )
        .changed();
    changed |= vec3_drag(ui, "scale", &mut t.scale, 0.05);
    changed |= coord_mode_widgets(ui, id, &mut t.coord_mode);
    changed
}

fn coord_mode_widgets(ui: &mut egui::Ui, id: LayerId, cm: &mut CoordMode) -> bool {
    let mut changed = false;
    let current_key = match cm {
        CoordMode::Passthrough => "Passthrough",
        CoordMode::Permute(_) => "Permute",
        CoordMode::Radial { .. } => "Radial",
    };
    egui::ComboBox::from_id_salt(("coord-mode", id.0))
        .selected_text(current_key)
        .show_ui(ui, |ui| {
            for (key, next) in &[
                ("Passthrough", CoordMode::Passthrough),
                ("Permute", CoordMode::Permute([Axis::U, Axis::V, Axis::W])),
                ("Radial", CoordMode::Radial { dim: RadialDim::D2, into: Axis::U }),
            ] {
                if ui.selectable_label(current_key == *key, *key).clicked()
                    && current_key != *key
                {
                    *cm = *next;
                    changed = true;
                }
            }
        });
    match cm {
        CoordMode::Passthrough => {}
        CoordMode::Permute(axes) => {
            ui.horizontal(|ui| {
                for (i, axis) in axes.iter_mut().enumerate() {
                    let key = ["out u", "out v", "out w"][i];
                    egui::ComboBox::from_id_salt(("permute", id.0, i))
                        .selected_text(format!("{key}={}", axis_label(*axis)))
                        .show_ui(ui, |ui| {
                            for a in [Axis::U, Axis::V, Axis::W] {
                                if ui.selectable_label(*axis == a, axis_label(a)).clicked()
                                    && *axis != a
                                {
                                    *axis = a;
                                    changed = true;
                                }
                            }
                        });
                }
            });
        }
        CoordMode::Radial { dim, into } => {
            changed |= enum_combo(
                ui,
                (id.0, "coord-radial-dim"),
                "radial dim",
                dim,
                &[RadialDim::D2, RadialDim::D3],
                |d| match d {
                    RadialDim::D2 => "2D",
                    RadialDim::D3 => "3D",
                },
            );
            changed |= enum_combo(
                ui,
                (id.0, "coord-radial-into"),
                "radial into",
                into,
                &[Axis::U, Axis::V, Axis::W],
                axis_label,
            );
        }
    }
    changed
}

fn axis_label(a: Axis) -> &'static str {
    match a {
        Axis::U => "U",
        Axis::V => "V",
        Axis::W => "W",
    }
}

fn mix_widgets(ui: &mut egui::Ui, graph: &Graph, id: LayerId, m: &mut Mix) -> bool {
    let mut changed = false;
    if let Some(new) = layer_ref::layer_ref_opt(ui, ("mix-a", id.0), "a", m.a, graph, Some(id)) {
        m.a = new;
        changed = true;
    }
    if let Some(new) = layer_ref::layer_ref_opt(ui, ("mix-b", id.0), "b", m.b, graph, Some(id)) {
        m.b = new;
        changed = true;
    }
    changed |= enum_combo(
        ui,
        (id.0, "mix-mode"),
        "mode",
        &mut m.mode,
        &[BlendMode::Add, BlendMode::Subtract, BlendMode::Multiply, BlendMode::Blend],
        |mode| match mode {
            BlendMode::Add => "Add",
            BlendMode::Subtract => "Subtract",
            BlendMode::Multiply => "Multiply",
            BlendMode::Blend => "Blend",
        },
    );
    if matches!(m.mode, BlendMode::Blend) {
        changed |= enum_combo(
            ui,
            (id.0, "mix-space"),
            "space",
            &mut m.space,
            &[BlendSpace::Oklch, BlendSpace::LinearSrgb, BlendSpace::Hsv],
            |s| match s {
                BlendSpace::Oklch => "Oklch",
                BlendSpace::LinearSrgb => "LinearSrgb",
                BlendSpace::Hsv => "Hsv",
            },
        );
        changed |= scalar_input_widget(ui, graph, id, "factor", &mut m.factor);
    }
    changed
}

fn map_widgets(ui: &mut egui::Ui, graph: &Graph, id: LayerId, m: &mut Map) -> bool {
    let mut changed = false;
    if let Some(new) = layer_ref::layer_ref_opt(ui, ("map-val", id.0), "value", m.value, graph, Some(id)) {
        m.value = new;
        changed = true;
    }
    if let Some(new) = layer_ref::layer_ref_opt(ui, ("map-pal", id.0), "palette", m.palette, graph, Some(id)) {
        m.palette = new;
        changed = true;
    }
    changed
}

fn min_max_widgets(ui: &mut egui::Ui, graph: &Graph, id: LayerId, mm: &mut MinMax) -> bool {
    let mut changed = false;
    if let Some(new) = layer_ref::layer_ref_opt(ui, ("mm-a", id.0), "a", mm.a, graph, Some(id)) {
        mm.a = new;
        changed = true;
    }
    if let Some(new) = layer_ref::layer_ref_opt(ui, ("mm-b", id.0), "b", mm.b, graph, Some(id)) {
        mm.b = new;
        changed = true;
    }
    changed |= enum_combo(
        ui,
        (id.0, "mm-mode"),
        "mode",
        &mut mm.mode,
        &[MinMaxMode::Min, MinMaxMode::Max],
        |m| match m {
            MinMaxMode::Min => "Min",
            MinMaxMode::Max => "Max",
        },
    );
    changed |= enum_combo(
        ui,
        (id.0, "mm-criterion"),
        "by",
        &mut mm.criterion,
        &[
            Criterion::Red,
            Criterion::Green,
            Criterion::Blue,
            Criterion::Saturation,
            Criterion::Value,
            Criterion::Luma,
            Criterion::Alpha,
            Criterion::Chroma,
        ],
        |c| match c {
            Criterion::Red => "Red",
            Criterion::Green => "Green",
            Criterion::Blue => "Blue",
            Criterion::Saturation => "Saturation",
            Criterion::Value => "Value",
            Criterion::Luma => "Luma",
            Criterion::Alpha => "Alpha",
            Criterion::Chroma => "Chroma",
        },
    );
    changed
}

fn h2n_widgets(ui: &mut egui::Ui, graph: &Graph, id: LayerId, h: &mut HeightToNormal) -> bool {
    let mut changed = false;
    if let Some(new) = layer_ref::layer_ref_opt(ui, ("h2n-src", id.0), "source", h.source, graph, Some(id)) {
        h.source = new;
        changed = true;
    }
    changed |= ui
        .add(egui::Slider::new(&mut h.strength, 0.0..=8.0).text("strength"))
        .changed();
    changed
}

// ---- Small helpers ------------------------------------------------------

pub fn scalar_input_widget(
    ui: &mut egui::Ui,
    graph: &Graph,
    self_id: LayerId,
    label: &str,
    si: &mut ScalarInput,
) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(label);
        match si {
            ScalarInput::Const(v) => {
                changed |= ui
                    .add(egui::Slider::new(v, 0.0..=1.0))
                    .changed();
                if ui.small_button("use layer").clicked() {
                    *si = ScalarInput::Layer(
                        graph.layers.first().map(|l| l.id).unwrap_or(self_id),
                    );
                    changed = true;
                }
            }
            ScalarInput::Layer(lref) => {
                if let Some(new) = layer_ref::layer_ref(
                    ui,
                    ("scalar", self_id.0, label),
                    label,
                    *lref,
                    graph,
                    Some(self_id),
                ) {
                    *lref = new;
                    changed = true;
                }
                if ui.small_button("use const").clicked() {
                    *si = ScalarInput::Const(0.5);
                    changed = true;
                }
            }
        }
    });
    changed
}

fn vec3_drag(ui: &mut egui::Ui, label: &str, v: &mut [f32; 3], speed: f32) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(label);
        for x in v.iter_mut() {
            changed |= ui
                .add(egui::DragValue::new(x).speed(speed as f64))
                .changed();
        }
    });
    changed
}

