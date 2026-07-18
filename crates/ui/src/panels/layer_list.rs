use texture_graph_core::{EvalCtx, Graph, LayerId, ScalarInput};

use crate::app::GpuBits;
use crate::panels::inspector;
use crate::previews::{PREVIEW_SIZE, PreviewCache};
use crate::state::{EditCmd, UiState};

const THUMB_PX: f32 = 48.0;

/// Left panel: linear layer list in `Graph::list_order` order, plus
/// output binding editor and add-layer button.
pub fn show(
    ui: &mut egui::Ui,
    graph: &Graph,
    state: &mut UiState,
    previews: &mut PreviewCache,
    eval_ctx: &EvalCtx,
    mut gpu: Option<&mut GpuBits>,
) {
    ui.heading("Output");
    output_editor(ui, graph, state);
    ui.separator();

    ui.horizontal(|ui| {
        ui.heading("Layers");
        add_layer_menu(ui, graph, state);
    });
    ui.separator();

    egui::ScrollArea::vertical().show(ui, |ui| {
        if graph.list_order.is_empty() {
            ui.weak("(no layers)");
            return;
        }
        for id in graph.list_order.iter().copied() {
            row(ui, graph, state, previews, eval_ctx, id, gpu.as_deref_mut());
            ui.separator();
        }
    });
}

// ---- Add-layer menu -----------------------------------------------------

fn add_layer_menu(ui: &mut egui::Ui, graph: &Graph, state: &mut UiState) {
    ui.menu_button("+ Layer", |ui| {
        for &variant in &[
            "Color",
            "Noise",
            "ColorRamp",
            "Transform",
            "Mix",
            "Map",
            "MinMax",
            "HeightToNormal",
        ] {
            if ui.button(variant).clicked() {
                let base = variant.to_ascii_lowercase();
                let name = unique_name(graph, &base);
                let kind = inspector::default_kind(variant, graph);
                state.push(EditCmd::AddLayer { name, kind });
                ui.close();
            }
        }
    });
}

fn unique_name(graph: &Graph, base: &str) -> String {
    if !graph.layers.iter().any(|l| l.name == base) {
        return base.to_string();
    }
    for n in 1..u32::MAX {
        let candidate = format!("{base} {n}");
        if !graph.layers.iter().any(|l| l.name == candidate) {
            return candidate;
        }
    }
    base.to_string()
}

// ---- Output binding editor ---------------------------------------------

fn output_editor(ui: &mut egui::Ui, graph: &Graph, state: &mut UiState) {
    let mut out = graph.output.clone();
    let mut changed = false;

    if let Some(new) = crate::widgets::layer_ref::layer_ref(
        ui,
        "output-color",
        "color",
        out.color,
        graph,
        None,
    ) {
        out.color = new;
        changed = true;
    }
    changed |= inspector::scalar_input_widget(
        ui,
        graph,
        LayerId(u64::MAX),
        "roughness",
        &mut out.roughness,
    );
    changed |= inspector::scalar_input_widget(
        ui,
        graph,
        LayerId(u64::MAX),
        "metallic",
        &mut out.metallic,
    );

    let mut has_normal = out.normal.is_some();
    let toggled = ui.checkbox(&mut has_normal, "normal").changed();
    if toggled {
        if has_normal {
            out.normal = graph.layers.first().map(|l| l.id);
        } else {
            out.normal = None;
        }
        changed = true;
    }
    if let Some(nref) = out.normal {
        if let Some(new) =
            crate::widgets::layer_ref::layer_ref(ui, "output-normal", "normal", nref, graph, None)
        {
            out.normal = Some(new);
            changed = true;
        }
    }
    if changed {
        state.push(EditCmd::SetOutput(out));
    }
}

// ---- Rows ---------------------------------------------------------------

fn row(
    ui: &mut egui::Ui,
    graph: &Graph,
    state: &mut UiState,
    previews: &mut PreviewCache,
    eval_ctx: &EvalCtx,
    id: LayerId,
    gpu: Option<&mut GpuBits>,
) {
    let Some(layer) = graph.get(id) else { return };
    let selected = state.selected == Some(id);
    let is_output = graph.output.color == id
        || graph.output.normal == Some(id)
        || matches!(graph.output.roughness, ScalarInput::Layer(x) if x == id)
        || matches!(graph.output.metallic, ScalarInput::Layer(x) if x == id);

    let tex_id = previews.get_or_build(ui.ctx(), graph, id, eval_ctx, gpu);

    ui.horizontal(|ui| {
        let thumb_size = egui::vec2(THUMB_PX, THUMB_PX);
        let thumb_response = if let Some(tid) = tex_id {
            ui.add(
                egui::Image::new((tid, egui::vec2(PREVIEW_SIZE as f32, PREVIEW_SIZE as f32)))
                    .fit_to_exact_size(thumb_size)
                    .sense(egui::Sense::click()),
            )
        } else {
            let (rect, resp) = ui.allocate_exact_size(thumb_size, egui::Sense::click());
            ui.painter().rect_filled(rect, 2.0, egui::Color32::DARK_GRAY);
            resp
        };
        if thumb_response.clicked() {
            state.selected = Some(id);
        }
        ui.vertical(|ui| {
            ui.horizontal(|ui| {
                let mut name = egui::RichText::new(&layer.name);
                if is_output {
                    name = name.strong();
                }
                if ui.selectable_label(selected, name).clicked() {
                    state.selected = Some(id);
                }
                // Push the row menu to the far right of the row.
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    row_menu(ui, state, id);
                });
            });
            ui.weak(
                egui::RichText::new(format!("{} - {}", layer.kind.category_label(), id))
                    .small(),
            );
        });
    });
    if selected {
        ui.indent(format!("layer-{}-inspector", id.0), |ui| {
            inspector::show(ui, graph, state, id);
        });
    }
}

/// Hamburger-style menu on each row. Only Delete for now — future items
/// (Duplicate, Rename, Focus in graph…) go here.
fn row_menu(ui: &mut egui::Ui, state: &mut UiState, id: LayerId) {
    let btn_id = format!("layer-{}-row-menu", id.0);
    let btn = ui
        .small_button("...")
        .on_hover_text("row actions");
    egui::Popup::from_toggle_button_response(&btn)
        .id(egui::Id::new(btn_id))
        .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
        .show(|ui| {
            if ui.button("Delete").clicked() {
                state.push(EditCmd::Remove(id));
                ui.close();
            }
        });
}
