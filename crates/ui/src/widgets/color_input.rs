//! Shared "color or layer" picker for `ColorInput` fields.
//!
//! `id_source` must be unique across the frame: pass a tuple of the parent
//! layer id and the field name, e.g. `(id.0, "ramp-stop", i)`.

use texture_graph_core::{ColorInput, Graph, LayerId, color::oklcha};

use crate::widgets::{color_edit, layer_ref};

/// Draw a `ColorInput` inline and mutate it in place.
///
/// * `label` — prefixes the layer combo in Layer mode; empty for none.
/// * `except` — hidden from the picker, so a layer can't reference itself.
pub fn color_input_widget(
    ui: &mut egui::Ui,
    graph: &Graph,
    id_source: impl std::hash::Hash + std::fmt::Debug + Copy,
    label: &str,
    input: &mut ColorInput,
    except: Option<LayerId>,
) -> bool {
    let mut changed = false;
    match input {
        ColorInput::Const(c) => {
            if !label.is_empty() {
                ui.label(label);
            }
            changed |= color_edit::oklcha_edit(ui, c);
            if ui.small_button("use layer").clicked() {
                let fallback = graph
                    .layers
                    .iter()
                    .find(|l| Some(l.id) != except)
                    .map(|l| l.id)
                    .or_else(|| graph.layers.first().map(|l| l.id))
                    .unwrap_or(LayerId(0));
                *input = ColorInput::Layer(fallback);
                changed = true;
            }
        }
        ColorInput::Layer(lref) => {
            if let Some(new) = layer_ref::layer_ref(ui, id_source, label, *lref, graph, except) {
                *lref = new;
                changed = true;
            }
            if ui.small_button("use const").clicked() {
                *input = ColorInput::Const(oklcha(0.5, 0.0, 0.0, 1.0));
                changed = true;
            }
        }
    }
    changed
}
