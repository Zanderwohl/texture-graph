//! Shared "color, layer or parameter" picker for `ColorInput` fields.
//!
//! `id_source` must be unique across the frame: pass a tuple of the parent
//! layer id and the field name, e.g. `(id.0, "ramp-stop", i)`.

use texture_graph_core::{ColorInput, EvalCtx, Graph, LayerId, ParamUse, color::oklcha};

use crate::widgets::{color_edit, layer_ref, param_ref};

/// `except` is left out of the layer list so a layer can't reference itself.
pub fn color_input_widget(
    ui: &mut egui::Ui,
    graph: &Graph,
    id_source: impl std::hash::Hash + std::fmt::Debug + Copy,
    label: &str,
    input: &mut ColorInput,
    except: Option<LayerId>,
    ctx: &EvalCtx,
) -> bool {
    let mut changed = false;
    // A mode switch replaces the enum the match arm is borrowing, so it is
    // assigned after the match.
    let mut swap_to: Option<ColorInput> = None;

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
                swap_to = Some(ColorInput::Layer(fallback));
            }
            // Hidden when no color parameter exists, since it could only fail.
            if let Some(first) = param_ref::first(graph, ParamUse::Color) {
                if ui.small_button("use param").clicked() {
                    swap_to = Some(ColorInput::Param(first));
                }
            }
        }
        ColorInput::Layer(lref) => {
            if let Some(new) = layer_ref::layer_ref(ui, id_source, label, *lref, graph, except) {
                *lref = new;
                changed = true;
            }
            if ui.small_button("use const").clicked() {
                swap_to = Some(ColorInput::Const(oklcha(0.5, 0.0, 0.0, 1.0)));
            }
        }
        ColorInput::Param(name) => {
            if let Some(new) =
                param_ref::param_ref(ui, id_source, label, name, graph, ParamUse::Color)
            {
                *name = new;
                changed = true;
            }
            if ui.small_button("use const").clicked() {
                // Freeze at the current value so unbinding does not change
                // the picture.
                let frozen = graph
                    .param_value(name, ctx)
                    .and_then(|v| v.as_color())
                    .unwrap_or_else(|| oklcha(0.5, 0.0, 0.0, 1.0));
                swap_to = Some(ColorInput::Const(frozen));
            }
        }
    }

    if let Some(next) = swap_to {
        *input = next;
        changed = true;
    }
    changed
}
