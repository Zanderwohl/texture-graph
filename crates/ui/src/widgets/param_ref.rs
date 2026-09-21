//! Picking a declared parameter for a `ScalarInput` / `ColorInput` socket.
//!
//! A socket only offers parameters of the sort it can read — the graph
//! rejects the other kind anyway, and a combo listing choices that will be
//! refused is worse than one that is short.

use texture_graph_core::{Graph, ParamKind, ParamUse};

/// Declared parameters this sort of socket can read, in name order.
pub fn usable(graph: &Graph, want: ParamUse) -> impl Iterator<Item = &str> {
    graph
        .params
        .values()
        .filter(move |d| want.accepts(&d.kind))
        .map(|d| d.name.as_str())
}

/// The name a "use parameter" button should switch to: the first one this
/// socket can read. `None` when the graph declares none, and the button
/// should not be drawn at all.
pub fn first(graph: &Graph, want: ParamUse) -> Option<String> {
    usable(graph, want).next().map(str::to_string)
}

/// Combo over the usable parameters. `Some` when the selection changed.
///
/// A `current` that is not in the list still shows — a graph loaded from
/// elsewhere may name something this build cannot offer, and silently
/// showing the first entry instead would look like the binding changed.
pub fn param_ref(
    ui: &mut egui::Ui,
    id_source: impl std::hash::Hash + std::fmt::Debug,
    label: &str,
    current: &str,
    graph: &Graph,
    want: ParamUse,
) -> Option<String> {
    let mut picked = None;
    egui::ComboBox::from_id_salt(id_source)
        .selected_text(format!("{label}: {current}"))
        .show_ui(ui, |ui| {
            for name in usable(graph, want) {
                let selected = name == current;
                if ui.selectable_label(selected, name).clicked() && !selected {
                    picked = Some(name.to_string());
                }
            }
        });
    picked
}

/// The range a scalar parameter declares, for a slider that edits its
/// value. Falls back to `0..=1`, which is what most of this UI uses.
pub fn scalar_range(graph: &Graph, name: &str) -> std::ops::RangeInclusive<f32> {
    match graph.params.get(name).map(|d| d.kind) {
        Some(ParamKind::Scalar { min, max }) if max > min => min..=max,
        _ => 0.0..=1.0,
    }
}

/// Compact "constant or parameter" control for a node-body row.
///
/// A `Layer` binding draws nothing — the wire on the canvas already says
/// so, and a node row is too narrow to repeat it.
///
/// `draw_const` is the caller's own editor for the constant case, because
/// a Mix factor wants a `0..=1` slider and a Wave input wants an unbounded
/// drag.
pub fn scalar_socket(
    ui: &mut egui::Ui,
    graph: &Graph,
    ctx: &texture_graph_core::EvalCtx,
    id_source: impl std::hash::Hash + std::fmt::Debug + Copy,
    si: &mut texture_graph_core::ScalarInput,
    draw_const: impl FnOnce(&mut egui::Ui, &mut f32) -> egui::Response,
) -> bool {
    use texture_graph_core::ScalarInput;

    let mut changed = false;
    // A mode switch replaces the enum the arm below is borrowing.
    let mut swap_to: Option<ScalarInput> = None;
    match si {
        ScalarInput::Const(v) => {
            changed |= draw_const(ui, v).changed();
            if let Some(first) = first(graph, ParamUse::Scalar)
                && ui
                    .small_button("ƒ")
                    .on_hover_text("read a parameter instead")
                    .clicked()
            {
                swap_to = Some(ScalarInput::Param(first));
            }
        }
        ScalarInput::Param(name) => {
            if let Some(new) = param_ref(ui, id_source, "", name, graph, ParamUse::Scalar) {
                *name = new;
                changed = true;
            }
            if ui
                .small_button("×")
                .on_hover_text("freeze at the current value")
                .clicked()
            {
                let frozen = graph
                    .param_value(name, ctx)
                    .and_then(|v| v.as_scalar())
                    .unwrap_or(0.5);
                swap_to = Some(ScalarInput::Const(frozen));
            }
        }
        ScalarInput::Layer(_) => {}
    }
    if let Some(next) = swap_to {
        *si = next;
        changed = true;
    }
    changed
}
