//! The graph's named parameters: what it declares, and what this session
//! has them bound to.
//!
//! A declaration travels with the graph file. A binding does not: it stands
//! in for one instance the consumer will bake, so the author can see what a
//! parameter does without running the game.
//!
//! Declarations go through [`EditCmd`] like every other graph edit;
//! bindings write straight to the [`EvalCtx`], because the file does not
//! change.

use texture_graph_core::color::oklcha;
use texture_graph_core::{EvalCtx, Graph, ParamDecl, ParamKind, ParamValue};

use crate::state::{EditCmd, UiState};
use crate::widgets::color_edit;

pub fn show(ui: &mut egui::Ui, graph: &Graph, state: &mut UiState, eval_ctx: &mut EvalCtx) {
    ui.heading("Parameters");
    ui.label(
        egui::RichText::new(
            "Declared here, bound per bake. One graph, N instances.",
        )
        .weak()
        .small(),
    );
    ui.separator();

    if graph.params.is_empty() {
        ui.label(egui::RichText::new("none declared").weak());
    }

    // Collected first: the loop below pushes edits that would otherwise
    // need a mutable borrow of the map it is walking.
    let decls: Vec<ParamDecl> = graph.params.values().cloned().collect();
    for decl in &decls {
        param_row(ui, graph, state, eval_ctx, decl);
    }

    ui.separator();
    ui.horizontal(|ui| {
        if ui.button("+ scalar").clicked() {
            state.push(EditCmd::DeclareParam(ParamDecl::scalar(
                unique_name(graph, "scalar"),
                0.0,
                1.0,
                0.5,
            )));
        }
        if ui.button("+ color").clicked() {
            state.push(EditCmd::DeclareParam(ParamDecl::color(
                unique_name(graph, "color"),
                oklcha(0.5, 0.0, 0.0, 1.0),
            )));
        }
    });
}

fn param_row(
    ui: &mut egui::Ui,
    graph: &Graph,
    state: &mut UiState,
    eval_ctx: &mut EvalCtx,
    decl: &ParamDecl,
) {
    let readers = graph.param_readers(&decl.name).count();
    egui::CollapsingHeader::new(format!("{} ({})", decl.name, decl.kind.label()))
        .id_salt(("param", &decl.name))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label("value");
                let bound = graph.param_value(&decl.name, eval_ctx);
                match (decl.kind, bound) {
                    (ParamKind::Scalar { min, max }, Some(ParamValue::Scalar(v))) => {
                        let mut v = v;
                        let range = if max > min { min..=max } else { 0.0..=1.0 };
                        if ui.add(egui::Slider::new(&mut v, range)).changed() {
                            eval_ctx
                                .params
                                .insert(decl.name.clone(), ParamValue::Scalar(v));
                        }
                    }
                    (ParamKind::Color, Some(ParamValue::Color(c))) => {
                        let mut c = c;
                        if color_edit::oklcha_edit(ui, &mut c) {
                            eval_ctx
                                .params
                                .insert(decl.name.clone(), ParamValue::Color(c));
                        }
                    }
                    _ => {}
                }
                // Present only while something overrides the default, so
                // the button's absence means "this is the default".
                if eval_ctx.params.contains_key(&decl.name)
                    && ui.small_button("reset").clicked()
                {
                    eval_ctx.params.remove(&decl.name);
                }
            });

            let mut edited = decl.clone();
            let mut changed = false;
            ui.horizontal(|ui| {
                ui.label("name");
                let mut name = decl.name.clone();
                if ui.text_edit_singleline(&mut name).lost_focus() && name != decl.name {
                    // Renaming rewrites every socket that reads it, which
                    // is why it is its own command rather than a decl swap.
                    state.push(EditCmd::RenameParam {
                        from: decl.name.clone(),
                        to: name,
                    });
                }
            });
            ui.horizontal(|ui| {
                ui.label("default");
                match (&mut edited.kind, &mut edited.default) {
                    (ParamKind::Scalar { min, max }, ParamValue::Scalar(v)) => {
                        let range = if *max > *min { *min..=*max } else { 0.0..=1.0 };
                        changed |= ui.add(egui::Slider::new(v, range)).changed();
                    }
                    (ParamKind::Color, ParamValue::Color(c)) => {
                        changed |= color_edit::oklcha_edit(ui, c);
                    }
                    // A decl whose default disagrees with its kind is
                    // rejected at declaration, so this is unreachable.
                    _ => {}
                }
            });
            if let ParamKind::Scalar { min, max } = &mut edited.kind {
                ui.horizontal(|ui| {
                    ui.label("range");
                    changed |= ui.add(egui::DragValue::new(min).speed(0.01)).changed();
                    changed |= ui.add(egui::DragValue::new(max).speed(0.01)).changed();
                });
            }
            ui.horizontal(|ui| {
                ui.label("note");
                let mut text = edited.description.clone().unwrap_or_default();
                if ui.text_edit_singleline(&mut text).changed() {
                    edited.description = (!text.is_empty()).then_some(text);
                    changed = true;
                }
            });
            if changed {
                state.push(EditCmd::SetParamDecl(decl.name.clone(), edited));
            }

            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new(match readers {
                        0 => "read by nothing".to_string(),
                        1 => "read by 1 socket".to_string(),
                        n => format!("read by {n} sockets"),
                    })
                    .weak()
                    .small(),
                );
                // Removing freezes every reader at this default, so the
                // graph keeps rendering the same picture.
                if ui.small_button("remove").clicked() {
                    state.push(EditCmd::RemoveParam(decl.name.clone()));
                }
            });
        });
}

/// `stem`, `stem 1`, `stem 2`, … as for layer names.
fn unique_name(graph: &Graph, stem: &str) -> String {
    if !graph.params.contains_key(stem) {
        return stem.to_string();
    }
    (1..).map(|n| format!("{stem} {n}")).find(|c| !graph.params.contains_key(c)).unwrap()
}
