//! `ComboBox` for picking another layer. Excludes `except` (typically the
//! layer we're editing) so trivial self-references are unreachable; the
//! graph's cycle detection catches deeper cases.

use texture_graph_core::{Graph, LayerId};

/// Nullable variant of [`layer_ref`]: shows a "(none)" entry above the
/// layer list. `None` inputs render as the missing-texture grid. Returns
/// `Some(new_value)` if the selection changed this frame.
pub fn layer_ref_opt(
    ui: &mut egui::Ui,
    id_source: impl std::hash::Hash + std::fmt::Debug,
    label: &str,
    current: Option<LayerId>,
    graph: &Graph,
    except: Option<LayerId>,
) -> Option<Option<LayerId>> {
    let current_name = match current {
        Some(id) => graph.get(id).map(|l| l.name.as_str()).unwrap_or("<missing>"),
        None => "(none)",
    };
    let mut new_selection: Option<Option<LayerId>> = None;
    egui::ComboBox::from_id_salt(id_source)
        .selected_text(format!("{label}: {current_name}"))
        .show_ui(ui, |ui| {
            let none_selected = current.is_none();
            if ui.selectable_label(none_selected, "(none)").clicked() && !none_selected {
                new_selection = Some(None);
            }
            for l in &graph.layers {
                if Some(l.id) == except {
                    continue;
                }
                let selected = current == Some(l.id);
                if ui.selectable_label(selected, &l.name).clicked() && !selected {
                    new_selection = Some(Some(l.id));
                }
            }
        });
    new_selection
}

/// Show a combo of "layer name" options. Returns `Some(new)` if the
/// selection changed this frame.
pub fn layer_ref(
    ui: &mut egui::Ui,
    id_source: impl std::hash::Hash + std::fmt::Debug,
    label: &str,
    current: LayerId,
    graph: &Graph,
    except: Option<LayerId>,
) -> Option<LayerId> {
    let current_name = graph
        .get(current)
        .map(|l| l.name.as_str())
        .unwrap_or("<missing>");
    let mut new_selection: Option<LayerId> = None;
    egui::ComboBox::from_id_salt(id_source)
        .selected_text(format!("{label}: {current_name}"))
        .show_ui(ui, |ui| {
            for l in &graph.layers {
                if Some(l.id) == except {
                    continue;
                }
                let selected = l.id == current;
                if ui.selectable_label(selected, &l.name).clicked() && !selected {
                    new_selection = Some(l.id);
                }
            }
        });
    new_selection
}
