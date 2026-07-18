//! `ComboBox` for picking another layer. Excludes `except` (typically the
//! layer we're editing) so trivial self-references are unreachable; the
//! graph's cycle detection catches deeper cases.

use texture_graph_core::{Graph, LayerId};

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
