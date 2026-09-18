//! Labelled enum combo, shared by the inspector and the graph canvas.

/// `id_salt` must uniquely identify this widget's slot (typically
/// `(layer_id.0, "field")`) so egui's per-widget memory doesn't bleed
/// between two combos with the same visible label.
pub fn enum_combo<T: Copy + PartialEq>(
    ui: &mut egui::Ui,
    id_salt: impl std::hash::Hash + std::fmt::Debug,
    label: &str,
    current: &mut T,
    options: &[T],
    to_label: impl Fn(T) -> &'static str,
) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(label);
        egui::ComboBox::from_id_salt(id_salt)
            .selected_text(to_label(*current))
            .show_ui(ui, |ui| {
                for &opt in options {
                    if ui
                        .selectable_label(*current == opt, to_label(opt))
                        .clicked()
                        && *current != opt
                    {
                        *current = opt;
                        changed = true;
                    }
                }
            });
    });
    changed
}
