//! The ColorRamp stop position field, shared by the inspector and the graph canvas.

/// Two decimals always: a ramp reads as a column of positions, and digits
/// appearing and vanishing make the column jump while you drag. Rounds the
/// display, not the stored `t`.
pub fn stop_t(ui: &mut egui::Ui, t: &mut f32) -> egui::Response {
    ui.add(
        egui::DragValue::new(t)
            .speed(0.01)
            .range(0.0..=1.0)
            .fixed_decimals(2),
    )
}
