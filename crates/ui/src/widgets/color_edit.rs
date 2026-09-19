//! Oklch color editor: an sRGB swatch button (egui's default picker) with
//! an "LCh…" popover for authoring out-of-sRGB-gamut Oklch values that
//! the sRGB picker can't reach.

use texture_graph_core::Color;

use crate::color_convert::{oklcha_to_srgba, srgba_to_oklcha};

/// Whether the color changed.
pub fn oklcha_edit(ui: &mut egui::Ui, color: &mut Color) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        let mut rgba = oklcha_to_srgba(*color);
        if ui.color_edit_button_rgba_unmultiplied(&mut rgba).changed() {
            *color = srgba_to_oklcha(rgba);
            changed = true;
        }
        // The escape hatch for colors the sRGB picker can't reach.
        let btn = ui.small_button("LCh…");
        egui::Popup::from_toggle_button_response(&btn)
            .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
            .show(|ui| {
                ui.set_min_width(240.0);
                let mut l = color.l;
                let mut c = color.chroma;
                let mut h = color.hue.into_degrees();
                let mut a = color.alpha;
                let mut any = false;
                any |= ui.add(egui::Slider::new(&mut l, -0.2..=1.2).text("L")).changed();
                any |= ui
                    .add(egui::Slider::new(&mut c, 0.0..=0.5).text("C (chroma)"))
                    .changed();
                any |= ui
                    .add(egui::Slider::new(&mut h, 0.0..=360.0).text("h (deg)"))
                    .changed();
                any |= ui.add(egui::Slider::new(&mut a, 0.0..=1.0).text("alpha")).changed();
                if any {
                    *color = Color::new(l, c, h, a);
                    changed = true;
                }
            });
    });
    changed
}
