//! Rate limit for rebakes driven by edits.
//!
//! A drag edits the graph every frame, and every edit makes the previews
//! stale. Baking on each of those frames is what made dragging stutter, so a
//! stale product rebakes at most once per [`BAKE_INTERVAL`]: the first edit
//! bakes at once, later ones wait out the interval, and a repaint is booked
//! for the end of the wait so the last edit of a drag still lands after the
//! pointer stops moving.
//!
//! Time comes from egui's input clock rather than `Instant`, which panics
//! on wasm.

/// Seconds between edit-driven bakes: ten a second.
pub const BAKE_INTERVAL: f64 = 0.1;

#[derive(Debug, Default)]
pub struct Throttle {
    last: Option<f64>,
}

impl Throttle {
    /// Whether a bake may run now. If it may, the caller is taken to have
    /// baked and the interval restarts. If not, a repaint is booked for when
    /// it may, so the caller can leave its product stale without the result
    /// waiting on the next input event.
    pub fn allow(&mut self, ctx: &egui::Context) -> bool {
        let now = ctx.input(|i| i.time);
        let wait = self.last.map_or(0.0, |t| t + BAKE_INTERVAL - now);
        if wait <= 0.0 {
            self.last = Some(now);
            true
        } else {
            ctx.request_repaint_after(std::time::Duration::from_secs_f64(wait));
            false
        }
    }

    /// Record a bake that ran regardless of the throttle, so edits right
    /// after it wait their turn too.
    pub fn mark(&mut self, ctx: &egui::Context) {
        self.last = Some(ctx.input(|i| i.time));
    }
}
