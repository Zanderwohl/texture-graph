//! Rate limit for rebakes driven by edits.
//!
//! A drag edits the graph every frame, and baking every frame makes dragging
//! stutter. A stale product rebakes at most once per [`BAKE_INTERVAL`], and a
//! repaint is booked for the end of the wait so the last edit of a drag still
//! bakes after the pointer stops.
//!
//! Time comes from egui's input clock because `Instant` panics on wasm.

/// Seconds between edit-driven bakes.
pub const BAKE_INTERVAL: f64 = 0.1;

#[derive(Debug, Default)]
pub struct Throttle {
    last: Option<f64>,
    /// Requests turned away since the last bake, for the log.
    deferred: u32,
}

impl Throttle {
    /// If true, the caller must bake now and the interval restarts. If false,
    /// a repaint is booked for when a bake may run.
    pub fn allow(&mut self, ctx: &egui::Context) -> bool {
        let now = ctx.input(|i| i.time);
        let wait = self.last.map_or(0.0, |t| t + BAKE_INTERVAL - now);
        if wait <= 0.0 {
            self.last = Some(now);
            if self.deferred > 0 {
                log::debug!("throttle release coalesced={}", self.deferred);
                self.deferred = 0;
            }
            true
        } else {
            self.deferred += 1;
            log::trace!("throttle defer wait_ms={:.0}", wait * 1000.0);
            ctx.request_repaint_after(std::time::Duration::from_secs_f64(wait));
            false
        }
    }

    /// Record a bake that ran without asking [`Self::allow`].
    pub fn mark(&mut self, ctx: &egui::Context) {
        self.last = Some(ctx.input(|i| i.time));
        if self.deferred > 0 {
            log::debug!("throttle bypass coalesced={}", self.deferred);
            self.deferred = 0;
        }
    }
}
