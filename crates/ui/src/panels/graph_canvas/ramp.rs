//! Logic for the ColorRamp gradient bar: crossing reorders, sorted
//! insertion, duplicate placement, gradient sampling.
//!
//! Free of egui so the reorder behavior is unit-testable; the widget lives
//! in `nodes.rs`.

use texture_graph_core::color::blend;
use texture_graph_core::{
    BlendSpace, Color, ColorInput, ColorStop, EvalCtx, Graph, Sample, evaluate,
};

/// Move `stops[i]` to `t`, bubble-swapping while strictly out of order, so a
/// drag across another stop reorders them. Each returned swap is a
/// transposition for `UiState::remap_ramp_consts`; the comparison is strict
/// so coincident stops don't thrash.
pub fn drag_stop_to(stops: &mut [ColorStop], mut i: usize, t: f32) -> (usize, Vec<(usize, usize)>) {
    let mut swaps = Vec::new();
    stops[i].t = t;
    while i > 0 && stops[i].t < stops[i - 1].t {
        stops.swap(i, i - 1);
        swaps.push((i, i - 1));
        i -= 1;
    }
    while i + 1 < stops.len() && stops[i].t > stops[i + 1].t {
        stops.swap(i, i + 1);
        swaps.push((i, i + 1));
        i += 1;
    }
    (i, swaps)
}

/// Where a stop at `t` keeps `stops` sorted, after any equal `t` so it
/// paints on top.
pub fn insert_index(stops: &[ColorStop], t: f32) -> usize {
    stops.partition_point(|s| s.t <= t)
}

/// How far a duplicate lands from its source, in ramp `t`. About one grab
/// radius at a typical bar width, so the copy is its own target at once.
pub const DUPLICATE_NUDGE: f32 = 0.05;

/// Slack for [`has_room`]. A candidate is exactly one nudge from its
/// source, and `0.55f32 - 0.5f32` is 0.04999995, which reads as a collision.
const DUPLICATE_EPS: f32 = 1e-4;

/// Whether a stop fits at `c`: on the ramp, and a clear nudge from every
/// existing stop. Any less and the two handles share a grab radius.
fn has_room(stops: &[ColorStop], c: f32) -> bool {
    (0.0..=1.0).contains(&c)
        && stops
            .iter()
            .all(|s| (s.t - c).abs() >= DUPLICATE_NUDGE - DUPLICATE_EPS)
}

/// A nudge right, else left, else the middle of the nearest gap with room
/// (nearest rather than widest, so the copy stays by its source). With no
/// room anywhere, returns the source's `t` and accepts the overlap.
pub fn duplicate_t(stops: &[ColorStop], i: usize) -> f32 {
    let t = stops[i].t;
    [t + DUPLICATE_NUDGE, t - DUPLICATE_NUDGE]
        .into_iter()
        .find(|c| has_room(stops, *c))
        .or_else(|| nearest_gap_mid(stops, t))
        .unwrap_or(t)
}

/// Counts the stretches out to 0.0 and 1.0. Ties go to the lower gap.
fn nearest_gap_mid(stops: &[ColorStop], t: f32) -> Option<f32> {
    // Every path that writes stops keeps them sorted, but a loaded graph is
    // whatever the file said.
    let mut ts: Vec<f32> = stops.iter().map(|s| s.t.clamp(0.0, 1.0)).collect();
    ts.push(0.0);
    ts.push(1.0);
    ts.sort_by(f32::total_cmp);
    ts.windows(2)
        .map(|w| (w[0] + w[1]) / 2.0)
        .filter(|c| has_room(stops, *c))
        .min_by(|a, b| (a - t).abs().total_cmp(&(b - t).abs()))
}

/// A wired stop evaluates its source once, at the stop's own `t`.
pub fn stop_display_color(graph: &Graph, stop: &ColorStop, ctx: &EvalCtx) -> Color {
    match &stop.color {
        ColorInput::Const(c) => *c,
        ColorInput::Layer(id) => evaluate(graph, *id, Sample::uv(stop.t, 0.5), ctx),
        ColorInput::Param(name) => graph
            .param_value(name, ctx)
            .and_then(|v| v.as_color())
            // Undeclared, so a hand-built file. Show the missing-texture
            // magenta rather than a plausible gray.
            .unwrap_or_else(|| texture_graph_core::color::oklcha(0.7017, 0.3223, 328.36, 1.0)),
    }
}

/// Mirrors `eval_ramp`'s segment selection, including holding the end
/// colors past either end, so the bar matches evaluation.
pub fn sample_display(display: &[(f32, Color)], space: BlendSpace, u: f32) -> Color {
    match display {
        [] => return texture_graph_core::color::oklcha(0.5, 0.0, 0.0, 1.0),
        [(_, c)] => return *c,
        _ => {}
    }
    let mut lo = 0usize;
    let mut hi = display.len() - 1;
    for i in 0..display.len() - 1 {
        let a = display[i].0;
        let b = display[i + 1].0;
        if u >= a && u <= b {
            lo = i;
            hi = i + 1;
            break;
        }
        if u < a && i == 0 {
            lo = 0;
            hi = 1;
            break;
        }
        if u > b && i == display.len() - 2 {
            lo = display.len() - 2;
            hi = display.len() - 1;
            break;
        }
    }
    let (ta, ca) = display[lo];
    let (tb, cb) = display[hi];
    let span = tb - ta;
    let t = if span.abs() < f32::EPSILON {
        0.0
    } else {
        ((u - ta) / span).clamp(0.0, 1.0)
    };
    blend(ca, cb, t, space)
}

#[cfg(test)]
mod tests {
    use super::*;
    use texture_graph_core::color::oklcha;

    fn stop(t: f32, l: f32) -> ColorStop {
        ColorStop { t, color: ColorInput::Const(oklcha(l, 0.0, 0.0, 1.0)) }
    }

    fn lightness(s: &ColorStop) -> f32 {
        match s.color {
            ColorInput::Const(c) => c.l,
            _ => panic!("const expected"),
        }
    }

    #[test]
    fn drag_without_crossing_keeps_index() {
        let mut stops = vec![stop(0.0, 0.1), stop(0.5, 0.2), stop(1.0, 0.3)];
        let (i, swaps) = drag_stop_to(&mut stops, 1, 0.4);
        assert_eq!(i, 1);
        assert!(swaps.is_empty());
        assert_eq!(stops[1].t, 0.4);
    }

    #[test]
    fn drag_right_across_neighbor_swaps() {
        let mut stops = vec![stop(0.0, 0.1), stop(0.5, 0.2), stop(1.0, 0.3)];
        let (i, swaps) = drag_stop_to(&mut stops, 0, 0.7);
        assert_eq!(i, 1);
        assert_eq!(swaps, vec![(0, 1)]);
        // Colors traveled with their stops.
        assert_eq!(lightness(&stops[0]), 0.2);
        assert_eq!(lightness(&stops[1]), 0.1);
        assert_eq!(stops[0].t, 0.5);
        assert_eq!(stops[1].t, 0.7);
    }

    #[test]
    fn drag_left_across_multiple_in_one_frame() {
        let mut stops = vec![stop(0.1, 0.1), stop(0.4, 0.2), stop(0.6, 0.3), stop(0.9, 0.4)];
        let (i, swaps) = drag_stop_to(&mut stops, 3, 0.0);
        assert_eq!(i, 0);
        assert_eq!(swaps, vec![(3, 2), (2, 1), (1, 0)]);
        assert_eq!(lightness(&stops[0]), 0.4);
        let ts: Vec<f32> = stops.iter().map(|s| s.t).collect();
        assert_eq!(ts, vec![0.0, 0.1, 0.4, 0.6]);
    }

    #[test]
    fn coincident_t_does_not_swap() {
        let mut stops = vec![stop(0.5, 0.1), stop(0.5, 0.2)];
        let (i, swaps) = drag_stop_to(&mut stops, 0, 0.5);
        assert_eq!(i, 0);
        assert!(swaps.is_empty());
    }

    #[test]
    fn clamped_endpoint_drag_reorders() {
        let mut stops = vec![stop(0.2, 0.1), stop(1.0, 0.2)];
        // Caller clamps to 0.0; dragging the right stop to the far left
        // still crosses the left stop.
        let (i, swaps) = drag_stop_to(&mut stops, 1, 0.0);
        assert_eq!(i, 0);
        assert_eq!(swaps, vec![(1, 0)]);
        assert_eq!(lightness(&stops[0]), 0.2);
    }

    #[test]
    fn duplicate_goes_right_when_there_is_room() {
        let stops = vec![stop(0.0, 0.1), stop(0.5, 0.2), stop(1.0, 0.3)];
        assert!((duplicate_t(&stops, 1) - 0.55).abs() < 1e-6);
    }

    #[test]
    fn duplicate_goes_left_when_the_right_is_taken() {
        // 0.55 is occupied, so the copy of the 0.5 stop goes to 0.45.
        let stops = vec![stop(0.5, 0.1), stop(0.55, 0.2)];
        assert!((duplicate_t(&stops, 0) - 0.45).abs() < 1e-6);
        // And the end stop can only go inward — 1.05 is off the ramp.
        let stops = vec![stop(0.0, 0.1), stop(1.0, 0.2)];
        assert!((duplicate_t(&stops, 1) - 0.95).abs() < 1e-6);
    }

    #[test]
    fn duplicate_falls_back_to_the_nearest_roomy_gap_when_boxed_in() {
        // Both slots beside 0.5 are taken and the adjacent gaps are too
        // tight, so the copy goes to the middle of [0.55, 0.9], nearer than
        // the wider [0.0, 0.45].
        let stops = vec![stop(0.45, 0.1), stop(0.5, 0.2), stop(0.55, 0.3), stop(0.9, 0.4)];
        assert!((duplicate_t(&stops, 1) - 0.725).abs() < 1e-6);
    }

    #[test]
    fn duplicate_of_a_packed_ramp_overlaps_rather_than_vanishing() {
        // Nowhere to go: every candidate is crowded and so is every gap.
        let stops: Vec<ColorStop> =
            (0..=100).map(|i| stop(i as f32 / 100.0, 0.5)).collect();
        let t = duplicate_t(&stops, 50);
        assert!((0.0..=1.0).contains(&t), "stayed on the ramp: {t}");
    }

    #[test]
    fn duplicate_lands_where_insert_index_keeps_the_ramp_sorted() {
        let stops = vec![stop(0.0, 0.1), stop(0.5, 0.2), stop(1.0, 0.3)];
        let t = duplicate_t(&stops, 1);
        let mut after = stops.clone();
        after.insert(insert_index(&stops, t), stop(t, 0.2));
        assert!(after.windows(2).all(|w| w[0].t <= w[1].t), "still sorted");
    }

    #[test]
    fn insert_index_orders_and_breaks_ties_after() {
        let stops = vec![stop(0.0, 0.1), stop(0.5, 0.2), stop(1.0, 0.3)];
        assert_eq!(insert_index(&stops, 0.25), 1);
        assert_eq!(insert_index(&stops, 0.5), 2); // after the equal stop
        assert_eq!(insert_index(&stops, 0.75), 2);
        assert_eq!(insert_index(&stops, 1.5), 3);
    }

    #[test]
    fn sample_display_interpolates_and_holds_ends() {
        let display = vec![(0.3, oklcha(0.2, 0.0, 0.0, 1.0)), (0.7, oklcha(0.8, 0.0, 0.0, 1.0))];
        let mid = sample_display(&display, BlendSpace::Oklch, 0.5);
        assert!((mid.l - 0.5).abs() < 1e-4);
        // Outside the outermost stops the ramp holds their colors flat.
        assert!((sample_display(&display, BlendSpace::Oklch, 0.0).l - 0.2).abs() < 1e-6);
        assert!((sample_display(&display, BlendSpace::Oklch, 1.0).l - 0.8).abs() < 1e-6);
        // Degenerate inputs don't panic.
        let one = vec![(0.5, oklcha(0.3, 0.0, 0.0, 1.0))];
        assert!((sample_display(&one, BlendSpace::Oklch, 0.9).l - 0.3).abs() < 1e-6);
        sample_display(&[], BlendSpace::Oklch, 0.5);
    }
}
