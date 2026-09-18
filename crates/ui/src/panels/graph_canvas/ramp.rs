//! Pure logic for the ColorRamp gradient-bar editor: stop dragging with
//! crossing reorders, sorted insertion, and gradient sampling for display.
//!
//! Kept free of egui so the reorder/swap behavior is unit-testable. The
//! widget itself lives in `nodes.rs`.

use texture_graph_core::color::blend;
use texture_graph_core::{
    BlendSpace, Color, ColorInput, ColorStop, EvalCtx, Graph, Sample, evaluate,
};

/// Move `stops[i]` to position `t`, then bubble-swap it while strictly out
/// of order with its neighbors — a drag that crosses other stops reorders
/// them, Blender-style. Returns `(new_index, swaps)`; each swap `(a, b)` is
/// a transposition to feed `UiState::remap_ramp_consts`. Strict comparisons
/// mean coincident-t stops don't thrash.
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

/// Index at which a new stop at `t` keeps `stops` sorted (after any equal
/// t, so the new stop paints on top).
pub fn insert_index(stops: &[ColorStop], t: f32) -> usize {
    stops.partition_point(|s| s.t <= t)
}

/// Display color of one stop: a `Const` passes through; a wired stop
/// evaluates its source once at the stop's own position along the ramp.
pub fn stop_display_color(graph: &Graph, stop: &ColorStop, ctx: &EvalCtx) -> Color {
    match stop.color {
        ColorInput::Const(c) => c,
        ColorInput::Layer(id) => evaluate(graph, id, Sample::uv(stop.t, 0.5), ctx),
    }
}

/// Ramp color at `u` over pre-resolved `(t, color)` pairs — mirrors
/// `eval_ramp`'s segment selection (holds the outermost stop's color past
/// either end) so the bar shows exactly what evaluation produces.
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
        // Drag stop 0 to 0.7: crosses stop at 0.5.
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
        // Drag the last stop all the way to 0.0 — crosses everything.
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
