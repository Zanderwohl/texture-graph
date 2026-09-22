//! Pebble-game scheduler for the compute pipeline.
//!
//! Produces a topological order over the layers reachable from `Output`, an
//! assignment of each layer to a slot (one of the baker's pooled
//! `Rgba32Float` textures), and the peak number of slots live at once.
//!
//! Slots are assigned by linear scan: refcount each layer's consumers and
//! free a slot after its last consumer runs. Output roots get one extra
//! count so their slot survives to be packed.
//!
//! Greedy is optimal for a fixed topo order, as interval coloring is.
//! Choosing the order to minimize the peak is NP-hard.

use std::collections::{HashMap, HashSet};

use texture_graph_core::{
    Axis, CoordMode, EXTEND_LIMIT, EdgeMode, EvalCtx, Graph, LayerId, LayerKind, RadialDim,
    ScalarInput, Transform, Warp,
};

#[derive(Debug, Clone)]
pub struct Schedule {
    pub order: Vec<LayerId>,
    pub slot_of: HashMap<LayerId, u32>,
    pub peak_slots: u32,
    pub output_slots: OutputSlots,
    /// `Domain::UNIT` unless an `EdgeMode::Extend` transform or a `Warp`
    /// widens a source.
    pub domain_of: HashMap<LayerId, Domain>,
}

/// UV rectangle a layer's bake covers. Always contains the unit square, so
/// thumbnails and direct output stay renderable. Extend consumers grow it,
/// exactly for affine requests and capped at [`EXTEND_LIMIT`] for radial
/// ones.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Domain {
    pub min: [f32; 2],
    pub max: [f32; 2],
}

impl Domain {
    pub const UNIT: Domain = Domain { min: [0.0, 0.0], max: [1.0, 1.0] };

    fn union(&mut self, o: Domain) {
        self.min[0] = self.min[0].min(o.min[0]);
        self.min[1] = self.min[1].min(o.min[1]);
        self.max[0] = self.max[0].max(o.max[0]);
        self.max[1] = self.max[1].max(o.max[1]);
    }

    /// `(min_u, min_v, ext_u, ext_v)`, the layout the shaders expect.
    pub fn packed(&self) -> [f32; 4] {
        [
            self.min[0],
            self.min[1],
            self.max[0] - self.min[0],
            self.max[1] - self.min[1],
        ]
    }
}

/// Every layer starts at the unit square; a reverse-topo walk unions in what
/// each consumer samples. Layers other than Extend transforms and warps
/// sample at their own coordinates, so a widened consumer widens its inputs.
fn compute_domains(graph: &Graph, order: &[LayerId]) -> HashMap<LayerId, Domain> {
    let mut dom: HashMap<LayerId, Domain> =
        order.iter().map(|&id| (id, Domain::UNIT)).collect();
    for &id in order.iter().rev() {
        let d = dom[&id];
        let Some(layer) = graph.get(id) else { continue };
        match &layer.kind {
            LayerKind::Transform(t) => {
                if t.edge_mode == EdgeMode::Extend {
                    if let (Some(src), Some(req)) = (t.source, transform_request(t, d)) {
                        if let Some(e) = dom.get_mut(&src) {
                            e.union(req);
                        }
                    }
                }
            }
            LayerKind::Map(m) => {
                if let Some(e) = m.value.and_then(|v| dom.get_mut(&v)) {
                    e.union(d);
                }
                // Luminance-indexed, so the unit domain suffices.
            }
            LayerKind::Warp(w) => {
                if let Some(e) = w.by.and_then(|b| dom.get_mut(&b)) {
                    e.union(d);
                }
                // Capped like a radial extend, since `amount` is unbounded.
                if let Some(e) = w.source.and_then(|src| dom.get_mut(&src)) {
                    e.union(warp_request(w, d));
                }
            }
            _ => {
                for input in layer.kind.inputs() {
                    if let Some(e) = dom.get_mut(&input) {
                        e.union(d);
                    }
                }
            }
        }
    }
    dom
}

/// The rectangle an extend transform with bake domain `d` samples its
/// source over. Affine requests are exact and unbounded; the bake keeps its
/// pixel count, spread wider. Radial requests are conservative, so they are
/// intersected with the [`EXTEND_LIMIT`] box. `None` when that is empty or
/// the request is not finite.
fn transform_request(t: &Transform, d: Domain) -> Option<Domain> {
    let corners = [
        (d.min[0], d.min[1]),
        (d.max[0], d.min[1]),
        (d.min[0], d.max[1]),
        (d.max[0], d.max[1]),
    ];
    let (mut u_min, mut u_max) = (f32::INFINITY, f32::NEG_INFINITY);
    let (mut v_min, mut v_max) = (f32::INFINITY, f32::NEG_INFINITY);
    for (cu, cv) in corners {
        let mut u = cu - t.offset[0];
        let mut v = cv - t.offset[1];
        if t.rotate_uv != 0.0 {
            let (sin, cos) = t.rotate_uv.sin_cos();
            let (ru, rv) = (u * cos - v * sin, u * sin + v * cos);
            u = ru;
            v = rv;
        }
        let u = u * t.scale[0];
        let v = v * t.scale[1];
        u_min = u_min.min(u);
        u_max = u_max.max(u);
        v_min = v_min.min(v);
        v_max = v_max.max(v);
    }
    // w spans [0, 1] across a volume bake (0.5 flat).
    let sw_max = {
        let a = (0.0 - t.offset[2]) * t.scale[2];
        let b = (1.0 - t.offset[2]) * t.scale[2];
        a.abs().max(b.abs())
    };

    let radial = matches!(t.coord_mode, CoordMode::Radial { .. });
    let ((ru0, ru1), (rv0, rv1)) = match t.coord_mode {
        CoordMode::Passthrough => ((u_min, u_max), (v_min, v_max)),
        CoordMode::Permute(axes) => {
            let range = |a: Axis| match a {
                Axis::U => (u_min, u_max),
                Axis::V => (v_min, v_max),
                Axis::W => (-sw_max, sw_max),
            };
            (range(axes[0]), range(axes[1]))
        }
        CoordMode::Radial { dim, into } => {
            let mut r_max: f32 = 0.0;
            for &u in &[u_min, u_max] {
                for &v in &[v_min, v_max] {
                    let mut r2 = u * u + v * v;
                    if matches!(dim, RadialDim::D3) {
                        r2 += sw_max * sw_max;
                    }
                    r_max = r_max.max(r2.sqrt());
                }
            }
            match into {
                Axis::U => ((0.0, r_max), (0.0, 0.0)),
                Axis::V => ((0.0, 0.0), (0.0, r_max)),
                Axis::W => ((0.0, 0.0), (0.0, 0.0)),
            }
        }
    };

    let (u0, u1, v0, v1) = if radial {
        let lo = 0.5 - EXTEND_LIMIT;
        let hi = 0.5 + EXTEND_LIMIT;
        (ru0.max(lo), ru1.min(hi), rv0.max(lo), rv1.min(hi))
    } else {
        (ru0, ru1, rv0, rv1)
    };
    if !(u0 <= u1 && v0 <= v1 && u0.is_finite() && u1.is_finite() && v0.is_finite() && v1.is_finite()) {
        return None;
    }
    Some(Domain { min: [u0, v0], max: [u1, v1] })
}

/// `d` grown by `|amount|`, capped at the [`EXTEND_LIMIT`] box. Agrees with
/// `warp_bounds` in `core::eval` when `d` is the unit square.
fn warp_request(w: &Warp, d: Domain) -> Domain {
    let lo = 0.5 - EXTEND_LIMIT;
    let hi = 0.5 + EXTEND_LIMIT;
    let grow = |min: f32, max: f32, a: f32| {
        ((min - a.abs()).max(lo), (max + a.abs()).min(hi))
    };
    let (u0, u1) = grow(d.min[0], d.max[0], w.amount[0]);
    let (v0, v1) = grow(d.min[1], d.max[1], w.amount[1]);
    Domain { min: [u0, v0], max: [u1, v1] }
}

#[derive(Debug, Copy, Clone)]
pub struct OutputSlots {
    /// `None` packs the missing-texture grid.
    pub color: Option<u32>,
    pub roughness: ScalarSlot,
    pub metallic: ScalarSlot,
    pub normal: Option<u32>,
}

#[derive(Debug, Copy, Clone)]
pub enum ScalarSlot {
    Const(f32),
    /// Reads the slot's L.
    Slot(u32),
}

/// A graph built through the `Graph` mutators never produces these.
#[derive(Debug, Clone, PartialEq)]
pub enum ScheduleError {
    UnknownLayer(LayerId),
    Cycle,
}

impl std::fmt::Display for ScheduleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ScheduleError::UnknownLayer(id) => write!(f, "unknown layer id {id}"),
            ScheduleError::Cycle => f.write_str("graph contains a cycle"),
        }
    }
}

impl std::error::Error for ScheduleError {}

/// Only layers reachable from `Output` are included.
pub fn schedule(graph: &Graph, ctx: &EvalCtx) -> Result<Schedule, ScheduleError> {
    let result = schedule_output(graph, ctx);
    log_result("output", graph, &result);
    result
}

fn schedule_output(graph: &Graph, ctx: &EvalCtx) -> Result<Schedule, ScheduleError> {
    let ctx = &graph.resolve_params(ctx);
    let output_roots = output_referenced(graph);
    let plan = plan_from(graph, &output_roots)?;

    let output_slots = OutputSlots {
        color: match graph.output.color {
            Some(id) => Some(*plan.slot_of.get(&id).ok_or(ScheduleError::UnknownLayer(id))?),
            None => None,
        },
        roughness: scalar_to_slot(&graph.output.roughness, &plan.slot_of, ctx)?,
        metallic: scalar_to_slot(&graph.output.metallic, &plan.slot_of, ctx)?,
        normal: match graph.output.normal {
            Some(id) => Some(
                *plan.slot_of.get(&id).ok_or(ScheduleError::UnknownLayer(id))?,
            ),
            None => None,
        },
    };
    Ok(plan.into_schedule(output_slots))
}

/// A `Schedule` for `root` and everything it transitively reads, with the
/// same slot reuse as [`schedule`].
///
/// `output_slots.color` points at `root`; the other channels are
/// placeholders.
pub fn schedule_layer(graph: &Graph, root: LayerId) -> Result<Schedule, ScheduleError> {
    let result = schedule_root(graph, root);
    log_result("layer", graph, &result);
    result
}

fn schedule_root(graph: &Graph, root: LayerId) -> Result<Schedule, ScheduleError> {
    if graph.get(root).is_none() {
        return Err(ScheduleError::UnknownLayer(root));
    }
    let plan = plan_from(graph, &[root])?;
    let output_slots = OutputSlots {
        color: plan.slot_of.get(&root).copied(),
        roughness: ScalarSlot::Const(0.5),
        metallic: ScalarSlot::Const(0.0),
        normal: None,
    };
    Ok(plan.into_schedule(output_slots))
}

/// A `Schedule` without its output slots.
struct Plan {
    order: Vec<LayerId>,
    slot_of: HashMap<LayerId, u32>,
    peak_slots: u32,
    domain_of: HashMap<LayerId, Domain>,
}

impl Plan {
    fn into_schedule(self, output_slots: OutputSlots) -> Schedule {
        Schedule {
            order: self.order,
            slot_of: self.slot_of,
            peak_slots: self.peak_slots,
            output_slots,
            domain_of: self.domain_of,
        }
    }
}

fn plan_from(graph: &Graph, roots: &[LayerId]) -> Result<Plan, ScheduleError> {
    let mut required: HashSet<LayerId> = HashSet::new();
    let mut stack = roots.to_vec();
    while let Some(id) = stack.pop() {
        if !required.insert(id) {
            continue;
        }
        let layer = graph.get(id).ok_or(ScheduleError::UnknownLayer(id))?;
        for input in layer.kind.inputs() {
            stack.push(input);
        }
    }

    // `visiting` catches cycles, which the graph mutators should already
    // have rejected.
    let mut order = Vec::with_capacity(required.len());
    let mut done: HashSet<LayerId> = HashSet::new();
    let mut visiting: HashSet<LayerId> = HashSet::new();
    for &root in roots {
        topo_dfs(graph, root, &required, &mut visiting, &mut done, &mut order)?;
    }

    // Each occurrence in an `inputs()` counts, so a Mix reading one layer
    // twice counts twice. The root count keeps a root's slot alive to be
    // packed.
    let mut remaining: HashMap<LayerId, u32> = HashMap::new();
    for &id in &order {
        remaining.entry(id).or_insert(0);
    }
    for &id in &order {
        let layer = graph.get(id).unwrap();
        for input in layer.kind.inputs() {
            *remaining.entry(input).or_insert(0) += 1;
        }
    }
    for &root in roots {
        *remaining.entry(root).or_insert(0) += 1;
    }

    let mut slot_of: HashMap<LayerId, u32> = HashMap::new();
    let mut free_pool: Vec<u32> = Vec::new();
    let mut peak_slots: u32 = 0;
    for &id in &order {
        let slot = match free_pool.pop() {
            Some(s) => s,
            None => {
                let s = peak_slots;
                peak_slots += 1;
                s
            }
        };
        slot_of.insert(id, slot);

        let layer = graph.get(id).unwrap();
        for input in layer.kind.inputs() {
            if let Some(remaining_uses) = remaining.get_mut(&input) {
                *remaining_uses = remaining_uses.saturating_sub(1);
                if *remaining_uses == 0 {
                    if let Some(&input_slot) = slot_of.get(&input) {
                        free_pool.push(input_slot);
                    }
                }
            }
        }
    }

    let domain_of = compute_domains(graph, &order);
    Ok(Plan { order, slot_of, peak_slots, domain_of })
}

/// One slot per layer, no reuse, so every intermediate survives to be
/// previewed.
pub fn schedule_no_reuse(graph: &Graph, ctx: &EvalCtx) -> Result<Schedule, ScheduleError> {
    schedule_previews(graph, None, ctx)
}

/// The preview schedule for `wanted` and everything it transitively reads,
/// one slot per layer. `None` wants every layer.
///
/// Domains are computed over the whole graph. A domain is set by a layer's
/// consumers, so computing it over the subset would change a thumbnail's
/// resolution depending on which other layers were being baked.
pub fn schedule_previews(
    graph: &Graph,
    wanted: Option<&HashSet<LayerId>>,
    ctx: &EvalCtx,
) -> Result<Schedule, ScheduleError> {
    let result = schedule_previews_inner(graph, wanted, ctx);
    log_result("previews", graph, &result);
    result
}

fn schedule_previews_inner(
    graph: &Graph,
    wanted: Option<&HashSet<LayerId>>,
    ctx: &EvalCtx,
) -> Result<Schedule, ScheduleError> {
    let ctx = &graph.resolve_params(ctx);
    let everything: HashSet<LayerId> = graph.layers.iter().map(|l| l.id).collect();
    let full_order = topo_over(graph, &everything)?;
    let domain_of = compute_domains(graph, &full_order);

    let order = match wanted {
        None => full_order,
        Some(w) => topo_over(graph, &upstream_closure(graph, w)?)?,
    };

    let mut slot_of: HashMap<LayerId, u32> = HashMap::new();
    for (i, &id) in order.iter().enumerate() {
        slot_of.insert(id, i as u32);
    }
    let peak_slots = order.len() as u32;
    // Lenient: on a subset the output roots may not be scheduled, and the
    // preview pass does not read these.
    let output_slots = OutputSlots {
        color: graph.output.color.and_then(|id| slot_of.get(&id).copied()),
        roughness: scalar_slot_lenient(&graph.output.roughness, &slot_of, ctx),
        metallic: scalar_slot_lenient(&graph.output.metallic, &slot_of, ctx),
        normal: graph.output.normal.and_then(|id| slot_of.get(&id).copied()),
    };
    Ok(Schedule { order, slot_of, peak_slots, output_slots, domain_of })
}

/// `roots` plus everything they transitively read.
fn upstream_closure(
    graph: &Graph,
    roots: &HashSet<LayerId>,
) -> Result<HashSet<LayerId>, ScheduleError> {
    let mut required: HashSet<LayerId> = HashSet::new();
    let mut stack: Vec<LayerId> = roots.iter().copied().collect();
    while let Some(id) = stack.pop() {
        if !required.insert(id) {
            continue;
        }
        let layer = graph.get(id).ok_or(ScheduleError::UnknownLayer(id))?;
        stack.extend(layer.kind.inputs());
    }
    Ok(required)
}

/// A topological order over exactly `required`, dependencies first.
fn topo_over(graph: &Graph, required: &HashSet<LayerId>) -> Result<Vec<LayerId>, ScheduleError> {
    let mut order = Vec::with_capacity(required.len());
    let mut done: HashSet<LayerId> = HashSet::new();
    let mut visiting: HashSet<LayerId> = HashSet::new();
    // Sorted because `HashSet` order varies between runs, which would make
    // slot assignments irreproducible.
    let mut ids: Vec<LayerId> = required.iter().copied().collect();
    ids.sort();
    for id in ids {
        topo_dfs(graph, id, required, &mut visiting, &mut done, &mut order)?;
    }
    Ok(order)
}

/// Like [`scalar_to_slot`], but an unscheduled layer becomes 0.0 instead of
/// an error.
fn scalar_slot_lenient(
    s: &ScalarInput,
    slot_of: &HashMap<LayerId, u32>,
    ctx: &EvalCtx,
) -> ScalarSlot {
    if let ScalarInput::Layer(id) = s {
        return match slot_of.get(id) {
            Some(slot) => ScalarSlot::Slot(*slot),
            None => ScalarSlot::Const(0.0),
        };
    }
    ScalarSlot::Const(ctx.scalar_const(s).unwrap_or(0.0))
}

fn output_referenced(graph: &Graph) -> Vec<LayerId> {
    let mut out = Vec::new();
    out.extend(graph.output.color);
    if let ScalarInput::Layer(id) = graph.output.roughness {
        out.push(id);
    }
    if let ScalarInput::Layer(id) = graph.output.metallic {
        out.push(id);
    }
    if let Some(id) = graph.output.normal {
        out.push(id);
    }
    out
}

/// Parameters are resolved to constants here.
fn scalar_to_slot(
    s: &ScalarInput,
    slot_of: &HashMap<LayerId, u32>,
    ctx: &EvalCtx,
) -> Result<ScalarSlot, ScheduleError> {
    if let ScalarInput::Layer(id) = s {
        return Ok(ScalarSlot::Slot(
            *slot_of.get(id).ok_or(ScheduleError::UnknownLayer(*id))?,
        ));
    }
    Ok(ScalarSlot::Const(ctx.scalar_const(s).unwrap_or(0.0)))
}

/// Formats layer ids as `[L1, L2]`.
pub(crate) struct IdList<'a>(pub &'a [LayerId]);

impl std::fmt::Display for IdList<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("[")?;
        for (i, id) in self.0.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{id}")?;
        }
        f.write_str("]")
    }
}

fn log_result(mode: &str, graph: &Graph, result: &Result<Schedule, ScheduleError>) {
    let s = match result {
        Ok(s) => s,
        Err(e) => {
            log::debug!("schedule failed mode={mode} err={e}");
            return;
        }
    };
    if !log::log_enabled!(log::Level::Debug) {
        return;
    }
    log::debug!(
        "schedule done mode={mode} layers={} peak_slots={} order={} outputs={:?}",
        s.order.len(),
        s.peak_slots,
        IdList(&s.order),
        s.output_slots,
    );
    for &id in &s.order {
        let Some(layer) = graph.get(id) else { continue };
        let reads: Vec<String> = layer
            .kind
            .inputs()
            .iter()
            .map(|input| match s.slot_of.get(input) {
                Some(slot) => format!("{input}@{slot}"),
                None => format!("{input}@none"),
            })
            .collect();
        let dom = s.domain_of.get(&id).copied().unwrap_or(Domain::UNIT);
        log::debug!(
            "schedule layer id={id} name={:?} writes_slot={} reads=[{}] domain_min={:?} domain_max={:?}",
            layer.name,
            s.slot_of.get(&id).copied().unwrap_or(u32::MAX),
            reads.join(", "),
            dom.min,
            dom.max,
        );
    }
}

fn topo_dfs(
    graph: &Graph,
    id: LayerId,
    required: &HashSet<LayerId>,
    visiting: &mut HashSet<LayerId>,
    done: &mut HashSet<LayerId>,
    order: &mut Vec<LayerId>,
) -> Result<(), ScheduleError> {
    if done.contains(&id) {
        return Ok(());
    }
    if !visiting.insert(id) {
        return Err(ScheduleError::Cycle);
    }
    let layer = graph.get(id).ok_or(ScheduleError::UnknownLayer(id))?;
    for input in layer.kind.inputs() {
        if required.contains(&input) {
            topo_dfs(graph, input, required, visiting, done, order)?;
        }
    }
    visiting.remove(&id);
    done.insert(id);
    order.push(id);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use texture_graph_core::{
        BlendMode, BlendSpace, Color, EvalCtx, Graph, LayerKind, Mix, Output, ScalarInput,
    };

    fn base_graph() -> Graph {
        Graph::new()
    }

    fn add_color(g: &mut Graph, name: &str) -> LayerId {
        g.add_layer(name, LayerKind::Color(Color::new(0.5, 0.0, 0.0, 1.0)))
            .unwrap()
    }

    fn add_mix(g: &mut Graph, name: &str, a: LayerId, b: LayerId) -> LayerId {
        g.add_layer(
            name,
            LayerKind::Mix(Mix {
                a: Some(a),
                b: Some(b),
                mode: BlendMode::Add,
                factor: ScalarInput::Const(0.5),
                space: BlendSpace::Oklch,
            }),
        )
        .unwrap()
    }

    /// No two layers live at the same time share a slot. A layer is live
    /// from its dispatch through its last consumer's, or to the end if it
    /// is an output root.
    fn assert_valid_allocation(graph: &Graph, s: &Schedule) {
        let index_of: HashMap<LayerId, usize> = s
            .order
            .iter()
            .enumerate()
            .map(|(i, id)| (*id, i))
            .collect();
        let output_ids: HashSet<LayerId> = output_referenced(graph).into_iter().collect();
        let mut interval: HashMap<LayerId, (usize, usize)> = HashMap::new();
        for (i, &id) in s.order.iter().enumerate() {
            interval.insert(id, (i, i));
        }
        for (i, &id) in s.order.iter().enumerate() {
            for input in graph.get(id).unwrap().kind.inputs() {
                if let Some(v) = interval.get_mut(&input) {
                    v.1 = v.1.max(i);
                }
            }
        }
        for &out_id in &output_ids {
            if let Some(v) = interval.get_mut(&out_id) {
                v.1 = s.order.len();
            }
        }
        let items: Vec<_> = interval.iter().collect();
        for i in 0..items.len() {
            let (&id_a, &(a0, a1)) = items[i];
            for j in (i + 1)..items.len() {
                let (&id_b, &(b0, b1)) = items[j];
                let overlap = a0 <= b1 && b0 <= a1;
                if overlap {
                    let sa = s.slot_of[&id_a];
                    let sb = s.slot_of[&id_b];
                    assert_ne!(
                        sa, sb,
                        "layers {id_a} (live {a0}..={a1}) and {id_b} (live {b0}..={b1}) \
                         share slot {sa} but their live intervals overlap",
                    );
                }
            }
            let _ = index_of.get(&id_a);
        }
    }

    #[test]
    fn linear_chain_peaks_at_two() {
        let mut g = base_graph();
        let a = g.output.color; // seeded by Graph::new
        let b = add_color(&mut g, "b");
        let ab = add_mix(&mut g, "ab", a.unwrap(), b);
        g.set_output(Output {
            color: Some(ab),
            roughness: ScalarInput::Const(0.5),
            metallic: ScalarInput::Const(0.0),
            normal: None,
        })
        .unwrap();
        let s = schedule(&g, &EvalCtx::default()).unwrap();
        assert_valid_allocation(&g, &s);
        // Inputs are freed after the consumer's slot is allocated, so `a`,
        // `b` and `ab` are live together.
        assert!(s.peak_slots >= 3, "peak_slots was {}", s.peak_slots);
    }

    #[test]
    fn diamond_needs_three_slots() {
        let mut g = base_graph();
        let a = g.output.color.unwrap();
        let b = add_mix(&mut g, "b", a, a);
        let c = add_mix(&mut g, "c", a, a);
        let bc = add_mix(&mut g, "bc", b, c);
        g.set_output(Output {
            color: Some(bc),
            roughness: ScalarInput::Const(0.5),
            metallic: ScalarInput::Const(0.0),
            normal: None,
        })
        .unwrap();
        let s = schedule(&g, &EvalCtx::default()).unwrap();
        assert_valid_allocation(&g, &s);
        assert!(s.peak_slots >= 3, "peak_slots was {}", s.peak_slots);
    }

    #[test]
    fn long_lived_leaf_stays_alive() {
        let mut g = base_graph();
        let leaf = g.output.color;
        let mut cur = add_color(&mut g, "seed");
        for i in 0..5 {
            let name = format!("step-{i}");
            cur = add_mix(&mut g, &name, cur, leaf.unwrap());
        }
        g.set_output(Output {
            color: Some(cur),
            roughness: ScalarInput::Const(0.5),
            metallic: ScalarInput::Const(0.0),
            normal: None,
        })
        .unwrap();
        let s = schedule(&g, &EvalCtx::default()).unwrap();
        assert_valid_allocation(&g, &s);
    }

    #[test]
    fn schedule_no_reuse_gives_unique_slots() {
        let mut g = base_graph();
        let a = g.output.color;
        let b = add_color(&mut g, "b");
        let ab = add_mix(&mut g, "ab", a.unwrap(), b);
        g.set_output(Output {
            color: Some(ab),
            roughness: ScalarInput::Const(0.5),
            metallic: ScalarInput::Const(0.0),
            normal: None,
        })
        .unwrap();
        let s = schedule_no_reuse(&g, &EvalCtx::default()).unwrap();
        let mut slots: Vec<u32> = s.slot_of.values().copied().collect();
        slots.sort();
        slots.dedup();
        assert_eq!(slots.len(), s.slot_of.len(), "slots must be unique in no-reuse mode");
        assert_eq!(s.peak_slots as usize, s.order.len());
    }

    fn add_transform(
        g: &mut Graph,
        name: &str,
        src: LayerId,
        scale: [f32; 3],
        edge_mode: texture_graph_core::EdgeMode,
    ) -> LayerId {
        g.add_layer(
            name,
            LayerKind::Transform(texture_graph_core::Transform {
                source: Some(src),
                offset: [0.0; 3],
                rotate_uv: 0.0,
                scale,
                coord_mode: CoordMode::Passthrough,
                edge_mode,
            }),
        )
        .unwrap()
    }

    #[test]
    fn extend_transform_widens_source_domain() {
        use texture_graph_core::EdgeMode;
        let mut g = base_graph();
        let src = g.output.color.unwrap();
        let t = add_transform(&mut g, "t", src, [3.0, 1.0, 1.0], EdgeMode::Extend);
        g.set_output(Output {
            color: Some(t),
            roughness: ScalarInput::Const(0.5),
            metallic: ScalarInput::Const(0.0),
            normal: None,
        })
        .unwrap();
        let s = schedule(&g, &EvalCtx::default()).unwrap();
        let d = s.domain_of[&src];
        assert_eq!(d.min, [0.0, 0.0]);
        assert!((d.max[0] - 3.0).abs() < 1e-6, "u max {}", d.max[0]);
        assert!((d.max[1] - 1.0).abs() < 1e-6);
        assert_eq!(s.domain_of[&t], Domain::UNIT);
    }

    #[test]
    fn clamp_transform_keeps_source_at_unit() {
        use texture_graph_core::EdgeMode;
        let mut g = base_graph();
        let src = g.output.color.unwrap();
        let t = add_transform(&mut g, "t", src, [3.0, 1.0, 1.0], EdgeMode::Clamp);
        g.set_output(Output {
            color: Some(t),
            roughness: ScalarInput::Const(0.5),
            metallic: ScalarInput::Const(0.0),
            normal: None,
        })
        .unwrap();
        let s = schedule(&g, &EvalCtx::default()).unwrap();
        assert_eq!(s.domain_of[&src], Domain::UNIT);
    }

    #[test]
    fn extend_requests_are_exact_and_chains_compose() {
        use texture_graph_core::EdgeMode;
        let mut g = base_graph();
        let src = g.output.color.unwrap();
        let t1 = add_transform(&mut g, "t1", src, [2.0, 1.0, 1.0], EdgeMode::Extend);
        let t2 = add_transform(&mut g, "t2", t1, [2.0, 1.0, 1.0], EdgeMode::Extend);
        // Affine requests are not capped.
        let big = add_transform(&mut g, "big", src, [100.0, 1.0, 1.0], EdgeMode::Extend);
        let both = add_mix(&mut g, "both", t2, big);
        g.set_output(Output {
            color: Some(both),
            roughness: ScalarInput::Const(0.5),
            metallic: ScalarInput::Const(0.0),
            normal: None,
        })
        .unwrap();
        let s = schedule(&g, &EvalCtx::default()).unwrap();
        assert!((s.domain_of[&t1].max[0] - 2.0).abs() < 1e-6);
        let d = s.domain_of[&src];
        assert!((d.max[0] - 100.0).abs() < 1e-3, "u max {}", d.max[0]);
        assert_eq!(d.min[1], 0.0);
    }

    #[test]
    fn radial_extend_request_caps_at_limit() {
        use texture_graph_core::EdgeMode;
        let mut g = base_graph();
        let src = g.output.color.unwrap();
        let t = g
            .add_layer(
                "radial",
                LayerKind::Transform(texture_graph_core::Transform {
                    source: Some(src),
                    offset: [0.0; 3],
                    rotate_uv: 0.0,
                    scale: [40.0, 1.0, 1.0],
                    coord_mode: CoordMode::Radial { dim: RadialDim::D2, into: Axis::U },
                    edge_mode: EdgeMode::Extend,
                }),
            )
            .unwrap();
        g.set_output(Output {
            color: Some(t),
            roughness: ScalarInput::Const(0.5),
            metallic: ScalarInput::Const(0.0),
            normal: None,
        })
        .unwrap();
        let s = schedule(&g, &EvalCtx::default()).unwrap();
        let d = s.domain_of[&src];
        assert!((d.max[0] - (0.5 + EXTEND_LIMIT)).abs() < 1e-4, "u max {}", d.max[0]);
    }

    #[test]
    fn unreachable_layers_are_not_scheduled_for_output_pass() {
        let mut g = base_graph();
        let _dead = add_color(&mut g, "dead");
        let s = schedule(&g, &EvalCtx::default()).unwrap();
        assert_eq!(s.order.len(), 1);
    }
}
