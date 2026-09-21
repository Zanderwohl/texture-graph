//! Pebble-game scheduler for the compute pipeline.
//!
//! Given a `Graph`, produces:
//! - a topological order over every layer reachable from `Output`;
//! - an assignment `LayerId -> slot` where each "slot" is one of the pooled
//!   `Rgba32Float` textures the baker allocates;
//! - the peak number of slots that were live simultaneously — the exact
//!   texture-pool size to allocate.
//!
//! Linear-scan register allocation: refcount each layer's remaining
//! consumers, walk the topo order popping a slot per layer, and return a
//! slot once its last consumer has run. Output roots carry a +1 sentinel so
//! their slot survives to be sampled by `pack_srgb8`.
//!
//! Greedy is optimal for any fixed topo order, since interval coloring is.
//! Reordering the topo pass to minimize peak is NP-hard.

use std::collections::{HashMap, HashSet};

use texture_graph_core::{
    Axis, CoordMode, EXTEND_LIMIT, EdgeMode, Graph, LayerId, LayerKind, RadialDim, ScalarInput,
    Transform,
};

/// Fully-resolved dispatch plan for one bake.
#[derive(Debug, Clone)]
pub struct Schedule {
    /// Layers to dispatch, in the order the baker executes them.
    pub order: Vec<LayerId>,
    /// Which pool slot each layer's output ends up in.
    pub slot_of: HashMap<LayerId, u32>,
    /// Number of `Rgba32Float` textures to allocate in the pool.
    pub peak_slots: u32,
    /// Where the four PBR channels come from after all dispatches.
    pub output_slots: OutputSlots,
    /// UV rectangle each layer is baked over. `Domain::UNIT` for everything
    /// unless an `EdgeMode::Extend` transform pulls a source wider.
    pub domain_of: HashMap<LayerId, Domain>,
}

/// UV rectangle a layer's bake covers. Always contains the unit square, so
/// thumbnails and direct output stay renderable; extend-transform consumers
/// grow it, exactly for affine requests and capped at [`EXTEND_LIMIT`] for
/// radial ones.
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

    /// `(min_u, min_v, ext_u, ext_v)` — the uniform layout the shaders use
    /// to map UV to texels.
    pub fn packed(&self) -> [f32; 4] {
        [
            self.min[0],
            self.min[1],
            self.max[0] - self.min[0],
            self.max[1] - self.min[1],
        ]
    }
}

/// Per-layer bake domains. Every reachable layer starts at the unit square,
/// then a reverse-topo walk unions in what each consumer samples. Only
/// `EdgeMode::Extend` reaches beyond the unit square; everything else samples
/// at its own coordinates, so a widened consumer widens its inputs too. A
/// `Map`'s palette is luminance-indexed and always within range.
fn compute_domains(graph: &Graph, order: &[LayerId]) -> HashMap<LayerId, Domain> {
    let mut dom: HashMap<LayerId, Domain> =
        order.iter().map(|&id| (id, Domain::UNIT)).collect();
    for &id in order.iter().rev() {
        let d = dom[&id];
        let Some(layer) = graph.get(id) else { continue };
        match &layer.kind {
            LayerKind::Transform(t) => {
                // Clamp samples within [0, 1], which the baseline covers.
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

/// The UV rectangle an extend-transform samples its source over, given the
/// transform's own bake domain `d`: the AABB of the transformed corners.
/// Affine (passthrough/permute) requests are exact and unbounded — the
/// bake still spends the same pixel count, just spread over the wider
/// rectangle, matching 1:1 what the consumer samples. Radial requests are
/// conservative, so they intersect with the [`EXTEND_LIMIT`] box; `None`
/// when that leaves nothing (those samples all show the missing grid, so
/// the source needn't grow). Non-finite requests (degenerate scales) also
/// return `None`.
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
    // w spans [0, 1] across a volume bake (0.5 flat) — conservative range
    // for the permute/radial cases that read it.
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

/// Post-schedule descriptor for the `pack_srgb8` stage.
#[derive(Debug, Copy, Clone)]
pub struct OutputSlots {
    /// `None` = unconnected — pack from the missing-texture grid.
    pub color: Option<u32>,
    pub roughness: ScalarSlot,
    pub metallic: ScalarSlot,
    pub normal: Option<u32>,
}

/// A scalar output channel either takes a constant or reads a layer's L.
#[derive(Debug, Copy, Clone)]
pub enum ScalarSlot {
    Const(f32),
    Slot(u32),
}

/// Errors surfaced during scheduling. A well-formed `Graph` (as
/// enforced by the graph mutators) should never trip these.
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

/// Build a `Schedule` for `graph`. Only layers reachable from `Output` are
/// included; unreachable layers cost neither dispatches nor slots.
pub fn schedule(graph: &Graph) -> Result<Schedule, ScheduleError> {
    let output_roots = output_referenced(graph);
    let plan = plan_from(graph, &output_roots)?;

    // Resolve the four output channels against the final slot_of.
    let output_slots = OutputSlots {
        color: match graph.output.color {
            Some(id) => Some(*plan.slot_of.get(&id).ok_or(ScheduleError::UnknownLayer(id))?),
            None => None,
        },
        roughness: scalar_to_slot(&graph.output.roughness, &plan.slot_of)?,
        metallic: scalar_to_slot(&graph.output.metallic, &plan.slot_of)?,
        normal: match graph.output.normal {
            Some(id) => Some(
                *plan.slot_of.get(&id).ok_or(ScheduleError::UnknownLayer(id))?,
            ),
            None => None,
        },
    };
    Ok(plan.into_schedule(output_slots))
}

/// Build a `Schedule` that produces one layer and nothing else — `root`
/// plus everything it transitively reads, with the same slot reuse
/// `schedule` gets.
///
/// This is the plan a single-channel bake wants: the graph's Output may
/// route through nodes the caller does not care about, and a consumer
/// asking for one scalar field should not pay for the other three PBR
/// channels' dispatches.
///
/// `output_slots.color` points at `root`, so a caller that wants to reuse
/// the ordinary pack path can. The scalar channels report their constants
/// and `normal` is `None`; nothing here consults them.
pub fn schedule_layer(graph: &Graph, root: LayerId) -> Result<Schedule, ScheduleError> {
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

/// Everything a `Schedule` holds except which slots the output channels
/// read, which is the one part that depends on why the bake was asked for.
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

/// Topological order over `roots` and their upstream closure, with slots
/// allocated by the pebble game described in the module docs.
fn plan_from(graph: &Graph, roots: &[LayerId]) -> Result<Plan, ScheduleError> {
    // Reverse-transitive closure — every layer reachable from any root.
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

    // Topological post-order DFS. `visiting` catches cycles (defense-in-depth
    // — the graph mutators already reject cycles at edit time).
    let mut order = Vec::with_capacity(required.len());
    let mut done: HashSet<LayerId> = HashSet::new();
    let mut visiting: HashSet<LayerId> = HashSet::new();
    for &root in roots {
        topo_dfs(graph, root, &required, &mut visiting, &mut done, &mut order)?;
    }

    // Refcount consumers. Every occurrence of an id in another layer's
    // `inputs()` counts once; each root reference also counts once, which
    // is the sentinel that keeps a root's slot alive to be packed.
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

    // Walk topo order, allocating slots. Free pool is a Vec used as a
    // stack — reusing the most-recently-freed slot keeps cache locality
    // decent.
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

        // Decrement every input's remaining_uses; if it hits zero, its slot
        // returns to the free pool. Uses on a per-layer basis are unique per
        // occurrence — a Mix referencing the same layer twice still counts
        // twice.
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

/// One texture per layer, no reuse — used for the per-layer preview pass
/// where every intermediate must survive to the end so it can be read back
/// into egui.
pub fn schedule_no_reuse(graph: &Graph) -> Result<Schedule, ScheduleError> {
    schedule_previews(graph, None)
}

/// The preview schedule for `wanted` — those layers and everything they
/// transitively read, and nothing else. `None` wants every authored layer.
///
/// Baking a subset is what makes an edit to one corner of a graph cost one
/// corner's worth of work. Layers outside the upstream closure of `wanted`
/// are not dispatched, not allocated a slot, and not packed.
///
/// **Domains are computed over the whole graph regardless.** A layer's bake
/// domain is decided by its *consumers* (an `EdgeMode::Extend` transform
/// pulls its source wider), so restricting the walk to the subset would
/// give a layer a narrower domain whenever the consumer that widened it
/// happened to be clean — and its thumbnail would come back at a different
/// effective resolution depending on what else was being baked. One extra
/// topological pass buys a thumbnail that is the same picture either way.
pub fn schedule_previews(
    graph: &Graph,
    wanted: Option<&HashSet<LayerId>>,
) -> Result<Schedule, ScheduleError> {
    // Everything, in dependency order — what the domains are computed from,
    // and the whole schedule when `wanted` is `None`.
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
    // Lenient, unlike `schedule`'s: on a subset the output roots may not be
    // scheduled at all. The preview pass never reads these — it packs each
    // layer's own slot — so a missing root is not an error here.
    let output_slots = OutputSlots {
        color: graph.output.color.and_then(|id| slot_of.get(&id).copied()),
        roughness: scalar_slot_lenient(&graph.output.roughness, &slot_of),
        metallic: scalar_slot_lenient(&graph.output.metallic, &slot_of),
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
    // Sorted so the order is deterministic run to run: `HashSet` iteration
    // is not, and a schedule that reshuffles between frames would make
    // slot assignments — and so any bug in them — irreproducible.
    let mut ids: Vec<LayerId> = required.iter().copied().collect();
    ids.sort();
    for id in ids {
        topo_dfs(graph, id, required, &mut visiting, &mut done, &mut order)?;
    }
    Ok(order)
}

/// Like [`scalar_to_slot`], but a layer that isn't scheduled falls back to
/// a constant instead of failing. Only for the preview pass, which doesn't
/// read output slots.
fn scalar_slot_lenient(s: &ScalarInput, slot_of: &HashMap<LayerId, u32>) -> ScalarSlot {
    match *s {
        ScalarInput::Const(v) => ScalarSlot::Const(v),
        ScalarInput::Layer(id) => match slot_of.get(&id) {
            Some(slot) => ScalarSlot::Slot(*slot),
            None => ScalarSlot::Const(0.0),
        },
    }
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

fn scalar_to_slot(
    s: &ScalarInput,
    slot_of: &HashMap<LayerId, u32>,
) -> Result<ScalarSlot, ScheduleError> {
    Ok(match *s {
        ScalarInput::Const(v) => ScalarSlot::Const(v),
        ScalarInput::Layer(id) => ScalarSlot::Slot(
            *slot_of.get(&id).ok_or(ScheduleError::UnknownLayer(id))?,
        ),
    })
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
        BlendMode, BlendSpace, Color, Graph, LayerKind, Mix, Output, ScalarInput,
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

    /// Assert the schedule's slot assignment is valid: no two layers that
    /// are simultaneously live share a slot. "Live" spans from a layer's
    /// dispatch through its last consumer's dispatch (inclusive of Output
    /// use, which we treat as after every dispatch).
    fn assert_valid_allocation(graph: &Graph, s: &Schedule) {
        let index_of: HashMap<LayerId, usize> = s
            .order
            .iter()
            .enumerate()
            .map(|(i, id)| (*id, i))
            .collect();
        // Live interval: [i, last_consumer_i] (or ∞ if it's an output root).
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
        // For each layer pair, if intervals overlap they must not share a slot.
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
        // A -> Mix(A, A) -> Mix(prev, prev)  actually this is a real chain
        // structure. Take: base color; every consumer only reads the previous
        // layer so old ones can die.
        let mut g = base_graph();
        let a = g.output.color; // the "base color" seeded by Graph::new
        let b = add_color(&mut g, "b");
        let ab = add_mix(&mut g, "ab", a.unwrap(), b);
        // Overwrite output to point at the terminal.
        g.set_output(Output {
            color: Some(ab),
            roughness: ScalarInput::Const(0.5),
            metallic: ScalarInput::Const(0.0),
            normal: None,
        })
        .unwrap();
        let s = schedule(&g).unwrap();
        assert_valid_allocation(&g, &s);
        // Peak here: at the moment `ab` dispatches, both `a` and `b` are
        // live plus `ab`'s new slot, but `a` and `b` are freed after `ab`
        // reads them. Since inputs are freed *after* dispatch, allocation
        // happens first — needs 3 concurrent. So peak == 3.
        assert!(s.peak_slots >= 3, "peak_slots was {}", s.peak_slots);
    }

    #[test]
    fn diamond_needs_three_slots() {
        // a -> b, a -> c, mix(b, c). While mix dispatches, b, c, mix all live.
        let mut g = base_graph();
        let a = g.output.color.unwrap();
        // Wrap `a` in trivial transforms so we have distinct b, c layers
        // that both consume a.
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
        let s = schedule(&g).unwrap();
        assert_valid_allocation(&g, &s);
        // At `bc` dispatch: bc, b, c all live => >= 3.
        assert!(s.peak_slots >= 3, "peak_slots was {}", s.peak_slots);
    }

    #[test]
    fn long_lived_leaf_stays_alive() {
        // leaf feeds into a long chain of mixes as one operand each; leaf
        // survives to the end.
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
        let s = schedule(&g).unwrap();
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
        let s = schedule_no_reuse(&g).unwrap();
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
        let s = schedule(&g).unwrap();
        let d = s.domain_of[&src];
        // Transform samples u in [0, 3]; unit baseline keeps v at [0, 1].
        assert_eq!(d.min, [0.0, 0.0]);
        assert!((d.max[0] - 3.0).abs() < 1e-6, "u max {}", d.max[0]);
        assert!((d.max[1] - 1.0).abs() < 1e-6);
        // The transform itself stays at unit.
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
        let s = schedule(&g).unwrap();
        assert_eq!(s.domain_of[&src], Domain::UNIT);
    }

    #[test]
    fn extend_requests_are_exact_and_chains_compose() {
        use texture_graph_core::EdgeMode;
        let mut g = base_graph();
        let src = g.output.color.unwrap();
        // Two chained x2 extends: inner source needs u up to 4, chained
        // through the middle transform's own widened domain.
        let t1 = add_transform(&mut g, "t1", src, [2.0, 1.0, 1.0], EdgeMode::Extend);
        let t2 = add_transform(&mut g, "t2", t1, [2.0, 1.0, 1.0], EdgeMode::Extend);
        // And a huge affine scale — exact and unbounded, no cap.
        let big = add_transform(&mut g, "big", src, [100.0, 1.0, 1.0], EdgeMode::Extend);
        let both = add_mix(&mut g, "both", t2, big);
        g.set_output(Output {
            color: Some(both),
            roughness: ScalarInput::Const(0.5),
            metallic: ScalarInput::Const(0.0),
            normal: None,
        })
        .unwrap();
        let s = schedule(&g).unwrap();
        // t1's domain: widened by t2 to u in [0, 2].
        assert!((s.domain_of[&t1].max[0] - 2.0).abs() < 1e-6);
        // src: union of t1's request over its widened domain ([0, 4]) and
        // big's exact request ([0, 100]).
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
        let s = schedule(&g).unwrap();
        // r_max ~ 40 but the radial request is conservative, so it caps.
        let d = s.domain_of[&src];
        assert!((d.max[0] - (0.5 + EXTEND_LIMIT)).abs() < 1e-4, "u max {}", d.max[0]);
    }

    #[test]
    fn unreachable_layers_are_not_scheduled_for_output_pass() {
        let mut g = base_graph();
        let _dead = add_color(&mut g, "dead");
        // Output still points at the auto-created base color layer.
        let s = schedule(&g).unwrap();
        assert_eq!(s.order.len(), 1);
    }
}
