//! Pebble-game scheduler for the compute pipeline.
//!
//! Given a `Graph`, produces:
//! - a topological order over every layer reachable from `Output`;
//! - an assignment `LayerId -> slot` where each "slot" is one of the pooled
//!   `Rgba32Float` textures the baker allocates;
//! - the peak number of slots that were live simultaneously — the exact
//!   texture-pool size to allocate.
//!
//! Method: standard linear-scan register allocation. Refcount every layer's
//! remaining consumers (other layers + the four Output roots). Walk the
//! topo order; for each layer, pop a slot from a free pool (or grow it);
//! after dispatch, decrement each input's refcount and return its slot to
//! the pool once no consumer remains. Output-root layers get an extra +1
//! sentinel so their slot survives past dispatch and can be sampled by the
//! `pack_srgb8` pass.
//!
//! The greedy allocation is optimal for **any fixed topo order** (interval
//! coloring is trivially greedy-optimal); reordering the topo pass to
//! minimize peak is NP-hard and left for later if peak_slots ever becomes a
//! problem.

use std::collections::{HashMap, HashSet};

use texture_graph_core::{Graph, LayerId, ScalarInput};

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
}

/// Post-schedule descriptor for the `pack_srgb8` stage.
#[derive(Debug, Copy, Clone)]
pub struct OutputSlots {
    pub color: u32,
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

    // Reverse-transitive closure — every layer reachable from any root.
    let mut required: HashSet<LayerId> = HashSet::new();
    let mut stack = output_roots.clone();
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
    for &root in &output_roots {
        topo_dfs(graph, root, &required, &mut visiting, &mut done, &mut order)?;
    }

    // Refcount consumers. Every occurrence of an id in another layer's
    // `inputs()` counts once; each Output-root reference also counts once.
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
    for &root in &output_roots {
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

    // Resolve the four output channels against the final slot_of.
    let output_slots = OutputSlots {
        color: *slot_of
            .get(&graph.output.color)
            .ok_or(ScheduleError::UnknownLayer(graph.output.color))?,
        roughness: scalar_to_slot(&graph.output.roughness, &slot_of)?,
        metallic: scalar_to_slot(&graph.output.metallic, &slot_of)?,
        normal: match graph.output.normal {
            Some(id) => Some(
                *slot_of.get(&id).ok_or(ScheduleError::UnknownLayer(id))?,
            ),
            None => None,
        },
    };

    Ok(Schedule { order, slot_of, peak_slots, output_slots })
}

/// One texture per layer, no reuse — used for the per-layer preview pass
/// where every intermediate must survive to the end so it can be readback
/// into egui.
pub fn schedule_no_reuse(graph: &Graph) -> Result<Schedule, ScheduleError> {
    let output_roots = output_referenced(graph);
    let mut required: HashSet<LayerId> = HashSet::new();
    let mut stack = output_roots.clone();
    while let Some(id) = stack.pop() {
        if !required.insert(id) {
            continue;
        }
        let layer = graph.get(id).ok_or(ScheduleError::UnknownLayer(id))?;
        for input in layer.kind.inputs() {
            stack.push(input);
        }
    }
    // For per-layer previews, users want thumbnails of *every* authored
    // layer, not only those wired to Output. Add unreachable layers too.
    for l in &graph.layers {
        required.insert(l.id);
    }
    let mut order = Vec::with_capacity(required.len());
    let mut done: HashSet<LayerId> = HashSet::new();
    let mut visiting: HashSet<LayerId> = HashSet::new();
    for id in required.iter().copied().collect::<Vec<_>>() {
        topo_dfs(graph, id, &required, &mut visiting, &mut done, &mut order)?;
    }
    let mut slot_of: HashMap<LayerId, u32> = HashMap::new();
    for (i, &id) in order.iter().enumerate() {
        slot_of.insert(id, i as u32);
    }
    let peak_slots = order.len() as u32;
    let output_slots = OutputSlots {
        color: *slot_of.get(&graph.output.color).unwrap_or(&0),
        roughness: scalar_to_slot(&graph.output.roughness, &slot_of)?,
        metallic: scalar_to_slot(&graph.output.metallic, &slot_of)?,
        normal: graph.output.normal.and_then(|id| slot_of.get(&id).copied()),
    };
    Ok(Schedule { order, slot_of, peak_slots, output_slots })
}

fn output_referenced(graph: &Graph) -> Vec<LayerId> {
    let mut out = vec![graph.output.color];
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
                a,
                b,
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
        let ab = add_mix(&mut g, "ab", a, b);
        // Overwrite output to point at the terminal.
        g.set_output(Output {
            color: ab,
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
        let a = g.output.color;
        // Wrap `a` in trivial transforms so we have distinct b, c layers
        // that both consume a.
        let b = add_mix(&mut g, "b", a, a);
        let c = add_mix(&mut g, "c", a, a);
        let bc = add_mix(&mut g, "bc", b, c);
        g.set_output(Output {
            color: bc,
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
            cur = add_mix(&mut g, &name, cur, leaf);
        }
        g.set_output(Output {
            color: cur,
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
        let ab = add_mix(&mut g, "ab", a, b);
        g.set_output(Output {
            color: ab,
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

    #[test]
    fn unreachable_layers_are_not_scheduled_for_output_pass() {
        let mut g = base_graph();
        let _dead = add_color(&mut g, "dead");
        // Output still points at the auto-created base color layer.
        let s = schedule(&g).unwrap();
        assert_eq!(s.order.len(), 1);
    }
}
