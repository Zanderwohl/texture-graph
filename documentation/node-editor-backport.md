# Backporting node-editor improvements from modular-block-game

## Context

The graph UI in `~/rust/modular-block-game` (MBG) was copied from this project and
then improved. This document records what diverged and what is worth bringing back.

**Source of truth for the port:** `~/rust/modular-block-game`, files under
`src/common/generator/`. The direct counterpart of `crates/ui/src/panels/graph_canvas/`
is `src/common/generator/canvas/` (`mod.rs`, `layout.rs`, `nodes.rs`, `wires.rs`),
with supporting code in `src/common/generator/{state,previews,catalog,widgets}.rs`.

Both projects are on **egui 0.35** (MBG via `bevy_egui 0.41`), so UI code ports
almost verbatim. The mechanical substitutions are:

| MBG | here |
| --- | --- |
| `bevy_egui::egui` | `egui` |
| `worldgen_core` | `texture_graph_core` |
| `crate::common::generator::state` | `crate::state` |
| `super::bake::Sampler` + `ResourceRegistry` | `EvalCtx` + `Option<&mut GpuBits>` |
| `CANVAS` (hardcoded `"main"`) | `state.active_canvas` (keep ours — it is more general) |

MBG's node model is typed (`ValueType` = `Scalar/Vector(Dims)`, `Range`, `Blocks`);
ours has exactly one value type (an Oklcha `Color`). Anything in MBG that leans on
the type lattice does not transfer directly.

## Not worth backporting (we are ahead)

- `graph_canvas/ramp.rs` — the draggable ColorRamp gradient bar. MBG has no
  counterpart; its stop strips (Gradient Pass, Block Map) are paint-only.
- `UiState::remap_ramp_consts` — needed here because `InputKey::RampStop(i)` is
  positional. MBG dropped it because its stops are plain params with no sockets.
- Multi-canvas (`active_canvas`, `AddCanvas`/`RemoveCanvas`). MBG hardcoded a
  single `"main"` canvas.
- `wires::apply_socket_edits` — MBG replaced it with `EditCmd::Connect` routed
  through a `Graph::connect` method in core. Cleaner, but ours batches a
  detach-and-reconnect on the same node into a single `SetKind` so the
  intermediate state never exists. **Keep ours.**

## The GPU side (separate effort, not part of this plan)

`worldgen-gpu` is a different architecture, not a backport: compute kernels over
flat storage buffers (one kernel per node kind, each concatenated onto a shared
`shaders/prelude.wgsl`), with a `Domain` carrying origin/extent/step plus a shape
code (`LINE`/`AREA`/`VOLUME`) so D1/D2/D3 are one code path via
`cell_xyz` / `cell_pos` / `sample_pos`. Ours is render passes over 2D textures,
with volumes done by re-running the whole 2D pipeline once per slice
(`crates/gpu/src/baker.rs:647-796`). Note the generalization is over 1D/2D/3D —
there is no 4D anywhere in MBG.

One idea transfers cheaply and independently: the parity discipline in
`crates/worldgen-gpu/tests/parity.rs`. Two rules get `assert_eq!` on
`f32::to_bits` between CPU and GPU — (1) fuse every multiply-add in a stated
association order (`mul_add` / `fma`, left-associated), (2) never divide by
anything a shader could rewrite as a reciprocal; invert per-node constants once
on the CPU and upload. Our `crates/gpu/src/bake_test.rs` asserts tolerances
(`max_delta <= 14`), which is the standard that let MBG's noise drift.

---

# Tier 1 — near-verbatim ports (DONE)

- [x] **1. Inline node rename**
- [x] **2. Bidirectional wire drag**
- [x] **3. Off-screen node culling**
- [x] **4. Shared per-frame row style**
- [x] **5. Status-line lifecycle**
- [x] **6. `EditCmd` housekeeping**
- [x] **7. Canvas background fill**

Landed across `crates/ui/src/state.rs`,
`crates/ui/src/panels/graph_canvas/{mod,nodes,wires}.rs`. Builds clean, no new
clippy warnings, 9 UI tests pass, app starts without panicking.

**Deviations from what is described below**, all deliberate:

- Item 2: we kept `apply_socket_edits` as the drop action, so there is no
  `EditCmd::Connect` and no `Graph::connect`. MBG's `refusal() -> Option<String>`
  became `wires::eligible(graph, node, src) -> bool`, because with one value
  type the only thing to check is a cycle. When a type system arrives, widen
  `eligible` back into `refusal` and the socket-rings-red plumbing is already
  in place.
- Item 3: the cull margin is `socket_hit_radius(zoom)` rather than MBG's fixed
  `SOCKET_R * 4.0`. The grab radius grows with zoom; the constant comes up short
  when zoomed in.
- Item 6: also added `UiState::reset_for_new_graph`, called from
  `EditCmd::Replace`. File > Open was leaving the selection, stashed consts, and
  any in-flight drag pointing at ids that mean something else in the new graph —
  the same class of bug, so it went in with the rest.
- `drain_into` now returns `bool` ("anything landed"). Nothing consumes it yet;
  it is what Tier 2 item 8 and an unsaved-changes indicator will key off.
- Two tests came along as a down payment on Tier 3: `rename_command` in
  `nodes.rs` and `hovered_output` in `wires.rs`.

### 1. Inline node rename

`EditCmd::Rename` already exists in `crates/ui/src/state.rs` and is wired to
`Graph::rename`, but **nothing emits it** — layers cannot currently be renamed at
all. Port from `canvas/nodes.rs`: `title()`, `title_rect()`, `rename_command()`,
plus `Renaming { node, text, focused }` in `state.rs` and a "Rename" entry in the
node context menu.

Non-obvious parts, already solved in MBG:
- Check `Escape` **before** `resp.lost_focus()` — Escape also surrenders focus, so
  the other order reads a cancel as a commit.
- `request_focus()` only when `!focused`; asking every frame makes the field
  impossible to blur, and blurring is one of the two ways to commit.
- `rename_command` returns `None` for unchanged-after-trim and for empty text, so
  clicking a name to read it and clicking away does not mark the file dirty.
- `draw_chrome` takes a `renaming: bool` and skips painting the title text while
  the field is up.
- `title()` runs after `body_interact` so the field wins the pointer over the drag.
- Skip entirely below `ZOOM_WIDGETS_MIN`.

### 2. Bidirectional wire drag

Today, pressing an *unconnected* input does nothing. Port from `state.rs` +
`canvas/wires.rs` + `canvas/nodes.rs::sockets`:

- `WireDrag` struct -> enum:
  `FromOutput { src, detached_from } | FromInput { node, key }`,
  with `fn detached_from(self) -> Option<(NodeRef, InputKey)>`.
- `wires::nearest()` — shared hit-test helper for both socket sets.
- `wires::hovered_output()` — the Output pseudo-node has none, which
  `NodeLayout::output_socket == None` already says.
- `wires::anchor()` — where the live wire is pinned for the whole drag.
- `Candidate { node, key, src, at }` — both directions converge here, so the drop
  path stays single.
- In `sockets()`: a *connected* input hands its wire over (`FromOutput` with
  `detached_from`); an *unconnected* one starts `FromInput`.
- `socket_style` / `output_socket_style` split, so an output socket rings on a
  backwards drag and an input socket does not react to one.

Keep our `apply_socket_edits` as the drop action (see "Not worth backporting").
A `refusal()`-style pre-flight with a message is Tier 2, item 10.

### 3. Off-screen node culling

In `draw_and_interact_nodes`, skip any node whose `rect.expand(SOCKET_R * 4.0)`
does not intersect `canvas_rect` — no interact, no paint, **no `get_or_build`**.
The margin covers sockets, which sit on the node edge.

Exempt nodes that are `held()`: the one being dragged (the drag resolves against
the pointer, so a node that stopped interacting on leaving the view would be
dropped there) and the one being renamed (a text field that stops existing takes
the keyboard focus with it).

### 4. Shared per-frame row style

`row_ui` currently clones a whole `egui::Style` per row, per node, per frame.
Hoist to `row_style(base: &egui::Style, zoom: f32) -> Arc<egui::Style>`, built
once in `draw_and_interact_nodes` and passed down; `row_ui` takes
`&Arc<egui::Style>` and calls `child.set_style(style.clone())`.

### 5. Status-line lifecycle

`UiState::last_error` is set on failure and never cleared — one refused
connection sits in the status row for the rest of the session. Port MBG's
`drain_into`: track `changed` / `refused`, clear `last_error` when a later edit
lands cleanly, and return `bool` ("anything landed") for the unsaved-changes
question.

### 6. `EditCmd` housekeeping

`EditCmd::apply` does not clean up after `Remove`: `state.selected`,
`state.preview_target` and `state.saved_consts` keep pointing at a dead
`LayerId`. Move `apply` onto `UiState` (as MBG did) so it can:

- `Remove`: clear `selected` / `preview_target` if they name the removed layer,
  and `saved_consts.retain(|(n, _), _| *n != NodeRef::Layer(id))`.
- `AddLayer`: set `selected` to the new id.

### 7. Canvas background fill

`painter.rect_filled(canvas_rect, 0.0, egui::Color32::from_gray(28))` right after
`ui.painter_at(canvas_rect)`, so nodes sit on a canvas rather than floating on
the panel default.

---

# Tier 2 — real wins, need adaptation (NOT STARTED)

### 8. Per-node preview staleness + bake budget

The biggest win. `crates/ui/src/previews.rs` is all-or-nothing: any
eval-affecting edit sets `stale`, and the next `get_or_build` re-bakes **every**
layer via `Baker::bake_previews`. Port from `src/common/generator/previews.rs`
and `state.rs`:

- `Entry { revision, forced, texture }` per node; `stale_at: HashMap<LayerId, u64>`;
  `is_fresh(id)`.
- `UiState::revision: u64` and `dirty_previews: Option<HashSet<LayerId>>` (`None`
  = all), computed in `drain_into` from `dirty_roots(&cmd)` plus a `downstream()`
  consumer walk. Take the **union of the pre- and post-apply graph** so a
  `Remove` still finds its readers — over-invalidating costs a bake, never a
  stale picture.
- `begin_frame(revision, dirty)` + `BAKES_PER_FRAME` budget; stale nodes keep
  showing the previous image.
- `retain_live(graph)` so a long session does not accumulate textures for deleted
  nodes.

Two caveats specific to us: our bulk GPU bake is a single submit for the whole
graph, so the per-node economics differ from MBG's — but the baker still builds
intermediates for every layer, so the win should still be large. And
`InputKey::RampStop(i)` is positional, so `dirty_roots` must account for ramp
reorders (MBG's comment explicitly notes this obligation does not apply to *its*
stop lists).

### 9. A `catalog` module

The node list is written down **twice**: `VARIANTS: &[&str]` in
`graph_canvas/mod.rs` and `VARIANTS: &[VariantSpec]` in `panels/inspector.rs`,
and `default_kind` is keyed by `&str`. Port `src/common/generator/catalog.rs`:
one `VARIANTS` list carrying a `Kind` enum (matched instead of the label, so
editing menu text cannot quietly produce a different node), a `Group` for
submenus, `default_kind(&Variant)`, and `unique_name` (move `crate::util`'s in).

### 10. `refusal()` pre-flight with a message

`wires::refusal(graph, node, key, src) -> Option<String>` runs while the wire is
still in the air, turns the target socket red, and supplies the drop-time error
text. We only check `would_cycle` and report a generic message *after* the drop.
The type half does not transfer (everything is a `Color`), but `set_input` can
already fail, and the shape is worth having before a type system arrives.

# Tier 3 — cheap tests (NOT STARTED)

From `canvas/layout.rs`, all applicable as-is:

- every kind in the catalog lays out exactly the sockets `input_sockets()`
  reports (a mismatch silently loses an input, with no error)
- `socket_label(key)` agrees with the model's own `socket.label`
- a conditional row actually changes `node_height` — we have these: Mix's `Blend`
  factor row, Transform's `CoordMode` rows

Plus `rename_command` and `hovered_output` tests in `nodes.rs` / `wires.rs`.
