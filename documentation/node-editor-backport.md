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

### CPU/GPU noise parity — DONE

I originally described the parity discipline here as transferring "cheaply and
independently". That was wrong, and the correction is worth keeping: the two
backends were not one algorithm with different rounding, they were **two
different algorithms**. The CPU ran the `noise` crate's `Simplex` in f64 off one
permutation table; the GPU ran Gustavson's textureless simplex in f32 off
another. `noise.wgsl`'s own header said so. FMA discipline cannot close that.

**What landed:** `crates/core/src/noise.rs` is now the specification — Gustavson
simplex in Rust, function for function against `crates/gpu/src/shaders/noise.wgsl`,
with the shader's exact float literals. `eval_noise` calls it and the `noise`
crate dependency is gone. The *GPU* was chosen as the authority because it is
what users actually see, so no saved graph changed appearance.

**The bug this fixed:** on a machine with no working wgpu backend, previews
showed visibly different noise than a GPU bake of the same graph.

**A second bug it uncovered:** a flat GPU bake samples 3D fields at `w = 0.5`,
but the CPU fallbacks reached for `Sample::uv`, which is `w = 0.0` — two
different slices of the same volume. D1 and D2 hid it (they ignore `w`); D3
showed a worst-case sRGB delta of 203. There is now a `FLAT_W` constant and a
`Sample::flat` constructor so the convention has one home, used by both
fallbacks and the baker.

**What parity is actually held:** `cpu_and_gpu_noise_agree` bakes on the GPU and
evaluates the same points on the CPU across D1/D2/D3 and both ranges — the two
agree **within one sRGB step**, most samples identical. It reports a histogram,
because the shape of the disagreement is the diagnostic.

**What is not held: bit-exactness.** Two things are still missing, both real
work rather than oversights:

1. A float read-back path, so the comparison is on the field rather than on
   8-bit pixels. `bake_output` hands back `Rgba8Unorm`.
2. Every multiply-add pinned into explicitly fused form on both sides. A shader
   compiler contracts `a*b + c` into an FMA whether asked to or not, so the spec
   has to contract too (`mul_add` / `fma`). Doing this means rewriting the
   shader's arithmetic, which shifts GPU output by an ulp or two.

The residual ±1 is also not all noise: Oklch→sRGB is a separate pair of
implementations, and the out-of-range checker is a deliberate GPU-only display
choice (the parity test skips out-of-gamut samples and says so).

**What cannot go exact here at all:** tolerances like the extend-transform
test's `max_delta <= 14` are not arithmetic. The GPU bakes into intermediate
textures and resamples them at texel centres while the CPU evaluates
analytically. MBG's flat-buffer compute design makes that cell-for-cell exact;
our render-to-texture pipeline does not, and closing it would mean the kernel
rewrite described above.

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

# Fixes found after the fact

## Widget ids shuffling for a frame (egui auto-id hazard)

**Symptom:** a red outline flashed over most of the canvas for one frame
whenever the inline rename field appeared or disappeared — including when
clicking the Material Output's title, which merely blurs a rename elsewhere.

**What it was:** egui's `warn_if_rect_changes_id` debug overlay (on in every
debug build, alongside `warn_on_id_clash`). It fires when the widget at a given
rect has a different `Id` than it did last frame. It paints a red stroke and no
text, which is how it is told apart from the id-clash warning.

**Root cause**, in `egui::Ui::new_child`:

```rust
IdSource::Child(id_salt) => {
    let stable_id = self.id.with(id_salt);
    let unique_id = stable_id.with(self.next_auto_id_salt); // parent's counter!
    (stable_id, unique_id)
}
...
self.next_auto_id_salt = self.next_auto_id_salt.wrapping_add(1);
```

A child `Ui` built with `id_salt` still derives its `unique_id` — and so the
auto-id seed of every widget *inside* it — from the parent's running counter.
`row_ui` built one salted child per node row, so every inline slider and
drag-value depended on how many child `Ui`s happened to exist before it that
frame. Showing the rename field added one, and 51 widgets shifted a slot along
while staying exactly where they were.

**Fix:** `UiBuilder::id` instead of `id_salt` for the row children (an explicit
id is independent of the parent), and one shared child `Ui` for the title so
both branches register the same widget footprint whether it is a click target
or a text field.

This also mattered for the Tier 1 off-screen culling, which changes child counts
by a different route, and for anything that changes a node's row count
mid-canvas (switching a Mix off Blend drops two rows).

**Lesson for this codebase:** in the canvas, never let a child `Ui`'s identity
come from a salt. Positions, visibility and row counts all vary per frame, so
the parent's auto-id counter is not stable and nothing may depend on it.

**Tests:** `canvas_tests` in `graph_canvas/mod.rs` drives the real canvas
through a real `egui::Context` headlessly and counts shapes painted in
`error_fg_color`. Both halves of the fix were verified by reverting them and
watching the right test fail:

- `starting_and_ending_a_rename_does_not_shuffle_widget_ids` — the title half
- `changing_a_nodes_row_count_does_not_shuffle_later_widget_ids` — the row half
  (a Mix switched off Blend, with the Output node's sliders downstream of it)
- `panning_nodes_out_of_view_is_quiet` — proves the cull path is quiet, but
  does *not* exercise the id hazard: panning moves every rect, so egui has no
  same-rect pair to compare. The comment on the test says so.
- `clicking_a_title_renames_the_layer` / `escape_abandons_a_rename` — the
  feature itself, end to end through the real widget.

This harness is the tool to reach for the next time the canvas misbehaves
visually; it is much faster than driving the app by hand.

# Tier 2 — real wins, need adaptation (DONE)

- [x] **8. Per-layer preview staleness** (done; no bake budget — see below)
- [x] **9. A `catalog` module**
- [x] **10. `refusal()` pre-flight with a message**

### 8. Per-layer preview staleness — DONE

Landed across `crates/ui/src/{previews,state,app}.rs` and
`crates/gpu/src/{schedule,baker}.rs`. 78 tests pass, clippy clean in every file
touched.

What it does now:

- `UiState` carries `revision: u64` and `dirty_previews: Option<HashSet<LayerId>>`
  (`None` = all). `drain_into` computes the set per landed command from
  `dirty_roots(&cmd)` plus a `downstream()` consumer walk, taking the union of
  the pre- and post-apply graph so a `Remove` still finds its readers.
- `PreviewCache` keeps a `(revision, forced)` stamp per entry and a
  `stale_at: HashMap<LayerId, u64>`. `begin_frame(graph, revision, dirty, gpu)`
  runs at the top of the app's frame, retires what `dirty` names, and drops
  entries for layers that no longer exist.
- The first request for a stale layer triggers one bulk `rebuild` for the whole
  stale set (`rebuilt_this_frame` guards the rest of the frame). Clean layers
  are not baked, not registered, and keep the texture they already had.
- `Baker::bake_previews` gained a `wanted: Option<&HashSet<LayerId>>`.
  `schedule_previews` schedules only `wanted` and their upstream closure, and
  only `wanted` get an output texture and a pack pass.

**Deviations from the plan as written above:**

- **No `BAKES_PER_FRAME` budget.** MBG needs one because it bakes per node; here
  one `bake_previews` call covers the whole stale set in a single submit, so a
  budget would add staleness and save nothing. Revisit only if a cold cache on a
  very large graph turns out to hitch.
- **Domains are still computed over the whole graph** inside `schedule_previews`,
  deliberately. A layer's bake domain is decided by its *consumers* (an
  `EdgeMode::Extend` transform pulls its source wider), so a subset walk would
  give a layer a narrower domain whenever the consumer that widened it was clean
  — and its thumbnail would come back at a different effective resolution
  depending on what else was stale. `a_subset_bake_gives_the_same_picture_as_a_full_one`
  in `bake_test.rs` is what holds that.
- **The `RampStop(i)` caveat was over-cautious.** `SetKind` dirties the whole
  layer, so a reordered ramp needs no positional bookkeeping for previews. The
  obligation is real only for `saved_consts` (`remap_ramp_consts`), which is
  unchanged.
- **A failed bake now retries next frame** instead of silently clearing the stale
  flag. The common failure is transient (device lost on resume) and a thumbnail
  that never comes back is worse than one attempt per frame.
- `PreviewCache::mark_stale` is gone; `app.rs` no longer needs the `was_clean`
  dance around `state.dirty`, which belongs to the big preview panel alone.

### 8-original. Per-node preview staleness + bake budget (plan as written)

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

### 9. A `catalog` module — DONE

`crates/ui/src/catalog.rs` now owns the node list: `Variant { label, kind }`,
a `Kind` enum with `Kind::of(&LayerKind)`, `default_kind(Kind)` and
`unique_name`. `graph_canvas/mod.rs` and `panels/inspector.rs` both read it;
`crates/ui/src/util.rs` is gone (it held only `unique_name`).

What this removed: two copies of the eight-entry list, a `&str` key
(`"ColorRamp"`) that the label duplicated exactly, and a `_ =>` fallback arm in
`default_kind` that silently produced a Color for anything unrecognized — so a
typo in one copy added the wrong node instead of failing to compile.
`default_kind` also took a `&Graph` it never used.

**Deviation: no `Group`, no submenus.** MBG has 28 node kinds and needs them;
we have eight, and four submenus of two would cost a hover and a pointer trip
to reach any entry. `VARIANTS` is flat and ordered. Add grouping when the
catalog outgrows one list — it is a field on `Variant` and a nested loop in the
context menu.

Four tests hold it: every default is the variant it was asked for
(`Kind::of(default_kind(k)) == k`), every `Kind` appears in `VARIANTS`, every
default is actually addable through `Graph::add_layer`, and `unique_name` walks
past what is taken. `Kind::of` is exhaustive over `LayerKind`, so a new node
kind in core does not compile until it has a `Kind`, and the menu test then
forces it into `VARIANTS`.

Note there is a third copy of these strings, deliberately left alone:
`LayerKind::category_label()` in core, which the node header paints as its
subtitle. That one is the model describing itself and is fine where it is.

### 10. `refusal()` pre-flight with a message — DONE

`wires::eligible(graph, node, src) -> bool` became
`wires::refusal(graph, node, src) -> Option<String>`. Both the socket ring
(`socket_style` / `output_socket_style` in `nodes.rs`) and the drop-time status
message read from it, so the colour and the sentence cannot disagree about
what is wrong — previously the message was a hardcoded string sitting a hundred
lines away from the check that decided the colour.

**A cycle is still the only thing a wire drop can be refused for**, and that is
a fact about the model rather than a gap here. `Graph::set_kind` can return
`UnknownId` (impossible from a drop — both ends exist), `Cycle` (pre-checked),
and `RampTooFewStops` (connecting a wire never removes a stop). Every layer
produces a `Color` and every socket takes one, so there is no type mismatch to
catch. When there is, it goes in `refusal` and everything downstream already
works.

**No `key` parameter**, unlike MBG's. It needs the key to look up the socket's
type; we have nothing that varies by key, and an unused parameter is dead
weight. Both call sites have the key to hand if that changes.

What did improve for the user: the message now names both ends of the loop
(`crackle already reads albedo — that would make a loop`) instead of
`connection would create a cycle`, and self-loops say so separately. On a graph
with a dozen nodes the old message left you hunting for which existing wire was
the other half.

Four tests, and the first two are the ones worth keeping: a socket that would
refuse says so in advance **and `set_kind` agrees**, and a socket that rings
black **really does take the drop**. Those hold the preview against the
authority, which is the property that matters when a type system lands.

### Not adopted from MBG's version

`EditCmd::Connect` + a `Graph::connect` method in core. Ours batches a
detach-and-reconnect on the same node into one `SetKind` so the intermediate
state never exists (`wires::apply_socket_edits`), which MBG gave up when it
moved connection handling into the model. Keep ours unless that changes.

# Tier 3 — cheap tests (DONE)

Landed alongside the tiers above: `rename_command` (nodes.rs),
`hovered_output` (wires.rs), the `refusal` agreement pair (wires.rs), the
dirty-closure trio (state.rs) and the catalog's four (catalog.rs).

The `graph_canvas/layout.rs` sweep is now in too — six tests: every catalog
default lays out exactly the sockets `input_sockets()` reports, the same for the
Output pseudo-node, `socket_label` agrees with the model's own labels, a
conditional row actually changes `node_height` (Mix's Blend rows, Transform's
Radial rows), a ramp lays out one socket per stop, and the general invariant
below.

## The bug the sweep found

`Mix::input_sockets()` reports a factor socket in **every** blend mode, but
`rows_for` only emitted a row for it under `BlendMode::Blend`. Wire a layer into
a Blend mix's factor and switch the mode to Add, and that edge became invisible
and unreachable while staying completely real:

- the canvas had no anchor to draw the wire from, so it vanished;
- `mix_widgets` in the inspector hides the factor for non-Blend too, so there
  was nowhere left to disconnect it;
- `LayerKind::inputs()` still counted it, so it remained a scheduling
  dependency and a cycle edge, keeping an otherwise-unused branch alive;
- `eval_mix` ignored it, so it had no effect on the picture.

Fixed in `rows_for`: the factor row appears under Blend **or** whenever
`m.factor` is a `ScalarInput::Layer`. `socket_row` already renders a connected
factor as a bare label (the slider is `Const`-only), so the stranded wire now
shows up with somewhere to be pulled off.

The tempting fix — making `input_sockets()` conditional on mode — is **unsafe**:
`Graph::remove` scrubs references to a deleted layer by walking
`input_sockets()`, so hiding the socket there would leave a dangling `LayerId`
in the file that later `set_kind`/`add_layer` validation would reject.

This is also why `every_node_lays_out_exactly_the_sockets_it_has` only sweeps
catalog defaults. The rule that holds everywhere is the weaker, truer one:
**a socket with a wire in it must have a row to hang that wire on**
(`a_connected_socket_always_has_a_row_to_hang_its_wire_on`, which checks Mix
across all four modes).

Mix is the only kind with a conditional *socket* row — ColorRamp's stop rows
track `stops.len()` exactly, and Transform's conditionals are parameter rows
with no socket to strand.
