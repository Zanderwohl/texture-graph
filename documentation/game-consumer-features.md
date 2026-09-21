# Features needed to drive a game's procedural textures

## Context

`~/rust/rusty-space` (Exotic Matters + Lightcone Frontier) has six hand-tuned
procedural noise effects, all of them inline WGSL evaluated per-fragment, every
frame. The goal is to author them here instead: a `.tgraph` per effect, baked to a
texture at load, sampled by the game's existing shaders. That also makes them
transmissible — a graph is serde data, so it can be shipped by hash from a CDN
rather than compiled into the client.

This document is the gap list: what that consumer needs that this crate does not
have yet, in the order it blocks work.

**Consumer facts that constrain the design:**

- bevy 0.19.1 → wgpu 29.0.4. Our `wgpu = "29.0.3"` already matches, so the
  consumer links one copy. Keep it that way ([Cargo.toml](../Cargo.toml) already
  says so).
- `DeviceCtx::from_shared` is enough to adopt bevy's device: `RenderAdapter`,
  `RenderDevice`, `RenderQueue` are all `Clone` and bevy inserts them into the
  **main** world, not just the render world.
- lc-client builds for `wasm32-unknown-unknown` (`tools/build-wasm.sh`). Anything
  on the consumer's path must compile there — no threads, no `std::fs`, no
  blocking device poll.
- Every one of the six sites wants **one scalar field**, not four PBR channels.
- Four of the six sample noise on a **direction vector**, not UV.

## The six sites, and what each one needs

| site | what it is | blocked on |
| --- | --- | --- |
| population grain, ring speckle | 1-octave value noise warping a volume envelope | §1 |
| planet surface | fbm on sphere direction; banded variant adds `sin()` and a domain warp | §1 §2 §3 §4 §5 |
| star corona | ridged fbm, 3 octaves, on star direction | §2 §5 |
| engine plume | value noise periodic in z (period 64), 2 octaves, animated | §1 §2 §5 |
| hull | the planet material with the pattern disabled | nothing new |

## 1. A value-noise kernel with an integer lattice period

**The one feature that unblocks the most.** Three problems collapse into it:

- **Tiling.** Gustavson simplex has no period, so any baked volume seams the
  moment a consumer samples outside `[0, 1]³` — which the population grain (freq
  6) and ring speckle (freq 12) both do. Value noise on an integer lattice tiles
  exactly by taking the lattice cell modulo a period.
- **The plume.** Its noise is *already* periodic value noise with `CHURN_PERIOD =
  64` in z, because the animation scrolls through z forever. Nothing but a
  periodic kernel reproduces it.
- **Fidelity.** All six sites are value noise today. Simplex changes the look of
  every one of them. The planets and corona can be re-tuned in the editor — that
  is the point of the editor — but a value kernel means the consumer can port
  a look first and re-tune it later, instead of both at once.

```rust
pub enum NoiseKernel {
    /// Gustavson simplex. Today's behaviour; the default a file without the
    /// field loads as.
    Simplex,
    /// Trilinear value noise on an integer lattice, smoothstep weights.
    Value,
}

pub struct Noise {
    pub dims: NoiseDims,
    pub kernel: NoiseKernel,
    /// Lattice period in cells, per axis; `0` = unbounded on that axis.
    /// The plume needs exactly z — `[0, 0, 64]` — so this is per-axis rather
    /// than one flag. `Value` only; `Simplex` rejects a nonzero period.
    pub period: [u32; 3],
    // ...existing fields
}
```

Semantics: with `frequency = f` and `period = p`, the lattice cell index is taken
`mod p`, so the field repeats every `p / f` units of sample space. Seamless across
the unit cube is the case `p == f` with `f` integral; the UI should say so rather
than silently rounding.

`Simplex` + `period` is an error, not a silent no-op — tiled simplex needs a 6D
kernel for 3D and we are not writing one. Reject it in `Graph::set_kind` with a new
`GraphError::PeriodicSimplex`.

**Parity is the cost.** Per [noise.rs](../crates/core/src/noise.rs)'s op-order
contract, the CPU and WGSL kernels are twins and must land in the same commit. The
hash matters: use an integer hash, not a float one, so the two sides are
bit-exact rather than within-one-sRGB-step. The consumer's plume hash is a good
one — `(x, y, z mod period)` mixed with 1597334677 / 3812015801 / 2654435761, then
the xor-shift finaliser `>>15 *2246822519`, `>>13 *3266489917`, `>>16`.

**Test:** `f(x) == f(x + period / frequency)` exactly, on both backends, and the
existing `cpu_and_gpu_noise_agree` extended over the new kernel.

## 2. Fractal parameters on the Noise node

Every site but two is fbm. Expressing 5 octaves today costs 5 `Noise` layers and 4
`Mix::Add`s — it works (`NoiseRange::Signed` exists for exactly this), but it is
nine nodes of bookkeeping for one concept, and the octave frequencies have to be
maintained by hand.

```rust
pub struct Fractal {
    /// 1..=8. 1 is a plain single-octave sample.
    pub octaves: u32,
    /// Frequency multiplier per octave. The consumer uses 2.13 and 2.4 —
    /// not just 2.0, so this is a float, not a shift.
    pub lacunarity: f32,
    /// Amplitude multiplier per octave.
    pub gain: f32,
    pub mode: FractalMode,
    /// Divide by the sum of amplitudes so the result stays in range.
    pub normalize: bool,
}

pub enum FractalMode {
    /// Σ aᵢ · n(fᵢx)
    Standard,
    /// Σ aᵢ · |n|
    Turbulence,
    /// Σ aᵢ · (1 - |2n - 1|)² — the corona's filaments.
    Ridged,
}
```

**Put this on `Noise`, not in a new node.** A generic `Fbm { source }` node cannot
work in this architecture: the baker evaluates each layer once per pixel over a
fixed domain, so a downstream node cannot resample its input at a different scale.
That is the same constraint already documented on `bake_volume` ("a Transform's w
offset/scale can't re-sample its input at a different w"). Octaves have to be
inside the kernel dispatch, where the coordinates are still live.

With `period` set, the period doubles per octave alongside the frequency, so a
tiling fractal stays tiling.

## 3. Domain warp

The banded-planet surface is `sin((y + turbulence·0.055)·18 + drift)`, where both
`turbulence` and `drift` are noise fields. `Transform::offset` is a const
`[f32; 3]`, so there is no way to say "displace the sample by what that layer
says".

```rust
pub struct Warp {
    /// Sampled at the displaced coordinate. `None` → missing-texture grid.
    pub source: Option<LayerId>,
    /// Displacement field. Its L (and, for `Vector`, C and hue) drive the offset.
    pub by: Option<LayerId>,
    pub mode: WarpMode,
    /// Per-axis scale on the displacement, in sample-space units.
    pub amount: [f32; 3],
}

pub enum WarpMode {
    /// One scalar (L) displaces all three axes equally.
    Scalar,
    /// L, C, hue → x, y, z independently.
    Vector,
}
```

The GPU side is a texture fetch at an offset UV, which the `Domain` machinery in
[schedule.rs](../crates/gpu/src/schedule.rs) already exists to support: widen
`source`'s bake domain by `amount` the same way `EdgeMode::Extend` widens a
Transform's, so the fetch always lands inside baked territory. Clamp at the domain
edge and say so in the doc comment — a warp that reaches past `EXTEND_LIMIT` gets
the missing-texture grid, consistent with the rest.

CPU side is trivial: evaluate `by`, add, evaluate `source` at the moved sample.

## 4. A waveform node

`sin(...)` has no expression in the current node set, and a ColorRamp with enough
stops to fake it is not an answer.

```rust
pub struct Wave {
    pub input: ScalarInput,
    pub shape: WaveShape,   // Sine | Triangle | Square | Sawtooth
    pub frequency: f32,
    pub phase: f32,
    /// Output range. Bands want [0, 1]; a signed wave composes with Mix::Add.
    pub range: NoiseRange,
}
```

Sine alone covers the known need; the other three are nearly free once the node
exists. Note `sin` is transcendental, so this node is explicitly **outside** the
noise op-order contract — CPU/GPU agreement here is "within one sRGB step", not
bit-exact, and the parity test should assert the looser bound rather than pretend.

## 5. Scalar and volume output, and a readback that works on wasm

The bake API today returns four sRGB8 channels. A consumer that wants one grayscale
field pays 4× the memory and 4× the pack passes, and for volumes that is the
difference between viable and not:

| | 128³ | 256³ |
| --- | --- | --- |
| 4 × RGBA8 (today) | 33 MB | 268 MB |
| 1 × R8 | 2 MB | 16 MB |

Wanted:

```rust
impl Baker {
    /// Bake one layer's scalar (Oklch L) to a single-channel texture.
    pub fn bake_scalar(
        &mut self,
        graph: &Graph,
        layer: LayerId,
        size: (u32, u32),
        format: ScalarFormat,   // R8Unorm | R16Float | R32Float
        ctx: &EvalCtx,
    ) -> Result<wgpu::Texture, BakeError>;

    /// The same over a volume. `res³` sampled at slice centres, as bake_volume.
    pub fn bake_scalar_volume(
        &mut self,
        graph: &Graph,
        layer: LayerId,
        res: u32,
        depth: u32,
        format: ScalarFormat,
        ctx: &EvalCtx,
    ) -> Result<wgpu::Texture, BakeError>;
}
```

Note for the consumer's WebGPU target: `R32Float` is not filterable there, so
`R8Unorm` or `R16Float` are the useful formats and `R32Float` is for tests.

**Readback.** [readback.rs](../crates/gpu/src/readback.rs) is 2D-only and calls
`device.poll(wait_indefinitely())`, which does not work on wasm. Two additions:

- `read_scalar` and a 3D variant, mirroring `read_rgba8`.
- an `async` variant of each that awaits the map callback instead of polling, so a
  wasm host can drive it. The blocking ones stay for tests and CLI bakes.

A consumer sharing bevy's device can skip readback entirely and adopt the
`wgpu::Texture` directly — which is the fast path, and the reason `bake_*` should
keep returning textures rather than images.

## 6. Named parameters

Six planet surface classes differ by palette and contrast; every body differs by
seed. Today that is either six near-identical graphs plus a re-bake per body, or
`EvalCtx::seed` and nothing else.

```rust
pub struct ParamDecl {
    pub name: String,
    pub kind: ParamKind,     // Scalar { min, max } | Color | Vec3
    pub default: ParamValue,
    pub description: Option<String>,
}

// In Graph, serialized with it:
pub params: BTreeMap<String, ParamDecl>,

// Bound at bake time:
pub struct EvalCtx {
    pub seed: u32,
    pub normal_epsilon: f32,
    pub params: BTreeMap<String, ParamValue>,   // unbound → default
}
```

Any `ScalarInput` / `ColorInput` gains a `Param(String)` variant alongside `Const`
and `Layer`. On the GPU this is a uniform read, so it costs nothing per pixel; the
win is that one authored graph serves N instances, and a host can expose the
parameters as a UI without knowing the graph's internals.

This is also what makes a graph worth transmitting: the wire carries the graph
once, and per-instance variation is a small parameter block.

## 7. Housekeeping the consumer will trip over

- **`file.rs` uses `std::fs`.** It compiles on wasm but cannot work there. Gate
  `load_from_path` / `save_to_path` behind a default-on `std-fs` feature;
  `load_from_str` / `save_to_string` stay unconditional. A wasm consumer then
  depends on `texture-graph-core` with `default-features = false`.
- **Format version.** RON tags enum variants by name, so adding `LayerKind`
  variants keeps old files loading in new builds — but new files will not load in
  old ones, which is what `CURRENT_FORMAT_VERSION` is for. Bump it once when this
  batch lands, not once per node.
- **`bake_output`'s `_ctx`.** The parameter *is* used (it reaches
  `dispatch_kind` → `dispatch_noise`). The underscore reads as dead and will get
  someone to delete a seed. Rename to `eval_ctx` like its siblings.
- **crates.io.** The consumer will depend by git rev until these land. Publishing
  `texture-graph-core` / `-gpu` (the `-ui` crate is not needed downstream) is the
  point at which the rev pin turns into a version, and is worth doing once §1–§5
  are stable rather than before.

## Suggested order

1. §1 value kernel + period, §5 scalar output. Together these are enough to
   replace the population grain and ring speckle, which is the smallest useful
   end-to-end slice and proves the bake-and-sample path in bevy.
2. §2 fractal. Unblocks the corona on its own.
3. §3 warp, §4 wave. Unblocks the banded planet surface, the most visible one.
4. §6 params. Turns six graphs into one and makes the wire format worth having.
5. §7 and publishing.

The plume is last in the consumer's own ordering, not because it is blocked — §1
and §2 cover it — but because what exists works and the animated path deserves its
own performance comparison first.
