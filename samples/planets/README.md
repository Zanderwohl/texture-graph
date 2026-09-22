# Planet samples

Two classes of planet, not two planets. Every noise is 3D fbm on the sample
point, so a sphere shows no seam or pole pinch; `EvalCtx::seed` picks a member
of the class and the parameters move it along the class's obvious axes.

| file | what | parameters |
| --- | --- | --- |
| `earthlike.tgraph` | Oceans, continents with ridged ranges, desert belts, polar ice. Oceans are glossy and flat; land has a normal map. | `ocean`, `ice`, `aridity` |
| `earthlike-clouds.tgraph` | Translucent cloud deck, stretched east-west, for a shell over `earthlike`. | `cover` |
| `marslike.tgraph` | Rust world: south-high dichotomy, canyon networks, dark provinces, dusty caps. | `ice`, `dark`, `dust` (color) |

The graphs are not committed. `crates/core/examples/planets.rs` generates
them here; edit that rather than the `.tgraph` files:

```sh
cargo run -p texture-graph-core --example planets
```

## Screenshots

```sh
cargo run -p texture-graph-ui -- screenshot samples/planets/earthlike.tgraph earth.png \
    --seed 2 --yaw 130 --tilt 23 --background 05060a \
    --shell samples/planets/earthlike-clouds.tgraph
```

`texture-graph screenshot --help` lists the options. `screenshots.sh <dir>` renders
the whole sample set and contact sheets.

## Limits

- The color graphs use ColorRamp and Map, so `Baker::bake_scalar_cube` refuses
  them; they preview as solid (volume) textures. The mountain bump comes from the
  HeightToNormal's source height, bumped by its 3D gradient in the solid preview.
- The GPU `Map` clamps its value to `[0, 1]` before the lookup and the CPU
  evaluator does not, so every remap here keeps its range inside `[0, 1]`.
