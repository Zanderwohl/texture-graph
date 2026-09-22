# texture-graph

## Before publishing to crates.io

Names `texture-graph`, `texture-graph-core`, `texture-graph-gpu` and `texture-graph-ui` were
all unclaimed on crates.io as of 2026-09-21.

### Description and metadata

- [ ] Add `description` to each crate's `Cargo.toml` (crates.io rejects a publish without one).
- [ ] Add `repository`, `homepage` / `documentation` and `readme` to `[workspace.package]`, and
      inherit them in each crate (`repository.workspace = true`, etc.).
- [ ] Add `keywords` (up to 5) and `categories` (e.g. `graphics`, `rendering`, `game-development`)
      per crate.
- [ ] Add `authors` if you want them shown.
- [ ] Add `rust-version`; edition 2024 means at least 1.85.

### README and docs

- [ ] Write this README: what it is, a screenshot of the editor, the core → gpu → ui layering,
      and a minimal example of loading a graph and baking it.
- [ ] Decide whether each crate gets its own README or points at this one (`readme = "../../README.md"`).
- [ ] Give `texture-graph-ui` a crate doc beyond one line; `core` and `gpu` already have one.
- [ ] Run `cargo doc --workspace --no-deps` and fix broken intra-doc links and warnings.
- [ ] Decide whether `documentation/` (design notes) should ship in the packages or stay repo-only.

### Licensing

- [ ] Add `LICENSE-MIT` and `LICENSE-APACHE` files; the manifests say `MIT OR Apache-2.0` but
      neither text is in the repo.

### Packaging

- [ ] Give the path dependencies a `version` too (`texture-graph-core = { path = "../core",
      version = "0.1.0", ... }`); crates.io refuses path-only dependencies.
- [ ] Publish in dependency order: `core`, then `gpu`, then `ui`.
- [ ] Stop committing `crates/ui/dist/` (21 MB of Trunk output, including a `.wasm`): add it to
      `.gitignore`, `git rm --cached` it, and make sure it can't end up in a package (`exclude`).
- [ ] Decide whether `texture-graph-ui` is published at all, or gets `publish = false`. If it is,
      the binary is named `texture-graph` but the crate isn't, so `cargo install texture-graph-ui`
      is the install command. Consider publishing it as `texture-graph` instead.
- [ ] Check what each package contains with `cargo package --list -p <crate>`.
- [ ] Dry-run each crate with `cargo publish --dry-run -p <crate>`.

### Cleanup

- [x] Comment pass (decimate-comments).
- [ ] Review the public API of `core` and `gpu`: everything `pub` becomes a semver promise at
      0.1. Make internals `pub(crate)` where nothing outside needs them.
- [ ] Consider `#[non_exhaustive]` on public enums that will grow (node kinds, params).
- [ ] Bump older dependencies before they're locked into a public API: `thiserror` 1 → 2,
      `ron` 0.8 → current.
- [ ] `cargo clippy --workspace --all-targets` clean.
- [ ] Add CI (test, clippy, doc, a wasm32 build of `ui`).
- [ ] Add a `CHANGELOG.md`.
