# Actor Refactor Milestone 2 Evidence

Date: 2026-08-23

Status: complete

## Scope

Milestone 2 removed the test-only renderer implementation of application command semantics. The
inventory started at 55 mutation emulators and now contains zero. The deleted paths covered
workspace topology and links, camera and plane navigation, channels and groups, native layers,
rendering preferences, label-resource changes, object presentation, filters, and selection.

Application tests that need both semantic and rendering behavior now use `ActorAppFixture`: a real
in-process `AppModel` executes the typed command, produces the canonical projection, and the
renderer consumes that projection. Pure validation and response behavior remain in model or actor
tests. Renderer tests remain responsible for projection application, resource installation,
readiness, clipping, transient interaction, and presentation behavior.

## Regression protection

`renderer_has_no_semantic_command_emulators` scans every Rust source file under
`src/app/renderer_bridge` and rejects functions in the known semantic mutation families. This is a
zero-baseline assertion rather than an allowlist, so the inventory cannot grow again.

The actor object tests additionally cover independently evaluated per-viewport filters, explicit
filter-source selection, the ambiguity error for an omitted viewport source, and standalone
filter-query selection. The application projection tests verify that those actor filter states are
consumed by the renderer.

## Defects exposed and corrected

- An actor-side active-channel change did not update the actor-native `channel:N` active layer,
  allowing projection to retain the wrong selected channel. The presentation command now updates
  both canonical values atomically.
- Actor-backed renderer tests found ordering assumptions in comparison setup. Object resources are
  now installed into both canonical and renderer fixtures before projection, and the multi-view
  filter test restores a single-view workspace before asserting screen-space geometry.

## Verification

The milestone tree passed:

- `cargo fmt --all -- --check`
- `cargo check --all-targets`
- `cargo test --lib -q` — 170 passed
- `cargo test --bin odon -q` — 187 passed, 4 ignored
- `cargo test --test data_contracts -q` — 10 passed

The Python SDK was not changed by this milestone. Its latest cumulative evidence remains 88 tests
passing; it will be rerun with generated-reference and surface checks at the final release gate.

## Next boundary

Milestone 3 removes the semantic mirrors themselves. It begins with workspace membership, titles,
active viewport, layout, ratio, and links, then proceeds through navigation, channels, rendering
preferences, and per-viewport layer presentation. Milestone 2 proves commands have one
implementation; it does not by itself prove that renderer projections contain no duplicate
semantic state.
