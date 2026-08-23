# Actor Refactor Milestone 0 Checkpoint

Status: complete

Date: 2026-08-23

Base commit: `7737aab` (`refactor: migrate presentation and saved views to actor`)

Checkpoint commit: `8ce14b6` (`refactor: complete actor control cutover checkpoint`)

## Checkpoint scope

This checkpoint contains the current control-actor migration, removal of the production legacy
semantic path, Python/API additions required by the migrated surface, the background-control
verifier and evidence, and the responsibility-based source splits performed before ownership
cleanup.

The source checkpoint includes:

- `src/`, including the new actor, model, renderer-bridge, application, resource, and test modules;
- `python/src/odon/` and the corresponding Python tests;
- `api/application-surface.json`;
- `scripts/generate_python_api_reference.py` and `scripts/verify_background_control.py`;
- the generated Python API reference;
- the actor architecture, ownership, and completion plans; and
- `docs/testing/background-control-acceptance.md` and its four macOS JSON evidence files.

## Preserved local and unrelated files

The following untracked material was present during the audit and is deliberately excluded from
the actor checkpoint. It is not deleted or modified:

- local GeoJSON, project, synthetic-data, statistics, and deep-link demonstration files at the
  repository root;
- `docs/assets/images/screenshots/`;
- `presentations/`;
- `review.files/`;
- `screenshots/`; and
- the unrelated annotation, documentation-screenshot, and rechunking helper scripts.

These files are not evidence for the actor migration and must not be added by a broad `git add .`.

## Automated verification

| Check | Result |
| --- | --- |
| `cargo fmt --all -- --check` | Passed |
| `cargo check --all-targets` | Passed |
| `cargo test --lib` | 169 passed, 0 failed |
| `cargo test --bin odon` | 186 passed, 0 failed, 4 ignored |
| `cargo test --test data_contracts` | 10 passed, 0 failed |
| Python `unittest` discovery under `python/tests` | 88 passed, 0 failed |
| `scripts/generate_python_api_reference.py --check` | Passed |
| `api/application-surface.json` and macOS evidence JSON parsing | Passed |
| `git diff --check` | Passed |
| Explicit multi-viewport frame-planning benchmark | Passed |

The explicit benchmark reported:

- single-viewport frame-plan EMA: 0.2474 ms; and
- split-viewport frame-plan EMA: 0.2414 ms.

## Architectural evidence included in the suites

- 263/263 registered application methods have actor routes and no legacy/hybrid variants.
- The complete TCP comparison workflow advances without a UI frame.
- Actor projections coalesce to the latest workspace while unconsumed.
- Resource and compute families progress without frames.
- Backpressure, cancellation, stale completion, and presentation waiting remain responsive.
- The production legacy dispatcher, renderer snapshot-event publisher, and native snapshot
  translators are absent.
- `RootApp` requires the actor and retains only the named platform-effect route.
- Source-organization guards cover the newly split responsibility boundaries.

## Milestone 0 closure

The source and evidence scope above was committed without the preserved local files. Milestone 1
then added the executable ownership ledger in commit `54a9388`. Further work proceeds by ownership
slice; broad file splitting has stopped.
