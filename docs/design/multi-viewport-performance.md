# Multi-Viewport Performance Evidence

Status: first-milestone diagnostic baseline, 2026-08-21

This document records reproducible evidence for the two-viewport milestone. The
numbers are diagnostics from one machine and build profile, not portable
performance guarantees. Runtime counters are the authoritative way to inspect a
particular dataset and machine.

## Reproduce

Run the frame-planning benchmark explicitly:

```bash
cargo test --bin odon benchmark_single_and_two_viewport_frame_planning \
  -- --ignored --nocapture
```

Run the deterministic decode-sharing acceptance tests:

```bash
cargo test --all-targets tile_worker_accepts_two_live_viewport_generations
cargo test --all-targets composited_tiff_loader_shares_decode_across_viewport_presentations
cargo test --all-targets motivating_two_property_comparison_runs_end_to_end_on_one_document
```

The ignored benchmark drives 40 egui passes at 1200 by 800 points in single
layout, clones the active viewport, then drives 40 passes in a horizontal split.
It measures Odon's CPU workspace planning time. It does not claim to measure GPU
driver presentation or display refresh latency.

## Recorded result

Environment:

- arm64, macOS 15.5 (24F74)
- Rust and Cargo 1.93.1
- debug test profile
- checked-in `fixtures/synthetic_5ch.ome.zarr`

Result from 2026-08-21:

| Metric | Single viewport | Two linked viewports |
| --- | ---: | ---: |
| Workspace frame-plan EMA | 0.1870 ms | 0.2354 ms |
| Document instances | 1 | 1 |
| Dataset/source instances | 1 | 1 |
| Decoded CPU tile-cache instances | 1 | 1 |

The run reported one decoded entry occupying 8,192 bytes, one decode request,
and one source read. The end-to-end two-presentation test separately requests
the same raw channel twice with different colours and asserts two distinct RGBA
outputs, exactly one source read, and one decoded-cache hit.

## Live instrumentation

`viewer.workspace.get` exposes:

- document, dataset, CPU-cache and GPU-cache instance counts;
- composed, decoded and GPU raw cache entry counts;
- decoded-cache bytes;
- decode requests, source reads and cache hits;
- primary object count and geometry-instance count; and
- last, exponential-moving-average and sample count for workspace frame
  planning.

These counters allow an IPython session to confirm that adding a viewport did
not reopen the source or duplicate object geometry:

```python
workspace = app.viewer.workspace.get()
workspace["shared_resources"]
workspace["performance"]
```

## Native GPU smoke

The debug application was launched with the checked-in synthetic source and
then controlled through the installed Python resource surface. Python created a
55/45 horizontal comparison, selected `DAPI + CD3` and `DAPI + PanCK`
independently, set distinct sampling preferences, configured the canonical
camera/plane/selection link group, and captured the workspace through egui.

The live workspace reported one dataset instance, one GPU raw-tile cache and 18
raw entries. The 1,584 by 1,032 workspace PNG showed two adjacent, aligned
canvases with the same camera and visibly different channel composites. This
smoke run exposed and led to a fix for inherited horizontal child layout;
`horizontal_split_stacks_each_header_above_an_adjacent_full_height_canvas` now
guards the corrected canvas geometry.

## Acceptance interpretation

The measured split adds lightweight presentation and canvas-planning work while
keeping the source, decoded raw samples, GPU raw cache, object geometry and
scientific edits document-owned. Different colours, contrast windows and object
properties remain viewport-owned. Slow and failed decode sharing, as well as
removal of one viewport while work is live, have deterministic regression
tests.

A manual GPU smoke run remains useful for release qualification on each target
GPU because a headless test cannot characterize driver-specific frame pacing.
