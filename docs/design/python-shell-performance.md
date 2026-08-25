# Python-controlled shell performance evidence

Status: local diagnostic baseline, 2026-08-25

This document records reproducible CPU and backpressure evidence for Odon's declarative
single-window shell. The values are a development-machine baseline, not portable performance
guarantees. The tests enforce broad five-second non-blocking budgets so unusually slow validation
or accidental blocking fails visibly without pretending that debug-build timing is a frame-time
contract.

## Reproduce

Run the maximum shell and command-surface timing gates:

```bash
cargo test maximum_ -- --nocapture
```

Run sustained slow-subscriber pressure:

```bash
cargo test slow_subscribers_drop_events_without_blocking_publishers -- --nocapture
```

Related deterministic pressure tests are:

```bash
cargo test high_frequency_component_interactions_coalesce_per_component
cargo test comparison_workflow_completes_over_tcp_without_a_ui_frame
cargo test extension_disconnect_and_reconnect_reconcile_focus_and_retained_readiness
```

## Recorded local result

Environment:

- arm64, macOS 15.5;
- Rust and Cargo 1.93.1;
- debug test profile; and
- no renderer, GPU presentation, or Python frame callback.

Result from 2026-08-25:

| Workload | Samples | Total | Average |
| --- | ---: | ---: | ---: |
| Validate a 256-node shell tree | 64 | 196,378 µs | 3,068 µs |
| Validate a 128-item toolbar | 128 | 56,299 µs | 439 µs |
| Reconcile the complete 32-node predicate allowance | 512 | 367,068 µs | 716 µs |
| Publish into a saturated one-event subscriber queue | 10,000 | 85,979 µs | 8,597 ns |

The saturated subscriber retained one event and recorded 9,999 dropped events. Publishing did not
wait for the consumer. The component-interaction pressure test independently submits 1,000
debounced updates for one component and proves that one map entry containing the newest value is
retained. The paused-frame TCP workflow proves that repeated actor mutations coalesce to one latest
render projection. Disconnect/reconnect coverage proves that transport loss reconciles focus and
retained extension readiness rather than leaving work waiting on the absent client.

## Interpretation and remaining evidence

The measurements cover validation, actor-side command-state projection, and bounded queue
behavior. They do not measure egui layout, operating-system menu realization, GPU frame pacing, or
network behavior on Windows and Linux. Those belong to the cross-platform rendered release suite
and remain explicitly tracked in the application-shell checklist.
