# Python SDK UI Action Safety and Workflow Recipes Plan

Status: implemented

Date: 2026-08-26

Target: Odon 0.x Python SDK stabilization

Implementation completed: 2026-08-26

The public implementation uses `UnsafeCallbackWaitError`, extension-owned runners with a default
`serial-worker` policy and a bounded queue of 128, built-in accumulation, explicit status/progress
component IDs, cancellation on removal/close, an inspectable Python source/style recipe, and
`odon.recipes.MarkerComparisonController`. The safe recipes and generic one-view controller are
the reference applications. An actor-owned compound source/style command was not required for this
milestone because neutral and final presentation barriers plus generation checks satisfy the
reference workflow; the documented non-atomic limitation remains explicit.

## Outcome

Python-authored Odon controls should make the safe path the obvious path. A developer should be
able to connect a button or select control to a resource load, analysis task, or presentation
change without knowing the SDK's internal callback-thread topology and without accidentally
blocking delivery of the task-completion event they are waiting for.

This work will add four layers of protection and convenience:

1. fail-fast detection when synchronous `Task.wait()` is called from an SDK event callback;
2. a standard serialized action worker for long-running UI actions;
3. normalized, typed UI interactions that hide the raw `action` versus `input` event envelope;
4. documented and tested recipes for navigation, task progress, coalescing, error handling, and
   safe object-source/style replacement.

A reusable marker-comparison controller will exercise the complete design by keeping the selected
ROI, marker, visible image channel, object source, continuous colour property, panel state, and
presentation readiness synchronized.

## Motivation

The motivating failure occurred in a Python-authored marker comparison panel. A native `Next
channel` button changed the image channel and then loaded a marker-scoped GeoParquet before applying
a continuous segmentation style.

The original control flow was:

```text
native button click
    -> synchronous SDK event callback
        -> start Odon object-load task
            -> Task.wait()
                -> wait for a tasks.* completion callback
```

Synchronous SDK callbacks are dispatched by one callback worker. `Task.wait()` also uses callbacks
on that worker to observe completion. If the task was not already complete when `wait()` refreshed
it, the current callback blocked the worker that needed to deliver its completion. Fast or cached
tasks sometimes finished before the refresh, which made the failure timing-dependent and therefore
particularly misleading.

A second failure appeared while replacing marker-scoped object tables. The new table did not
contain the previous marker's continuous colour property, but the retained presentation still
referred to it. Loading the new resource before resetting the old style left the actor model ready
while presentation could not advance.

These are SDK usability problems as well as application bugs. The low-level APIs are individually
valid, but their unsafe composition is neither rejected nor made sufficiently obvious.

## Goals

- Prevent a synchronous callback from deadlocking its own task-completion delivery.
- Make long-running native UI actions concise and safe by default.
- Serialize related state mutations unless a developer explicitly chooses concurrency.
- Let rapid navigation express whether every action matters or only the latest destination matters.
- Give every action a consistent busy, progress, success, failure, and cancellation lifecycle.
- Normalize UI events into a stable Python model with `component_id`, `action`, `value`, and `kind`.
- Preserve access to raw protocol events for advanced clients and backward compatibility.
- Provide a safe object-source/style swap pattern with separate semantic and presentation barriers.
- Demonstrate the design in a realistic one-view marker/fill comparison controller.
- Cover synchronous and asynchronous SDK behaviour explicitly rather than assuming parity.

## Non-Goals

- Running arbitrary Python on Odon's render thread.
- Hiding that resource loads and analyses are asynchronous Odon-managed tasks.
- Making every extension action concurrent.
- Moving large image or object data through event JSON.
- Adding a domain-specific marker controller to the control protocol itself.
- Replacing native `command` or `bind` actions that already execute entirely inside Odon.
- Making a timed-out task imply cancellation; cancellation remains explicit and cooperative.

## Design Principles

### Event callbacks acknowledge; workers perform

A raw SDK callback should decode or validate the interaction, enqueue work, and return. It should
not perform a blocking resource load, analysis, export, screenshot, or presentation wait.

### Safety must not depend on task speed

The same code must behave correctly whether a task completes before the first refresh, after one
second, or after one minute. Cached completion must not turn an unsafe pattern into an apparently
working one.

### Related viewer mutations are serialized

Channel, object source, object style, and UI status form one logical transition in comparison
workflows. They should not be mutated by competing workers unless the application deliberately
implements versioned concurrency.

### Semantic completion and presentation completion stay distinct

An object-load task can finish before the renderer presents the resulting geometry and property
payload. Helpers and recipes must name and await both barriers where rendered pixels matter.

### High-level helpers remain inspectable

Workers, coalescing, and contexts should be small Python abstractions over the existing task and
event contracts. Applications must still be able to inspect task IDs, errors, queue state, and raw
events.

## Workstream 1: Fail-Fast `Task.wait()` Guard

### Contract

Synchronous `Task.wait()` will detect whether it is running inside a synchronous SDK event
callback. If so, it will raise a structured SDK error before subscribing or blocking:

```text
Task.wait() cannot run inside an Odon event callback because task completion is delivered by the
same callback worker. Submit the operation to an action worker, use a native command/binding, or
return from the callback and wait elsewhere.
```

The check should be unconditional inside callback scope, even if a refresh might reveal that the
task has already completed. This removes timing-dependent behaviour.

### Implementation

- Add an internal callback-scope marker to `python/src/odon/events.py`.
- Set and clear the marker around each user callback with `try/finally`.
- Do not infer callback scope from a public thread name.
- Add an internal query such as `events.in_callback` for SDK components.
- Add a specific error, provisionally `UnsafeCallbackWaitError`, in the SDK error hierarchy.
- Check callback scope at the beginning of `Task.wait()` in `python/src/odon/tasks.py`.
- Include the task ID and remediation options in the error without exposing transport details.
- Confirm the async contract separately. `AsyncTask.wait()` should remain legal inside an async
  callback only if the event consumer continues dispatching while the callback awaits. Add a guard
  there as well if tests show equivalent starvation.

### Compatibility

Raw callbacks that only inspect state or issue non-blocking calls are unaffected. Code that relied
on a task completing quickly enough inside a callback will now fail deterministically with a useful
message instead of sometimes hanging.

## Workstream 2: Normalized UI Interactions

### Problem

Native components currently expose transport-oriented event envelopes. Buttons arrive under an
extension event ending in `.action`; selects, radios, and sliders arrive under `.input`. The
semantic action name is nested in `event.data["action"]["event"]`, while the component value is a
separate field.

Applications should not need to reproduce this decoding logic.

### Public model

Add a detached typed value, provisionally `UiInteraction`:

```python
@dataclass(frozen=True)
class UiInteraction:
    extension_id: str
    component_id: str
    action: str | None
    value: Any
    kind: Literal["action", "input"]
    event: Event
```

The original `Event` remains available for correlation IDs, revisions, source, and advanced data.

### API

Provide one or both of these forms after usability testing:

```python
extension.on_interaction(callback)
extension.on_action("next-channel", callback)
```

The extension helper will:

- subscribe to the correct `ui.extension:<extension-id>.*` wildcard;
- decode and validate the event envelope once;
- filter by semantic action or component ID;
- pass a `UiInteraction` to user code;
- return a removable subscription handle;
- expose malformed events through diagnostics rather than silently discarding them.

Existing `app.events.subscribe(...)` remains supported.

## Workstream 3: Serialized Action Execution

### Public API direction

Add a standard action runner owned by an extension or contribution. The intended concise form is:

```python
@extension.on_action("next-channel", execution="serial-worker")
def next_channel(context, interaction):
    context.status("Loading…")
    task = app.objects.load(path)
    task.wait()
    viewer.objects.color_by_continuous(property_name)
    context.status("Ready")
```

This exact spelling is provisional, but the behaviour is required.

### Execution policies

| Policy | Use | Contract |
| --- | --- | --- |
| `callback` | Tiny non-blocking handlers | Runs on the SDK callback worker; blocking waits are rejected. |
| `worker` | Independent background actions | Runs outside callback delivery; actions may overlap. |
| `serial-worker` | Ordered viewer workflows | Runs outside callback delivery; one action executes at a time. |

`serial-worker` should be the documented default for Python-authored UI actions that mutate Odon
state.

### Worker lifecycle

The SDK-owned worker will:

- start lazily when the first worker-backed action is registered;
- accept actions through a bounded queue;
- expose queue depth, running action, submitted/completed counts, and dropped/coalesced counts;
- catch exceptions without killing the worker;
- publish failures to an error callback and optional status component;
- stop accepting work when the extension disconnects or is removed;
- support bounded shutdown and cooperative cancellation of the active Odon task;
- never close the Odon client merely because one action fails.

### Action context

Provide an `ActionContext` with deliberately small responsibilities:

```python
context.status("Loading CD183…")
context.progress(0.5, "Installing object properties")
context.check_cancelled()
context.result("Ready")
```

The context should include:

- the normalized interaction;
- a monotonically increasing action generation;
- cancellation state;
- optional contribution/component IDs for status and progress patches;
- the currently retained Odon task, when present;
- structured error reporting.

It must not hide the underlying `Task` from applications that need its full snapshot.

## Workstream 4: Queueing and Coalescing Policies

Not every click should have the same queue semantics.

### Required policies

- `all`: execute every action in submission order.
- `latest`: retain only the newest pending action for a key.
- `accumulate`: combine relative navigation deltas, such as five `next` clicks becoming `+5`.
- `reject-while-busy`: ignore or reject new submissions with an explicit reason.

Example:

```python
@extension.on_action(
    "next-channel",
    execution="serial-worker",
    queue_key="marker-navigation",
    coalesce="accumulate",
    delta=1,
)
def next_channel(context, interaction, delta):
    controller.move_marker(delta)
```

For a select control, `latest` is usually appropriate. For an export button, `reject-while-busy`
or `all` may be appropriate. The SDK must not guess silently; recipes should recommend a policy.

### Stale-result protection

Each submitted action receives a generation. A latest-wins workflow can check the generation before
committing its final presentation. Results from an older generation may finish, but must not restore
an obsolete marker, channel, or status after a newer request has won.

## Workstream 5: Busy, Progress, Error, and Cancellation Conventions

Provide a context manager for the common lifecycle:

```python
with context.busy("Loading CD183…", ready="Ready"):
    task = app.objects.load(path)
    context.attach(task)
    result = task.wait(progress=context.report_task)
```

The helper will:

- set the busy/status component before work starts;
- optionally disable controls whose queue policy cannot accept more work;
- keep coalescing navigation controls enabled when appropriate;
- map task progress to a progress component;
- distinguish failure, cancellation, timeout, and disconnect;
- restore a ready or failed state in `finally`;
- retain a concise error for the panel and the structured exception for logs/tests.

Timeout messages must state that the retained task may still be running. Cancellation remains an
explicit call on the attached task.

## Workstream 6: Safe Object Source and Style Replacement

### Required transition

When the next object source does not contain the property referenced by the current style, use this
ordered transition:

```text
1. mark the panel busy;
2. hide the object overlay or reset its colour mapping to single mode;
3. wait for that neutral presentation to be presented;
4. clear/replace the object source;
5. wait for the retained load task to complete;
6. wait for resource, geometry, and presentation readiness;
7. apply the new continuous property, domain, and palette;
8. reveal the overlay;
9. wait for the final presentation revision;
10. publish the synchronized ready state.
```

### Initial delivery

Document and test this as a recipe using existing typed resources. The recipe must verify that the
requested property exists in the loaded source before committing the final style.

### Possible high-level method

After the recipe proves useful in more than one workflow, evaluate an atomic helper:

```python
viewer.objects.replace_source_and_style(
    path,
    color_mapping={
        "mode": "continuous",
        "property": property_name,
        "palette": "viridis",
        "domain": domain,
    },
    visible=True,
)
```

An SDK-only helper can order existing calls but cannot make the actor transition atomic. If users
must never observe an intermediate state, add one actor-owned command that validates the source and
desired style together, retains the task, and commits a generation-tagged projection.

## Workstream 7: Official Recipes

Add a focused guide under `docs/examples/` and executable examples under `python/examples/`.

### Recipe 1: Long-running button action

- Register a native button.
- Run its task on a serial worker.
- Patch status/progress.
- Handle timeout, cancellation, failure, and success.

### Recipe 2: Resource load from a UI interaction

- Start a retained resource task.
- Wait outside callback scope.
- Read the source state back.
- Await presentation only when pixels matter.

### Recipe 3: Previous/next navigation

- Use a stable ordered list.
- Wrap at the ends.
- Accumulate rapid relative clicks.
- Update the select control and underlying semantic state together.

### Recipe 4: Latest-wins select control

- Coalesce rapid select changes by key.
- Cancel work cooperatively where supported.
- Ignore stale generations at commit time.

### Recipe 5: Busy/status/error panel

- Keep the native panel responsive.
- Disable only incompatible controls.
- Present a short user-facing error and retain structured diagnostics.

### Recipe 6: Safe object source/style swap

- Neutralize a property-dependent style.
- Replace the source.
- verify the new numeric property;
- apply an explicit continuous domain;
- wait for the final presented revision.

Every recipe must include a deliberately unsafe counterexample and the exact error or failure mode
the safe version prevents.

## Workstream 8: Marker Comparison Controller

Build a reusable reference controller in `python/examples/` or a provisional `odon.recipes`
namespace. Do not make it a core protocol concept.

### Inputs

- ordered ROIs and markers;
- marker-to-image-channel mapping;
- marker/fill-to-object-property mapping;
- object source resolver;
- robust/full domains and palettes;
- component IDs for ROI, marker, fill, status, and progress;
- queue/coalescing policy.

### Owned state

```python
MarkerComparisonState(
    roi_id="18S1746__ROI3",
    marker="CD183 (APC) [S]",
    fill="nimbus",
    generation=12,
    phase="ready",
)
```

### Invariants

- The marker select and active/visible image channel describe the same marker.
- The object source contains the selected fill property.
- The continuous style references that exact property.
- The ready status is published only after final presentation readiness.
- An older generation cannot overwrite a newer selection.
- Fill changes reuse the current object resource when all fill properties are already loaded.
- Channel changes remain one-view operations; no second viewer is created.

This controller is the end-to-end acceptance example for normalized interactions, serialized
actions, coalescing, task waits, source/style transitions, and status handling.

## Test Plan

### Task/callback safety

- Start a task from a synchronous callback and assert `Task.wait()` raises immediately.
- Make the task already complete and assert the same deterministic error.
- Assert the error contains the task ID and worker remediation.
- Verify the callback worker continues delivering subsequent events.
- Verify ordinary `Task.wait()` outside callback scope is unchanged.
- Verify or explicitly reject the equivalent async callback pattern.

### Normalized interactions

- Decode button, toggle, select, radio, slider, and text-input events.
- Preserve component ID, semantic action, value, source event, and revision.
- Reject or diagnose malformed envelopes.
- Verify unsubscribe/removal stops delivery.
- Preserve raw wildcard subscriptions.

### Action worker

- Execute long tasks without blocking event delivery.
- Preserve serial ordering.
- Contain handler exceptions and continue with the next action.
- Exercise bounded queue behaviour.
- Stop safely during an active task and with pending work.
- Verify extension disconnect policies do not leak a worker or callback.

### Coalescing

- Accumulate repeated next/previous deltas correctly, including cancellation to zero.
- Retain only the latest select value for a queue key.
- Reject duplicate exports under `reject-while-busy`.
- Report submitted, executed, coalesced, and rejected counts accurately.
- Prevent stale generations from committing status or presentation.

### Object source/style transition

- Replace a source while the old continuous property is absent from the new source.
- Assert the neutral style is presented before the source replacement.
- Assert the new property exists before final style commit.
- Assert model, resources, geometry, canvas, and presentation all become ready.
- Cover failure, cancellation, property absence, and a second request during loading.

### End-to-end marker workflow

- Press next/previous repeatedly and verify marker, channel, source, property, and status agree.
- Switch raw, flat-field, and Nimbus fills without reloading geometry when already available.
- Rapidly request several markers and verify the configured coalescing policy.
- Change ROI while a marker load is pending and reject stale completion.
- Capture an Odon window screenshot after readiness and inspect the visible channel and panel state.

## Documentation Changes

- Add a callback-thread warning to the events and tasks sections of the Python API guide.
- Document all execution and coalescing policies next to declarative UI actions.
- Add `UiInteraction`, action worker, context, and error classes to the generated member reference.
- Add the six recipes to `docs/examples/` with runnable counterparts in `python/examples/`.
- Update Python API limitations to distinguish callback, worker, actor, and render-thread execution.
- Link the marker comparison example from the Python-authored native GUI guide.

## Implementation Sequence

### Milestone 1: deterministic guardrail

- Add callback-scope tracking and `UnsafeCallbackWaitError`.
- Guard synchronous `Task.wait()`.
- Add regression tests for incomplete and already-complete tasks.
- Add the immediate documentation warning.

Exit condition: blocking from a synchronous callback fails instantly and subsequent callbacks still
run.

### Milestone 2: normalized interaction model

- Add `UiInteraction` and envelope decoding.
- Add extension-scoped subscription handles and filters.
- Cover all interactive component families.
- Preserve raw event APIs.

Exit condition: a button and select can use the same typed handler without inspecting raw event
names or nested dictionaries.

### Milestone 3: action worker and lifecycle

- Add callback, worker, and serial-worker execution policies.
- Add `ActionContext`, queue diagnostics, cancellation, and structured failure handling.
- Integrate extension removal/disconnection cleanup.
- Add sync/async contract tests.

Exit condition: a UI action can wait for a retained task without blocking event delivery, and one
failed action does not stop later actions.

### Milestone 4: coalescing and status conventions

- Add all/latest/accumulate/reject-while-busy policies.
- Add action generations and stale-result guards.
- Add busy/progress/status helpers.
- Test rapid navigation and select changes.

Exit condition: rapid marker navigation reaches the intended destination without replaying every
intermediate resource load or committing stale results.

### Milestone 5: source/style recipe and marker controller

- Publish the safe object source/style replacement recipe.
- Build the marker comparison reference controller.
- Add live Odon integration and screenshot verification.
- Decide from evidence whether an actor-owned atomic replacement command is required.

Exit condition: the reference workflow can cycle ROI, marker, and fill repeatedly while channel,
source, style, panel state, and rendered presentation remain synchronized.

## Acceptance Criteria

- `Task.wait()` can no longer deadlock the synchronous event callback worker.
- The error for unsafe use is immediate, deterministic, structured, and actionable.
- Long-running extension actions require no hand-written thread or queue in the common case.
- Buttons and value controls expose one normalized interaction model.
- Serial actions preserve ordering and survive individual handler failures.
- Rapid navigation has an explicit, tested coalescing policy.
- Status never reports `Ready` before the final presentation revision is ready.
- A source/style swap cannot retain a property unavailable in the new source.
- Raw events and low-level task APIs remain available.
- Documentation contains runnable safe recipes and unsafe counterexamples.
- The marker comparison example passes repeated and rapid-interaction integration tests.

## Likely Code Areas

- `python/src/odon/events.py`: callback scope, normalized dispatch, subscription handles.
- `python/src/odon/tasks.py`: synchronous wait guard and structured error.
- `python/src/odon/async_events.py` and `async_tasks.py`: explicit async safety contract.
- `python/src/odon/ui.py`: `UiInteraction`, extension action registration, worker lifecycle.
- `python/src/odon/errors.py`: callback-wait/action execution errors.
- `python/tests/test_task_completion_contracts.py`: task/callback regression coverage.
- `python/tests/test_shell_ui.py`: normalized interactions and extension lifecycle.
- New focused action-worker and recipe tests under `python/tests/`.
- `docs/reference/python-api.md`, generated member reference, limitations, and examples.

## Resolved Decisions

- The structured error is `UnsafeCallbackWaitError`, directly under `OdonError`.
- `serial-worker` is the default for `extension.on_action()`; `callback` and `worker` remain
  explicit alternatives.
- Each extension owns its runner; contributions are optional patch targets rather than worker
  owners.
- The default queue bound is 128. Overflow is rejected as `ActionQueueFullError` and counted.
- `accumulate` is built in. An active relative action completes, while pending deltas for its key
  combine and may cancel to zero.
- Extension/action removal and client close reject new work, discard pending work, request
  cooperative cancellation of attached tasks, and perform bounded shutdown.
- Status and progress component IDs are explicit.
- Source/style replacement remains an inspectable Python recipe for this milestone; its lack of
  actor-level atomicity is documented.
- The reusable reference controller lives in `odon.recipes`; applications supply their own domain
  mappings and source resolvers.
