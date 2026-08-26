# Safe Python UI action recipes

Python-authored native controls should acknowledge interactions quickly and move resource or
presentation work onto an extension-owned action worker. The complete runnable counterpart to all
six recipes is
[`python/examples/ui_action_recipes.py`](../../python/examples/ui_action_recipes.py).

## Choose execution and queue semantics explicitly

| Execution | Meaning | Appropriate work |
| --- | --- | --- |
| `callback` | Run on synchronous event callback delivery. `Task.wait()` is rejected. | Tiny validation or enqueue-only handlers. |
| `worker` | Run independently in the SDK pool; actions may overlap. | Unrelated computations or exports. |
| `serial-worker` | Run off callback delivery, one extension action at a time. This is the default. | Viewer, source, style, channel, and panel transitions. |

| Coalescing | Meaning | Typical control |
| --- | --- | --- |
| `all` | Retain every accepted action in order. | Audit/save operations that must not be skipped. |
| `latest` | Keep only the newest pending generation for a queue key. | Select, radio, slider, or text value. |
| `accumulate` | Add pending relative deltas; opposite clicks may cancel to zero. | Previous/next buttons. |
| `reject-while-busy` | Reject another action for the same busy key with `ActionRejectedError`. | Export or one-shot analysis. |

`extension.action_status()` returns submitted, executed, completed, failed, cancelled, rejected,
coalesced, queue-depth, running-action, and closed diagnostics. `executed` counts handlers that
actually started; coalesced or queue-rejected submissions do not inflate it. Removing an action,
removing its extension, or closing the client stops submissions, removes callbacks, cancels pending
work, and cooperatively cancels an attached Odon task when that task supports cancellation.
Shutdown has a bounded wait; Python cannot forcibly terminate an uncooperative thread.

## 1. Long-running button action

```python
@extension.on_action(
    "run-analysis",
    execution="serial-worker",
    coalesce="reject-while-busy",
    contribution=panel,
    status_component_id="status",
    progress_component_id="progress",
)
def run(context, interaction):
    with context.busy("Loading…"):
        task = context.attach(app.objects.load(path))
        result = task.wait(timeout=60, progress=context.report_task)
```

`context.attach()` preserves access to the underlying task and lets extension shutdown request
cooperative cancellation. A wait timeout only means Python stopped waiting; its message states that
Odon work may still be running. Call `task.cancel()` explicitly when cancellation is desired.

Unsafe counterexample:

```python
app.events.subscribe("ui.extension:org.example.*", lambda event: task.wait())
```

This deterministically raises `UnsafeCallbackWaitError` immediately, even when the task happens to
be complete. Without the guard, an incomplete task could wait for a completion event queued behind
the callback that was doing the waiting.

## 2. Resource load from a UI interaction

```python
@extension.on_action("source-selected", coalesce="latest", queue_key="source")
def load(context, interaction):
    task = context.attach(app.objects.load(interaction.value))
    task.wait(timeout=60, progress=context.report_task)
    context.ensure_current()
    source = app.objects.get_state()            # semantic/resource barrier
    readiness = odon.wait_for_viewer_readiness(app)  # presentation barrier
    context.result("Ready")
```

Task completion and rendered pixels are different contracts. Read the source after the task, and
wait for model, resources, geometry, canvas, and presentation only when the final visual result
matters.

Unsafe counterexample: treating `app.objects.load(path)` as a completed load and styling
immediately can race the immutable object resource and renderer geometry installation. The style
may reference a property the renderer has not installed yet.

## 3. Previous/next navigation

Register both directions with the same handler and queue key:

```python
def move(context, interaction):
    index = (state.index + int(context.delta)) % len(channels)
    channel = channels[index]
    viewer.set_visible_channels([channel], mode="only")
    viewer.set_active_channel(channel)
    context.patch({"channel": channel})

extension.on_action(
    "previous-channel", move,
    queue_key="channel-navigation", coalesce="accumulate", delta=-1,
)
extension.on_action(
    "next-channel", move,
    queue_key="channel-navigation", coalesce="accumulate", delta=1,
)
```

The already-running relative step is allowed to finish. Clicks waiting behind it are accumulated
against the state that step commits: five pending Next clicks can become `+5`, and pending Next
followed by Previous can reduce to zero before another resource load. This preserves every relative
click while still skipping intermediate pending destinations. Patch the native select from the
same winning commit that changes semantic state.

Unsafe counterexample: launching one thread per click lets loads complete out of order, so a slower
old channel can overwrite the newest select value and visible-channel list.

## 4. Latest-wins select control

```python
@extension.on_action(
    "marker-selected",
    queue_key="marker-comparison",
    coalesce="latest",
    contribution=panel,
    status_component_id="status",
)
def select_marker(context, interaction):
    task = context.attach(load_marker(interaction.value))
    task.wait(progress=context.report_task)
    context.ensure_current()
    context.commit(lambda: install_marker(interaction.value))
```

Every accepted submission gets a monotonically increasing generation. `ensure_current()` raises
`StaleActionError` after a newer generation wins. `context.patch()` and `context.commit()` check the
generation while the synchronous runner's submission state is stable, preventing an old action
from restoring an obsolete status or local commit.

Unsafe counterexample: checking a shared `selected_marker` once before a slow load has a time-of-
check/time-of-use race. The selection can change while the load is running, after the check but
before the final style mutation.

## 5. Busy, status, progress, error, and cancellation

```python
diagnostics = []

def failed(error, context):
    diagnostics.append(error)  # structured exception for logs/tests
    if context is not None:
        context.patch({"status": f"Failed: {type(error).__name__}"})

extension.on_action(
    "export",
    export,
    coalesce="reject-while-busy",
    contribution=panel,
    status_component_id="status",
    progress_component_id="progress",
    on_error=failed,
)
```

Handler exceptions are contained: the action is marked failed, the worker continues to its next
action, and the Odon client remains connected. Cancellation, stale generations, queue rejection,
queue overflow, task failure, timeout, and disconnect remain distinct error conditions.

Unsafe counterexample: allowing an exception to escape a hand-written daemon worker terminates that
worker. The first failure then makes every later button appear dead, often with no native error.

## 6. Safe object source and continuous-style swap

```python
odon.replace_object_source_and_style(
    app,
    path,
    property_name,
    palette="viridis",
    domain=(low, high),
    context=context,
    presentation_objects=viewer.objects,
)
```

The helper performs this inspectable sequence:

1. hide the active object presentation and reset its mapping to `single`;
2. wait until the neutral projection is presented;
3. clear and load the new source through a retained task;
4. wait for semantic resource, geometry, canvas, and presentation readiness;
5. page the new source's properties and require the requested property to be numeric;
6. apply the explicit continuous mapping, reveal the overlay, and wait for the final projection;
7. publish `Ready` only if the action generation still wins.

If verification fails, `ObjectPropertyUnavailableError` includes the requested and available
properties, and the overlay remains hidden with its neutral mapping. The result retains the task,
task result, object state, property descriptor, and final readiness evidence.

Unsafe counterexample:

```python
app.objects.load(next_path).wait()
viewer.objects.color_by_continuous(old_property)
```

When `old_property` is absent from the new table, the resource and retained presentation disagree.
Depending on timing, the renderer rejects the projection, stays hidden, or displays stale styling.

## Marker comparison controller

`odon.MarkerComparisonController` is a reusable reference pattern, not a protocol feature. It is
configured with ordered ROIs/markers/fills and callables that resolve image channels, object
sources, numeric properties, domains, and palettes. It owns one semantic state:

```python
odon.MarkerComparisonState(
    roi_id="18S1746__ROI3",
    marker="CD183 (APC) [S]",
    fill="nimbus",
    generation=12,
    phase="ready",
)
```

`install_actions()` supplies latest selects plus accumulated previous/next controls on one serial
queue key. Fill changes reuse a source when that source already contains every fill property;
marker or ROI changes use the safe source/style transition. A ready state commits channel, source,
property, style, and panel values together, and the controller never creates a second viewport.
Applications can extend this pattern with their own marker-scoped table generation and additional
palette or range controls without adding a second viewer.
