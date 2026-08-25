# Python application-shell workflows

Odon ships reusable layouts for common review, analysis, comparison, mosaic-triage, and
presentation sessions. Each builder returns a complete `ShellLayout`; applying it remains an
explicit revision-guarded transaction.

```python
import odon
from odon import layouts

with odon.connect() as app:
    current = app.ui.shell.get(mode="single")
    desired = layouts.review()
    app.ui.shell.replace_layout(
        desired,
        mode="single",
        if_revision=current.revision,
        transaction_id="install-review-workspace",
    )
```

An extension panel can be inserted into the appropriate workflow tabs using the stable mount ID
returned by registration:

```python
contribution = extension.register(panel, contribution_id="quality-control")
desired = layouts.analysis(panel_mounts=[contribution.shell_mount])
app.ui.shell.replace_layout(desired, mode="single")
```

Other supplied layouts are constructed the same way:

```python
layouts.comparison()
layouts.mosaic_triage()
layouts.presentation(mode="single", show_toolbar=False)
layouts.presentation(mode="mosaic", show_toolbar=True)
```

The review, analysis, comparison, and mosaic-triage layouts include actor-owned default extension
hosts. Contributions explicitly supplied through `panel_mounts` are excluded from their default
host at render time, preventing duplicate UI. Presentation layouts intentionally contain only the
required canvas and optional native toolbar.

## Dataset-specific controlled-shell example

[`examples/python_shell_control.py`](../../examples/python_shell_control.py) opens the checked-in
five-channel OME-Zarr fixture, installs a nested review shell, mounts a Python-defined command
toolbar, and binds the right inspector panel's visibility to the actor-owned checked state of
`viewer.scale_bar.toggle`. The toolbar deliberately mixes:

- ready native `project.save`;
- protected control command `app.shell.recover`;
- checked native `viewer.scale_bar.toggle`; and
- predicate-disabled `viewer.masks.export_geojson` while no mask layer exists.

Validate and inspect the desired state without launching Odon:

```bash
uv run --project python python examples/python_shell_control.py --plan-only
```

Run the complete workflow against a running app, or let it launch the repository debug binary:

```bash
uv run --project python python examples/python_shell_control.py
uv run --project python python examples/python_shell_control.py --launch
```

The live macOS run on 2026-08-25 produced this actor-owned output before restoring the previous
layout and toolbar:

```json
{
  "active_region_id": "layout:workflow.review.canvas",
  "dataset": "synthetic_5ch.ome.zarr",
  "layout_nodes": 23,
  "layout_root": "layout:workflow.review.root",
  "mode": "single",
  "opened": true,
  "right_panel_visible": true,
  "scale_bar_checked": true,
  "shell_revision": 2,
  "toolbar_commands": [
    "project.save",
    "app.shell.recover",
    "viewer.scale_bar.toggle",
    "viewer.masks.export_geojson"
  ],
  "toolbar_groups": 2,
  "toolbar_revision": 2
}
```

The optional `--capture` mode records both effective states from one run. In the first frame the
scale-bar action is selected, mask export is actor-disabled, and the bound inspector occupies the
right side. The example then executes the same checked command through `ui.commands.execute`.

| Checked command and visible inspector | Unchecked command and hidden inspector |
| --- | --- |
| ![Python-controlled shell with the bound inspector visible](../assets/images/screenshots/raw/python-shell-binding-expanded-macos.png) | ![Python-controlled shell after the actor hides the bound inspector](../assets/images/screenshots/raw/python-shell-binding-hidden-macos.png) |

After the command, the actor reported:

```json
{
  "right_panel_visible": false,
  "scale_bar_checked": false,
  "shell_revision": 2
}
```

The shell document revision remains 2 because the command changes actor-derived effective
visibility, not the persisted desired layout. No Python frame callback or follow-up layout patch is
involved; the canvas takes the released inspector space during native reconciliation.

## Actor-owned default extension host lifecycle

[`examples/python_extension_host_control.py`](../../examples/python_extension_host_control.py)
registers a native egui panel from a separate Python extension session and lets the review layout's
existing `builtin:extension-host.left-sections` mount place it. The controller selects that host by
its stable desired-tree node ID; the contribution is not inserted as an explicit layout node.

Inspect the typed plan without a running app, run the lifecycle against an existing instance, or
launch the repository debug binary and pause at each capture stage:

```bash
uv run --project python python examples/python_extension_host_control.py --plan-only
uv run --project python python examples/python_extension_host_control.py
uv run --project python python examples/python_extension_host_control.py --launch --capture
```

The macOS run records three states while preserving shell revision 3:

1. `ready`: the default host renders the Python panel and owns actor focus;
2. `disconnected`: closing only the extension session retains the same shell mount and renders an
   explicit disconnected banner; and
3. `reconnected`: a new session registering the same extension ID and version reclaims the retained
   contribution under a new owner session, removes the banner, and restores interaction.

| Ready | Retained after disconnect | Reclaimed after reconnect |
| --- | --- | --- |
| ![Ready Python panel in the actor-owned extension host](../assets/images/screenshots/raw/python-extension-host-ready-macos.png) | ![Retained extension host showing its disconnected state](../assets/images/screenshots/raw/python-extension-host-disconnected-macos.png) | ![Compatible Python extension session reconnected to the retained host](../assets/images/screenshots/raw/python-extension-host-reconnected-macos.png) |

This capture also found and fixed a compatibility ownership bug: passive projection of the legacy
Layers/Project enum used to overwrite a desired-tree selection that named the extension host.
Passive compatibility projection now updates only its compatibility fields. An explicit legacy tab
command still deliberately updates the desired layout, and a regression test enforces both sides of
that rule.

## Split, selection, focus, collapse, and protected recovery

[`examples/python_shell_interaction_control.py`](../../examples/python_shell_interaction_control.py)
adds a collapsible left region to the nested review layout and records four revision-guarded states.
Its Python patches target the same actor-owned split, tab, collapse, active-region, and focus fields
that native gestures commit; the existing Rust interaction tests prove native split release, tab
click, collapse click, and leaf activation submit that same patch/event path. The example itself is
therefore reproducible Python-initiated state evidence, not a claim that an automated pointer
gesture produced these particular frames.

Inspect the typed plan, run it against an existing app, or launch Odon and pause at every frame:

```bash
uv run --project python python examples/python_shell_interaction_control.py --plan-only
uv run --project python python examples/python_shell_interaction_control.py
uv run --project python python examples/python_shell_interaction_control.py --launch --capture
```

The macOS run on 2026-08-25 produced revisions 2 through 5:

1. `baseline` installs the open 23-node shell with Layers selected and split ratio 0.24;
2. `resized-selected-focused` atomically changes the ratio to 0.36, selects Project, and sets both
   actor-owned active and focused IDs to that Project mount;
3. `collapsed` retains the 0.36 split and Project selection while collapsing `Review panels` and
   transferring active/focused state to the canvas; and
4. `recovered` invokes protected `app.shell.recover` through `ui.commands.execute`, receives the
   `control` handler result, and installs the two-node canvas-only recovery tree.

| Baseline: ratio 0.24 and Layers selected | Ratio 0.36, Project selected and focused |
| --- | --- |
| ![Baseline nested shell before interaction-state patches](../assets/images/screenshots/raw/python-shell-interaction-baseline-macos.png) | ![Wider left split with Project selected and actor focus](../assets/images/screenshots/raw/python-shell-interaction-resized-focused-macos.png) |

| Collapsed Review panels at ratio 0.36 | Protected canvas-only recovery layout |
| --- | --- |
| ![Collapsed left review region](../assets/images/screenshots/raw/python-shell-interaction-collapsed-macos.png) | ![Protected minimal recovery layout after shared command dispatch](../assets/images/screenshots/raw/python-shell-interaction-recovered-macos.png) |

The stage output is the machine-readable half of the evidence: it reports the exact split ratio,
selection, collapse flag, active/focused IDs, layout root, revision, recovery handler type, and
recovery canvas mount. The workflow imports the prior shell in `finally` and terminates only an app
process it launched itself.

## Isolated startup restore and mode retention

[`examples/python_shell_startup_mode_control.py`](../../examples/python_shell_startup_mode_control.py)
proves durable application-profile restoration across a real process restart, then switches from
single-view to project mode and back. It launches both processes with a temporary
`ODON_SETTINGS_PATH`, so the qualification run neither reads nor changes the normal OS-user Odon
settings file. The first process activates single-view mode, installs a distinctive 23-node
analysis layout, saves it as an application profile, and selects that profile for single-view
startup. The second process is the evidence process.

```bash
uv run --project python python examples/python_shell_startup_mode_control.py --plan-only
uv run --project python python examples/python_shell_startup_mode_control.py
uv run --project python python examples/python_shell_startup_mode_control.py --capture
```

On macOS, the second process reported `status: restored`, schema version 1, no migration or
protected recovery, the expected `layout:workflow.analysis.root`, split ratio 0.34, and Project tab
selection when single-view mode was first activated. It then set Project active/focused at revision
3, switched to the default project tree, and returned to the same single-view tree at revision 3
with those per-mode IDs retained.

| Application profile restored in the second process | Default project-mode shell |
| --- | --- |
| ![Isolated application profile restored at first single-view activation](../assets/images/screenshots/raw/python-shell-startup-restored-macos.png) | ![Project shell between single-view activations](../assets/images/screenshots/raw/python-shell-mode-project-macos.png) |

| Same single-view shell after returning from project mode |
| --- |
| ![Startup-restored single-view shell retained across the mode transition](../assets/images/screenshots/raw/python-shell-mode-single-returned-macos.png) |

The initial restore intentionally restores the layout document rather than transient focus. The
workflow sets active/focused state after restoration, then proves that actor-owned state is retained
when the mode is deactivated and reactivated. The temporary settings directory and both launched
processes are cleaned up in `finally` paths.
