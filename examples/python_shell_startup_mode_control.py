"""Qualify isolated startup-profile restoration and per-mode shell retention.

This workflow launches Odon twice with a temporary ``ODON_SETTINGS_PATH``. The first process saves
and selects an application-scoped single-view profile. The second process proves that profile is
restored on first single-view activation, then transitions to project mode and back without losing
the single-view shell. No normal user settings are read or changed.

Validate the plan or run the complete two-process workflow:

    uv run --project python python examples/python_shell_startup_mode_control.py --plan-only
    uv run --project python python examples/python_shell_startup_mode_control.py
    uv run --project python python examples/python_shell_startup_mode_control.py --capture
"""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import odon
from odon import layouts, ui


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "fixtures" / "synthetic_5ch.ome.zarr"
DEFAULT_EXECUTABLE = ROOT / "target" / "debug" / "odon"
SETTINGS_ENV = "ODON_SETTINGS_PATH"
PROFILE_NAME = "Python startup qualification"

PREFIX = "layout:workflow.analysis"
BODY_ID = f"{PREFIX}.body"
LEFT_COLLAPSIBLE_ID = f"{PREFIX}.startup-left-collapsible"
LEFT_PANEL_ID = f"{PREFIX}.left"
LEFT_TABS_ID = f"{PREFIX}.left-tabs"
PROJECT_ID = f"{PREFIX}.left.native.1"
CANVAS_ID = f"{PREFIX}.canvas"
STARTUP_RATIO = 0.34


def startup_layout() -> ui.ShellLayout:
    """Build the visibly distinct layout persisted for startup restoration."""

    base = layouts.analysis()
    nodes: list[ui.ShellLayoutNode] = []
    for node in base.nodes:
        if node.id == BODY_ID:
            node = replace(
                node,
                children=(LEFT_COLLAPSIBLE_ID, node.children[1]),
                split=ui.ShellSplit(
                    ratio=STARTUP_RATIO,
                    resizable=True,
                    axis="horizontal",
                ),
            )
        elif node.id == LEFT_TABS_ID:
            node = replace(node, selected_id=PROJECT_ID)
        nodes.append(node)
    nodes.append(
        ui.ShellLayoutNode.collapsible(
            LEFT_COLLAPSIBLE_ID,
            LEFT_PANEL_ID,
            title="Startup-restored analysis",
        )
    )
    return ui.ShellLayout(base.root_id, tuple(nodes))


def plan_summary() -> dict[str, Any]:
    layout = startup_layout()
    split = layout.node(BODY_ID).split
    return {
        "dataset": DEFAULT_DATASET.name,
        "mode": "single",
        "layout_root": layout.root_id,
        "layout_nodes": len(layout.nodes),
        "profile_name": PROFILE_NAME,
        "settings_environment_variable": SETTINGS_ENV,
        "startup_ratio": split.ratio if split is not None else None,
        "startup_selected_id": layout.node(LEFT_TABS_ID).selected_id,
        "transition_focus_id": PROJECT_ID,
        "transition_sequence": ["project", "single", "project", "single"],
    }


def _terminate_launched(app: odon.Client) -> None:
    process = app.launched_process
    app.close()
    if process is not None and process.poll() is None:
        process.terminate()
        process.wait(timeout=5)


def _snapshot_state(
    app: odon.Client, stage: str, *, settings_path: Path
) -> dict[str, Any]:
    application = app.application.get_state()
    mode = application["mode"]
    shell = app.ui.shell.get(mode=mode)
    settings = app.application.get_settings()
    restore = settings["shell_layout_startup_restore"]["results"].get(mode)
    result: dict[str, Any] = {
        "stage": stage,
        "application_mode": mode,
        "shell_revision": shell.revision,
        "layout_root": shell.layout.root_id if shell.layout is not None else None,
        "active_region_id": shell.active_region_id,
        "focused_node_id": shell.focused_node_id,
        "startup_restore": restore,
        "isolated_settings": settings.get("settings_path") == str(settings_path),
    }
    if shell.layout is not None and shell.layout.root_id == startup_layout().root_id:
        split = shell.layout.node(BODY_ID).split
        result.update(
            {
                "split_ratio": split.ratio if split is not None else None,
                "selected_id": shell.layout.node(LEFT_TABS_ID).selected_id,
            }
        )
    return result


def _print_stage(
    app: odon.Client, stage: str, *, settings_path: Path, capture: bool
) -> None:
    print(
        json.dumps(
            _snapshot_state(app, stage, settings_path=settings_path),
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    if capture:
        print(f"ODON_SHELL_STARTUP_MODE_CAPTURE_STAGE={stage}", flush=True)
        input()


def _configure_startup_profile(
    executable: Path, settings_path: Path, dataset: Path
) -> None:
    app = odon.launch(
        executable,
        timeout=30.0,
        env={SETTINGS_ENV: str(settings_path)},
    )
    try:
        app.datasets.open_ome_zarr(dataset).wait(timeout=120.0)
        current = app.ui.shell.get(mode="single")
        installed = app.ui.shell.replace_layout(
            startup_layout(),
            mode="single",
            if_revision=current.revision,
            transaction_id="python-shell-startup-install",
        )
        saved = app.ui.shell.save_profile(
            PROFILE_NAME,
            scope="application",
            mode="single",
        )
        configured = app.application.update_settings(
            shell_layout_startup_profiles={"single": PROFILE_NAME}
        )
        settings = app.application.get_settings()
        if not saved.get("persisted"):
            raise RuntimeError("application startup profile was not persisted")
        if configured["shell_layout_startup_profiles"] != {"single": PROFILE_NAME}:
            raise RuntimeError("single-view startup profile selection was not persisted")
        if settings.get("settings_path") != str(settings_path) or not settings_path.exists():
            raise RuntimeError("Odon did not use the isolated settings path")
        print(
            json.dumps(
                {
                    "stage": "configured",
                    "layout_root": startup_layout().root_id,
                    "profile_name": PROFILE_NAME,
                    "shell_revision": installed.revision,
                    "isolated_settings": True,
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
    finally:
        _terminate_launched(app)


def run_live(*, executable: Path, dataset: Path, capture: bool) -> None:
    if not executable.exists():
        raise FileNotFoundError(executable)
    if not dataset.exists():
        raise FileNotFoundError(dataset)

    with tempfile.TemporaryDirectory(prefix="odon-startup-qualification-") as directory:
        settings_path = Path(directory) / "settings.json"
        _configure_startup_profile(executable, settings_path, dataset)

        app = odon.launch(
            executable,
            timeout=30.0,
            env={SETTINGS_ENV: str(settings_path)},
        )
        try:
            initial = app.application.get_state()
            if initial["mode"] != "project":
                raise RuntimeError(f"Odon did not start in project mode: {initial['mode']!r}")

            app.datasets.open_ome_zarr(dataset).wait(timeout=120.0)
            app.channels.set_visible(["DAPI", "CD3"], mode="only")
            app.viewer.fit()
            restored = _snapshot_state(
                app, "startup-restored-single", settings_path=settings_path
            )
            if restored["layout_root"] != startup_layout().root_id:
                raise RuntimeError("configured startup profile was not restored")
            if restored["startup_restore"]["status"] != "restored":
                raise RuntimeError("actor did not report successful startup restoration")
            _print_stage(
                app,
                "startup-restored-single",
                settings_path=settings_path,
                capture=capture,
            )

            current = app.ui.shell.get(mode="single")
            focused = app.ui.shell.patch_layout(
                active_region_id=PROJECT_ID,
                focused_node_id=PROJECT_ID,
                mode="single",
                if_revision=current.revision,
                transaction_id="python-shell-mode-transition-focus",
            )
            print(
                json.dumps(
                    {
                        "stage": "single-focused-before-transition",
                        "shell_revision": focused.revision,
                        "active_region_id": focused.active_region_id,
                        "focused_node_id": focused.focused_node_id,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                flush=True,
            )

            app.application.show_project_page()
            _print_stage(
                app,
                "project-mode",
                settings_path=settings_path,
                capture=capture,
            )

            app.datasets.open_ome_zarr(dataset).wait(timeout=120.0)
            app.channels.set_visible(["DAPI", "CD3"], mode="only")
            app.viewer.fit()
            returned = _snapshot_state(
                app, "single-returned", settings_path=settings_path
            )
            if returned["layout_root"] != startup_layout().root_id:
                raise RuntimeError("single-view layout was lost across the mode transition")
            if returned["startup_restore"]["status"] != "restored":
                raise RuntimeError("startup restoration result changed after reactivation")
            if returned["active_region_id"] != PROJECT_ID:
                raise RuntimeError("single-view active region was lost across mode transition")
            if returned["focused_node_id"] != PROJECT_ID:
                raise RuntimeError("single-view focus was lost across mode transition")
            _print_stage(
                app,
                "single-returned",
                settings_path=settings_path,
                capture=capture,
            )
        finally:
            _terminate_launched(app)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--executable", type=Path, default=DEFAULT_EXECUTABLE)
    args = parser.parse_args()
    if args.plan_only:
        print(json.dumps(plan_summary(), indent=2, sort_keys=True))
        return
    run_live(executable=args.executable, dataset=args.dataset, capture=args.capture)


if __name__ == "__main__":
    main()
