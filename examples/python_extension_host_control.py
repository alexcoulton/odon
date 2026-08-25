"""Qualify an actor-owned default extension host and retained reconnect lifecycle.

Run Odon separately and connect:

    uv run --project python python examples/python_extension_host_control.py

Or launch the repository debug binary and pause at each evidence stage:

    uv run --project python python examples/python_extension_host_control.py --launch --capture

The workflow restores the previous shell, removes its extension, and terminates only an Odon
process that it launched itself.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import odon
from odon import layouts, ui


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "fixtures" / "synthetic_5ch.ome.zarr"
DEFAULT_EXECUTABLE = ROOT / "target" / "debug" / "odon"
EXTENSION_ID = "org.odon.python_shell_host_example"
EXTENSION_VERSION = "1.0.0"
CONTRIBUTION_ID = "host-proof"
EXPECTED_SHELL_MOUNT = f"extension:{EXTENSION_ID}/{CONTRIBUTION_ID}"
LEFT_TABS_ID = "layout:workflow.review.left-tabs"
LEFT_HOST_ID = "layout:workflow.review.left.default-extensions"


def extension_panel() -> ui.Panel:
    """Build the declarative panel rendered through the default left-sections host."""

    return ui.Panel(
        "host-proof-panel",
        title="Python extension host",
        children=[
            ui.Markdown(
                "explanation",
                "**This panel is contributed by Python and placed by Odon's actor-owned host.**",
            ),
            ui.Status("connection", "Connected and ready"),
            ui.Button(
                "fit",
                "Fit native viewer",
                action=ui.command("viewer.camera.fit"),
            ),
        ],
    )


def plan_summary() -> dict[str, Any]:
    layout = layouts.review()
    host = layout.node(LEFT_HOST_ID)
    panel = extension_panel().to_dict()
    return {
        "dataset": DEFAULT_DATASET.name,
        "mode": "single",
        "layout_root": layout.root_id,
        "layout_nodes": len(layout.nodes),
        "host_node_id": host.id,
        "host_mount": host.mount,
        "host_parent_id": LEFT_TABS_ID,
        "contribution_location": "left.sections",
        "expected_shell_mount": EXPECTED_SHELL_MOUNT,
        "component_ids": [panel["id"], *(child["id"] for child in panel["children"])],
    }


def _wait_for_contribution(
    app: odon.Client, expected: str | None, timeout: float = 5.0
) -> dict[str, Any] | None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        item = next(
            (
                dict(candidate)
                for candidate in app.ui.list_contributions()
                if candidate["shell_mount"] == EXPECTED_SHELL_MOUNT
            ),
            None,
        )
        if expected is None and item is None:
            return None
        if item is not None and item.get("readiness") == expected:
            return item
        time.sleep(0.05)
    raise TimeoutError(f"extension contribution did not reach readiness={expected!r}")


def _connect_extension(app: odon.Client, suffix: str) -> odon.Client:
    return odon.connect(
        instance=app.hello.instance_id,
        timeout=10.0,
        client_name=f"python-shell-host-{suffix}",
    )


def _register_extension(client: odon.Client) -> ui.Extension:
    return client.ui.register_extension(
        id=EXTENSION_ID,
        name="Python shell host example",
        version=EXTENSION_VERSION,
        capabilities=("ui.panels", "viewer.read", "viewer.write"),
        disconnect_policy="retain",
    )


def _print_stage(stage: str, data: dict[str, Any], capture: bool) -> None:
    print(json.dumps({"stage": stage, **data}, indent=2, sort_keys=True), flush=True)
    if capture:
        print(f"ODON_EXTENSION_HOST_CAPTURE_STAGE={stage}", flush=True)
        input()


def run_live(*, launch: bool, executable: Path, dataset: Path, capture: bool) -> None:
    if not dataset.exists():
        raise FileNotFoundError(dataset)
    if launch and not executable.exists():
        raise FileNotFoundError(executable)

    app = (
        odon.launch(executable, timeout=30.0)
        if launch
        else odon.connect(client_name="python-shell-host-controller")
    )
    launched_process = app.launched_process
    original_layout: ui.ShellLayoutDocument | None = None
    extension_client: odon.Client | None = None
    replacement_client: odon.Client | None = None
    replacement_extension: ui.Extension | None = None
    try:
        opened = app.datasets.open_ome_zarr(dataset).wait(timeout=120.0)
        original_layout = app.ui.shell.export_layout(mode="single")

        extension_client = _connect_extension(app, "initial")
        extension = _register_extension(extension_client)
        contribution = extension.register(
            extension_panel(),
            location="left.sections",
            contribution_id=CONTRIBUTION_ID,
        )
        if contribution.shell_mount != EXPECTED_SHELL_MOUNT:
            raise RuntimeError(
                f"unexpected extension shell mount: {contribution.shell_mount!r}"
            )

        current = app.ui.shell.get(mode="single")
        installed = app.ui.shell.replace_layout(
            layouts.review(),
            mode="single",
            if_revision=current.revision,
            transaction_id="python-shell-host-install",
        )
        selected = app.ui.shell.patch_layout(
            selected={LEFT_TABS_ID: LEFT_HOST_ID},
            focused_node_id=LEFT_HOST_ID,
            mode="single",
            if_revision=installed.revision,
            transaction_id="python-shell-host-select",
        )
        app.channels.set_visible(["DAPI", "CD3"], mode="only")
        app.viewer.fit()

        ready = _wait_for_contribution(app, "ready")
        _print_stage(
            "ready",
            {
                **plan_summary(),
                "opened": bool(opened),
                "shell_revision": selected.revision,
                "selected_host_id": selected.layout.node(LEFT_TABS_ID).selected_id
                if selected.layout is not None
                else None,
                "focused_node_id": selected.focused_node_id,
                "contribution_readiness": ready["readiness"] if ready else None,
                "owner_session_id": ready["ownership"]["owner_session_id"]
                if ready
                else None,
            },
            capture,
        )

        extension_client.close()
        extension_client = None
        disconnected = _wait_for_contribution(app, "disconnected")
        _print_stage(
            "disconnected",
            {
                "shell_revision": app.ui.shell.get(mode="single").revision,
                "contribution_readiness": disconnected["readiness"]
                if disconnected
                else None,
                "shell_mount_retained": disconnected["shell_mount"]
                if disconnected
                else None,
            },
            capture,
        )

        replacement_client = _connect_extension(app, "replacement")
        replacement_extension = _register_extension(replacement_client)
        reconnected = _wait_for_contribution(app, "ready")
        _print_stage(
            "reconnected",
            {
                "shell_revision": app.ui.shell.get(mode="single").revision,
                "contribution_readiness": reconnected["readiness"]
                if reconnected
                else None,
                "shell_mount_restored": reconnected["shell_mount"]
                if reconnected
                else None,
                "owner_session_id": reconnected["ownership"]["owner_session_id"]
                if reconnected
                else None,
            },
            capture,
        )
    finally:
        try:
            if replacement_extension is not None:
                replacement_extension.remove()
                _wait_for_contribution(app, None)
            if extension_client is not None:
                extension_client.close()
            if replacement_client is not None:
                replacement_client.close()
            if original_layout is not None:
                current = app.ui.shell.get(mode="single")
                app.ui.shell.import_layout(
                    original_layout,
                    mode="single",
                    if_revision=current.revision,
                    transaction_id="python-shell-host-restore",
                )
        finally:
            app.close()
            if launched_process is not None and launched_process.poll() is None:
                launched_process.terminate()
                launched_process.wait(timeout=5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--executable", type=Path, default=DEFAULT_EXECUTABLE)
    args = parser.parse_args()
    if args.plan_only:
        print(json.dumps(plan_summary(), indent=2, sort_keys=True))
        return
    run_live(
        launch=args.launch,
        executable=args.executable,
        dataset=args.dataset,
        capture=args.capture,
    )


if __name__ == "__main__":
    main()
