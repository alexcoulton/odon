"""Create a linked two-viewer channel comparison in a running Odon instance.

Run the complete example from the repository with:

    uv run --project python python examples/two_viewer_comparison.py

Alternatively, open this file in Neovim and send its ``# %%`` cells to an
IPython session with vim-slime. The repository's five-channel synthetic
OME-Zarr fixture is selected by default. Odon must already be running with its
control API enabled; the script opens the fixture itself.

The example changes only viewport presentation and workspace layout. It does
not modify image data, segmentation geometry, annotations, or masks.
"""

# %% Imports and configuration.
from __future__ import annotations

import argparse
from pprint import pprint
from pathlib import Path
from typing import Any

import odon


def _repository_fixture() -> Path:
    """Find the checked-in fixture from a script or repository IPython shell."""
    roots: list[Path] = []
    if "__file__" in globals():
        roots.append(Path(__file__).resolve().parents[1])
    working_directory = Path.cwd().resolve()
    roots.extend((working_directory, *working_directory.parents))
    for root in roots:
        candidate = root / "fixtures" / "synthetic_5ch.ome.zarr"
        if candidate.exists():
            return candidate
    return working_directory / "fixtures" / "synthetic_5ch.ome.zarr"


# The checked-in fixture works without configuration. Replace this value to use
# another local OME-Zarr or TIFF source, or pass a path on the command line.
IMAGE_SOURCE: str | Path = _repository_fixture()
IMAGE_KIND = "auto"  # "auto", "ome-zarr", or "tiff"
OPEN_TIMEOUT_SECONDS = 120.0

# Set either value to an exact channel name or channel index. Leave both as
# None to use the first two channels. A single-channel image reuses that channel
# in both viewports with different presentation settings.
LEFT_CHANNEL: str | int | None = None
RIGHT_CHANNEL: str | int | None = None

LEFT_TITLE = "Channel A"
RIGHT_TITLE = "Channel B"
LAYOUT = "horizontal"  # "horizontal" or "vertical"
SPLIT_RATIO = 0.5


# %% Normal-script argument handling; this cell does nothing inside IPython.
def _running_in_ipython() -> bool:
    try:
        get_ipython  # type: ignore[name-defined]  # noqa: B018
    except NameError:
        return False
    return True


if not _running_in_ipython():
    parser = argparse.ArgumentParser(
        description="Open an image in Odon and create a linked two-viewer comparison."
    )
    parser.add_argument(
        "source",
        nargs="?",
        default=str(IMAGE_SOURCE),
        help=(
            "Local OME-Zarr directory or TIFF file to open in Odon "
            "(default: fixtures/synthetic_5ch.ome.zarr)."
        ),
    )
    parser.add_argument(
        "--kind",
        choices=("auto", "ome-zarr", "tiff"),
        default=IMAGE_KIND,
        help="Dataset format; auto recognizes .tif and .tiff as TIFF.",
    )
    arguments = parser.parse_args()
    IMAGE_SOURCE = arguments.source
    IMAGE_KIND = arguments.kind


# %% Connect to the running Odon process.
app = odon.connect()
print(
    f"Connected to {app.hello.app_name} {app.hello.app_version} "
    f"(instance {app.hello.instance_id})"
)


# %% Open the configured image and wait for Odon to finish loading it.
source_text = str(IMAGE_SOURCE)
source_path = Path(source_text).expanduser()
if not source_path.exists():
    raise FileNotFoundError(f"Image source does not exist: {source_path}")

dataset_kind = IMAGE_KIND
if dataset_kind == "auto":
    dataset_kind = (
        "tiff" if source_path.suffix.lower() in {".tif", ".tiff"} else "ome-zarr"
    )

if dataset_kind == "tiff":
    open_task = app.datasets.open_tiff(source_path)
else:
    open_task = app.datasets.open_ome_zarr(source_path)


def report_open_progress(snapshot: odon.TaskSnapshot) -> None:
    progress = (
        f"{snapshot.progress * 100:.0f}%"
        if snapshot.progress is not None
        else "progress unavailable"
    )
    print(f"Opening image: {snapshot.phase} ({progress})")


open_result = open_task.wait(
    timeout=OPEN_TIMEOUT_SECONDS,
    progress=report_open_progress,
)
print(f"Opened {source_path}")
if open_result is not None:
    pprint(open_result, sort_dicts=False, depth=3)


# %% Discover channels and choose the two presentations.
channel_state: dict[str, Any] = app.channels.list()
channel_items = channel_state.get("channels", [])
available_channels = [item["name"] for item in channel_items]

if not available_channels:
    raise RuntimeError(
        "Odon finished opening the source but reported no image channels. "
        "Confirm that the source is a supported image dataset."
    )

left_channel = LEFT_CHANNEL if LEFT_CHANNEL is not None else available_channels[0]
right_channel = (
    RIGHT_CHANNEL
    if RIGHT_CHANNEL is not None
    else available_channels[min(1, len(available_channels) - 1)]
)

print("Available channels:")
for index, name in enumerate(available_channels):
    print(f"  {index}: {name}")
print(f"\nLeft viewport:  {left_channel!r}")
print(f"Right viewport: {right_channel!r}")
if len(available_channels) == 1:
    print("The image has one channel, so both viewports will present it differently.")


# %% Create the comparison, or reuse the existing two viewport slots.
workspace = app.viewer.workspace.get()
viewport_entries = workspace.get("viewports", [])

if len(viewport_entries) == 1:
    comparison = app.viewer.viewports.compare(
        layout=LAYOUT,
        ratio=SPLIT_RATIO,
        titles=(LEFT_TITLE, RIGHT_TITLE),
        linked=("camera", "plane", "selection"),
    )
    left = comparison.left
    right = comparison.right
elif len(viewport_entries) == 2:
    left = app.viewer.viewports.handle(viewport_entries[0]["viewport_id"])
    right = app.viewer.viewports.handle(viewport_entries[1]["viewport_id"])
    left.rename(LEFT_TITLE)
    right.rename(RIGHT_TITLE)
    app.viewer.workspace.set_layout(LAYOUT, ratio=SPLIT_RATIO)
    app.viewer.viewport_links.update(
        viewports=(left, right),
        fields=("camera", "plane", "selection"),
    )
else:
    raise RuntimeError(
        f"Expected one or two viewports, but Odon reported {len(viewport_entries)}."
    )

print(f"Using stable viewport IDs {left.id!r} and {right.id!r}")


# %% Give each canvas an independent channel presentation.
left.set_visible_channels([left_channel])
right.set_visible_channels([right_channel])

# Colours are presentation-local: changing a colour on one viewport does not
# change the same channel's appearance in the other viewport.
left.set_channel_color(left_channel, [80, 140, 255])
right.set_channel_color(right_channel, [255, 90, 120])

# Rendering preferences are also local to each canvas.
left.set_rendering(smooth_pixels=False, show_hud=True)
right.set_rendering(smooth_pixels=True, show_hud=True)

# Camera and plane are linked, so fitting either viewport updates both.
fit_result = left.fit_camera()
fit_payload = fit_result.get("result", fit_result)
if not isinstance(fit_payload, dict) or "center_world_lvl0" not in fit_payload:
    raise RuntimeError(f"Odon did not complete the viewport camera fit: {fit_result!r}")

print("\nTwo-viewer comparison configured.")
print("Pan or zoom either canvas: the other canvas should follow.")
print("The left and right channel presentations remain independent.")


# %% Inspect the resulting independent state and shared resources.
print("\nLeft viewport channels:")
pprint(left.list_channels(), sort_dicts=False, depth=4)

print("\nRight viewport channels:")
pprint(right.list_channels(), sort_dicts=False, depth=4)

workspace = app.viewer.workspace.get()
print("\nWorkspace summary:")
pprint(
    {
        "layout": workspace.get("layout"),
        "active_viewport_id": workspace.get("active_viewport_id"),
        "viewports": [
            {
                "viewport_id": item.get("viewport_id"),
                "title": item.get("title"),
            }
            for item in workspace.get("viewports", [])
        ],
        "shared_resources": workspace.get("shared_resources"),
        "performance": workspace.get("performance"),
    },
    sort_dicts=False,
    depth=5,
)


# %% Optional: use different segmentation properties in the two canvases.
# Replace these property names with columns returned by
# app.objects.list_properties(), then uncomment the calls.
#
# left.objects.set_style(
#     fill_cells=True,
#     fill_opacity=0.65,
#     color_property="marker_a",
# )
# right.objects.set_style(
#     fill_cells=True,
#     fill_opacity=0.65,
#     color_property="marker_b",
# )


# %% Optional: unlink the cameras for independent navigation.
# Selection remains document-shared in the current milestone.
# app.viewer.viewport_links.update(
#     viewports=(left, right),
#     fields=("plane", "selection"),
# )
# right.set_camera(center=(500, 500), zoom=2.0)


# %% Optional cleanup: remove the second viewport and close this client.
# right.remove()
# app.close()
