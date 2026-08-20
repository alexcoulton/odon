"""Interactive Odon API scratchpad for Vim-Slime and IPython.

Start Odon from the repository root with:

    cargo run -- fixtures/synthetic_5ch.ome.zarr

Start IPython in another tmux pane from an activated development environment:

    source .venv/bin/activate
    ipython --no-autoindent

Then open this file in Neovim and send one ``# %%`` section at a time. Do not
run the entire file: later sections remove resources and close the connection.
"""

# %% Imports and paths
from pathlib import Path

import numpy as np
import odon
from odon import ui

ROOT = Path("/Users/alexcoulton/work/external.repos/odon.pub")
DATASET = ROOT / "fixtures/synthetic_5ch.ome.zarr"


# %% Discover running Odon instances without printing their bearer tokens
[
    (instance.instance_id, instance.pid, instance.port)
    for instance in odon.list_instances()
]


# %% Connect to the sole running instance
app = odon.connect()
app.hello


# %% Inspect application and viewer state
state = app.application.get_state()
state


# %% Inspect and manipulate channels
app.channels.list()
app.channels.set_visible(["DAPI", "CD3"])
app.channels.set_active("CD3")
app.channels.set_contrast("CD3", minimum=0, maximum=25000)


# %% Manipulate the camera and rendering
app.viewer.fit()
app.viewer.set_camera(center=(256, 256), zoom=1.5)
app.viewer.set_smooth_pixels(False)


# %% Print native Odon changes and task progress in IPython
def print_event(event: odon.Event) -> None:
    print(
        f"\nEVENT {event.name}: "
        f"revision={event.revision}, source={event.source}"
    )


app.events.subscribe(
    [
        "viewer.camera.*",
        "viewer.channels.*",
        "viewer.selection.*",
        "tasks.*",
    ],
    print_event,
)


# %% Start a retained task; this line returns promptly
task = app.application.open_ome_zarr(DATASET)
task


# %% Inspect the task without waiting
task.refresh()


# %% Wait for real completion; only the IPython prompt blocks, not Odon
result = task.wait(timeout=120)
result


# %% Create a synthetic label array
yy, xx = np.ogrid[:512, :512]
labels = np.zeros((512, 512), dtype=np.uint16)
labels[(xx - 170) ** 2 + (yy - 220) ** 2 < 70**2] = 1
labels[(xx - 340) ** 2 + (yy - 280) ** 2 < 55**2] = 2


# %% Register the NumPy result through a managed temporary OME-Zarr resource
resource = app.data.register_numpy(
    labels,
    axes=("y", "x"),
    scale=(1.0, 1.0),
    translation=(0.0, 0.0),
    provenance={"created_by": "Vim-Slime IPython test"},
)


# %% Add the resource as a live Odon label layer
layer = app.layers.add(
    resource,
    name="IPython labels",
    kind="labels",
    opacity=0.65,
)


# %% Update the layer
layer.update(opacity=0.9)


# %% Register a Python-defined native Odon panel
extension = app.ui.register_extension(
    id="org.example.nvim-test",
    name="Neovim Test",
    version="0.1.0",
    capabilities=("ui.panels", "viewer.read", "viewer.write"),
    disconnect_policy="remove",
)

panel = ui.Panel(
    "nvim-panel",
    title="Neovim + IPython",
    children=[
        ui.Text("message", "This panel was created from IPython."),
        ui.Slider(
            "zoom",
            "Camera zoom",
            minimum=0.1,
            maximum=10.0,
            value=1.5,
            action=ui.bind("viewer.camera", property="zoom"),
            event_policy=ui.Immediate(),
        ),
        ui.Button(
            "hello",
            "Send event to Python",
            action=ui.emit("hello-from-odon"),
        ),
        ui.Status("status", "Ready"),
    ],
)

contribution = extension.register(panel, location="right.tabs")


# %% Print button and other Python-facing UI events
app.events.subscribe(
    "ui.extension:org.example.nvim-test.*",
    lambda event: print("\nUI EVENT:", event.name, event.data),
)


# %% Patch retained native UI state from Python
contribution.patch_values(
    {
        "message": "Updated live from Neovim",
        "status": "Everything is connected",
    }
)


# %% Optional cleanup: send these lines only when finished
extension.remove()
layer.remove()
resource.remove()
app.close()
