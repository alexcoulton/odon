# odon-client

`odon-client` is the pure-Python SDK for controlling and extending a separately
running Odon microscopy viewer. Odon does not bundle Python, and the SDK does
not bundle Odon.

```python
import odon

with odon.connect() as app:
    print(app.application.get_state())
    app.channels.set_visible(["DAPI", "CD3"])
    app.viewer.set_camera(center=(500, 700), zoom=0.5)
```

Each Odon process publishes a private authenticated discovery manifest for its
dynamic loopback endpoint. Use `odon.list_instances()` or
`odon.connect(instance="…")` when several instances are open. You can also
start an installed executable with `odon.launch(path)` or
`await odon.launch_async(path)`.

Async calls share a persistent connection and do not block the Python event
loop:

```python
import asyncio
import odon

async def main():
    async with odon.connect_async() as app:
        camera, channels = await asyncio.gather(
            app.viewer.get_camera(), app.channels.list()
        )
        await app.viewer.set_camera(zoom=1.0)
        print(camera, channels)

asyncio.run(main())
```

Long operations return retained task handles. Sync code uses `task.wait()`;
async tasks are directly awaitable. Completion and progress are pushed by Odon,
not polled:

```python
task = app.projects.open("experiment.odon")
task.wait(timeout=120)

async_task = await app.screenshots.capture("view.png")
await async_task
```

External analysis sends large results by reference:

```python
space = odon.CoordinateSpace(axes=("y", "x"), scale=(0.5, 0.5))
labels = app.data.register(
    "file:///data/cellpose.zarr",
    format="ome-zarr",
    coordinate_space=space,
)
layer = app.layers.add(labels, name="Cellpose", kind="labels")
```

Install the `arrays` extra to use sync or async `register_numpy(...)`. It writes
a managed temporary Zarr resource and cleans it with the session. Project-owned
resource and layer descriptors persist in Odon project JSON.

Declarative components in `odon.ui` are validated and rendered as native egui
controls by Rust. They support Rust-local Odon commands and state bindings as
well as events delivered to Python. `odon.run_extension(...)` provides a
reconnecting lifecycle runner for separately packaged extensions.

The control API is currently experimental. See
`docs/reference/python-api.md` and `docs/design/control-protocol-v1.md` in the
Odon repository for the full surface, guarantees, and current limitations.
