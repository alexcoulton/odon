# odon-cellpose

`odon-cellpose` is Odon's reference external analysis extension. It runs in the
user's Python environment—not inside the Rust process—and exercises the full
extension path: authenticated discovery, native declarative UI, pushed events,
Cellpose inference, managed temporary Zarr exchange, provenance, and a rendered
label layer.

Development install from the Odon repository:

```bash
python -m pip install -e './python[arrays]' -e ./extensions/odon-cellpose
```

With Odon running and a local OME-Zarr dataset open:

```bash
odon-cellpose
```

The command reconnects if Odon restarts, re-registers its right-side native
panel, and exits cleanly on SIGINT or SIGTERM. Use the panel to choose the model,
diameter, CPU/GPU execution, and viewport or whole-image extent. Run and Cancel
events are handled in the external process. The previous preview resource is
removed after a new label layer replaces it.

Cancellation is cooperative: it is checked before and after Cellpose inference,
but cannot interrupt `model.eval()` while that library call is executing. The
current reference implementation also reads the requested extent into memory
and writes a session-temporary preview. Tiled inference and a user-selected
durable output location remain future production work.

Programmatic construction is also supported:

```python
import odon
from odon_cellpose import CellposeExtension

app = odon.connect()
extension = CellposeExtension(app)
```

Call `extension.close()` and `app.close()` when not using the lifecycle runner.
