# Odon MCP Integration

Odon includes an MCP server for AI-assisted control of a running Odon GUI. An
MCP client such as Codex can inspect the current viewer state, change channels,
adjust contrast, open project ROIs, move the camera, query object selections, and
capture screenshots.

The MCP server is a control layer for an already-running Odon window. It is not
a replacement for the GUI and it does not render images by itself.

Users normally open only the Odon desktop app. The MCP helper is launched by the
MCP client, not by the user.

## Architecture

```text
MCP client
  launches odon_mcp over stdio
    odon_mcp forwards tool calls to
      running Odon GUI control bridge on 127.0.0.1:17870
```

There are two Odon-side executables:

| Process | Role |
| --- | --- |
| `odon` | The GUI application. It owns projects, image data, object data, GPU rendering, and screenshots. |
| `odon_mcp` | The MCP server. It speaks JSON-RPC/MCP over stdin/stdout and forwards tool calls to the GUI. |

Most MCP tools require that Odon is already open and has a project, image, or
mosaic loaded.

## Installation

For normal use, install or open the Odon desktop application first. Packaged
releases include the `odon_mcp` helper alongside the GUI. The user does not
manually launch `odon_mcp`; the MCP client launches it when needed.

### Packaged Releases

Install Odon from the release artifact for your platform, then configure your
MCP client to launch the bundled `odon_mcp` binary.

Typical paths are:

| Platform | MCP command path |
| --- | --- |
| macOS DMG/app install | `/Applications/odon.app/Contents/MacOS/odon_mcp` |
| Windows installer | `C:\Program Files\Odon\odon_mcp.exe` |
| Linux `.deb` | `/usr/lib/odon/odon_mcp` |

Restart the MCP client after changing its configuration.

The GUI still needs to be running. Open Odon normally from the desktop launcher,
then let the MCP client launch `odon_mcp`. The MCP helper connects to the GUI
bridge on `127.0.0.1:17870`; it should not start a second Odon GUI process.

### Client Configuration Examples

The examples below all configure a local stdio MCP server. Use the command path
for your platform from the table above.

#### Codex

Codex can add stdio MCP servers from the CLI:

```bash
codex mcp add odon -- /Applications/odon.app/Contents/MacOS/odon_mcp
```

On Windows:

```powershell
codex mcp add odon -- "C:\Program Files\Odon\odon_mcp.exe"
```

For manual configuration, edit `~/.codex/config.toml` or a project-scoped
`.codex/config.toml`:

```toml
[mcp_servers.odon]
command = "/Applications/odon.app/Contents/MacOS/odon_mcp"
args = []
```

Windows TOML strings need escaped backslashes:

```toml
[mcp_servers.odon]
command = "C:\\Program Files\\Odon\\odon_mcp.exe"
args = []
```

In Codex, use `/mcp` to confirm that the server is connected.

#### Claude Code

Claude Code can add a local stdio server with `claude mcp add`. The `--`
separator is important; everything after it is the command that launches the MCP
server.

```bash
claude mcp add odon -- /Applications/odon.app/Contents/MacOS/odon_mcp
```

On Linux:

```bash
claude mcp add odon -- /usr/lib/odon/odon_mcp
```

On Windows PowerShell:

```powershell
claude mcp add odon -- "C:\Program Files\Odon\odon_mcp.exe"
```

Use `claude mcp list` from a terminal or `/mcp` inside Claude Code to check the
server status. If you use project-scoped MCP configuration, Claude Code may ask
you to approve the server before it is active.

#### Claude JSON Configuration

Claude Code and Claude Desktop-style configurations use an `mcpServers` JSON
object. Depending on the client and scope, this may live in a project
`.mcp.json` file or in the client's user settings. Claude Desktop commonly uses
`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS and
`%APPDATA%\Claude\claude_desktop_config.json` on Windows.

macOS:

```json
{
  "mcpServers": {
    "odon": {
      "command": "/Applications/odon.app/Contents/MacOS/odon_mcp",
      "args": []
    }
  }
}
```

Windows:

```json
{
  "mcpServers": {
    "odon": {
      "command": "C:\\Program Files\\Odon\\odon_mcp.exe",
      "args": []
    }
  }
}
```

Linux:

```json
{
  "mcpServers": {
    "odon": {
      "command": "/usr/lib/odon/odon_mcp",
      "args": []
    }
  }
}
```

#### Gemini CLI

Gemini CLI can add local stdio servers with `gemini mcp add`:

```bash
gemini mcp add odon /Applications/odon.app/Contents/MacOS/odon_mcp
```

On Linux:

```bash
gemini mcp add odon /usr/lib/odon/odon_mcp
```

On Windows PowerShell:

```powershell
gemini mcp add odon "C:\Program Files\Odon\odon_mcp.exe"
```

Gemini CLI also supports JSON configuration in `settings.json`:

```json
{
  "mcpServers": {
    "odon": {
      "command": "/Applications/odon.app/Contents/MacOS/odon_mcp",
      "args": [],
      "timeout": 30000,
      "trust": false
    }
  }
}
```

Use `gemini mcp list` or `/mcp list` to check connection status. Stdio MCP
servers may show as disconnected when the current folder is not trusted; trust
the folder if Gemini CLI prompts you to do so.

#### Cursor

Cursor uses the `mcpServers` JSON shape. For a global setup, edit
`~/.cursor/mcp.json`. For a project-scoped setup, create `.cursor/mcp.json` in
the project directory.

```json
{
  "mcpServers": {
    "odon": {
      "command": "/Applications/odon.app/Contents/MacOS/odon_mcp",
      "args": []
    }
  }
}
```

Use the Windows or Linux command path from the table above on those platforms.
After editing the file, reload Cursor or refresh MCP servers from Cursor's MCP
settings.

#### Windsurf / Cascade

Windsurf Cascade reads MCP servers from
`~/.codeium/windsurf/mcp_config.json`.

```json
{
  "mcpServers": {
    "odon": {
      "command": "/Applications/odon.app/Contents/MacOS/odon_mcp",
      "args": []
    }
  }
}
```

Use the Windows or Linux command path from the table above on those platforms.
After editing `mcp_config.json`, restart Windsurf or reload MCP servers from the
Cascade MCP settings page.

#### VS Code / GitHub Copilot

VS Code's MCP configuration uses a `servers` object rather than `mcpServers`.
For a workspace-scoped setup, create or edit `.vscode/mcp.json`:

```json
{
  "servers": {
    "odon": {
      "type": "stdio",
      "command": "/Applications/odon.app/Contents/MacOS/odon_mcp",
      "args": []
    }
  }
}
```

Use the Windows or Linux command path from the table above on those platforms.
After saving the file, start the server from the MCP view or use the MCP server
actions shown by VS Code.

#### Other JSON MCP Clients

Many MCP-capable coding clients use the same `mcpServers` JSON shape:

```json
{
  "mcpServers": {
    "odon": {
      "command": "/path/to/odon_mcp",
      "args": []
    }
  }
}
```

Use the installed `odon_mcp` path for your operating system. If the client has a
trust or approval flow for local stdio servers, approve Odon after reviewing the
path.

### Development Checkout

Build the GUI and MCP helper from the checkout:

```bash
cargo build --bin odon --bin odon_mcp
```

For development builds, start the GUI from the checkout:

```bash
cargo run --bin odon
```

Configure your MCP client to launch the MCP server from the same checkout:

```text
command: cargo
args: ["run", "--quiet", "--bin", "odon_mcp"]
cwd: /path/to/odon.pub
```

For clients that use TOML-style MCP configuration, the same idea is:

```toml
[mcp_servers.odon]
command = "cargo"
args = ["run", "--quiet", "--bin", "odon_mcp"]
cwd = "/path/to/odon.pub"
```

Restart the MCP client after changing its configuration.

When the development GUI starts successfully from a terminal, it prints:

```text
odon control bridge listening on 127.0.0.1:17870
```

Packaged desktop launches do not require a terminal. On Windows release builds,
the GUI is built as a Windows desktop subsystem executable so double-clicking
`odon.exe` should not open a cmd window.

## Smoke Tests

You can test the MCP server without Codex or another full MCP client.

List available tools:

```bash
printf '%s\n' \
'{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","clientInfo":{"name":"smoke","version":"0.1"},"capabilities":{}}}' \
'{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}' \
| cargo run --quiet --bin odon_mcp
```

Call a tool against a running Odon GUI:

```bash
printf '%s\n' \
'{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","clientInfo":{"name":"smoke","version":"0.1"},"capabilities":{}}}' \
'{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"get_current_view","arguments":{}}}' \
| cargo run --quiet --bin odon_mcp
```

If Odon is not running, `tools/list` can still work because it only talks to
`odon_mcp`, but `tools/call` requests that need the GUI will fail.

## Typical Usage

Start Odon, open a project or image, then ask the MCP client to operate on the
current view.

Example tasks:

```text
Use the Odon MCP. Show only CD3, CD4, CD8, and DAPI, then fit the image to view.
```

```text
Use the Odon MCP. Inspect the current ROI, list the visible channels, report the
active channel contrast, and capture a screenshot.
```

```text
Use the Odon MCP. Open ROI_042 from the current project, show object overlays,
colour the relevant marker channels, and save the project.
```

## Tool Groups

The exact tool list is provided by the `odon_mcp` binary through MCP
`tools/list`. Current tools are grouped below by workflow.

### Viewer State

| Tool | Purpose |
| --- | --- |
| `get_current_view` | Return current mode, project state, viewer state, and visible channels. |
| `get_loading_state` | Return diagnostics for loading or busy indicators. |
| `get_side_panels` | Report whether the left and right panels are visible. |
| `set_side_panels` | Show or hide side panels. |
| `get_smooth_pixels` | Report smooth pixel interpolation state. |
| `set_smooth_pixels` | Enable or disable smooth pixel interpolation. |
| `set_right_tab` | Set the active right-panel tab in single-image mode. |

### Opening Data And ROIs

| Tool | Purpose |
| --- | --- |
| `open_project` | Open a local Odon project JSON in the active window. |
| `open_ome_zarr` | Open a local OME-Zarr dataset in the active window. |
| `open_tiff` | Open a local TIFF or OME-TIFF file in the active window. |
| `open_mosaic_samplesheet` | Open a mosaic from a local samplesheet CSV. |
| `list_project_rois` | List ROIs from the current project. |
| `open_roi` | Open a project ROI by id, display name, path fragment, or sample. |
| `show_project_page` | Return to the Project page from the active single-image or mosaic viewer. |
| `save_project` | Save the current project to its existing project JSON path. |

### Mosaic Layout

| Tool | Purpose |
| --- | --- |
| `set_mosaic_right_tab` | Set the active right-panel tab in mosaic mode. |
| `configure_mosaic_layout` | Configure grouping, sorting, labels, layout mode, and column count in mosaic mode. |

### Channels And Contrast

| Tool | Purpose |
| --- | --- |
| `list_channels` | List channels from the active single-image or mosaic viewer. |
| `list_visible_channels` | List currently visible channels. |
| `get_active_channel` | Return the active channel. |
| `set_active_channel` | Set the active channel by index, exact name, or marker-like selector. |
| `set_visible_channels` | Show, hide, or replace the visible channel set. |
| `get_channel_contrast` | Return contrast limits for a selected channel. |
| `set_channel_contrast` | Set contrast limits for a selected channel. |
| `get_channel_intensity_stats` | Compute coarse image-level intensity statistics in single-image view. |
| `set_channel_order` | Set manual/listed channel order or a built-in sort mode. |
| `list_channel_groups` | List project-backed channel groups. |
| `set_channel_group` | Create or update a project-backed channel group. |

### Camera And Screenshots

| Tool | Purpose |
| --- | --- |
| `get_camera` | Return camera center and zoom. |
| `set_camera` | Set camera center and/or zoom. |
| `zoom_in` | Zoom in around the current viewport center. |
| `zoom_out` | Zoom out around the current viewport center. |
| `fit_to_view` | Fit the current image or mosaic to the viewport. |
| `capture_screenshot` | Queue a canvas screenshot. |
| `capture_window_screenshot` | Queue a full Odon viewport screenshot, including panels and Project-page UI. |
| `capture_project_screenshot` | Return to the Project page, then queue a full Odon viewport screenshot. |

### Objects and Overlays

| Tool | Purpose |
| --- | --- |
| `get_object_overlay_visibility` | Return object, labels, GeoJSON, or combined overlay visibility. |
| `set_object_overlay_visibility` | Show or hide object/segmentation overlays. |
| `get_object_selection` | Return selected object IDs and centroids for the active selectable layer. |
| `query_object_ids_in_rect` | Return object IDs intersecting a world or screen rectangle. |
| `query_object_ids_in_view` | Return object IDs intersecting the current viewport. |
| `select_object_ids_in_rect` | Select object IDs intersecting a rectangle. |
| `clear_object_selection` | Clear selected objects on the active selectable object layer. |

Object selection and rectangle-query tools require an active selectable object
layer. They return an error if the current mode or layer does not support the
requested operation.

## Common Tool Arguments

Open a project:

```json
{"path": "/path/to/project.json"}
```

Open a mosaic from a samplesheet:

```json
{"path": "/path/to/samplesheet.csv", "columns": 10}
```

Set the active right-panel tab in single-image mode:

```json
{"tab": "analysis"}
```

Set the active right-panel tab in mosaic mode:

```json
{"tab": "layout"}
```

Configure mosaic layout:

```json
{
  "group_by": "response",
  "sort_by": "patient_id",
  "sort_by_secondary": "core_replicate",
  "layout": "fit_cells",
  "columns": 10,
  "show_group_labels": true,
  "show_text_labels": true,
  "label_columns": ["id", "response"]
}
```

Channel tools accept one of several selectors:

```json
{"index": 0}
{"name": "C005 - CD45"}
{"channel": "CD45"}
{"marker": "CD45"}
```

Set visible channels:

```json
{"channels": ["CD3", "CD4", "CD8"], "mode": "only"}
```

Set contrast:

```json
{"channel": "CD45", "min": 0, "max": 1200}
```

Set camera:

```json
{"center_world_lvl0": [1200, 2400], "zoom": 0.35}
```

Capture a canvas screenshot:

```json
{"path": "/path/to/screenshot.png"}
```

If `capture_screenshot` is called without `path`, Odon uses the configured quick
screenshot output location.

Capture the full Odon window content, including panels:

```json
{"path": "/path/to/window-screenshot.png"}
```

Use `capture_project_screenshot` when you want a Project page screenshot in one
step:

```json
{"path": "/path/to/project-page.png"}
```

Alternatively, use `show_project_page` followed by `capture_window_screenshot`
when you want to inspect or adjust the Project page before capturing it.
Full-window screenshots require an explicit `path` and are written after egui
returns the next viewport screenshot event, so the tool response reports that
the screenshot was queued.

## Common Problems

### Odon Bridge Unavailable

The MCP server could not connect to `127.0.0.1:17870`.

Fix: open the Odon desktop application first. For a development checkout, start
the GUI with:

```bash
cargo run --bin odon
```

### Tool List Works But Tool Calls Fail

`tools/list` only checks the MCP server. Most `tools/call` requests also require
the running GUI and an open project, image, or mosaic.

Fix: open Odon and load data before asking the MCP client to inspect or modify
viewer state.

### Unknown Odon Control Method

The MCP server knows about a tool, but the running GUI does not know the
forwarded bridge method. This usually means the GUI and `odon_mcp` came from
different builds.

Fix: restart both the GUI and MCP client from the same checkout or release.

### Screenshot Did Not Appear

Screenshots are queued in the GUI. If making many captures, wait for each output
file before requesting the next one. The GUI has one pending screenshot slot.

### A Tool Times Out

The local GUI bridge waits for a response from the app. If Odon is busy loading
or blocked, the bridge can time out.

Fix: wait until the viewer is responsive, then call `get_loading_state` or retry
the operation.

## Security Notes

The control bridge binds to `127.0.0.1:17870`, so it is intended for local
desktop control. Do not expose it to a network.

MCP clients can ask Odon to open local files, capture screenshots, and save the
current project. Use trusted MCP clients and review automation instructions
before running them against sensitive data.
