# Controls

This page is a quick reference for common Odon controls. For step-by-step
workflows, use the workflow pages linked at the end.

## Canvas Navigation

| Action | Control |
| --- | --- |
| Pan | Left-drag on the canvas. |
| Zoom | Mouse wheel or trackpad pinch. |
| Fit the active view target | Press `F` or click `Fit (F)`. |
| Fit the current target | Double-click the image, ROI, or supported target. |

Shortcuts are ignored while a text field or numeric input is actively using the
keyboard.

## Top Bar

The top bar provides compact controls for the current viewer.

| Control | Purpose |
| --- | --- |
| `Back` | Return from the current viewer when a back target is available. |
| `Fit (F)` | Fit the current image or mosaic target to the viewport. |
| Plane selector | Switch view plane when the dataset supports more than one view plane. |
| Slice slider and arrow buttons | Move through the selected slice axis for multidimensional data. |
| `Prev. Channel` / `Next Channel` | Step through image channels. |
| `Smooth` | Toggle linear pixel filtering while zooming. Disable it for crisp nearest-neighbour pixels. |
| `Contrast` | Open compact contrast controls for the active channel, visible channels, or selected group. |
| Side-panel toggles | Show or hide the left and right panels when available. |

The compact `Contrast` menu supports `Reset`, `Set max`, a range slider, and
numeric min/max drag inputs. Hold `Ctrl` while dragging numeric contrast values
for finer adjustment.

## Left Panel

The left panel contains tools, layers, and project actions.

Use the layer list to:

- select the active layer
- show or hide layers
- reorder layers
- group or ungroup channels when supported
- choose which layer the right-panel `Properties` tab controls

If expected controls are missing, select the relevant layer first.

## Tool Buttons

Tool availability depends on the active layer and view mode. Some editing tools
are disabled outside XY view.

| Tool | Use |
| --- | --- |
| Select | Click objects or mask polygons. Drag to rectangle-select object centroids when a selectable object layer is active. |
| Pan | Navigate the canvas without editing layers. |
| Move | Move selected visible layer(s); on mask layers, drag a polygon to move it. |
| Transform | Scale or rotate the active channel when supported. |
| Polygon | Draw a mask polygon. |
| Lasso select | Draw a freehand lasso to select object centroids. |

Object selection tools require an active selectable object layer. For rectangle
selection, hold `Shift` while finishing the drag to add to the current
selection. `Esc` cancels an in-progress rectangle or lasso gesture; when no
gesture is active, `Esc` clears the current object or polygon selection.

## Right Panel

The right panel changes with the current viewer and active layer.

| Tab | Purpose |
| --- | --- |
| `Properties` | Layer-specific display, contrast, colour, transform, mask, object, and overlay controls. |
| `Views` | Capture and apply saved project view states. |
| `Layout` | Arrange mosaic ROIs by metadata. Available in mosaic mode. |
| `Analysis` | Object-property histograms, scatter review, marker mapping, calls, and selection workflows. |
| `Measurements` | Summarize image channels over polygon objects and export enriched results. |
| `Memory` | Inspect tile loading and optionally pin selected channels/levels in RAM. |
| `ROI Selector` | Navigate project ROIs and related project context. |

## Channel Controls

Use the channel list and image-channel `Properties` tab to:

- toggle channel visibility
- select the active channel
- change channel colour
- adjust contrast minimum and maximum
- inspect the histogram
- add notes
- create or use channel groups
- sort or filter channel names

Image-channel contrast, colour, visibility, and ordering affect display/project
state. They do not modify source image data.

## Object Controls

Object-backed layers expose controls only after compatible objects are loaded.

Common object controls include:

- object visibility
- opacity, outline width, and fill settings
- property-based colouring through `Color by`
- legend visibility
- object filtering
- selection display
- `Analysis` tab workflows
- `Measurements` tab workflows

Object selections feed review and analysis workflows. They do not edit source
segmentation geometry.

## Mask Polygons

| Action | Control |
| --- | --- |
| Add vertices | Select the polygon tool, then click on the canvas. |
| Close polygon | Double-click, press `Enter`, or click the highlighted first vertex. |
| Cancel in-progress polygon | Press `Esc`. |
| Remove last in-progress point | Press `Backspace`. |
| Select polygon | Make a mask layer active, switch to select or pan, then click a polygon edge or interior. |
| Edit polygon | Drag a selected polygon vertex handle. |
| Move polygon | Make a mask layer active, switch to move, then drag inside the polygon. |
| Delete selected polygon | Press `Delete`, press `Backspace`, use the mask layer properties panel, or right-click and choose `Delete polygon`. |
| Undo mask or layer-move edit | Press `Ctrl+Z` on Windows/Linux or `Cmd+Z` on macOS. |

`Esc` clears the current polygon selection when you are not actively drawing.

## Transform Tool

When the transform tool is active for a supported channel layer:

- drag inside the transform box to translate
- drag corners to scale
- drag the rotation handle to rotate

Transforms affect viewer/project alignment state, not the source image file.

## Mosaic Controls

| Action | Control |
| --- | --- |
| Fit the full mosaic | Press `F` or click `Fit Mosaic (F)`. |
| Fit one ROI | Double-click an ROI. |
| Jump to previous ROI | Click `Prev. Core`. |
| Jump to next ROI | Click `Next. Core`. |
| Arrange ROIs | Use the right-panel `Layout` tab. |
| Pin channels/levels | Use the right-panel `Memory` tab. |
| Return from mosaic | Use `Back` when available. |

Project ROI selection controls:

| Action | Control |
| --- | --- |
| Select one ROI | Click the ROI. |
| Add or remove one ROI from selection | `Cmd`-click on macOS, or `Ctrl`-click on Windows/Linux. |
| Select a range | Click one ROI, then `Shift`-click another ROI. |
| Select all visible ROIs | Click `Select all`, click `Select visible`, or press `Cmd+A`/`Ctrl+A` when the ROI browser has focus. |
| Open one ROI | Double-click a supported ROI row, or select it and click `Open`. |

## Keyboard Shortcuts

| Shortcut | Action |
| --- | --- |
| `F` | Fit the active view target. |
| `Ctrl+W` / `Cmd+W` | Open close confirmation. Press again to confirm close. |
| `Ctrl+M` | Open Analysis mapping settings where available. |
| `Ctrl+A` / `Cmd+A` | Select all visible ROIs in the project ROI browser when it has focus. |
| `Enter` | Close an in-progress mask polygon. |
| `Esc` | Cancel polygon drawing or clear polygon selection. |
| `Backspace` | Remove the last in-progress polygon point, or delete a selected polygon in pan/select mode. |
| `Delete` | Delete the selected polygon. |
| `Ctrl+Z` / `Cmd+Z` | Undo the previous mask or layer-move edit. |

## Related Pages

- [Viewing Channels](../workflows/viewing.md)
- [Mosaic Mode](../workflows/mosaic.md)
- [Objects and Overlays](../workflows/objects-and-overlays.md)
- [Mask Polygons](../workflows/mask-polygons.md)
- [Thresholding](../workflows/thresholding.md)
- [Odon MCP](codex-mcp-odon.md)
