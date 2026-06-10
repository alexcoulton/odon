# Mask Polygons

Mask polygons are editable overlay regions for exclusion, artefact marking, and
other region-based review tasks. They live on mask layers, can be saved in
project state, and can be exported as GeoJSON for downstream analysis.

Use mask polygons when you need to mark regions on an image by hand. For
object-level filtering or cell category review, use object overlays and object
filters instead.

## Start From The GUI

1. Open an image or project ROI.
2. Make sure the left and right panels are visible.
3. In the layer list, select an existing mask layer, or choose the polygon tool
   to create an editable mask layer when needed.
4. Draw, select, edit, or export polygons from the active mask layer.

Mask polygon editing is a single-image workflow. In mosaic mode, use masks for
review context and open an individual ROI for detailed mask editing.

## Create A Mask Polygon

1. Select the polygon tool.
2. Click on the canvas to add polygon vertices.
3. Close the polygon with double-click, `Enter`, or by clicking the highlighted
   first vertex.

While drawing:

- the first vertex highlights when the cursor is close enough to close the polygon
- `Backspace` removes the last point
- `Esc` cancels the in-progress polygon
- a polygon needs at least three points before it can close

When a polygon is completed, Odon adds it to the active editable mask layer. If
there is no suitable active mask layer, Odon creates one.

## Select And Edit A Polygon

1. Make the mask layer active in the left panel.
2. Switch to the pan tool.
3. Click a polygon edge or interior to select it.
4. Click and drag a vertex handle to reshape the polygon.

The selected polygon is drawn with a stronger outline and visible vertex handles.
Dragging the first vertex also moves the duplicated closing vertex so the ring
remains closed.

To move one whole polygon without moving the rest of the mask layer, switch to
the move tool and drag inside the polygon. If the mask layer has multiple
polygons, only the polygon under the pointer is translated.

Drag empty canvas space to pan when the pointer is not on a polygon edge,
interior, or vertex handle.

## Delete A Polygon

With a mask polygon selected:

- press `Delete` or `Backspace`
- use `Delete polygon` in the mask layer properties panel
- right-click the canvas and choose `Delete polygon`

## Undo Mask Edits

Use `Ctrl+Z` on Windows/Linux or `Cmd+Z` on macOS to undo the previous mask edit.

Undo covers:

- adding a completed polygon
- deleting a polygon
- dragging a vertex
- moving a polygon
- clearing, reloading, creating, or deleting mask layers
- threshold-created mask layers

`Esc` clears the current polygon selection when you are not actively drawing.

## GeoJSON Workflow

Mask layers can be loaded from GeoJSON and exported back to GeoJSON. This is
useful for hand-authored exclusion regions or artefact regions that need to move
between Odon and downstream analysis code.

Use project save when you want to preserve mask state inside an Odon project. Use
GeoJSON export when another tool needs the geometry.

## Threshold-Created Masks

Some threshold workflows can create a mask preview and then apply it as an
editable mask layer. After applying a threshold-created mask, use the same mask
polygon controls to clean up generated regions by selecting, moving, deleting,
or reshaping polygons.

## Practical Tips

- Keep one mask concept per layer, such as artefact, tissue exclusion, or review
  region.
- Select the mask layer before editing so clicks target the expected polygons.
- Export to GeoJSON before handing masks to another tool.
- Use undo immediately if a polygon edit affects the wrong region.

## Related Pages

- [Objects and Overlays](objects-and-overlays.md)
- [Viewing Channels](viewing.md)
- [Controls](../reference/controls.md)
