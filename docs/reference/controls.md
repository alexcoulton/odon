# Controls

## Navigation

- Pan: left-drag
- Zoom: mouse wheel or trackpad pinch
- Fit view: `F`
- Double click: fit to the current target

## Layer Interaction

- select a layer in the left panel to make it active
- use visibility toggles to hide or show layers
- drag layer rows to reorder them

## Mask Polygons

- Add vertices: select the polygon tool, then click on the canvas
- Close polygon: double-click, press `Enter`, or click the highlighted first vertex
- Cancel in-progress polygon: `Esc`
- Remove last in-progress point: `Backspace`
- Select polygon: make a mask layer active, switch to pan, then click a polygon edge or interior
- Edit polygon: drag a selected polygon vertex handle
- Move polygon: make a mask layer active, switch to move, then drag inside a polygon
- Delete selected polygon: `Delete`, `Backspace`, or right-click and choose `Delete polygon`
- Undo mask edit: `Ctrl+Z` on Windows/Linux or `Cmd+Z` on macOS

## Transform Tool

When the transform tool is active for a supported channel layer:

- drag inside the box to translate
- drag corners to scale
- drag the rotation handle to rotate

## Mosaic Navigation

- `F`: fit the mosaic
- double click an ROI: fit that ROI to the viewport
- `Prev. Core`: jump to the previous ROI
- `Next. Core`: jump to the next ROI
- right panel `Layout`: group and sort ROIs by samplesheet/project metadata
- right panel `Memory`: optionally pin selected channels and levels in RAM

## Panels

- Left panel: layers and project actions
- Right panel: properties, layout, views, memory, threshold controls, and ROI-related tools
- Top bar: quick actions and compact contrast controls when panels are hidden
- `Ctrl+M`: open the Analysis mapping settings dialog for segmentation objects or the active object-backed SpatialData shape layer
