# Objects And Overlays

`odon` can display more than image channels. You can layer objects, labels, points, and masks on top of the image canvas.

## Common Overlay Types

- segmentation outlines from NGFF labels
- polygon or point objects from GeoParquet, Parquet, GeoJSON, or CSV
- exclusion or artefact masks from GeoJSON
- point overlays such as transcript-like data

## Object Data

For object-centric review, the most useful inputs are usually:

- GeoParquet for rich polygon or point data with scalar properties
- GeoJSON for simple interchange
- CSV for point-only data with `x` and `y` columns

These layers can participate in selection, inspection, and some analysis workflows.

## NGFF Labels

If the dataset includes label groups under `labels/`, the viewer can render outlines for those label layers.

Useful notes:

- label groups are discoverable rather than assumed to be called `cells`
- the viewer renders outlines rather than filled masks
- label appearance can be adjusted from the layer properties

## Masks

The viewer supports exclusion or artefact masks saved as GeoJSON.

Current workflow:

- load masks from the `ROI Selector` tab
- draw or edit masks using the `Draw mask` tool
- export masks to GeoJSON when needed

## Points

Point overlays are available through both direct point loading and specialized dataset adapters.

Examples include:

- generic point tables
- thresholded cell-centroid style views
- Xenium transcript-style overlays

## Recommended Overlay Workflow

1. Open the image dataset first.
2. Add segmentation or object layers.
3. Select the active overlay layer in the left panel.
4. Use the right panel to adjust appearance or analysis settings.
5. Keep image channels beneath non-image overlays for clarity.
