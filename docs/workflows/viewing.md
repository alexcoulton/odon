# Viewing Channels

This guide covers the normal single-image viewer workflow: open one ROI, choose
channels, adjust contrast, inspect overlays, and navigate large images.

For multi-ROI review, see [Mosaic Mode](mosaic.md).

## Open One ROI

Use the Project panel as the primary way to open a single image:

1. Open Odon.
2. Click `Load Project...`, `Import Samplesheet CSV...`, or
   `Add OME-Zarr Root...`.
3. Select one ROI in the Project panel.
4. Click `Open`.

You can also drag a supported local dataset into the Odon window when that is
more convenient.

After the image opens, press `F` if the image is not already fitted to the
viewport.

## Viewer Layout

The single-image viewer has three main working areas:

| Area | Purpose |
| --- | --- |
| Left panel | Layer list, project actions, masks, object layers, and tool selection. |
| Center canvas | Image channels, overlays, masks, object outlines, and labels. |
| Right panel | `Properties`, `Views`, analysis, measurements, thresholding, and memory controls. |

The top bar contains quick controls for fitting, plane or slice navigation,
channel stepping, side-panel visibility, and compact contrast controls.

## Navigate The Canvas

| Action | Control |
| --- | --- |
| Pan | Left-drag on the canvas. |
| Zoom | Mouse wheel or trackpad pinch. |
| Fit the view | Press `F` or use the fit button in the top bar. |
| Fit the current target | Double-click the image or target ROI. |

The active layer affects what canvas tools do. If a tool does not seem to affect
the expected data, click the target layer in the left panel first.

## Work With Layers

The left panel lists image channels and non-image overlays. Use it to:

- select the active layer
- show or hide layers
- reorder layers
- select channel groups
- choose tools such as pan, move-layer, transform, and mask drawing

Image channels are drawn below non-image overlays. Object layers, masks, labels,
and point layers can be displayed over the image.

Layer changes are viewer/project state. They do not rewrite the original image
or object files unless you explicitly save or export an editable layer such as a
mask.

## Channel Visibility And Ordering

Use the channel list to decide which channels are rendered.

Common actions:

- toggle one channel on or off
- select a channel to edit its properties
- filter the channel list by name
- sort channels by name or visibility
- group related channels for repeated review

Reducing the number of visible channels is often the fastest way to improve
responsiveness on very large images, remote data, or high-channel-count
datasets.

## Contrast And Colour

Select an image channel, then use the right-panel `Properties` tab to adjust its
display.

Image-channel properties include:

- visibility
- colour
- contrast minimum and maximum
- histogram
- channel grouping
- notes
- transform controls when available

Contrast and colour affect display only. They do not modify the image data.

Odon composites visible channels additively. This means each visible channel
contributes to the final rendered image using its chosen colour and contrast
window.

The histogram is used as an interactive guide for display adjustment. For large
images, Odon avoids requiring a full level-0 image read before you can start
viewing; image tiles and display statistics are requested around the current
viewer workflow instead of blocking startup on whole-image work.

## Quick Contrast Controls

The top bar can show compact contrast controls, especially when side panels are
hidden. These are useful for quick review without reopening the full Properties
panel.

Use the full right-panel controls when you need to inspect the histogram, edit
precise numeric limits, or adjust related channel settings.

## Z Planes And View Planes

Odon is primarily an XY image viewer. If an OME-Zarr dataset includes a `z` axis,
the top bar can show plane or slice controls.

Use these controls to inspect one plane at a time. Odon does not present this as
a full volumetric renderer; it is a plane-by-plane viewer for multidimensional
OME-Zarr data.

Some layer operations are XY-specific. If an overlay, transform, mask, or object
operation is unavailable while viewing another plane mode, switch back to XY and
try again.

## Channel Transforms

If a channel layer supports transforms, select the channel, choose the
`Transform` tool, and interact with the transform box on the canvas:

- drag inside the box to translate
- drag corners to scale
- drag the rotation handle to rotate

Use this for visual alignment corrections between channels or overlays.
Transforms affect viewer/project alignment state, not the source image file.

## Overlays In Single-Image View

Single-image view is the best place for detailed overlay work.

Typical overlay workflows include:

- segmentation objects from GeoParquet or Parquet
- GeoJSON outlines
- CSV point overlays
- OME-Zarr labels
- editable mask polygons
- object selection, filtering, measurement, and export

Select an overlay layer in the left panel to see its controls in the
right-panel `Properties` tab. If expected overlay controls are missing, the most
common cause is that a different layer is active.

For details, see [Objects and Overlays](objects-and-overlays.md),
[Object and Overlay Data](../data-sources/object-and-overlay-data.md), and
[Mask Polygons](mask-polygons.md).

## Large-Image Behaviour

Odon is designed around pyramid and tile-based viewing.

When you open or navigate a large image:

- coarse levels can appear before fine levels
- visible tiles are loaded on demand
- nearby or finer-resolution tiles may be prefetched
- contrast changes usually do not require reloading decoded image tiles
- fewer visible channels usually means less tile work

If navigation feels slow:

1. Hide channels that are not needed for the current review task.
2. Fit the image and wait briefly for visible tiles to finish loading.
3. Use coarser overview levels for broad navigation when available.
4. Use the right-panel `Memory` tab only when repeated review of selected
   channels/levels justifies pinning data in RAM.

Avoid pinning high-resolution levels for very large images unless the memory
estimate is acceptable for your machine.

## Troubleshooting

### The Canvas Is Blank

Check that at least one image channel is visible, the image is fitted to the
viewport, and the selected ROI opened successfully.

### The Wrong Controls Are Shown

Click the target layer in the left panel. The `Properties` tab changes based on
the active layer.

### A Channel Looks Too Dim Or Saturated

Select the channel and adjust the contrast minimum and maximum in the
`Properties` tab. If multiple bright channels are visible, hide unrelated
channels while setting contrast for the channel you are reviewing.

### Objects Or Masks Look Offset

Check whether the active layer has a transform, offset, or move state. Also
confirm that the object or mask file belongs to the current ROI and uses the
same coordinate system as the image.

### Loading Feels Slow

Reduce visible channels, avoid unnecessary high-resolution memory pinning, and
wait for visible tiles to finish loading after large zoom or pan changes.

## Related Pages

- [Getting Started](../getting-started.md)
- [OME-Zarr](../data-sources/ome-zarr.md)
- [Objects and Overlays](objects-and-overlays.md)
- [Object and Overlay Data](../data-sources/object-and-overlay-data.md)
- [Mask Polygons](mask-polygons.md)
- [Controls](../reference/controls.md)
- [Current Limitations](../advanced/current-limitations.md)
