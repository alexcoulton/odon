# Viewing Channels

This page covers the core single-dataset workflow: open an image, navigate it, and inspect channels quickly.

## Navigation

- Pan: left-drag
- Zoom: mouse wheel or trackpad pinch
- Fit view: `F`
- Fit the current view target: double click

## Working With Layers

The left panel contains the layer list and layer tools.

You can:

- change which layer is active
- toggle layer visibility
- reorder layers
- adjust per-channel colors
- choose tools such as pan, move-layer, transform, and mask drawing

## Contrast And Color

For image channels, the active layer exposes contrast and color controls in the right panel.

Current behavior:

- channels are composited additively
- channel colors and visibility can be changed independently
- on the GPU path, contrast updates are fast because they do not require reloading image tiles

## Channel Transforms

If a channel layer supports transforms, select the `Transform` tool and interact with the overlay box:

- drag inside the box to translate
- drag corners to scale
- drag the rotation handle to rotate

This is useful when channels need manual alignment adjustments.

## Tips For Large Datasets

- the viewer draws coarse levels first and refines as data loads
- visible tiles are loaded on demand rather than decoding the whole pyramid
- if zooming feels incomplete, pause briefly to let higher-resolution tiles arrive

## Related Pages

- [OME-Zarr](../data-sources/ome-zarr.md)
- [Controls](../reference/controls.md)
- [Current Limitations](../advanced/current-limitations.md)
