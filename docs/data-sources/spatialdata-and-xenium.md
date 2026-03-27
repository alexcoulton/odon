# SpatialData And Xenium

`odon` includes support for SpatialData and selected Xenium Explorer workflows, but these should be treated as format adapters around the core viewer rather than the primary product center.

## SpatialData

Current SpatialData support includes discovery of:

- `images/*`
- `points/*`
- `shapes/*`
- label-like elements where present

This is useful when you want to reuse the same viewer for a SpatialData container that still maps well onto image-plus-overlay inspection.

## Xenium

Current Xenium support is centered on:

- dataset discovery from `experiment.xenium`
- morphology image loading
- cell polygon loading
- transcript point loading

This makes the viewer useful for quick exploratory inspection, especially when you want one desktop tool across multiple spatial modalities.

## Recommendation

If your main workflow is multiplex imaging or spatial proteomics, start with OME-Zarr and object overlays first. Use SpatialData and Xenium support when your data already lives in those ecosystems and you need compatibility rather than a different product model.
