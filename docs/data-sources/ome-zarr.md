# OME-Zarr

OME-Zarr is the main image format for `odon`.

## What Works Well

- multiscale image pyramids
- on-demand tile loading
- per-channel display controls
- responsive single-view inspection
- mosaic workflows across many ROIs

## Viewer Assumptions

The current viewer is focused on XY inspection, with support for scrubbing through OME-Zarr `z` planes when present.

In practice, that means:

- the first `multiscales[]` entry is used as the main image
- XY viewing is the primary path
- `z` can be inspected plane-by-plane with a slider, rather than as a full volumetric renderer
- other axes such as `t` are still treated more narrowly than in a full multidimensional viewer

## Opening Local OME-Zarr

```bash
cargo run -- "/path/to/ROI1.ome.zarr"
```

You can also open OME-Zarr through a project file or as part of a mosaic.

## Rendering Behavior

The viewer uses a tile-based, coarse-to-fine approach:

- only visible tiles are requested
- coarse levels appear first
- finer levels replace them as data arrives

This keeps large pyramids usable during zoom and pan.

## Related Workflows

- [Viewing Channels](../workflows/viewing.md)
- [Mosaic Mode](../workflows/mosaic.md)
- [Current Limitations](../advanced/current-limitations.md)
