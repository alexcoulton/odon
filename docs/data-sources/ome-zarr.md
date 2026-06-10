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

Use the GUI for normal opening:

1. Open Odon.
2. In the Project panel, click `Add OME-Zarr Root...`.
3. Choose either one OME-Zarr dataset folder or a parent folder containing
   multiple OME-Zarr ROI folders.
4. Select one ROI and click `Open`, or select multiple OME-Zarr ROIs and click
   `Open mosaic (N)`.

You can also drag a supported local OME-Zarr dataset into the Odon window to add
it to the current project list.

Command-line opening is available for scripting and debugging; see
[CLI](../reference/cli.md).

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
