# Current Limitations

This page lists the main constraints a user should know before treating `odon` as a general-purpose spatial viewer.

## Viewer Model

- the main viewing path is 2D XY rather than a fully general multidimensional viewer
- channels are currently composited additively
- image channels are drawn below non-image overlays

## Mosaic Mode

- mosaic mode is currently strongest for imagery
- some overlay workflows are more complete in single-view mode
- the current mosaic implementation depends on the GPU rendering path

## Labels And Overlays

- NGFF labels are currently rendered as outlines rather than filled masks
- label rendering does not yet provide every refinement you might expect from a dedicated label viewer

## Remote And Session State

- remote workflows exist, but credentials are session-only
- some project-linked workflows assume local or stable dataset organization

## Scope

`odon` is best understood as a fast viewer and lightweight annotation tool. If your workflow requires a full pathology suite, a full multidimensional image authoring environment, or heavy downstream statistics inside the same app, you will likely want companion tools around it.
