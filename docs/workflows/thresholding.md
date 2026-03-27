# Thresholding

`odon` already includes threshold-oriented workflows aimed at rapid marker review rather than a full analysis suite.

## What The Current Workflow Supports

The current implementation is strongest for:

- loading marker-associated values from object or centroid data
- reviewing thresholds channel by channel
- previewing threshold effects in the viewer
- saving per-ROI or per-marker state in project-driven workflows

## Where It Appears In The UI

Depending on the data you have loaded, threshold-related controls appear in the right-side tabs, especially around:

- `Cell Thresholds`
- object analysis or layer-specific properties

## Best Use Case

The most natural use case today is a spatial proteomics review loop:

1. open an image dataset
2. load an object or centroid-backed data source
3. move marker by marker
4. compare image context with the thresholded data view
5. save or export the resulting threshold state for downstream work

## Practical Expectation

This should be treated as lightweight viewer-side threshold authoring. If you need a full phenotype logic engine or heavier statistical analysis, `odon` is better used upstream of that work rather than as the complete environment for it.

## Related Pages

- [Objects and Overlays](objects-and-overlays.md)
- [Projects and Samplesheets](../data-sources/projects-and-samplesheets.md)
- [Current Limitations](../advanced/current-limitations.md)
