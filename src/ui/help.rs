use eframe::egui;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HelpTopic {
    GettingStarted,
    Navigation,
    Layers,
    AutoLevel,
    ImportOpen,
    Samplesheets,
    ObjectCache,
    Tools,
    PropertiesPanel,
    ViewsPanel,
    AnalysisPanel,
    MeasurementsPanel,
    MemoryPanel,
    RoiSelectorPanel,
    Thresholding,
    MaskPolygons,
    Shortcuts,
}

impl HelpTopic {
    const ALL: &'static [HelpTopic] = &[
        HelpTopic::GettingStarted,
        HelpTopic::Navigation,
        HelpTopic::Layers,
        HelpTopic::AutoLevel,
        HelpTopic::ImportOpen,
        HelpTopic::Samplesheets,
        HelpTopic::ObjectCache,
        HelpTopic::Tools,
        HelpTopic::PropertiesPanel,
        HelpTopic::ViewsPanel,
        HelpTopic::AnalysisPanel,
        HelpTopic::MeasurementsPanel,
        HelpTopic::MemoryPanel,
        HelpTopic::RoiSelectorPanel,
        HelpTopic::Thresholding,
        HelpTopic::MaskPolygons,
        HelpTopic::Shortcuts,
    ];

    pub fn title(self) -> &'static str {
        match self {
            HelpTopic::GettingStarted => "Getting Started",
            HelpTopic::Navigation => "Navigation",
            HelpTopic::Layers => "Layers And Overlays",
            HelpTopic::AutoLevel => "Auto Level",
            HelpTopic::ImportOpen => "Import And Open Controls",
            HelpTopic::Samplesheets => "Samplesheet CSV Format",
            HelpTopic::ObjectCache => "Object Cache",
            HelpTopic::Tools => "Toolbar Tools",
            HelpTopic::PropertiesPanel => "Properties Tab",
            HelpTopic::ViewsPanel => "Views Tab",
            HelpTopic::AnalysisPanel => "Analysis Tab",
            HelpTopic::MeasurementsPanel => "Measurements Tab",
            HelpTopic::MemoryPanel => "Memory Tab",
            HelpTopic::RoiSelectorPanel => "ROI Selector Tab",
            HelpTopic::Thresholding => "Thresholding",
            HelpTopic::MaskPolygons => "Mask Polygons",
            HelpTopic::Shortcuts => "Keyboard And Mouse Shortcuts",
        }
    }

    fn summary(self) -> &'static str {
        match self {
            HelpTopic::GettingStarted => {
                "A short orientation to the main odon workflow: open data, choose layers, navigate, and inspect overlays."
            }
            HelpTopic::Navigation => {
                "How to move around the image canvas and fit the current view."
            }
            HelpTopic::Layers => {
                "How layer visibility, ordering, active-layer selection, and overlay properties work."
            }
            HelpTopic::AutoLevel => {
                "Automatically chooses the image pyramid level to draw for the current zoom."
            }
            HelpTopic::ImportOpen => {
                "Adds datasets to a project and opens ROIs, saved views, remote datasets, or mosaics."
            }
            HelpTopic::Samplesheets => {
                "The CSV format used to bulk-add ROIs and carry metadata into projects and mosaics."
            }
            HelpTopic::ObjectCache => {
                "Preloads segmentation object files so repeated project ROI review is faster."
            }
            HelpTopic::Tools => "What each toolbar tool does and when to use it.",
            HelpTopic::PropertiesPanel => "Layer-specific controls for the currently active layer.",
            HelpTopic::ViewsPanel => "Save and reopen reusable project views for ROIs.",
            HelpTopic::AnalysisPanel => {
                "Object-backed analysis controls for cell calls, selections, and marker review."
            }
            HelpTopic::MeasurementsPanel => {
                "Measurement tools for loaded segmentation objects and object-backed shape layers."
            }
            HelpTopic::MemoryPanel => {
                "Controls for tile loading, pinned levels, and memory-sensitive workflows."
            }
            HelpTopic::RoiSelectorPanel => {
                "Project-driven ROI navigation, label loading, and mask loading/saving."
            }
            HelpTopic::Thresholding => {
                "Creates review masks from visible pixels in the active image channel."
            }
            HelpTopic::MaskPolygons => {
                "Draws, selects, edits, deletes, and exports polygon mask regions."
            }
            HelpTopic::Shortcuts => {
                "Common keyboard and mouse controls available across the viewer."
            }
        }
    }

    fn sections(self) -> &'static [HelpSection] {
        match self {
            HelpTopic::GettingStarted => &[
                HelpSection {
                    heading: "Typical session",
                    bullets: &[
                        "Start by opening a single supported dataset, or open a project when you need a saved multi-ROI workspace.",
                        "Use the layer list to decide what is visible, then click one layer to make it active.",
                        "Use the right panel for the active layer's contrast, color, transform, mask, object, and analysis controls.",
                        "Use the canvas for navigation, object selection, mask drawing, threshold preview, and visual review.",
                    ],
                },
                HelpSection {
                    heading: "Where things are",
                    bullets: &[
                        "The left panel contains the layer list, toolbar, project controls, ROI browser, and project save/load controls.",
                        "The right panel contains tabs for layer properties, saved views, analysis, measurements, memory, and ROI navigation.",
                        "The top bar contains view fit, plane or slice navigation, pyramid-level controls, channel selection, panel toggles, and quick contrast controls.",
                        "The status text near project and workflow controls reports import failures, save paths, and long-running action results.",
                    ],
                },
                HelpSection {
                    heading: "New-user checks",
                    bullets: &[
                        "If a control is disabled, first check that the correct layer is active and that the dataset type supports that operation.",
                        "For project workflows, select or focus an ROI before opening, capturing a view, loading masks, or saving masks.",
                        "For object analysis and measurements, load segmentation objects or choose an object-backed SpatialData shape layer first.",
                    ],
                },
            ],
            HelpTopic::Navigation => &[
                HelpSection {
                    heading: "Canvas controls",
                    bullets: &[
                        "With the pan tool active, left-drag empty canvas space to pan.",
                        "Mouse wheel or trackpad pinch zooms around the pointer; zooming may briefly show a coarser pyramid level while detailed tiles load.",
                        "Press F to fit the current image, mosaic, or active view target into the canvas.",
                        "Double-click fits the current view target unless a drawing or editing tool uses the double-click for its own action.",
                        "When an editable handle is under the pointer, dragging edits that handle instead of panning.",
                    ],
                },
                HelpSection {
                    heading: "Large images",
                    bullets: &[
                        "The viewer draws available coarse pyramid tiles first and refines as higher-resolution tiles arrive.",
                        "Remote HTTP, S3, and R2 datasets may take longer to refine at high zoom because each visible tile has to be fetched.",
                        "If navigation feels slow, reduce visible channels, use Auto level, or pin only the levels and channels you need in the Memory tab.",
                    ],
                },
            ],
            HelpTopic::Layers => &[
                HelpSection {
                    heading: "Layer list",
                    bullets: &[
                        "Click a layer row to make it active; the active layer is the one edited by most toolbar and right-panel controls.",
                        "Use visibility toggles to show or hide layers without deleting them.",
                        "Drag layer rows to reorder draw order when an overlay should appear above or below another layer.",
                        "Use channel groups when related channels should share grouping, organization, or review context.",
                    ],
                },
                HelpSection {
                    heading: "Overlay layers",
                    bullets: &[
                        "Object, point, mask, annotation, label, and SpatialData layers can be drawn over image channels.",
                        "Many overlays expose color, opacity, width, selection, label, source, or analysis controls in the Properties tab.",
                        "Mask polygon editing only works when a mask layer is active and the pan tool is selected.",
                        "Object selection tools only work when the active layer is a loaded segmentation object layer or an object-backed shape layer.",
                    ],
                },
            ],
            HelpTopic::AutoLevel => &[
                HelpSection {
                    heading: "When to use it",
                    bullets: &[
                        "Keep Auto level enabled for normal navigation and review.",
                        "The viewer chooses a coarse or fine image pyramid level based on the current zoom and visible area.",
                        "Disable Auto level only when you need to inspect a specific stored resolution level or compare how a level was generated.",
                    ],
                },
                HelpSection {
                    heading: "Notes",
                    bullets: &[
                        "Manual level selection changes display resolution only; it does not modify the source dataset.",
                        "High-resolution levels can take longer to load and can use more memory, especially for remote storage or many visible channels.",
                        "If the image looks blocky while zooming, wait for refinement tiles or fit/zoom again after loading catches up.",
                    ],
                },
            ],
            HelpTopic::ImportOpen => &[
                HelpSection {
                    heading: "Project inputs",
                    bullets: &[
                        "Import Samplesheet CSV replaces the current project ROI list with rows from a CSV file; see Samplesheet CSV Format for required columns.",
                        "Export Samplesheet CSV writes local project ROIs back out as `id,path,...metadata` rows; remote ROIs are skipped.",
                        "Add OME-Zarr Root scans a local directory tree for OME-Zarr roots and adds each discovered ROI.",
                        "Open TIFF / OME-TIFF opens a local TIFF file directly in the single-dataset viewer and adds it to the current project list.",
                        "Drag a supported local dataset into the window to add it quickly; supported local inputs include OME-Zarr roots, Xenium folders, and TIFF files.",
                        "Open Remote accepts HTTP(S) OME-Zarr URLs, or S3/R2 endpoint, region, bucket, prefix, and credentials.",
                    ],
                },
                HelpSection {
                    heading: "Opening",
                    bullets: &[
                        "Select one ROI and choose Open to inspect it in the single-dataset viewer.",
                        "Select multiple OME-Zarr ROIs and choose Open mosaic to compare them in one canvas.",
                        "Use the ROI browser filters to narrow the list, then select the rows you want to open.",
                        "Saved project views can reopen an ROI with captured channel, camera, object, and overlay settings.",
                    ],
                },
                HelpSection {
                    heading: "Segmentation inputs",
                    bullets: &[
                        "Add Seg Search Root adds folders that Odon should search when matching segmentation files to ROIs.",
                        "Auto-match Seg tries to connect project ROIs with segmentation/object files using configured search roots and ROI metadata.",
                        "The Object Cache can preload GeoParquet or Parquet segmentation objects so repeated ROI review is faster.",
                        "Use Edit config (JSON) for advanced project metadata edits when the UI does not expose a field directly.",
                    ],
                },
            ],
            HelpTopic::Samplesheets => &[
                HelpSection {
                    heading: "Required columns",
                    bullets: &[
                        "A samplesheet must be a CSV file with a header row and at least two columns.",
                        "Column 1 is interpreted as `id`; use a stable ROI name such as `ROI_001` or `patient7_coreA`.",
                        "Column 2 is interpreted as `path`; use a supported local dataset root or file path.",
                        "The recommended header is `id,path` followed by any metadata columns you need.",
                        "Rows with an empty `id` or empty `path` are skipped; if no usable rows remain, import fails.",
                    ],
                },
                HelpSection {
                    heading: "Paths",
                    bullets: &[
                        "Relative `path` values are resolved relative to the samplesheet CSV file, not the current working directory.",
                        "Absolute paths are accepted as written.",
                        "For local project import, paths should point to supported datasets such as OME-Zarr roots, Xenium folders, or TIFF files.",
                        "For mosaic opening from the command line, rows must resolve to OME-Zarr datasets because mosaics use OME-Zarr sources.",
                    ],
                },
                HelpSection {
                    heading: "Metadata columns",
                    bullets: &[
                        "Every column after `id` and `path` is stored as ROI metadata and can be used for browsing, sorting, grouping, or project context.",
                        "`dataset` is a special metadata column; when present it sets the project dataset name for that ROI.",
                        "If `dataset` is blank or missing during project import, Odon uses the project's default dataset name.",
                        "`segpath` is a special metadata column; when present it points to a segmentation/object file or folder for that ROI.",
                        "Relative `segpath` values are resolved relative to the samplesheet CSV file.",
                    ],
                },
                HelpSection {
                    heading: "Example",
                    bullets: &[
                        "Header: `id,path,dataset,condition,batch,segpath`",
                        "Row: `ROI_001,rois/ROI_001.ome.zarr,cohort_a,treated,B1,segmentations/ROI_001.parquet`",
                        "Row: `ROI_002,/data/rois/ROI_002.ome.zarr,cohort_a,control,B1,/data/segs/ROI_002.parquet`",
                        "Use Export Samplesheet CSV on an existing project to generate a valid starting template.",
                    ],
                },
                HelpSection {
                    heading: "Import behavior",
                    bullets: &[
                        "Import Samplesheet CSV clears the current project ROI list before adding rows from the file.",
                        "After import, the first ROI is focused and selected so it can be opened immediately.",
                        "The import reports how many ROIs were added and how many metadata columns were found.",
                        "If import fails, check for a missing header row, fewer than two columns, no usable `id`/`path` rows, or quoting errors in the CSV.",
                    ],
                },
            ],
            HelpTopic::ObjectCache => &[
                HelpSection {
                    heading: "What it does",
                    bullets: &[
                        "Object Cache preloads project-linked GeoParquet or Parquet segmentation object files.",
                        "Use it when moving through many project ROIs that have segmentation/object files and repeated object loading feels slow.",
                        "The cache is for viewer performance; it does not change the segmentation source files or project metadata.",
                        "The summary line shows how many compatible files were found and how much cached object data is on disk.",
                    ],
                },
                HelpSection {
                    heading: "Modes",
                    bullets: &[
                        "Full geometry loads polygon geometry so object outlines, fills, measurements, and geometry-aware workflows can open faster.",
                        "Centroid points loads lighter point representations for faster browsing when full polygon geometry is not needed immediately.",
                        "Load properties lazily keeps table columns deferred until needed; this is usually the better default for wide object tables.",
                        "Disable lazy properties only when you know you will need many object columns immediately and have enough memory and IO bandwidth.",
                    ],
                },
                HelpSection {
                    heading: "Buttons and status",
                    bullets: &[
                        "Preload object segmentations starts loading all discovered compatible object files using the selected mode.",
                        "The preload button is enabled only when compatible files are available and the project has been saved to a path.",
                        "Clear object cache removes cached object data so it can be rebuilt with different settings or freed from disk.",
                        "The progress bar shows completed files; the status line reports cached, failed, loading, mode, and property-loading state.",
                    ],
                },
                HelpSection {
                    heading: "When to use it",
                    bullets: &[
                        "Use Object Cache after importing ROIs and matching segmentation paths, before a long review session.",
                        "If no files are available, add segmentation search roots, include `segpath` in the samplesheet, or run Auto-match Seg.",
                        "If failures are reported, check that segmentation paths exist and that the object files are readable GeoParquet or Parquet files.",
                    ],
                },
            ],
            HelpTopic::Tools => &[
                HelpSection {
                    heading: "Pan",
                    bullets: &[
                        "The default navigation and selection tool.",
                        "Left-drag pans the image unless the active layer has an editable object, polygon, or transform handle under the pointer.",
                        "Click object-backed layers to select objects, or click mask polygons to select editable polygon regions.",
                        "Use pan mode for most editing operations after choosing the layer you want to edit.",
                    ],
                },
                HelpSection {
                    heading: "Move layers",
                    bullets: &[
                        "Select the visible layer or layers you want to move in the layer list.",
                        "Choose the move tool, then drag on the canvas.",
                        "The layer offset changes in viewer/project state; the underlying source image, mask, or object file is not rewritten.",
                        "Use this for visual alignment corrections between channels, masks, labels, and object layers.",
                    ],
                },
                HelpSection {
                    heading: "Transform channel",
                    bullets: &[
                        "Available for supported channel layers in XY view.",
                        "Drag inside the transform box to translate.",
                        "Drag corners to scale and use the rotation handle to rotate.",
                        "Use the Properties tab to inspect or reset transform values when available.",
                    ],
                },
                HelpSection {
                    heading: "Draw mask polygon",
                    bullets: &[
                        "Creates editable mask polygons on the active mask layer, or creates an editable mask layer when the workflow needs one.",
                        "Click to add vertices.",
                        "Close with double-click, Enter, or by clicking the highlighted first vertex.",
                        "Switch back to pan mode to select, drag vertices, delete polygons, or use the right-click context menu.",
                    ],
                },
                HelpSection {
                    heading: "Rect and lasso select",
                    bullets: &[
                        "Available for active object-backed layers.",
                        "Rect select chooses cells by centroid inside a dragged rectangle.",
                        "Lasso select chooses cells by centroid inside a freehand polygon.",
                        "Selections feed the Analysis tab and object review workflows; they do not edit source segmentation geometry.",
                    ],
                },
            ],
            HelpTopic::PropertiesPanel => &[
                HelpSection {
                    heading: "Active layer",
                    bullets: &[
                        "The Properties tab changes based on the active layer in the layer list.",
                        "The active layer also determines which canvas tools can operate on it.",
                        "If expected controls are missing, click the target layer in the layer list first.",
                        "Use transform controls to inspect or reset layer offsets when the active layer supports transforms.",
                    ],
                },
                HelpSection {
                    heading: "Common controls",
                    bullets: &[
                        "Image channels expose contrast windows, color, channel grouping, transforms, histograms, notes, and quick display controls.",
                        "Mask and overlay layers expose visibility, opacity, line width, fill/outline color, source paths, and mask-specific actions.",
                        "Object-backed layers can expose selection, rendering, marker mapping, threshold/call, export, and analysis-related settings.",
                        "Threshold Regions appears under image-channel properties because it captures the visible region from the active channel.",
                    ],
                },
                HelpSection {
                    heading: "Edits versus display",
                    bullets: &[
                        "Contrast, color, visibility, and level settings affect display and project state, not the original image data.",
                        "Moving or transforming a layer changes viewer/project alignment state, not the source file on disk.",
                        "Mask polygon edits change editable mask layers and can be saved or exported through mask/project workflows.",
                    ],
                },
            ],
            HelpTopic::ViewsPanel => &[
                HelpSection {
                    heading: "Saved views",
                    bullets: &[
                        "Views capture reusable display state for project ROIs.",
                        "A view can store channel visibility, channel aliases, color settings, camera position, object display, selected value filters, and related overlay choices.",
                        "Use views when you want to reopen an ROI in a consistent review state.",
                        "Views are most useful for repeated review patterns such as tumour overview, marker detail, or object-call validation.",
                    ],
                },
                HelpSection {
                    heading: "Project workflow",
                    bullets: &[
                        "Open a project ROI before capturing a live viewer state.",
                        "Configure the viewer, click Capture, enter or review the view name and aliases, then Save view in the Project Views window.",
                        "Saved views are project-level metadata; they do not copy or modify image data.",
                        "Apply a view from the right panel, double-click it in the list, or open it from project ROI controls.",
                        "If Apply is disabled, make sure a project ROI is currently open or focused.",
                    ],
                },
            ],
            HelpTopic::AnalysisPanel => &[
                HelpSection {
                    heading: "Availability",
                    bullets: &[
                        "Analysis controls are available for loaded segmentation objects or object-backed SpatialData shape layers.",
                        "If no compatible object layer is loaded, the tab explains what is missing.",
                        "For SpatialData, make the object-backed shape layer active before opening the Analysis tab.",
                        "For project ROIs, use segmentation search roots, Auto-match Seg, Load Labels, or object loading workflows to make objects available.",
                    ],
                },
                HelpSection {
                    heading: "What it is for",
                    bullets: &[
                        "Review marker values and threshold/call state in the context of the image.",
                        "Create and manage selection elements for downstream review.",
                        "Map object properties to channels when names differ between image data and object tables.",
                        "Zoom to selected objects and compare object-derived properties with visible image channels.",
                    ],
                },
                HelpSection {
                    heading: "Common causes of confusion",
                    bullets: &[
                        "Image channels alone are not enough for this tab; analysis needs object geometry plus object properties.",
                        "A selected object layer can be visible while another layer is active; click the object layer if controls are not appearing.",
                        "Changing thresholds or calls in the viewer is review state and should be saved or exported through the relevant object/project workflow.",
                    ],
                },
            ],
            HelpTopic::MeasurementsPanel => &[
                HelpSection {
                    heading: "Availability",
                    bullets: &[
                        "Measurements are available for loaded segmentation objects and object-backed SpatialData shape layers.",
                        "The active or loaded object source determines which polygons and properties are measured.",
                        "Measurements require both object geometry and image channels to sample from.",
                    ],
                },
                HelpSection {
                    heading: "How to use it",
                    bullets: &[
                        "Load segmentation objects or choose an object-backed SpatialData shape layer first.",
                        "Choose the image channels and object source you want to summarize.",
                        "Use measurements to summarize image signal over polygon objects for review or downstream analysis.",
                        "Export measurement results when you need to continue analysis outside Odon.",
                    ],
                },
                HelpSection {
                    heading: "Interpretation",
                    bullets: &[
                        "Measurements depend on current object geometry, layer alignment, and channel data availability.",
                        "If results look offset, check layer transforms, object offsets, and whether the correct ROI/object source is active.",
                        "Use this as a viewer-side measurement and QC workflow rather than a complete statistical analysis environment.",
                    ],
                },
            ],
            HelpTopic::MemoryPanel => &[
                HelpSection {
                    heading: "Tile loading",
                    bullets: &[
                        "Tile workers load visible OME-Zarr chunks as you navigate.",
                        "Prefetch can load nearby or finer-resolution tiles ahead of time.",
                        "More aggressive worker and prefetch settings may feel smoother but can increase memory, disk, and network pressure.",
                        "Reducing visible channels is often the quickest way to lower tile load and memory use.",
                    ],
                },
                HelpSection {
                    heading: "Pinned levels",
                    bullets: &[
                        "Pinned levels keep selected channels and pyramid levels resident in CPU memory.",
                        "Use pinned data for repeated review of the same channels when memory allows.",
                        "Unpin data when the memory summary shows pressure or when switching tasks.",
                        "Pinned levels are manual; Odon estimates RAM use and warns about risk, but you decide whether to load.",
                        "In mosaics, you can pin selected channels for all ROIs or only the focused ROI.",
                    ],
                },
                HelpSection {
                    heading: "Choosing a level",
                    bullets: &[
                        "Lower-numbered levels are usually higher resolution and require more memory.",
                        "Higher-numbered levels are coarser and can be useful for fast overview review.",
                        "Pin only the channels and levels needed for the current review task, then unload them afterward.",
                    ],
                },
            ],
            HelpTopic::RoiSelectorPanel => &[
                HelpSection {
                    heading: "ROI navigation",
                    bullets: &[
                        "The ROI Selector follows project metadata for the current dataset.",
                        "Use previous and next controls to step through ROIs without returning to the project panel.",
                        "Open ROI switches the single viewer to the selected project ROI.",
                        "The selector is most useful after opening a project or samplesheet-driven workspace.",
                        "If no ROIs appear, return to the Project panel and import or add ROIs first.",
                    ],
                },
                HelpSection {
                    heading: "Labels and masks",
                    bullets: &[
                        "Load Labels loads segmentation label data for the current ROI when a label source is available.",
                        "Load Masks loads project-linked exclusion, artefact, or review masks for the current ROI.",
                        "Save Masks writes editable mask polygons back to the project mask target when supported.",
                        "Legacy Save Masks appends editable non-file-backed polygons to the project mask target for local datasets.",
                        "If Load or Save fails, check that the ROI is local, project metadata contains the expected mask or label path, and the target path is writable.",
                    ],
                },
                HelpSection {
                    heading: "Project context",
                    bullets: &[
                        "The ROI Selector does not create new ROIs; it navigates ROIs already present in the project.",
                        "Samplesheet metadata is carried into project ROIs and can help identify which ROI you are reviewing.",
                        "Use saved views with ROI navigation when each ROI should open with the same display recipe.",
                    ],
                },
            ],
            HelpTopic::Thresholding => &[
                HelpSection {
                    heading: "Workflow",
                    bullets: &[
                        "Select the image channel you want to threshold and make sure you are in XY view.",
                        "Pan and zoom so the region you want is visible, then start a threshold preview from the visible region.",
                        "Adjust Threshold to include pixels at or above the chosen value.",
                        "Adjust Min component pixels to remove small connected components from the preview.",
                        "Apply the preview to create a new editable mask layer.",
                    ],
                },
                HelpSection {
                    heading: "Preview behavior",
                    bullets: &[
                        "The preview operates on the active channel and the currently visible canvas region, not the whole dataset.",
                        "Refresh from visible region after panning, zooming, changing channel, or changing the area you want to capture.",
                        "The canvas overlay is a raster preview; after applying, Odon converts it into editable mask polygons.",
                        "If the visible region is very large or high resolution, preview generation may take longer.",
                    ],
                },
                HelpSection {
                    heading: "Practical use",
                    bullets: &[
                        "Use threshold regions for rapid review masks, artefact regions, or exclusion regions.",
                        "After applying a threshold mask, switch to mask polygon editing if you need to clean up the generated polygons.",
                        "Treat this as viewer-side mask authoring rather than a full statistical thresholding workflow.",
                    ],
                },
            ],
            HelpTopic::MaskPolygons => &[
                HelpSection {
                    heading: "Drawing",
                    bullets: &[
                        "Choose the polygon tool and click on the canvas to add vertices.",
                        "The first vertex changes color when the pointer is close enough to close the polygon.",
                        "Close with double-click, Enter, or by clicking the highlighted first vertex.",
                        "Use Backspace to remove the last in-progress point and Esc to cancel.",
                        "A polygon needs at least three points before it can close.",
                    ],
                },
                HelpSection {
                    heading: "Editing",
                    bullets: &[
                        "Make a mask layer active, switch to pan, and click a polygon to select it.",
                        "Click an edge, handle, or polygon interior to select the polygon.",
                        "Drag vertex handles to reshape the selected polygon; dragging the first vertex keeps the closing vertex in sync.",
                        "Drag empty canvas space to pan when you are not on a polygon handle.",
                        "Delete with Delete, Backspace, the right-click menu, or the mask layer properties panel.",
                        "Undo mask or layer-move edits with Ctrl+Z or Cmd+Z.",
                    ],
                },
                HelpSection {
                    heading: "Saving and export",
                    bullets: &[
                        "Editable mask polygons live on mask layers until saved, exported, or stored in project state.",
                        "Use Save Masks from project/ROI workflows when masks should be written back to the project target.",
                        "Use GeoJSON import/export when masks need to move between Odon and downstream analysis code.",
                        "Project save preserves project-linked mask state, but it is still worth exporting when another tool needs the geometry.",
                    ],
                },
            ],
            HelpTopic::Shortcuts => &[
                HelpSection {
                    heading: "Global",
                    bullets: &[
                        "F fits the active view target.",
                        "Ctrl+W or Cmd+W opens the close confirmation.",
                        "Ctrl+M opens analysis mapping settings where available.",
                        "Ctrl+A or Cmd+A selects all visible ROIs in the project ROI browser when the browser has focus and no text field is active.",
                        "Mouse wheel or trackpad pinch zooms around the pointer.",
                    ],
                },
                HelpSection {
                    heading: "Mask polygons",
                    bullets: &[
                        "Enter closes an in-progress polygon.",
                        "Esc cancels drawing or clears polygon selection.",
                        "Backspace removes the last in-progress point, or deletes a selected polygon in pan mode.",
                        "Delete deletes the selected polygon.",
                        "Ctrl+Z or Cmd+Z undoes the previous mask or layer-move edit.",
                    ],
                },
                HelpSection {
                    heading: "Project and views",
                    bullets: &[
                        "Double-click a project ROI row to open it when supported by the project panel.",
                        "Double-click a saved view to apply it to the focused project ROI.",
                        "Use text fields normally; shortcuts are ignored while the UI wants keyboard input.",
                    ],
                },
            ],
        }
    }
}

struct HelpSection {
    heading: &'static str,
    bullets: &'static [&'static str],
}

pub fn help_button(ui: &mut egui::Ui, topic: HelpTopic) -> bool {
    ui.small_button("?")
        .on_hover_text(format!("Open help: {}", topic.title()))
        .clicked()
}

pub fn show_help_window(ctx: &egui::Context, active_topic: &mut Option<HelpTopic>) {
    let Some(mut topic) = *active_topic else {
        return;
    };

    let mut open = true;
    egui::Window::new("Odon Documentation")
        .id(egui::Id::new("odon.help.window"))
        .collapsible(false)
        .resizable(true)
        .default_size(egui::vec2(760.0, 520.0))
        .min_width(480.0)
        .min_height(320.0)
        .open(&mut open)
        .show(ctx, |ui| {
            let available = ui.available_size();
            ui.allocate_ui_with_layout(
                available,
                egui::Layout::left_to_right(egui::Align::Min),
                |ui| {
                    let nav_width = 210.0;
                    ui.allocate_ui_with_layout(
                        egui::vec2(nav_width, available.y),
                        egui::Layout::top_down(egui::Align::Min),
                        |ui| {
                            ui.set_min_size(egui::vec2(nav_width, available.y));
                            ui.set_max_width(nav_width);
                            ui.heading("Topics");
                            ui.separator();
                            let scroll_height = ui.available_height();
                            egui::ScrollArea::vertical()
                                .id_salt("odon.help.topic_list")
                                .max_height(scroll_height)
                                .auto_shrink([false, false])
                                .show(ui, |ui| {
                                    for candidate in HelpTopic::ALL {
                                        if ui
                                            .selectable_label(
                                                topic == *candidate,
                                                candidate.title(),
                                            )
                                            .clicked()
                                        {
                                            topic = *candidate;
                                        }
                                    }
                                });
                        },
                    );

                    ui.separator();

                    let content_size = egui::vec2(ui.available_width(), available.y);
                    ui.allocate_ui_with_layout(
                        content_size,
                        egui::Layout::top_down(egui::Align::Min),
                        |ui| {
                            ui.set_min_size(content_size);
                            ui.heading(
                                egui::RichText::new(topic.title()).color(egui::Color32::WHITE),
                            );
                            ui.label(topic.summary());
                            ui.add_space(8.0);
                            let scroll_height = ui.available_height();
                            egui::ScrollArea::vertical()
                                .id_salt("odon.help.topic_content")
                                .max_height(scroll_height)
                                .auto_shrink([false, false])
                                .show(ui, |ui| {
                                    for section in topic.sections() {
                                        ui.add_space(8.0);
                                        ui.strong(section.heading);
                                        for bullet in section.bullets {
                                            ui.horizontal_wrapped(|ui| {
                                                ui.label("-");
                                                ui.label(*bullet);
                                            });
                                        }
                                    }
                                });
                        },
                    );
                },
            );
        });

    if !open {
        *active_topic = None;
    } else {
        *active_topic = Some(topic);
    }
}
