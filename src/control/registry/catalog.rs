//! Authoritative application-method descriptors.

use std::sync::LazyLock;

use super::*;

pub static METHODS: LazyLock<Vec<MethodDescriptor>> = LazyLock::new(|| {
    vec![
        method!(
            "app.get_state",
            "Get current application and viewer state.",
            "viewer.read",
            false,
            false,
            None,
            ALL_MODES,
            Empty
        ),
        method!(
            "app.settings.get",
            "Inspect persistent application preferences.",
            "application.settings.read",
            false,
            false,
            None,
            ALL_MODES,
            Empty
        ),
        method!(
            "app.settings.set",
            "Validate, persist, and apply application preferences.",
            "application.settings.write",
            true,
            false,
            Some("application.settings.changed"),
            ALL_MODES,
            AppSettings
        ),
        method!(
            "app.recent_projects.list",
            "List recently opened project files.",
            "application.settings.read",
            false,
            false,
            None,
            ALL_MODES,
            Empty
        ),
        method!(
            "app.recent_projects.forget",
            "Forget one recently opened project path.",
            "application.settings.write",
            true,
            false,
            Some("application.recent_projects.changed"),
            ALL_MODES,
            Path
        ),
        method!(
            "app.recent_projects.clear",
            "Clear the recent-project list.",
            "application.settings.write",
            true,
            false,
            Some("application.recent_projects.changed"),
            ALL_MODES,
            Empty
        ),
        method!(
            "app.lifecycle.get",
            "Inspect dirty state and safe close options.",
            "application.lifecycle.read",
            false,
            false,
            None,
            ALL_MODES,
            Empty
        ),
        method!(
            "app.lifecycle.request_close",
            "Request that the Odon window close with an explicit save decision.",
            "application.close",
            true,
            false,
            Some("application.close.requested"),
            READY_MODES,
            LifecycleRequest
        ),
        method!(
            "app.lifecycle.request_quit",
            "Request that Odon quit with an explicit save decision.",
            "application.quit",
            true,
            false,
            Some("application.quit.requested"),
            READY_MODES,
            LifecycleRequest
        ),
        MethodDescriptor {
            name: "app.get_method_availability",
            summary: "Describe whether control methods are available in the current mode.",
            capability: "system.introspect",
            mutates: false,
            starts_task: false,
            mcp_exposed: false,
            stability: Stability::Provisional,
            request_shape: RequestShape::MethodAvailability,
            event: None,
            available_in: ALL_MODES,
            since: "0.2.0",
            execution_class: execution_class("app.get_method_availability", false),
        },
        method!(
            "project.rois.list",
            "List project ROIs.",
            "project.read",
            false,
            false,
            None,
            READY_MODES,
            Empty
        ),
        method!(
            "project.get",
            "Get project metadata and lifecycle state.",
            "project.read",
            false,
            false,
            None,
            READY_MODES,
            Empty
        ),
        method!(
            "project.create",
            "Create a new empty project workspace.",
            "project.write",
            true,
            false,
            Some("application.mode.changed"),
            READY_MODES,
            ProjectCreate
        ),
        method!(
            "project.save_as",
            "Save the active project to an explicit path.",
            "project.write",
            true,
            false,
            Some("project.saved"),
            READY_MODES,
            Path
        ),
        method!(
            "project.update_metadata",
            "Update supported project metadata and search roots.",
            "project.write",
            true,
            false,
            Some("project.changed"),
            READY_MODES,
            ProjectMetadata
        ),
        method!(
            "project.samplesheets.inspect",
            "Parse and validate a samplesheet without changing the active project.",
            "project.read",
            false,
            false,
            None,
            ALL_MODES,
            SamplesheetInspect
        ),
        method!(
            "project.samplesheets.validate",
            "Validate samplesheet identity, paths, and metadata without changing the project.",
            "project.read",
            false,
            false,
            None,
            ALL_MODES,
            SamplesheetInspect
        ),
        method!(
            "project.samplesheets.import",
            "Replace project ROIs from a validated samplesheet.",
            "project.write",
            true,
            true,
            Some("project.rois.changed"),
            READY_MODES,
            Path
        ),
        method!(
            "project.samplesheets.export",
            "Export local project ROIs and metadata to a samplesheet.",
            "project.export",
            true,
            false,
            Some("project.samplesheet.exported"),
            READY_MODES,
            SamplesheetExport
        ),
        method!(
            "project.discovery.add_root",
            "Discover OME-Zarr datasets recursively and add them as project ROIs.",
            "project.write",
            true,
            true,
            Some("project.rois.changed"),
            READY_MODES,
            Path
        ),
        method!(
            "project.objects.preload.get",
            "Inspect available and cached project object segmentations.",
            "project.objects.read",
            false,
            false,
            None,
            READY_MODES,
            Empty
        ),
        method!(
            "project.objects.preload.list_sources",
            "List preload-eligible project segmentation sources.",
            "project.objects.read",
            false,
            false,
            None,
            READY_MODES,
            MosaicItems
        ),
        method!(
            "project.objects.preload.start",
            "Preload project object geometry or centroids and wait for completion.",
            "project.objects.write",
            true,
            true,
            Some("project.objects.preload.changed"),
            READY_MODES,
            ObjectPreloadStart
        ),
        method!(
            "project.objects.preload.clear",
            "Clear preloaded project objects from memory.",
            "project.objects.write",
            true,
            false,
            Some("project.objects.preload.changed"),
            READY_MODES,
            Empty
        ),
        method!(
            "project.rois.get",
            "Get one project ROI by stable ID.",
            "project.read",
            false,
            false,
            None,
            READY_MODES,
            ProjectRoiId
        ),
        method!(
            "project.rois.add",
            "Add a project ROI.",
            "project.write",
            true,
            false,
            Some("project.rois.changed"),
            READY_MODES,
            ProjectRoiAdd
        ),
        method!(
            "project.rois.update",
            "Update a project ROI.",
            "project.write",
            true,
            false,
            Some("project.rois.changed"),
            READY_MODES,
            ProjectRoiUpdate
        ),
        method!(
            "project.rois.remove",
            "Remove a project ROI.",
            "project.write",
            true,
            false,
            Some("project.rois.changed"),
            READY_MODES,
            ProjectRoiId
        ),
        method!(
            "project.rois.reorder",
            "Set the exact project ROI order.",
            "project.write",
            true,
            false,
            Some("project.rois.changed"),
            READY_MODES,
            ProjectRoiOrder
        ),
        method!(
            "project.rois.get_selection",
            "Get focused and selected project ROIs.",
            "project.read",
            false,
            false,
            None,
            READY_MODES,
            Empty
        ),
        method!(
            "project.rois.select",
            "Select project ROIs by stable ID.",
            "project.write",
            true,
            false,
            Some("project.rois.selection_changed"),
            READY_MODES,
            ProjectRoiSelect
        ),
        method!(
            "project.rois.focus",
            "Focus a project ROI by stable ID.",
            "project.write",
            true,
            false,
            Some("project.rois.selection_changed"),
            READY_MODES,
            ProjectRoiId
        ),
        method!(
            "project.rois.next",
            "Focus the next project ROI.",
            "project.write",
            true,
            false,
            Some("project.rois.selection_changed"),
            READY_MODES,
            StepPlane
        ),
        method!(
            "project.rois.previous",
            "Focus the previous project ROI.",
            "project.write",
            true,
            false,
            Some("project.rois.selection_changed"),
            READY_MODES,
            StepPlane
        ),
        method!(
            "project.rois.open_selected_mosaic",
            "Open selected project ROIs as a mosaic.",
            "project.write",
            true,
            true,
            Some("application.mode.changed"),
            READY_MODES,
            Empty
        ),
        method!(
            "viewer.channels.list",
            "List channels.",
            "viewer.channels.read",
            false,
            false,
            None,
            READY_MODES,
            Empty
        ),
        method!(
            "viewer.channels.list_visible",
            "List visible channels.",
            "viewer.channels.read",
            false,
            false,
            None,
            READY_MODES,
            Empty
        ),
        method!(
            "viewer.planes.get",
            "Get the active view plane, slice, extent, and supported orientations.",
            "viewer.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Empty
        ),
        method!(
            "viewer.planes.set",
            "Set the active view orientation and/or slice.",
            "viewer.write",
            true,
            false,
            Some("viewer.planes.changed"),
            SINGLE_MODE,
            SetPlane
        ),
        method!(
            "viewer.planes.next",
            "Move forward through slices in the active view orientation.",
            "viewer.write",
            true,
            false,
            Some("viewer.planes.changed"),
            SINGLE_MODE,
            StepPlane
        ),
        method!(
            "viewer.planes.previous",
            "Move backward through slices in the active view orientation.",
            "viewer.write",
            true,
            false,
            Some("viewer.planes.changed"),
            SINGLE_MODE,
            StepPlane
        ),
        method!(
            "viewer.planes.operation_availability",
            "Describe XY-only operation safeguards for the active multidimensional view plane.",
            "viewer.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Empty
        ),
        method!(
            "viewer.panels.get",
            "Get side-panel visibility.",
            "viewer.read",
            false,
            false,
            None,
            VIEWER_MODES,
            Empty
        ),
        method!(
            "viewer.panels.set",
            "Set side-panel visibility.",
            "viewer.write",
            true,
            false,
            Some("viewer.panels.changed"),
            VIEWER_MODES,
            SetSidePanels
        ),
        method!(
            "viewer.rendering.get_smooth_pixels",
            "Get image interpolation state.",
            "viewer.read",
            false,
            false,
            None,
            VIEWER_MODES,
            Empty
        ),
        method!(
            "viewer.rendering.set_smooth_pixels",
            "Set image interpolation state.",
            "viewer.write",
            true,
            false,
            Some("viewer.rendering.changed"),
            VIEWER_MODES,
            SetSmoothPixels
        ),
        method!(
            "viewer.rendering.get_state",
            "Inspect renderer, additive compositing, interpolation, and deterministic-capture readiness.",
            "viewer.read",
            false,
            false,
            None,
            VIEWER_MODES,
            Empty
        ),
        method!(
            "app.get_loading_state",
            "Get loading diagnostics.",
            "viewer.read",
            false,
            false,
            None,
            ALL_MODES,
            Empty
        ),
        method!(
            "viewer.channels.get_active",
            "Get the active channel.",
            "viewer.channels.read",
            false,
            false,
            None,
            READY_MODES,
            Object
        ),
        method!(
            "viewer.channels.set_active",
            "Set the active channel.",
            "viewer.channels.write",
            true,
            false,
            Some("viewer.channels.changed"),
            VIEWER_MODES,
            Object
        ),
        method!(
            "viewer.channels.set_visible",
            "Set channel visibility.",
            "viewer.channels.write",
            true,
            false,
            Some("viewer.channels.changed"),
            VIEWER_MODES,
            SetVisibleChannels
        ),
        method!(
            "viewer.channels.set_color",
            "Set a channel's additive-compositing colour.",
            "viewer.channels.write",
            true,
            false,
            Some("viewer.channels.changed"),
            VIEWER_MODES,
            SetChannelColor
        ),
        method!(
            "viewer.channels.set_note",
            "Set a channel note.",
            "viewer.channels.write",
            true,
            false,
            Some("viewer.channels.changed"),
            VIEWER_MODES,
            SetChannelNote
        ),
        method!(
            "viewer.channels.get_transform",
            "Get a channel's translation, scale, and rotation.",
            "viewer.channels.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.channels.set_transform",
            "Set a channel's translation, scale, and rotation.",
            "viewer.channels.write",
            true,
            false,
            Some("viewer.channels.changed"),
            SINGLE_MODE,
            SetChannelTransform
        ),
        method!(
            "viewer.channels.reset_transform",
            "Reset a channel transform to identity.",
            "viewer.channels.write",
            true,
            false,
            Some("viewer.channels.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "project.open",
            "Open a project.",
            "project.write",
            true,
            true,
            Some("application.mode.changed"),
            ALL_MODES,
            Object
        ),
        method!(
            "datasets.open_ome_zarr",
            "Open an OME-Zarr dataset.",
            "application.open",
            true,
            true,
            Some("application.mode.changed"),
            ALL_MODES,
            Object
        ),
        method!(
            "datasets.inspect",
            "Inspect a local dataset source and discover supported elements without opening it.",
            "datasets.read",
            false,
            false,
            None,
            ALL_MODES,
            Path
        ),
        method!(
            "datasets.open_spatialdata",
            "Open a selected SpatialData image with typed image, label, shape, and point elements.",
            "application.open",
            true,
            true,
            Some("application.mode.changed"),
            ALL_MODES,
            SpatialDataOpen
        ),
        method!(
            "datasets.open_xenium",
            "Open a Xenium experiment with explicit imagery and overlay choices.",
            "application.open",
            true,
            true,
            Some("application.mode.changed"),
            ALL_MODES,
            XeniumOpen
        ),
        method!(
            "datasets.open_http",
            "Open a remote HTTP(S) OME-Zarr source.",
            "application.open",
            true,
            true,
            Some("application.mode.changed"),
            ALL_MODES,
            HttpOpen
        ),
        method!(
            "datasets.s3.get_session",
            "Inspect redacted session-only S3 connection metadata.",
            "datasets.credentials.read",
            false,
            false,
            None,
            ALL_MODES,
            Empty
        ),
        method!(
            "datasets.s3.configure_session",
            "Configure session-only S3 credentials without persisting or returning secrets.",
            "datasets.credentials.write",
            true,
            false,
            Some("datasets.credentials.changed"),
            ALL_MODES,
            S3Session
        ),
        method!(
            "datasets.s3.clear_session",
            "Remove session-only S3 credentials from Odon memory.",
            "datasets.credentials.write",
            true,
            false,
            Some("datasets.credentials.changed"),
            ALL_MODES,
            Empty
        ),
        method!(
            "datasets.s3.list",
            "List one S3 prefix using the configured session credentials.",
            "datasets.remote.read",
            false,
            true,
            None,
            ALL_MODES,
            S3Prefix
        ),
        method!(
            "datasets.open_s3",
            "Open an S3 OME-Zarr prefix using session credentials.",
            "application.open",
            true,
            true,
            Some("application.mode.changed"),
            ALL_MODES,
            S3Prefix
        ),
        method!(
            "deep_links.parse",
            "Parse and validate an Odon deep link into its structured public model.",
            "deep_links.read",
            false,
            false,
            None,
            ALL_MODES,
            DeepLinkUri
        ),
        method!(
            "deep_links.resolve",
            "Resolve a deep link against its project and return an unambiguous ROI without changing application state.",
            "deep_links.read",
            false,
            false,
            None,
            READY_MODES,
            DeepLinkApply
        ),
        method!(
            "deep_links.filters.get",
            "Extract the typed object-filter state carried by a deep link without applying it.",
            "deep_links.read",
            false,
            false,
            None,
            ALL_MODES,
            DeepLinkApply
        ),
        method!(
            "deep_links.generate",
            "Generate a canonical Odon deep link from structured or current viewer state.",
            "deep_links.read",
            false,
            false,
            None,
            READY_MODES,
            DeepLinkGenerate
        ),
        method!(
            "deep_links.apply",
            "Apply a validated deep link as an atomic actor transaction and settle after its model and resources are ready.",
            "application.write",
            true,
            true,
            Some("application.state.changed"),
            ALL_MODES,
            DeepLinkApply
        ),
        method!(
            "datasets.open_tiff",
            "Open a TIFF dataset.",
            "application.open",
            true,
            true,
            Some("application.mode.changed"),
            ALL_MODES,
            TiffOpen
        ),
        method!(
            "datasets.open_mosaic_samplesheet",
            "Open a mosaic samplesheet.",
            "application.open",
            true,
            true,
            Some("application.mode.changed"),
            ALL_MODES,
            Object
        ),
        method!(
            "project.rois.open",
            "Open a project ROI.",
            "project.write",
            true,
            true,
            Some("project.active_roi.changed"),
            ALL_MODES,
            Object
        ),
        method!(
            "project.save",
            "Save the active project.",
            "project.write",
            true,
            false,
            Some("project.saved"),
            READY_MODES,
            Empty
        ),
        method!(
            "project.views.list",
            "List saved project view presets.",
            "project.read",
            false,
            false,
            None,
            READY_MODES,
            Empty
        ),
        method!(
            "project.views.get",
            "Get a saved project view preset.",
            "project.read",
            false,
            false,
            None,
            READY_MODES,
            ProjectViewSelector
        ),
        method!(
            "project.views.create",
            "Create or replace a saved project view preset from a specification.",
            "project.write",
            true,
            false,
            Some("project.views.changed"),
            READY_MODES,
            ProjectViewCreate
        ),
        method!(
            "project.views.capture",
            "Capture the current single-image viewer as a saved project view preset.",
            "project.write",
            true,
            false,
            Some("project.views.changed"),
            SINGLE_MODE,
            ProjectViewCapture
        ),
        method!(
            "project.views.rename",
            "Rename a saved project view preset.",
            "project.write",
            true,
            false,
            Some("project.views.changed"),
            READY_MODES,
            ProjectViewRename
        ),
        method!(
            "project.views.delete",
            "Delete a saved project view preset.",
            "project.write",
            true,
            false,
            Some("project.views.changed"),
            READY_MODES,
            ProjectViewSelector
        ),
        method!(
            "project.views.apply",
            "Apply a saved project view preset to the current single-image viewer.",
            "project.write",
            true,
            false,
            Some("project.views.applied"),
            SINGLE_MODE,
            ProjectViewSelector
        ),
        method!(
            "viewer.channels.get_contrast",
            "Get channel contrast.",
            "viewer.channels.read",
            false,
            false,
            None,
            VIEWER_MODES,
            Object
        ),
        method!(
            "viewer.channels.set_contrast",
            "Set channel contrast.",
            "viewer.channels.write",
            true,
            false,
            Some("viewer.channels.changed"),
            VIEWER_MODES,
            Object
        ),
        method!(
            "viewer.objects.get_visibility",
            "Get object overlay visibility.",
            "viewer.layers.read",
            false,
            false,
            None,
            VIEWER_MODES,
            Object
        ),
        method!(
            "viewer.native_layers.list",
            "List Odon-native layers in their channel and overlay stacks.",
            "viewer.layers.read",
            false,
            false,
            None,
            VIEWER_MODES,
            Empty
        ),
        method!(
            "viewer.native_layers.get",
            "Get one Odon-native layer.",
            "viewer.layers.read",
            false,
            false,
            None,
            VIEWER_MODES,
            NativeLayerSelector
        ),
        method!(
            "viewer.native_layers.set_active",
            "Set the active Odon-native layer.",
            "viewer.layers.write",
            true,
            false,
            Some("viewer.layers.changed"),
            VIEWER_MODES,
            NativeLayerSelector
        ),
        method!(
            "viewer.native_layers.set_visibility",
            "Set an Odon-native layer's visibility.",
            "viewer.layers.write",
            true,
            false,
            Some("viewer.layers.changed"),
            VIEWER_MODES,
            NativeLayerVisibility
        ),
        method!(
            "viewer.native_layers.set_order",
            "Set the exact order of the native channel or overlay stack.",
            "viewer.layers.write",
            true,
            false,
            Some("viewer.layers.changed"),
            VIEWER_MODES,
            NativeLayerOrder
        ),
        method!(
            "viewer.native_layers.set_offset",
            "Set an Odon-native layer's world translation.",
            "viewer.layers.write",
            true,
            false,
            Some("viewer.layers.changed"),
            SINGLE_MODE,
            NativeLayerOffset
        ),
        method!(
            "viewer.native_layers.reset_offset",
            "Reset an Odon-native layer's world translation to its loaded baseline.",
            "viewer.layers.write",
            true,
            false,
            Some("viewer.layers.changed"),
            SINGLE_MODE,
            NativeLayerSelector
        ),
        method!(
            "viewer.objects.set_visibility",
            "Set object overlay visibility.",
            "viewer.layers.write",
            true,
            false,
            Some("viewer.layers.changed"),
            VIEWER_MODES,
            Object
        ),
        method!(
            "viewer.objects.get_state",
            "Get bounded object source, loading, rendering, styling, filter, and selection state.",
            "viewer.objects.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.objects.source.load",
            "Load a CSV, GeoJSON, Parquet, or GeoParquet object source and settle when parsing finishes.",
            "viewer.objects.write",
            true,
            true,
            Some("viewer.objects.source.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.objects.source.reload",
            "Reload the current object source and settle when parsing finishes.",
            "viewer.objects.write",
            true,
            true,
            Some("viewer.objects.source.changed"),
            SINGLE_MODE,
            Empty
        ),
        method!(
            "viewer.objects.source.clear",
            "Clear the current object source and all derived object state.",
            "viewer.objects.write",
            true,
            false,
            Some("viewer.objects.source.changed"),
            SINGLE_MODE,
            Empty
        ),
        method!(
            "viewer.objects.source.cancel_load",
            "Cooperatively cancel the current object-source load.",
            "viewer.objects.write",
            true,
            false,
            Some("viewer.objects.source.changed"),
            SINGLE_MODE,
            Empty
        ),
        method!(
            "viewer.objects.style.get",
            "Get complete object appearance, color-property, and bounded legend state.",
            "viewer.objects.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.objects.style.set",
            "Set object visibility, stroke, fill, selection overlay, and color-property appearance.",
            "viewer.objects.write",
            true,
            false,
            Some("viewer.objects.style.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.objects.legend.set",
            "Set visibility and color overrides for object color-property legend values.",
            "viewer.objects.write",
            true,
            false,
            Some("viewer.objects.style.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.objects.rendering.get_fast",
            "Get fast object-rendering mode.",
            "viewer.objects.read",
            false,
            false,
            None,
            VIEWER_MODES,
            Object
        ),
        method!(
            "viewer.objects.rendering.set_fast",
            "Set fast object-rendering mode.",
            "viewer.objects.write",
            true,
            false,
            Some("viewer.objects.rendering.changed"),
            VIEWER_MODES,
            Object
        ),
        method!(
            "viewer.objects.properties.list",
            "List the object property schema with bounded pagination and lazy-load state.",
            "viewer.objects.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.objects.properties.load",
            "Load one lazy object property column and settle when its values are available.",
            "viewer.objects.write",
            true,
            true,
            Some("viewer.objects.properties.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.objects.properties.values",
            "Read a bounded page of typed values for one loaded object property.",
            "viewer.objects.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.objects.get_selection",
            "Get selected objects.",
            "viewer.objects.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.objects.query_rect",
            "Query objects in a rectangle.",
            "viewer.objects.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.objects.query_view",
            "Query objects in the viewport.",
            "viewer.objects.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.objects.query_lasso",
            "Query objects intersecting a world-coordinate lasso with bounded results.",
            "viewer.objects.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.objects.select_rect",
            "Select objects in a rectangle.",
            "viewer.objects.write",
            true,
            false,
            Some("viewer.selection.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.objects.select_lasso",
            "Select objects intersecting a world-coordinate lasso with explicit set semantics.",
            "viewer.objects.write",
            true,
            false,
            Some("viewer.selection.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.objects.clear_selection",
            "Clear object selection.",
            "viewer.objects.write",
            true,
            false,
            Some("viewer.selection.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.objects.selection.select_ids",
            "Select objects by stable IDs with replace, add, remove, or toggle semantics.",
            "viewer.objects.write",
            true,
            false,
            Some("viewer.selection.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.objects.selection.select_filtered",
            "Apply an explicitly sourced viewport filter or standalone query to selection.",
            "viewer.objects.write",
            true,
            false,
            Some("viewer.selection.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.objects.focus.set",
            "Focus an object by stable ID or index and optionally fit it in the viewport.",
            "viewer.objects.write",
            true,
            false,
            Some("viewer.objects.focus.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.objects.focus.clear",
            "Clear primary object focus without clearing the selection set.",
            "viewer.objects.write",
            true,
            false,
            Some("viewer.objects.focus.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.objects.selection.state.replace",
            "Atomically replace committed primary object selection with generation checking.",
            "viewer.objects.write",
            true,
            false,
            Some("viewer.selection.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.objects.get_filter",
            "Get object filter state.",
            "viewer.objects.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.objects.set_filter",
            "Set an object filter query.",
            "viewer.objects.write",
            true,
            false,
            Some("viewer.layers.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.objects.clear_filter",
            "Clear object filtering.",
            "viewer.objects.write",
            true,
            false,
            Some("viewer.layers.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.objects.filters.set_model",
            "Set the complete typed simple-clause or boolean-query object filter model.",
            "viewer.objects.write",
            true,
            false,
            Some("viewer.objects.filter.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.objects.filters.get_revision",
            "Get the monotonic object-filter revision and bounded visible/hidden counts shared by downstream consumers.",
            "viewer.objects.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.masks.layers.list",
            "List editable and read-only mask layers with complete presentation state.",
            "viewer.masks.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Empty
        ),
        method!(
            "viewer.masks.layers.get",
            "Get one mask layer by stable ID.",
            "viewer.masks.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.masks.layers.create",
            "Create an editable mask layer.",
            "viewer.masks.write",
            true,
            false,
            Some("viewer.masks.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.masks.layers.update",
            "Update mask layer name, presentation, editability, or offset.",
            "viewer.masks.write",
            true,
            false,
            Some("viewer.masks.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.masks.layers.delete",
            "Delete a mask layer and its polygons.",
            "viewer.masks.write",
            true,
            false,
            Some("viewer.masks.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.masks.polygons.list",
            "List a bounded page of mask polygons in local and world coordinates.",
            "viewer.masks.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.masks.polygons.add",
            "Add a closed polygon to an editable mask layer.",
            "viewer.masks.write",
            true,
            false,
            Some("viewer.masks.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.masks.polygons.update",
            "Replace the vertices of one editable mask polygon.",
            "viewer.masks.write",
            true,
            false,
            Some("viewer.masks.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.masks.polygons.remove",
            "Remove one polygon from an editable mask layer.",
            "viewer.masks.write",
            true,
            false,
            Some("viewer.masks.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.masks.selection.get",
            "Get the selected mask polygon and optional selected vertex.",
            "viewer.masks.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Empty
        ),
        method!(
            "viewer.masks.selection.set",
            "Select one mask polygon and optional vertex by layer ID and index.",
            "viewer.masks.write",
            true,
            false,
            Some("viewer.masks.selection.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.masks.selection.clear",
            "Clear the selected mask polygon and vertex.",
            "viewer.masks.write",
            true,
            false,
            Some("viewer.masks.selection.changed"),
            SINGLE_MODE,
            Empty
        ),
        method!(
            "viewer.masks.undo",
            "Undo the most recent mask or mask-offset edit.",
            "viewer.masks.write",
            true,
            false,
            Some("viewer.masks.changed"),
            SINGLE_MODE,
            Empty
        ),
        method!(
            "viewer.masks.state.replace",
            "Atomically replace committed mask state with optional generation conflict checking.",
            "viewer.masks.write",
            true,
            false,
            Some("viewer.masks.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.masks.import_geojson",
            "Import GeoJSON polygon or line geometry as a mask layer.",
            "viewer.masks.write",
            true,
            false,
            Some("viewer.masks.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.masks.export_geojson",
            "Export one mask layer or all mask layers as GeoJSON.",
            "viewer.masks.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.masks.persistence.get",
            "Inspect mask persistence state for the current dataset and project.",
            "viewer.masks.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Empty
        ),
        method!(
            "viewer.masks.persistence.sync",
            "Synchronize live mask layers into the current project in memory.",
            "viewer.masks.write",
            true,
            false,
            Some("viewer.masks.changed"),
            SINGLE_MODE,
            Empty
        ),
        method!(
            "viewer.masks.persistence.append_geojson",
            "Append editable non-file-backed masks to a project GeoJSON, clear only the saved polygons, and reload the read-only source layer.",
            "viewer.masks.write",
            true,
            false,
            Some("viewer.masks.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.labels.list",
            "List discovered NGFF label groups and current render state.",
            "viewer.labels.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Empty
        ),
        method!(
            "viewer.labels.get",
            "Inspect current NGFF label selection, loading, visibility, and alignment state.",
            "viewer.labels.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Empty
        ),
        method!(
            "viewer.labels.load",
            "Load one discovered NGFF label group into the shared label renderer.",
            "viewer.labels.write",
            true,
            false,
            Some("viewer.labels.changed"),
            SINGLE_MODE,
            LabelLoad
        ),
        method!(
            "viewer.labels.unload",
            "Unload the active NGFF label group and release its loader state.",
            "viewer.labels.write",
            true,
            false,
            Some("viewer.labels.changed"),
            SINGLE_MODE,
            Empty
        ),
        method!(
            "viewer.labels.set_visibility",
            "Set NGFF label visibility, loading the selected group when necessary.",
            "viewer.labels.write",
            true,
            false,
            Some("viewer.labels.changed"),
            SINGLE_MODE,
            LabelVisibility
        ),
        method!(
            "viewer.thresholds.levels.list",
            "List image levels and whole-image threshold safety limits.",
            "viewer.thresholds.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Empty
        ),
        method!(
            "viewer.screenshot.settings.get",
            "Inspect canvas screenshot overlay, scaling, quick-save, and readiness settings.",
            "viewer.screenshot",
            false,
            false,
            None,
            VIEWER_MODES,
            Empty
        ),
        method!(
            "viewer.scale_bar.get",
            "Inspect canvas scale-bar visibility and availability.",
            "viewer.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Empty
        ),
        method!(
            "viewer.scale_bar.set",
            "Set canvas scale-bar visibility.",
            "viewer.write",
            true,
            false,
            Some("viewer.scale_bar.changed"),
            SINGLE_MODE,
            SetScaleBar
        ),
        method!(
            "viewer.screenshot.settings.set",
            "Set canvas screenshot overlay, scaling, and quick-save folder options.",
            "viewer.screenshot",
            true,
            false,
            Some("viewer.screenshot.settings.changed"),
            VIEWER_MODES,
            ScreenshotSettings
        ),
        method!(
            "memory.tiles.get",
            "Inspect tile workers, cache occupancy, target level, and prefetch policy.",
            "memory.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Empty
        ),
        method!(
            "memory.tiles.set",
            "Set tile worker count, prefetch policy, and pinned-level fallback.",
            "memory.write",
            true,
            false,
            Some("memory.tiles.changed"),
            SINGLE_MODE,
            TileLoading
        ),
        method!(
            "memory.get",
            "Inspect system RAM, selected channel estimates, and pinned-level lifecycle.",
            "memory.read",
            false,
            false,
            None,
            VIEWER_MODES,
            Empty
        ),
        method!(
            "memory.pin",
            "Load selected channels from one pyramid level into CPU RAM.",
            "memory.write",
            true,
            true,
            Some("memory.changed"),
            VIEWER_MODES,
            MemoryPin
        ),
        method!(
            "memory.unpin",
            "Unload one pinned pyramid level from CPU RAM.",
            "memory.write",
            true,
            false,
            Some("memory.changed"),
            VIEWER_MODES,
            MemoryUnpin
        ),
        method!(
            "memory.unpin_all",
            "Unload all pinned pyramid levels from CPU RAM.",
            "memory.write",
            true,
            false,
            Some("memory.changed"),
            VIEWER_MODES,
            Empty
        ),
        method!(
            "viewer.thresholds.preview.get",
            "Get threshold-preview configuration, source extent, and bounded summary statistics.",
            "viewer.thresholds.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Empty
        ),
        method!(
            "viewer.thresholds.preview.configure",
            "Configure threshold scope, level, channel, value, and component filtering.",
            "viewer.thresholds.write",
            true,
            false,
            Some("viewer.thresholds.preview.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.thresholds.preview.start",
            "Read the selected channel region and start an interactive threshold preview.",
            "viewer.thresholds.write",
            true,
            true,
            Some("viewer.thresholds.preview.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.thresholds.preview.refresh",
            "Reload source pixels for the active threshold preview.",
            "viewer.thresholds.write",
            true,
            true,
            Some("viewer.thresholds.preview.changed"),
            SINGLE_MODE,
            Empty
        ),
        method!(
            "viewer.thresholds.preview.apply",
            "Filter components, polygonize the preview, and create an editable mask layer.",
            "viewer.thresholds.write",
            true,
            true,
            Some("viewer.masks.changed"),
            SINGLE_MODE,
            Empty
        ),
        method!(
            "viewer.thresholds.preview.cancel",
            "Cancel and clear the active threshold preview.",
            "viewer.thresholds.write",
            true,
            false,
            Some("viewer.thresholds.preview.changed"),
            SINGLE_MODE,
            Empty
        ),
        method!(
            "viewer.analysis.get",
            "Get persisted calls, named selections, channel mappings, and analysis readiness.",
            "viewer.analysis.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.analysis.set",
            "Atomically replace calls, named selections, mappings, and live-analysis options.",
            "viewer.analysis.write",
            true,
            false,
            Some("viewer.analysis.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.analysis.histogram",
            "Compute a bounded histogram for a numeric property over the active filtered set.",
            "viewer.analysis.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.analysis.suggest_thresholds",
            "Suggest quantile or one-dimensional K-means thresholds for a numeric property.",
            "viewer.analysis.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.analysis.warmup.get",
            "Inspect project-linked property-analysis cache warmup progress.",
            "viewer.analysis.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.analysis.warmup.start",
            "Start project-linked property-analysis cache warmup.",
            "viewer.analysis.write",
            true,
            true,
            Some("viewer.analysis.warmup.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.analysis.presets.import",
            "Import a call preset JSON file.",
            "viewer.analysis.write",
            true,
            false,
            Some("viewer.analysis.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.analysis.presets.export",
            "Export calls as a reusable preset JSON file.",
            "viewer.analysis.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.measurements.get",
            "Inspect polygon intensity measurement configuration and progress.",
            "viewer.measurements.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.measurements.configure",
            "Configure metric, image level, filtered scope, concurrency, and output prefix.",
            "viewer.measurements.write",
            true,
            false,
            Some("viewer.measurements.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.measurements.start",
            "Start background mean or exact-median polygon intensity measurement.",
            "viewer.measurements.write",
            true,
            true,
            Some("viewer.measurements.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.measurements.cancel",
            "Cooperatively cancel the active polygon intensity measurement.",
            "viewer.measurements.write",
            true,
            false,
            Some("viewer.measurements.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.measurements.properties.list",
            "List numeric properties generated by the configured measurement prefix.",
            "viewer.measurements.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Object
        ),
        method!(
            "exports.objects.columns",
            "List source, geometry, measurement, call, and named-selection export columns.",
            "exports.objects.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Object
        ),
        method!(
            "exports.objects.get_state",
            "Inspect enriched object export progress and status.",
            "exports.objects.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Object
        ),
        method!(
            "exports.objects.start",
            "Export all, filtered, or selected objects to enriched CSV or GeoParquet.",
            "exports.objects.write",
            true,
            true,
            Some("exports.objects.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "exports.objects.export_csv",
            "Export all, filtered, or selected objects and derived columns to CSV.",
            "exports.objects.write",
            true,
            true,
            Some("exports.objects.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "exports.objects.export_geoparquet",
            "Export all, filtered, or selected objects with WKB geometry and GeoParquet metadata.",
            "exports.objects.write",
            true,
            true,
            Some("exports.objects.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.channels.intensity_stats",
            "Get channel intensity statistics.",
            "viewer.analysis.read",
            false,
            true,
            None,
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.channels.set_order",
            "Set channel ordering.",
            "viewer.channels.write",
            true,
            false,
            Some("viewer.channels.changed"),
            VIEWER_MODES,
            Object
        ),
        method!(
            "viewer.channels.presentation.get",
            "Inspect channel-list search, sort, and effective ordering.",
            "viewer.channels.read",
            false,
            false,
            None,
            VIEWER_MODES,
            Empty
        ),
        method!(
            "viewer.channels.presentation.set",
            "Set channel-list search and sort presentation state.",
            "viewer.channels.write",
            true,
            false,
            Some("viewer.channels.changed"),
            VIEWER_MODES,
            ChannelPresentation
        ),
        method!(
            "viewer.channels.list_groups",
            "List channel groups.",
            "viewer.channels.read",
            false,
            false,
            None,
            READY_MODES,
            Empty
        ),
        method!(
            "viewer.channels.set_group",
            "Set channel grouping.",
            "viewer.channels.write",
            true,
            false,
            Some("viewer.channels.changed"),
            VIEWER_MODES,
            Object
        ),
        method!(
            "viewer.camera.get",
            "Get camera state.",
            "viewer.read",
            false,
            false,
            None,
            VIEWER_MODES,
            Empty
        ),
        method!(
            "viewer.camera.set",
            "Set camera state.",
            "viewer.write",
            true,
            false,
            Some("viewer.camera.changed"),
            VIEWER_MODES,
            SetCamera
        ),
        method!(
            "viewer.camera.zoom_in",
            "Zoom in.",
            "viewer.write",
            true,
            false,
            Some("viewer.camera.changed"),
            VIEWER_MODES,
            Object
        ),
        method!(
            "viewer.camera.zoom_out",
            "Zoom out.",
            "viewer.write",
            true,
            false,
            Some("viewer.camera.changed"),
            VIEWER_MODES,
            Object
        ),
        method!(
            "viewer.camera.fit",
            "Fit content to the viewport.",
            "viewer.write",
            true,
            false,
            Some("viewer.camera.changed"),
            VIEWER_MODES,
            Empty
        ),
        method!(
            "viewer.workspace.get",
            "Get the current viewer workspace, layout, links, and viewport snapshots.",
            "viewer.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Empty
        ),
        method!(
            "viewer.workspace.layout.set",
            "Set the current single or two-viewport layout.",
            "viewer.write",
            true,
            false,
            Some("viewer.workspace.layout.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.workspace.layout.get",
            "Get the current viewport workspace layout and ordered viewport IDs.",
            "viewer.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Empty
        ),
        method!(
            "viewer.workspace.swap",
            "Swap the two viewport positions in the current layout.",
            "viewer.write",
            true,
            false,
            Some("viewer.workspace.layout.changed"),
            SINGLE_MODE,
            Empty
        ),
        method!(
            "viewer.viewports.list",
            "List native viewports and their navigation and presentation snapshots.",
            "viewer.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Empty
        ),
        method!(
            "viewer.viewports.get",
            "Get one viewport by stable ID.",
            "viewer.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.viewports.create",
            "Clone a viewport into a horizontal or vertical comparison layout.",
            "viewer.write",
            true,
            false,
            Some("viewer.viewports.created"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.viewports.clone",
            "Clone an explicit viewport into a horizontal or vertical comparison layout.",
            "viewer.write",
            true,
            false,
            Some("viewer.viewports.created"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.viewports.rename",
            "Rename a viewport.",
            "viewer.write",
            true,
            false,
            Some("viewer.viewports.presentation.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.viewports.remove",
            "Remove a viewport while preserving the final remaining view.",
            "viewer.write",
            true,
            false,
            Some("viewer.viewports.removed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.viewports.set_active",
            "Set the active viewport used by native panels and legacy viewer methods.",
            "viewer.write",
            true,
            false,
            Some("viewer.viewports.active_changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.viewport_links.set",
            "Configure camera, plane, and shared-selection links between viewports.",
            "viewer.write",
            true,
            false,
            Some("viewer.viewport_links.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.viewport_links.get",
            "Get camera, plane, and shared-selection links for the workspace.",
            "viewer.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Empty
        ),
        method!(
            "viewer.viewport_links.list",
            "List the workspace's fixed comparison link group.",
            "viewer.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Empty
        ),
        method!(
            "viewer.viewport_links.create",
            "Configure the fixed comparison link group for the two workspace viewports.",
            "viewer.write",
            true,
            false,
            Some("viewer.viewport_links.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.viewport_links.update",
            "Update fields in the fixed comparison link group.",
            "viewer.write",
            true,
            false,
            Some("viewer.viewport_links.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.viewport_links.remove",
            "Disable optional navigation links while retaining document-shared selection.",
            "viewer.write",
            true,
            false,
            Some("viewer.viewport_links.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.viewports.camera.get",
            "Get camera state for an explicit viewport.",
            "viewer.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.viewports.camera.set",
            "Set camera state for an explicit viewport and propagate configured links.",
            "viewer.write",
            true,
            false,
            Some("viewer.viewports.navigation.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.viewports.camera.fit",
            "Fit content in an explicit viewport and propagate configured camera links.",
            "viewer.write",
            true,
            false,
            Some("viewer.viewports.navigation.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.viewports.planes.get",
            "Get plane state for an explicit viewport.",
            "viewer.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.viewports.planes.set",
            "Set plane state for an explicit viewport and propagate configured links.",
            "viewer.write",
            true,
            false,
            Some("viewer.viewports.navigation.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.viewports.channels.get",
            "Get channel presentation for an explicit viewport.",
            "viewer.channels.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.viewports.channels.set_visible",
            "Set visible channels for an explicit viewport.",
            "viewer.channels.write",
            true,
            false,
            Some("viewer.viewports.presentation.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.viewports.channels.set",
            "Set the visible channel collection for an explicit viewport.",
            "viewer.channels.write",
            true,
            false,
            Some("viewer.viewports.presentation.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.viewports.channels.set_active",
            "Set the active channel in an explicit viewport.",
            "viewer.channels.write",
            true,
            false,
            Some("viewer.viewports.presentation.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.viewports.channels.set_color",
            "Set channel color in an explicit viewport.",
            "viewer.channels.write",
            true,
            false,
            Some("viewer.viewports.presentation.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.viewports.channels.set_contrast",
            "Set channel contrast in an explicit viewport.",
            "viewer.channels.write",
            true,
            false,
            Some("viewer.viewports.presentation.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.viewports.channels.set_order",
            "Set channel order in an explicit viewport.",
            "viewer.channels.write",
            true,
            false,
            Some("viewer.viewports.presentation.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.viewports.channels.list_groups",
            "List channel-group presentation for an explicit viewport.",
            "viewer.channels.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.viewports.channels.set_group",
            "Set channel-group membership and color presentation in an explicit viewport.",
            "viewer.channels.write",
            true,
            false,
            Some("viewer.viewports.presentation.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.viewports.objects.style.get",
            "Get object presentation for an explicit viewport.",
            "viewer.objects.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.viewports.rendering.get",
            "Get sampling, scale-bar, HUD, and tile-debug preferences for an explicit viewport.",
            "viewer.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.viewports.rendering.set",
            "Set sampling, scale-bar, HUD, and tile-debug preferences for an explicit viewport.",
            "viewer.write",
            true,
            false,
            Some("viewer.viewports.presentation.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.viewports.objects.style.set",
            "Set independent object presentation for an explicit viewport.",
            "viewer.objects.write",
            true,
            false,
            Some("viewer.viewports.presentation.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.viewports.objects.legend.set",
            "Set independent object-property palette entries for an explicit viewport.",
            "viewer.objects.write",
            true,
            false,
            Some("viewer.viewports.presentation.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.viewports.objects.filter.get",
            "Get the independent segmentation-object filter for an explicit viewport.",
            "viewer.objects.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.viewports.objects.filter.set",
            "Set an independent segmentation-object filter for an explicit viewport.",
            "viewer.objects.write",
            true,
            false,
            Some("viewer.viewports.presentation.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.viewports.objects.filter.clear",
            "Clear the segmentation-object filter for an explicit viewport.",
            "viewer.objects.write",
            true,
            false,
            Some("viewer.viewports.presentation.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.viewports.layers.list",
            "List channels and overlays with presentation state for an explicit viewport.",
            "viewer.layers.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.viewports.layers.get",
            "Get one native layer and its complete presentation for an explicit viewport.",
            "viewer.layers.read",
            false,
            false,
            None,
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.viewports.layers.set",
            "Set one native layer's independent presentation in an explicit viewport.",
            "viewer.layers.write",
            true,
            false,
            Some("viewer.viewports.presentation.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.viewports.layers.set_visibility",
            "Set native-layer visibility in an explicit viewport.",
            "viewer.layers.write",
            true,
            false,
            Some("viewer.viewports.presentation.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.viewports.layers.set_order",
            "Set native-layer order in an explicit viewport.",
            "viewer.layers.write",
            true,
            false,
            Some("viewer.viewports.presentation.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.viewports.layers.set_active",
            "Set the active native layer in an explicit viewport.",
            "viewer.layers.write",
            true,
            false,
            Some("viewer.viewports.presentation.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.viewports.layers.state.replace",
            "Atomically replace actor-owned native-layer presentation for one viewport.",
            "viewer.layers.write",
            true,
            false,
            Some("viewer.viewports.presentation.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "viewer.ui.set_right_tab",
            "Set the single-view right tab.",
            "viewer.write",
            true,
            false,
            Some("viewer.ui.changed"),
            SINGLE_MODE,
            Object
        ),
        method!(
            "mosaic.ui.set_right_tab",
            "Set the mosaic right tab.",
            "viewer.write",
            true,
            false,
            Some("mosaic.ui.changed"),
            MOSAIC_MODE,
            Object
        ),
        method!(
            "mosaic.layout.configure",
            "Configure mosaic layout.",
            "mosaic.write",
            true,
            false,
            Some("mosaic.layout.changed"),
            MOSAIC_MODE,
            MosaicLayout
        ),
        method!(
            "mosaic.get_state",
            "Get complete mosaic layout, ROI, and focus state.",
            "viewer.read",
            false,
            false,
            None,
            MOSAIC_MODE,
            Empty
        ),
        method!(
            "mosaic.items.list",
            "List positioned mosaic items with stable ordering and pagination.",
            "viewer.read",
            false,
            false,
            None,
            MOSAIC_MODE,
            MosaicItems
        ),
        method!(
            "mosaic.selection.get",
            "Get selected mosaic ROIs.",
            "viewer.read",
            false,
            false,
            None,
            MOSAIC_MODE,
            Empty
        ),
        method!(
            "mosaic.selection.set",
            "Select mosaic ROIs using stable IDs and replace, add, remove, toggle, all, or range semantics.",
            "mosaic.write",
            true,
            false,
            Some("mosaic.selection.changed"),
            MOSAIC_MODE,
            MosaicSelect
        ),
        method!(
            "mosaic.selection.clear",
            "Clear the mosaic ROI selection.",
            "mosaic.write",
            true,
            false,
            Some("mosaic.selection.changed"),
            MOSAIC_MODE,
            Empty
        ),
        method!(
            "mosaic.focus.get",
            "Get the focused mosaic ROI.",
            "viewer.read",
            false,
            false,
            None,
            MOSAIC_MODE,
            Empty
        ),
        method!(
            "mosaic.focus.set",
            "Focus a mosaic ROI by stable ROI ID or index.",
            "mosaic.write",
            true,
            false,
            Some("mosaic.focus.changed"),
            MOSAIC_MODE,
            MosaicFocus
        ),
        method!(
            "mosaic.focus.next",
            "Focus the next mosaic ROI.",
            "mosaic.write",
            true,
            false,
            Some("mosaic.focus.changed"),
            MOSAIC_MODE,
            StepPlane
        ),
        method!(
            "mosaic.focus.previous",
            "Focus the previous mosaic ROI.",
            "mosaic.write",
            true,
            false,
            Some("mosaic.focus.changed"),
            MOSAIC_MODE,
            StepPlane
        ),
        method!(
            "mosaic.focus.fit",
            "Fit the focused mosaic ROI to the viewport.",
            "mosaic.write",
            true,
            false,
            Some("viewer.camera.changed"),
            MOSAIC_MODE,
            Empty
        ),
        method!(
            "mosaic.focus.clear",
            "Clear focused mosaic ROI without changing selection.",
            "mosaic.write",
            true,
            false,
            Some("mosaic.focus.changed"),
            MOSAIC_MODE,
            Empty
        ),
        method!(
            "mosaic.fit_all",
            "Fit all mosaic items to the viewport.",
            "mosaic.write",
            true,
            false,
            Some("viewer.camera.changed"),
            MOSAIC_MODE,
            Empty
        ),
        method!(
            "mosaic.objects.get_state",
            "Get per-ROI mosaic object-source, loading, and allocation state.",
            "viewer.objects.read",
            false,
            false,
            None,
            MOSAIC_MODE,
            Empty
        ),
        method!(
            "mosaic.objects.load_selected",
            "Load object segmentations for the selected mosaic ROIs and settle when all requested reads finish.",
            "viewer.objects.write",
            true,
            true,
            Some("mosaic.objects.changed"),
            MOSAIC_MODE,
            Empty
        ),
        method!(
            "mosaic.objects.cancel_load",
            "Cancel remaining scheduled object loads while allowing an in-flight disk read to finish.",
            "viewer.objects.write",
            true,
            false,
            Some("mosaic.objects.changed"),
            MOSAIC_MODE,
            Empty
        ),
        method!(
            "app.navigation.show_project",
            "Show the project page.",
            "application.write",
            true,
            false,
            Some("application.mode.changed"),
            READY_MODES,
            Empty
        ),
        method!(
            "viewer.screenshot.capture",
            "Capture the viewer canvas.",
            "viewer.screenshot",
            true,
            true,
            Some("viewer.screenshot.completed"),
            VIEWER_MODES,
            CaptureScreenshot
        ),
        method!(
            "viewer.workspace.screenshot.capture",
            "Capture the composed multi-viewport canvas workspace.",
            "viewer.screenshot",
            true,
            true,
            Some("viewer.screenshot.completed"),
            SINGLE_MODE,
            CaptureScreenshot
        ),
        method!(
            "app.screenshot.capture",
            "Capture the Odon window.",
            "application.screenshot",
            true,
            true,
            Some("viewer.screenshot.completed"),
            READY_MODES,
            Object
        ),
        method!(
            "project.screenshot.capture",
            "Capture the project page.",
            "application.screenshot",
            true,
            true,
            Some("viewer.screenshot.completed"),
            READY_MODES,
            Object
        ),
    ]
});
