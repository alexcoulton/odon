use std::collections::BTreeSet;

use serde::Serialize;
use serde_json::{Value, json};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Stability {
    Experimental,
    Provisional,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestShape {
    Empty,
    SetSidePanels,
    SetSmoothPixels,
    SetVisibleChannels,
    SetCamera,
    CaptureScreenshot,
    Object,
}

#[derive(Debug, Clone, Copy)]
pub struct MethodDescriptor {
    pub name: &'static str,
    pub summary: &'static str,
    pub capability: &'static str,
    pub mutates: bool,
    pub starts_task: bool,
    pub mcp_exposed: bool,
    pub stability: Stability,
    pub request_shape: RequestShape,
}

macro_rules! method {
    ($name:literal, $summary:literal, $capability:literal, $mutates:expr, $shape:ident) => {
        MethodDescriptor {
            name: $name,
            summary: $summary,
            capability: $capability,
            mutates: $mutates,
            starts_task: false,
            mcp_exposed: true,
            stability: Stability::Provisional,
            request_shape: RequestShape::$shape,
        }
    };
}

pub static METHODS: &[MethodDescriptor] = &[
    method!(
        "get_current_view",
        "Get current application and viewer state.",
        "viewer.read",
        false,
        Empty
    ),
    method!(
        "list_project_rois",
        "List project ROIs.",
        "project.read",
        false,
        Empty
    ),
    method!(
        "list_channels",
        "List channels.",
        "viewer.channels.read",
        false,
        Empty
    ),
    method!(
        "list_visible_channels",
        "List visible channels.",
        "viewer.channels.read",
        false,
        Empty
    ),
    method!(
        "get_side_panels",
        "Get side-panel visibility.",
        "viewer.read",
        false,
        Empty
    ),
    method!(
        "set_side_panels",
        "Set side-panel visibility.",
        "viewer.write",
        true,
        SetSidePanels
    ),
    method!(
        "get_smooth_pixels",
        "Get image interpolation state.",
        "viewer.read",
        false,
        Empty
    ),
    method!(
        "set_smooth_pixels",
        "Set image interpolation state.",
        "viewer.write",
        true,
        SetSmoothPixels
    ),
    method!(
        "get_loading_state",
        "Get loading diagnostics.",
        "viewer.read",
        false,
        Empty
    ),
    method!(
        "get_active_channel",
        "Get the active channel.",
        "viewer.channels.read",
        false,
        Object
    ),
    method!(
        "set_active_channel",
        "Set the active channel.",
        "viewer.channels.write",
        true,
        Object
    ),
    method!(
        "set_visible_channels",
        "Set channel visibility.",
        "viewer.channels.write",
        true,
        SetVisibleChannels
    ),
    method!(
        "open_project",
        "Open a project.",
        "project.write",
        true,
        Object
    ),
    method!(
        "open_ome_zarr",
        "Open an OME-Zarr dataset.",
        "application.open",
        true,
        Object
    ),
    method!(
        "open_tiff",
        "Open a TIFF dataset.",
        "application.open",
        true,
        Object
    ),
    method!(
        "open_mosaic_samplesheet",
        "Open a mosaic samplesheet.",
        "application.open",
        true,
        Object
    ),
    method!(
        "open_roi",
        "Open a project ROI.",
        "project.write",
        true,
        Object
    ),
    method!(
        "save_project",
        "Save the active project.",
        "project.write",
        true,
        Empty
    ),
    method!(
        "get_channel_contrast",
        "Get channel contrast.",
        "viewer.channels.read",
        false,
        Object
    ),
    method!(
        "set_channel_contrast",
        "Set channel contrast.",
        "viewer.channels.write",
        true,
        Object
    ),
    method!(
        "get_object_overlay_visibility",
        "Get object overlay visibility.",
        "viewer.layers.read",
        false,
        Object
    ),
    method!(
        "set_object_overlay_visibility",
        "Set object overlay visibility.",
        "viewer.layers.write",
        true,
        Object
    ),
    method!(
        "get_object_selection",
        "Get selected objects.",
        "viewer.objects.read",
        false,
        Object
    ),
    method!(
        "query_object_ids_in_rect",
        "Query objects in a rectangle.",
        "viewer.objects.read",
        false,
        Object
    ),
    method!(
        "query_object_ids_in_view",
        "Query objects in the viewport.",
        "viewer.objects.read",
        false,
        Object
    ),
    method!(
        "select_object_ids_in_rect",
        "Select objects in a rectangle.",
        "viewer.objects.write",
        true,
        Object
    ),
    method!(
        "clear_object_selection",
        "Clear object selection.",
        "viewer.objects.write",
        true,
        Object
    ),
    method!(
        "get_object_filter",
        "Get object filter state.",
        "viewer.objects.read",
        false,
        Object
    ),
    method!(
        "set_object_filter_query",
        "Set an object filter query.",
        "viewer.objects.write",
        true,
        Object
    ),
    method!(
        "clear_object_filter",
        "Clear object filtering.",
        "viewer.objects.write",
        true,
        Object
    ),
    method!(
        "get_channel_intensity_stats",
        "Get channel intensity statistics.",
        "viewer.analysis.read",
        false,
        Object
    ),
    method!(
        "set_channel_order",
        "Set channel ordering.",
        "viewer.channels.write",
        true,
        Object
    ),
    method!(
        "list_channel_groups",
        "List channel groups.",
        "viewer.channels.read",
        false,
        Empty
    ),
    method!(
        "set_channel_group",
        "Set channel grouping.",
        "viewer.channels.write",
        true,
        Object
    ),
    method!(
        "get_camera",
        "Get camera state.",
        "viewer.read",
        false,
        Empty
    ),
    method!(
        "set_camera",
        "Set camera state.",
        "viewer.write",
        true,
        SetCamera
    ),
    method!("zoom_in", "Zoom in.", "viewer.write", true, Object),
    method!("zoom_out", "Zoom out.", "viewer.write", true, Object),
    method!(
        "fit_to_view",
        "Fit content to the viewport.",
        "viewer.write",
        true,
        Empty
    ),
    method!(
        "set_right_tab",
        "Set the single-view right tab.",
        "viewer.write",
        true,
        Object
    ),
    method!(
        "set_mosaic_right_tab",
        "Set the mosaic right tab.",
        "viewer.write",
        true,
        Object
    ),
    method!(
        "configure_mosaic_layout",
        "Configure mosaic layout.",
        "mosaic.write",
        true,
        Object
    ),
    method!(
        "show_project_page",
        "Show the project page.",
        "application.write",
        true,
        Empty
    ),
    method!(
        "capture_screenshot",
        "Capture the viewer canvas.",
        "viewer.screenshot",
        true,
        CaptureScreenshot
    ),
    method!(
        "capture_window_screenshot",
        "Capture the Odon window.",
        "application.screenshot",
        true,
        Object
    ),
    method!(
        "capture_project_screenshot",
        "Capture the project page.",
        "application.screenshot",
        true,
        Object
    ),
];

pub static PROTOCOL_METHODS: &[(&str, &str, &str, bool, bool)] = &[
    (
        "system.hello",
        "Authenticate and negotiate a control protocol version.",
        "system.connect",
        false,
        false,
    ),
    (
        "system.get_capabilities",
        "List capabilities granted by this Odon build.",
        "system.introspect",
        false,
        false,
    ),
    (
        "system.list_methods",
        "List control methods and request schemas.",
        "system.introspect",
        false,
        false,
    ),
    (
        "system.describe_methods",
        "List control methods and request schemas.",
        "system.introspect",
        false,
        false,
    ),
    (
        "system.describe_events",
        "Describe initial event families and envelope fields.",
        "system.introspect",
        false,
        false,
    ),
    (
        "ui.describe_schema",
        "Describe declarative UI schema v1 limits and vocabulary.",
        "ui.read",
        false,
        false,
    ),
    (
        "system.batch",
        "Execute application commands in order.",
        "system.batch",
        true,
        false,
    ),
    (
        "system.get_diagnostics",
        "Inspect bounded control-server state.",
        "system.introspect",
        false,
        false,
    ),
    (
        "events.subscribe",
        "Subscribe to server-pushed event patterns.",
        "events.read",
        false,
        false,
    ),
    (
        "events.unsubscribe",
        "Remove event subscriptions.",
        "events.read",
        false,
        false,
    ),
    (
        "events.get_status",
        "Inspect event queue diagnostics.",
        "events.read",
        false,
        false,
    ),
    (
        "tasks.start",
        "Run an application command as a retained task.",
        "tasks.write",
        true,
        true,
    ),
    (
        "tasks.get",
        "Get a task snapshot.",
        "tasks.read",
        false,
        false,
    ),
    (
        "tasks.list",
        "List retained tasks.",
        "tasks.read",
        false,
        false,
    ),
    (
        "tasks.cancel",
        "Cancel a queued task.",
        "tasks.write",
        true,
        false,
    ),
    (
        "tasks.forget",
        "Forget a terminal task.",
        "tasks.write",
        true,
        false,
    ),
    (
        "data.resources.register",
        "Register referenced external data.",
        "data.write",
        true,
        false,
    ),
    (
        "data.resources.list",
        "List registered external data.",
        "data.read",
        false,
        false,
    ),
    (
        "data.resources.get",
        "Get an external data descriptor.",
        "data.read",
        false,
        false,
    ),
    (
        "data.resources.remove",
        "Remove an unreferenced data descriptor.",
        "data.write",
        true,
        false,
    ),
    (
        "viewer.layers.add",
        "Add a referenced viewer layer.",
        "viewer.layers.write",
        true,
        false,
    ),
    (
        "viewer.layers.list",
        "List referenced viewer layers.",
        "viewer.layers.read",
        false,
        false,
    ),
    (
        "viewer.layers.get",
        "Get a referenced viewer layer.",
        "viewer.layers.read",
        false,
        false,
    ),
    (
        "viewer.layers.update",
        "Update layer state or style.",
        "viewer.layers.write",
        true,
        false,
    ),
    (
        "viewer.layers.remove",
        "Remove a referenced layer.",
        "viewer.layers.write",
        true,
        false,
    ),
    (
        "viewer.layers.reorder",
        "Set referenced layer order.",
        "viewer.layers.write",
        true,
        false,
    ),
    (
        "ui.extensions.register",
        "Register a declarative UI extension.",
        "ui.panels",
        true,
        false,
    ),
    (
        "ui.extensions.list",
        "List UI extensions.",
        "ui.read",
        false,
        false,
    ),
    (
        "ui.extensions.remove",
        "Remove an owned UI extension.",
        "ui.panels",
        true,
        false,
    ),
    (
        "ui.contributions.register",
        "Register a validated component tree.",
        "ui.panels",
        true,
        false,
    ),
    (
        "ui.contributions.list",
        "List component trees.",
        "ui.read",
        false,
        false,
    ),
    (
        "ui.contributions.patch_values",
        "Atomically patch retained UI values.",
        "ui.panels",
        true,
        false,
    ),
    (
        "ui.contributions.remove",
        "Remove an owned component tree.",
        "ui.panels",
        true,
        false,
    ),
];

pub static METHOD_ALIASES: &[(&str, &str)] = &[
    ("app.get_state", "get_current_view"),
    ("app.get_loading_state", "get_loading_state"),
    ("app.show_project_page", "show_project_page"),
    ("project.get_state", "list_project_rois"),
    ("project.rois.list", "list_project_rois"),
    ("project.open", "open_project"),
    ("project.save", "save_project"),
    ("project.rois.open", "open_roi"),
    ("viewer.get_state", "get_current_view"),
    ("viewer.camera.get", "get_camera"),
    ("viewer.camera.set", "set_camera"),
    ("viewer.camera.fit", "fit_to_view"),
    ("viewer.camera.zoom_in", "zoom_in"),
    ("viewer.camera.zoom_out", "zoom_out"),
    ("viewer.channels.list", "list_channels"),
    ("viewer.channels.list_visible", "list_visible_channels"),
    ("viewer.channels.get_active", "get_active_channel"),
    ("viewer.channels.set_active", "set_active_channel"),
    ("viewer.channels.set_visible", "set_visible_channels"),
    ("viewer.channels.get_contrast", "get_channel_contrast"),
    ("viewer.channels.set_contrast", "set_channel_contrast"),
    ("viewer.channels.set_order", "set_channel_order"),
    ("viewer.channels.list_groups", "list_channel_groups"),
    ("viewer.channels.set_group", "set_channel_group"),
    ("viewer.objects.get_selection", "get_object_selection"),
    ("viewer.objects.clear_selection", "clear_object_selection"),
    ("viewer.objects.get_filter", "get_object_filter"),
    ("viewer.objects.set_filter", "set_object_filter_query"),
    ("viewer.objects.clear_filter", "clear_object_filter"),
    ("viewer.screenshot.capture", "capture_screenshot"),
    ("app.screenshot.capture", "capture_window_screenshot"),
    ("project.screenshot.capture", "capture_project_screenshot"),
];

pub fn method(name: &str) -> Option<&'static MethodDescriptor> {
    let canonical = canonical_method(name);
    METHODS
        .iter()
        .find(|descriptor| descriptor.name == canonical)
}

pub fn canonical_method(name: &str) -> &str {
    METHOD_ALIASES
        .iter()
        .find_map(|(alias, canonical)| (*alias == name).then_some(*canonical))
        .unwrap_or(name)
}

pub fn capabilities() -> Vec<String> {
    METHODS
        .iter()
        .map(|descriptor| descriptor.capability)
        .chain(PROTOCOL_METHODS.iter().map(|method| method.2))
        .chain(["system.introspect"])
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(str::to_string)
        .collect()
}

pub fn catalog_json() -> Value {
    Value::Array(
        METHODS
            .iter()
            .map(|descriptor| {
                json!({
                    "name": descriptor.name,
                    "summary": descriptor.summary,
                    "capability": descriptor.capability,
                    "mutates": descriptor.mutates,
                    "starts_task": descriptor.starts_task,
                    "mcp_exposed": descriptor.mcp_exposed,
                    "stability": descriptor.stability,
                    "request_schema": request_schema_for(descriptor),
                })
            })
            .chain(PROTOCOL_METHODS.iter().map(
                |(name, summary, capability, mutates, starts_task)| {
                    json!({
                        "name": name,
                        "summary": summary,
                        "capability": capability,
                        "mutates": mutates,
                        "starts_task": starts_task,
                        "mcp_exposed": false,
                        "stability": Stability::Experimental,
                        "request_schema": {"type": "object"},
                    })
                },
            ))
            .chain(METHOD_ALIASES.iter().filter_map(|(alias, canonical)| {
                method(canonical).map(|descriptor| {
                    json!({
                        "name": alias,
                        "canonical_name": canonical,
                        "summary": descriptor.summary,
                        "capability": descriptor.capability,
                        "mutates": descriptor.mutates,
                        "starts_task": descriptor.starts_task,
                        "mcp_exposed": false,
                        "stability": Stability::Experimental,
                        "request_schema": request_schema_for(descriptor),
                    })
                })
            }))
            .collect(),
    )
}

fn request_schema(shape: RequestShape) -> Value {
    match shape {
        RequestShape::Empty => json!({
            "type": "object",
            "properties": {},
            "additionalProperties": false,
        }),
        RequestShape::SetSidePanels => json!({
            "type": "object",
            "properties": {"left": {"type": "boolean"}, "right": {"type": "boolean"}},
            "additionalProperties": false,
        }),
        RequestShape::SetSmoothPixels => json!({
            "type": "object",
            "properties": {"smooth": {"type": "boolean"}},
            "required": ["smooth"],
            "additionalProperties": false,
        }),
        RequestShape::SetVisibleChannels => json!({
            "type": "object",
            "properties": {
                "channels": {"type": "array", "items": {"anyOf": [{"type": "string"}, {"type": "integer", "minimum": 0}]}},
                "mode": {"type": "string", "enum": ["only", "show", "hide"]}
            },
            "required": ["channels"],
            "additionalProperties": false,
        }),
        RequestShape::SetCamera => json!({
            "type": "object",
            "properties": {
                "center_world_lvl0": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                "center_x": {"type": "number"}, "center_y": {"type": "number"},
                "zoom": {"type": "number", "exclusiveMinimum": 0},
                "zoom_screen_per_lvl0_px": {"type": "number", "exclusiveMinimum": 0}
            },
            "additionalProperties": false,
        }),
        RequestShape::CaptureScreenshot => json!({
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "additionalProperties": false,
        }),
        RequestShape::Object => json!({"type": "object"}),
    }
}

fn request_schema_for(descriptor: &MethodDescriptor) -> Value {
    let mut schema = request_schema(descriptor.request_shape);
    if descriptor.mutates
        && let Some(properties) = schema.get_mut("properties").and_then(Value::as_object_mut)
    {
        properties.insert(
            "if_revision".to_string(),
            json!({"type": "integer", "minimum": 0}),
        );
    }
    schema
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_names_are_unique_and_capabilities_are_sorted() {
        let names = METHODS
            .iter()
            .map(|descriptor| descriptor.name)
            .chain(PROTOCOL_METHODS.iter().map(|method| method.0))
            .chain(METHOD_ALIASES.iter().map(|method| method.0))
            .collect::<BTreeSet<_>>();
        assert_eq!(
            names.len(),
            METHODS.len() + PROTOCOL_METHODS.len() + METHOD_ALIASES.len()
        );

        let capabilities = capabilities();
        assert!(capabilities.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(capabilities.contains(&"system.introspect".to_string()));
    }

    #[test]
    fn every_method_has_introspection_metadata() {
        for descriptor in METHODS {
            assert!(!descriptor.name.is_empty());
            assert!(!descriptor.summary.is_empty(), "{}", descriptor.name);
            assert!(!descriptor.capability.is_empty(), "{}", descriptor.name);
            assert!(request_schema(descriptor.request_shape).is_object());
        }
    }
}
