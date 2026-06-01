use serde_json::{Value, json};

const SERVER_NAME: &str = "odon-mcp";
const SERVER_VERSION: &str = env!("CARGO_PKG_VERSION");

pub fn handle_json_rpc_line(line: &str) -> Option<Value> {
    let request = match serde_json::from_str::<Value>(line) {
        Ok(request) => request,
        Err(err) => {
            return Some(json_rpc_error(
                Value::Null,
                -32700,
                format!("Parse error: {err}"),
            ));
        }
    };
    let id = request.get("id").cloned().unwrap_or(Value::Null);
    let Some(method) = request.get("method").and_then(Value::as_str) else {
        return Some(json_rpc_error(id, -32600, "Invalid request"));
    };

    match method {
        "initialize" => Some(json_rpc_result(
            id,
            json!({
                "protocolVersion": request
                    .get("params")
                    .and_then(|params| params.get("protocolVersion"))
                    .cloned()
                    .unwrap_or_else(|| json!("2025-06-18")),
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": {
                    "name": SERVER_NAME,
                    "version": SERVER_VERSION
                }
            }),
        )),
        "notifications/initialized" => None,
        "ping" => Some(json_rpc_result(id, json!({}))),
        "tools/list" => Some(json_rpc_result(id, tools_list())),
        "tools/call" => Some(handle_tool_call(
            id,
            request.get("params").cloned().unwrap_or(Value::Null),
        )),
        _ => Some(json_rpc_error(
            id,
            -32601,
            format!("Method not found: {method}"),
        )),
    }
}

fn tools_list() -> Value {
    json!({
        "tools": [
            tool_schema(
                "get_current_view",
                "Return the current Odon mode, project, current viewer state, and visible channels."
            ),
            tool_schema(
                "list_project_rois",
                "List ROIs from the currently loaded Odon project."
            ),
            tool_schema(
                "list_channels",
                "List channels from the active single-image or mosaic viewer."
            ),
            tool_schema(
                "list_visible_channels",
                "List visible channels from the active single-image or mosaic viewer."
            ),
            tool_schema(
                "get_side_panels",
                "Return whether the left and right side panels are visible."
            ),
            set_side_panels_tool_schema(),
            tool_schema(
                "get_smooth_pixels",
                "Return whether smooth pixel interpolation is enabled."
            ),
            set_smooth_pixels_tool_schema(),
            tool_schema(
                "get_loading_state",
                "Return diagnostic state explaining why the active viewer is showing loading or busy indicators."
            ),
            channel_selector_tool_schema(
                "get_active_channel",
                "Return the active channel from the active single-image or mosaic viewer."
            ),
            channel_selector_tool_schema(
                "set_active_channel",
                "Set the active channel by index, exact channel name, or marker-like channel selector."
            ),
            set_visible_channels_tool_schema(),
            open_ome_zarr_tool_schema(),
            open_roi_tool_schema(),
            tool_schema(
                "save_project",
                "Save the current Odon project to its existing project JSON path."
            ),
            channel_selector_tool_schema(
                "get_channel_contrast",
                "Return contrast limits for a selected channel, or the active channel if no selector is provided."
            ),
            set_channel_contrast_tool_schema(),
            object_overlay_visibility_tool_schema(
                "get_object_overlay_visibility",
                "Return object/segmentation overlay visibility state for the active viewer."
            ),
            set_object_overlay_visibility_tool_schema(),
            object_selection_tool_schema(
                "get_object_selection",
                "Return the current selectable object layer and selected object IDs/centroids."
            ),
            object_rect_tool_schema(
                "query_object_ids_in_rect",
                "Return object IDs whose rendered geometry intersects a world or screen rectangle without changing selection."
            ),
            object_selection_tool_schema(
                "query_object_ids_in_view",
                "Return object IDs whose rendered geometry intersects the current canvas viewport without changing selection."
            ),
            object_rect_select_tool_schema(),
            object_selection_tool_schema(
                "clear_object_selection",
                "Clear selected objects on the active selectable object layer."
            ),
            channel_intensity_stats_tool_schema(),
            set_channel_order_tool_schema(),
            tool_schema(
                "list_channel_groups",
                "List project-backed channel groups and their current members."
            ),
            set_channel_group_tool_schema(),
            tool_schema(
                "get_camera",
                "Return the current camera center and zoom."
            ),
            set_camera_tool_schema(),
            zoom_tool_schema(
                "zoom_in",
                "Zoom in around the current viewport center."
            ),
            zoom_tool_schema(
                "zoom_out",
                "Zoom out around the current viewport center."
            ),
            tool_schema(
                "fit_to_view",
                "Fit the current image or mosaic to the viewport."
            ),
            capture_screenshot_tool_schema()
        ]
    })
}

fn tool_schema(name: &str, description: &str) -> Value {
    json!({
        "name": name,
        "description": description,
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": false
        }
    })
}

fn channel_selector_tool_schema(name: &str, description: &str) -> Value {
    json!({
        "name": name,
        "description": description,
        "inputSchema": {
            "type": "object",
            "properties": {
                "index": {"type": "integer", "minimum": 0},
                "name": {"type": "string"},
                "channel": {"type": "string"},
                "marker": {"type": "string"}
            },
            "additionalProperties": false
        }
    })
}

fn set_visible_channels_tool_schema() -> Value {
    json!({
        "name": "set_visible_channels",
        "description": "Set visible channels. mode='only' replaces visibility; mode='show' or 'hide' edits the current visibility set.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "channels": {
                    "type": "array",
                    "items": {
                        "anyOf": [
                            {"type": "string"},
                            {"type": "integer", "minimum": 0}
                        ]
                    }
                },
                "mode": {
                    "type": "string",
                    "enum": ["only", "show", "hide"],
                    "default": "only"
                }
            },
            "required": ["channels"],
            "additionalProperties": false
        }
    })
}

fn set_side_panels_tool_schema() -> Value {
    json!({
        "name": "set_side_panels",
        "description": "Show or hide the left and/or right side panels in the active single-image or mosaic viewer.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "left": {"type": "boolean"},
                "right": {"type": "boolean"}
            },
            "additionalProperties": false
        }
    })
}

fn set_smooth_pixels_tool_schema() -> Value {
    json!({
        "name": "set_smooth_pixels",
        "description": "Enable or disable smooth pixel interpolation in the active single-image or mosaic viewer.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "smooth": {"type": "boolean"}
            },
            "required": ["smooth"],
            "additionalProperties": false
        }
    })
}

fn open_ome_zarr_tool_schema() -> Value {
    json!({
        "name": "open_ome_zarr",
        "description": "Open a local OME-Zarr dataset in the active Odon window, replacing the current viewer.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to a local OME-Zarr directory, or to its .zattrs/zarr.json metadata file."
                }
            },
            "required": ["path"],
            "additionalProperties": false
        }
    })
}

fn open_roi_tool_schema() -> Value {
    json!({
        "name": "open_roi",
        "description": "Open a project ROI by ROI/display/path fragment, optionally disambiguated by sample.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "roi": {"type": "string"},
                "id": {"type": "string"},
                "name": {"type": "string"},
                "sample": {"type": "string"}
            },
            "additionalProperties": false
        }
    })
}

fn set_channel_contrast_tool_schema() -> Value {
    json!({
        "name": "set_channel_contrast",
        "description": "Set contrast limits for a channel by index, name, channel, or marker.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "index": {"type": "integer", "minimum": 0},
                "name": {"type": "string"},
                "channel": {"type": "string"},
                "marker": {"type": "string"},
                "min": {"type": "number"},
                "max": {"type": "number"}
            },
            "required": ["min", "max"],
            "additionalProperties": false
        }
    })
}

fn channel_intensity_stats_tool_schema() -> Value {
    json!({
        "name": "get_channel_intensity_stats",
        "description": "Compute coarse image-level intensity statistics for a selected channel in the active single-image viewer. Defaults to the coarsest pyramid level for quick QC.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "index": {"type": "integer", "minimum": 0},
                "name": {"type": "string"},
                "channel": {"type": "string"},
                "marker": {"type": "string"},
                "level": {"type": "integer", "minimum": 0}
            },
            "additionalProperties": false
        }
    })
}

fn object_overlay_visibility_tool_schema(name: &str, description: &str) -> Value {
    json!({
        "name": name,
        "description": description,
        "inputSchema": {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "enum": ["objects", "labels", "geojson", "all"],
                    "default": "objects"
                }
            },
            "additionalProperties": false
        }
    })
}

fn set_object_overlay_visibility_tool_schema() -> Value {
    json!({
        "name": "set_object_overlay_visibility",
        "description": "Show or hide object/segmentation overlays in the active viewer.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "visible": {"type": "boolean"},
                "target": {
                    "type": "string",
                    "enum": ["objects", "labels", "geojson", "all"],
                    "default": "objects"
                }
            },
            "required": ["visible"],
            "additionalProperties": false
        }
    })
}

fn object_selection_tool_schema(name: &str, description: &str) -> Value {
    json!({
        "name": name,
        "description": description,
        "inputSchema": {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "enum": ["active", "objects", "segmentation_objects", "spatial_shape"],
                    "default": "active"
                },
                "layer_id": {"type": "integer", "minimum": 0},
                "id": {"type": "integer", "minimum": 0},
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10000,
                    "default": 64
                }
            },
            "additionalProperties": false
        }
    })
}

fn object_rect_tool_schema(name: &str, description: &str) -> Value {
    json!({
        "name": name,
        "description": description,
        "inputSchema": {
            "type": "object",
            "properties": object_rect_tool_properties(false),
            "additionalProperties": false
        }
    })
}

fn object_rect_select_tool_schema() -> Value {
    json!({
        "name": "select_object_ids_in_rect",
        "description": "Select object IDs whose rendered geometry intersects a world or screen rectangle using the same logic as rectangle selection in the app.",
        "inputSchema": {
            "type": "object",
            "properties": object_rect_tool_properties(true),
            "additionalProperties": false
        }
    })
}

fn object_rect_tool_properties(include_additive: bool) -> Value {
    let mut properties = serde_json::Map::from_iter([
        (
            "target".to_string(),
            json!({
                "type": "string",
                "enum": ["active", "objects", "segmentation_objects", "spatial_shape"],
                "default": "active"
            }),
        ),
        (
            "layer_id".to_string(),
            json!({"type": "integer", "minimum": 0}),
        ),
        ("id".to_string(), json!({"type": "integer", "minimum": 0})),
        (
            "world_rect".to_string(),
            json!({
                "type": "array",
                "items": {"type": "number"},
                "minItems": 4,
                "maxItems": 4
            }),
        ),
        (
            "screen_rect".to_string(),
            json!({
                "type": "array",
                "items": {"type": "number"},
                "minItems": 4,
                "maxItems": 4
            }),
        ),
        ("min_x".to_string(), json!({"type": "number"})),
        ("min_y".to_string(), json!({"type": "number"})),
        ("max_x".to_string(), json!({"type": "number"})),
        ("max_y".to_string(), json!({"type": "number"})),
        ("x0".to_string(), json!({"type": "number"})),
        ("y0".to_string(), json!({"type": "number"})),
        ("x1".to_string(), json!({"type": "number"})),
        ("y1".to_string(), json!({"type": "number"})),
        (
            "limit".to_string(),
            json!({
                "type": "integer",
                "minimum": 1,
                "maximum": 10000,
                "default": 64
            }),
        ),
    ]);
    if include_additive {
        properties.insert(
            "additive".to_string(),
            json!({"type": "boolean", "default": false}),
        );
    }
    serde_json::Value::Object(properties)
}

fn set_channel_order_tool_schema() -> Value {
    json!({
        "name": "set_channel_order",
        "description": "Set channel list ordering. Provide channels to pin/list manually, or sort to use a built-in sort mode.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "channels": {
                    "type": "array",
                    "items": {
                        "anyOf": [
                            {"type": "string"},
                            {"type": "integer", "minimum": 0}
                        ]
                    }
                },
                "mode": {
                    "type": "string",
                    "enum": ["listed_first", "exact"],
                    "default": "listed_first"
                },
                "sort": {
                    "type": "string",
                    "enum": ["manual", "name_asc", "name_desc", "visible_first", "hidden_first"]
                }
            },
            "additionalProperties": false
        }
    })
}

fn set_channel_group_tool_schema() -> Value {
    json!({
        "name": "set_channel_group",
        "description": "Create/update a project-backed channel group and assign channels to it.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "group": {"type": "string"},
                "group_id": {"type": "integer", "minimum": 1},
                "channels": {
                    "type": "array",
                    "items": {
                        "anyOf": [
                            {"type": "string"},
                            {"type": "integer", "minimum": 0}
                        ]
                    }
                },
                "color": {"type": "string"},
                "color_rgb": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 0, "maximum": 255},
                    "minItems": 3,
                    "maxItems": 3
                },
                "inherit_color": {"type": "boolean", "default": true},
                "replace_group_members": {"type": "boolean", "default": false}
            },
            "required": ["channels"],
            "additionalProperties": false
        }
    })
}

fn set_camera_tool_schema() -> Value {
    json!({
        "name": "set_camera",
        "description": "Set camera center and/or zoom.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "center_world_lvl0": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 2,
                    "maxItems": 2
                },
                "center_x": {"type": "number"},
                "center_y": {"type": "number"},
                "zoom": {"type": "number"},
                "zoom_screen_per_lvl0_px": {"type": "number"}
            },
            "additionalProperties": false
        }
    })
}

fn zoom_tool_schema(name: &str, description: &str) -> Value {
    json!({
        "name": name,
        "description": description,
        "inputSchema": {
            "type": "object",
            "properties": {
                "factor": {
                    "type": "number",
                    "default": 1.5
                }
            },
            "additionalProperties": false
        }
    })
}

fn capture_screenshot_tool_schema() -> Value {
    json!({
        "name": "capture_screenshot",
        "description": "Queue a canvas screenshot. Provide path to save to a specific PNG; otherwise uses the configured quick screenshot folder.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"}
            },
            "additionalProperties": false
        }
    })
}

fn handle_tool_call(id: Value, params: Value) -> Value {
    let Some(name) = params.get("name").and_then(Value::as_str) else {
        return json_rpc_error(id, -32602, "tools/call requires params.name");
    };
    let arguments = params
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| json!({}));
    let method = match name {
        "get_current_view"
        | "list_project_rois"
        | "list_channels"
        | "list_visible_channels"
        | "get_side_panels"
        | "set_side_panels"
        | "get_smooth_pixels"
        | "set_smooth_pixels"
        | "get_loading_state"
        | "get_active_channel"
        | "set_active_channel"
        | "set_visible_channels"
        | "open_ome_zarr"
        | "open_roi"
        | "save_project"
        | "get_channel_contrast"
        | "set_channel_contrast"
        | "get_object_overlay_visibility"
        | "set_object_overlay_visibility"
        | "get_object_selection"
        | "query_object_ids_in_rect"
        | "query_object_ids_in_view"
        | "select_object_ids_in_rect"
        | "clear_object_selection"
        | "get_channel_intensity_stats"
        | "set_channel_order"
        | "list_channel_groups"
        | "set_channel_group"
        | "get_camera"
        | "set_camera"
        | "zoom_in"
        | "zoom_out"
        | "fit_to_view"
        | "capture_screenshot" => name,
        _ => {
            return json_rpc_error(id, -32602, format!("Unknown tool: {name}"));
        }
    };
    match crate::mcp::client::call_running_odon(method, arguments) {
        Ok(result) => json_rpc_result(
            id,
            json!({
                "content": [
                    {
                        "type": "text",
                        "text": serde_json::to_string_pretty(&result)
                            .unwrap_or_else(|_| result.to_string())
                    }
                ]
            }),
        ),
        Err(err) => json_rpc_result(
            id,
            json!({
                "isError": true,
                "content": [
                    {
                        "type": "text",
                        "text": format!("{err}")
                    }
                ]
            }),
        ),
    }
}

fn json_rpc_result(id: Value, result: Value) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": result,
    })
}

fn json_rpc_error(id: Value, code: i64, message: impl Into<String>) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": {
            "code": code,
            "message": message.into(),
        },
    })
}
