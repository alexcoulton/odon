//! JSON request-schema generation for registry introspection.

use super::*;

fn request_schema(shape: RequestShape) -> Value {
    match shape {
        RequestShape::Empty => json!({
            "type": "object",
            "properties": {},
            "additionalProperties": false,
        }),
        RequestShape::ShellGet => json!({
            "type":"object",
            "properties":{"mode":{"type":"string","enum":["project","single","mosaic"]}},
            "additionalProperties":false,
        }),
        RequestShape::MenuReplace => json!({
            "type":"object",
            "properties":{
                "if_command_revision":{"type":"integer","minimum":1},
                "transaction_id":{"type":"string","minLength":1,"maxLength":128},
                "menu":{"type":"object"}
            },
            "required":["menu"],
            "additionalProperties":false,
        }),
        RequestShape::ToolbarReplace => json!({
            "type":"object",
            "properties":{
                "if_command_revision":{"type":"integer","minimum":1},
                "transaction_id":{"type":"string","minLength":1,"maxLength":128},
                "toolbar":{"type":"object"}
            },
            "required":["toolbar"],
            "additionalProperties":false,
        }),
        RequestShape::PaletteReplace => json!({
            "type":"object",
            "properties":{
                "if_command_revision":{"type":"integer","minimum":1},
                "transaction_id":{"type":"string","minLength":1,"maxLength":128},
                "palette":{"type":"object"}
            },
            "required":["palette"],
            "additionalProperties":false,
        }),
        RequestShape::CommandRegister => json!({
            "type":"object",
            "properties":{
                "extension_id":{"type":"string","minLength":1,"maxLength":256},
                "if_command_revision":{"type":"integer","minimum":1},
                "transaction_id":{"type":"string","minLength":1,"maxLength":128},
                "command":{"type":"object"}
            },
            "required":["extension_id","command"],
            "additionalProperties":false,
        }),
        RequestShape::CommandRemove => json!({
            "type":"object",
            "properties":{
                "extension_id":{"type":"string","minLength":1,"maxLength":256},
                "command_id":{"type":"string","minLength":1,"maxLength":256},
                "if_command_revision":{"type":"integer","minimum":1},
                "transaction_id":{"type":"string","minLength":1,"maxLength":128}
            },
            "required":["extension_id","command_id"],
            "additionalProperties":false,
        }),
        RequestShape::CommandExecute => json!({
            "type":"object",
            "properties":{
                "command_id":{"type":"string","minLength":1,"maxLength":256},
                "checked":{"type":"boolean"}
            },
            "required":["command_id"],
            "additionalProperties":false,
        }),
        RequestShape::CommandCleanup => json!({
            "type":"object",
            "properties":{
                "extensions":{"type":"array","maxItems":256,"items":{"type":"object"}}
            },
            "required":["extensions"],
            "additionalProperties":false,
        }),
        RequestShape::CommandSync => json!({
            "type":"object",
            "properties":{"context":{"type":"object"}},
            "required":["context"],
            "additionalProperties":false,
        }),
        RequestShape::ShellImportLayout => json!({
            "type":"object",
            "properties":{
                "mode":{"type":"string","enum":["project","single","mosaic"]},
                "if_shell_revision":{"type":"integer","minimum":1},
                "transaction_id":{"type":"string","minLength":1,"maxLength":128},
                "document":{"type":"object"}
            },
            "required":["document"],
            "additionalProperties":false,
        }),
        RequestShape::ShellPatch => json!({
            "type":"object",
            "properties":{
                "mode":{"type":"string","enum":["project","single","mosaic"]},
                "if_shell_revision":{"type":"integer","minimum":1},
                "transaction_id":{"type":"string","minLength":1,"maxLength":128},
                "visibility":{
                    "type":"object",
                    "propertyNames":{"type":"string","minLength":1,"maxLength":256},
                    "additionalProperties":{"type":"boolean"}
                },
                "orders":{
                    "type":"object",
                    "propertyNames":{"type":"string","minLength":1,"maxLength":256},
                    "additionalProperties":{
                        "type":"array",
                        "items":{"type":"string","minLength":1,"maxLength":256},
                        "uniqueItems":true
                    }
                },
                "selected":{
                    "type":"object",
                    "propertyNames":{"type":"string","minLength":1,"maxLength":256},
                    "additionalProperties":{"type":"string","minLength":1,"maxLength":256}
                }
            },
            "additionalProperties":false,
        }),
        RequestShape::ShellReplaceLayout => json!({
            "type":"object",
            "properties":{
                "mode":{"type":"string","enum":["project","single","mosaic"]},
                "if_shell_revision":{"type":"integer","minimum":1},
                "transaction_id":{"type":"string","minLength":1,"maxLength":128},
                "desired_tree":{"type":"object"}
            },
            "required":["desired_tree"],
            "additionalProperties":false,
        }),
        RequestShape::ShellPatchLayout => json!({
            "type":"object",
            "properties":{
                "mode":{"type":"string","enum":["project","single","mosaic"]},
                "if_shell_revision":{"type":"integer","minimum":1},
                "transaction_id":{"type":"string","minLength":1,"maxLength":128},
                "visibility":{"type":"object","additionalProperties":{"type":"boolean"}},
                "selected":{"type":"object","additionalProperties":{"type":"string","minLength":1,"maxLength":256}},
                "sizes":{"type":"object","additionalProperties":{"type":"object"}},
                "splits":{"type":"object","additionalProperties":{"type":"object"}},
                "collapsed":{"type":"object","additionalProperties":{"type":"boolean"}},
                "configurations":{"type":"object","additionalProperties":{"type":"object"}},
                "active_region_id":{"type":"string","minLength":1,"maxLength":256},
                "focused_node_id":{"type":"string","minLength":1,"maxLength":256},
                "clear_focus":{"type":"boolean"}
            },
            "additionalProperties":false,
        }),
        RequestShape::ShellProfileList => json!({
            "type":"object",
            "properties":{"scope":{"type":"string","enum":["session","application","project"],"default":"session"}},
            "additionalProperties":false,
        }),
        RequestShape::ShellProfileSave => json!({
            "type":"object",
            "properties":{
                "name":{"type":"string","minLength":1,"maxLength":128},
                "scope":{"type":"string","enum":["session","application","project"],"default":"session"},
                "mode":{"type":"string","enum":["project","single","mosaic"]}
            },
            "required":["name"],
            "additionalProperties":false,
        }),
        RequestShape::ShellProfileLoad => json!({
            "type":"object",
            "properties":{
                "name":{"type":"string","minLength":1,"maxLength":128},
                "scope":{"type":"string","enum":["session","application","project"],"default":"session"},
                "mode":{"type":"string","enum":["project","single","mosaic"]},
                "if_shell_revision":{"type":"integer","minimum":1},
                "transaction_id":{"type":"string","minLength":1,"maxLength":128}
            },
            "required":["name"],
            "additionalProperties":false,
        }),
        RequestShape::ShellProfileRemove => json!({
            "type":"object",
            "properties":{
                "name":{"type":"string","minLength":1,"maxLength":128},
                "scope":{"type":"string","enum":["session","application","project"],"default":"session"}
            },
            "required":["name"],
            "additionalProperties":false,
        }),
        RequestShape::ShellReset => json!({
            "type":"object",
            "properties":{
                "mode":{"type":"string","enum":["project","single","mosaic"]},
                "if_shell_revision":{"type":"integer","minimum":1},
                "transaction_id":{"type":"string","minLength":1,"maxLength":128}
            },
            "additionalProperties":false,
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
            "properties": {
                "path": {"type": "string"},
                "overwrite": {"type": "boolean", "default": false}
            },
            "additionalProperties": false,
        }),
        RequestShape::AppSettings => json!({
            "type": "object",
            "properties": {
                "fast_object_rendering": {"type": "boolean"},
                "shell_layout_startup_profiles": {
                    "type":"object",
                    "properties":{
                        "project":{"type":"string","minLength":1,"maxLength":128},
                        "single":{"type":"string","minLength":1,"maxLength":128},
                        "mosaic":{"type":"string","minLength":1,"maxLength":128}
                    },
                    "additionalProperties":false
                },
                "auto_contrast": {
                    "type": "object",
                    "properties": {
                        "enabled_on_open": {"type": "boolean"},
                        "method": {"type": "string", "enum": ["zero_to_p97", "p1_to_p99", "zero_to_max"]},
                        "lower_percentile": {"type": "integer", "minimum": 0, "maximum": 99},
                        "upper_percentile": {"type": "integer", "minimum": 1, "maximum": 100}
                    },
                    "additionalProperties": false
                }
            },
            "additionalProperties": false,
        }),
        RequestShape::LifecycleRequest => json!({
            "type": "object",
            "properties": {"save": {"type": "string", "enum": ["prompt", "save", "discard"], "default": "prompt"}},
            "additionalProperties": false,
        }),
        RequestShape::SetScaleBar => json!({
            "type": "object",
            "properties": {"visible": {"type": "boolean"}},
            "required": ["visible"],
            "additionalProperties": false,
        }),
        RequestShape::ScreenshotSettings => json!({
            "type": "object",
            "properties": {
                "output_dir": {"type": ["string", "null"]},
                "include_scale_bar": {"type": "boolean"},
                "include_legend": {"type": "boolean"},
                "scale_bar_scale": {"type": "number", "minimum": 0.5, "maximum": 3.0},
                "legend_scale": {"type": "number", "minimum": 0.5, "maximum": 3.0}
            },
            "additionalProperties": false,
        }),
        RequestShape::TileLoading => json!({
            "type": "object",
            "properties": {
                "workers": {"type": "integer", "minimum": 1, "maximum": 12},
                "prefetch_mode": {"type": "string", "enum": ["off", "target_halo", "target_and_finer_halo"]},
                "prefetch_aggressiveness": {"type": "string", "enum": ["conservative", "balanced", "aggressive"]},
                "prefer_pinned_finer_levels": {"type": "boolean"}
            },
            "additionalProperties": false,
        }),
        RequestShape::MemoryPin => json!({
            "type": "object",
            "properties": {
                "level": {"type": "integer", "minimum": 0},
                "channels": {"type": "array", "items": {"anyOf": [{"type": "string"}, {"type": "integer", "minimum": 0}]}, "uniqueItems": true},
                "scope": {"type": "string", "enum": ["focused", "item", "all"], "default": "focused"},
                "item": {"anyOf": [{"type": "string", "minLength": 1}, {"type": "integer", "minimum": 0}]},
                "force": {"type": "boolean", "default": false}
            },
            "required": ["level"],
            "additionalProperties": false,
        }),
        RequestShape::MemoryUnpin => json!({
            "type": "object",
            "properties": {
                "level": {"type": "integer", "minimum": 0},
                "scope": {"type": "string", "enum": ["focused", "item", "all"], "default": "focused"},
                "item": {"anyOf": [{"type": "string", "minLength": 1}, {"type": "integer", "minimum": 0}]}
            },
            "required": ["level"],
            "additionalProperties": false,
        }),
        RequestShape::LabelLoad => json!({
            "type": "object",
            "properties": {"name": {"type": "string", "minLength": 1}},
            "additionalProperties": false,
        }),
        RequestShape::LabelVisibility => json!({
            "type": "object",
            "properties": {
                "visible": {"type": "boolean"},
                "name": {"type": "string", "minLength": 1}
            },
            "required": ["visible"],
            "additionalProperties": false,
        }),
        RequestShape::ChannelPresentation => json!({
            "type": "object",
            "properties": {
                "search": {"type": "string", "maxLength": 4096},
                "sort": {"type": "string", "enum": ["manual", "name_asc", "name_desc", "visible_first", "hidden_first"]}
            },
            "minProperties": 1,
            "additionalProperties": false,
        }),
        RequestShape::MethodAvailability => json!({
            "type": "object",
            "properties": {
                "methods": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                    "maxItems": 256,
                    "uniqueItems": true
                }
            },
            "additionalProperties": false,
        }),
        RequestShape::SetPlane => json!({
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["xy", "xz", "yz"]},
                "slice": {"type": "integer", "minimum": 0}
            },
            "anyOf": [{"required": ["mode"]}, {"required": ["slice"]}],
            "additionalProperties": false,
        }),
        RequestShape::StepPlane => json!({
            "type": "object",
            "properties": {
                "step": {"type": "integer", "minimum": 1, "default": 1},
                "wrap": {"type": "boolean", "default": false}
            },
            "additionalProperties": false,
        }),
        RequestShape::SetChannelColor => json!({
            "type": "object",
            "properties": {
                "index": {"type": "integer", "minimum": 0},
                "channel_index": {"type": "integer", "minimum": 0},
                "name": {"type": "string"},
                "channel": {"anyOf": [{"type": "string"}, {"type": "integer", "minimum": 0}]},
                "marker": {"type": "string"},
                "color_rgb": {"type": "array", "items": {"type": "integer", "minimum": 0, "maximum": 255}, "minItems": 3, "maxItems": 3}
            },
            "required": ["color_rgb"],
            "additionalProperties": false,
        }),
        RequestShape::SetChannelNote => json!({
            "type": "object",
            "properties": {
                "index": {"type": "integer", "minimum": 0},
                "channel_index": {"type": "integer", "minimum": 0},
                "name": {"type": "string"},
                "channel": {"anyOf": [{"type": "string"}, {"type": "integer", "minimum": 0}]},
                "marker": {"type": "string"},
                "note": {"type": "string", "maxLength": 16384}
            },
            "required": ["note"],
            "additionalProperties": false,
        }),
        RequestShape::SetChannelTransform => json!({
            "type": "object",
            "properties": {
                "viewport_id": {"type": "string", "minLength": 1, "maxLength": 128},
                "if_presentation_revision": {"type": "integer", "minimum": 1},
                "index": {"type": "integer", "minimum": 0},
                "channel_index": {"type": "integer", "minimum": 0},
                "name": {"type": "string"},
                "channel": {"anyOf": [{"type": "string"}, {"type": "integer", "minimum": 0}]},
                "marker": {"type": "string"},
                "offset_world": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                "scale": {"type": "array", "items": {"type": "number", "minimum": 0.01, "maximum": 100}, "minItems": 2, "maxItems": 2},
                "rotation_rad": {"type": "number"}
            },
            "additionalProperties": false,
        }),
        RequestShape::NativeLayerSelector => json!({
            "type": "object",
            "properties": {
                "layer_id": {"type": "string", "minLength": 1},
                "id": {"type": "string", "minLength": 1}
            },
            "oneOf": [{"required": ["layer_id"]}, {"required": ["id"]}],
            "additionalProperties": false,
        }),
        RequestShape::NativeLayerVisibility => json!({
            "type": "object",
            "properties": {
                "layer_id": {"type": "string", "minLength": 1},
                "id": {"type": "string", "minLength": 1},
                "visible": {"type": "boolean"}
            },
            "required": ["visible"],
            "oneOf": [{"required": ["layer_id"]}, {"required": ["id"]}],
            "additionalProperties": false,
        }),
        RequestShape::NativeLayerOrder => json!({
            "type": "object",
            "properties": {
                "stack": {"type": "string", "enum": ["channels", "overlays"]},
                "layers": {"type": "array", "items": {"type": "string", "minLength": 1}, "maxItems": 4096, "uniqueItems": true}
            },
            "required": ["stack", "layers"],
            "additionalProperties": false,
        }),
        RequestShape::NativeLayerOffset => json!({
            "type": "object",
            "properties": {
                "layer_id": {"type": "string", "minLength": 1},
                "id": {"type": "string", "minLength": 1},
                "offset_world": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}
            },
            "required": ["offset_world"],
            "oneOf": [{"required": ["layer_id"]}, {"required": ["id"]}],
            "additionalProperties": false,
        }),
        RequestShape::ProjectViewSelector => json!({
            "type": "object",
            "properties": {
                "index": {"type": "integer", "minimum": 0},
                "name": {"type": "string", "minLength": 1}
            },
            "oneOf": [{"required": ["index"]}, {"required": ["name"]}],
            "additionalProperties": false,
        }),
        RequestShape::ProjectViewCreate => json!({
            "type": "object",
            "properties": {
                "name": {"type": "string", "minLength": 1},
                "spec": {"type": "object"}
            },
            "required": ["name"],
            "additionalProperties": false,
        }),
        RequestShape::ProjectViewCapture => json!({
            "type": "object",
            "properties": {
                "name": {"type": "string", "minLength": 1},
                "viewport_id": {"type": "string", "minLength": 1, "maxLength": 128}
            },
            "required": ["name"],
            "additionalProperties": false,
        }),
        RequestShape::ProjectViewRename => json!({
            "type": "object",
            "properties": {
                "index": {"type": "integer", "minimum": 0},
                "name": {"type": "string", "minLength": 1},
                "new_name": {"type": "string", "minLength": 1}
            },
            "required": ["new_name"],
            "oneOf": [{"required": ["index"]}, {"required": ["name"]}],
            "additionalProperties": false,
        }),
        RequestShape::ProjectCreate => json!({
            "type": "object",
            "properties": {"default_dataset": {"type": "string", "minLength": 1}},
            "additionalProperties": false,
        }),
        RequestShape::Path => json!({
            "type": "object",
            "properties": {"path": {"type": "string", "minLength": 1}},
            "required": ["path"],
            "additionalProperties": false,
        }),
        RequestShape::ProjectMetadata => json!({
            "type": "object",
            "properties": {
                "default_dataset": {"type": ["string", "null"]},
                "secondary_dataset": {"type": ["string", "null"]},
                "default_threshold_marker": {"type": ["string", "null"]},
                "mosaic_segmentation_search_roots": {"type": "array", "items": {"type": "string", "minLength": 1}, "maxItems": 4096}
            },
            "additionalProperties": false,
        }),
        RequestShape::SamplesheetInspect => json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "minLength": 1},
                "offset": {"type": "integer", "minimum": 0, "default": 0},
                "limit": {"type": "integer", "minimum": 1, "maximum": 10000, "default": 200}
            },
            "required": ["path"],
            "additionalProperties": false,
        }),
        RequestShape::SamplesheetExport => json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "minLength": 1},
                "overwrite": {"type": "boolean", "default": false}
            },
            "required": ["path"],
            "additionalProperties": false,
        }),
        RequestShape::SpatialDataOpen => json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "minLength": 1},
                "image": {"type": "string", "minLength": 1},
                "extra_images": {"type": "array", "items": {"type": "string", "minLength": 1}, "uniqueItems": true},
                "labels": {"type": ["string", "null"]},
                "shapes": {"type": "array", "items": {"type": "string", "minLength": 1}, "uniqueItems": true},
                "points": {"type": ["string", "null"]},
                "points_max": {"type": "integer", "minimum": 0, "maximum": 200000000, "default": 200000}
            },
            "required": ["path", "image"],
            "additionalProperties": false,
        }),
        RequestShape::XeniumOpen => json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "minLength": 1},
                "imagery": {"type": "string", "enum": ["auto", "ome_zarr", "tiff"], "default": "auto"},
                "load_cells": {"type": "boolean", "default": true},
                "load_transcripts": {"type": "boolean", "default": true}
            },
            "required": ["path"],
            "additionalProperties": false,
        }),
        RequestShape::HttpOpen => json!({
            "type": "object",
            "properties": {"url": {"type": "string", "minLength": 1}},
            "required": ["url"],
            "additionalProperties": false,
        }),
        RequestShape::S3Session => json!({
            "type": "object",
            "properties": {
                "endpoint": {"type": "string", "minLength": 1},
                "region": {"type": "string", "minLength": 1, "default": "auto"},
                "bucket": {"type": "string", "minLength": 1},
                "access_key": {"type": "string", "minLength": 1},
                "secret_key": {"type": "string", "minLength": 1}
            },
            "required": ["endpoint", "bucket", "access_key", "secret_key"],
            "additionalProperties": false,
        }),
        RequestShape::S3Prefix => json!({
            "type": "object",
            "properties": {"prefix": {"type": "string", "default": ""}},
            "additionalProperties": false,
        }),
        RequestShape::TiffOpen => json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "minLength": 1},
                "z": {"type": "integer", "minimum": 0, "default": 0},
                "t": {"type": "integer", "minimum": 0, "default": 0}
            },
            "required": ["path"],
            "additionalProperties": false,
        }),
        RequestShape::ProjectRoiId => json!({
            "type": "object",
            "properties": {"id": {"type": "string", "minLength": 1}},
            "required": ["id"],
            "additionalProperties": false,
        }),
        RequestShape::ProjectRoiAdd => json!({
            "type": "object",
            "properties": {
                "id": {"type": "string", "minLength": 1},
                "path": {"type": "string", "minLength": 1},
                "display_name": {"type": "string"},
                "dataset": {"type": "string"},
                "segmentation_path": {"type": "string"},
                "metadata": {"type": "object", "additionalProperties": {"type": "string"}},
                "replacement": {"type": "object", "description": "Complete project ROI used by the native command adapter"}
            },
            "oneOf": [
                {"required": ["id", "path"], "not": {"required": ["replacement"]}},
                {
                    "required": ["replacement"],
                    "not": {"anyOf": [
                        {"required": ["id"]},
                        {"required": ["path"]},
                        {"required": ["display_name"]},
                        {"required": ["dataset"]},
                        {"required": ["segmentation_path"]},
                        {"required": ["metadata"]}
                    ]}
                }
            ],
            "additionalProperties": false,
        }),
        RequestShape::ProjectRoiUpdate => json!({
            "type": "object",
            "properties": {
                "target_id": {"type": "string", "minLength": 1},
                "changes": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "minLength": 1},
                        "path": {"type": "string", "minLength": 1},
                        "display_name": {"type": ["string", "null"]},
                        "dataset": {"type": ["string", "null"]},
                        "segmentation_path": {"type": ["string", "null"]},
                        "metadata": {"type": "object", "additionalProperties": {"type": "string"}}
                    },
                    "minProperties": 1,
                    "additionalProperties": false
                },
                "replacement": {"type": "object", "description": "Complete project ROI used by the native command adapter"}
            },
            "required": ["target_id"],
            "oneOf": [
                {"required": ["changes"], "not": {"required": ["replacement"]}},
                {"required": ["replacement"], "not": {"required": ["changes"]}}
            ],
            "additionalProperties": false,
        }),
        RequestShape::ProjectRoiOrder => json!({
            "type": "object",
            "properties": {"ids": {"type": "array", "items": {"type": "string", "minLength": 1}, "maxItems": 100000, "uniqueItems": true}},
            "required": ["ids"],
            "additionalProperties": false,
        }),
        RequestShape::ProjectRoiSelect => json!({
            "type": "object",
            "properties": {
                "ids": {"type": "array", "items": {"type": "string", "minLength": 1}, "maxItems": 100000, "uniqueItems": true},
                "mode": {"type": "string", "enum": ["replace", "add", "remove", "toggle"], "default": "replace"}
            },
            "required": ["ids"],
            "additionalProperties": false,
        }),
        RequestShape::MosaicFocus => json!({
            "type": "object",
            "properties": {
                "index": {"type": "integer", "minimum": 0},
                "roi_id": {"type": "string", "minLength": 1},
                "id": {"type": "string", "minLength": 1},
                "fit": {"type": "boolean", "default": true}
            },
            "oneOf": [{"required": ["index"]}, {"required": ["roi_id"]}, {"required": ["id"]}],
            "additionalProperties": false,
        }),
        RequestShape::MosaicItems => json!({
            "type": "object",
            "properties": {
                "offset": {"type": "integer", "minimum": 0, "default": 0},
                "limit": {"type": "integer", "minimum": 1, "maximum": 10000, "default": 200}
            },
            "additionalProperties": false,
        }),
        RequestShape::MosaicSelect => json!({
            "type": "object",
            "properties": {
                "ids": {"type": "array", "items": {"type": "string", "minLength": 1}, "uniqueItems": true},
                "mode": {"type": "string", "enum": ["replace", "add", "remove", "toggle", "all", "range"], "default": "replace"},
                "start": {"type": "string", "minLength": 1},
                "end": {"type": "string", "minLength": 1}
            },
            "additionalProperties": false,
        }),
        RequestShape::MosaicLayout => json!({
            "type": "object",
            "properties": {
                "group_by": {"type": "string"},
                "sort_by": {"type": "string", "minLength": 1},
                "sort_by_secondary": {"type": "string", "minLength": 1},
                "sort_secondary_enabled": {"type": "boolean"},
                "show_group_labels": {"type": "boolean"},
                "show_text_labels": {"type": "boolean"},
                "group_gap": {"type": "number", "minimum": 0},
                "columns": {"type": "integer", "minimum": 1},
                "layout": {"type": "string", "enum": ["fit_cells", "native_pixels"]},
                "layout_mode": {"type": "string", "enum": ["fit_cells", "native_pixels"]},
                "label_columns": {"type": "array", "items": {"type": "string", "minLength": 1}, "uniqueItems": true},
                "fit": {"type": "boolean", "default": true}
            },
            "additionalProperties": false,
        }),
        RequestShape::ObjectPreloadStart => json!({
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["full_geometry", "centroid_points"], "default": "full_geometry"},
                "lazy_properties": {"type": "boolean", "default": true}
            },
            "additionalProperties": false,
        }),
        RequestShape::DeepLinkUri => json!({
            "type": "object",
            "properties": {"url": {"type": "string", "pattern": "^[Oo][Dd][Oo][Nn]:(//)?"}},
            "required": ["url"],
            "additionalProperties": false,
        }),
        RequestShape::DeepLinkGenerate => json!({
            "type": "object",
            "properties": {
                "request": {"type": "object"},
                "include_project": {"type": "boolean", "default": true},
                "roi": {"type": ["string", "null"]}
            },
            "additionalProperties": false,
        }),
        RequestShape::DeepLinkApply => json!({
            "type": "object",
            "properties": {
                "url": {"type": "string", "pattern": "^[Oo][Dd][Oo][Nn]:(//)?"},
                "request": {"type": "object"}
            },
            "oneOf": [{"required": ["url"]}, {"required": ["request"]}],
            "additionalProperties": false,
        }),
        RequestShape::Object => json!({
            "type": "object",
            "properties": {},
            "additionalProperties": true,
        }),
    }
}

pub(super) fn request_schema_for(descriptor: &MethodDescriptor) -> Value {
    let mut schema = request_schema(descriptor.request_shape);
    if descriptor.mutates
        && let Some(properties) = schema.get_mut("properties").and_then(Value::as_object_mut)
    {
        properties.insert(
            "if_revision".to_string(),
            json!({"type": "integer", "minimum": 0}),
        );
    }
    let explicit_viewport_method = descriptor.name.starts_with("viewer.viewports.")
        && !matches!(
            descriptor.name,
            "viewer.viewports.list" | "viewer.viewports.create"
        );
    if explicit_viewport_method {
        if let Some(properties) = schema.get_mut("properties").and_then(Value::as_object_mut) {
            properties.insert(
                "viewport_id".to_string(),
                json!({"type": "string", "minLength": 1, "maxLength": 128}),
            );
        }
        schema["required"] = json!(["viewport_id"]);
    }
    if matches!(
        descriptor.name,
        "viewer.viewports.camera.set"
            | "viewer.viewports.camera.fit"
            | "viewer.viewports.planes.set"
    ) && let Some(properties) = schema.get_mut("properties").and_then(Value::as_object_mut)
    {
        properties.insert(
            "if_navigation_revision".to_string(),
            json!({"type": "integer", "minimum": 1}),
        );
    }
    if matches!(
        descriptor.name,
        "viewer.workspace.layout.set" | "viewer.viewports.create" | "viewer.viewports.clone"
    ) && let Some(properties) = schema.get_mut("properties").and_then(Value::as_object_mut)
    {
        properties.insert(
            "ratio".to_string(),
            json!({"type": "number", "minimum": 0.1, "maximum": 0.9}),
        );
        if descriptor.name == "viewer.workspace.layout.set" {
            properties.insert(
                "viewports".to_string(),
                json!({
                    "type": "array",
                    "items": {"type": "string", "minLength": 1, "maxLength": 128},
                    "minItems": 1,
                    "maxItems": 2,
                    "uniqueItems": true
                }),
            );
        }
    }
    if matches!(
        descriptor.name,
        "viewer.viewport_links.create"
            | "viewer.viewport_links.update"
            | "viewer.viewport_links.remove"
    ) && let Some(properties) = schema.get_mut("properties").and_then(Value::as_object_mut)
    {
        properties.insert(
            "link_group_id".to_string(),
            json!({"type": "string", "const": "comparison-navigation"}),
        );
        properties.insert(
            "viewports".to_string(),
            json!({
                "type": "array",
                "items": {"type": "string", "minLength": 1, "maxLength": 128},
                "minItems": 2,
                "maxItems": 2,
                "uniqueItems": true
            }),
        );
        properties.insert(
            "fields".to_string(),
            json!({
                "type": "array",
                "items": {"type": "string", "enum": ["camera", "plane", "selection"]},
                "uniqueItems": true
            }),
        );
        if descriptor.name == "viewer.viewport_links.create" {
            schema["required"] = json!(["viewports", "fields"]);
        } else if descriptor.name == "viewer.viewport_links.update" {
            schema["required"] = json!(["fields"]);
        }
    }
    if matches!(
        descriptor.name,
        "viewer.objects.selection.select_filtered"
            | "viewer.analysis.histogram"
            | "viewer.analysis.suggest_thresholds"
            | "viewer.measurements.start"
            | "exports.objects.start"
            | "exports.objects.export_csv"
            | "exports.objects.export_geoparquet"
    ) && let Some(properties) = schema.get_mut("properties").and_then(Value::as_object_mut)
    {
        properties.insert(
            "viewport_id".to_string(),
            json!({"type": "string", "minLength": 1, "maxLength": 128}),
        );
        properties.insert("filter_query".to_string(), json!({"type": "string"}));
        properties.insert("use_all_objects".to_string(), json!({"type": "boolean"}));
        properties.insert(
            "use_active_viewport_filter".to_string(),
            json!({"type": "boolean"}),
        );
    }
    if matches!(
        descriptor.name,
        "viewer.viewports.rename"
            | "viewer.viewports.channels.set_visible"
            | "viewer.viewports.channels.set"
            | "viewer.viewports.channels.set_active"
            | "viewer.viewports.channels.set_color"
            | "viewer.viewports.channels.set_contrast"
            | "viewer.viewports.channels.set_order"
            | "viewer.viewports.channels.set_group"
            | "viewer.viewports.rendering.set"
            | "viewer.viewports.objects.style.set"
            | "viewer.viewports.objects.legend.set"
            | "viewer.viewports.objects.filter.set"
            | "viewer.viewports.objects.filter.clear"
            | "viewer.viewports.layers.set_visibility"
            | "viewer.viewports.layers.set"
            | "viewer.viewports.layers.set_order"
            | "viewer.viewports.layers.set_active"
            | "viewer.viewports.layers.state.replace"
    ) && let Some(properties) = schema.get_mut("properties").and_then(Value::as_object_mut)
    {
        properties.insert(
            "if_presentation_revision".to_string(),
            json!({"type": "integer", "minimum": 1}),
        );
    }
    schema
}
