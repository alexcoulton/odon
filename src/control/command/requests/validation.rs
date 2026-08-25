//! Request-shape validation and method-specific semantic checks.

use super::*;

mod commands;
mod shell;

use shell::{
    validate_shell_id, validate_shell_mode, validate_shell_profile_name,
    validate_shell_profile_scope, validate_shell_transaction_id,
};

fn validate_optional_nullable_string(
    method: &str,
    name: &str,
    value: Option<&Value>,
) -> Result<(), ControlError> {
    if value.is_some_and(|value| !(value.is_null() || value.is_string())) {
        return Err(ControlError::invalid_params(
            method,
            format!("{name} must be a string or null"),
        ));
    }
    Ok(())
}

fn validate_project_view_selector(
    method: &str,
    index: Option<usize>,
    name: Option<&str>,
) -> Result<(), ControlError> {
    if index.is_some() == name.is_some() {
        return Err(ControlError::invalid_params(
            method,
            "provide exactly one of index or name",
        ));
    }
    if name.is_some_and(|name| name.trim().is_empty()) {
        return Err(ControlError::invalid_params(
            method,
            "view preset name must not be empty",
        ));
    }
    Ok(())
}

fn validate_native_layer_selector(
    method: &str,
    layer_id: Option<&str>,
    id: Option<&str>,
) -> Result<(), ControlError> {
    if layer_id.or(id).is_none_or(|value| value.trim().is_empty()) {
        return Err(ControlError::invalid_params(
            method,
            "a non-empty layer_id is required",
        ));
    }
    if layer_id.is_some() && id.is_some() {
        return Err(ControlError::invalid_params(
            method,
            "provide layer_id or id, not both",
        ));
    }
    Ok(())
}

fn has_channel_selector(
    index: Option<usize>,
    channel_index: Option<usize>,
    name: Option<&str>,
    channel: Option<&ChannelSelector>,
    marker: Option<&str>,
) -> bool {
    index.is_some()
        || channel_index.is_some()
        || name.is_some_and(|value| !value.trim().is_empty())
        || channel.is_some()
        || marker.is_some_and(|value| !value.trim().is_empty())
}

pub(in crate::control::command) fn validate_params(
    method: &str,
    shape: RequestShape,
    params: &Value,
) -> Result<(), ControlError> {
    let invalid = |error: serde_json::Error| {
        ControlError::invalid_params(method, format!("invalid parameters: {error}"))
    };
    match shape {
        RequestShape::Empty => {
            if params.as_object().is_some_and(|object| !object.is_empty()) {
                return Err(ControlError::invalid_params(
                    method,
                    "this method does not accept parameters",
                ));
            }
        }
        RequestShape::ShellGet => {
            let request: ShellGetRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            validate_shell_mode(method, request.mode.as_deref())?;
        }
        RequestShape::MenuReplace => {
            commands::menu_replace(method, params)?;
        }
        RequestShape::ToolbarReplace => {
            commands::toolbar_replace(method, params)?;
        }
        RequestShape::PaletteReplace => {
            commands::palette_replace(method, params)?;
        }
        RequestShape::CommandRegister => {
            commands::register(method, params)?;
        }
        RequestShape::CommandRemove => {
            commands::remove(method, params)?;
        }
        RequestShape::CommandExecute => {
            commands::execute(method, params)?;
        }
        RequestShape::CommandCleanup => {
            commands::cleanup(method, params)?;
        }
        RequestShape::CommandSync => {
            commands::sync(method, params)?;
        }
        RequestShape::ShellImportLayout => {
            let request: ShellImportLayoutRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            validate_shell_mode(method, request.mode.as_deref())?;
            validate_shell_transaction_id(method, request.transaction_id.as_deref())?;
            if request.if_shell_revision == Some(0) {
                return Err(ControlError::invalid_params(
                    method,
                    "if_shell_revision must be at least 1",
                ));
            }
            if !request.document.is_object() {
                return Err(ControlError::invalid_params(
                    method,
                    "document must be an object",
                ));
            }
        }
        RequestShape::ShellPatch => {
            let request: ShellPatchRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            validate_shell_mode(method, request.mode.as_deref())?;
            validate_shell_transaction_id(method, request.transaction_id.as_deref())?;
            if request.if_shell_revision == Some(0) {
                return Err(ControlError::invalid_params(
                    method,
                    "if_shell_revision must be at least 1",
                ));
            }
            if let Some(visibility) = request.visibility {
                for id in visibility.keys() {
                    validate_shell_id(method, "visibility", id)?;
                }
            }
            if let Some(orders) = request.orders {
                for (id, children) in orders {
                    validate_shell_id(method, "orders", &id)?;
                    let mut unique = std::collections::BTreeSet::new();
                    for child in children {
                        validate_shell_id(method, "orders child", &child)?;
                        if !unique.insert(child) {
                            return Err(ControlError::invalid_params(
                                method,
                                format!("order for '{id}' contains a duplicate child ID"),
                            ));
                        }
                    }
                }
            }
            if let Some(selected) = request.selected {
                for (id, child) in selected {
                    validate_shell_id(method, "selected container", &id)?;
                    validate_shell_id(method, "selected child", &child)?;
                }
            }
        }
        RequestShape::ShellReplaceLayout => {
            let request: ShellReplaceLayoutRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            validate_shell_mode(method, request.mode.as_deref())?;
            validate_shell_transaction_id(method, request.transaction_id.as_deref())?;
            if request.if_shell_revision == Some(0) {
                return Err(ControlError::invalid_params(
                    method,
                    "if_shell_revision must be at least 1",
                ));
            }
            if !request.desired_tree.is_object() {
                return Err(ControlError::invalid_params(
                    method,
                    "desired_tree must be an object",
                ));
            }
        }
        RequestShape::ShellPatchLayout => {
            let request: ShellPatchLayoutRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            validate_shell_mode(method, request.mode.as_deref())?;
            validate_shell_transaction_id(method, request.transaction_id.as_deref())?;
            if request.if_shell_revision == Some(0) {
                return Err(ControlError::invalid_params(
                    method,
                    "if_shell_revision must be at least 1",
                ));
            }
            for id in request
                .visibility
                .iter()
                .flat_map(|values| values.keys())
                .chain(request.selected.iter().flat_map(|values| values.keys()))
                .chain(request.sizes.iter().flat_map(|values| values.keys()))
                .chain(request.splits.iter().flat_map(|values| values.keys()))
                .chain(request.collapsed.iter().flat_map(|values| values.keys()))
                .chain(
                    request
                        .configurations
                        .iter()
                        .flat_map(|values| values.keys()),
                )
            {
                validate_shell_id(method, "layout node", id)?;
            }
            if let Some(selected) = request.selected {
                for child in selected.values() {
                    validate_shell_id(method, "selected layout child", child)?;
                }
            }
            if let Some(id) = request.active_region_id {
                validate_shell_id(method, "active_region_id", &id)?;
            }
            if let Some(id) = request.focused_node_id {
                validate_shell_id(method, "focused_node_id", &id)?;
            }
            if request.clear_focus == Some(true) && params.get("focused_node_id").is_some() {
                return Err(ControlError::invalid_params(
                    method,
                    "clear_focus and focused_node_id are mutually exclusive",
                ));
            }
            for (label, values) in [
                ("sizes", request.sizes),
                ("splits", request.splits),
                ("configurations", request.configurations),
            ] {
                if let Some(values) = values {
                    for (id, value) in values {
                        if !value.is_object() {
                            return Err(ControlError::invalid_params(
                                method,
                                format!("{label} value for '{id}' must be an object"),
                            ));
                        }
                    }
                }
            }
        }
        RequestShape::ShellProfileList => {
            let request: ShellProfileListRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            validate_shell_profile_scope(method, request.scope.as_deref())?;
        }
        RequestShape::ShellProfileSave => {
            let request: ShellProfileSaveRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            validate_shell_profile_name(method, &request.name)?;
            validate_shell_profile_scope(method, request.scope.as_deref())?;
            validate_shell_mode(method, request.mode.as_deref())?;
        }
        RequestShape::ShellProfileLoad => {
            let request: ShellProfileLoadRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            validate_shell_profile_name(method, &request.name)?;
            validate_shell_profile_scope(method, request.scope.as_deref())?;
            validate_shell_mode(method, request.mode.as_deref())?;
            validate_shell_transaction_id(method, request.transaction_id.as_deref())?;
            if request.if_shell_revision == Some(0) {
                return Err(ControlError::invalid_params(
                    method,
                    "if_shell_revision must be at least 1",
                ));
            }
        }
        RequestShape::ShellProfileRemove => {
            let request: ShellProfileRemoveRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            validate_shell_profile_name(method, &request.name)?;
            validate_shell_profile_scope(method, request.scope.as_deref())?;
        }
        RequestShape::ShellReset => {
            let request: ShellResetRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            validate_shell_mode(method, request.mode.as_deref())?;
            validate_shell_transaction_id(method, request.transaction_id.as_deref())?;
            if request.if_shell_revision == Some(0) {
                return Err(ControlError::invalid_params(
                    method,
                    "if_shell_revision must be at least 1",
                ));
            }
        }
        RequestShape::SetSidePanels => {
            let request: SetSidePanelsRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if request.left.is_none() && request.right.is_none() {
                return Err(ControlError::invalid_params(
                    method,
                    "left and/or right is required",
                ));
            }
        }
        RequestShape::SetSmoothPixels => {
            let request: SetSmoothPixelsRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            let _ = request.smooth;
        }
        RequestShape::SetVisibleChannels => {
            let request: SetVisibleChannelsRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if request.channels.is_empty() && request.mode.as_deref() != Some("only") {
                return Err(ControlError::invalid_params(
                    method,
                    "channels must not be empty unless mode is 'only'",
                ));
            }
            if let Some(mode) = request.mode.as_deref()
                && !matches!(mode, "only" | "show" | "hide")
            {
                return Err(ControlError::invalid_params(
                    method,
                    "mode must be 'only', 'show', or 'hide'",
                ));
            }
            for channel in request.channels {
                match channel {
                    ChannelSelector::Name(name) if name.trim().is_empty() => {
                        return Err(ControlError::invalid_params(
                            method,
                            "channel names must not be empty",
                        ));
                    }
                    ChannelSelector::Name(_) => {}
                    ChannelSelector::Index(index) => {
                        let _ = index;
                    }
                }
            }
        }
        RequestShape::SetCamera => {
            let request: SetCameraRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            let values = request
                .center_world_lvl0
                .into_iter()
                .flatten()
                .chain(request.center_x)
                .chain(request.center_y);
            if values.into_iter().any(|value| !value.is_finite()) {
                return Err(ControlError::invalid_params(
                    method,
                    "camera coordinates must be finite",
                ));
            }
            for zoom in [request.zoom, request.zoom_screen_per_lvl0_px]
                .into_iter()
                .flatten()
            {
                if !zoom.is_finite() || zoom <= 0.0 {
                    return Err(ControlError::invalid_params(
                        method,
                        "zoom must be finite and greater than zero",
                    ));
                }
            }
        }
        RequestShape::CaptureScreenshot => {
            let request: CaptureScreenshotRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if request
                .path
                .as_deref()
                .is_some_and(|path| path.trim().is_empty())
            {
                return Err(ControlError::invalid_params(
                    method,
                    "path must not be empty",
                ));
            }
            let _ = request.viewport_id;
            let _ = request.overwrite;
        }
        RequestShape::AppSettings => {
            let request: AppSettingsRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if let Some(auto) = request.auto_contrast {
                if let Some(method_name) = auto.method
                    && !matches!(
                        method_name.as_str(),
                        "zero_to_p97" | "p1_to_p99" | "zero_to_max"
                    )
                {
                    return Err(ControlError::invalid_params(
                        method,
                        "unknown auto-contrast method",
                    ));
                }
                if auto.lower_percentile.is_some_and(|value| value > 99)
                    || auto
                        .upper_percentile
                        .is_some_and(|value| value == 0 || value > 100)
                {
                    return Err(ControlError::invalid_params(
                        method,
                        "auto-contrast percentiles must be in range",
                    ));
                }
                if let (Some(lower), Some(upper)) = (auto.lower_percentile, auto.upper_percentile)
                    && lower >= upper
                {
                    return Err(ControlError::invalid_params(
                        method,
                        "lower_percentile must be less than upper_percentile",
                    ));
                }
                let _ = auto.enabled_on_open;
            }
            let _ = request.fast_object_rendering;
            let _ = request.shell_layout_startup_profiles;
        }
        RequestShape::LifecycleRequest => {
            let request: LifecycleRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if request
                .save
                .as_deref()
                .is_some_and(|value| !matches!(value, "prompt" | "save" | "discard"))
            {
                return Err(ControlError::invalid_params(
                    method,
                    "save must be prompt, save, or discard",
                ));
            }
        }
        RequestShape::SetScaleBar => {
            let request: SetScaleBarRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            let _ = request.visible;
        }
        RequestShape::ScreenshotSettings => {
            let request: ScreenshotSettingsRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if request
                .output_dir
                .as_ref()
                .and_then(|value| value.as_deref())
                .is_some_and(|path| path.trim().is_empty())
            {
                return Err(ControlError::invalid_params(
                    method,
                    "output_dir must not be empty",
                ));
            }
            for value in [request.scale_bar_scale, request.legend_scale]
                .into_iter()
                .flatten()
            {
                if !value.is_finite() || !(0.5..=3.0).contains(&value) {
                    return Err(ControlError::invalid_params(
                        method,
                        "screenshot scales must be between 0.5 and 3.0",
                    ));
                }
            }
            let _ = (request.include_scale_bar, request.include_legend);
        }
        RequestShape::TileLoading => {
            let request: TileLoadingRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if request
                .workers
                .is_some_and(|value| !(1..=12).contains(&value))
            {
                return Err(ControlError::invalid_params(
                    method,
                    "workers must be from 1 to 12",
                ));
            }
            if request.prefetch_mode.as_deref().is_some_and(|value| {
                !matches!(value, "off" | "target_halo" | "target_and_finer_halo")
            }) {
                return Err(ControlError::invalid_params(
                    method,
                    "unknown prefetch_mode",
                ));
            }
            if request
                .prefetch_aggressiveness
                .as_deref()
                .is_some_and(|value| !matches!(value, "conservative" | "balanced" | "aggressive"))
            {
                return Err(ControlError::invalid_params(
                    method,
                    "unknown prefetch_aggressiveness",
                ));
            }
            let _ = request.prefer_pinned_finer_levels;
        }
        RequestShape::MemoryPin => {
            let request: MemoryPinRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if request
                .scope
                .as_deref()
                .is_some_and(|value| !matches!(value, "focused" | "item" | "all"))
            {
                return Err(ControlError::invalid_params(
                    method,
                    "scope must be focused, item, or all",
                ));
            }
            if request.scope.as_deref() == Some("item") && request.item.is_none() {
                return Err(ControlError::invalid_params(
                    method,
                    "item is required when scope is item",
                ));
            }
            if request
                .item
                .as_ref()
                .is_some_and(|value| !(value.is_string() || value.is_u64()))
            {
                return Err(ControlError::invalid_params(
                    method,
                    "item must be a string or non-negative integer",
                ));
            }
            let _ = (request.level, request.channels, request.force);
        }
        RequestShape::MemoryUnpin => {
            let request: MemoryUnpinRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if request
                .scope
                .as_deref()
                .is_some_and(|value| !matches!(value, "focused" | "item" | "all"))
            {
                return Err(ControlError::invalid_params(
                    method,
                    "scope must be focused, item, or all",
                ));
            }
            if request.scope.as_deref() == Some("item") && request.item.is_none() {
                return Err(ControlError::invalid_params(
                    method,
                    "item is required when scope is item",
                ));
            }
            if request
                .item
                .as_ref()
                .is_some_and(|value| !(value.is_string() || value.is_u64()))
            {
                return Err(ControlError::invalid_params(
                    method,
                    "item must be a string or non-negative integer",
                ));
            }
            let _ = request.level;
        }
        RequestShape::LabelLoad => {
            let request: LabelLoadRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if request
                .name
                .as_deref()
                .is_some_and(|name| name.trim().is_empty())
            {
                return Err(ControlError::invalid_params(
                    method,
                    "label name must not be empty",
                ));
            }
        }
        RequestShape::LabelVisibility => {
            let request: LabelVisibilityRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if request
                .name
                .as_deref()
                .is_some_and(|name| name.trim().is_empty())
            {
                return Err(ControlError::invalid_params(
                    method,
                    "label name must not be empty",
                ));
            }
            let _ = request.visible;
        }
        RequestShape::ChannelPresentation => {
            let request: ChannelPresentationRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if request.search.is_none() && request.sort.is_none() {
                return Err(ControlError::invalid_params(
                    method,
                    "search and/or sort is required",
                ));
            }
            if request
                .search
                .as_ref()
                .is_some_and(|search| search.len() > 4096)
            {
                return Err(ControlError::invalid_params(method, "search is too long"));
            }
            if request.sort.as_deref().is_some_and(|sort| {
                !matches!(
                    sort,
                    "manual" | "name_asc" | "name_desc" | "visible_first" | "hidden_first"
                )
            }) {
                return Err(ControlError::invalid_params(
                    method,
                    "unknown channel sort mode",
                ));
            }
        }
        RequestShape::MethodAvailability => {
            let request: MethodAvailabilityRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if let Some(methods) = request.methods {
                if methods.len() > 256 {
                    return Err(ControlError::invalid_params(
                        method,
                        "methods must contain at most 256 entries",
                    ));
                }
                let mut seen = std::collections::BTreeSet::new();
                for candidate in methods {
                    if candidate.trim().is_empty() {
                        return Err(ControlError::invalid_params(
                            method,
                            "method names must not be empty",
                        ));
                    }
                    let Some(descriptor) = registry::method(&candidate) else {
                        return Err(ControlError::invalid_params(
                            method,
                            format!("unknown method '{candidate}'"),
                        ));
                    };
                    if !seen.insert(descriptor.name) {
                        return Err(ControlError::invalid_params(
                            method,
                            format!("method '{}' is duplicated", descriptor.name),
                        ));
                    }
                }
            }
        }
        RequestShape::SetPlane => {
            let request: SetPlaneRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if request.mode.is_none() && request.slice.is_none() {
                return Err(ControlError::invalid_params(
                    method,
                    "mode and/or slice is required",
                ));
            }
            if let Some(mode) = request.mode.as_deref()
                && !matches!(mode.to_ascii_lowercase().as_str(), "xy" | "xz" | "yz")
            {
                return Err(ControlError::invalid_params(
                    method,
                    "mode must be 'xy', 'xz', or 'yz'",
                ));
            }
        }
        RequestShape::StepPlane => {
            let request: StepPlaneRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if request.step == 0 {
                return Err(ControlError::invalid_params(
                    method,
                    "step must be greater than zero",
                ));
            }
            let _ = request.wrap;
        }
        RequestShape::SetChannelColor => {
            let request: SetChannelColorRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if !has_channel_selector(
                request.index,
                request.channel_index,
                request.name.as_deref(),
                request.channel.as_ref(),
                request.marker.as_deref(),
            ) {
                return Err(ControlError::invalid_params(
                    method,
                    "a channel selector is required",
                ));
            }
            let _ = request.color_rgb;
        }
        RequestShape::SetChannelNote => {
            let request: SetChannelNoteRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if !has_channel_selector(
                request.index,
                request.channel_index,
                request.name.as_deref(),
                request.channel.as_ref(),
                request.marker.as_deref(),
            ) {
                return Err(ControlError::invalid_params(
                    method,
                    "a channel selector is required",
                ));
            }
            if request.note.len() > 16_384 {
                return Err(ControlError::invalid_params(
                    method,
                    "note must be at most 16384 bytes",
                ));
            }
        }
        RequestShape::SetChannelTransform => {
            let request: SetChannelTransformRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if request.if_presentation_revision.is_some()
                && request.viewport_id.as_deref().is_none_or(str::is_empty)
            {
                return Err(ControlError::invalid_params(
                    method,
                    "if_presentation_revision requires viewport_id",
                ));
            }
            if !has_channel_selector(
                request.index,
                request.channel_index,
                request.name.as_deref(),
                request.channel.as_ref(),
                request.marker.as_deref(),
            ) {
                return Err(ControlError::invalid_params(
                    method,
                    "a channel selector is required",
                ));
            }
            if request.offset_world.is_none()
                && request.scale.is_none()
                && request.rotation_rad.is_none()
            {
                return Err(ControlError::invalid_params(
                    method,
                    "offset_world, scale, and/or rotation_rad is required",
                ));
            }
            if request
                .offset_world
                .into_iter()
                .flatten()
                .chain(request.rotation_rad)
                .any(|value| !value.is_finite())
            {
                return Err(ControlError::invalid_params(
                    method,
                    "transform values must be finite",
                ));
            }
            if request
                .scale
                .into_iter()
                .flatten()
                .any(|value| !value.is_finite() || !(0.01..=100.0).contains(&value))
            {
                return Err(ControlError::invalid_params(
                    method,
                    "scale values must be finite and between 0.01 and 100",
                ));
            }
        }
        RequestShape::NativeLayerSelector => {
            let request: NativeLayerSelectorRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            validate_native_layer_selector(
                method,
                request.layer_id.as_deref(),
                request.id.as_deref(),
            )?;
        }
        RequestShape::NativeLayerVisibility => {
            let request: NativeLayerVisibilityRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            validate_native_layer_selector(
                method,
                request.layer_id.as_deref(),
                request.id.as_deref(),
            )?;
            let _ = request.visible;
        }
        RequestShape::NativeLayerOrder => {
            let request: NativeLayerOrderRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if !matches!(request.stack.as_str(), "channels" | "overlays") {
                return Err(ControlError::invalid_params(
                    method,
                    "stack must be 'channels' or 'overlays'",
                ));
            }
            if request.layers.len() > 4096 {
                return Err(ControlError::invalid_params(
                    method,
                    "layers must contain at most 4096 entries",
                ));
            }
            let mut unique = std::collections::BTreeSet::new();
            if request
                .layers
                .iter()
                .any(|layer| layer.trim().is_empty() || !unique.insert(layer))
            {
                return Err(ControlError::invalid_params(
                    method,
                    "layers must contain unique, non-empty layer IDs",
                ));
            }
        }
        RequestShape::NativeLayerOffset => {
            let request: NativeLayerOffsetRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            validate_native_layer_selector(
                method,
                request.layer_id.as_deref(),
                request.id.as_deref(),
            )?;
            if request
                .offset_world
                .into_iter()
                .any(|value| !value.is_finite())
            {
                return Err(ControlError::invalid_params(
                    method,
                    "offset_world values must be finite",
                ));
            }
        }
        RequestShape::ProjectViewSelector => {
            let request: ProjectViewSelectorRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            validate_project_view_selector(method, request.index, request.name.as_deref())?;
        }
        RequestShape::ProjectViewCreate => {
            let request: ProjectViewCreateRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if request.name.trim().is_empty() {
                return Err(ControlError::invalid_params(
                    method,
                    "name must not be empty",
                ));
            }
            if request.spec.as_ref().is_some_and(|spec| !spec.is_object()) {
                return Err(ControlError::invalid_params(
                    method,
                    "spec must be an object",
                ));
            }
        }
        RequestShape::ProjectViewCapture => {
            let request: ProjectViewCaptureRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if request.name.trim().is_empty() {
                return Err(ControlError::invalid_params(
                    method,
                    "name must not be empty",
                ));
            }
            if request
                .viewport_id
                .as_deref()
                .is_some_and(|value| value.trim().is_empty() || value.len() > 128)
            {
                return Err(ControlError::invalid_params(
                    method,
                    "viewport_id must contain 1 to 128 characters",
                ));
            }
        }
        RequestShape::ProjectViewRename => {
            let request: ProjectViewRenameRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            validate_project_view_selector(method, request.index, request.name.as_deref())?;
            if request.new_name.trim().is_empty() {
                return Err(ControlError::invalid_params(
                    method,
                    "new_name must not be empty",
                ));
            }
        }
        RequestShape::ProjectCreate => {
            let request: ProjectCreateRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if request.config.is_some() && request.default_dataset.is_some() {
                return Err(ControlError::invalid_params(
                    method,
                    "config and default_dataset are mutually exclusive",
                ));
            }
            if request
                .default_dataset
                .as_deref()
                .is_some_and(|value| value.trim().is_empty())
            {
                return Err(ControlError::invalid_params(
                    method,
                    "default_dataset must not be empty",
                ));
            }
        }
        RequestShape::Path => {
            let request: PathRequest = serde_json::from_value(params.clone()).map_err(invalid)?;
            if request.path.trim().is_empty() {
                return Err(ControlError::invalid_params(
                    method,
                    "path must not be empty",
                ));
            }
        }
        RequestShape::ProjectMetadata => {
            let request: ProjectMetadataRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            validate_optional_nullable_string(
                method,
                "default_dataset",
                request.default_dataset.as_ref(),
            )?;
            validate_optional_nullable_string(
                method,
                "secondary_dataset",
                request.secondary_dataset.as_ref(),
            )?;
            validate_optional_nullable_string(
                method,
                "default_threshold_marker",
                request.default_threshold_marker.as_ref(),
            )?;
            if request
                .mosaic_segmentation_search_roots
                .as_ref()
                .is_some_and(|roots| {
                    roots.len() > 4096 || roots.iter().any(|root| root.trim().is_empty())
                })
            {
                return Err(ControlError::invalid_params(
                    method,
                    "search roots must contain at most 4096 non-empty paths",
                ));
            }
        }
        RequestShape::SamplesheetInspect => {
            let request: SamplesheetInspectRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if request.path.trim().is_empty() || request.limit == 0 || request.limit > 10_000 {
                return Err(ControlError::invalid_params(
                    method,
                    "path must not be empty and limit must be between 1 and 10000",
                ));
            }
            let _ = request.offset;
        }
        RequestShape::SamplesheetExport => {
            let request: SamplesheetExportRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if request.path.trim().is_empty() {
                return Err(ControlError::invalid_params(
                    method,
                    "path must not be empty",
                ));
            }
            let _ = request.overwrite;
        }
        RequestShape::SpatialDataOpen => {
            let request: SpatialDataOpenRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if request.path.trim().is_empty()
                || request.image.trim().is_empty()
                || request.points_max > 200_000_000
                || request
                    .extra_images
                    .iter()
                    .chain(&request.shapes)
                    .any(|name| name.trim().is_empty())
                || request
                    .labels
                    .as_deref()
                    .into_iter()
                    .chain(request.points.as_deref())
                    .any(|name| name.trim().is_empty())
            {
                return Err(ControlError::invalid_params(
                    method,
                    "SpatialData paths and element names must be non-empty and points_max must not exceed 200000000",
                ));
            }
        }
        RequestShape::XeniumOpen => {
            let request: XeniumOpenRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if request.path.trim().is_empty()
                || !matches!(request.imagery.as_str(), "auto" | "ome_zarr" | "tiff")
            {
                return Err(ControlError::invalid_params(
                    method,
                    "path must not be empty and imagery must be auto, ome_zarr, or tiff",
                ));
            }
            let _ = (request.load_cells, request.load_transcripts);
        }
        RequestShape::HttpOpen => {
            let request: HttpOpenRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            let url = request.url.trim();
            if !(url.starts_with("http://") || url.starts_with("https://")) {
                return Err(ControlError::invalid_params(
                    method,
                    "url must begin with http:// or https://",
                ));
            }
        }
        RequestShape::S3Session => {
            let request: S3SessionRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if [
                request.endpoint.as_str(),
                request.region.as_str(),
                request.bucket.as_str(),
                request.access_key.as_str(),
                request.secret_key.as_str(),
            ]
            .iter()
            .any(|value| value.trim().is_empty())
            {
                return Err(ControlError::invalid_params(
                    method,
                    "endpoint, region, bucket, access_key, and secret_key must not be empty",
                ));
            }
        }
        RequestShape::S3Prefix => {
            let request: S3PrefixRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if request.prefix.contains('\0') {
                return Err(ControlError::invalid_params(
                    method,
                    "prefix must not contain NUL characters",
                ));
            }
        }
        RequestShape::TiffOpen => {
            let request: TiffOpenRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if request.path.trim().is_empty() {
                return Err(ControlError::invalid_params(
                    method,
                    "path must not be empty",
                ));
            }
            let _ = (request.z, request.t);
        }
        RequestShape::ProjectRoiId => {
            let request: ProjectRoiIdRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if request.id.trim().is_empty() {
                return Err(ControlError::invalid_params(method, "id must not be empty"));
            }
        }
        RequestShape::ProjectRoiAdd => {
            let request: ProjectRoiAddRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            let has_ordinary_fields = request.id.is_some()
                || request.path.is_some()
                || request.display_name.is_some()
                || request.dataset.is_some()
                || request.segmentation_path.is_some()
                || request.metadata.is_some();
            match (&request.replacement, has_ordinary_fields) {
                (Some(replacement), false)
                    if !replacement.id.trim().is_empty()
                        && replacement.dataset_source().is_some() => {}
                (None, true)
                    if request
                        .id
                        .as_deref()
                        .is_some_and(|id| !id.trim().is_empty())
                        && request
                            .path
                            .as_deref()
                            .is_some_and(|path| !path.trim().is_empty()) => {}
                (Some(_), _) => {
                    return Err(ControlError::invalid_params(
                        method,
                        "replacement cannot be combined with ordinary ROI fields and must contain an ID and source",
                    ));
                }
                _ => {
                    return Err(ControlError::invalid_params(
                        method,
                        "provide either replacement or non-empty id and path",
                    ));
                }
            }
        }
        RequestShape::ProjectRoiUpdate => {
            let request: ProjectRoiUpdateRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if request.target_id.trim().is_empty() {
                return Err(ControlError::invalid_params(
                    method,
                    "target_id must not be empty",
                ));
            }
            match (request.changes, request.replacement) {
                (Some(changes), None) => {
                    let changes_object = params
                        .get("changes")
                        .and_then(Value::as_object)
                        .expect("deserialized project ROI changes must be an object");
                    if changes_object.is_empty() {
                        return Err(ControlError::invalid_params(
                            method,
                            "changes must not be empty",
                        ));
                    }
                    for name in ["display_name", "dataset", "segmentation_path"] {
                        let value = changes_object.get(name);
                        validate_optional_nullable_string(method, name, value)?;
                    }
                    let _ = (
                        changes.id,
                        changes.path,
                        changes.display_name,
                        changes.dataset,
                        changes.segmentation_path,
                        changes.metadata,
                    );
                }
                (None, Some(replacement))
                    if !replacement.id.trim().is_empty()
                        && replacement.dataset_source().is_some() => {}
                (None, Some(_)) => {
                    return Err(ControlError::invalid_params(
                        method,
                        "replacement must contain a non-empty ID and source",
                    ));
                }
                _ => {
                    return Err(ControlError::invalid_params(
                        method,
                        "provide exactly one of changes or replacement",
                    ));
                }
            }
        }
        RequestShape::ProjectRoiOrder => {
            let request: ProjectRoiOrderRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            let mut unique = std::collections::BTreeSet::new();
            if request.ids.len() > 100_000
                || request
                    .ids
                    .iter()
                    .any(|id| id.trim().is_empty() || !unique.insert(id))
            {
                return Err(ControlError::invalid_params(
                    method,
                    "ids must contain at most 100000 unique, non-empty ROI IDs",
                ));
            }
        }
        RequestShape::ProjectRoiSelect => {
            let request: ProjectRoiSelectRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if !matches!(
                request.mode.as_str(),
                "replace" | "add" | "remove" | "toggle"
            ) {
                return Err(ControlError::invalid_params(
                    method,
                    "mode must be replace, add, remove, or toggle",
                ));
            }
            let mut unique = std::collections::BTreeSet::new();
            if request.ids.len() > 100_000
                || request
                    .ids
                    .iter()
                    .any(|id| id.trim().is_empty() || !unique.insert(id))
            {
                return Err(ControlError::invalid_params(
                    method,
                    "ids must contain at most 100000 unique, non-empty ROI IDs",
                ));
            }
        }
        RequestShape::MosaicFocus => {
            let request: MosaicFocusRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            let selector_count = usize::from(request.index.is_some())
                + usize::from(request.roi_id.is_some())
                + usize::from(request.id.is_some());
            if selector_count != 1 {
                return Err(ControlError::invalid_params(
                    method,
                    "provide exactly one of index, roi_id, or id",
                ));
            }
            if request
                .roi_id
                .as_deref()
                .or(request.id.as_deref())
                .is_some_and(|id| id.trim().is_empty())
            {
                return Err(ControlError::invalid_params(
                    method,
                    "mosaic ROI ID must not be empty",
                ));
            }
            let _ = request.fit;
        }
        RequestShape::MosaicItems => {
            let request: MosaicItemsRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if request.limit == 0 || request.limit > 10_000 {
                return Err(ControlError::invalid_params(
                    method,
                    "limit must be between 1 and 10000",
                ));
            }
            let _ = request.offset;
        }
        RequestShape::MosaicSelect => {
            let request: MosaicSelectRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            let mut unique = std::collections::BTreeSet::new();
            if request
                .ids
                .iter()
                .any(|id| id.trim().is_empty() || !unique.insert(id.as_str()))
            {
                return Err(ControlError::invalid_params(
                    method,
                    "ids must be unique and non-empty",
                ));
            }
            match request.mode.as_str() {
                "all"
                    if request.ids.is_empty()
                        && request.start.is_none()
                        && request.end.is_none() => {}
                "range"
                    if request.ids.is_empty()
                        && request
                            .start
                            .as_deref()
                            .is_some_and(|id| !id.trim().is_empty())
                        && request
                            .end
                            .as_deref()
                            .is_some_and(|id| !id.trim().is_empty()) => {}
                "replace" | "add" | "remove" | "toggle"
                    if !request.ids.is_empty()
                        && request.start.is_none()
                        && request.end.is_none() => {}
                _ => {
                    return Err(ControlError::invalid_params(
                        method,
                        "all takes no IDs, range requires start/end, and other modes require ids",
                    ));
                }
            }
        }
        RequestShape::MosaicLayout => {
            let request: MosaicLayoutRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if params.as_object().is_none_or(|params| params.is_empty()) {
                return Err(ControlError::invalid_params(
                    method,
                    "at least one layout property is required",
                ));
            }
            if request.columns == Some(0)
                || request
                    .group_gap
                    .is_some_and(|gap| !gap.is_finite() || gap < 0.0)
                || request.layout.is_some() && request.layout_mode.is_some()
                || request
                    .label_columns
                    .as_ref()
                    .is_some_and(|columns| columns.iter().any(|column| column.trim().is_empty()))
            {
                return Err(ControlError::invalid_params(
                    method,
                    "layout fields are invalid or conflicting",
                ));
            }
            let _ = (
                request.group_by,
                request.sort_by,
                request.sort_by_secondary,
                request.sort_secondary_enabled,
                request.show_group_labels,
                request.show_text_labels,
                request.fit,
            );
        }
        RequestShape::ObjectPreloadStart => {
            let request: ObjectPreloadStartRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if !matches!(request.mode.as_str(), "full_geometry" | "centroid_points") {
                return Err(ControlError::invalid_params(
                    method,
                    "mode must be full_geometry or centroid_points",
                ));
            }
            let _ = request.lazy_properties;
        }
        RequestShape::DeepLinkUri => {
            let request: DeepLinkUriRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if request.url.trim().is_empty() {
                return Err(ControlError::invalid_params(
                    method,
                    "url must not be empty",
                ));
            }
            crate::deep_link::DeepLinkRequest::parse_arg(&request.url)
                .map_err(|error| ControlError::invalid_params(method, error.to_string()))?
                .ok_or_else(|| {
                    ControlError::invalid_params(method, "url must use the odon: scheme")
                })?;
        }
        RequestShape::DeepLinkGenerate => {
            let request: DeepLinkGenerateRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if request
                .roi
                .as_deref()
                .is_some_and(|roi| roi.trim().is_empty())
            {
                return Err(ControlError::invalid_params(
                    method,
                    "roi must not be empty",
                ));
            }
            let _ = (request.request, request.include_project);
        }
        RequestShape::DeepLinkApply => {
            let request: DeepLinkApplyRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if request.url.is_some() == request.request.is_some() {
                return Err(ControlError::invalid_params(
                    method,
                    "provide exactly one of url or request",
                ));
            }
            if let Some(url) = request.url {
                crate::deep_link::DeepLinkRequest::parse_arg(&url)
                    .map_err(|error| ControlError::invalid_params(method, error.to_string()))?
                    .ok_or_else(|| {
                        ControlError::invalid_params(method, "url must use the odon: scheme")
                    })?;
            }
        }
        RequestShape::Object => {}
    }
    Ok(())
}
