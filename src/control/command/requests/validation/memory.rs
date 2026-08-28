//! Memory and image-tile policy request validation.

use super::super::{ImageTileCacheRequest, TileLoadingRequest};
use crate::control::ControlError;

const MIN_CACHE_BYTES: u64 = 128 * 1024 * 1024;
const MAX_CACHE_BYTES: u64 = 4 * 1024 * 1024 * 1024;

fn valid_cache_mode(value: &str) -> bool {
    matches!(
        value,
        "automatic" | "conservative" | "balanced" | "performance" | "custom"
    )
}

fn valid_channel_history(value: &str) -> bool {
    matches!(value, "automatic" | "current_only" | "current_and_previous")
}

pub(super) fn validate_image_tile_cache(
    method: &str,
    cache: ImageTileCacheRequest,
) -> Result<(), ControlError> {
    if cache
        .mode
        .as_deref()
        .is_some_and(|value| !valid_cache_mode(value))
    {
        return Err(ControlError::invalid_params(
            method,
            "unknown image tile cache mode",
        ));
    }
    if cache
        .custom_budget_bytes
        .is_some_and(|value| !(MIN_CACHE_BYTES..=MAX_CACHE_BYTES).contains(&value))
    {
        return Err(ControlError::invalid_params(
            method,
            "image tile cache budget must be between 128 MiB and 4 GiB",
        ));
    }
    if cache
        .channel_history
        .as_deref()
        .is_some_and(|value| !valid_channel_history(value))
    {
        return Err(ControlError::invalid_params(
            method,
            "unknown image tile channel history",
        ));
    }
    Ok(())
}

pub(super) fn validate_tile_loading(
    method: &str,
    request: TileLoadingRequest,
) -> Result<(), ControlError> {
    if request
        .workers
        .is_some_and(|value| !(1..=12).contains(&value))
    {
        return Err(ControlError::invalid_params(
            method,
            "workers must be from 1 to 12",
        ));
    }
    if request
        .prefetch_mode
        .as_deref()
        .is_some_and(|value| !matches!(value, "off" | "target_halo" | "target_and_finer_halo"))
    {
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
    if request
        .cache_mode
        .as_deref()
        .is_some_and(|value| !valid_cache_mode(value))
    {
        return Err(ControlError::invalid_params(method, "unknown cache_mode"));
    }
    if request
        .cache_budget_bytes
        .is_some_and(|value| !(MIN_CACHE_BYTES..=MAX_CACHE_BYTES).contains(&value))
    {
        return Err(ControlError::invalid_params(
            method,
            "cache_budget_bytes must be between 128 MiB and 4 GiB",
        ));
    }
    if request
        .channel_history
        .as_deref()
        .is_some_and(|value| !valid_channel_history(value))
    {
        return Err(ControlError::invalid_params(
            method,
            "unknown channel_history",
        ));
    }
    let _ = request.prefer_pinned_finer_levels;
    Ok(())
}
