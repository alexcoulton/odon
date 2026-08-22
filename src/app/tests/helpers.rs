use super::*;

pub(super) fn actor_call(
    model: &mut AppModel,
    method: &str,
    params: serde_json::Value,
) -> serde_json::Value {
    model
        .dispatch(method, &params)
        .unwrap_or_else(|| panic!("{method} should be actor-owned"))
        .unwrap_or_else(|error| panic!("{method} failed: {error}"))
        .response
}

pub(super) fn workspace_topology(workspace: &serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "revision": workspace["revision"],
        "layout": workspace["layout"],
        "ratio": workspace["ratio"],
        "active_viewport_id": workspace["active_viewport_id"],
        "max_viewports": workspace["max_viewports"],
        "links": workspace["links"],
        "viewports": workspace["viewports"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|viewport| serde_json::json!({
                "viewport_id": viewport["viewport_id"],
                "title": viewport["title"],
                "active": viewport["active"],
                "navigation_revision": viewport["navigation_revision"],
                "presentation_revision": viewport["presentation_revision"],
                "plane": viewport["plane"],
                "channels": viewport["channels"],
                "rendering": viewport["rendering"],
            }))
            .collect::<Vec<_>>(),
    })
}
