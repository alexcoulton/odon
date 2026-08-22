use super::*;

pub fn execution_diagnostics(diagnostics: &ActorDiagnostics) -> Value {
    let method_routes = crate::control::registry::METHODS
        .iter()
        .map(|descriptor| {
            (
                descriptor.name.to_string(),
                Value::String(
                    crate::control::registry::execution_route_summary(descriptor).to_string(),
                ),
            )
        })
        .chain(
            crate::control::registry::PROTOCOL_METHODS
                .iter()
                .map(|descriptor| {
                    (
                        descriptor.0.to_string(),
                        Value::String(
                            if MIGRATED_METHODS.contains(&descriptor.0) {
                                "actor"
                            } else {
                                "control_service"
                            }
                            .to_string(),
                        ),
                    )
                }),
        )
        .collect::<serde_json::Map<_, _>>();
    let route_matrix = crate::control::registry::METHODS
        .iter()
        .map(|descriptor| {
            (
                descriptor.name.to_string(),
                crate::control::registry::execution_route_json(descriptor),
            )
        })
        .collect::<serde_json::Map<_, _>>();
    json!({
        "actor_methods": MIGRATED_METHODS,
        "compatibility_fallback": "legacy_ui",
        "model_commands_require_ui_frame": false,
        "presentation_is_asynchronous": true,
        "method_routes": method_routes,
        "route_matrix":route_matrix,
        "metrics": diagnostics.snapshot(),
    })
}
