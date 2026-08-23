use super::*;
use std::ops::{Deref, DerefMut};

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

pub(super) struct ActorAppFixture {
    app: OmeZarrViewerApp,
    model: AppModel,
}

impl ActorAppFixture {
    pub(super) fn new(mut app: OmeZarrViewerApp) -> Self {
        let mut model = AppModel::project();
        model
            .bootstrap_dataset(&app.dataset)
            .expect("actor fixture installs the dataset");
        let projection = model
            .render_workspace_snapshot()
            .expect("installed dataset has an actor projection");
        app.apply_control_actor_workspace_projection(&projection)
            .expect("actor fixture starts from the canonical projection");
        Self { app, model }
    }

    pub(super) fn actor_command(
        &mut self,
        method: &str,
        params: serde_json::Value,
    ) -> serde_json::Value {
        let response = actor_call(&mut self.model, method, params);
        self.apply_latest_projection();
        response
    }

    pub(super) fn try_actor_command(
        &mut self,
        method: &str,
        params: serde_json::Value,
    ) -> Result<serde_json::Value, odon::control::ControlError> {
        let result = self
            .model
            .dispatch(method, &params)
            .unwrap_or_else(|| panic!("{method} should be actor-owned"))
            .map(|outcome| outcome.response);
        if result.is_ok() {
            self.apply_latest_projection();
        }
        result
    }

    pub(super) fn actor_query(
        &mut self,
        method: &str,
        params: serde_json::Value,
    ) -> serde_json::Value {
        actor_call(&mut self.model, method, params)
    }

    pub(super) fn apply_latest_projection(&mut self) {
        let projection = self
            .model
            .render_workspace_snapshot()
            .expect("actor fixture has a render workspace");
        self.app
            .apply_control_actor_workspace_projection(&projection)
            .expect("actor fixture projection applies");
    }

    pub(super) fn install_object_resource(&mut self, resource: odon::model::ControlObjectResource) {
        let source = resource.source.display().to_string();
        let (document_generation, resource_generation) =
            self.model.begin_object_resource_load(source);
        assert!(
            self.app
                .install_control_actor_object_resource(resource_generation, &resource),
            "renderer installs the actor-prepared object resource"
        );
        assert!(
            self.model.install_object_resource_for_generation(
                document_generation,
                resource_generation,
                std::sync::Arc::new(resource),
            ),
            "actor model installs the prepared object resource"
        );
        self.apply_latest_projection();
    }
}

impl Deref for ActorAppFixture {
    type Target = OmeZarrViewerApp;

    fn deref(&self) -> &Self::Target {
        &self.app
    }
}

impl DerefMut for ActorAppFixture {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.app
    }
}
