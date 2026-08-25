//! Actor-facing command and platform-menu resources.

use super::*;

impl AppModel {
    pub(crate) fn command_surface_projection(&self) -> Value {
        self.command_surface
            .evaluated_projection(&self.command_evaluation_context(&[], true))
    }

    pub(super) fn command_surface_schema(&self) -> Value {
        self.command_surface.schema()
    }

    pub(super) fn command_list(&self) -> Value {
        self.command_surface
            .evaluated_commands_snapshot(&self.command_evaluation_context(&[], true))
    }

    pub(crate) fn command_list_for_session(&self, capabilities: &[String]) -> Value {
        self.command_surface
            .evaluated_commands_snapshot(&self.command_evaluation_context(capabilities, false))
    }

    pub(super) fn platform_menu(&self) -> Value {
        self.command_surface.menu_snapshot()
    }

    pub(super) fn command_toolbar(&self) -> Value {
        self.command_surface.toolbar_snapshot()
    }

    pub(super) fn command_palette(&self) -> Value {
        self.command_surface.palette_snapshot()
    }

    pub(super) fn replace_platform_menu(&mut self, params: &Value) -> Result<Value, ControlError> {
        self.command_surface.replace_menu(params)
    }

    pub(super) fn replace_command_toolbar(
        &mut self,
        params: &Value,
    ) -> Result<Value, ControlError> {
        self.command_surface.replace_toolbar(params)
    }

    pub(super) fn replace_command_palette(
        &mut self,
        params: &Value,
    ) -> Result<Value, ControlError> {
        self.command_surface.replace_palette(params)
    }

    pub(crate) fn register_extension_command(
        &mut self,
        params: &Value,
        context: &crate::control::ExtensionCommandContext,
    ) -> Result<Value, ControlError> {
        self.command_surface
            .register_extension_command(params, context)
    }

    pub(crate) fn remove_extension_command(
        &mut self,
        params: &Value,
        context: &crate::control::ExtensionCommandContext,
    ) -> Result<Value, ControlError> {
        self.command_surface
            .remove_extension_command(params, context)
    }

    pub(crate) fn cleanup_extension_commands(
        &mut self,
        extensions: &[crate::control::UiExtensionCleanup],
    ) -> Value {
        self.command_surface.cleanup_extensions(extensions)
    }

    pub(crate) fn sync_extension_commands(
        &mut self,
        context: &crate::control::ExtensionCommandContext,
    ) -> Value {
        self.command_surface.sync_extension(context)
    }

    pub(crate) fn command_invocation(
        &self,
        params: &Value,
        capabilities: &[String],
        native: bool,
    ) -> Result<crate::model::CommandInvocation, ControlError> {
        self.command_surface.invocation(
            params,
            &self.command_evaluation_context(capabilities, native),
        )
    }

    fn command_evaluation_context(
        &self,
        capabilities: &[String],
        native: bool,
    ) -> crate::model::CommandEvaluationContext {
        let (objects, labels, masks, object_selection, scale_bar, left_panel, right_panel) = self
            .dataset
            .as_ref()
            .map_or((false, false, false, 0, false, false, false), |dataset| {
                let active = dataset.workspace.active();
                (
                    dataset.object_resource.is_some(),
                    dataset.label_resource.is_some(),
                    dataset
                        .masks
                        .projection_json()
                        .get("layers")
                        .and_then(Value::as_array)
                        .is_some_and(|layers| !layers.is_empty()),
                    dataset.object_selection.selection_count(),
                    active.state.show_scale_bar,
                    dataset.show_left_panel,
                    dataset.show_right_panel,
                )
            });
        let mosaic_objects = self.mosaic.object_resources();
        let state = json!({
            "mode":self.mode().as_str(),
            "resources":{
                "project":self.project_initialized,
                "dataset":self.dataset.is_some(),
                "mosaic":self.mosaic.resource().is_some(),
                "objects":objects || !mosaic_objects.is_empty(),
                "labels":labels,
                "masks":masks,
                "gpu":self.renderer_gpu_available,
            },
            "selection":{
                "objects":{"count":object_selection.saturating_add(self.mosaic.selected_object_count())},
                "mosaic_items":{"count":self.mosaic.selected_item_count()},
            },
            "presentation":{
                "scale_bar":{"checked":scale_bar},
                "left_panel":{"visible":left_panel},
                "right_panel":{"visible":right_panel},
            },
        });
        if native {
            crate::model::CommandEvaluationContext::native(self.mode().as_str(), state)
        } else {
            crate::model::CommandEvaluationContext::session(
                self.mode().as_str(),
                capabilities.iter().cloned(),
                state,
            )
        }
    }
}
