use std::io::{BufRead, BufReader, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use crossbeam_channel::Sender;
use eframe::egui;
use serde_json::{Value, json};

use crate::control::registry;
use crate::control::{
    ControlCommand, ControlError, ControlErrorKind, EventHub, HelloResponse, HelloServerInfo,
    JsonRpcRequest, ResourceRegistry, TaskRegistry, TaskState, UiRegistry, json_rpc_error,
    json_rpc_result,
};

pub use super::runtime::{DEFAULT_ADDR, OdonControlBridge, OdonControlRuntime};
pub use crate::control::OdonControlRequest;

mod dispatch;
mod services;
mod tasks;
mod transport;
mod waits;

use dispatch::{dispatch_to_app, handle_control_line};
use services::{
    cancel_task, forget_task, get_task, list_extension_layouts, list_tasks, patch_ui_values,
    register_contribution, register_extension, register_extension_layout, remove_contribution,
    remove_extension, remove_extension_layout, set_extension_readiness, subscribe_events,
    unsubscribe_events,
};
use tasks::{run_batch, start_task};
pub(super) use transport::spawn_control_server;
#[cfg(test)]
use waits::application_state_is_ready;
use waits::{
    wait_for_application_ready, wait_for_control_operation, wait_for_deep_link_application,
    wait_for_mosaic_object_load, wait_for_object_property_load, wait_for_object_source_load,
    wait_for_output_path, wait_for_project_object_preload,
};

const MAX_INLINE_PAYLOAD_BYTES: u64 = 1_048_576;

#[derive(Debug)]
struct ControlServerIdentity {
    instance_id: String,
    expected_token: Option<String>,
    allow_legacy: bool,
    event_hub: Arc<EventHub>,
    task_registry: Arc<TaskRegistry>,
    task_service: crate::control::TaskServiceHandle,
    resource_registry: Arc<ResourceRegistry>,
    ui_registry: Arc<UiRegistry>,
    actor_diagnostics: Arc<crate::control::actor::ActorDiagnostics>,
}

pub(super) struct ControlServerServices {
    pub(super) event_hub: Arc<EventHub>,
    pub(super) task_registry: Arc<TaskRegistry>,
    pub(super) task_service: crate::control::TaskServiceHandle,
    pub(super) resource_registry: Arc<ResourceRegistry>,
    pub(super) ui_registry: Arc<UiRegistry>,
    pub(super) actor_diagnostics: Arc<crate::control::actor::ActorDiagnostics>,
}

pub(super) struct ControlServerPublication {
    pub(super) local_addr: SocketAddr,
    pub(super) manifest: Option<crate::control::discovery::InstanceManifestGuard>,
}

#[derive(Clone)]
struct ConnectionState {
    hello_complete: bool,
    close_after_response: bool,
    allow_legacy: bool,
    hello_server: HelloServerInfo,
    event_hub: Arc<EventHub>,
    task_registry: Arc<TaskRegistry>,
    task_service: crate::control::TaskServiceHandle,
    resource_registry: Arc<ResourceRegistry>,
    ui_registry: Arc<UiRegistry>,
    actor_diagnostics: Arc<crate::control::actor::ActorDiagnostics>,
}

impl ConnectionState {
    fn new(identity: &ControlServerIdentity, _outbound: Sender<Value>) -> anyhow::Result<Self> {
        Ok(Self {
            hello_complete: false,
            close_after_response: false,
            allow_legacy: identity.allow_legacy,
            hello_server: HelloServerInfo {
                instance_id: identity.instance_id.clone(),
                session_id: crate::control::discovery::random_uuid_like()?,
                expected_token: identity.expected_token.clone(),
                max_inline_payload_bytes: MAX_INLINE_PAYLOAD_BYTES,
            },
            event_hub: Arc::clone(&identity.event_hub),
            task_registry: Arc::clone(&identity.task_registry),
            task_service: identity.task_service.clone(),
            resource_registry: Arc::clone(&identity.resource_registry),
            ui_registry: Arc::clone(&identity.ui_registry),
            actor_diagnostics: Arc::clone(&identity.actor_diagnostics),
        })
    }

    fn unauthenticated_test() -> Self {
        let (outbound, _rx) = crossbeam_channel::bounded(8);
        let event_hub = EventHub::shared();
        let task_registry = TaskRegistry::shared(Arc::clone(&event_hub));
        let identity = ControlServerIdentity {
            instance_id: "test-instance".to_string(),
            expected_token: None,
            allow_legacy: true,
            event_hub: Arc::clone(&event_hub),
            task_service: crate::control::TaskServiceHandle::spawn_standalone(Arc::clone(
                &task_registry,
            )),
            task_registry,
            resource_registry: ResourceRegistry::shared(Arc::clone(&event_hub)),
            ui_registry: UiRegistry::shared(Arc::clone(&event_hub)),
            actor_diagnostics: crate::control::actor::ActorDiagnostics::shared(),
        };
        let state = Self::new(&identity, outbound.clone()).expect("create test connection state");
        identity
            .event_hub
            .register(state.hello_server.session_id.clone(), outbound);
        state
    }
}

#[cfg(test)]
#[path = "bridge/tests.rs"]
mod tests;
