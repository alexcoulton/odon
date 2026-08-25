//! Transport-independent control contracts for Odon automation clients.

pub mod actor;
mod command;
pub mod discovery;
mod error;
mod events;
mod protocol;
pub mod registry;
mod request;
mod resources;
mod surface;
mod tasks;
mod ui;

pub use command::ControlCommand;
pub use error::{ControlError, ControlErrorKind};
pub use events::{EventEnvelope, EventHub};
pub use protocol::{
    ClientInfo, HelloRequest, HelloResponse, HelloServerInfo, JSONRPC_VERSION, JsonRpcRequest,
    PROTOCOL_VERSION, json_rpc_error, json_rpc_result,
};
pub use request::OdonControlRequest;
pub use resources::{
    CoordinateSpace, DataResourceSnapshot, LayerSnapshot, Ownership, ResourceRegistry,
};
pub use surface::{
    ApplicationSurfaceEntry, ApplicationSurfaceManifest, SurfaceStatus, application_surface,
    application_surface_json,
};
pub use tasks::{TaskRegistry, TaskServiceHandle, TaskSnapshot, TaskState};
pub use ui::{
    Component, ContributionSnapshot, DisconnectPolicy, ExtensionCommandContext, ExtensionSnapshot,
    UiAction, UiExtensionCleanup, UiRegistry, UiSessionCleanup,
};
