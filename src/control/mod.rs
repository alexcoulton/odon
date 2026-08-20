//! Transport-independent control contracts for Odon automation clients.

mod command;
pub mod discovery;
mod error;
mod events;
mod protocol;
pub mod registry;
mod resources;
mod tasks;
mod ui;

pub use command::ControlCommand;
pub use error::{ControlError, ControlErrorKind};
pub use events::{EventEnvelope, EventHub};
pub use protocol::{
    ClientInfo, HelloRequest, HelloResponse, HelloServerInfo, JSONRPC_VERSION, JsonRpcRequest,
    PROTOCOL_VERSION, json_rpc_error, json_rpc_result,
};
pub use resources::{
    CoordinateSpace, DataResourceSnapshot, LayerSnapshot, Ownership, ResourceRegistry,
};
pub use tasks::{TaskRegistry, TaskSnapshot, TaskState};
pub use ui::{Component, ContributionSnapshot, ExtensionSnapshot, UiAction, UiRegistry};
