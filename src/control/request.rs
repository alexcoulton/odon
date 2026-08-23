//! Transport-independent command envelope consumed by the control actor.

use std::sync::Arc;

use crossbeam_channel::Sender;
use serde_json::Value;

use super::{ControlCommand, ControlError, EventHub, TaskRegistry};

/// One command submitted to the canonical application actor.
///
/// Native UI, TCP, Python, and future transports all construct the same envelope. Transport
/// modules may supply session and request metadata, but they do not own command execution.
#[derive(Debug)]
pub struct OdonControlRequest {
    pub command: ControlCommand,
    pub reply: Sender<Result<Value, ControlError>>,
    pub session_id: String,
    pub request_id: Option<Value>,
    pub event_hub: Arc<EventHub>,
    pub task_registry: Arc<TaskRegistry>,
    pub task_id: Option<String>,
}
