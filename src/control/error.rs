use serde::Serialize;
use serde_json::{Value, json};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ControlErrorKind {
    ParseError,
    InvalidRequest,
    MethodNotFound,
    InvalidParams,
    Application,
    AuthenticationFailed,
    HandshakeRequired,
    IncompatibleProtocol,
    NotReady,
    Unsupported,
    Conflict,
    Cancelled,
    ResourceNotFound,
    PermissionDenied,
    WrongMode,
    ResourceLimit,
    Internal,
    Timeout,
}

impl ControlErrorKind {
    pub fn json_rpc_code(self) -> i64 {
        match self {
            Self::ParseError => -32700,
            Self::InvalidRequest => -32600,
            Self::MethodNotFound => -32601,
            Self::InvalidParams => -32602,
            Self::Application => -32000,
            Self::AuthenticationFailed => -32003,
            Self::HandshakeRequired => -32001,
            Self::IncompatibleProtocol => -32002,
            Self::NotReady => -32010,
            Self::Unsupported => -32011,
            Self::Conflict => -32013,
            Self::Cancelled => -32014,
            Self::ResourceNotFound => -32015,
            Self::PermissionDenied => -32016,
            Self::WrongMode => -32017,
            Self::ResourceLimit => -32018,
            Self::Timeout => -32012,
            Self::Internal => -32603,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ControlError {
    pub kind: ControlErrorKind,
    pub message: String,
    pub data: Option<Value>,
}

impl ControlError {
    pub fn new(kind: ControlErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            data: None,
        }
    }

    pub fn with_data(mut self, data: Value) -> Self {
        self.data = Some(data);
        self
    }

    pub fn invalid_params(method: &str, message: impl Into<String>) -> Self {
        Self::new(ControlErrorKind::InvalidParams, message).with_data(json!({
            "method": method,
        }))
    }

    pub fn to_json_rpc_error(&self) -> Value {
        let mut data = self.data.clone().unwrap_or_else(|| json!({}));
        if let Some(object) = data.as_object_mut() {
            object.insert("kind".to_string(), json!(self.kind));
        }
        json!({
            "code": self.kind.json_rpc_code(),
            "message": self.message,
            "data": data,
        })
    }
}

impl std::fmt::Display for ControlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for ControlError {}
