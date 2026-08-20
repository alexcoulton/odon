use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::{ControlError, ControlErrorKind};
use crate::control::registry;

pub const JSONRPC_VERSION: &str = "2.0";
pub const PROTOCOL_VERSION: u32 = 1;

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    #[serde(default)]
    pub id: Option<Value>,
    pub method: String,
    #[serde(default = "empty_params")]
    pub params: Value,
}

fn empty_params() -> Value {
    json!({})
}

impl JsonRpcRequest {
    pub fn validate(&self) -> Result<(), ControlError> {
        if self.jsonrpc != JSONRPC_VERSION {
            return Err(ControlError::new(
                ControlErrorKind::InvalidRequest,
                "jsonrpc must be '2.0'",
            ));
        }
        if self.method.trim().is_empty() {
            return Err(ControlError::new(
                ControlErrorKind::InvalidRequest,
                "method must not be empty",
            ));
        }
        if self
            .id
            .as_ref()
            .is_some_and(|id| !id.is_null() && !id.is_string() && !id.is_number())
        {
            return Err(ControlError::new(
                ControlErrorKind::InvalidRequest,
                "id must be a string, number, or null",
            ));
        }
        if !self.params.is_object() && !self.params.is_null() {
            return Err(ControlError::new(
                ControlErrorKind::InvalidParams,
                "params must be an object",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ClientInfo {
    pub name: String,
    pub version: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HelloRequest {
    #[serde(default)]
    pub token: Option<String>,
    pub client: ClientInfo,
    pub protocol_versions: Vec<u32>,
}

#[derive(Debug, Clone)]
pub struct HelloServerInfo {
    pub instance_id: String,
    pub session_id: String,
    pub expected_token: Option<String>,
    pub max_inline_payload_bytes: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct HelloResponse {
    pub protocol_version: u32,
    pub app_name: &'static str,
    pub app_version: &'static str,
    pub control_api_version: &'static str,
    pub instance_id: String,
    pub session_id: String,
    pub capabilities: Vec<String>,
    pub max_inline_payload_bytes: u64,
    pub permission_policy: &'static str,
    pub client: ClientInfo,
}

impl HelloResponse {
    pub fn negotiate(params: Value, server: &HelloServerInfo) -> Result<Self, ControlError> {
        let request: HelloRequest = serde_json::from_value(params).map_err(|error| {
            ControlError::invalid_params("system.hello", format!("invalid hello request: {error}"))
        })?;
        if request.client.name.trim().is_empty() || request.client.version.trim().is_empty() {
            return Err(ControlError::invalid_params(
                "system.hello",
                "client name and version must not be empty",
            ));
        }
        if let Some(expected) = server.expected_token.as_deref()
            && !request
                .token
                .as_deref()
                .is_some_and(|actual| constant_time_eq(actual.as_bytes(), expected.as_bytes()))
        {
            return Err(ControlError::new(
                ControlErrorKind::AuthenticationFailed,
                "invalid Odon instance token",
            ));
        }
        if !request.protocol_versions.contains(&PROTOCOL_VERSION) {
            return Err(ControlError::new(
                ControlErrorKind::IncompatibleProtocol,
                "client and Odon do not share a control protocol version",
            )
            .with_data(json!({
                "client_protocol_versions": request.protocol_versions,
                "server_protocol_versions": [PROTOCOL_VERSION],
            })));
        }
        Ok(Self {
            protocol_version: PROTOCOL_VERSION,
            app_name: "odon",
            app_version: env!("CARGO_PKG_VERSION"),
            control_api_version: "0.1.0",
            instance_id: server.instance_id.clone(),
            session_id: server.session_id.clone(),
            capabilities: registry::capabilities(),
            max_inline_payload_bytes: server.max_inline_payload_bytes,
            permission_policy: "local_authenticated_standard",
            client: request.client,
        })
    }
}

fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    left.iter()
        .zip(right)
        .fold(0u8, |difference, (left, right)| difference | (left ^ right))
        == 0
}

pub fn json_rpc_result(id: Value, result: Value) -> Value {
    json!({
        "jsonrpc": JSONRPC_VERSION,
        "id": id,
        "result": result,
    })
}

pub fn json_rpc_error(id: Value, error: &ControlError) -> Value {
    json!({
        "jsonrpc": JSONRPC_VERSION,
        "id": id,
        "error": error.to_json_rpc_error(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hello_negotiates_protocol_and_capabilities() {
        let response = HelloResponse::negotiate(
            json!({
                "token": "secret",
                "client": {"name": "test-client", "version": "1.2.3"},
                "protocol_versions": [1]
            }),
            &server_info(Some("secret")),
        )
        .expect("compatible handshake");

        assert_eq!(response.protocol_version, 1);
        assert_eq!(response.client.name, "test-client");
        assert_eq!(response.instance_id, "instance-test");
        assert!(response.capabilities.contains(&"viewer.read".to_string()));
    }

    #[test]
    fn hello_rejects_incompatible_protocols() {
        let error = HelloResponse::negotiate(
            json!({
                "client": {"name": "test-client", "version": "1.2.3"},
                "protocol_versions": [99]
            }),
            &server_info(None),
        )
        .expect_err("incompatible handshake");

        assert_eq!(error.kind, ControlErrorKind::IncompatibleProtocol);
        assert_eq!(error.kind.json_rpc_code(), -32002);
    }

    #[test]
    fn hello_rejects_missing_or_wrong_authentication_token() {
        for token in [None, Some("wrong")] {
            let error = HelloResponse::negotiate(
                json!({
                    "token": token,
                    "client": {"name": "test-client", "version": "1.2.3"},
                    "protocol_versions": [1]
                }),
                &server_info(Some("secret")),
            )
            .expect_err("authentication must fail");
            assert_eq!(error.kind, ControlErrorKind::AuthenticationFailed);
        }
    }

    fn server_info(token: Option<&str>) -> HelloServerInfo {
        HelloServerInfo {
            instance_id: "instance-test".to_string(),
            session_id: "session-test".to_string(),
            expected_token: token.map(str::to_string),
            max_inline_payload_bytes: 1_048_576,
        }
    }
}
