use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;

use serde_json::{Value, json};

pub fn call_running_odon(method: &str, params: Value) -> anyhow::Result<Value> {
    let requested_instance = std::env::var("ODON_INSTANCE_ID").ok();
    let instance = crate::control::discovery::select_instance(requested_instance.as_deref())?;
    let mut stream = TcpStream::connect(instance.socket_addr()?)?;
    let hello = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "system.hello",
        "params": {
            "token": instance.token,
            "client": {
                "name": "odon-mcp",
                "version": env!("CARGO_PKG_VERSION")
            },
            "protocol_versions": [1]
        }
    });
    writeln!(stream, "{}", hello)?;
    stream.flush()?;
    let mut reader = BufReader::new(stream.try_clone()?);
    let hello_response = read_response(&mut reader)?;
    response_result(hello_response)?;

    let request = json!({
        "jsonrpc": "2.0",
        "id": 2,
        "method": method,
        "params": params,
    });
    writeln!(stream, "{}", request)?;
    stream.flush()?;
    response_result(read_response(&mut reader)?)
}

fn read_response(reader: &mut BufReader<TcpStream>) -> anyhow::Result<Value> {
    let mut line = String::new();
    reader.read_line(&mut line)?;
    if line.trim().is_empty() {
        anyhow::bail!("Odon control connection closed without a response");
    }
    Ok(serde_json::from_str(line.trim())?)
}

fn response_result(response: Value) -> anyhow::Result<Value> {
    if let Some(error) = response.get("error") {
        let message = error
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or("unknown Odon control error");
        let kind = error
            .get("data")
            .and_then(|data| data.get("kind"))
            .and_then(Value::as_str)
            .unwrap_or("CONTROL_ERROR");
        anyhow::bail!("{kind}: {message}");
    }
    response
        .get("result")
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("Odon returned an invalid JSON-RPC response"))
}
