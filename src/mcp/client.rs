use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;

use serde_json::{Value, json};

use crate::mcp::bridge::DEFAULT_ADDR;

pub fn call_running_odon(method: &str, params: Value) -> anyhow::Result<Value> {
    let mut stream = TcpStream::connect(DEFAULT_ADDR)?;
    let request = json!({
        "method": method,
        "params": params,
    });
    writeln!(stream, "{}", request)?;
    stream.flush()?;
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    reader.read_line(&mut line)?;
    let response: Value = serde_json::from_str(line.trim())?;
    if response.get("ok").and_then(Value::as_bool) == Some(true) {
        Ok(response.get("result").cloned().unwrap_or(Value::Null))
    } else {
        let error = response
            .get("error")
            .and_then(Value::as_str)
            .unwrap_or("unknown Odon bridge error");
        anyhow::bail!("{error}")
    }
}
