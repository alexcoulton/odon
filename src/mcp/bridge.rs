use std::io::{BufRead, BufReader, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::thread;
use std::time::Duration;

use crossbeam_channel::{Receiver, Sender};
use eframe::egui;
use serde_json::{Value, json};

pub const DEFAULT_ADDR: &str = "127.0.0.1:17870";

#[derive(Debug)]
pub struct OdonControlBridge {
    rx: Receiver<OdonControlRequest>,
    local_addr: SocketAddr,
}

#[derive(Debug)]
pub struct OdonControlRequest {
    pub method: String,
    pub params: Value,
    pub reply: Sender<Value>,
}

impl OdonControlBridge {
    pub fn spawn_default(ctx: egui::Context) -> anyhow::Result<Self> {
        Self::spawn(DEFAULT_ADDR, ctx)
    }

    pub fn spawn(addr: &str, ctx: egui::Context) -> anyhow::Result<Self> {
        let listener = TcpListener::bind(addr)?;
        let local_addr = listener.local_addr()?;
        listener.set_nonblocking(false)?;
        let (tx, rx) = crossbeam_channel::unbounded::<OdonControlRequest>();
        let addr = addr.to_string();
        thread::Builder::new()
            .name("odon-control-bridge".to_string())
            .spawn(move || serve_control_bridge(listener, tx, addr, ctx))
            .map_err(anyhow::Error::from)?;
        Ok(Self { rx, local_addr })
    }

    pub fn try_recv(&self) -> Result<OdonControlRequest, crossbeam_channel::TryRecvError> {
        self.rx.try_recv()
    }

    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }
}

fn serve_control_bridge(
    listener: TcpListener,
    tx: Sender<OdonControlRequest>,
    addr: String,
    ctx: egui::Context,
) {
    eprintln!("odon control bridge listening on {addr}");
    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let tx = tx.clone();
                let ctx = ctx.clone();
                let _ = thread::Builder::new()
                    .name("odon-control-client".to_string())
                    .spawn(move || handle_control_client(stream, tx, ctx));
            }
            Err(err) => eprintln!("odon control bridge accept failed: {err}"),
        }
    }
}

fn handle_control_client(stream: TcpStream, tx: Sender<OdonControlRequest>, ctx: egui::Context) {
    let Ok(write_stream) = stream.try_clone() else {
        return;
    };
    let mut writer = write_stream;
    let reader = BufReader::new(stream);
    for line in reader.lines() {
        let response = match line {
            Ok(line) => handle_control_line(&line, &tx, &ctx),
            Err(err) => json!({"ok": false, "error": format!("read failed: {err}")}),
        };
        if writeln!(writer, "{}", response).is_err() {
            return;
        }
        let _ = writer.flush();
    }
}

fn handle_control_line(line: &str, tx: &Sender<OdonControlRequest>, ctx: &egui::Context) -> Value {
    let value = match serde_json::from_str::<Value>(line) {
        Ok(value) => value,
        Err(err) => return json!({"ok": false, "error": format!("invalid JSON: {err}")}),
    };
    let Some(method) = value.get("method").and_then(Value::as_str) else {
        return json!({"ok": false, "error": "missing method"});
    };
    let params = value.get("params").cloned().unwrap_or(Value::Null);
    let (reply_tx, reply_rx) = crossbeam_channel::bounded::<Value>(1);
    if tx
        .send(OdonControlRequest {
            method: method.to_string(),
            params,
            reply: reply_tx,
        })
        .is_err()
    {
        return json!({"ok": false, "error": "Odon app is not accepting control requests"});
    }
    ctx.request_repaint();
    match reply_rx.recv_timeout(Duration::from_secs(5)) {
        Ok(value) => json!({"ok": true, "result": value}),
        Err(_) => json!({"ok": false, "error": "Odon app did not respond in time"}),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    fn read_json(reader: &mut BufReader<TcpStream>) -> Value {
        let mut line = String::new();
        reader.read_line(&mut line).expect("read bridge response");
        serde_json::from_str(line.trim()).expect("parse bridge response")
    }

    #[test]
    fn tcp_bridge_validates_envelopes_and_roundtrips_app_replies() {
        let bridge = OdonControlBridge::spawn("127.0.0.1:0", egui::Context::default())
            .expect("spawn bridge on ephemeral port");
        let mut stream = TcpStream::connect(bridge.local_addr()).expect("connect bridge client");
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .expect("set read timeout");
        let mut reader = BufReader::new(stream.try_clone().expect("clone bridge socket"));

        writeln!(stream, "{{").expect("write malformed JSON");
        let malformed = read_json(&mut reader);
        assert_eq!(malformed["ok"], false);
        assert!(
            malformed["error"]
                .as_str()
                .unwrap()
                .contains("invalid JSON")
        );

        writeln!(stream, "{}", json!({"params": {}})).expect("write missing method");
        let missing = read_json(&mut reader);
        assert_eq!(missing, json!({"ok": false, "error": "missing method"}));

        writeln!(
            stream,
            "{}",
            json!({"method": "set_camera", "params": {"center_x": 12.5}})
        )
        .expect("write valid request");
        stream.flush().expect("flush valid request");
        let deadline = Instant::now() + Duration::from_secs(2);
        let request = loop {
            match bridge.try_recv() {
                Ok(request) => break request,
                Err(crossbeam_channel::TryRecvError::Empty) if Instant::now() < deadline => {
                    std::thread::yield_now();
                }
                Err(error) => panic!("bridge request not delivered: {error}"),
            }
        };
        assert_eq!(request.method, "set_camera");
        assert_eq!(request.params["center_x"], 12.5);
        request
            .reply
            .send(json!({"center_world_lvl0": [12.5, 0.0]}))
            .expect("reply from app");
        let response = read_json(&mut reader);
        assert_eq!(response["ok"], true);
        assert_eq!(response["result"]["center_world_lvl0"], json!([12.5, 0.0]));
    }
}
