use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::thread;
use std::time::Duration;

use crossbeam_channel::{Receiver, Sender};
use eframe::egui;
use serde_json::{Value, json};

pub const DEFAULT_ADDR: &str = "127.0.0.1:17870";

#[derive(Debug)]
pub struct OdonControlBridge {
    rx: Receiver<OdonControlRequest>,
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
        listener.set_nonblocking(false)?;
        let (tx, rx) = crossbeam_channel::unbounded::<OdonControlRequest>();
        let addr = addr.to_string();
        thread::Builder::new()
            .name("odon-control-bridge".to_string())
            .spawn(move || serve_control_bridge(listener, tx, addr, ctx))
            .map_err(anyhow::Error::from)?;
        Ok(Self { rx })
    }

    pub fn try_recv(&self) -> Result<OdonControlRequest, crossbeam_channel::TryRecvError> {
        self.rx.try_recv()
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
