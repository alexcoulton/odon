use std::io::{BufRead, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::Duration;

use crate::deep_link::DeepLinkRequest;

const DEFAULT_ADDR: &str = "127.0.0.1:17871";
const MESSAGE_PREFIX: &str = "ODON_DEEP_LINK_V1\t";
const ACK: &str = "OK\n";

pub fn send_to_running(raw_url: &str) -> bool {
    let Ok(addr) = DEFAULT_ADDR.parse::<SocketAddr>() else {
        return false;
    };
    let Ok(mut stream) = TcpStream::connect_timeout(&addr, Duration::from_millis(250)) else {
        return false;
    };

    let _ = stream.set_write_timeout(Some(Duration::from_millis(250)));
    let _ = stream.set_read_timeout(Some(Duration::from_millis(500)));
    if stream.write_all(MESSAGE_PREFIX.as_bytes()).is_err()
        || stream.write_all(raw_url.as_bytes()).is_err()
        || stream.write_all(b"\n").is_err()
        || stream.flush().is_err()
    {
        return false;
    }

    let mut ack = String::new();
    let mut reader = std::io::BufReader::new(stream);
    reader.read_line(&mut ack).is_ok() && ack == ACK
}

pub fn spawn_listener() -> Option<Receiver<DeepLinkRequest>> {
    let listener = match TcpListener::bind(DEFAULT_ADDR) {
        Ok(listener) => listener,
        Err(_) => {
            if send_to_running("odon://open") {
                return None;
            }
            return None;
        }
    };

    let (tx, rx) = mpsc::channel();
    std::thread::Builder::new()
        .name("odon-deep-link-ipc".to_string())
        .spawn(move || {
            for stream in listener.incoming().flatten() {
                handle_client(stream, &tx);
            }
        })
        .ok()?;

    Some(rx)
}

fn handle_client(mut stream: TcpStream, tx: &Sender<DeepLinkRequest>) {
    let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
    let mut raw = String::new();
    {
        let Ok(reader_stream) = stream.try_clone() else {
            return;
        };
        let mut reader = std::io::BufReader::new(reader_stream);
        if reader.read_line(&mut raw).is_err() {
            return;
        }
    }

    let Some(raw_url) = raw.trim().strip_prefix(MESSAGE_PREFIX) else {
        return;
    };

    if !raw_url.is_empty()
        && raw_url != "odon://open"
        && let Ok(Some(request)) = DeepLinkRequest::parse_arg(raw_url)
    {
        let _ = tx.send(request);
    }

    let _ = stream.write_all(ACK.as_bytes());
    let _ = stream.flush();
}
