use std::io::{BufRead, Write};
use std::sync::mpsc::{self, Receiver};

use crate::deep_link::DeepLinkRequest;

#[cfg(unix)]
pub fn send_to_running(raw_url: &str) -> bool {
    use std::fs;
    use std::io::ErrorKind;
    use std::os::unix::net::UnixStream;

    match UnixStream::connect(socket_path()) {
        Ok(mut stream) => {
            let _ = stream.write_all(raw_url.as_bytes());
            let _ = stream.write_all(b"\n");
            true
        }
        Err(err) => {
            if matches!(
                err.kind(),
                ErrorKind::ConnectionRefused | ErrorKind::NotFound
            ) {
                let _ = fs::remove_file(socket_path());
            }
            false
        }
    }
}

#[cfg(not(unix))]
pub fn send_to_running(_raw_url: &str) -> bool {
    false
}

#[cfg(unix)]
pub fn spawn_listener() -> Option<Receiver<DeepLinkRequest>> {
    use std::fs;
    use std::io::ErrorKind;
    use std::os::unix::net::UnixListener;
    use std::thread;

    let path = socket_path();
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }

    let listener = match UnixListener::bind(&path) {
        Ok(listener) => listener,
        Err(err) if err.kind() == ErrorKind::AddrInUse => {
            if send_to_running("odon://open") {
                return None;
            }
            let _ = fs::remove_file(&path);
            UnixListener::bind(&path).ok()?
        }
        Err(_) => return None,
    };

    let (tx, rx) = mpsc::channel();
    thread::Builder::new()
        .name("odon-deep-link-ipc".to_string())
        .spawn(move || {
            for stream in listener.incoming().flatten() {
                let mut raw = String::new();
                let mut reader = std::io::BufReader::new(stream);
                if reader.read_line(&mut raw).is_ok() {
                    let raw = raw.trim();
                    if raw.is_empty() || raw == "odon://open" {
                        continue;
                    }
                    if let Ok(Some(request)) = DeepLinkRequest::parse_arg(raw) {
                        let _ = tx.send(request);
                    }
                }
            }
        })
        .ok()?;

    Some(rx)
}

#[cfg(not(unix))]
pub fn spawn_listener() -> Option<Receiver<DeepLinkRequest>> {
    None
}

#[cfg(unix)]
fn socket_path() -> std::path::PathBuf {
    let user = std::env::var("USER")
        .or_else(|_| std::env::var("USERNAME"))
        .unwrap_or_else(|_| "user".to_string());
    std::path::PathBuf::from(format!("/tmp/odon-{user}.deeplink.sock"))
}
