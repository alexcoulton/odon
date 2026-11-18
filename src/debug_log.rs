use std::fs::{File, OpenOptions};
use std::io::Write;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    Error = 1,
    Warn = 2,
    Info = 3,
    Debug = 4,
    Trace = 5,
}

#[derive(Debug, Clone, Copy)]
pub struct LogConfig {
    pub level: LogLevel,
    pub debug_io: bool,
}

static CONFIG: OnceLock<LogConfig> = OnceLock::new();
static START: OnceLock<Instant> = OnceLock::new();
static LOG_FILE: OnceLock<Mutex<Option<File>>> = OnceLock::new();
const LOG_FILE_PATH: &str = "odon.log";

pub fn init(level: LogLevel, debug_io: bool) {
    let _ = START.set(Instant::now());
    let _ = CONFIG.set(LogConfig { level, debug_io });
    // The working directory is the natural place when Odon is started from a terminal. An
    // application launched from the Finder, Start menu or a desktop entry usually runs in a
    // directory it cannot write to (`/` on macOS), where the log used to be lost silently; fall
    // back to the per-user data directory so there is always a log to look at.
    let opened = log_file_candidates().into_iter().find_map(|path| {
        if let Some(parent) = path.parent().filter(|parent| !parent.as_os_str().is_empty()) {
            let _ = std::fs::create_dir_all(parent);
        }
        OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&path)
            .ok()
            .map(|file| (path, file))
    });
    match opened {
        Some((path, file)) => {
            let _ = LOG_FILE.set(Mutex::new(Some(file)));
            eprintln!(
                "[   0.000][thread][Info] writing logs to {}",
                path.display()
            );
        }
        None => {
            let _ = LOG_FILE.set(Mutex::new(None));
            eprintln!("[   0.000][thread][Warn] no writable location for {LOG_FILE_PATH}; logging to stderr only");
        }
    }
}

fn log_file_candidates() -> Vec<std::path::PathBuf> {
    let mut candidates = vec![std::path::PathBuf::from(LOG_FILE_PATH)];
    if let Some(directory) = dirs::data_local_dir() {
        candidates.push(directory.join("odon").join("logs").join(LOG_FILE_PATH));
    }
    candidates.push(std::env::temp_dir().join(LOG_FILE_PATH));
    candidates
}

pub fn config() -> LogConfig {
    *CONFIG.get_or_init(|| LogConfig {
        level: LogLevel::Warn,
        debug_io: false,
    })
}

pub fn enabled(level: LogLevel) -> bool {
    level <= config().level
}

pub fn debug_io_enabled() -> bool {
    config().debug_io
}

fn ts_s() -> f32 {
    START.get_or_init(Instant::now).elapsed().as_secs_f32()
}

pub fn log_line(level: LogLevel, msg: impl AsRef<str>) {
    if !enabled(level) {
        return;
    }
    let cur = std::thread::current();
    let thread = cur.name().unwrap_or("thread");
    let line = format!("[{:8.3}][{thread}][{level:?}] {}", ts_s(), msg.as_ref());
    eprintln!("{line}");
    if let Some(file) = LOG_FILE.get() {
        if let Ok(mut file) = file.lock() {
            if let Some(file) = file.as_mut() {
                let _ = writeln!(file, "{line}");
                let _ = file.flush();
            }
        }
    }
}

#[macro_export]
macro_rules! log_error {
    ($($arg:tt)*) => {{
        $crate::debug_log::log_line($crate::debug_log::LogLevel::Error, format!($($arg)*));
    }};
}

#[macro_export]
macro_rules! log_warn {
    ($($arg:tt)*) => {{
        $crate::debug_log::log_line($crate::debug_log::LogLevel::Warn, format!($($arg)*));
    }};
}

#[macro_export]
macro_rules! log_info {
    ($($arg:tt)*) => {{
        $crate::debug_log::log_line($crate::debug_log::LogLevel::Info, format!($($arg)*));
    }};
}

#[macro_export]
macro_rules! log_debug {
    ($($arg:tt)*) => {{
        $crate::debug_log::log_line($crate::debug_log::LogLevel::Debug, format!($($arg)*));
    }};
}

#[macro_export]
macro_rules! log_trace {
    ($($arg:tt)*) => {{
        $crate::debug_log::log_line($crate::debug_log::LogLevel::Trace, format!($($arg)*));
    }};
}
