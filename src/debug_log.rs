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
    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(LOG_FILE_PATH)
        .ok();
    let _ = LOG_FILE.set(Mutex::new(file));
    if LOG_FILE
        .get()
        .and_then(|file| file.lock().ok())
        .and_then(|file| file.as_ref().map(|_| ()))
        .is_some()
    {
        eprintln!("[   0.000][thread][Info] writing logs to {LOG_FILE_PATH}");
    }
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
