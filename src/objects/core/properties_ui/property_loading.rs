use super::*;

const MAX_CONCURRENT_LAZY_PROPERTY_LOADS: usize = 4;

struct LazyPropertyLoadLimiter {
    active: std::sync::Mutex<usize>,
    wake: std::sync::Condvar,
}

impl LazyPropertyLoadLimiter {
    fn acquire(&'static self, cancel: &AtomicBool) -> Option<LazyPropertyLoadPermit<'static>> {
        let mut active = self
            .active
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        while *active >= MAX_CONCURRENT_LAZY_PROPERTY_LOADS {
            if cancel.load(Ordering::Relaxed) {
                return None;
            }
            let (next, _) = self
                .wake
                .wait_timeout(active, std::time::Duration::from_millis(20))
                .unwrap_or_else(|error| error.into_inner());
            active = next;
        }
        if cancel.load(Ordering::Relaxed) {
            return None;
        }
        *active += 1;
        Some(LazyPropertyLoadPermit { limiter: self })
    }
}

pub(super) struct LazyPropertyLoadPermit<'a> {
    limiter: &'a LazyPropertyLoadLimiter,
}

impl Drop for LazyPropertyLoadPermit<'_> {
    fn drop(&mut self) {
        let mut active = self
            .limiter
            .active
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        *active = active.saturating_sub(1);
        self.limiter.wake.notify_all();
    }
}

static LAZY_PROPERTY_LOAD_LIMITER: std::sync::LazyLock<LazyPropertyLoadLimiter> =
    std::sync::LazyLock::new(|| LazyPropertyLoadLimiter {
        active: std::sync::Mutex::new(0),
        wake: std::sync::Condvar::new(),
    });

pub(super) fn acquire_lazy_property_load(
    cancel: &AtomicBool,
) -> Option<LazyPropertyLoadPermit<'static>> {
    LAZY_PROPERTY_LOAD_LIMITER.acquire(cancel)
}
