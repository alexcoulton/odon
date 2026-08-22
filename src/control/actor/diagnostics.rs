use super::*;

#[derive(Debug, Default)]
pub struct ActorDiagnostics {
    pub(super) alive: AtomicBool,
    pub(super) actor_requests: AtomicU64,
    pub(super) legacy_requests: AtomicU64,
    pub(super) rejected_requests: AtomicU64,
    pub(super) workers_started: AtomicU64,
    pub(super) workers_completed: AtomicU64,
    pub(super) stale_worker_completions: AtomicU64,
    pub(super) projections_published: AtomicU64,
    pub(super) projections_coalesced: AtomicU64,
    pub(super) queue_wait_ns_total: AtomicU64,
    pub(super) queue_wait_ns_max: AtomicU64,
    pub(super) model_time_ns_total: AtomicU64,
    pub(super) model_time_ns_max: AtomicU64,
    pub(super) reply_time_ns_total: AtomicU64,
    pub(super) reply_time_ns_max: AtomicU64,
    pub(super) presentation_wait_ns_total: AtomicU64,
    pub(super) presentation_wait_ns_max: AtomicU64,
    pub(super) presentation_wait_samples: AtomicU64,
    pub(super) pending_projection: Mutex<Option<(u64, Instant)>>,
}

impl ActorDiagnostics {
    pub fn shared() -> Arc<Self> {
        Arc::new(Self::default())
    }

    pub(super) fn set_alive(&self, alive: bool) {
        self.alive.store(alive, Ordering::Release);
    }

    pub(super) fn record_queue_wait(&self, duration: Duration) {
        record_duration(&self.queue_wait_ns_total, &self.queue_wait_ns_max, duration);
    }

    pub(super) fn record_model_time(&self, duration: Duration) {
        record_duration(&self.model_time_ns_total, &self.model_time_ns_max, duration);
    }

    pub(super) fn record_reply_time(&self, duration: Duration) {
        record_duration(&self.reply_time_ns_total, &self.reply_time_ns_max, duration);
    }

    pub(super) fn projection_published(&self, revision: u64, coalesced: bool) {
        self.projections_published.fetch_add(1, Ordering::Relaxed);
        if coalesced {
            self.projections_coalesced.fetch_add(1, Ordering::Relaxed);
        }
        *self
            .pending_projection
            .lock()
            .expect("actor diagnostics projection lock poisoned") =
            Some((revision, Instant::now()));
    }

    pub(super) fn projection_presented(&self, revision: u64) {
        let pending = self
            .pending_projection
            .lock()
            .expect("actor diagnostics projection lock poisoned")
            .take();
        let Some((pending_revision, published_at)) = pending else {
            return;
        };
        if revision < pending_revision {
            *self
                .pending_projection
                .lock()
                .expect("actor diagnostics projection lock poisoned") =
                Some((pending_revision, published_at));
            return;
        }
        record_duration(
            &self.presentation_wait_ns_total,
            &self.presentation_wait_ns_max,
            published_at.elapsed(),
        );
        self.presentation_wait_samples
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> Value {
        let actor_requests = self.actor_requests.load(Ordering::Relaxed);
        let presentation_samples = self.presentation_wait_samples.load(Ordering::Relaxed);
        json!({
            "alive": self.alive.load(Ordering::Acquire),
            "requests": {
                "actor": actor_requests,
                "legacy_ui": self.legacy_requests.load(Ordering::Relaxed),
                "rejected": self.rejected_requests.load(Ordering::Relaxed),
            },
            "workers": {
                "started": self.workers_started.load(Ordering::Relaxed),
                "completed": self.workers_completed.load(Ordering::Relaxed),
                "stale_completions": self.stale_worker_completions.load(Ordering::Relaxed),
            },
            "projections": {
                "published": self.projections_published.load(Ordering::Relaxed),
                "coalesced": self.projections_coalesced.load(Ordering::Relaxed),
                "waiting_for_presentation": self.pending_projection.lock().expect("actor diagnostics projection lock poisoned").is_some(),
            },
            "timing_ms": {
                "queue_wait": timing_json(
                    self.queue_wait_ns_total.load(Ordering::Relaxed),
                    self.queue_wait_ns_max.load(Ordering::Relaxed),
                    actor_requests + self.legacy_requests.load(Ordering::Relaxed),
                ),
                "model": timing_json(
                    self.model_time_ns_total.load(Ordering::Relaxed),
                    self.model_time_ns_max.load(Ordering::Relaxed),
                    actor_requests,
                ),
                "reply": timing_json(
                    self.reply_time_ns_total.load(Ordering::Relaxed),
                    self.reply_time_ns_max.load(Ordering::Relaxed),
                    actor_requests,
                ),
                "presentation_wait": timing_json(
                    self.presentation_wait_ns_total.load(Ordering::Relaxed),
                    self.presentation_wait_ns_max.load(Ordering::Relaxed),
                    presentation_samples,
                ),
            },
        })
    }
}

fn record_duration(total: &AtomicU64, max: &AtomicU64, duration: Duration) {
    let nanos = duration.as_nanos().min(u128::from(u64::MAX)) as u64;
    total.fetch_add(nanos, Ordering::Relaxed);
    max.fetch_max(nanos, Ordering::Relaxed);
}

fn timing_json(total_ns: u64, max_ns: u64, samples: u64) -> Value {
    json!({
        "samples": samples,
        "average": (samples > 0).then(|| total_ns as f64 / samples as f64 / 1_000_000.0),
        "max": (samples > 0).then(|| max_ns as f64 / 1_000_000.0),
    })
}
