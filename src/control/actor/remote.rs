use super::*;
use crate::data::remote_store::S3SessionCredentials;

pub(super) struct RemoteSessionState {
    generation: u64,
    credentials: Option<S3SessionCredentials>,
    pending_s3_open_generation: Option<u64>,
}

impl Default for RemoteSessionState {
    fn default() -> Self {
        Self {
            generation: 0,
            credentials: None,
            pending_s3_open_generation: None,
        }
    }
}

impl RemoteSessionState {
    pub(super) fn snapshot(&self) -> Value {
        let configured = self.credentials.is_some();
        json!({
            "configured": configured,
            "endpoint": self.credentials.as_ref().map(|session| session.endpoint.as_str()),
            "region": self.credentials.as_ref().map(|session| session.region.as_str()),
            "bucket": self.credentials.as_ref().map(|session| session.bucket.as_str()),
            "credentials": if configured { "session_only_redacted" } else { "none" },
            "persisted": false,
            "generation": self.generation,
        })
    }

    pub(super) fn credentials(&self) -> Result<(u64, S3SessionCredentials), ControlError> {
        self.credentials
            .clone()
            .map(|credentials| (self.generation, credentials))
            .ok_or_else(|| {
                ControlError::new(
                    ControlErrorKind::NotReady,
                    "S3 session credentials are not configured",
                )
            })
    }

    pub(super) fn is_current(&self, generation: u64) -> bool {
        self.credentials.is_some() && self.generation == generation
    }

    pub(super) fn configure(&mut self, params: &Value, model: &mut AppModel) -> Value {
        self.invalidate_dependent_work(model, "S3 session was replaced");
        self.generation = self.generation.wrapping_add(1).max(1);
        self.credentials = Some(S3SessionCredentials::normalized(
            params["endpoint"].as_str().unwrap_or_default(),
            params["region"].as_str().unwrap_or("auto"),
            params["bucket"].as_str().unwrap_or_default(),
            params["access_key"].as_str().unwrap_or_default(),
            params["secret_key"].as_str().unwrap_or_default(),
        ));
        self.snapshot()
    }

    pub(super) fn clear(&mut self, model: &mut AppModel) -> Value {
        self.invalidate_dependent_work(model, "S3 session was cleared");
        self.generation = self.generation.wrapping_add(1).max(1);
        self.credentials = None;
        json!({
            "cleared": true,
            "configured": false,
            "persisted": false,
            "generation": self.generation,
        })
    }

    pub(super) fn mark_s3_open_pending(&mut self, document_generation: u64) {
        self.pending_s3_open_generation = Some(document_generation);
    }

    pub(super) fn finish_s3_open(&mut self, document_generation: u64) {
        if self.pending_s3_open_generation == Some(document_generation) {
            self.pending_s3_open_generation = None;
        }
    }

    fn invalidate_dependent_work(&mut self, model: &mut AppModel, reason: &str) {
        model.cancel_pending_remote_listings(reason);
        model.cancel_pending_deep_link_apply(reason);
        if let Some(generation) = self.pending_s3_open_generation.take() {
            model.fail_dataset_open_for_generation(generation, reason);
        }
    }
}

pub(super) enum RemoteOpenSpec {
    Http {
        url: String,
    },
    S3 {
        credentials: S3SessionCredentials,
        prefix: String,
    },
}

#[derive(Clone)]
pub(super) enum RemoteOpenIdentity {
    Http {
        url: String,
    },
    S3 {
        endpoint: String,
        region: String,
        bucket: String,
        prefix: String,
    },
}

pub(super) fn begin_remote_list(
    model: &mut AppModel,
    session: &RemoteSessionState,
    request: OdonControlRequest,
    load_job_tx: &Sender<LoadJob>,
    diagnostics: &ActorDiagnostics,
) {
    let (session_generation, credentials) = match session.credentials() {
        Ok(session) => session,
        Err(error) => {
            reject_actor_request(request, diagnostics, error);
            return;
        }
    };
    if load_job_tx.is_full() {
        reject_worker_submission(request, diagnostics);
        return;
    }
    let prefix = normalize_s3_prefix(
        request
            .command
            .params()
            .get("prefix")
            .and_then(Value::as_str)
            .unwrap_or_default(),
    );
    let operation_scope = format!("{session_generation}:{prefix}");
    let operation_generation = model.begin_remote_listing(operation_scope.clone());
    match load_job_tx.try_send(LoadJob::RemoteList {
        session_generation,
        operation_generation,
        operation_scope: operation_scope.clone(),
        request,
        credentials,
        prefix,
    }) {
        Ok(()) => {
            diagnostics.workers_started.fetch_add(1, Ordering::Relaxed);
        }
        Err(error) => {
            let LoadJob::RemoteList { request, .. } = error.into_inner() else {
                unreachable!("remote listing submission returns its own job")
            };
            model.cancel_remote_listing(
                &operation_scope,
                operation_generation,
                "Remote worker queue is unavailable",
            );
            reject_worker_submission(request, diagnostics);
        }
    }
}

pub(super) fn begin_remote_http_open(
    model: &mut AppModel,
    request: OdonControlRequest,
    load_job_tx: &Sender<LoadJob>,
    diagnostics: &ActorDiagnostics,
) {
    if load_job_tx.is_full() {
        reject_worker_submission(request, diagnostics);
        return;
    }
    let url = request.command.params()["url"]
        .as_str()
        .unwrap_or_default()
        .trim()
        .trim_end_matches('/')
        .to_string();
    let generation = model.begin_dataset_open(url.clone());
    submit_remote_open(
        model,
        request,
        load_job_tx,
        diagnostics,
        generation,
        None,
        RemoteOpenSpec::Http { url },
    );
}

pub(super) fn begin_remote_s3_open(
    model: &mut AppModel,
    session: &mut RemoteSessionState,
    request: OdonControlRequest,
    load_job_tx: &Sender<LoadJob>,
    diagnostics: &ActorDiagnostics,
) {
    let (session_generation, credentials) = match session.credentials() {
        Ok(session) => session,
        Err(error) => {
            reject_actor_request(request, diagnostics, error);
            return;
        }
    };
    if load_job_tx.is_full() {
        reject_worker_submission(request, diagnostics);
        return;
    }
    let prefix = normalize_s3_prefix(
        request
            .command
            .params()
            .get("prefix")
            .and_then(Value::as_str)
            .unwrap_or_default(),
    );
    let source = format!("s3://{}/{prefix}", credentials.bucket);
    let generation = model.begin_dataset_open(source);
    session.mark_s3_open_pending(generation);
    submit_remote_open(
        model,
        request,
        load_job_tx,
        diagnostics,
        generation,
        Some(session_generation),
        RemoteOpenSpec::S3 {
            credentials,
            prefix,
        },
    );
}

fn submit_remote_open(
    model: &mut AppModel,
    request: OdonControlRequest,
    load_job_tx: &Sender<LoadJob>,
    diagnostics: &ActorDiagnostics,
    generation: u64,
    session_generation: Option<u64>,
    spec: RemoteOpenSpec,
) {
    match load_job_tx.try_send(LoadJob::RemoteOpen {
        generation,
        session_generation,
        request,
        spec,
    }) {
        Ok(()) => {
            diagnostics.workers_started.fetch_add(1, Ordering::Relaxed);
        }
        Err(error) => {
            let LoadJob::RemoteOpen { request, .. } = error.into_inner() else {
                unreachable!("remote open submission returns its own job")
            };
            model.fail_dataset_open_for_generation(
                generation,
                "Remote dataset worker queue is unavailable",
            );
            reject_worker_submission(request, diagnostics);
        }
    }
}

pub(super) fn normalize_s3_prefix(prefix: &str) -> String {
    prefix.trim().trim_matches('/').to_string()
}
