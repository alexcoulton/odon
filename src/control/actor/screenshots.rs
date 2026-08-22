use super::*;

pub(super) fn begin_screenshot_settings_update(
    model: &mut AppModel,
    request: OdonControlRequest,
    load_job_tx: &Sender<LoadJob>,
    diagnostics: &ActorDiagnostics,
) {
    if load_job_tx.is_full() {
        reject_worker_submission(request, diagnostics);
        return;
    }
    let normalized_output_dir = match request.command.params().get("output_dir") {
        Some(Value::Null) => Some(None),
        Some(Value::String(path)) => Some(Some(expand_path(path))),
        Some(_) => {
            reject_actor_request(
                request,
                diagnostics,
                ControlError::invalid_params(
                    "viewer.screenshot.settings.set",
                    "output_dir must be a path string or null",
                ),
            );
            return;
        }
        None => None,
    };
    let (generation, preferences) = match model
        .begin_screenshot_settings_update(request.command.params(), normalized_output_dir)
    {
        Ok(operation) => operation,
        Err(error) => {
            reject_actor_request(request, diagnostics, error);
            return;
        }
    };
    match load_job_tx.try_send(LoadJob::ScreenshotSettingsValidate {
        generation,
        request,
        preferences,
    }) {
        Ok(()) => {
            diagnostics.workers_started.fetch_add(1, Ordering::Relaxed);
        }
        Err(error) => {
            let LoadJob::ScreenshotSettingsValidate {
                generation,
                request,
                ..
            } = error.into_inner()
            else {
                unreachable!("screenshot settings submission returns its own job")
            };
            model.fail_screenshot_settings_for_generation(
                generation,
                "Screenshot settings worker queue is unavailable",
            );
            reject_worker_submission(request, diagnostics);
        }
    }
}
