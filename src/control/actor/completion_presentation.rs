use super::completion::{CompletionContext, reject_cancelled_request, request_is_cancelled};
use super::presentation::screenshot_result;
use super::*;

pub(super) fn finish(completion: LoadCompletion, context: CompletionContext<'_>) {
    let LoadCompletion::ScreenshotWrite {
        request,
        spec,
        result,
    } = completion
    else {
        unreachable!("presentation completion router received another domain")
    };

    match result {
        Ok(bytes) => {
            context
                .model
                .finish_presentation_capture(spec.capture_id, "Screenshot capture completed");
            finish_request(
                request,
                screenshot_result(&spec, bytes),
                context.diagnostics,
            );
        }
        Err(_) if request_is_cancelled(&request) => {
            context
                .model
                .cancel_presentation_capture(spec.capture_id, "Screenshot capture was cancelled");
            reject_cancelled_request(request, context.diagnostics, "screenshot capture");
        }
        Err(error) => {
            let message = format!(
                "failed to write screenshot {}: {error}",
                spec.path.display()
            );
            context
                .model
                .fail_presentation_capture(spec.capture_id, &message);
            let kind = if !spec.overwrite && spec.path.exists() {
                ControlErrorKind::Conflict
            } else {
                ControlErrorKind::Application
            };
            reject_actor_request(
                request,
                context.diagnostics,
                ControlError::new(kind, message)
                    .with_data(json!({"path":spec.path.to_string_lossy()})),
            );
        }
    }
}
