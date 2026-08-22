use super::*;

pub(super) fn begin_channel_intensity(
    model: &AppModel,
    request: OdonControlRequest,
    load_job_tx: &Sender<LoadJob>,
    render_document: &Option<Arc<RenderDocument>>,
    diagnostics: &ActorDiagnostics,
) {
    let Some(document) = render_document.as_ref().cloned() else {
        diagnostics
            .rejected_requests
            .fetch_add(1, Ordering::Relaxed);
        diagnostics.record_reply_time(request.command.queue_age());
        let _ = request.reply.send(Err(ControlError::new(
            ControlErrorKind::NotReady,
            "channel intensity statistics require an opened dataset resource",
        )));
        return;
    };
    let spec = match model.channel_intensity_spec(document.dataset(), request.command.params()) {
        Ok(spec) => spec,
        Err(error) => {
            diagnostics
                .rejected_requests
                .fetch_add(1, Ordering::Relaxed);
            diagnostics.record_reply_time(request.command.queue_age());
            let _ = request.reply.send(Err(error));
            return;
        }
    };
    let generation = model.document_generation();
    match load_job_tx.try_send(LoadJob::ChannelIntensity {
        generation,
        request,
        document,
        spec,
    }) {
        Ok(()) => {
            diagnostics.workers_started.fetch_add(1, Ordering::Relaxed);
        }
        Err(crossbeam_channel::TrySendError::Full(LoadJob::ChannelIntensity {
            request, ..
        })) => {
            diagnostics
                .rejected_requests
                .fetch_add(1, Ordering::Relaxed);
            diagnostics.record_reply_time(request.command.queue_age());
            let _ = request.reply.send(Err(ControlError::new(
                ControlErrorKind::NotReady,
                "Odon's resource worker queue is full; retry later",
            )));
        }
        Err(crossbeam_channel::TrySendError::Disconnected(LoadJob::ChannelIntensity {
            request,
            ..
        })) => {
            diagnostics
                .rejected_requests
                .fetch_add(1, Ordering::Relaxed);
            diagnostics.record_reply_time(request.command.queue_age());
            let _ = request.reply.send(Err(ControlError::new(
                ControlErrorKind::NotReady,
                "Odon's resource workers are unavailable",
            )));
        }
        Err(_) => unreachable!("submitted channel intensity job changed variant"),
    }
}
