use super::*;
#[test]
#[ignore = "diagnostic benchmark; run explicitly with --ignored --nocapture"]
fn benchmark_single_and_two_viewport_frame_planning() {
    fn run_frames(app: &mut OmeZarrViewerApp, ctx: &egui::Context, count: usize) -> f32 {
        for _ in 0..count {
            ctx.begin_pass(egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(1200.0, 800.0),
                )),
                ..Default::default()
            });
            egui::CentralPanel::default().show(ctx, |ui| {
                app.ui_viewport_workspace(ui, ctx);
            });
            let _ = ctx.end_pass();
        }
        app.viewport_frame_plan_ema_ms
    }

    let ctx = egui::Context::default();
    let mut app = fixture_app();
    let single_ms = run_frames(&mut app, &ctx, 40);
    let left = app.control_viewport_workspace_snapshot()["active_viewport_id"]
        .as_str()
        .unwrap()
        .to_string();
    let created = app.control_create_viewport(&serde_json::json!({
        "viewport_id": left,
        "layout": "horizontal",
    }));
    assert!(created.get("error").is_none(), "{created:#}");
    let split_ms = run_frames(&mut app, &ctx, 40);
    let resources = app.control_viewport_workspace_snapshot()["shared_resources"].clone();
    println!(
        "multi-viewport benchmark: single_frame_plan_ema_ms={single_ms:.4} split_frame_plan_ema_ms={split_ms:.4} resources={resources}"
    );
    assert_eq!(resources["document_instances"], 1);
    assert_eq!(resources["dataset_instances"], 1);
    assert_eq!(resources["cpu_decoded_tile_cache_instances"], 1);
}
