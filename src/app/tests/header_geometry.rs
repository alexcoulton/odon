use super::*;
#[test]
fn horizontal_split_stacks_each_header_above_an_adjacent_full_height_canvas() {
    let mut app = fixture_actor_app();
    let left = app.control_viewport_workspace_snapshot()["active_viewport_id"]
        .as_str()
        .unwrap()
        .to_string();
    let created = app.actor_command(
        "viewer.viewports.clone",
        serde_json::json!({
            "viewport_id": left,
            "layout": "horizontal",
            "ratio": 0.55,
        }),
    );
    assert!(created.get("error").is_none(), "{created:#}");

    let ctx = egui::Context::default();
    ctx.begin_pass(egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1200.0, 800.0),
        )),
        ..Default::default()
    });
    egui::CentralPanel::default().show(&ctx, |ui| {
        app.ui_viewport_workspace(ui, &ctx);
    });
    let _ = ctx.end_pass();

    let canvases = app
        .viewport_workspace
        .as_ref()
        .unwrap()
        .viewports()
        .iter()
        .map(|viewport| viewport.state.last_canvas_rect.unwrap())
        .collect::<Vec<_>>();
    assert_eq!(canvases.len(), 2);
    assert!(
        (canvases[0].top() - canvases[1].top()).abs() <= 2.0,
        "horizontal split canvas rectangles must align: {canvases:?}"
    );
    assert!((canvases[0].bottom() - canvases[1].bottom()).abs() < 1.0);
    assert!(canvases[0].height() > 700.0);
    assert!(canvases[1].height() > 700.0);
    assert!(canvases[0].right() < canvases[1].left());
    assert!(canvases[1].left() - canvases[0].right() < 30.0);
    let content_width = canvases[0].width() + canvases[1].width();
    assert!((canvases[0].width() / content_width - 0.55).abs() < 0.01);

    let before_activation = canvases;
    let activated = app.actor_command(
        "viewer.viewports.set_active",
        serde_json::json!({
            "viewport_id": left,
        }),
    );
    assert!(activated.get("error").is_none(), "{activated:#}");
    ctx.begin_pass(egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1200.0, 800.0),
        )),
        ..Default::default()
    });
    egui::CentralPanel::default().show(&ctx, |ui| {
        app.ui_viewport_workspace(ui, &ctx);
    });
    let _ = ctx.end_pass();

    let after_activation = app
        .viewport_workspace
        .as_ref()
        .unwrap()
        .viewports()
        .iter()
        .map(|viewport| viewport.state.last_canvas_rect.unwrap())
        .collect::<Vec<_>>();
    assert_eq!(before_activation.len(), after_activation.len());
    for (before, after) in before_activation.iter().zip(&after_activation) {
        assert!(
            (before.top() - after.top()).abs() <= f32::EPSILON
                && (before.bottom() - after.bottom()).abs() <= f32::EPSILON
                && (before.left() - after.left()).abs() <= f32::EPSILON
                && (before.right() - after.right()).abs() <= f32::EPSILON,
            "activating a viewport must not resize its header or canvas: before={before:?}, after={after:?}"
        );
    }
}
