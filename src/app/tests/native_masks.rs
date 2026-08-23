use super::*;

fn mask_projection(generation: u64, first_x: f32) -> serde_json::Value {
    serde_json::json!({
        "generation":generation,
        "active_layer_id":1,
        "layers":[{
            "id":1,
            "name":"Editable mask",
            "visible":true,
            "opacity":0.8,
            "width_screen_px":2.0,
            "display_mode":"translucent_fill",
            "color_rgb":[255,210,60],
            "offset_world":[0.0,0.0],
            "editable":true,
            "polygons_world":[[[first_x,0.0],[10.0,0.0],[10.0,10.0],[first_x,0.0]]],
            "source_geojson":null,
        }],
        "selection":null,
        "dirty":false,
        "undo_available":false,
    })
}

#[test]
fn native_mask_drag_defers_projection_and_commits_one_generation_checked_transaction() {
    let mut app = fixture_app();
    app.apply_control_actor_masks_projection(&mask_projection(2, 0.0))
        .unwrap();
    let selection = MaskPolygonSelection {
        layer_id: 1,
        polygon_idx: 0,
    };
    assert!(app.begin_mask_vertex_drag(selection, 0));
    assert!(app.move_mask_polygon_vertex(selection, 0, egui::pos2(3.0, 4.0)));

    app.apply_control_actor_masks_projection(&mask_projection(3, 20.0))
        .unwrap();
    assert_eq!(app.control_actor_mask_generation, 2);
    assert_eq!(
        app.mask_layers[0].polygons_world[0][0],
        egui::pos2(3.0, 4.0)
    );
    assert_eq!(
        app.pending_control_actor_mask_projection.as_ref().unwrap()["generation"],
        3
    );

    assert!(app.finish_mask_polygon_gesture());
    let intents = app.take_native_control_intents();
    assert_eq!(intents.len(), 1);
    assert_eq!(intents[0].method, "viewer.masks.state.replace");
    assert_eq!(intents[0].params["expected_generation"], 2);
    assert_eq!(intents[0].params["sync_project"], true);
    assert_eq!(
        intents[0].params["state"]["layers"][0]["polygons_world"][0][0],
        serde_json::json!([3.0, 4.0])
    );
    odon::control::ControlCommand::decode(intents[0].method, intents[0].params.clone()).unwrap();

    assert_eq!(app.control_actor_mask_generation, 3);
    assert_eq!(
        app.mask_layers[0].polygons_world[0][0],
        egui::pos2(20.0, 0.0)
    );
    assert!(app.pending_control_actor_mask_projection.is_none());
    assert!(!app.mask_polygon_gesture_active());
}

#[test]
fn native_mask_draw_selection_and_delete_submit_actor_commands_before_local_mutation() {
    let mut app = fixture_app();
    app.apply_control_actor_masks_projection(&mask_projection(2, 0.0))
        .unwrap();

    app.drawing_mask_layer = Some(1);
    app.drawing_mask_polygon = vec![
        egui::pos2(20.0, 20.0),
        egui::pos2(30.0, 20.0),
        egui::pos2(30.0, 30.0),
    ];
    assert!(app.finish_drawing_mask_polygon());
    assert_eq!(app.mask_layers[0].polygons_world.len(), 1);
    let intents = app.take_native_control_intents();
    assert_eq!(intents.len(), 1);
    assert_eq!(intents[0].method, "viewer.masks.polygons.add");
    assert_eq!(intents[0].params["expected_generation"], 2);
    assert_eq!(intents[0].params["sync_project"], true);
    assert_eq!(intents[0].params["coordinate_space"], "local");
    odon::control::ControlCommand::decode(intents[0].method, intents[0].params.clone()).unwrap();

    let canvas = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(100.0, 100.0));
    assert!(app.select_mask_polygon_at(1, egui::pos2(5.0, 5.0), egui::pos2(50.0, 50.0), canvas,));
    assert!(app.selected_mask_polygon.is_none());
    let intents = app.take_native_control_intents();
    assert_eq!(intents.len(), 1);
    assert_eq!(intents[0].method, "viewer.masks.selection.set");
    assert_eq!(intents[0].params["expected_generation"], 2);
    assert_eq!(intents[0].params["sync_project"], serde_json::Value::Null);
    assert_eq!(intents[0].params["id"], 1);
    assert_eq!(intents[0].params["index"], 0);

    app.selected_mask_polygon = Some(MaskPolygonSelection {
        layer_id: 1,
        polygon_idx: 0,
    });
    assert!(app.delete_selected_mask_polygon());
    assert_eq!(app.mask_layers[0].polygons_world.len(), 1);
    let intents = app.take_native_control_intents();
    assert_eq!(intents.len(), 1);
    assert_eq!(intents[0].method, "viewer.masks.polygons.remove");
    assert_eq!(intents[0].params["expected_generation"], 2);
    assert_eq!(intents[0].params["sync_project"], true);

    assert!(app.delete_mask_layer(1));
    assert_eq!(app.mask_layers.len(), 1);
    let intents = app.take_native_control_intents();
    assert_eq!(intents.len(), 1);
    assert_eq!(intents[0].method, "viewer.masks.layers.delete");
    assert_eq!(intents[0].params["expected_generation"], 2);
    assert_eq!(intents[0].params["sync_project"], true);
}
