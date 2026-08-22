use super::*;
#[test]
fn label_control_state_and_channel_presentation_are_bounded() {
    let mut app = fixture_app();
    let labels = app.control_labels_json();
    assert!(labels["available"].is_array());
    assert_eq!(labels["gpu_available"], false);
    assert_eq!(labels["busy"], false);
    let unloaded = app.control_unload_labels();
    assert!(unloaded["labels"].is_object());
    assert_eq!(unloaded["labels"]["visible"], false);
    assert!(
        app.control_load_labels(&serde_json::json!({"name": "missing"}))
            .get("error")
            .is_some()
    );

    let actor_labels = LabelZarrDataset::try_open(Arc::clone(&app.store), "cells")
        .unwrap()
        .expect("fixture label metadata");
    let actor_resource = odon::model::ControlLabelResource {
        dataset: actor_labels,
        store: Arc::clone(&app.store),
    };
    assert!(
        app.install_control_actor_label_resource(7, &actor_resource)
            .unwrap()
    );
    assert_eq!(app.control_labels_json()["loaded"], "cells");
    assert_eq!(app.control_labels_json()["actor_owned"], true);
    assert!(app.unload_control_actor_label_resource(8));
    assert!(app.control_labels_json()["loaded"].is_null());

    let presentation = app.control_set_channel_presentation(
        &serde_json::json!({"search": "nuc", "sort": "visible_first"}),
    );
    assert_eq!(presentation["search"], "nuc");
    assert_eq!(presentation["sort"], "visible_first");
    let before = app.control_channel_presentation_json();
    assert!(
        app.control_set_channel_presentation(&serde_json::json!({"sort": "unknown"}))
            .get("error")
            .is_some()
    );
    assert_eq!(app.control_channel_presentation_json(), before);
}
