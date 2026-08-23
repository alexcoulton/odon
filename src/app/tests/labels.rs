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
    app.seg_label_prompt_open = true;
    app.seg_label_prompt_always = true;
    assert!(
        app.install_control_actor_label_resource(7, &actor_resource)
            .unwrap()
    );
    assert!(!app.seg_label_prompt_open);
    assert!(!app.seg_label_prompt_always);
    assert_eq!(app.control_labels_json()["loaded"], "cells");
    assert_eq!(app.control_labels_json()["actor_owned"], true);
    app.seg_label_prompt_open = true;
    app.seg_label_prompt_always = true;
    assert!(app.unload_control_actor_label_resource(8));
    assert!(!app.seg_label_prompt_open);
    assert!(!app.seg_label_prompt_always);
    assert!(app.control_labels_json()["loaded"].is_null());

    // Replayed/coalesced projections must also dismiss a prompt that opened while the actor
    // resource was being prepared, even when the semantic generation is unchanged.
    app.seg_label_prompt_open = true;
    assert!(!app.unload_control_actor_label_resource(8));
    assert!(!app.seg_label_prompt_open);

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
