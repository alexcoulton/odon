use super::*;
#[test]
fn label_resources_load_toggle_and_unload_without_a_ui_frame() {
    let channels = spawn_test_actor();
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/synthetic_5ch.ome.zarr");
    let (open, open_rx) = request("datasets.open_ome_zarr", json!({"path":fixture}));
    channels.request_tx.send(open).unwrap();
    open_rx
        .recv_timeout(Duration::from_secs(10))
        .unwrap()
        .unwrap();

    let (list, list_rx) = request("viewer.labels.list", json!({}));
    channels.request_tx.send(list).unwrap();
    let listed = list_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    assert_eq!(listed["available"], json!(["cells"]));
    assert_eq!(listed["selected"], "cells");
    assert!(listed["loaded"].is_null());

    let (load, load_rx) = request("viewer.labels.load", json!({"name":"cells"}));
    channels.request_tx.send(load).unwrap();
    let loaded = load_rx
        .recv_timeout(Duration::from_secs(2))
        .unwrap()
        .unwrap();
    assert_eq!(loaded["loaded"], "cells");
    assert_eq!(loaded["visible"], true);
    assert_eq!(loaded["busy"], false);
    assert_eq!(loaded["actor_owned"], true);

    let projection = channels.presentation_rx.try_recv().unwrap();
    assert_eq!(
        projection
            .label_resource
            .as_ref()
            .map(|resource| resource.dataset.label_name.as_str()),
        Some("cells")
    );
    let workspace = projection.workspace.as_ref().unwrap();
    assert_eq!(workspace["labels"]["loaded"], "cells");
    let label_layer = workspace["viewports"][0]["native_layers"]
        .as_array()
        .unwrap()
        .iter()
        .find(|layer| layer["layer_id"] == "segmentation_labels")
        .unwrap();
    assert_eq!(label_layer["visible"], true);

    let (hide, hide_rx) = request("viewer.labels.set_visibility", json!({"visible":false}));
    channels.request_tx.send(hide).unwrap();
    assert_eq!(
        hide_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap()
            .unwrap()["visible"],
        false
    );
    let (show, show_rx) = request("viewer.labels.set_visibility", json!({"visible":true}));
    channels.request_tx.send(show).unwrap();
    assert_eq!(
        show_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap()
            .unwrap()["visible"],
        true
    );

    let (unload, unload_rx) = request("viewer.labels.unload", json!({}));
    channels.request_tx.send(unload).unwrap();
    let unloaded = unload_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    assert_eq!(unloaded["unloaded"], "cells");
    assert!(unloaded["labels"]["loaded"].is_null());
    assert_eq!(unloaded["labels"]["visible"], false);
    let projection = channels.presentation_rx.try_recv().unwrap();
    assert!(projection.label_resource.is_none());
    assert_eq!(
        projection.workspace.as_ref().unwrap()["labels"]["actor_owned"],
        true
    );
}
