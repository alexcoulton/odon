use super::*;
#[test]
fn project_roi_transactions_complete_without_draining_the_ui_queue() {
    let channels = spawn_test_actor();
    let (create, create_rx) = request(
        "project.create",
        json!({"default_dataset":"background-project"}),
    );
    channels.request_tx.send(create).unwrap();
    assert_eq!(
        create_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap()
            .unwrap()["project"]["metadata"]["default_dataset"],
        "background-project"
    );
    for (id, path) in [("a", "/tmp/a.zarr"), ("b", "/tmp/b.zarr")] {
        let (add, add_rx) = request(
            "project.rois.add",
            json!({"id":id,"path":path,"display_name":id.to_uppercase()}),
        );
        channels.request_tx.send(add).unwrap();
        assert_eq!(
            add_rx
                .recv_timeout(Duration::from_secs(1))
                .unwrap()
                .unwrap()["roi"]["id"],
            id
        );
    }
    let (select, select_rx) = request("project.rois.select", json!({"ids":["b"],"mode":"replace"}));
    channels.request_tx.send(select).unwrap();
    assert_eq!(
        select_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap()
            .unwrap()["selected"],
        json!(["b"])
    );
    let (view, view_rx) = request(
        "project.views.create",
        json!({
            "name":"Comparison",
            "spec": {
                "channel_ref":{"label":"DAPI","alias":"stale"},
                "visible_channel_refs":[{"label":"DAPI","alias":"nuclei"}]
            }
        }),
    );
    channels.request_tx.send(view).unwrap();
    assert_eq!(
        view_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap()
            .unwrap()["spec"]["channel_ref"]["alias"],
        "nuclei"
    );
    let (views, views_rx) = request("project.views.list", json!({}));
    channels.request_tx.send(views).unwrap();
    assert_eq!(
        views_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap()
            .unwrap()["views"][0]["name"],
        "Comparison"
    );
    let (list, list_rx) = request("project.rois.list", json!({}));
    channels.request_tx.send(list).unwrap();
    assert_eq!(
        list_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap()
            .unwrap()["roi_count"],
        2
    );
    assert_eq!(channels.presentation_rx.len(), 1);
    let projection = channels.presentation_rx.try_recv().unwrap();
    assert_eq!(projection.mode, ModelMode::Project);
    assert_eq!(projection.project.rois.len(), 2);
    assert_eq!(projection.project.selected_source_keys.len(), 1);
    assert_eq!(projection.project.view_presets.len(), 1);
}
