use super::*;

#[test]
fn duplicate_roi_ids_select_by_source_without_changing_project_records() {
    let channels = spawn_test_actor();
    let rois = ["a", "b", "c"].map(|sample| {
        let mut roi = ProjectRoi {
            id: "ROI1.ome.zarr".to_string(),
            ..Default::default()
        };
        roi.set_dataset_source(DatasetSource::Local(PathBuf::from(format!(
            "/tmp/{sample}/ROI1.ome.zarr"
        ))));
        roi
    });
    let keys = rois
        .iter()
        .map(|roi| roi.source_key().unwrap())
        .collect::<Vec<_>>();
    channels
        .model_tx
        .send(ActorModelUpdate::BootstrapProject(ProjectModelSnapshot {
            config: ProjectConfig {
                rois: rois.to_vec(),
                ..Default::default()
            },
            rois: rois.to_vec(),
            state: json!({"browser":{},"roi_views":{}}),
            load_generation: 1,
            ..Default::default()
        }))
        .unwrap();
    let (list, list_rx) = request("project.rois.list", json!({}));
    channels.request_tx.send(list).unwrap();
    list_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();

    for (mode, selected, expected) in [
        ("replace", vec![keys[1].clone()], vec![false, true, false]),
        ("add", vec![keys[2].clone()], vec![false, true, true]),
        ("toggle", vec![keys[1].clone()], vec![false, false, true]),
        ("replace", keys.clone(), vec![true, true, true]),
        ("remove", vec![keys[0].clone()], vec![false, true, true]),
    ] {
        let (select, select_rx) = request(
            "project.rois.select",
            json!({"source_keys":selected,"mode":mode}),
        );
        channels.request_tx.send(select).unwrap();
        select_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap()
            .unwrap();
        let (list, list_rx) = request("project.rois.list", json!({}));
        channels.request_tx.send(list).unwrap();
        let list = list_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap()
            .unwrap();
        assert_eq!(
            list["rois"]
                .as_array()
                .unwrap()
                .iter()
                .map(|roi| roi["selected"].as_bool().unwrap())
                .collect::<Vec<_>>(),
            expected
        );
    }

    for params in [
        json!({"ids":["ROI1.ome.zarr"]}),
        json!({"source_keys":[keys[0],"local:/tmp/missing"]}),
    ] {
        let (select, select_rx) = request("project.rois.select", params);
        channels.request_tx.send(select).unwrap();
        assert!(
            select_rx
                .recv_timeout(Duration::from_secs(1))
                .unwrap()
                .is_err()
        );
    }
    let projection = channels.presentation_rx.try_recv().unwrap();
    assert_eq!(projection.project.selected_source_keys, keys[1..].to_vec());
    assert_eq!(
        serde_json::to_value(projection.project.rois).unwrap(),
        serde_json::to_value(rois).unwrap()
    );
}

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
