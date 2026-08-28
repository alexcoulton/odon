#[test]
fn renderer_object_loader_exports_a_send_only_control_index() {
    let path = std::env::temp_dir().join(format!(
        "odon-control-objects-{}-{}.geojson",
        std::process::id(),
        std::thread::current().name().unwrap_or("test")
    ));
    std::fs::write(
        &path,
        serde_json::to_vec(&serde_json::json!({
            "type":"FeatureCollection",
            "features":[
                {"type":"Feature","id":"cell-a","properties":{"phenotype":"tumour","score":0.9},"geometry":{"type":"Polygon","coordinates":[[[0,0],[10,0],[10,10],[0,10],[0,0]]] }},
                {"type":"Feature","id":"cell-b","properties":{"phenotype":"immune","score":0.2},"geometry":{"type":"Polygon","coordinates":[[[20,20],[30,20],[30,30],[20,30],[20,20]]] }}
            ]
        }))
        .unwrap(),
    )
    .unwrap();
    let resource = crate::objects::load_control_object_resource(path.clone(), 1.0).unwrap();
    assert_eq!(resource.features.len(), 2);
    assert_eq!(resource.features[0].id, "cell-a");
    assert_eq!(
        resource.features[0].bbox_world,
        eframe::egui::Rect::from_min_max(
            eframe::egui::pos2(0.0, 0.0),
            eframe::egui::pos2(10.0, 10.0),
        )
    );
    assert_eq!(resource.features[0].polygons_world.len(), 1);
    assert_eq!(resource.features[0].polygons_world[0].len(), 5);
    assert_eq!(
        resource.property_value(0, "phenotype"),
        Some(serde_json::json!("tumour"))
    );
    assert!(resource.property_names.iter().any(|name| name == "score"));
    assert!(resource.renderer_payload.is_some());
    let mut layer = crate::objects::ObjectsLayer::default();
    assert!(layer.install_control_resource(&resource));
    assert_eq!(layer.object_count(), 2);
    std::fs::remove_file(path).unwrap();
}
