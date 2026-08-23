use super::*;

#[test]
fn registry_names_are_unique_and_capabilities_are_sorted() {
    let names = METHODS
        .iter()
        .map(|descriptor| descriptor.name)
        .chain(PROTOCOL_METHODS.iter().map(|method| method.0))
        .chain(METHOD_ALIASES.iter().map(|method| method.0))
        .collect::<BTreeSet<_>>();
    assert_eq!(
        names.len(),
        METHODS.len() + PROTOCOL_METHODS.len() + METHOD_ALIASES.len()
    );

    let capabilities = capabilities();
    assert!(capabilities.windows(2).all(|pair| pair[0] < pair[1]));
    assert!(capabilities.contains(&"system.introspect".to_string()));
}

#[test]
fn every_method_has_introspection_metadata() {
    for descriptor in METHODS.iter() {
        assert!(!descriptor.name.is_empty());
        assert!(
            descriptor.name.contains('.'),
            "canonical method is not hierarchical: {}",
            descriptor.name
        );
        assert!(!descriptor.summary.is_empty(), "{}", descriptor.name);
        assert!(!descriptor.capability.is_empty(), "{}", descriptor.name);
        assert!(!descriptor.available_in.is_empty(), "{}", descriptor.name);
        assert!(
            !descriptor
                .execution_class
                .readiness_requirements()
                .is_empty(),
            "{}",
            descriptor.name
        );
        assert!(request_schema_for(descriptor).is_object());
        assert!(
            !descriptor
                .completion_contract()
                .completion_point()
                .is_empty()
        );
        assert!(matches!(
            descriptor.cancellation_contract(),
            "cooperative" | "not_applicable"
        ));
    }
    assert_eq!(
        method("viewer.viewports.camera.fit")
            .unwrap()
            .execution_class,
        ExecutionClass::Geometry
    );
    assert_eq!(
        method("datasets.open_ome_zarr").unwrap().execution_class,
        ExecutionClass::Resource
    );
    assert_eq!(
        method("viewer.screenshot.capture").unwrap().execution_class,
        ExecutionClass::Presentation
    );
    assert_eq!(
        method("viewer.screenshot.capture")
            .unwrap()
            .completion_contract(),
        CompletionContract::PresentationDependent
    );
    assert_eq!(
        method("datasets.open_ome_zarr")
            .unwrap()
            .completion_contract(),
        CompletionContract::RetainedBackground
    );
    assert_eq!(
        method("viewer.screenshot.settings.set")
            .unwrap()
            .completion_contract(),
        CompletionContract::ResourceReady
    );
    assert_eq!(
        method("viewer.screenshot.settings.get")
            .unwrap()
            .completion_contract(),
        CompletionContract::ImmediateSemantic
    );
    let catalog = catalog_json();
    assert!(catalog.as_array().unwrap().iter().all(|entry| {
        entry.get("execution_route").is_some()
            && entry["execution_route"]["by_mode"].is_object()
            && entry["completion_contract"].is_string()
            && entry["completion_point"].is_string()
            && entry["cancellation"].is_string()
    }));
}

#[test]
fn execution_routes_are_mode_aware_and_object_targets_are_actor_owned() {
    let camera = method("viewer.camera.set").unwrap();
    assert_eq!(
        execution_owner(camera, "single", &json!({}), false),
        ExecutionOwner::Actor
    );
    assert_eq!(
        execution_owner(camera, "mosaic", &json!({}), false),
        ExecutionOwner::Actor
    );
    let selection = method("viewer.objects.get_selection").unwrap();
    assert_eq!(
        execution_owner(
            selection,
            "single",
            &json!({"target":"segmentation_objects"}),
            false,
        ),
        ExecutionOwner::Actor
    );
    assert_eq!(
        execution_owner(
            selection,
            "single",
            &json!({"target":"spatial_shape","layer_id":7}),
            false,
        ),
        ExecutionOwner::Actor
    );
    assert_eq!(execution_route_summary(selection), "actor");
    assert_eq!(execution_route_json(selection)["variants"], json!([]));
    let route = execution_route_json(camera);
    assert_eq!(route["by_mode"]["single"]["default_owner"], "actor");
    assert_eq!(route["by_mode"]["mosaic"]["default_owner"], "actor");
    assert_eq!(execution_route_summary(camera), "actor");

    let memory = method("memory.pin").unwrap();
    assert_eq!(
        execution_owner(memory, "single", &json!({}), false),
        ExecutionOwner::Actor
    );
    assert_eq!(
        execution_owner(memory, "mosaic", &json!({}), false),
        ExecutionOwner::Actor
    );
    let route = execution_route_json(memory);
    assert_eq!(route["by_mode"]["single"]["default_owner"], "actor");
    assert_eq!(route["by_mode"]["mosaic"]["default_owner"], "actor");
}

#[test]
fn mosaic_has_no_legacy_control_routes() {
    let legacy = METHODS
        .iter()
        .filter(|descriptor| descriptor.available_in.contains(&"mosaic"))
        .filter(|descriptor| {
            execution_owner(descriptor, "mosaic", &json!({}), false) == ExecutionOwner::LegacyUi
        })
        .map(|descriptor| descriptor.name)
        .collect::<Vec<_>>();
    assert!(legacy.is_empty(), "legacy mosaic routes remain: {legacy:?}");
}

#[test]
fn registered_application_surface_has_no_legacy_execution_routes() {
    let legacy = METHODS
        .iter()
        .flat_map(|descriptor| {
            ["project", "single", "mosaic", "transition"]
                .into_iter()
                .filter_map(move |mode| {
                    (execution_owner(descriptor, mode, &json!({}), false)
                        == ExecutionOwner::LegacyUi)
                        .then_some((descriptor.name, mode))
                })
        })
        .collect::<Vec<_>>();
    assert!(
        legacy.is_empty(),
        "registered legacy application routes remain: {legacy:?}"
    );
    assert!(
        METHODS
            .iter()
            .all(|descriptor| execution_route_json(descriptor)["variants"] == json!([])),
        "conditional legacy target routes remain"
    );
}

#[test]
fn flat_names_are_deprecated_aliases_for_hierarchical_methods() {
    assert_eq!(canonical_method("set_camera"), "viewer.camera.set");
    assert_eq!(canonical_method("viewer.camera.set"), "viewer.camera.set");
    assert!(aliases_for("viewer.camera.set").contains(&"set_camera"));
    assert_eq!(method("set_camera").unwrap().name, "viewer.camera.set");
}

#[test]
fn availability_reports_mode_and_accepts_alias_filters() {
    let requested = vec!["get_camera".to_string(), "project.save".to_string()];
    let single = availability_catalog("single", Some(&requested));
    let methods = single["methods"].as_array().unwrap();
    assert_eq!(methods.len(), 2);
    assert!(methods.iter().all(|method| method["available"] == true));

    let project = availability_catalog("project", Some(&requested));
    let camera = project["methods"]
        .as_array()
        .unwrap()
        .iter()
        .find(|method| method["method"] == "viewer.camera.get")
        .unwrap();
    assert_eq!(camera["available"], false);
    assert_eq!(camera["reason"], "wrong_mode");

    let transition = availability_catalog("transition", Some(&requested));
    let camera = transition["methods"]
        .as_array()
        .unwrap()
        .iter()
        .find(|method| method["method"] == "viewer.camera.get")
        .unwrap();
    assert_eq!(camera["reason"], "not_ready");
}

#[test]
fn multi_viewport_registry_contracts_expose_ids_revisions_events_and_modes() {
    let camera = method("viewer.viewports.camera.set").unwrap();
    assert!(camera.mutates);
    assert_eq!(camera.event, Some("viewer.viewports.navigation.changed"));
    assert_eq!(camera.available_in, SINGLE_MODE);
    let camera_schema = request_schema_for(camera);
    assert_eq!(camera_schema["required"], json!(["viewport_id"]));
    assert_eq!(
        camera_schema["properties"]["if_navigation_revision"]["minimum"],
        1
    );

    let style = method("viewer.viewports.objects.style.set").unwrap();
    assert_eq!(style.event, Some("viewer.viewports.presentation.changed"));
    let style_schema = request_schema_for(style);
    assert_eq!(
        style_schema["properties"]["if_presentation_revision"]["minimum"],
        1
    );

    let links = request_schema_for(method("viewer.viewport_links.create").unwrap());
    assert_eq!(links["required"], json!(["viewports", "fields"]));
    assert_eq!(links["properties"]["viewports"]["minItems"], 2);

    for name in [
        "viewer.workspace.get",
        "viewer.workspace.layout.get",
        "viewer.workspace.layout.set",
        "viewer.workspace.swap",
        "viewer.viewports.list",
        "viewer.viewports.get",
        "viewer.viewports.create",
        "viewer.viewports.clone",
        "viewer.viewports.rename",
        "viewer.viewports.remove",
        "viewer.viewports.set_active",
        "viewer.viewport_links.set",
        "viewer.viewport_links.get",
        "viewer.viewport_links.list",
        "viewer.viewport_links.create",
        "viewer.viewport_links.update",
        "viewer.viewport_links.remove",
        "viewer.viewports.camera.get",
        "viewer.viewports.camera.set",
        "viewer.viewports.camera.fit",
        "viewer.viewports.planes.get",
        "viewer.viewports.planes.set",
        "viewer.viewports.channels.get",
        "viewer.viewports.channels.set_visible",
        "viewer.viewports.channels.set",
        "viewer.viewports.channels.set_active",
        "viewer.viewports.channels.set_color",
        "viewer.viewports.channels.set_contrast",
        "viewer.viewports.channels.set_order",
        "viewer.viewports.channels.list_groups",
        "viewer.viewports.channels.set_group",
        "viewer.viewports.rendering.get",
        "viewer.viewports.rendering.set",
        "viewer.viewports.objects.style.get",
        "viewer.viewports.objects.style.set",
        "viewer.viewports.objects.legend.set",
        "viewer.viewports.objects.filter.get",
        "viewer.viewports.objects.filter.set",
        "viewer.viewports.objects.filter.clear",
        "viewer.viewports.layers.list",
        "viewer.viewports.layers.get",
        "viewer.viewports.layers.set",
        "viewer.viewports.layers.set_visibility",
        "viewer.viewports.layers.set_order",
        "viewer.viewports.layers.set_active",
        "viewer.workspace.screenshot.capture",
    ] {
        assert!(method(name).is_some(), "missing registry method {name}");
    }
}

#[test]
fn actor_capability_registry_has_unique_known_methods() {
    let application_methods = METHODS
        .iter()
        .map(|descriptor| descriptor.name)
        .collect::<BTreeSet<_>>();
    let protocol_methods = PROTOCOL_METHODS
        .iter()
        .map(|descriptor| descriptor.0)
        .collect::<BTreeSet<_>>();
    let mut seen = BTreeSet::new();

    for name in ACTOR_CAPABLE_METHODS {
        assert!(seen.insert(*name), "duplicate actor-capable method {name}");
        assert!(
            application_methods.contains(name) || protocol_methods.contains(name),
            "actor-capable method is absent from every registry: {name}"
        );
    }
}
