use super::*;
use std::path::PathBuf;
use std::time::Instant;

fn read_json(reader: &mut BufReader<TcpStream>) -> Value {
    let mut line = String::new();
    reader.read_line(&mut line).expect("read bridge response");
    serde_json::from_str(line.trim()).expect("parse bridge response")
}

#[test]
fn optional_tcp_failure_keeps_the_local_actor_available() {
    let occupied = TcpListener::bind("127.0.0.1:0").expect("reserve test address");
    let address = occupied.local_addr().expect("reserved address");
    let ctx = egui::Context::default();
    let bridge = OdonControlBridge::spawn_inner(
        &address.to_string(),
        ctx.clone(),
        false,
        false,
        None,
        None,
        None,
    )
    .expect("optional TCP exposure must not prevent actor startup");

    assert!(bridge.server_error().is_some());
    assert!(bridge.instance_manifest().is_none());
    let reply = bridge
        .submit_native_command_with_reply(&ctx, "project.get", serde_json::json!({}))
        .expect("submit command to local actor");
    reply
        .recv_timeout(Duration::from_secs(2))
        .expect("local actor reply after TCP failure")
        .expect("local actor command succeeds");
}

#[test]
fn native_ingress_reaches_actor_without_root_update() {
    let ctx = egui::Context::default();
    let runtime =
        OdonControlBridge::spawn_inner("127.0.0.1:0", ctx.clone(), false, false, None, None, None)
            .expect("spawn local actor");
    let ingress = runtime.native_command_ingress();
    assert!(ingress.submit(
        "project.rois.add",
        json!({"id":"native-no-frame","path":"/tmp/native-no-frame.ome.zarr"}),
    ));
    let deadline = Instant::now() + Duration::from_secs(2);
    while ingress.contains_pending("project.rois.add") && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(2));
    }
    assert!(!ingress.contains_pending("project.rois.add"));

    let reply = runtime
        .submit_native_command_with_reply(&ctx, "project.rois.list", json!({}))
        .expect("query actor after native ingress command");
    let project = reply
        .recv_timeout(Duration::from_secs(2))
        .expect("actor query reply")
        .expect("actor query succeeds");
    assert_eq!(project["rois"][0]["id"], "native-no-frame");
}

#[test]
fn tcp_bridge_validates_envelopes_and_roundtrips_app_replies() {
    let bridge = OdonControlBridge::spawn("127.0.0.1:0", egui::Context::default())
        .expect("spawn bridge on ephemeral port");
    let mut stream = TcpStream::connect(bridge.local_addr()).expect("connect bridge client");
    stream
        .set_read_timeout(Some(Duration::from_secs(2)))
        .expect("set read timeout");
    let mut reader = BufReader::new(stream.try_clone().expect("clone bridge socket"));

    writeln!(stream, "{{").expect("write malformed JSON");
    let malformed = read_json(&mut reader);
    assert_eq!(malformed["jsonrpc"], "2.0");
    assert!(
        malformed["error"]["message"]
            .as_str()
            .unwrap()
            .contains("invalid JSON")
    );
    assert_eq!(malformed["error"]["code"], -32700);

    writeln!(stream, "{}", json!({"params": {}})).expect("write missing method");
    let missing = read_json(&mut reader);
    assert_eq!(missing, json!({"ok": false, "error": "missing method"}));

    // Application methods now execute through the actor and must not require
    // the native UI receiver to be drained.
    writeln!(
        stream,
        "{}",
        json!({"method": "app.get_state", "params": {}})
    )
    .expect("write valid request");
    stream.flush().expect("flush valid request");
    let response = read_json(&mut reader);
    assert_eq!(response["ok"], true);
    assert_eq!(response["result"]["mode"], "project");
}

#[test]
fn json_rpc_requires_hello_and_exposes_introspection() {
    let (tx, _rx) = crossbeam_channel::unbounded();
    let ctx = egui::Context::default();
    let mut state = ConnectionState::unauthenticated_test();

    let before_hello = handle_control_line(
        &json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "get_current_view",
            "params": {}
        })
        .to_string(),
        &tx,
        &ctx,
        &mut state,
    )
    .expect("request response");
    assert_eq!(before_hello["error"]["data"]["kind"], "HANDSHAKE_REQUIRED");

    let hello = handle_control_line(
        &json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "system.hello",
            "params": {
                "client": {"name": "bridge-test", "version": "1.0.0"},
                "protocol_versions": [1]
            }
        })
        .to_string(),
        &tx,
        &ctx,
        &mut state,
    )
    .expect("hello response");
    assert_eq!(hello["result"]["protocol_version"], 1);
    assert!(state.hello_complete);

    let methods = handle_control_line(
        &json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "system.list_methods",
            "params": {}
        })
        .to_string(),
        &tx,
        &ctx,
        &mut state,
    )
    .expect("method response");
    assert!(
        methods["result"]["methods"]
            .as_array()
            .is_some_and(|methods| methods.iter().any(|method| method["name"] == "get_camera"))
    );
}

#[test]
fn authenticated_connections_execute_actor_requests_without_ui_delivery() {
    let bridge = OdonControlBridge::spawn("127.0.0.1:0", egui::Context::default())
        .expect("spawn bridge on ephemeral port");
    let mut stream = TcpStream::connect(bridge.local_addr()).expect("connect bridge client");
    stream
        .set_read_timeout(Some(Duration::from_secs(2)))
        .expect("set read timeout");
    let mut reader = BufReader::new(stream.try_clone().expect("clone bridge socket"));

    writeln!(
        stream,
        "{}",
        json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "system.hello",
            "params": {
                "client": {"name": "concurrency-test", "version": "1"},
                "protocol_versions": [1]
            }
        })
    )
    .expect("write hello");
    stream.flush().expect("flush hello");
    assert_eq!(read_json(&mut reader)["id"], 1);

    writeln!(
        stream,
        "{}",
        json!({"jsonrpc": "2.0", "id": 2, "method": "app.get_state", "params": {}})
    )
    .expect("write first request");
    writeln!(
        stream,
        "{}",
        json!({"jsonrpc": "2.0", "id": 3, "method": "app.get_loading_state", "params": {}})
    )
    .expect("write second request");
    stream.flush().expect("flush concurrent requests");

    let first = read_json(&mut reader);
    let second = read_json(&mut reader);
    let responses = [first, second]
        .into_iter()
        .map(|response| (response["id"].as_u64().unwrap(), response))
        .collect::<std::collections::HashMap<_, _>>();
    assert_eq!(responses.len(), 2);
    assert_eq!(responses[&2]["result"]["mode"], "project");
    assert_eq!(responses[&3]["result"]["mode"], "project");
}

#[test]
fn comparison_workflow_completes_over_tcp_without_a_ui_frame() {
    let bridge = OdonControlBridge::spawn("127.0.0.1:0", egui::Context::default())
        .expect("spawn bridge on ephemeral port");
    let mut stream = TcpStream::connect(bridge.local_addr()).expect("connect bridge client");
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .expect("set read timeout");
    let mut reader = BufReader::new(stream.try_clone().expect("clone bridge socket"));
    let mut next_id = 1_u64;
    let mut rpc = |method: &str, params: Value| {
        let id = next_id;
        next_id += 1;
        writeln!(
            stream,
            "{}",
            json!({"jsonrpc":"2.0", "id":id, "method":method, "params":params})
        )
        .expect("write request");
        stream.flush().expect("flush request");
        let response = read_json(&mut reader);
        assert_eq!(response["id"], id);
        assert!(response.get("error").is_none(), "RPC error: {response}");
        response["result"].clone()
    };

    rpc(
        "system.hello",
        json!({"client":{"name":"paused-render-test","version":"1"}, "protocol_versions":[1]}),
    );
    rpc(
        "project.create",
        json!({"default_dataset":"paused-frame-project"}),
    );
    for (id, path) in [("roi-a", "/tmp/a.zarr"), ("roi-b", "/tmp/b.zarr")] {
        rpc(
            "project.rois.add",
            json!({"id":id,"path":path,"metadata":{"condition":"paused"}}),
        );
    }
    rpc(
        "project.rois.select",
        json!({"ids":["roi-b"],"mode":"replace"}),
    );
    rpc(
        "data.resources.register",
        json!({
            "resource_id":"resource:paused-project",
            "uri":"file:///tmp/paused-project.geojson",
            "format":"geojson",
            "ownership":"project",
            "coordinate_space":{"axes":["y","x"]}
        }),
    );
    rpc(
        "viewer.layers.add",
        json!({
            "layer_id":"layer:paused-project",
            "name":"Paused project layer",
            "kind":"shapes",
            "data_resource_id":"resource:paused-project",
            "ownership":"project"
        }),
    );
    let saved_project = std::env::temp_dir().join(format!(
        "odon-paused-frame-project-{}-{}.json",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    rpc("project.save_as", json!({"path":saved_project}));
    let saved_value: Value =
        serde_json::from_str(&std::fs::read_to_string(&saved_project).unwrap()).unwrap();
    assert_eq!(
        saved_value["config"]["control_resources"][0]["resource_id"],
        "resource:paused-project"
    );
    assert_eq!(
        saved_value["config"]["control_layers"][0]["layer_id"],
        "layer:paused-project"
    );
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/synthetic_5ch.ome.zarr");
    let task = rpc(
        "tasks.start",
        json!({"method":"datasets.open_ome_zarr", "params":{"path":fixture}}),
    );
    let task_id = task["task_id"].as_str().expect("task id").to_string();
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let task = rpc("tasks.get", json!({"task_id":task_id}));
        match task["state"].as_str() {
            Some("completed") => break,
            Some("failed" | "cancelled") => panic!("open task did not complete: {task}"),
            _ if Instant::now() < deadline => thread::sleep(Duration::from_millis(10)),
            _ => panic!("open task timed out without a UI frame: {task}"),
        }
    }

    let channels = rpc("viewer.channels.list", json!({}));
    assert!(
        channels["channels"]
            .as_array()
            .is_some_and(|items| items.len() >= 2)
    );
    let intensity = rpc(
        "viewer.channels.intensity_stats",
        json!({"channel":0,"level":0}),
    );
    assert!(intensity["n"].as_u64().is_some_and(|count| count > 0));
    let workspace = rpc("viewer.workspace.get", json!({}));
    let left = workspace["active_viewport_id"]
        .as_str()
        .unwrap()
        .to_string();
    rpc(
        "viewer.viewports.rename",
        json!({"viewport_id":left, "title":"Channel A"}),
    );
    let cloned = rpc(
        "viewer.viewports.clone",
        json!({"viewport_id":left, "title":"Channel B", "layout":"horizontal", "ratio":0.5}),
    );
    let right = cloned["viewport_id"].as_str().unwrap().to_string();
    rpc(
        "viewer.viewport_links.create",
        json!({"viewports":[left,right], "fields":["camera","plane","selection"]}),
    );
    rpc(
        "viewer.viewports.channels.set_visible",
        json!({"viewport_id":left,"channels":[0],"mode":"only"}),
    );
    rpc(
        "viewer.viewports.channels.set_visible",
        json!({"viewport_id":right,"channels":[1],"mode":"only"}),
    );
    rpc(
        "viewer.viewports.channels.set_color",
        json!({"viewport_id":left,"channel":0,"color_rgb":[80,140,255]}),
    );
    rpc(
        "viewer.viewports.channels.set_color",
        json!({"viewport_id":right,"channel":1,"color_rgb":[255,90,120]}),
    );
    rpc(
        "viewer.channels.set_note",
        json!({"channel":1,"note":"right comparison marker"}),
    );
    rpc(
        "viewer.channels.set_transform",
        json!({"channel":1,"offset_world":[2.0,-1.0],"scale":[1.1,0.9],"rotation_rad":0.1}),
    );
    rpc(
        "viewer.viewports.channels.set_order",
        json!({"viewport_id":left,"channels":[4,3,2,1,0],"mode":"exact"}),
    );
    rpc(
        "viewer.viewports.channels.set_group",
        json!({"viewport_id":right,"channels":[0,1],"name":"Comparison"}),
    );
    rpc(
        "viewer.channels.presentation.set",
        json!({"search":"CD","sort":"visible_first"}),
    );
    rpc("viewer.panels.set", json!({"left":false,"right":false}));
    rpc(
        "viewer.viewports.rendering.set",
        json!({"viewport_id":left,"smooth_pixels":false,"show_hud":true}),
    );
    rpc(
        "viewer.viewports.rendering.set",
        json!({"viewport_id":right,"smooth_pixels":true,"show_hud":true}),
    );
    let fitted = rpc("viewer.viewports.camera.fit", json!({"viewport_id":left}));
    assert!(
        fitted["result"]["zoom_screen_per_lvl0_px"]
            .as_f64()
            .unwrap()
            > 0.0
    );
    let mut final_workspace = rpc("viewer.workspace.get", json!({}));
    let final_project = rpc("project.get", json!({}));
    let final_rois = rpc("project.rois.list", json!({}));
    assert_eq!(final_workspace["layout"], "horizontal");
    assert_eq!(final_workspace["viewports"].as_array().unwrap().len(), 2);
    assert_eq!(
        final_project["metadata"]["default_dataset"],
        "paused-frame-project"
    );
    assert_eq!(final_rois["roi_count"], 2);
    assert_eq!(final_rois["rois"][1]["selected"], true);

    assert_eq!(
        bridge.pending_presentation_len(),
        1,
        "covered-window updates should coalesce to one latest render projection"
    );
    let projection = bridge
        .try_recv_presentation()
        .expect("latest render projection remains available");
    assert!(projection.document.is_some());
    assert_eq!(projection.project.rois.len(), 2);
    assert_eq!(projection.project.selected_source_keys.len(), 1);
    final_workspace.as_object_mut().unwrap().remove("_control");
    assert_eq!(projection.workspace.unwrap(), final_workspace);
    let diagnostics = rpc("system.get_diagnostics", json!({}));
    for method in [
        "viewer.channels.set_note",
        "viewer.channels.intensity_stats",
        "viewer.channels.set_transform",
        "viewer.viewports.channels.set_order",
        "viewer.viewports.channels.set_group",
        "viewer.channels.presentation.set",
        "viewer.panels.set",
        "project.create",
        "project.rois.add",
        "project.rois.select",
        "project.save_as",
    ] {
        assert_eq!(
            diagnostics["dispatch"]["method_routes"][method], "actor",
            "{method} must not depend on RootApp::update"
        );
    }
    let _ = std::fs::remove_file(saved_project);
}

#[test]
fn protocol_registries_roundtrip_data_layers_and_declarative_ui() {
    let ctx = egui::Context::default();
    let mut state = ConnectionState::unauthenticated_test();
    let actor = crate::control::actor::spawn_control_actor(
        Arc::new(|| {}),
        Arc::clone(&state.resource_registry),
    )
    .unwrap();
    let tx = actor.request_tx;

    let call = |state: &mut ConnectionState, id: u64, method: &str, params: Value| {
        handle_control_line(
            &json!({
                "jsonrpc": "2.0", "id": id, "method": method, "params": params
            })
            .to_string(),
            &tx,
            &ctx,
            state,
        )
        .expect("JSON-RPC response")
    };
    assert!(
        call(
            &mut state,
            1,
            "system.hello",
            json!({
                "client": {"name": "conformance", "version": "1"},
                "protocol_versions": [1]
            }),
        )["result"]
            .is_object()
    );
    let methods = call(&mut state, 2, "system.list_methods", json!({}));
    assert!(
        methods["result"]["methods"]
            .as_array()
            .is_some_and(|items| { items.iter().any(|item| item["name"] == "viewer.camera.fit") })
    );
    let resource = call(
        &mut state,
        3,
        "data.resources.register",
        json!({
            "resource_id": "resource:test", "uri": "file:///tmp/test.zarr",
            "format": "ome-zarr",
            "coordinate_space": {"axes": ["y", "x"], "scale": [1.0, 1.0]}
        }),
    );
    assert_eq!(resource["result"]["resource_id"], "resource:test");
    let layer = call(
        &mut state,
        4,
        "viewer.layers.add",
        json!({
            "layer_id": "layer:test", "name": "Test", "kind": "labels",
            "data_resource_id": "resource:test"
        }),
    );
    assert_eq!(layer["result"]["layer_id"], "layer:test");
    let extension = call(
        &mut state,
        5,
        "ui.extensions.register",
        json!({
            "id": "org.example.test", "name": "Test", "version": "1",
            "capabilities": ["ui.panels"]
        }),
    );
    assert_eq!(extension["result"]["id"], "org.example.test");
    let contribution = call(
        &mut state,
        6,
        "ui.contributions.register",
        json!({
            "extension_id": "org.example.test", "location": "right.tabs",
            "root": {"id": "root", "type": "panel", "children": [
                {"id": "run", "type": "button", "label": "Run",
                 "action": {"type": "emit", "event": "run"}}
            ]}
        }),
    );
    assert_eq!(contribution["result"]["extension_id"], "org.example.test");
}

#[test]
fn application_readiness_prefers_actor_work_readiness_and_accepts_older_canvas_shapes() {
    let loading_without_canvas = json!({
        "mode": "single",
        "loading": {"busy": false, "canvas_ready": false},
    });
    assert!(!application_state_is_ready(
        &loading_without_canvas,
        "single"
    ));

    let ready = json!({
        "mode": "single",
        "loading": {"busy": false, "canvas_ready": true},
    });
    assert!(application_state_is_ready(&ready, "single"));

    let background_ready = json!({
        "mode": "single",
        "loading": {
            "busy": false,
            "model_ready": true,
            "resources_ready": true,
            "geometry_ready": true,
            "presentation_ready": false,
            "canvas_ready": false,
        },
    });
    assert!(application_state_is_ready(&background_ready, "single"));

    let still_loading = json!({
        "mode": "single",
        "loading": {"busy": true, "canvas_ready": true},
    });
    assert!(!application_state_is_ready(&still_loading, "single"));

    let project = json!({"mode": "project", "busy": false});
    assert!(application_state_is_ready(&project, "project"));
}
