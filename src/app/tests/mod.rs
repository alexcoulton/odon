use super::*;
use odon::model::AppModel;

mod helpers;
use helpers::*;

fn fixture_app() -> OmeZarrViewerApp {
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/synthetic_5ch.ome.zarr");
    let (dataset, store) = OmeZarrDataset::open_local(&fixture).expect("open OME-Zarr fixture");
    OmeZarrViewerApp::new_runtime(&egui::Context::default(), false, dataset, store)
}

fn fixture_actor_app() -> ActorAppFixture {
    ActorAppFixture::new(fixture_app())
}

fn visible_channel_names(app: &OmeZarrViewerApp) -> Vec<String> {
    app.control_visible_channel_snapshot()
        .as_array()
        .expect("visible channel array")
        .iter()
        .map(|channel| {
            channel["name"]
                .as_str()
                .expect("visible channel name")
                .to_string()
        })
        .collect()
}

mod active_compatibility;
mod active_keys;
mod actor_viewport;
mod actor_workspace;
mod camera_fit;
mod canvas_union;
mod channel_controls;
mod channel_groups;
mod comparison;
mod comparison_workflow;
mod deep_link_apply;
mod deep_link_channels;
mod explicit_layers;
mod frame_benchmark;
mod header_geometry;
mod labels;
mod legacy_project_view;
mod native_layers;
mod native_masks;
mod native_selection;
mod native_viewport;
mod object_loader;
mod objects;
mod overlays;
mod project_actions;
mod project_roundtrip;
mod projection;
mod rapid_comparison;
mod removing_viewport;
mod rendering_preferences;
mod scheduler;
mod screenshots;
mod source_organization;
mod transient_gestures;
mod viewport_lifecycle;
mod viewport_links;
mod viewport_revisions;
mod workspace_persistence;
