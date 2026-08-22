use super::*;
use crate::control::{ControlCommand, EventHub, TaskRegistry};
use std::time::Duration;

fn request(
    method: &str,
    params: Value,
) -> (OdonControlRequest, Receiver<Result<Value, ControlError>>) {
    let event_hub = EventHub::shared();
    let tasks = TaskRegistry::shared(Arc::clone(&event_hub));
    let (reply, rx) = crossbeam_channel::bounded(1);
    (
        OdonControlRequest {
            command: ControlCommand::decode(method, params).unwrap(),
            reply,
            session_id: "test".to_string(),
            request_id: None,
            event_hub,
            task_registry: tasks,
            task_id: None,
        },
        rx,
    )
}

fn spawn_test_actor() -> ControlActorChannels {
    let events = EventHub::shared();
    let resources = ResourceRegistry::shared(events);
    spawn_control_actor(Arc::new(|| {}), resources).unwrap()
}

fn spawn_test_actor_with_objects() -> ControlActorChannels {
    let events = EventHub::shared();
    let resources = ResourceRegistry::shared(events);
    struct TestObjectLoader;
    impl ObjectResourceLoader for TestObjectLoader {
        fn load(
            &self,
            path: PathBuf,
            downsample_factor: f32,
        ) -> anyhow::Result<ControlObjectResource> {
            Ok(ControlObjectResource {
                source: path,
                downsample_factor,
                features: Arc::new(vec![
                    crate::model::ControlObjectFeature {
                        id: "cell-a".to_string(),
                        bbox_world: [0.0, 0.0, 10.0, 10.0],
                        centroid_world: [5.0, 5.0],
                        polygons_world: Arc::new(vec![vec![
                            [0.0, 0.0],
                            [10.0, 0.0],
                            [10.0, 10.0],
                            [0.0, 10.0],
                            [0.0, 0.0],
                        ]]),
                        point_position_world: Some([5.0, 5.0]),
                        area_px: 100.0,
                        perimeter_px: 40.0,
                        properties: json!({"phenotype":"tumour","score":0.9})
                            .as_object()
                            .unwrap()
                            .clone(),
                    },
                    crate::model::ControlObjectFeature {
                        id: "cell-b".to_string(),
                        bbox_world: [20.0, 20.0, 30.0, 30.0],
                        centroid_world: [25.0, 25.0],
                        polygons_world: Arc::new(vec![vec![
                            [20.0, 20.0],
                            [30.0, 20.0],
                            [30.0, 30.0],
                            [20.0, 30.0],
                            [20.0, 20.0],
                        ]]),
                        point_position_world: Some([25.0, 25.0]),
                        area_px: 100.0,
                        perimeter_px: 40.0,
                        properties: json!({"phenotype":"immune","score":0.2})
                            .as_object()
                            .unwrap()
                            .clone(),
                    },
                ]),
                property_names: Arc::new(vec![
                    "id".to_string(),
                    "phenotype".to_string(),
                    "score".to_string(),
                ]),
                renderer_payload: None,
            })
        }

        fn evaluate_filter(
            &self,
            resource: Arc<ControlObjectResource>,
            model: Value,
        ) -> anyhow::Result<ControlObjectFilterResult> {
            let query = model
                .get("query")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let matching_indices = resource
                .features
                .iter()
                .enumerate()
                .filter_map(|(index, feature)| {
                    (query.is_empty()
                        || feature
                            .properties
                            .get("phenotype")
                            .and_then(Value::as_str)
                            .is_some_and(|value| query.contains(value)))
                    .then_some(index)
                })
                .collect::<Vec<_>>();
            Ok(ControlObjectFilterResult {
                model: json!({"mode":"query","query":query}),
                matching_indices: Arc::new(matching_indices),
                active: !query.is_empty(),
            })
        }
    }
    let loader: Arc<dyn ObjectResourceLoader> = Arc::new(TestObjectLoader);
    spawn_control_actor_with_object_loader(Arc::new(|| {}), resources, Some(loader)).unwrap()
}

fn spawn_test_actor_with_remote(backend: Arc<dyn RemoteDatasetBackend>) -> ControlActorChannels {
    let events = EventHub::shared();
    let resources = ResourceRegistry::shared(events);
    spawn_control_actor_with_services(
        Arc::new(|| {}),
        resources,
        None,
        None,
        None,
        Some(backend),
        None,
    )
    .unwrap()
}

fn spawn_test_actor_with_alternate(
    backend: Arc<dyn AlternateDatasetBackend>,
) -> ControlActorChannels {
    let events = EventHub::shared();
    let resources = ResourceRegistry::shared(events);
    spawn_control_actor_with_services(
        Arc::new(|| {}),
        resources,
        None,
        None,
        None,
        None,
        Some(backend),
    )
    .unwrap()
}

mod alternate_datasets;
mod analysis;
mod backpressure;
mod dataset_inspection;
mod deep_link_apply;
mod deep_link_resolution;
mod deep_link_state;
mod events;
mod labels;
mod lifecycle;
mod masks;
mod measurements;
mod memory;
mod mosaics;
mod object_exports;
mod objects;
mod project_preload;
mod project_roi_open;
mod project_rois;
mod project_roundtrip;
mod readiness;
mod remote;
mod routing;
mod samplesheets;
mod saved_views;
mod screenshots;
mod settings;
mod task_cancellation;
mod thresholds;
mod workspace;
