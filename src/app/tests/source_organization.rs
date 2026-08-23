use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

fn source(path: impl AsRef<Path>) -> String {
    fs::read_to_string(path.as_ref())
        .unwrap_or_else(|error| panic!("read {}: {error}", path.as_ref().display()))
}

fn rust_files(path: &Path) -> Vec<PathBuf> {
    let mut files = fs::read_dir(path)
        .unwrap_or_else(|error| panic!("read {}: {error}", path.display()))
        .map(|entry| entry.expect("directory entry").path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "rs"))
        .collect::<Vec<_>>();
    files.sort();
    files
}

fn struct_field_names(contents: &str, struct_name: &str) -> BTreeSet<String> {
    let marker = format!("pub struct {struct_name} {{");
    let body = contents
        .split_once(&marker)
        .unwrap_or_else(|| panic!("missing {marker}"))
        .1;
    let mut fields = BTreeSet::new();
    for line in body.lines() {
        let line = line.trim();
        if line == "}" {
            break;
        }
        if line.is_empty() || line.starts_with("#") {
            continue;
        }
        let Some((candidate, _)) = line.split_once(':') else {
            continue;
        };
        let candidate = candidate.trim();
        if !candidate.is_empty()
            && candidate
                .chars()
                .all(|character| character == '_' || character.is_ascii_alphanumeric())
        {
            assert!(
                fields.insert(candidate.to_string()),
                "duplicate source field {struct_name}.{candidate}"
            );
        }
    }
    fields
}

#[test]
fn app_source_stays_split_by_responsibility() {
    let app_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/app");
    let facade = source(app_dir.join("mod.rs"));
    assert!(
        facade.lines().count() <= 4_000,
        "src/app/mod.rs has regrown into an implementation monolith"
    );
    assert!(
        !facade.contains("impl eframe::App for OmeZarrViewerApp"),
        "the frame lifecycle belongs in app/update.rs"
    );

    for path in rust_files(&app_dir) {
        let contents = source(&path);
        assert!(
            contents.lines().count() <= 3_500,
            "{} is too large; split it at a responsibility boundary",
            path.display()
        );
        assert!(
            !contents.contains("#[test]"),
            "production app tests belong under app/tests: {}",
            path.display()
        );
    }
}

#[test]
fn layer_runtime_stays_split_by_responsibility() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let facade = source(root.join("src/app/layer_runtime.rs"));
    let module_dir = root.join("src/app/layer_runtime");

    assert!(
        facade.lines().count() <= 40,
        "app/layer_runtime.rs must remain a responsibility-module façade"
    );
    assert!(!facade.contains("impl OmeZarrViewerApp"));
    assert!(!facade.contains("#[test]"));

    for path in rust_files(&module_dir) {
        let contents = source(&path);
        assert!(
            contents.lines().count() <= 500,
            "{} is too large; split the layer runtime at a command, UI, metadata, offset, geometry, or ordering boundary",
            path.display()
        );
        assert!(contents.contains("impl OmeZarrViewerApp"));
        assert!(
            !contents.contains("#[test]"),
            "layer-runtime behavior tests belong under app/tests: {}",
            path.display()
        );
    }

    let commands = source(module_dir.join("commands.rs"));
    let contrast = source(module_dir.join("contrast.rs"));
    let metadata = source(module_dir.join("metadata.rs"));
    let offsets = source(module_dir.join("offsets.rs"));
    let geometry = source(module_dir.join("geometry.rs"));
    let ordering = source(module_dir.join("ordering.rs"));
    assert!(commands.contains("fn submit_native_layer_visibility("));
    assert!(!commands.contains("fn ui_top_bar_quick_contrast("));
    assert!(contrast.contains("fn ui_top_bar_quick_contrast("));
    assert!(!contrast.contains("fn layer_offset_world("));
    assert!(metadata.contains("fn layer_display_name("));
    assert!(offsets.contains("fn commit_layer_offsets("));
    assert!(geometry.contains("fn union_visible_world_for_visible_channels_xform("));
    assert!(ordering.contains("fn rebuild_layer_orders("));
}

#[test]
fn annotation_points_layer_stays_split_by_responsibility() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let facade = source(root.join("src/annotations/mod.rs"));
    let module_dir = root.join("src/annotations");

    assert!(
        facade.lines().count() <= 1_100,
        "annotations/mod.rs has regrown into a UI, hit-test, color, parquet, and GL monolith"
    );
    assert!(facade.contains("impl AnnotationPointsLayer"));
    assert!(!facade.contains("ParquetRecordBatchReaderBuilder"));
    assert!(!facade.contains("struct PointsRadius"));
    assert!(!facade.contains("fn turbo_rgb_u8("));
    assert!(!facade.contains("#[test]"));

    for path in rust_files(&module_dir) {
        let contents = source(&path);
        assert!(
            contents.lines().count() <= 1_000,
            "{} is too large; split the annotation adapter at a model, selection, color, parquet, or GL boundary",
            path.display()
        );
        assert!(
            !contents.contains("#[test]"),
            "annotation behavior tests belong in a dedicated test module: {}",
            path.display()
        );
    }

    let selection = source(module_dir.join("selection.rs"));
    let colors = source(module_dir.join("colors.rs"));
    let parquet = source(root.join("src/data/annotations/parquet.rs"));
    let gl = source(module_dir.join("gl.rs"));
    assert!(selection.contains("fn pick_nearest_in_roi("));
    assert!(!selection.contains("ParquetRecordBatchReaderBuilder"));
    assert!(colors.contains("pub fn build_category_luts("));
    assert!(parquet.contains("fn load_annotations_parquet("));
    assert!(parquet.contains("ParquetRecordBatchReaderBuilder"));
    assert!(gl.contains("struct AnnotationGlRenderer"));
    assert!(!gl.contains("fn load_annotations_parquet("));
}

#[test]
fn tile_gl_renderer_stays_split_by_responsibility() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let facade = source(root.join("src/render/tiles_gl.rs"));
    let module_dir = root.join("src/render/tiles_gl");

    assert!(
        facade.lines().count() <= 250,
        "render/tiles_gl.rs has regrown into a paint, resource, geometry, upload, and shader monolith"
    );
    assert!(facade.contains("pub struct TilesGl"));
    assert!(!facade.contains("impl TilesGl {"));
    assert!(!facade.contains("struct Inner"));
    assert!(!facade.contains("const VERT_330"));
    assert!(!facade.contains("#[test]"));

    for path in rust_files(&module_dir) {
        let contents = source(&path);
        assert!(
            contents.lines().count() <= 800,
            "{} is too large; split the tile renderer at an orchestration, resource, geometry, upload, or shader boundary",
            path.display()
        );
        assert!(
            !contents.contains("#[test]"),
            "tile renderer behavior tests belong in a dedicated render test module: {}",
            path.display()
        );
    }

    let orchestration = source(module_dir.join("orchestration.rs"));
    let backend = source(module_dir.join("backend.rs"));
    let geometry = source(module_dir.join("geometry.rs"));
    let upload = source(module_dir.join("upload.rs"));
    let shaders = source(module_dir.join("shaders.rs"));
    assert!(orchestration.contains("impl TilesGl {"));
    assert!(orchestration.contains("pub fn paint_with_channel_transforms_screen("));
    assert!(!orchestration.contains("const VERT_330"));
    assert!(backend.contains("struct Inner"));
    assert!(backend.contains("struct GlObjects"));
    assert!(!backend.contains("const VERT_330"));
    assert!(geometry.contains("fn tile_vertices_ndc("));
    assert!(!geometry.contains("create_texture"));
    assert!(upload.contains("fn upload_r16_texture("));
    assert!(!upload.contains("compile_program"));
    assert!(shaders.contains("fn compile_program("));
    assert!(shaders.contains("const VERT_330"));
    assert!(!shaders.contains("RawTileCache"));
}

#[test]
fn line_bin_renderers_stay_split_by_responsibility() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let facade = source(root.join("src/render/line_bins_gl.rs"));
    let module_dir = root.join("src/render/line_bins_gl");

    assert!(
        facade.lines().count() <= 30,
        "render/line_bins_gl.rs must remain a public renderer façade"
    );
    assert!(facade.contains("pub use bins::"));
    assert!(facade.contains("pub use objects::"));
    assert!(!facade.contains("impl LineBinsGlRenderer"));
    assert!(!facade.contains("const VERT_330"));
    assert!(!facade.contains("#[test]"));

    for path in rust_files(&module_dir) {
        let contents = source(&path);
        assert!(
            contents.lines().count() <= 700,
            "{} is too large; split line-bin rendering at a generic, object-state, or GL-program boundary",
            path.display()
        );
        assert!(
            !contents.contains("#[test]"),
            "line-bin renderer tests belong in a dedicated render test module: {}",
            path.display()
        );
    }

    let bins = source(module_dir.join("bins.rs"));
    let objects = source(module_dir.join("objects.rs"));
    let program = source(module_dir.join("program.rs"));
    assert!(bins.contains("pub struct LineBinsGlRenderer"));
    assert!(bins.contains("const VERT_330"));
    assert!(!bins.contains("pub struct ObjectLineBinsGlRenderer"));
    assert!(objects.contains("pub struct ObjectLineBinsGlRenderer"));
    assert!(objects.contains("const OBJECT_LINE_VERT_330"));
    assert!(!objects.contains("pub struct LineBinsGlRenderer"));
    assert!(program.contains("fn compile_program("));
    assert!(program.contains("fn compile_program_with_attributes("));
    assert!(!program.contains("const VERT_330"));
    assert!(!program.contains("LruCache"));
}

#[test]
fn viewport_canvas_establishes_a_hard_clip_before_painting() {
    let source = include_str!("../canvas.rs");
    let allocation = source
        .find("ui.allocate_exact_size(available, egui::Sense::click_and_drag())")
        .expect("viewport canvas allocation");
    let clipping = source
        .find("ui.shrink_clip_rect(rect)")
        .expect("viewport canvas hard clip");
    let background = source
        .find(".rect_filled(rect, 0.0, egui::Color32::from_gray(10))")
        .expect("viewport canvas background paint");
    assert!(allocation < clipping);
    assert!(clipping < background);

    assert!(
        source.lines().count() <= 1_500,
        "app/canvas.rs has regrown into an interaction/render/capture monolith"
    );
    assert!(!source.contains("let can_lasso_select"));
    assert!(!source.contains("let can_transform_primary"));

    let interactions = include_str!("../canvas/interactions.rs");
    assert!(interactions.lines().count() <= 800);
    assert!(interactions.contains("fn handle_canvas_interactions("));
    assert!(!interactions.contains("ui.shrink_clip_rect(rect)"));
    assert!(!interactions.contains("ScreenshotWorkerMsg::SavePng"));
}

#[test]
fn root_app_source_stays_split_by_responsibility() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let facade_path = root.join("src/root_app.rs");
    let facade = source(&facade_path);
    assert!(
        facade.lines().count() <= 4_000,
        "src/root_app.rs has regrown into a frame-orchestration monolith"
    );
    assert!(
        !facade.contains("#[test]"),
        "RootApp tests belong under root_app/tests.rs"
    );

    let module_dir = root.join("src/root_app");
    for path in rust_files(&module_dir) {
        if path.file_name().is_some_and(|name| name == "tests.rs") {
            continue;
        }
        let contents = source(&path);
        assert!(
            contents.lines().count() <= 2_000,
            "{} is too large; split it at a responsibility boundary",
            path.display()
        );
        assert!(
            !contents.contains("#[test]"),
            "production RootApp tests belong under root_app/tests.rs: {}",
            path.display()
        );
    }
}

#[test]
fn canonical_app_model_stays_split_by_domain() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let facade = source(root.join("src/model/app.rs"));
    assert!(
        facade.lines().count() <= 3_500,
        "src/model/app.rs has regrown into a semantic implementation monolith"
    );
    assert!(
        !facade.contains("impl AppModel {"),
        "AppModel behavior belongs in responsibility modules under model/app"
    );

    let module_dir = root.join("src/model/app");
    for path in rust_files(&module_dir) {
        if path.file_name().is_some_and(|name| name == "tests.rs") {
            continue;
        }
        let contents = source(&path);
        assert!(
            contents.lines().count() <= 2_500,
            "{} is too large; split it at a model-domain boundary",
            path.display()
        );
        assert!(
            !contents.contains("#[test]"),
            "AppModel tests belong under model/app/tests.rs: {}",
            path.display()
        );
    }

    let viewport_commands = source(module_dir.join("viewport_commands.rs"));
    assert!(
        viewport_commands.lines().count() <= 1_000,
        "viewport_commands.rs has regrown into a topology/navigation/presentation monolith"
    );
    assert!(!viewport_commands.contains("fn workspace_snapshot("));
    assert!(!viewport_commands.contains("fn set_camera(&mut self"));
    assert!(!viewport_commands.contains("fn set_visible_channels("));
    assert!(!viewport_commands.contains("#[test]"));

    for path in rust_files(&module_dir.join("viewport_commands")) {
        let contents = source(&path);
        assert!(
            contents.lines().count() <= 800,
            "{} is too large; split it at a viewport-command domain boundary",
            path.display()
        );
        assert!(
            !contents.contains("#[test]"),
            "viewport command tests belong under model/app/tests.rs: {}",
            path.display()
        );
    }
}

#[test]
fn mosaic_app_stays_split_by_responsibility() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let facade = source(root.join("src/mosaic/mod.rs"));
    assert!(
        facade.lines().count() <= 1_200,
        "src/mosaic/mod.rs has regrown into a multi-ROI implementation monolith"
    );
    assert!(!facade.contains("impl MosaicViewerApp {"));
    assert!(!facade.contains("impl eframe::App for MosaicViewerApp"));
    assert!(!facade.contains("#[test]"));

    for path in rust_files(&root.join("src/mosaic")) {
        if path
            .file_name()
            .is_some_and(|name| name == "mod.rs" || name == "tests.rs")
        {
            continue;
        }
        let contents = source(&path);
        assert!(
            contents.lines().count() <= 2_500,
            "{} is too large; split it at a mosaic responsibility boundary",
            path.display()
        );
    }

    let construction = source(root.join("src/mosaic/construction.rs"));
    assert!(
        construction.lines().count() <= 150,
        "mosaic construction façade has regrown into source-specific implementation"
    );
    assert!(!construction.contains("fn from_control_resource("));
    assert!(!construction.contains("fn from_local_paths("));
    assert!(!construction.contains("fn from_remote_s3_sources("));
    assert!(!construction.contains("fn from_project_rois("));
    assert!(!construction.contains("Ok(Self {"));

    let construction_dir = root.join("src/mosaic/construction");
    for path in rust_files(&construction_dir) {
        let contents = source(&path);
        assert!(
            contents.lines().count() <= 700,
            "{} is too large; split it at a mosaic-construction boundary",
            path.display()
        );
        assert!(
            !contents.contains("#[test]"),
            "mosaic construction tests belong under mosaic/tests.rs: {}",
            path.display()
        );
    }

    let assembly = source(construction_dir.join("assembly.rs"));
    assert!(assembly.contains("struct PreparedMosaicConstruction"));
    assert!(assembly.contains("fn from_prepared_construction("));
    for adapter in [
        "actor_resource.rs",
        "config.rs",
        "local.rs",
        "project.rs",
        "remote.rs",
        "samplesheet.rs",
    ] {
        let contents = source(construction_dir.join(adapter));
        assert!(contents.contains("PreparedMosaicConstruction"));
        assert!(
            !contents.contains("tiles_gl:"),
            "shared MosaicViewerApp defaults belong in construction/assembly.rs: {adapter}"
        );
    }

    let control = source(root.join("src/mosaic/control.rs"));
    let control_dir = root.join("src/mosaic/control");
    assert!(
        control.lines().count() <= 30,
        "mosaic/control.rs must remain a responsibility-module façade"
    );
    assert!(!control.contains("impl MosaicViewerApp {"));
    assert!(!control.contains("#[test]"));
    for path in rust_files(&control_dir) {
        let contents = source(&path);
        assert!(
            contents.lines().count() <= 800,
            "{} is too large; split mosaic control integration at an intent, project, snapshot, screenshot, or host boundary",
            path.display()
        );
        assert!(
            !contents.contains("#[test]"),
            "mosaic control behavior tests belong under mosaic/tests.rs: {}",
            path.display()
        );
    }

    let intents = source(control_dir.join("intents.rs"));
    let project = source(control_dir.join("project.rs"));
    let snapshots = source(control_dir.join("snapshots.rs"));
    let screenshots = source(control_dir.join("screenshots.rs"));
    let host = source(control_dir.join("host.rs"));
    assert!(intents.contains("fn submit_native_control_intent("));
    assert!(!intents.contains("fn set_project_space("));
    assert!(project.contains("fn take_project_space("));
    assert!(project.contains("fn set_project_space("));
    assert!(!project.contains("fn control_channel_snapshot("));
    assert!(snapshots.contains("fn control_channel_snapshot("));
    for retired_renderer_emulator in [
        "fn control_configure_layout(",
        "fn control_set_visible_channels(",
        "fn control_set_channel_contrast(",
        "fn control_set_focused_roi(",
        "fn control_step_focused_roi(",
    ] {
        assert!(
            !snapshots.contains(retired_renderer_emulator),
            "mosaic command semantics must remain actor-only: {retired_renderer_emulator}"
        );
    }
    assert!(
        !source(root.join("src/mosaic/construction/actor_resource.rs"))
            .contains("fn control_actor_semantic_snapshot(")
    );
    assert!(!snapshots.contains("fn request_screenshot_png("));
    assert!(screenshots.contains("fn request_screenshot_png("));
    assert!(screenshots.contains("fn request_actor_screenshot("));
    assert!(host.contains("fn take_request("));
    assert!(host.contains("fn set_fast_object_rendering("));

    let model = source(root.join("src/model/mosaic.rs"));
    assert!(
        model.lines().count() <= 1_400,
        "src/model/mosaic.rs has regrown into a semantic mosaic monolith"
    );
    assert!(!model.contains("fn prepare_memory_pin("));
    assert!(!model.contains("fn configure_layout("));
    assert!(!model.contains("fn channels_snapshot("));
    assert!(!model.contains("#[test]"));

    for path in rust_files(&root.join("src/model/mosaic")) {
        if path.file_name().is_some_and(|name| name == "tests.rs") {
            continue;
        }
        let contents = source(&path);
        assert!(
            contents.lines().count() <= 1_000,
            "{} is too large; split it at a mosaic-model domain boundary",
            path.display()
        );
        assert!(
            !contents.contains("#[test]"),
            "mosaic-model tests belong under model/mosaic/tests.rs: {}",
            path.display()
        );
    }
}

#[test]
fn mask_model_stays_split_by_responsibility() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let facade = source(root.join("src/model/masks.rs"));
    let module_dir = root.join("src/model/masks");

    assert!(
        facade.lines().count() <= 100,
        "model/masks.rs has regrown into a state, command, validation, I/O, and test monolith"
    );
    assert!(facade.contains("pub(crate) struct MaskModel"));
    assert!(!facade.contains("impl MaskModel {"));
    assert!(!facade.contains("#[test]"));

    for path in rust_files(&module_dir) {
        if path.file_name().is_some_and(|name| name == "tests.rs") {
            continue;
        }
        let contents = source(&path);
        assert!(
            contents.lines().count() <= 500,
            "{} is too large; split the mask model at a state, command, validation, or GeoJSON boundary",
            path.display()
        );
        assert!(
            !contents.contains("#[test]"),
            "mask-model tests belong under model/masks/tests.rs: {}",
            path.display()
        );
    }

    let state = source(module_dir.join("state.rs"));
    let commands = source(module_dir.join("commands.rs"));
    let validation = source(module_dir.join("validation.rs"));
    let geojson = source(module_dir.join("geojson.rs"));
    let tests = source(module_dir.join("tests.rs"));
    assert!(state.contains("fn replace("));
    assert!(state.contains("fn reconcile_appended_file("));
    assert!(!state.contains("fn dispatch("));
    assert!(commands.contains("fn dispatch("));
    assert!(commands.contains("fn add_polygon("));
    assert!(!commands.contains("fn load_geojson_mask_polylines("));
    assert!(validation.contains("fn parse_vertices("));
    assert!(validation.contains("fn validate_layers("));
    assert!(!validation.contains("fs::read_to_string"));
    assert!(geojson.contains("fn load_geojson_mask_polylines("));
    assert!(geojson.contains("fn parse_geojson_points("));
    assert!(!geojson.contains("impl MaskModel {"));
    assert!(tests.contains("mask_crud_selection_and_undo_are_renderer_independent"));
    assert!(tests.contains("append_reconciliation_preserves_edits_made_after_the_worker_snapshot"));
}

#[test]
fn deep_link_surface_stays_split_by_responsibility() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let facade = source(root.join("src/deep_link.rs"));
    let module_dir = root.join("src/deep_link");

    assert!(
        facade.lines().count() <= 200,
        "deep_link.rs has regrown into a DTO, serializer, resolver, parser, and test monolith"
    );
    assert!(facade.contains("pub struct DeepLinkRequest"));
    assert!(facade.contains("pub use resolution::"));
    assert!(!facade.contains("pub fn to_url("));
    assert!(!facade.contains("fn parse_color_rgb("));
    assert!(!facade.contains("#[test]"));

    for path in rust_files(&module_dir) {
        if path.file_name().is_some_and(|name| name == "tests.rs") {
            continue;
        }
        let contents = source(&path);
        assert!(
            contents.lines().count() <= 500,
            "{} is too large; split deep links at a canonicalization, resolution, parsing, or semantic boundary",
            path.display()
        );
        assert!(
            !contents.contains("#[test]"),
            "deep-link tests belong under deep_link/tests.rs: {}",
            path.display()
        );
    }

    let canonical = source(module_dir.join("canonical.rs"));
    let resolution = source(module_dir.join("resolution.rs"));
    let parsing = source(module_dir.join("parsing.rs"));
    let semantics = source(module_dir.join("semantics.rs"));
    let tests = source(module_dir.join("tests.rs"));
    assert!(canonical.contains("pub fn to_url("));
    assert!(canonical.contains("fn append_option("));
    assert!(!canonical.contains("fn parse_deep_link("));
    assert!(resolution.contains("pub fn resolve_roi_target("));
    assert!(resolution.contains("ProjectRoi"));
    assert!(!resolution.contains("fn parse_deep_link("));
    assert!(parsing.contains("fn parse_deep_link("));
    assert!(parsing.contains("fn parse_color_rgb("));
    assert!(!parsing.contains("pub fn resolve_roi_target("));
    assert!(semantics.contains("fn object_filter_model("));
    assert!(semantics.contains("fn requested_bundled_label("));
    assert!(tests.contains("canonical_url_round_trips_public_state"));
    assert!(tests.contains("parses_channel_visibility_and_contrast"));
}

#[test]
fn project_model_stays_split_by_responsibility() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let facade = source(root.join("src/model/project.rs"));
    let module_dir = root.join("src/model/project");

    assert!(
        facade.lines().count() <= 100,
        "model/project.rs has regrown into a state, command, and validation monolith"
    );
    assert!(facade.contains("pub struct ProjectModelSnapshot"));
    assert!(facade.contains("pub(crate) struct ProjectModel"));
    assert!(facade.contains("pub(crate) use state::normalized_loaded_project_snapshot"));
    assert!(!facade.contains("impl ProjectModel {"));
    assert!(!facade.contains("#[test]"));

    for path in rust_files(&module_dir) {
        let contents = source(&path);
        assert!(
            contents.lines().count() <= 700,
            "{} is too large; split the project model at a state, command, navigation, or validation boundary",
            path.display()
        );
        assert!(
            !contents.contains("#[test]"),
            "project-model behavior tests belong under model/app/tests.rs: {}",
            path.display()
        );
    }

    let state = source(module_dir.join("state.rs"));
    let commands = source(module_dir.join("commands.rs"));
    let validation = source(module_dir.join("validation.rs"));
    assert!(state.contains("fn install_loaded("));
    assert!(state.contains("fn persistence_payload("));
    assert!(state.contains("fn normalized_loaded_project_snapshot("));
    assert!(!state.contains("fn dispatch("));
    assert!(commands.contains("fn dispatch("));
    assert!(commands.contains("fn select_rois("));
    assert!(commands.contains("fn create_view("));
    assert!(!commands.contains("classify_local_dataset_path"));
    assert!(validation.contains("fn validate_replacement_rois("));
    assert!(validation.contains("fn roi_from_params("));
    assert!(!validation.contains("impl ProjectModel {"));
}

#[test]
fn object_layer_core_stays_split_by_responsibility() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let object_facade = source(root.join("src/objects/mod.rs"));
    assert!(
        object_facade.lines().count() <= 1_400,
        "src/objects/mod.rs has regrown into a state/service/storage/test monolith"
    );
    assert!(!object_facade.contains("impl ObjectResourceLoader for NativeObjectControlService"));
    assert!(!object_facade.contains("impl ObjectPropertyStore"));
    assert!(!object_facade.contains("impl Default for ObjectsLayer"));
    assert!(!object_facade.contains("#[test]"));

    for module in ["control_service.rs", "property_store.rs", "defaults.rs"] {
        let path = root.join("src/objects").join(module);
        let contents = source(&path);
        assert!(
            contents.lines().count() <= 600,
            "{} is too large; split it at an object-layer domain boundary",
            path.display()
        );
        assert!(
            !contents.contains("#[test]"),
            "object-layer tests belong under objects/tests.rs: {}",
            path.display()
        );
    }

    let facade = source(root.join("src/objects/core.rs"));
    assert!(
        facade.lines().count() <= 1_000,
        "src/objects/core.rs has regrown into an object-runtime implementation monolith"
    );
    assert!(!facade.contains("impl ObjectsLayer {"));
    assert!(!facade.contains("#[test]"));

    for path in rust_files(&root.join("src/objects/core")) {
        if path.file_name().is_some_and(|name| name == "tests.rs") {
            continue;
        }
        let contents = source(&path);
        assert!(
            contents.lines().count() <= 2_500,
            "{} is too large; split it at an object-layer responsibility boundary",
            path.display()
        );
        assert!(
            !contents.contains("#[test]"),
            "object-layer tests belong under objects/core/tests.rs: {}",
            path.display()
        );
    }

    let properties_ui = source(root.join("src/objects/core/properties_ui.rs"));
    assert!(
        properties_ui.lines().count() <= 650,
        "object properties UI has regrown into a UI/style/filter/color monolith"
    );
    assert!(!properties_ui.contains("fn control_style_snapshot_json("));
    assert!(!properties_ui.contains("fn object_property_label("));
    assert!(!properties_ui.contains("fn active_color_legend_entries("));
    assert!(!properties_ui.contains("#[test]"));

    for path in rust_files(&root.join("src/objects/core/properties_ui")) {
        let contents = source(&path);
        assert!(
            contents.lines().count() <= 700,
            "{} is too large; split it at an object-presentation responsibility boundary",
            path.display()
        );
        assert!(
            !contents.contains("#[test]"),
            "object presentation tests belong under object test modules: {}",
            path.display()
        );
    }

    let analysis = source(root.join("src/objects/analysis.rs"));
    assert!(
        analysis.lines().count() <= 2_500,
        "src/objects/analysis.rs has regrown into a UI/data/algorithm monolith"
    );
    assert!(!analysis.contains("fn compute_threshold_selection_indices("));
    assert!(!analysis.contains("fn available_numeric_object_property_keys("));
    assert!(!analysis.contains("#[test]"));

    for path in rust_files(&root.join("src/objects/analysis")) {
        let contents = source(&path);
        assert!(
            contents.lines().count() <= 1_500,
            "{} is too large; split it at an object-analysis responsibility boundary",
            path.display()
        );
        assert!(
            !contents.contains("#[test]"),
            "object-analysis tests belong under the object test modules: {}",
            path.display()
        );
    }

    let render = source(root.join("src/objects/render.rs"));
    assert!(
        render.lines().count() <= 1_800,
        "src/objects/render.rs has regrown into a drawing/selection/geometry monolith"
    );
    assert!(!render.contains("fn object_intersects_rect_for_selection("));
    assert!(!render.contains("fn build_render_lods("));
    assert!(!render.contains("#[test]"));

    for path in rust_files(&root.join("src/objects/render")) {
        if path.file_name().is_some_and(|name| name == "tests.rs") {
            continue;
        }
        let contents = source(&path);
        assert!(
            contents.lines().count() <= 1_000,
            "{} is too large; split it at an object-render responsibility boundary",
            path.display()
        );
        assert!(
            !contents.contains("#[test]"),
            "object-render tests belong under objects/render/tests.rs: {}",
            path.display()
        );
    }
}

#[test]
fn project_space_stays_split_by_responsibility() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let facade = source(root.join("src/project/space.rs"));
    assert!(
        facade.lines().count() <= 1_200,
        "src/project/space.rs has regrown into a project-state/UI monolith"
    );
    assert!(!facade.contains("impl ProjectSpace {"));
    assert!(!facade.contains("#[test]"));

    for path in rust_files(&root.join("src/project/space")) {
        if path.file_name().is_some_and(|name| name == "tests.rs") {
            continue;
        }
        let contents = source(&path);
        assert!(
            contents.lines().count() <= 1_500,
            "{} is too large; split it at a ProjectSpace responsibility boundary",
            path.display()
        );
        assert!(
            !contents.contains("#[test]"),
            "ProjectSpace tests belong under project/space/tests.rs: {}",
            path.display()
        );
    }
}

#[test]
fn control_registry_stays_split_by_responsibility() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let facade = source(root.join("src/control/registry.rs"));
    assert!(
        facade.lines().count() <= 800,
        "src/control/registry.rs has regrown into a catalog/schema/test monolith"
    );
    assert!(
        !facade.contains("pub static METHODS"),
        "application descriptors belong in control/registry/catalog.rs"
    );
    assert!(
        !facade.contains("fn request_schema("),
        "request schemas belong in control/registry/schema.rs"
    );
    assert!(
        !facade.contains("#[test]"),
        "registry tests belong under control/registry/tests.rs"
    );

    for path in rust_files(&root.join("src/control/registry")) {
        if path.file_name().is_some_and(|name| name == "tests.rs") {
            continue;
        }
        let contents = source(&path);
        assert!(
            contents.lines().count() <= 3_000,
            "{} is too large; split it at a registry responsibility boundary",
            path.display()
        );
        assert!(
            !contents.contains("#[test]"),
            "registry tests belong under control/registry/tests.rs: {}",
            path.display()
        );
    }

    let command = source(root.join("src/control/command.rs"));
    assert!(
        command.lines().count() <= 300,
        "src/control/command.rs has regrown into a request/validation/test monolith"
    );
    assert!(!command.contains("struct SetSidePanelsRequest"));
    assert!(!command.contains("fn validate_params("));
    assert!(!command.contains("#[test]"));

    let requests = source(root.join("src/control/command/requests.rs"));
    assert!(
        requests.lines().count() <= 700,
        "control request DTOs need another domain split"
    );
    assert!(!requests.contains("fn validate_params("));
    assert!(!requests.contains("#[test]"));

    let validation = source(root.join("src/control/command/requests/validation.rs"));
    assert!(
        validation.lines().count() <= 1_400,
        "control request validation needs another domain split"
    );
    assert!(!validation.contains("#[test]"));
}

#[test]
fn declarative_ui_stays_split_by_responsibility() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let facade = source(root.join("src/control/ui.rs"));
    assert!(
        facade.lines().count() <= 700,
        "src/control/ui.rs has regrown into a registry/render/validation/test monolith"
    );
    assert!(!facade.contains("pub fn render(&self"));
    assert!(!facade.contains("fn render_component("));
    assert!(!facade.contains("fn validate_tree("));
    assert!(!facade.contains("#[test]"));

    let render = source(root.join("src/control/ui/render.rs"));
    assert!(render.lines().count() <= 500);
    assert!(render.contains("pub fn render(&self"));
    assert!(!render.contains("fn render_component("));
    assert!(!render.contains("#[test]"));

    for (path, limit) in [
        ("src/control/ui/render/components.rs", 600),
        ("src/control/ui/validation.rs", 450),
    ] {
        let contents = source(root.join(path));
        assert!(
            contents.lines().count() <= limit,
            "{path} is too large; split it at a declarative-UI responsibility boundary"
        );
        assert!(
            !contents.contains("#[test]"),
            "declarative UI tests belong under control/ui/tests.rs: {path}"
        );
    }
}

#[test]
fn actor_resource_worker_stays_split_by_domain() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let worker = source(root.join("src/control/actor/worker.rs"));
    assert!(
        worker.lines().count() <= 1_300,
        "src/control/actor/worker.rs has regrown into a dispatch/domain-compute monolith"
    );
    assert!(worker.contains("fn spawn_resource_workers("));
    assert!(!worker.contains("fn load_pinned_memory_on_worker("));
    assert!(!worker.contains("fn compute_analysis_on_worker("));
    assert!(!worker.contains("fn measure_objects_on_worker("));
    assert!(!worker.contains("fn write_screenshot_on_worker("));
    assert!(!worker.contains("#[test]"));

    for path in rust_files(&root.join("src/control/actor/worker")) {
        let contents = source(&path);
        assert!(
            contents.lines().count() <= 500,
            "{} is too large; split it at an actor-worker domain boundary",
            path.display()
        );
        assert!(
            !contents.contains("#[test]"),
            "actor-worker tests belong under control/actor/tests: {}",
            path.display()
        );
    }
}

#[test]
fn tiff_pyramid_stays_split_by_responsibility() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let facade = source(root.join("src/xenium/tiff_pyramid.rs"));
    assert!(
        facade.lines().count() <= 500,
        "src/xenium/tiff_pyramid.rs has regrown into an inspection/loader/test monolith"
    );
    assert!(!facade.contains("fn build_levels_from_main_ifds("));
    assert!(!facade.contains("fn parse_ome_xml("));
    assert!(!facade.contains("fn tiff_tile_loader_thread("));
    assert!(!facade.contains("#[test]"));

    for path in rust_files(&root.join("src/xenium/tiff_pyramid")) {
        if path.file_name().is_some_and(|name| name == "tests.rs") {
            continue;
        }
        let contents = source(&path);
        assert!(
            contents.lines().count() <= 800,
            "{} is too large; split it at a TIFF subsystem boundary",
            path.display()
        );
        assert!(
            !contents.contains("#[test]"),
            "TIFF tests belong under xenium/tiff_pyramid/tests.rs: {}",
            path.display()
        );
    }
}

#[test]
fn cell_threshold_panel_stays_split_by_responsibility() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let facade = source(root.join("src/custom/cell_thresholds.rs"));
    let data = source(root.join("src/custom/cell_thresholds/data.rs"));
    let threshold_files = source(root.join("src/custom/cell_thresholds/threshold_files.rs"));

    assert!(
        facade.lines().count() <= 900,
        "the cell-threshold panel has regrown into a UI, parquet, and file-format monolith"
    );
    assert!(facade.contains("impl CellThresholdsPanel"));
    assert!(!facade.contains("ParquetRecordBatchReaderBuilder"));
    assert!(!facade.contains("fn parse_csv("));

    assert!(data.lines().count() <= 800);
    assert!(data.contains("ParquetRecordBatchReaderBuilder"));
    assert!(data.contains("fn load_points_for_marker("));
    assert!(!data.contains("impl CellThresholdsPanel"));
    assert!(!data.contains("fn load_thresholds_csv("));

    assert!(threshold_files.lines().count() <= 400);
    assert!(threshold_files.contains("fn load_thresholds_csv("));
    assert!(threshold_files.contains("fn load_auto_thresholds_json("));
    assert!(!threshold_files.contains("ParquetRecordBatchReaderBuilder"));
    assert!(!threshold_files.contains("impl CellThresholdsPanel"));

    for (path, contents) in [
        ("src/custom/cell_thresholds.rs", facade),
        ("src/custom/cell_thresholds/data.rs", data),
        (
            "src/custom/cell_thresholds/threshold_files.rs",
            threshold_files,
        ),
    ] {
        assert!(
            !contents.contains("#[test]"),
            "cell-threshold tests belong in a dedicated test module: {path}"
        );
    }
}

#[test]
fn spatialdata_layers_stay_split_by_responsibility() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let facade = source(root.join("src/spatialdata/layers.rs"));
    assert!(
        facade.lines().count() <= 300,
        "src/spatialdata/layers.rs has regrown into a collection/render/preparation monolith"
    );
    assert!(!facade.contains("pub struct SpatialShapesLayer"));
    assert!(!facade.contains("pub struct SpatialPointsLayer"));
    assert!(!facade.contains("fn prepare_spatial_points_payload("));
    assert!(!facade.contains("#[test]"));

    for (module, limit) in [("shapes.rs", 600), ("points.rs", 1_200)] {
        let path = root.join("src/spatialdata/layers").join(module);
        let contents = source(&path);
        assert!(
            contents.lines().count() <= limit,
            "{} is too large; split it at a SpatialData layer boundary",
            path.display()
        );
        assert!(
            !contents.contains("#[test]"),
            "SpatialData layer tests belong in dedicated test modules: {}",
            path.display()
        );
    }

    let preparation = source(root.join("src/spatialdata/layers/points/prepare.rs"));
    assert!(preparation.lines().count() <= 400);
    assert!(!preparation.contains("pub fn draw("));
    assert!(!preparation.contains("pub fn ui_properties("));
    assert!(!preparation.contains("#[test]"));
}

#[test]
fn spatialdata_parquet_shapes_stay_split_by_responsibility() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let facade = source(root.join("src/spatialdata/parquet_shapes.rs"));
    let geometry = source(root.join("src/spatialdata/parquet_shapes/geometry.rs"));
    let tests = source(root.join("src/spatialdata/parquet_shapes/tests.rs"));

    assert!(
        facade.lines().count() <= 1_000,
        "the SpatialData shapes façade has regrown into a parquet/WKB/test monolith"
    );
    assert!(facade.contains("ParquetRecordBatchReaderBuilder"));
    assert!(facade.contains("pub fn load_shapes_objects("));
    assert!(!facade.contains("struct Cursor<'a>"));
    assert!(!facade.contains("fn read_geom("));
    assert!(!facade.contains("#[test]"));

    assert!(geometry.lines().count() <= 900);
    assert!(geometry.contains("struct Cursor<'a>"));
    assert!(geometry.contains("fn read_geom("));
    assert!(geometry.contains("fn centroid_summary_from_wkb("));
    assert!(!geometry.contains("ParquetRecordBatchReaderBuilder"));
    assert!(!geometry.contains("pub fn load_shapes_objects("));
    assert!(!geometry.contains("#[test]"));

    assert!(tests.contains("#[test]"));
    assert!(tests.contains("centroid_summary_reads_polygon_without_persisting_rings"));
}

#[test]
fn renderer_bridge_is_a_projection_only_boundary() {
    let app_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/app");
    let renderer_bridge = source(app_dir.join("renderer_bridge/mod.rs"));
    let renderer_masks = source(app_dir.join("renderer_bridge/masks.rs"));
    assert!(renderer_bridge.contains("must not implement application command semantics"));
    for removed_selection_bridge in [
        "record_native_object_selection_intent",
        "control_object_selection_projection_snapshot",
        "record_native_mask_intent",
        "native_mask_actor_intent_emitted",
        "record_native_layers_intent",
        "control_native_layers_projection_snapshot",
    ] {
        assert!(
            !renderer_masks.contains(removed_selection_bridge),
            "native selection, mask, and layer edits must commit at their interaction point: {removed_selection_bridge}"
        );
    }

    let facade = source(app_dir.join("mod.rs"));
    assert!(!facade.contains("mod legacy_control"));
    assert!(!facade.contains("native_mask_actor_intent_emitted"));
    assert!(
        !source(app_dir.join("viewport_runtime.rs")).contains("record_native_viewport_intents"),
        "viewport edits must commit at their interaction points, not through a snapshot diff"
    );

    let actor_facade =
        source(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/control/actor/mod.rs"));
    assert!(
        actor_facade.lines().count() <= 300,
        "the control actor façade must not regain domain handlers"
    );

    let native_layer_projection = source(app_dir.join("actor_layer_projection.rs"));
    assert!(
        !native_layer_projection.contains("control_set_"),
        "actor native-layer projection must not call renderer command emulators"
    );
}

#[test]
fn native_workspace_topology_has_no_renderer_mutation_fallback() {
    let app_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/app");
    let runtime = source(app_dir.join("viewport_runtime.rs"));
    let ui = source(app_dir.join("viewport_ui.rs"));
    let combined = format!("{runtime}\n{ui}");

    for removed_handler in [
        "fn split_active_viewport(",
        "fn remove_viewport(",
        "fn set_viewport_layout(",
        "fn set_viewport_links(",
    ] {
        assert!(
            !combined.contains(removed_handler),
            "native viewport topology must not regain renderer handler {removed_handler}"
        );
    }
    for forbidden_mutation in [
        "workspace.clone_viewport(",
        "workspace.set_layout(",
        "workspace.set_links(",
        "workspace.swap_order(",
        "workspace.set_active(",
        "workspace.rename(",
        "workspace.set_split_ratio(",
        "workspace.remove(&viewport_id)",
    ] {
        assert!(
            !combined.contains(forbidden_mutation),
            "native viewport topology must commit through the actor: {forbidden_mutation}"
        );
    }
    assert!(
        !ui.contains("native_viewport_actor_owned()"),
        "viewport chrome must submit topology commands even before its first actor projection"
    );
    let renderer_queries = source(app_dir.join("renderer_bridge/viewports.rs"));
    for forbidden_authority in [
        "bump_navigation_revision(",
        "bump_presentation_revision(",
        "copy_linked_navigation_from(",
        "ViewportControlDomain",
    ] {
        assert!(
            !combined.contains(forbidden_authority)
                && !renderer_queries.contains(forbidden_authority),
            "renderer viewport code must not own revision or linked-state semantics: {forbidden_authority}"
        );
    }
    let navigation = source(app_dir.join("navigation.rs"));
    let interactions = source(app_dir.join("canvas/interactions.rs"));
    let update = source(app_dir.join("update.rs"));
    assert!(!navigation.contains("native_viewport_actor_owned()"));
    assert!(!interactions.contains("native_viewport_actor_owned()"));
    assert!(!update.contains("set_view_plane_mode("));
    assert!(!update.contains("set_active_view_slice_level0("));
    assert!(!source(app_dir.join("image_runtime.rs")).contains("fn set_view_plane_mode("));
    for actor_only_channel_source in [
        "mod.rs",
        "datasets.rs",
        "layer_runtime/contrast.rs",
        "projects.rs",
        "viewport_runtime.rs",
    ] {
        assert!(
            !source(app_dir.join(actor_only_channel_source))
                .contains("native_viewport_actor_owned"),
            "native channel controls must not regain projection-readiness mutation fallbacks: {actor_only_channel_source}"
        );
    }
    for actor_only_layer_source in [
        "layer_runtime/commands.rs",
        "layer_runtime/offsets.rs",
        "layer_properties.rs",
        "layers_ui.rs",
        "mask_interaction.rs",
        "canvas/interactions.rs",
    ] {
        assert!(
            !source(app_dir.join(actor_only_layer_source)).contains("native_layers_actor_owned"),
            "native layer controls must not regain projection-readiness mutation fallbacks: {actor_only_layer_source}"
        );
    }
    let update = source(app_dir.join("update.rs"));
    for forbidden_presentation_write in [
        "self.show_left_panel =",
        "self.show_right_panel =",
        "self.left_tab =",
        "self.right_tab =",
        "self.smooth_pixels =",
        "self.show_scale_bar =",
        "self.show_hud =",
        "self.show_tile_debug =",
    ] {
        assert!(
            !update.contains(forbidden_presentation_write),
            "native presentation controls must wait for actor projection: {forbidden_presentation_write}"
        );
    }
    assert!(!source(app_dir.join("project_integration.rs")).contains("self.right_tab ="));

    let root = source(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/root_app.rs"));
    assert!(root.contains("app.control_renderer_observation_snapshot()"));
    assert!(root.contains("report_renderer_observation("));
    assert!(!root.contains("app.set_show_scale_bar(visible)"));
    assert!(!root.contains("view_show_scale_bar"));
    assert!(root.contains("menu.set_scale_bar_visible(visible)"));
    assert!(root.contains("runtime.bootstrap_project_model(snapshot)"));
    assert!(!root.contains("control_actor_semantic_snapshot"));
    assert!(root.contains("runtime.bootstrap_mosaic_model(mosaic.control_actor_resource())"));
    assert!(
        !root.contains("app.set_project_space(")
            && !root.contains("single.set_project_space(")
            && !root.contains("restored.set_project_space("),
        "RootApp must attach projected project state without asking the renderer to restore semantic workspace state"
    );
    assert!(
        !root.contains("app.control_viewport_workspace_snapshot(),"),
        "dataset bootstrap must restore actor-owned project state instead of importing a renderer workspace"
    );

    let actor_runtime =
        source(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/control/actor/runtime.rs"));
    let bootstrap_variant = actor_runtime
        .split("BootstrapDataset {")
        .nth(1)
        .and_then(|tail| tail.split("},").next())
        .expect("actor runtime declares BootstrapDataset");
    assert!(
        !bootstrap_variant.contains("workspace"),
        "renderer workspace data must not cross the production dataset-bootstrap boundary"
    );
    let datasets = source(app_dir.join("datasets.rs"));
    assert!(
        !datasets.contains("self.apply_view_state_from_project_space()"),
        "renderer-side dataset switching must wait for the actor's persisted-view projection"
    );
}

#[test]
fn viewport_render_history_is_explicitly_separated_from_projected_state() {
    let app = source(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/app/mod.rs"));
    let render_state = app
        .split("struct ViewportRenderState {")
        .nth(1)
        .and_then(|tail| tail.split("struct ViewerViewportState {").next())
        .expect("viewport render state precedes projected viewport state");
    for field in [
        "last_canvas_rect",
        "active_render_id",
        "previous_render_id",
        "previous_view_selection",
        "last_target_level",
        "fallback_ceiling_level",
        "last_visible_world_tiles",
        "zoom_out_floor_level",
        "zoom_out_floor_until",
    ] {
        assert!(
            render_state.contains(field),
            "renderer history field must remain isolated: {field}"
        );
    }
    let projected_state = app
        .split("struct ViewerViewportState {")
        .nth(1)
        .and_then(|tail| tail.split("impl ViewerViewportState {").next())
        .expect("projected viewport state declaration exists");
    assert!(projected_state.contains("render: ViewportRenderState"));
    assert!(!projected_state.contains("last_canvas_rect:"));
    assert!(!projected_state.contains("active_render_id:"));

    let transient_state = app
        .split("struct ViewportTransientState {")
        .nth(1)
        .and_then(|tail| tail.split("struct ViewerViewportState {").next())
        .expect("viewport transient state precedes projected viewport state");
    for field in [
        "draft_view_slice_level0",
        "selected_channel_layers",
        "selected_channel_group_id",
        "selected_overlay_layers",
        "object_filter_cache",
    ] {
        assert!(
            transient_state.contains(field),
            "transient interaction field must remain isolated: {field}"
        );
    }
    assert!(projected_state.contains("transient: ViewportTransientState"));
    assert!(!projected_state.contains("draft_view_slice_level0:"));
    assert!(!projected_state.contains("object_filter_cache:"));

    let viewport_ui =
        source(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/app/viewport_ui.rs"));
    let viewport_runtime =
        source(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/app/viewport_runtime.rs"));
    let actor_projection =
        source(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/app/actor_projection.rs"));
    assert!(viewport_ui.contains("state.capture_runtime(self)"));
    assert!(viewport_runtime.contains("state.capture_runtime(self)"));
    assert!(actor_projection.contains("active.state.capture_runtime(self)"));
    for frame_path in [&viewport_ui, &viewport_runtime] {
        assert!(
            !frame_path.contains("state = ViewerViewportState::capture(self)"),
            "frame-driven workspace synchronization must not copy projected semantics from the renderer"
        );
    }
}

#[test]
fn renderer_has_no_semantic_command_emulators() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/app/renderer_bridge");
    let mutation_prefixes = [
        "control_set_",
        "control_create_",
        "control_update_",
        "control_remove_",
        "control_rename_",
        "control_reset_",
        "control_swap_",
        "control_zoom",
        "control_fit_",
        "control_load_",
        "control_unload_",
        "control_select_",
        "control_step_",
        "control_configure_",
    ];
    let mut found = BTreeSet::new();

    for path in rust_files(&root) {
        let filename = path
            .file_name()
            .and_then(|name| name.to_str())
            .expect("renderer bridge filename");
        for line in source(&path).lines() {
            let Some(after_fn) = line.split_once("fn ").map(|(_, rest)| rest) else {
                continue;
            };
            let Some(name) = after_fn.split_once('(').map(|(name, _)| name.trim()) else {
                continue;
            };
            if mutation_prefixes
                .iter()
                .any(|prefix| name.starts_with(prefix))
            {
                found.insert(format!("{filename}::{name}"));
            }
        }
    }

    assert!(
        found.is_empty(),
        "renderer semantic command emulators are forbidden: {found:?}"
    );
}

#[test]
fn native_object_selection_and_analysis_have_no_renderer_commit_fallback() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let selection = source(root.join("src/app/selection.rs"));
    assert!(!selection.contains("actor_owns_object_selection_target"));
    for forbidden in [
        ".select_in_world_rect(",
        ".select_in_world_lasso(",
        ".select_at(",
        ".clear_selection()",
        ".select_objects_by_ids(",
    ] {
        assert!(
            !selection.contains(forbidden),
            "native object selection must submit an actor command before renderer mutation: {forbidden}"
        );
    }

    let properties = source(root.join("src/objects/core/properties_ui.rs"));
    assert!(!properties.contains("self.clear_selection()"));
    assert!(!properties.contains("self.select_filtered_objects()"));
    assert!(properties.contains("ObjectUiAction::ClearSelection"));
    assert!(properties.contains("ObjectUiAction::SelectFiltered"));

    let update = source(root.join("src/app/update.rs"));
    assert!(update.contains("objects.apply_project_analysis_state("));
    assert!(update.contains("method: \"viewer.analysis.set\""));

    let model = source(root.join("src/model/app.rs"));
    assert!(model.contains("analyses: HashMap<ObjectTarget, AnalysisModel>"));
    assert!(!model.contains("analysis: AnalysisModel"));
    assert!(model.contains("pub analysis_generation: u64"));
    assert!(model.contains("pub analysis_state: Value"));

    let projection = source(root.join("src/app/actor_projection.rs"));
    assert!(projection.contains("projected.analysis_generation > installed_analysis"));
    assert!(projection.contains("objects.apply_project_analysis_state("));
    assert!(
        !projection.contains(".load_path(PathBuf::from(source)"),
        "an actor object projection must install its shared resource, never restart renderer I/O"
    );
}

#[test]
fn native_mask_semantics_and_io_have_no_renderer_commit_fallback() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let masks = source(root.join("src/app/mask_interaction.rs"));
    for forbidden in [
        "mask_actor_owned",
        "next_mask_layer_id",
        "mask_layers_project_dirty",
        "undo_stack",
        "push_mask_undo_snapshot",
        "push_layer_offsets_undo_snapshot",
        "mark_mask_layers_project_dirty",
        "fn create_editable_mask_layer",
        "sync_mask_layers_into_project_space",
    ] {
        assert!(
            !masks.contains(forbidden),
            "native mask semantics must not retain a renderer fallback: {forbidden}"
        );
    }
    for method in [
        "viewer.masks.layers.create",
        "viewer.masks.layers.delete",
        "viewer.masks.polygons.add",
        "viewer.masks.polygons.remove",
        "viewer.masks.selection.set",
        "viewer.masks.selection.clear",
        "viewer.masks.state.replace",
        "viewer.masks.undo",
        "viewer.masks.export_geojson",
    ] {
        assert!(
            masks.contains(method),
            "native mask operation must enter its typed actor command: {method}"
        );
    }

    let screenshots = source(root.join("src/app/screenshots.rs"));
    assert!(!screenshots.contains("save_mask_layers_geojson"));
    assert!(!screenshots.contains("export_masks_geojson"));
    assert!(!screenshots.contains("export_mask_layer_geojson"));

    let datasets = source(root.join("src/app/datasets.rs"));
    assert!(!datasets.contains("ensure_exclusion_masks_loaded"));
    assert!(!datasets.contains("fs::write"));
}

#[test]
fn annotation_semantics_resources_and_persistence_are_actor_owned() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let app = source(root.join("src/app/mod.rs"));
    let mosaic = source(root.join("src/mosaic/mod.rs"));
    let renderer = source(root.join("src/annotations/mod.rs"));
    let actor_model = source(root.join("src/model/annotations.rs"));
    let single_project = source(root.join("src/app/projects.rs"));
    let mosaic_project = source(root.join("src/mosaic/control/project.rs"));

    assert!(!app.contains("next_annotation_layer_id"));
    assert!(!mosaic.contains("next_annotation_layer_id"));
    assert!(actor_model.contains("next_id: u64"));
    assert!(actor_model.contains("resource: Option<Arc<ControlAnnotationResource>>"));
    for forbidden in [
        "load_annotations_parquet(",
        "ParquetRecordBatchReaderBuilder",
        "schema_rx",
        "load_rx",
        "std::thread::spawn",
    ] {
        assert!(
            !renderer.contains(forbidden),
            "annotation decoding and worker ownership must remain outside the renderer: {forbidden}"
        );
    }

    assert!(single_project.contains("let annotation_layers = self"));
    assert!(!single_project.contains(".map(|layer| self.project_annotation_layer_state(layer))"));
    assert!(mosaic_project.contains("let annotation_layers = self"));
    assert!(!mosaic_project.contains("project_annotation_layer_state"));
    assert!(!mosaic_project.contains("restore_annotation_layers"));

    let single_properties = source(root.join("src/app/layer_properties.rs"));
    let mosaic_properties = source(root.join("src/mosaic/panels.rs"));
    for method in [
        "viewer.annotations.layers.update",
        "viewer.annotations.source.inspect",
        "viewer.annotations.source.load",
        "viewer.annotations.source.reload",
        "viewer.annotations.layers.delete",
    ] {
        assert!(single_properties.contains(method));
        assert!(mosaic_properties.contains(method));
    }
}

#[test]
fn mosaic_object_style_and_selection_have_no_renderer_commit_fallback() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let canvas = source(root.join("src/mosaic/canvas.rs"));
    let panels = source(root.join("src/mosaic/panels.rs"));
    let overlay = source(root.join("src/mosaic/segmentation_geojson.rs"));
    let model = source(root.join("src/model/mosaic.rs"));

    assert!(!canvas.contains(".select_at("));
    assert!(!canvas.contains("seg_geojson.clear_selection("));
    assert!(canvas.contains("mosaic.objects.selection.replace"));
    assert!(canvas.contains("mosaic.objects.selection.clear"));
    assert!(panels.contains("mosaic.objects.style.set"));
    assert!(panels.contains("mosaic.objects.selection.clear"));
    assert!(!overlay.contains("pub fn select_at("));
    assert!(!overlay.contains("pub fn clear_selection("));
    assert!(model.contains("object_style: Value"));
    assert!(model.contains("object_selections: HashMap<usize, ObjectSelectionModel>"));
}

#[test]
fn segmentation_resources_have_no_frame_driven_filesystem_loader() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let single = source(root.join("src/objects/geojson.rs"));
    let mosaic = source(root.join("src/mosaic/segmentation_geojson.rs"));
    let canvas = source(root.join("src/mosaic/canvas.rs"));
    let actor = source(root.join("src/control/actor/worker.rs"));

    for forbidden in [
        "load_rx",
        "seg-geojson-loader",
        "load_in_thread",
        "ensure_visible_items_loading",
        "load_objects_with_transform",
    ] {
        assert!(
            !single.contains(forbidden) && !mosaic.contains(forbidden),
            "segmentation renderer reintroduced frame-driven resource work: {forbidden}"
        );
    }
    assert!(canvas.contains("\"mosaic.objects.load\""));
    assert!(actor.contains("LoadJob::SegmentationGeoJson"));
    assert!(actor.contains("load_geojson_polyline_coordinates_world"));
}

#[test]
fn native_settings_are_actor_persisted_and_projection_driven() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root_app = source(root.join("src/root_app.rs"));
    let actor = source(root.join("src/control/actor/application.rs"));

    assert!(root_app.contains("settings_draft: AppSettings"));
    assert!(root_app.contains("\"app.settings.set\""));
    assert!(root_app.contains("self.settings_draft = self.app_settings.clone()"));
    for forbidden in ["fn persist_app_settings", "self.app_settings.save()"] {
        assert!(
            !root_app.contains(forbidden),
            "native settings must not regain GUI-side persistence: {forbidden}"
        );
    }
    assert!(actor.contains("LoadJob::SettingsSave"));
}

#[test]
fn mosaic_actor_tasks_do_not_require_a_renderer_pending_item_mirror() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mosaic = source(root.join("src/mosaic/mod.rs"));
    let projection = source(root.join("src/mosaic/construction/actor_resource.rs"));
    let update = source(root.join("src/mosaic/update.rs"));

    for source in [&mosaic, &projection, &update] {
        assert!(!source.contains("pending_object_load_ids"));
    }
    assert!(projection.contains("reconcile_actor_load_state"));
}

#[test]
fn project_object_preload_is_one_actor_projection_adapter() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root_app = source(root.join("src/root_app.rs"));

    assert!(root_app.contains("struct ProjectObjectPreloadRenderProjection"));
    assert!(root_app.contains("resources: HashMap<"));
    assert!(root_app.contains("ui: ProjectObjectCacheUiState"));
    assert!(root_app.contains("fn apply_project_object_preload_projection("));
    for obsolete in [
        "object_preload_cache:",
        "object_preload_settings:",
        "object_preload_available_count:",
        "object_preload_on_disk_bytes:",
        "object_preload_total:",
        "object_preload_done:",
        "object_preload_failed:",
        "object_preload_loading:",
    ] {
        assert!(
            !root_app.contains(obsolete),
            "project preload state must not split back into host-owned fields: {obsolete}"
        );
    }
}

#[test]
fn remote_dataset_io_is_actor_backed_and_not_renderer_local() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let app = source(root.join("src/app/mod.rs"));
    let viewer_project = source(root.join("src/app/project_integration.rs"));
    let root_remote = source(root.join("src/root_app/remote.rs"));

    assert!(!root.join("src/app/remote.rs").exists());
    assert!(!app.contains("mod remote;"));
    assert!(viewer_project.contains("ViewerRequest::OpenRemoteDialog"));
    for method in [
        "datasets.s3.configure_session",
        "datasets.s3.list",
        "datasets.open_http",
        "datasets.open_s3",
    ] {
        assert!(
            root_remote.contains(method),
            "shared remote dialog must submit the actor command: {method}"
        );
    }
    for forbidden in [
        "build_http_store(",
        "build_s3_store(",
        "build_s3_browser(",
        "list_s3_prefix(",
        "OmeZarrDataset::open_with_store(",
    ] {
        assert!(
            !root_remote.contains(forbidden),
            "remote dialog must not perform dataset I/O locally: {forbidden}"
        );
    }
}

#[test]
fn label_discovery_and_loading_are_actor_owned() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let construction = source(root.join("src/app/construction.rs"));
    let datasets = source(root.join("src/app/datasets.rs"));
    let lifecycle = source(root.join("src/app/lifecycle.rs"));
    let project = source(root.join("src/app/project_integration.rs"));
    let properties = source(root.join("src/app/layer_properties.rs"));
    let projection = source(root.join("src/app/actor_projection.rs"));
    let worker = source(root.join("src/control/actor/worker.rs"));

    for renderer in [&construction, &datasets, &lifecycle, &project, &properties] {
        for forbidden in [
            "discover_label_names_local",
            "LabelZarrDataset::try_open",
            "fn load_segmentation_labels(",
            "fn load_root_segmentation_labels(",
        ] {
            assert!(
                !renderer.contains(forbidden),
                "label discovery/loading must not return to a renderer path: {forbidden}"
            );
        }
    }
    assert!(datasets.contains("\"viewer.labels.load\""));
    assert!(properties.contains("\"viewer.labels.load\""));
    assert!(properties.contains("\"viewer.labels.set_visibility\""));
    assert!(projection.contains("projection.get(\"labels\")"));
    assert!(worker.contains("LoadJob::Labels"));
}

#[test]
fn channel_histogram_and_auto_contrast_are_actor_owned() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let construction = source(root.join("src/app/construction.rs"));
    let lifecycle = source(root.join("src/app/lifecycle.rs"));
    let tile_runtime = source(root.join("src/app/tile_runtime.rs"));
    let update = source(root.join("src/app/update.rs"));
    let projection = source(root.join("src/app/actor_projection.rs"));
    let channels = source(root.join("src/control/actor/channels.rs"));
    let runtime = source(root.join("src/control/actor/runtime.rs"));
    let worker = source(root.join("src/control/actor/worker.rs"));

    for renderer in [&construction, &lifecycle, &tile_runtime, &update] {
        for forbidden in [
            "hist_loader",
            "chanmax_loader",
            "spawn_histogram_loader",
            "spawn_channel_max_loader",
            "spawn_tiff_histogram_loader",
            "spawn_tiff_channel_max_loader",
            "drain_channel_maxes",
            "request_auto_contrast",
        ] {
            assert!(
                !renderer.contains(forbidden),
                "channel compute must not return to a renderer worker path: {forbidden}"
            );
        }
    }
    assert!(tile_runtime.contains("viewer.channels.intensity_stats"));
    assert!(projection.contains("apply_control_actor_channel_compute"));
    assert!(channels.contains("LoadJob::AutoContrast"));
    assert!(runtime.contains("enqueue_auto_contrast_on_open"));
    assert!(worker.contains("LoadJob::ChannelIntensity"));
    assert!(worker.contains("LoadJob::AutoContrast"));
}

#[test]
fn document_tiff_plane_and_tile_policy_commits_are_actor_only() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let datasets = source(root.join("src/app/datasets.rs"));
    let tiff = source(root.join("src/app/tiff.rs"));
    let memory = source(root.join("src/app/memory_ui.rs"));
    let image_runtime = source(root.join("src/app/image_runtime.rs"));

    assert!(!datasets.contains("switch_dataset_with_store"));
    assert!(tiff.contains("\"datasets.open_tiff\""));
    for forbidden in [
        "fn apply_tiff_plane_selection(",
        "build_tiff_runtime_assets(",
        "self.dataset =",
        "self.store =",
        "self.loader =",
        "self.raw_loader =",
    ] {
        assert!(
            !tiff.contains(forbidden),
            "TIFF plane controls must not rebuild document resources locally: {forbidden}"
        );
    }
    assert!(memory.contains("\"memory.tiles.set\""));
    for forbidden in [
        "self.tile_loader_threads =",
        "self.tile_prefetch_mode =",
        "self.tile_prefetch_aggressiveness =",
        "self.respawn_tile_loaders()",
    ] {
        assert!(
            !memory.contains(forbidden),
            "tile policy UI must wait for actor projection: {forbidden}"
        );
    }
    assert!(image_runtime.contains("apply_control_actor_tile_loading_policy"));
}

#[test]
fn threshold_controls_wait_for_actor_projection_and_shared_resources() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let controls = source(root.join("src/app/contrast_ui.rs"));
    let projection = source(root.join("src/app/actor_projection.rs"));
    let worker = source(root.join("src/control/actor/worker.rs"));

    for method in [
        "viewer.thresholds.preview.start",
        "viewer.thresholds.preview.configure",
        "viewer.thresholds.preview.refresh",
        "viewer.thresholds.preview.apply",
        "viewer.thresholds.preview.cancel",
    ] {
        assert!(controls.contains(method));
    }
    for forbidden in [
        "self.threshold_region_min_pixels =",
        "self.threshold_region_scope =",
        "self.threshold_region_full_level =",
        "self.threshold_region_status.clear()",
        "self.threshold_region_preview.as_mut()",
    ] {
        assert!(
            !controls.contains(forbidden),
            "threshold UI must not optimistically mutate actor projection/resource state: {forbidden}"
        );
    }
    assert!(projection.contains("ControlThresholdPreviewResource"));
    assert!(worker.contains("LoadJob::Threshold"));
}

#[test]
fn production_control_path_has_no_legacy_ui_dispatcher() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root_app = source(root.join("src/root_app.rs"));
    let actor_runtime = source(root.join("src/control/actor/runtime.rs"));
    let actor_dispatch = source(root.join("src/control/actor/dispatch.rs"));
    let protocol_bridge = source(root.join("src/mcp/bridge.rs"));

    for symbol in [
        "reply_to_control_request",
        "process_control_requests",
        "process_deferred_control_requests",
        "deferred_control_requests",
        "legacy_rx",
        "legacy_tx",
        "forward_legacy_request",
        "publish_native_event",
        "control_observed_state",
    ] {
        assert!(
            !root_app.contains(symbol)
                && !actor_runtime.contains(symbol)
                && !actor_dispatch.contains(symbol)
                && !protocol_bridge.contains(symbol),
            "legacy control symbol remains reachable: {symbol}"
        );
    }
}

#[test]
fn local_control_runtime_and_tcp_bridge_remain_separate() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let actor = source(root.join("src/control/actor/mod.rs"));
    let request = source(root.join("src/control/request.rs"));
    let runtime = source(root.join("src/mcp/runtime.rs"));
    let bridge = source(root.join("src/mcp/bridge.rs"));
    let bridge_dir = root.join("src/mcp/bridge");

    assert!(request.contains("pub struct OdonControlRequest"));
    assert!(!actor.contains("crate::mcp::OdonControlRequest"));
    assert!(runtime.contains("pub struct OdonControlRuntime"));
    assert!(runtime.lines().count() <= 800);
    assert!(!runtime.contains("TcpListener"));
    assert!(!runtime.contains("TcpStream"));
    assert!(!runtime.contains("JsonRpcRequest"));
    assert!(!runtime.contains("fn handle_control_line"));

    assert!(
        bridge.lines().count() <= 250,
        "src/mcp/bridge.rs has regrown beyond its shared-state façade"
    );
    assert!(!bridge.contains("pub struct OdonControlRuntime"));
    assert!(!bridge.contains("spawn_control_actor_with_services"));
    assert!(!bridge.contains("#[test]"));
    for symbol in [
        "fn handle_control_client(",
        "fn handle_json_rpc_request(",
        "fn start_task(",
        "fn wait_for_application_ready(",
        "fn register_extension(",
    ] {
        assert!(
            !bridge.contains(symbol),
            "bridge responsibility belongs in a dedicated module: {symbol}"
        );
    }

    for path in rust_files(&bridge_dir) {
        if path.file_name().is_some_and(|name| name == "tests.rs") {
            continue;
        }
        let contents = source(&path);
        assert!(
            contents.lines().count() <= 800,
            "{} is too large; split the bridge at a transport, dispatch, task, wait, or service boundary",
            path.display()
        );
        assert!(
            !contents.contains("#[test]"),
            "production bridge tests belong under mcp/bridge/tests.rs: {}",
            path.display()
        );
    }

    let transport = source(bridge_dir.join("transport.rs"));
    let dispatch = source(bridge_dir.join("dispatch.rs"));
    let tasks = source(bridge_dir.join("tasks.rs"));
    let waits = source(bridge_dir.join("waits.rs"));
    let services = source(bridge_dir.join("services.rs"));
    assert!(transport.contains("TcpListener"));
    assert!(!transport.contains("ControlCommand::decode"));
    assert!(dispatch.contains("fn handle_json_rpc_request("));
    assert!(!dispatch.contains("TcpListener"));
    assert!(tasks.contains("fn start_task("));
    assert!(waits.contains("fn wait_for_application_ready("));
    assert!(services.contains("fn register_extension("));
}

#[test]
fn actor_owned_project_actions_bypass_host_semantic_relays() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let project_space = source(root.join("src/project/space/control_persistence.rs"));
    let viewer_integration = source(root.join("src/app/project_integration.rs"));
    let roi_integration = source(root.join("src/app/datasets.rs"));
    let mosaic = source(root.join("src/mosaic/panels.rs"));
    let root_app = source(root.join("src/root_app.rs"));

    assert!(project_space.contains("pub fn submit_action_control_intent"));
    for host in [viewer_integration, roi_integration, mosaic, root_app] {
        assert!(
            host.contains("submit_action_control_intent"),
            "actor-owned project actions must enter the project command outbox before any host fallback relay"
        );
    }
}

#[test]
fn root_requires_the_actor_and_host_requests_are_platform_only() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root_app = source(root.join("src/root_app.rs"));
    let viewer = source(root.join("src/app/mod.rs"));
    let mosaic = source(root.join("src/mosaic/mod.rs"));

    assert!(root_app.contains("control_runtime: OdonControlRuntime"));
    for forbidden in [
        "control_runtime: Option<OdonControlRuntime>",
        "control_runtime.is_some()",
        "control_runtime.is_none()",
        "control_bridge: Option<OdonControlBridge>",
        "control_bridge.is_some()",
        "control_bridge.is_none()",
        "pending_deep_link",
        "fn open_single(",
        "fn open_project_roi(",
        "fn open_mosaic_from_project(",
    ] {
        assert!(
            !root_app.contains(forbidden),
            "RootApp must not regain an optional actor or direct semantic fallback: {forbidden}"
        );
    }

    for forbidden in [
        "ViewerRequest::OpenProjectRoi",
        "ViewerRequest::OpenProjectRoiView",
        "ViewerRequest::OpenProject(",
        "ViewerRequest::SaveProject(",
        "ViewerRequest::OpenLocalPath",
        "ViewerRequest::OpenProjectMosaic",
        "ViewerRequest::PreloadObjectSegmentations",
        "ViewerRequest::ClearObjectCache",
    ] {
        assert!(
            !root_app.contains(forbidden) && !viewer.contains(forbidden),
            "viewer host requests must not carry actor-owned semantics: {forbidden}"
        );
    }
    for forbidden in [
        "MosaicRequest::OpenProjectRoi",
        "MosaicRequest::OpenProjectRoiView",
        "MosaicRequest::OpenProject(",
        "MosaicRequest::SaveProject(",
        "MosaicRequest::OpenLocalPath",
        "MosaicRequest::OpenProjectMosaic",
        "MosaicRequest::PreloadObjectSegmentations",
        "MosaicRequest::ClearObjectCache",
    ] {
        assert!(
            !root_app.contains(forbidden) && !mosaic.contains(forbidden),
            "mosaic host requests must not carry actor-owned semantics: {forbidden}"
        );
    }
}

#[test]
fn application_state_ownership_ledger_covers_every_host_field_exactly_once() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let ledger: serde_json::Value =
        serde_json::from_str(&source(root.join("api/state-ownership-ledger.json")))
            .expect("state ownership ledger must be valid JSON");
    assert_eq!(ledger["schema_version"], 1);

    let owners = ledger["owners"]
        .as_object()
        .expect("ownership ledger owners must be an object");
    let targets = [
        ("OmeZarrViewerApp", "src/app/mod.rs"),
        ("RootApp", "src/root_app.rs"),
        ("MosaicViewerApp", "src/mosaic/mod.rs"),
    ];
    let allowed_classes = BTreeSet::from([
        "actor_projection",
        "mixed_compatibility",
        "platform_effect",
        "renderer",
        "shared_resource",
        "transient_ui",
    ]);
    let allowed_dispositions = BTreeSet::from(["delete", "narrow", "replace", "retain"]);
    let required_metadata = [
        "canonical_writer",
        "projection_source",
        "renderer_consumers",
        "persistence_consumers",
        "native_commit_point",
    ];
    let mut total_fields = 0usize;

    for (struct_name, source_path) in targets {
        let actual = struct_field_names(&source(root.join(source_path)), struct_name);
        let entries = owners
            .get(struct_name)
            .and_then(serde_json::Value::as_array)
            .unwrap_or_else(|| panic!("missing ownership entries for {struct_name}"));
        let mut classified = BTreeMap::<String, String>::new();
        let mut ids = BTreeSet::new();

        for entry in entries {
            let id = entry["id"]
                .as_str()
                .unwrap_or_else(|| panic!("{struct_name} ownership entry is missing an id"));
            assert!(
                ids.insert(id.to_string()),
                "duplicate ownership id {struct_name}.{id}"
            );
            let current_class = entry["current_class"]
                .as_str()
                .unwrap_or_else(|| panic!("{struct_name}.{id} is missing current_class"));
            assert!(
                allowed_classes.contains(current_class),
                "invalid current_class {current_class} for {struct_name}.{id}"
            );
            let disposition = entry["disposition"]
                .as_str()
                .unwrap_or_else(|| panic!("{struct_name}.{id} is missing disposition"));
            assert!(
                allowed_dispositions.contains(disposition),
                "invalid disposition {disposition} for {struct_name}.{id}"
            );
            let milestone = entry["milestone"]
                .as_u64()
                .unwrap_or_else(|| panic!("{struct_name}.{id} is missing milestone"));
            assert!(
                (2..=8).contains(&milestone),
                "invalid milestone {milestone} for {struct_name}.{id}"
            );
            if milestone == 3 {
                assert_ne!(
                    current_class, "mixed_compatibility",
                    "Milestone 3 ownership must no longer mix actor projections with local drafts: {struct_name}.{id}"
                );
                assert_eq!(
                    disposition, "retain",
                    "Milestone 3 ownership row remains open: {struct_name}.{id}"
                );
            }
            for key in required_metadata {
                assert!(
                    entry[key]
                        .as_str()
                        .is_some_and(|value| !value.trim().is_empty()),
                    "{struct_name}.{id} is missing {key}"
                );
            }
            let fields = entry["fields"]
                .as_array()
                .unwrap_or_else(|| panic!("{struct_name}.{id} fields must be an array"));
            assert!(!fields.is_empty(), "{struct_name}.{id} has no fields");
            for field in fields {
                let field = field
                    .as_str()
                    .unwrap_or_else(|| panic!("{struct_name}.{id} contains a non-string field"));
                assert!(
                    classified
                        .insert(field.to_string(), id.to_string())
                        .is_none(),
                    "{struct_name}.{field} is classified more than once"
                );
            }
        }

        let classified = classified.into_keys().collect::<BTreeSet<_>>();
        let missing = actual.difference(&classified).cloned().collect::<Vec<_>>();
        let unknown = classified.difference(&actual).cloned().collect::<Vec<_>>();
        assert!(
            missing.is_empty() && unknown.is_empty(),
            "{struct_name} ownership mismatch; missing={missing:?}, unknown={unknown:?}"
        );
        total_fields += actual.len();
    }

    assert_eq!(
        total_fields, 288,
        "review the ownership ledger when host fields change"
    );
}
