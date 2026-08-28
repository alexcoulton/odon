//! Object source inspection, parsing, worker loading, and renderer/control resource preparation.

use super::*;

pub(super) fn check_cancel(cancel: &AtomicBool) -> anyhow::Result<()> {
    if cancel.load(Ordering::Relaxed) {
        anyhow::bail!("object load cancelled");
    }
    Ok(())
}

pub(super) fn load_in_thread(
    path: PathBuf,
    downsample_factor: f32,
    load_options: Option<ObjectLoadOptions>,
    request_id: u64,
    cancel: &AtomicBool,
) -> anyhow::Result<LoadResult> {
    // Format dispatch is based on the path. Regardless of source, each branch is normalized into
    // `GeoJsonObjectFeature` records plus enough metadata to rebuild display/analysis state.
    check_cancel(cancel)?;
    let (display_mode, objects, lazy_parquet_source) = if is_parquet_objects_path(&path) {
        let parquet_options = match load_options.as_ref() {
            Some(ObjectLoadOptions::Parquet(options)) => Some(options),
            _ => None,
        };
        let display_mode = parquet_options
            .map(|opts| opts.display_mode)
            .unwrap_or(ObjectDisplayMode::Polygons);
        let schema = inspect_shapes_object_schema(&path)?;
        check_cancel(cancel)?;
        let loaded_property_columns = parquet_loaded_property_columns(parquet_options, &schema);
        let objects = match parquet_options {
            Some(ObjectParquetLoadOptions {
                display_mode: ObjectDisplayMode::Points,
                source: ObjectParquetSource::Geometry(shape_options),
            }) => parse_geoparquet_centroid_point_objects(&path, shape_options, cancel)?,
            Some(ObjectParquetLoadOptions {
                display_mode: ObjectDisplayMode::Points,
                source:
                    ObjectParquetSource::XYColumns {
                        x_column,
                        y_column,
                        property_columns,
                    },
            }) => parse_geoparquet_xy_point_features(
                &path,
                x_column,
                y_column,
                property_columns.as_deref(),
                cancel,
            )?,
            _ => parse_geoparquet_objects(&path, parquet_options, cancel)?,
        };
        (
            display_mode,
            objects,
            Some(LazyParquetSource {
                available_property_columns: schema.property_columns,
                numeric_property_columns: schema.numeric_property_columns,
                loaded_property_columns,
            }),
        )
    } else if is_csv_objects_path(&path) {
        let csv_options = match load_options.as_ref() {
            Some(ObjectLoadOptions::Csv(options)) => Some(options),
            _ => None,
        };
        (
            ObjectDisplayMode::Points,
            parse_csv_objects(&path, csv_options, cancel)?,
            None,
        )
    } else {
        check_cancel(cancel)?;
        (
            ObjectDisplayMode::Polygons,
            parse_geojson_objects(&path, downsample_factor)?,
            None,
        )
    };
    load_result_from_objects(
        request_id,
        path,
        downsample_factor,
        SpatialDataTransform2::default(),
        display_mode,
        objects,
        lazy_parquet_source,
        cancel,
    )
}

pub fn preload_objects_from_path(
    path: PathBuf,
    downsample_factor: f32,
    settings: ObjectPreloadSettings,
) -> anyhow::Result<PreloadedObjectLayer> {
    let cancel = AtomicBool::new(false);
    let result = match settings.mode {
        ObjectPreloadMode::FullGeometry => {
            if is_parquet_objects_path(&path) {
                let load_options = minimal_parquet_load_options(&path)
                    .ok()
                    .map(ObjectLoadOptions::Parquet);
                let mut result =
                    load_in_thread(path.clone(), downsample_factor, load_options, 0, &cancel)?;
                if !settings.lazy_properties {
                    load_all_parquet_property_columns_into_result(&path, &mut result, &cancel)?;
                }
                Ok(result)
            } else {
                load_in_thread(path, downsample_factor, None, 0, &cancel)
            }
        }
        ObjectPreloadMode::CentroidPoints => load_centroid_points_in_thread(
            path,
            downsample_factor,
            settings.lazy_properties,
            0,
            &cancel,
        ),
    }?;
    Ok(PreloadedObjectLayer { result })
}

pub fn load_control_object_resource(
    path: PathBuf,
    downsample_factor: f32,
) -> anyhow::Result<odon::model::ControlObjectResource> {
    let preloaded = preload_objects_from_path(
        path.clone(),
        downsample_factor,
        ObjectPreloadSettings {
            mode: ObjectPreloadMode::FullGeometry,
            lazy_properties: false,
        },
    )?;
    control_resource_from_preloaded(path, downsample_factor, preloaded)
}

pub fn load_control_object_resource_with_options(
    path: PathBuf,
    downsample_factor: f32,
    options: Option<&serde_json::Value>,
) -> anyhow::Result<odon::model::ControlObjectResource> {
    let Some(options) = options else {
        return load_control_object_resource(path, downsample_factor);
    };
    if let Some(preload) = options.get("project_preload") {
        let mode = match preload
            .get("mode")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("full_geometry")
        {
            "full_geometry" => ObjectPreloadMode::FullGeometry,
            "centroid_points" => ObjectPreloadMode::CentroidPoints,
            other => anyhow::bail!("unknown project object preload mode '{other}'"),
        };
        let settings = ObjectPreloadSettings {
            mode,
            lazy_properties: preload
                .get("lazy_properties")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(true),
        };
        let preloaded = preload_objects_from_path(path.clone(), downsample_factor, settings)?;
        return control_resource_from_preloaded(path, downsample_factor, preloaded);
    }
    let property_columns = options
        .get("property_columns")
        .and_then(serde_json::Value::as_array)
        .map(|columns| {
            columns
                .iter()
                .map(|column| {
                    column
                        .as_str()
                        .map(str::to_string)
                        .ok_or_else(|| anyhow!("property_columns must contain strings"))
                })
                .collect::<anyhow::Result<Vec<_>>>()
        })
        .transpose()?;
    let format = options
        .get("format")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| anyhow!("object loader options require format"))?;
    let load_options = match format {
        "geoparquet" => {
            let display_mode = match options
                .get("display_mode")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("polygons")
            {
                "polygons" => ObjectDisplayMode::Polygons,
                "points" => ObjectDisplayMode::Points,
                other => anyhow::bail!("unknown GeoParquet display_mode '{other}'"),
            };
            let source = match options
                .get("source")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("geometry")
            {
                "geometry" => ObjectParquetSource::Geometry(ShapesLoadOptions {
                    transform: SpatialDataTransform2::default(),
                    geometry_column: required_loader_option(options, "geometry_column")?,
                    property_columns,
                }),
                "xy_columns" => ObjectParquetSource::XYColumns {
                    x_column: required_loader_option(options, "x_column")?,
                    y_column: required_loader_option(options, "y_column")?,
                    property_columns,
                },
                other => anyhow::bail!("unknown GeoParquet source '{other}'"),
            };
            ObjectLoadOptions::Parquet(ObjectParquetLoadOptions {
                display_mode,
                source,
            })
        }
        "csv" => ObjectLoadOptions::Csv(ObjectCsvLoadOptions {
            x_column: required_loader_option(options, "x_column")?,
            y_column: required_loader_option(options, "y_column")?,
            property_columns,
        }),
        other => anyhow::bail!("unknown object loader format '{other}'"),
    };
    let cancel = AtomicBool::new(false);
    let result = load_in_thread(
        path.clone(),
        downsample_factor,
        Some(load_options),
        0,
        &cancel,
    )?;
    control_resource_from_preloaded(path, downsample_factor, PreloadedObjectLayer { result })
}

/// Prepare a SpatialData shape as a canonical actor resource and renderer preload in one pass.
///
/// The transform is applied while parsing, so both background queries and the eventual renderer
/// use the same image-relative world coordinates.
pub fn load_control_spatialdata_object_resource(
    path: PathBuf,
    transform: SpatialDataTransform2,
) -> anyhow::Result<odon::model::ControlObjectResource> {
    let cancel = AtomicBool::new(false);
    let result = load_spatialdata_in_thread(path.clone(), transform, 0, &cancel)?;
    control_resource_from_preloaded(path, 1.0, PreloadedObjectLayer { result })
}

pub(super) fn required_loader_option(
    options: &serde_json::Value,
    name: &str,
) -> anyhow::Result<String> {
    options
        .get(name)
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .ok_or_else(|| anyhow!("object loader option '{name}' is required"))
}

pub(super) fn control_resource_from_preloaded(
    path: PathBuf,
    downsample_factor: f32,
    preloaded: PreloadedObjectLayer,
) -> anyhow::Result<odon::model::ControlObjectResource> {
    let result = preloaded.result.clone();
    let loaded_properties = result.property_store.loaded_keys();
    let mut property_names = result
        .object_property_keys
        .iter()
        .chain(loaded_properties.iter())
        .cloned()
        .collect::<BTreeSet<_>>();
    property_names.insert("id".to_string());
    let features = Arc::clone(&result.objects);
    let property_names = property_names.into_iter().collect::<Vec<_>>();
    let property_source: Arc<dyn odon::model::ControlObjectPropertySource> =
        Arc::new(result.property_store.clone());
    let numeric_summaries = odon::model::ControlObjectResource::build_numeric_summaries(
        features.as_ref(),
        &property_names,
        property_source.as_ref(),
    );
    let memory_diagnostics = Arc::clone(&result.memory_diagnostics);
    Ok(odon::model::ControlObjectResource {
        source: path,
        downsample_factor,
        features,
        property_names: Arc::new(property_names),
        property_source,
        numeric_summaries,
        memory_diagnostics,
        renderer_payload: Some(Arc::new(preloaded)),
    })
}

pub(super) fn load_centroid_points_in_thread(
    path: PathBuf,
    downsample_factor: f32,
    lazy_properties: bool,
    request_id: u64,
    cancel: &AtomicBool,
) -> anyhow::Result<LoadResult> {
    if !is_parquet_objects_path(&path) {
        return load_in_thread(path, downsample_factor, None, request_id, cancel);
    }

    let schema = inspect_shapes_object_schema(&path)?;
    let x_column = preferred_xy_column_exact(
        &schema.numeric_property_columns,
        &["x_centroid", "x", "x_centroid_image", "centroid_x"],
    );
    let y_column = preferred_xy_column_exact(
        &schema.numeric_property_columns,
        &["y_centroid", "y", "y_centroid_image", "centroid_y"],
    );
    let initial_property_columns = Some(preferred_object_id_property_columns(
        &schema.property_columns,
    ));
    let loaded_property_columns = initial_property_columns
        .clone()
        .unwrap_or_else(|| schema.property_columns.clone())
        .into_iter()
        .collect::<HashSet<_>>();
    if let (Some(x_column), Some(y_column)) = (x_column, y_column) {
        let objects = parse_geoparquet_xy_point_features(
            &path,
            x_column.as_str(),
            y_column.as_str(),
            initial_property_columns.as_deref(),
            cancel,
        )?;
        let lazy_parquet_source = Some(LazyParquetSource {
            available_property_columns: schema.property_columns.clone(),
            numeric_property_columns: schema.numeric_property_columns.clone(),
            loaded_property_columns,
        });
        let mut result = load_result_from_objects(
            request_id,
            path.clone(),
            downsample_factor,
            SpatialDataTransform2::default(),
            ObjectDisplayMode::Points,
            objects,
            lazy_parquet_source,
            cancel,
        )?;
        if !lazy_properties {
            load_all_parquet_property_columns_into_result(&path, &mut result, cancel)?;
        }
        return Ok(result);
    }

    if schema.geometry_candidates.is_empty() {
        anyhow::bail!("No centroid columns or supported geometry columns found in GeoParquet.");
    }
    let geometry_column = preferred_geometry_column(&schema);
    let objects = parse_geoparquet_centroid_point_objects(
        &path,
        &ShapesLoadOptions {
            transform: SpatialDataTransform2::default(),
            geometry_column: geometry_column.clone(),
            property_columns: initial_property_columns,
        },
        cancel,
    )?;
    if objects.is_empty() {
        anyhow::bail!("no centroid point objects found in GeoParquet");
    }
    let lazy_parquet_source = Some(LazyParquetSource {
        available_property_columns: schema.property_columns.clone(),
        numeric_property_columns: schema.numeric_property_columns.clone(),
        loaded_property_columns,
    });
    let mut result = load_result_from_objects(
        request_id,
        path.clone(),
        downsample_factor,
        SpatialDataTransform2::default(),
        ObjectDisplayMode::Points,
        objects,
        lazy_parquet_source,
        cancel,
    )?;
    if !lazy_properties {
        load_all_parquet_property_columns_into_result(&path, &mut result, cancel)?;
    }
    Ok(result)
}

pub(super) fn load_spatialdata_in_thread(
    path: PathBuf,
    transform: SpatialDataTransform2,
    request_id: u64,
    cancel: &AtomicBool,
) -> anyhow::Result<LoadResult> {
    let objects = parse_spatialdata_objects(&path, transform, cancel)?;
    load_result_from_objects(
        request_id,
        path,
        1.0,
        transform,
        ObjectDisplayMode::Polygons,
        objects,
        None,
        cancel,
    )
}

pub(super) fn load_result_from_objects(
    request_id: u64,
    path: PathBuf,
    downsample_factor: f32,
    display_transform: SpatialDataTransform2,
    display_mode: ObjectDisplayMode,
    mut objects: Vec<GeoJsonObjectFeature>,
    lazy_parquet_source: Option<LazyParquetSource>,
    cancel: &AtomicBool,
) -> anyhow::Result<LoadResult> {
    check_cancel(cancel)?;
    let mut property_store = ObjectPropertyStore::from_available_columns(
        lazy_parquet_source
            .as_ref()
            .map(|source| source.available_property_columns.clone())
            .unwrap_or_default(),
    );
    if let Some(source) = lazy_parquet_source.as_ref() {
        // GeoParquet geometry decoding never constructs row maps. Hydrate the small selected
        // identity set directly into typed columns, indexed back through stable source rows. The
        // canonical id already lives on the feature itself and is exposed synthetically.
        for property in &source.loaded_property_columns {
            if property == "id" {
                continue;
            }
            check_cancel(cancel)?;
            let values = load_parquet_property_values(&path, property, cancel)
                .with_context(|| format!("failed to load identity column '{property}'"))?;
            property_store.insert_column(
                property.clone(),
                object_property_column_from_loaded_values(&objects, &values),
            );
        }
        for object in &mut objects {
            object.inline_properties.clear();
        }
    }
    let bounds = objects.iter().map(|o| o.bbox_world).collect::<Vec<_>>();
    let bounds_local =
        union_rects(&bounds).ok_or_else(|| anyhow!("no valid object bounds after parsing"))?;
    check_cancel(cancel)?;
    let bins = ObjectIndexBins::build(&bounds, 512.0)
        .ok_or_else(|| anyhow!("no valid object bounds after parsing"))?;
    check_cancel(cancel)?;
    let render_lods = if display_mode == ObjectDisplayMode::Points {
        Vec::new()
    } else {
        build_render_lods(&objects)?
    };
    check_cancel(cancel)?;
    let object_fill_mesh = if display_mode == ObjectDisplayMode::Polygons {
        build_object_fill_mesh(&objects).ok()
    } else {
        None
    };
    check_cancel(cancel)?;
    let object_selection_lods = if display_mode == ObjectDisplayMode::Polygons {
        build_object_selection_render_lods(&objects).ok()
    } else {
        None
    };
    check_cancel(cancel)?;
    let (point_positions_world, point_values, point_lods) =
        build_object_point_payload(&objects, display_transform);
    check_cancel(cancel)?;
    let mut object_property_keys = discover_property_keys(&objects);
    object_property_keys.extend(property_store.loaded_keys());
    object_property_keys.sort();
    object_property_keys.dedup();
    let mut scalar_property_keys = discover_scalar_property_keys(&objects);
    scalar_property_keys.extend(property_store.numeric_keys());
    scalar_property_keys.sort();
    scalar_property_keys.dedup();
    let mut color_property_keys = discover_categorical_color_keys(&objects);
    color_property_keys.extend(
        property_store
            .loaded_keys()
            .into_iter()
            .filter(|key| property_store.loaded_column_is_categorical(key, 24)),
    );
    color_property_keys.sort();
    color_property_keys.dedup();
    let mut result = LoadResult {
        request_id,
        render_resource_cache_id: next_object_render_resource_cache_id(),
        path,
        downsample_factor,
        display_transform,
        display_mode,
        objects: Arc::new(objects),
        bins: Arc::new(bins),
        render_lods,
        object_fill_mesh,
        object_selection_lods,
        point_positions_world,
        point_values,
        point_lods,
        object_property_keys,
        scalar_property_keys,
        color_property_keys,
        property_store,
        lazy_parquet_source,
        bounds_local,
        memory_diagnostics: Arc::new(Default::default()),
    };
    result.memory_diagnostics = Arc::new(crate::objects::memory::load_result_memory_diagnostics(
        &result,
    ));
    Ok(result)
}

pub(super) fn load_all_parquet_property_columns_into_result(
    path: &Path,
    result: &mut LoadResult,
    cancel: &AtomicBool,
) -> anyhow::Result<()> {
    let Some(source) = result.lazy_parquet_source.as_mut() else {
        return Ok(());
    };
    let available_columns = source.available_property_columns.clone();
    for property_key in available_columns {
        check_cancel(cancel)?;
        if result.property_store.has_loaded(&property_key) {
            source.loaded_property_columns.insert(property_key);
            continue;
        }
        let values = load_parquet_property_values(path, &property_key, cancel)
            .with_context(|| format!("failed to load property column '{property_key}'"))?;
        let column = object_property_column_from_loaded_values(result.objects.as_ref(), &values);
        let is_categorical = column.is_categorical(24);
        result
            .property_store
            .insert_column(property_key.clone(), column);
        source.loaded_property_columns.insert(property_key.clone());
        if is_categorical
            && !result
                .color_property_keys
                .iter()
                .any(|key| key == &property_key)
        {
            result.color_property_keys.push(property_key);
        }
    }
    result.color_property_keys.sort();
    Ok(())
}

pub(super) fn preferred_geometry_column(schema: &ShapesObjectSchema) -> String {
    schema
        .geometry_candidates
        .iter()
        .find(|name| name.as_str() == "geometry")
        .cloned()
        .or_else(|| schema.geometry_candidates.first().cloned())
        .unwrap_or_else(|| "geometry".to_string())
}

pub(super) fn preferred_object_id_property_columns(property_columns: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    for key in [
        "id",
        "instance_id",
        "instance_id_polygon",
        "cell_id",
        "label",
        "name",
        "polygon_name",
    ] {
        if property_columns.iter().any(|name| name == key) {
            out.push(key.to_string());
        }
    }
    out
}

pub(super) fn minimal_parquet_load_options(
    path: &Path,
) -> anyhow::Result<ObjectParquetLoadOptions> {
    let schema = inspect_shapes_object_schema(path)?;
    if schema.geometry_candidates.is_empty() {
        anyhow::bail!("No supported binary geometry columns found in GeoParquet.");
    }
    let geometry_column = preferred_geometry_column(&schema);
    let property_columns = preferred_object_id_property_columns(&schema.property_columns);
    Ok(ObjectParquetLoadOptions {
        display_mode: ObjectDisplayMode::Polygons,
        source: ObjectParquetSource::Geometry(ShapesLoadOptions {
            transform: SpatialDataTransform2::default(),
            geometry_column,
            property_columns: Some(property_columns),
        }),
    })
}

pub(super) fn parquet_loaded_property_columns(
    options: Option<&ObjectParquetLoadOptions>,
    schema: &ShapesObjectSchema,
) -> HashSet<String> {
    match options {
        Some(ObjectParquetLoadOptions {
            source: ObjectParquetSource::Geometry(shape_options),
            ..
        }) => shape_options
            .property_columns
            .clone()
            .unwrap_or_else(|| schema.property_columns.clone())
            .into_iter()
            .collect(),
        Some(ObjectParquetLoadOptions {
            source:
                ObjectParquetSource::XYColumns {
                    property_columns, ..
                },
            ..
        }) => property_columns
            .clone()
            .unwrap_or_else(|| schema.property_columns.clone())
            .into_iter()
            .collect(),
        None => schema.property_columns.iter().cloned().collect(),
    }
}

pub(super) fn load_parquet_property_values_for_loaded_objects(
    path: &Path,
    property_key: &str,
) -> anyhow::Result<LoadedPropertyValues> {
    let cancel = AtomicBool::new(false);
    load_parquet_property_values(path, property_key, &cancel)
}

fn load_parquet_property_values(
    path: &Path,
    property_key: &str,
    cancel: &AtomicBool,
) -> anyhow::Result<LoadedPropertyValues> {
    if let Some(values) = load_shapes_f32_property_column(path, property_key, cancel)? {
        return Ok(LoadedPropertyValues::F32(values));
    }
    load_shapes_property_values_by_row(path, property_key, cancel)
        .map(LoadedPropertyValues::ValuesByRow)
}

pub(super) fn object_property_column_from_loaded_values(
    objects: &[GeoJsonObjectFeature],
    values: &LoadedPropertyValues,
) -> ObjectPropertyColumn {
    match values {
        LoadedPropertyValues::F32(values) => ObjectPropertyColumn::F32(Arc::new(
            NullableF32Column::from_optional_values(objects.iter().map(|object| {
                object
                    .source_row_index
                    .and_then(|row_index| values.get(row_index))
            })),
        )),
        LoadedPropertyValues::ValuesByRow(values) => {
            ObjectPropertyColumn::from_values_by_row(objects, values)
        }
    }
}

pub(super) fn parse_geojson_objects(
    path: &Path,
    downsample_factor: f32,
) -> anyhow::Result<Vec<GeoJsonObjectFeature>> {
    if !path.exists() {
        anyhow::bail!("missing GeoJSON file: {}", path.to_string_lossy());
    }
    let text = std::fs::read_to_string(path)
        .map_err(anyhow::Error::from)
        .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).map_err(anyhow::Error::from))?;
    let feats = text
        .get("features")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    let scale = downsample_factor.max(1e-6);

    let mut out = Vec::new();
    for (feature_index, feat) in feats.iter().enumerate() {
        let Some(geom) = feat.get("geometry") else {
            continue;
        };
        let polygons_world = parse_feature_polygons(geom, scale);
        if polygons_world.is_empty() {
            continue;
        }
        let Some((bbox_world, area_px, perimeter_px, centroid_world)) =
            summarize_geometry(&polygons_world)
        else {
            continue;
        };

        let properties = feat
            .get("properties")
            .and_then(|v| v.as_object())
            .cloned()
            .unwrap_or_default();
        let id = feature_id(feat, &properties, feature_index);

        out.push(GeoJsonObjectFeature {
            id,
            polygons_world,
            point_position_world: None,
            bbox_world,
            area_px,
            perimeter_px,
            centroid_world,
            inline_properties: properties,
            source_row_index: None,
        });
    }

    if out.is_empty() {
        anyhow::bail!("no supported polygon objects in GeoJSON");
    }
    Ok(out)
}

pub(super) fn parse_spatialdata_objects(
    path: &Path,
    transform: SpatialDataTransform2,
    cancel: &AtomicBool,
) -> anyhow::Result<Vec<GeoJsonObjectFeature>> {
    if !path.exists() {
        anyhow::bail!(
            "missing SpatialData shapes parquet: {}",
            path.to_string_lossy()
        );
    }
    let loaded = load_shapes_objects(
        path,
        &ShapesLoadOptions {
            transform,
            ..ShapesLoadOptions::default()
        },
        cancel,
    )?;
    let mut out = Vec::with_capacity(loaded.len());
    for obj in loaded {
        check_cancel(cancel)?;
        let Some((bbox_world, area_px, perimeter_px, centroid_world)) =
            summarize_geometry(&obj.polygons_world)
        else {
            continue;
        };
        out.push(GeoJsonObjectFeature {
            id: obj.id,
            polygons_world: obj.polygons_world,
            point_position_world: obj.point_position_world,
            bbox_world,
            area_px,
            perimeter_px,
            centroid_world,
            inline_properties: obj.properties,
            source_row_index: None,
        });
    }
    if out.is_empty() {
        anyhow::bail!("no polygon objects found in SpatialData shapes parquet");
    }
    Ok(out)
}

pub(super) fn parse_geoparquet_objects(
    path: &Path,
    options: Option<&ObjectParquetLoadOptions>,
    cancel: &AtomicBool,
) -> anyhow::Result<Vec<GeoJsonObjectFeature>> {
    // GeoParquet can contribute either polygon geometries or point-like objects built from XY
    // columns. Both paths are normalized into the same object feature struct so the rest of the
    // layer does not care which representation produced them.
    if !path.exists() {
        anyhow::bail!("missing GeoParquet file: {}", path.to_string_lossy());
    }
    let default_options = ObjectParquetLoadOptions {
        display_mode: ObjectDisplayMode::Polygons,
        source: ObjectParquetSource::Geometry(ShapesLoadOptions::default()),
    };
    let loaded = match &options.unwrap_or(&default_options).source {
        ObjectParquetSource::Geometry(shape_options) => {
            load_shapes_objects(path, shape_options, cancel)?
        }
        ObjectParquetSource::XYColumns {
            x_column,
            y_column,
            property_columns,
        } => load_shapes_xy_point_objects(
            path,
            x_column,
            y_column,
            property_columns.as_deref(),
            cancel,
        )?,
    };
    let mut out = Vec::with_capacity(loaded.len());
    for obj in loaded {
        check_cancel(cancel)?;
        let Some((bbox_world, area_px, perimeter_px, centroid_world)) =
            summarize_geometry(&obj.polygons_world)
        else {
            continue;
        };
        out.push(GeoJsonObjectFeature {
            id: obj.id,
            polygons_world: obj.polygons_world,
            point_position_world: obj.point_position_world,
            bbox_world,
            area_px,
            perimeter_px,
            centroid_world,
            inline_properties: obj.properties,
            source_row_index: obj.source_row_index,
        });
    }
    if out.is_empty() {
        anyhow::bail!("no polygon objects found in GeoParquet");
    }
    Ok(out)
}

pub(super) fn parse_geoparquet_xy_point_features(
    path: &Path,
    x_column: &str,
    y_column: &str,
    property_columns: Option<&[String]>,
    cancel: &AtomicBool,
) -> anyhow::Result<Vec<GeoJsonObjectFeature>> {
    if !path.exists() {
        anyhow::bail!("missing GeoParquet file: {}", path.to_string_lossy());
    }
    let loaded = load_shapes_xy_point_features(path, x_column, y_column, property_columns, cancel)?;
    let mut out = Vec::with_capacity(loaded.len());
    for obj in loaded {
        check_cancel(cancel)?;
        out.push(GeoJsonObjectFeature {
            id: obj.id,
            polygons_world: Vec::new(),
            point_position_world: Some(obj.point_world),
            bbox_world: obj.bbox_world,
            area_px: obj.area_px,
            perimeter_px: obj.perimeter_px,
            centroid_world: obj.point_world,
            inline_properties: obj.properties,
            source_row_index: obj.source_row_index,
        });
    }
    if out.is_empty() {
        anyhow::bail!("no point objects found in GeoParquet");
    }
    Ok(out)
}

pub(super) fn parse_geoparquet_centroid_point_objects(
    path: &Path,
    options: &ShapesLoadOptions,
    cancel: &AtomicBool,
) -> anyhow::Result<Vec<GeoJsonObjectFeature>> {
    if !path.exists() {
        anyhow::bail!("missing GeoParquet file: {}", path.to_string_lossy());
    }
    let loaded = load_shapes_centroid_point_objects(path, options, cancel)?;
    let mut out = Vec::with_capacity(loaded.len());
    for obj in loaded {
        check_cancel(cancel)?;
        out.push(GeoJsonObjectFeature {
            id: obj.id,
            polygons_world: Vec::new(),
            point_position_world: Some(obj.point_world),
            bbox_world: obj.bbox_world,
            area_px: obj.area_px,
            perimeter_px: obj.perimeter_px,
            centroid_world: obj.point_world,
            inline_properties: obj.properties,
            source_row_index: obj.source_row_index,
        });
    }
    if out.is_empty() {
        anyhow::bail!("no centroid point objects found in GeoParquet");
    }
    Ok(out)
}

#[derive(Debug)]
pub(super) struct CsvObjectSchema {
    pub(super) property_columns: Vec<String>,
    pub(super) numeric_columns: Vec<String>,
}

pub(super) fn inspect_csv_object_schema(path: &Path) -> anyhow::Result<CsvObjectSchema> {
    if !path.exists() {
        anyhow::bail!("missing CSV file: {}", path.to_string_lossy());
    }
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)
        .with_context(|| format!("failed to open CSV: {}", path.to_string_lossy()))?;
    let headers = reader
        .headers()
        .with_context(|| format!("failed to read CSV headers: {}", path.to_string_lossy()))?
        .iter()
        .map(|s| s.trim().to_string())
        .collect::<Vec<_>>();
    let mut numeric_ok = vec![true; headers.len()];
    let mut numeric_seen = vec![false; headers.len()];
    for record in reader.records().take(1024) {
        let record = record.with_context(|| {
            format!(
                "failed to read CSV records while inspecting: {}",
                path.to_string_lossy()
            )
        })?;
        for (idx, value) in record.iter().enumerate() {
            let trimmed = value.trim();
            if trimmed.is_empty() {
                continue;
            }
            if trimmed.parse::<f64>().is_ok() {
                if let Some(seen) = numeric_seen.get_mut(idx) {
                    *seen = true;
                }
            } else if let Some(ok) = numeric_ok.get_mut(idx) {
                *ok = false;
            }
        }
    }
    let numeric_columns = headers
        .iter()
        .enumerate()
        .filter(|(idx, _)| {
            numeric_ok.get(*idx).copied().unwrap_or(false)
                && numeric_seen.get(*idx).copied().unwrap_or(false)
        })
        .map(|(_, name)| name.clone())
        .collect::<Vec<_>>();
    Ok(CsvObjectSchema {
        property_columns: headers,
        numeric_columns,
    })
}

pub(super) fn parse_csv_objects(
    path: &Path,
    options: Option<&ObjectCsvLoadOptions>,
    cancel: &AtomicBool,
) -> anyhow::Result<Vec<GeoJsonObjectFeature>> {
    // CSV import is point-oriented: infer X/Y columns, then lift each row into an object feature
    // whose geometry is represented primarily by a point position rather than polygon rings.
    if !path.exists() {
        anyhow::bail!("missing CSV file: {}", path.to_string_lossy());
    }
    let schema = inspect_csv_object_schema(path)?;
    let x_names = [
        "x_centroid",
        "x",
        "x_centroid_image",
        "centroid_x",
        "xcoord",
    ];
    let y_names = [
        "y_centroid",
        "y",
        "y_centroid_image",
        "centroid_y",
        "ycoord",
    ];
    let x_column = options
        .map(|opts| opts.x_column.clone())
        .or_else(|| preferred_xy_column_exact(&schema.property_columns, &x_names))
        .or_else(|| preferred_xy_column(&schema.numeric_columns, &x_names))
        .ok_or_else(|| anyhow!("CSV is missing a usable X column"))?;
    let y_column = options
        .map(|opts| opts.y_column.clone())
        .or_else(|| preferred_xy_column_exact(&schema.property_columns, &y_names))
        .or_else(|| preferred_xy_column(&schema.numeric_columns, &y_names))
        .ok_or_else(|| anyhow!("CSV is missing a usable Y column"))?;
    let selected_property_columns = options
        .and_then(|opts| opts.property_columns.as_ref())
        .cloned()
        .unwrap_or_else(|| schema.property_columns.clone());

    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)
        .with_context(|| format!("failed to open CSV: {}", path.to_string_lossy()))?;
    let headers = reader
        .headers()
        .with_context(|| format!("failed to read CSV headers: {}", path.to_string_lossy()))?
        .iter()
        .map(|s| s.trim().to_string())
        .collect::<Vec<_>>();
    let x_idx = headers
        .iter()
        .position(|name| name == &x_column)
        .with_context(|| format!("missing x column '{x_column}'"))?;
    let y_idx = headers
        .iter()
        .position(|name| name == &y_column)
        .with_context(|| format!("missing y column '{y_column}'"))?;

    let selected = selected_property_columns
        .iter()
        .cloned()
        .collect::<HashSet<_>>();
    let mut out = Vec::new();
    for (row_index, record) in reader.records().enumerate() {
        check_cancel(cancel)?;
        let record = record.with_context(|| {
            format!(
                "failed reading CSV row {}: {}",
                row_index + 1,
                path.to_string_lossy()
            )
        })?;
        let Some(x) = record.get(x_idx).and_then(|s| s.trim().parse::<f32>().ok()) else {
            continue;
        };
        let Some(y) = record.get(y_idx).and_then(|s| s.trim().parse::<f32>().ok()) else {
            continue;
        };
        if !x.is_finite() || !y.is_finite() {
            continue;
        }

        let mut properties = serde_json::Map::new();
        properties.insert(x_column.clone(), serde_json::Value::from(x as f64));
        properties.insert(y_column.clone(), serde_json::Value::from(y as f64));
        for (idx, name) in headers.iter().enumerate() {
            if idx == x_idx || idx == y_idx {
                continue;
            }
            if !selected.contains(name) {
                continue;
            }
            let Some(raw) = record.get(idx) else {
                continue;
            };
            if raw.trim().is_empty() {
                continue;
            }
            properties.insert(name.clone(), csv_cell_to_json(raw));
        }

        let id = object_id_from_properties_local(&properties)
            .unwrap_or_else(|| (row_index + 1).to_string());
        properties.insert("id".to_string(), serde_json::Value::String(id.clone()));
        let center = egui::pos2(x, y);
        let polygons_world = vec![circle_polyline_local(center, 4.0, 8)];
        let Some((bbox_world, area_px, perimeter_px, centroid_world)) =
            summarize_geometry(&polygons_world)
        else {
            continue;
        };
        out.push(GeoJsonObjectFeature {
            id,
            polygons_world,
            point_position_world: Some(center),
            bbox_world,
            area_px,
            perimeter_px,
            centroid_world,
            inline_properties: properties,
            source_row_index: Some(row_index),
        });
    }

    if out.is_empty() {
        anyhow::bail!("no point objects found in CSV");
    }
    Ok(out)
}

pub(super) fn is_parquet_objects_path(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| matches!(ext.to_ascii_lowercase().as_str(), "parquet" | "geoparquet"))
        .unwrap_or(false)
}

pub(super) fn is_csv_objects_path(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("csv"))
        .unwrap_or(false)
}

pub(super) fn csv_cell_to_json(raw: &str) -> serde_json::Value {
    let trimmed = raw.trim();
    if let Ok(value) = trimmed.parse::<i64>() {
        return serde_json::Value::from(value);
    }
    if let Ok(value) = trimmed.parse::<u64>() {
        return serde_json::Value::from(value);
    }
    if let Ok(value) = trimmed.parse::<f64>() {
        if value.is_finite() {
            return serde_json::Value::from(value);
        }
    }
    if trimmed.eq_ignore_ascii_case("true") {
        return serde_json::Value::Bool(true);
    }
    if trimmed.eq_ignore_ascii_case("false") {
        return serde_json::Value::Bool(false);
    }
    serde_json::Value::String(trimmed.to_string())
}

pub(super) fn object_id_from_properties_local(
    properties: &serde_json::Map<String, serde_json::Value>,
) -> Option<String> {
    for key in [
        "id",
        "instance_id",
        "instance_id_polygon",
        "cell_id",
        "label",
        "name",
        "polygon_name",
    ] {
        if let Some(value) = properties.get(key) {
            match value {
                serde_json::Value::String(v) => return Some(v.clone()),
                other => return Some(other.to_string()),
            }
        }
    }
    None
}

pub(super) fn circle_polyline_local(
    center: egui::Pos2,
    radius_world: f32,
    segments: usize,
) -> Vec<egui::Pos2> {
    let n = segments.max(8);
    let mut pts = Vec::with_capacity(n + 1);
    for i in 0..=n {
        let t = (i as f32) * std::f32::consts::TAU / (n as f32);
        pts.push(egui::pos2(
            center.x + radius_world * t.cos(),
            center.y + radius_world * t.sin(),
        ));
    }
    pts
}

pub(super) fn fuzzy_filter_names(query: &str, names: &[String]) -> Vec<String> {
    let trimmed = query.trim();
    if trimmed.is_empty() {
        return names.to_vec();
    }
    let mut ranked = names
        .iter()
        .filter_map(|name| fuzzy_name_score_local(trimmed, name).map(|score| (score, name)))
        .collect::<Vec<_>>();
    ranked.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(b.1)));
    ranked.into_iter().map(|(_, name)| name.clone()).collect()
}

pub(super) fn fuzzy_name_score_local(query: &str, candidate: &str) -> Option<i32> {
    let q = query.trim().to_ascii_lowercase();
    let c = candidate.trim().to_ascii_lowercase();
    if q.is_empty() {
        return Some(0);
    }
    if c == q {
        return Some(0);
    }
    if c.starts_with(&q) {
        return Some(10 + (c.len() as i32 - q.len() as i32).max(0));
    }
    if let Some(idx) = c.find(&q) {
        return Some(100 + idx as i32 + (c.len() as i32 - q.len() as i32).max(0));
    }
    let mut pos = 0usize;
    let mut score = 300i32;
    for ch in q.chars() {
        let rest = &c[pos..];
        let found = rest.find(ch)?;
        score += found as i32;
        pos += found + ch.len_utf8();
    }
    Some(score + (c.len() as i32 - q.len() as i32).max(0))
}

pub(super) fn preferred_xy_column(columns: &[String], preferred_names: &[&str]) -> Option<String> {
    for preferred in preferred_names {
        if let Some(found) = columns
            .iter()
            .find(|name| name.eq_ignore_ascii_case(preferred))
        {
            return Some(found.clone());
        }
    }
    columns.first().cloned()
}

pub(super) fn preferred_xy_column_exact(
    columns: &[String],
    preferred_names: &[&str],
) -> Option<String> {
    for preferred in preferred_names {
        if let Some(found) = columns
            .iter()
            .find(|name| name.eq_ignore_ascii_case(preferred))
        {
            return Some(found.clone());
        }
    }
    None
}

pub(in crate::objects) fn build_object_point_payload(
    objects: &[GeoJsonObjectFeature],
    _display_transform: SpatialDataTransform2,
) -> (
    Arc<Vec<egui::Pos2>>,
    Arc<Vec<f32>>,
    Arc<Vec<FeaturePointLod>>,
) {
    let positions = objects
        .iter()
        .map(object_proxy_position_world)
        .collect::<Vec<_>>();
    let values = vec![1.0f32; positions.len()];
    (Arc::new(positions), Arc::new(values), Arc::new(Vec::new()))
}

pub(in crate::objects) fn object_proxy_position_world(obj: &GeoJsonObjectFeature) -> egui::Pos2 {
    if obj.polygons_world.is_empty() {
        obj.point_position_world.unwrap_or(obj.centroid_world)
    } else {
        obj.bbox_world.center()
    }
}
