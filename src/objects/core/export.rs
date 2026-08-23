//! Selection summaries, table indexing, CSV/GeoParquet export, and export task UI.

use super::*;

impl ObjectsLayer {
    pub(super) fn selection_summary(&self) -> (usize, f32, f32) {
        let Some(objects) = self.objects.as_ref() else {
            return (0, 0.0, 0.0);
        };
        let mut count = 0usize;
        let mut total = 0.0f32;
        for idx in &self.selected_object_indices {
            if let Some(obj) = objects.get(*idx) {
                count += 1;
                total += obj.area_px;
            }
        }
        let mean = if count > 0 { total / count as f32 } else { 0.0 };
        (count, total, mean)
    }

    pub(super) fn table_indices(&mut self) -> &[usize] {
        if !self.table_cache_dirty {
            return &self.table_indices_cache;
        }
        let mut out = if let Some(filtered) = self.filtered_ordered_indices.as_ref() {
            filtered.as_ref().clone()
        } else if let Some(objects) = self.objects.as_ref() {
            (0..objects.len()).collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        out.sort_by(|a, b| {
            let sel_a = self.selected_object_indices.contains(a);
            let sel_b = self.selected_object_indices.contains(b);
            sel_b.cmp(&sel_a).then_with(|| a.cmp(b))
        });
        self.table_indices_cache = out;
        self.table_cache_dirty = false;
        &self.table_indices_cache
    }

    pub(super) fn export_selected_with_dialog(&mut self, default_dir: &Path) -> anyhow::Result<()> {
        let indices = self
            .selected_object_indices
            .iter()
            .copied()
            .collect::<Vec<_>>();
        self.export_indices_with_dialog(default_dir, "seg_objects_selected.geojson", &indices)
    }

    pub(super) fn export_filtered_with_dialog(&mut self, default_dir: &Path) -> anyhow::Result<()> {
        let indices = if let Some(filtered) = self.filtered_ordered_indices.as_ref() {
            filtered.as_ref().clone()
        } else if let Some(objects) = self.objects.as_ref() {
            (0..objects.len()).collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        self.export_indices_with_dialog(default_dir, "seg_objects_filtered.geojson", &indices)
    }

    pub(super) fn export_indices_with_dialog(
        &mut self,
        default_dir: &Path,
        default_name: &str,
        indices: &[usize],
    ) -> anyhow::Result<()> {
        if indices.is_empty() {
            anyhow::bail!("no objects to export");
        }
        let start_dir = self
            .loaded_geojson
            .as_ref()
            .and_then(|p| p.parent())
            .unwrap_or(default_dir);
        let Some(path) = FileDialog::new()
            .add_filter("GeoJSON", &["geojson", "json"])
            .set_title("Export Segmentation Objects")
            .set_directory(start_dir)
            .set_file_name(default_name)
            .save_file()
        else {
            return Ok(());
        };
        save_geojson_objects(
            path.as_path(),
            self.objects.as_deref().map_or(&[], |v| v),
            indices,
        )?;
        self.status = format!(
            "Exported {} object(s) to {}",
            indices.len(),
            path.to_string_lossy()
        );
        Ok(())
    }

    pub fn export_objects_geoparquet_with_dialog(&mut self) -> anyhow::Result<()> {
        self.open_object_export_dialog(ObjectExportFormat::GeoParquet)
    }

    pub fn export_objects_csv_with_dialog(&mut self) -> anyhow::Result<()> {
        let Some(objects) = self.objects.as_ref() else {
            anyhow::bail!("no objects loaded");
        };
        if objects.is_empty() {
            anyhow::bail!("no objects loaded");
        }
        self.open_object_export_dialog(ObjectExportFormat::Csv)
    }

    pub(super) fn default_object_export_dir(&self) -> &Path {
        self.loaded_geojson
            .as_deref()
            .and_then(Path::parent)
            .unwrap_or_else(|| Path::new("."))
    }

    pub(super) fn default_object_export_stem(&self) -> String {
        let fallback = "seg_objects".to_string();
        let Some(path) = self.loaded_geojson.as_ref() else {
            return fallback;
        };
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            return fallback;
        };
        let trimmed = name
            .strip_suffix(".geoparquet")
            .or_else(|| name.strip_suffix(".parquet"))
            .or_else(|| name.strip_suffix(".geojson"))
            .or_else(|| name.strip_suffix(".json"))
            .or_else(|| name.strip_suffix(".csv"))
            .unwrap_or(name);
        let trimmed = trimmed.trim_matches('.');
        if trimmed.is_empty() {
            fallback
        } else {
            trimmed.to_string()
        }
    }

    pub(super) fn export_objects_csv(
        snapshot: &ObjectExportSnapshot,
        path: &Path,
        selected_columns: &HashSet<String>,
    ) -> anyhow::Result<()> {
        // CSV export writes the normalized export table directly: one logical object row and one
        // scalar column per exported property. Geometry is omitted here in favor of tabular tools.
        let table =
            Self::build_object_export_table_from_snapshot(snapshot, selected_columns, false)?;
        let mut writer = csv::Writer::from_path(path)
            .with_context(|| format!("failed to create CSV: {}", path.to_string_lossy()))?;
        let headers = table
            .columns
            .iter()
            .map(|column| column.name.as_str())
            .collect::<Vec<_>>();
        writer.write_record(headers)?;
        for row_idx in 0..table.row_count {
            let row = table
                .columns
                .iter()
                .map(|column| {
                    export_scalar_to_csv(
                        column.values.get(row_idx).and_then(|value| value.as_ref()),
                    )
                })
                .collect::<Vec<_>>();
            writer.write_record(row)?;
        }
        writer.flush()?;
        Ok(())
    }

    pub(super) fn export_objects_geoparquet(
        snapshot: &ObjectExportSnapshot,
        path: &Path,
        selected_columns: &HashSet<String>,
    ) -> anyhow::Result<()> {
        // GeoParquet export shares the same normalized property table as CSV, but prefixes it with
        // a WKB geometry column so downstream spatial tools can preserve shape information.
        let table =
            Self::build_object_export_table_from_snapshot(snapshot, selected_columns, true)?;
        let mut fields = Vec::with_capacity(table.columns.len() + 1);
        let mut arrays = Vec::with_capacity(table.columns.len() + 1);

        fields.push(Field::new(
            "geometry",
            arrow_schema::DataType::Binary,
            false,
        ));
        let mut geometry_builder = BinaryBuilder::new();
        for geom in &table.geometry_wkb {
            geometry_builder.append_value(geom);
        }
        arrays.push(Arc::new(geometry_builder.finish()) as arrow_array::ArrayRef);

        for column in &table.columns {
            let (field, array) = export_column_to_arrow_array(column)?;
            fields.push(field);
            arrays.push(array);
        }

        let schema = Arc::new(Schema::new(fields));
        let batch = RecordBatch::try_new(Arc::clone(&schema), arrays)?;

        let geometry_types_json = table
            .geometry_types
            .iter()
            .map(|name| format!("\"{name}\""))
            .collect::<Vec<_>>()
            .join(",");
        let geo_metadata = format!(
            "{{\"version\":\"1.0.0\",\"primary_column\":\"geometry\",\"columns\":{{\"geometry\":{{\"encoding\":\"WKB\",\"geometry_types\":[{}],\"crs\":null}}}}}}",
            geometry_types_json
        );
        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .set_key_value_metadata(Some(vec![KeyValue {
                key: "geo".to_string(),
                value: Some(geo_metadata),
            }]))
            .build();

        let file = std::fs::File::create(path)
            .with_context(|| format!("failed to create parquet: {}", path.to_string_lossy()))?;
        let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
        writer.write(&batch)?;
        writer.close()?;
        Ok(())
    }

    pub(super) fn build_object_export_table_from_snapshot(
        snapshot: &ObjectExportSnapshot,
        selected_columns: &HashSet<String>,
        include_geometry: bool,
    ) -> anyhow::Result<ObjectExportTable> {
        let objects = snapshot.objects.as_ref();
        if objects.is_empty() {
            anyhow::bail!("no objects loaded");
        }
        let rows = snapshot
            .row_indices
            .iter()
            .filter_map(|index| objects.get(*index).map(|object| (*index, object)))
            .collect::<Vec<_>>();
        if rows.is_empty() {
            anyhow::bail!("the requested export scope contains no objects");
        }

        let property_keys = snapshot.property_keys.clone();
        let mut used_names = property_keys.iter().cloned().collect::<HashSet<_>>();

        let mut columns = Vec::new();
        for key in &property_keys {
            if selected_columns.contains(key) {
                columns.push(ExportColumn {
                    name: key.clone(),
                    values: rows
                        .iter()
                        .map(|(idx, obj)| {
                            if key == "id" {
                                Some(ExportScalar::String(obj.id.clone()))
                            } else {
                                export_scalar_from_property_store(
                                    &snapshot.property_store,
                                    key,
                                    *idx,
                                )
                                .or_else(|| {
                                    obj.inline_properties.get(key).map(export_scalar_from_json)
                                })
                            }
                        })
                        .collect(),
                });
            }
        }

        let mut geometry_types_cache: Option<Vec<String>> = None;

        let geometry_type_column_name = unique_export_name("_odon_geometry_type", &mut used_names);
        if selected_columns.contains(&geometry_type_column_name) {
            if geometry_types_cache.is_none() {
                geometry_types_cache = Some(
                    rows.iter()
                        .map(|(_, obj)| export_geometry_type_label(obj).to_string())
                        .collect::<Vec<_>>(),
                );
            }
            let geometry_types = geometry_types_cache
                .as_ref()
                .expect("geometry types cached");
            columns.push(ExportColumn {
                name: geometry_type_column_name,
                values: geometry_types
                    .iter()
                    .map(|name| Some(ExportScalar::String(name.clone())))
                    .collect(),
            });
        }

        let centroid_x_name = unique_export_name("_odon_centroid_x", &mut used_names);
        if selected_columns.contains(&centroid_x_name) {
            columns.push(ExportColumn {
                name: centroid_x_name,
                values: rows
                    .iter()
                    .map(|(_, obj)| Some(ExportScalar::Float64(obj.centroid_world.x as f64)))
                    .collect(),
            });
        }

        let centroid_y_name = unique_export_name("_odon_centroid_y", &mut used_names);
        if selected_columns.contains(&centroid_y_name) {
            columns.push(ExportColumn {
                name: centroid_y_name,
                values: rows
                    .iter()
                    .map(|(_, obj)| Some(ExportScalar::Float64(obj.centroid_world.y as f64)))
                    .collect(),
            });
        }

        if rows
            .iter()
            .any(|(_, obj)| obj.point_position_world.is_some())
        {
            let point_x_name = unique_export_name("_odon_point_x", &mut used_names);
            if selected_columns.contains(&point_x_name) {
                columns.push(ExportColumn {
                    name: point_x_name,
                    values: rows
                        .iter()
                        .map(|(_, obj)| {
                            obj.point_position_world
                                .map(|pos| ExportScalar::Float64(pos.x as f64))
                        })
                        .collect(),
                });
            }
            let point_y_name = unique_export_name("_odon_point_y", &mut used_names);
            if selected_columns.contains(&point_y_name) {
                columns.push(ExportColumn {
                    name: point_y_name,
                    values: rows
                        .iter()
                        .map(|(_, obj)| {
                            obj.point_position_world
                                .map(|pos| ExportScalar::Float64(pos.y as f64))
                        })
                        .collect(),
                });
            }
        }

        let area_name = unique_export_name("_odon_area_px", &mut used_names);
        if selected_columns.contains(&area_name) {
            columns.push(ExportColumn {
                name: area_name,
                values: rows
                    .iter()
                    .map(|(_, obj)| Some(ExportScalar::Float64(obj.area_px as f64)))
                    .collect(),
            });
        }

        let perimeter_name = unique_export_name("_odon_perimeter_px", &mut used_names);
        if selected_columns.contains(&perimeter_name) {
            columns.push(ExportColumn {
                name: perimeter_name,
                values: rows
                    .iter()
                    .map(|(_, obj)| Some(ExportScalar::Float64(obj.perimeter_px as f64)))
                    .collect(),
            });
        }

        let selected_name = unique_export_name("_odon_selected", &mut used_names);
        if selected_columns.contains(&selected_name) {
            columns.push(ExportColumn {
                name: selected_name,
                values: rows
                    .iter()
                    .map(|(idx, _)| {
                        Some(ExportScalar::Bool(
                            snapshot.selected_object_indices.contains(idx),
                        ))
                    })
                    .collect(),
            });
        }

        if !snapshot.analysis_property_thresholds.is_empty() {
            let live_name = unique_export_name(
                &live_threshold_call_export_column_name(
                    &snapshot.analysis_property_thresholds,
                    snapshot.analysis_live_threshold_channel_name.as_deref(),
                ),
                &mut used_names,
            );
            if selected_columns.contains(&live_name) {
                columns.push(ExportColumn {
                    name: live_name,
                    values: rows
                        .iter()
                        .map(|(idx, obj)| {
                            Some(ExportScalar::Bool(object_passes_threshold_rules(
                                &snapshot.property_store,
                                *idx,
                                obj,
                                &snapshot.analysis_property_thresholds,
                            )))
                        })
                        .collect(),
                });
            }
        }

        for element in &snapshot.analysis_threshold_elements {
            let column_name =
                unique_export_name(&threshold_call_export_column_name(element), &mut used_names);
            if selected_columns.contains(&column_name) {
                columns.push(ExportColumn {
                    name: column_name,
                    values: rows
                        .iter()
                        .map(|(idx, obj)| {
                            if threshold_call_marks_failed(element) {
                                Some(ExportScalar::String("FAIL".to_string()))
                            } else {
                                Some(ExportScalar::Bool(object_passes_threshold_rules(
                                    &snapshot.property_store,
                                    *idx,
                                    obj,
                                    &element.rules,
                                )))
                            }
                        })
                        .collect(),
                });
            }
        }

        for element in &snapshot.selection_elements {
            let column_name = unique_export_name(
                &format!("_odon_selection_{}", sanitize_export_key(&element.name)),
                &mut used_names,
            );
            if selected_columns.contains(&column_name) {
                let selected_ids = element.object_ids.iter().cloned().collect::<HashSet<_>>();
                columns.push(ExportColumn {
                    name: column_name,
                    values: rows
                        .iter()
                        .map(|(_, obj)| Some(ExportScalar::Bool(selected_ids.contains(&obj.id))))
                        .collect(),
                });
            }
        }

        let geometry_wkb = if include_geometry {
            rows.iter()
                .map(|(_, object)| encode_object_wkb(object))
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        let geometry_types = if include_geometry {
            if geometry_types_cache.is_none() {
                geometry_types_cache = Some(
                    rows.iter()
                        .map(|(_, obj)| export_geometry_type_label(obj).to_string())
                        .collect::<Vec<_>>(),
                );
            }
            geometry_types_cache
                .as_ref()
                .expect("geometry types cached")
                .iter()
                .cloned()
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect()
        } else {
            Vec::new()
        };

        Ok(ObjectExportTable {
            row_count: rows.len(),
            columns,
            geometry_wkb,
            geometry_types,
        })
    }

    pub(super) fn build_object_export_column_names(&self) -> anyhow::Result<Vec<String>> {
        let snapshot = self.object_export_snapshot()?;
        let objects = snapshot.objects.as_ref();
        let property_keys = snapshot.property_keys.clone();
        let mut used_names = property_keys.iter().cloned().collect::<HashSet<_>>();
        let mut columns = property_keys.clone();
        let mut push_name = |base_name: &str| {
            let name = unique_export_name(base_name, &mut used_names);
            columns.push(name);
        };

        push_name("_odon_geometry_type");
        push_name("_odon_centroid_x");
        push_name("_odon_centroid_y");
        if objects.iter().any(|obj| obj.point_position_world.is_some()) {
            push_name("_odon_point_x");
            push_name("_odon_point_y");
        }
        push_name("_odon_area_px");
        push_name("_odon_perimeter_px");
        push_name("_odon_selected");

        if !snapshot.analysis_property_thresholds.is_empty() {
            push_name(&live_threshold_call_export_column_name(
                &snapshot.analysis_property_thresholds,
                snapshot.analysis_live_threshold_channel_name.as_deref(),
            ));
        }

        for element in &snapshot.analysis_threshold_elements {
            push_name(&threshold_call_export_column_name(element));
        }
        for element in &snapshot.selection_elements {
            push_name(&format!(
                "_odon_selection_{}",
                sanitize_export_key(&element.name)
            ));
        }

        Ok(columns)
    }

    pub(super) fn open_object_export_dialog(
        &mut self,
        format: ObjectExportFormat,
    ) -> anyhow::Result<()> {
        if self.is_exporting() {
            anyhow::bail!("an export is already in progress");
        }
        let columns = self
            .build_object_export_column_names()?
            .into_iter()
            .map(|name| ObjectExportColumnSelection {
                name,
                selected: true,
            })
            .collect();
        self.object_export_dialog = Some(ObjectExportDialog { format, columns });
        Ok(())
    }

    pub fn ui_export_dialog(
        &mut self,
        ctx: &egui::Context,
        actor_managed: bool,
    ) -> Option<NativeObjectExportIntent> {
        let Some(mut dialog) = self.object_export_dialog.clone() else {
            return None;
        };
        let mut actor_intent = None;

        let mut keep_open = true;
        let mut close_requested = false;
        let mut export_requested = false;
        let title = match dialog.format {
            ObjectExportFormat::GeoParquet => "Export Enriched GeoParquet",
            ObjectExportFormat::Csv => "Export Enriched CSV",
        };

        egui::Window::new(title)
            .collapsible(false)
            .resizable(true)
            .default_width(560.0)
            .default_height(520.0)
            .open(&mut keep_open)
            .show(ctx, |ui| {
                ui.label("Choose which enriched columns to include in the export.");
                ui.small(
                    "GeoParquet always includes geometry. CSV exports only the selected tabular columns.",
                );
                ui.separator();
                ui.horizontal(|ui| {
                    if ui.button("Select all").clicked() {
                        for column in &mut dialog.columns {
                            column.selected = true;
                        }
                    }
                    if ui.button("Select none").clicked() {
                        for column in &mut dialog.columns {
                            column.selected = false;
                        }
                    }
                });
                let selected_count = dialog.columns.iter().filter(|column| column.selected).count();
                ui.small(format!(
                    "{selected_count} of {} columns selected",
                    dialog.columns.len()
                ));
                ui.separator();
                let list_height = ui.available_height().clamp(200.0, 480.0);
                ui.set_min_height(list_height);
                egui::ScrollArea::vertical()
                    .id_salt(("object_export_columns", title))
                    .auto_shrink([false, false])
                    .max_height(list_height)
                    .show(ui, |ui| {
                        for column in &mut dialog.columns {
                            ui.checkbox(&mut column.selected, &column.name);
                        }
                    });
                ui.separator();
                ui.horizontal(|ui| {
                    if ui.button("Cancel").clicked() {
                        close_requested = true;
                    }
                    let export_label = match dialog.format {
                        ObjectExportFormat::GeoParquet => "Choose file and export",
                        ObjectExportFormat::Csv => "Choose file and export",
                    };
                    if ui
                        .add_enabled(selected_count > 0, egui::Button::new(export_label))
                        .clicked()
                    {
                        export_requested = true;
                    }
                });
            });

        if export_requested {
            let selected_columns = dialog
                .columns
                .iter()
                .filter(|column| column.selected)
                .map(|column| column.name.clone())
                .collect::<HashSet<_>>();
            let default_name = match dialog.format {
                ObjectExportFormat::GeoParquet => {
                    format!("{}.enriched.geoparquet", self.default_object_export_stem())
                }
                ObjectExportFormat::Csv => {
                    format!("{}.enriched.csv", self.default_object_export_stem())
                }
            };
            let mut file_dialog = FileDialog::new().set_directory(self.default_object_export_dir());
            file_dialog = match dialog.format {
                ObjectExportFormat::GeoParquet => file_dialog
                    .add_filter("GeoParquet", &["geoparquet", "parquet"])
                    .set_title("Export Objects GeoParquet"),
                ObjectExportFormat::Csv => file_dialog
                    .add_filter("CSV", &["csv"])
                    .set_title("Export Objects CSV"),
            };
            let path = file_dialog.set_file_name(&default_name).save_file();
            if let Some(path) = path {
                if actor_managed {
                    let method = match dialog.format {
                        ObjectExportFormat::GeoParquet => "exports.objects.export_geoparquet",
                        ObjectExportFormat::Csv => "exports.objects.export_csv",
                    };
                    actor_intent = Some(NativeObjectExportIntent {
                        method,
                        params: serde_json::json!({
                            "path":path,
                            "scope":"all",
                            "columns":selected_columns.into_iter().collect::<Vec<_>>(),
                        }),
                    });
                    self.status = format!("Exporting objects to {}...", path.to_string_lossy());
                    close_requested = true;
                } else {
                    match self.start_object_export(dialog.format, path.clone(), selected_columns) {
                        Ok(()) => {
                            close_requested = true;
                        }
                        Err(err) => {
                            self.status = format!("Export failed: {err}");
                        }
                    }
                }
            }
        }

        if !keep_open || close_requested {
            self.object_export_dialog = None;
        } else {
            self.object_export_dialog = Some(dialog);
        }
        actor_intent
    }

    pub(crate) fn apply_control_actor_export_state(&mut self, state: &serde_json::Value) {
        self.actor_object_export_running = state
            .get("running")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        if let Some(status) = state.get("status").and_then(serde_json::Value::as_str) {
            self.status = status.to_string();
        }
    }

    pub(super) fn start_object_export(
        &mut self,
        format: ObjectExportFormat,
        path: PathBuf,
        selected_columns: HashSet<String>,
    ) -> anyhow::Result<()> {
        if self.is_exporting() {
            anyhow::bail!("an export is already in progress");
        }
        if selected_columns.is_empty() {
            anyhow::bail!("no export columns selected");
        }
        let snapshot = self.object_export_snapshot()?;
        let object_count = snapshot.row_indices.len();
        let request_id = self.object_export_request_id.wrapping_add(1).max(1);
        self.object_export_request_id = request_id;
        let (tx, rx) = crossbeam_channel::unbounded();
        self.object_export_rx = Some(rx);
        self.status = format!("Exporting {} object(s)...", object_count);
        std::thread::spawn(move || {
            let error = match format {
                ObjectExportFormat::GeoParquet => {
                    Self::export_objects_geoparquet(&snapshot, path.as_path(), &selected_columns)
                }
                ObjectExportFormat::Csv => {
                    Self::export_objects_csv(&snapshot, path.as_path(), &selected_columns)
                }
            }
            .err()
            .map(|err| err.to_string());
            let _ = tx.send(ObjectExportEvent::Finished {
                request_id,
                path,
                object_count,
                error,
            });
        });
        Ok(())
    }

    pub(super) fn object_export_snapshot(&self) -> anyhow::Result<ObjectExportSnapshot> {
        let Some(objects) = self.objects.as_ref() else {
            anyhow::bail!("no objects loaded");
        };
        if objects.is_empty() {
            anyhow::bail!("no objects loaded");
        }
        let mut property_keys = self.object_property_keys.clone();
        for key in self.property_store.loaded_keys() {
            match property_keys.binary_search_by(|existing| existing.as_str().cmp(&key)) {
                Ok(_) => {}
                Err(idx) => property_keys.insert(idx, key),
            }
        }
        if let Err(idx) = property_keys.binary_search_by(|existing| existing.as_str().cmp("id")) {
            property_keys.insert(idx, "id".to_string());
        }
        Ok(ObjectExportSnapshot {
            objects: Arc::clone(objects),
            row_indices: (0..objects.len()).collect(),
            property_keys,
            property_store: self.property_store.clone(),
            selected_object_indices: self.selected_object_indices.clone(),
            analysis_property_thresholds: self.analysis_property_thresholds.clone(),
            analysis_live_threshold_channel_name: self.analysis_live_threshold_channel_name.clone(),
            analysis_threshold_elements: self.analysis_threshold_elements.clone(),
            selection_elements: self.selection_elements.clone(),
        })
    }

    pub fn is_exporting(&self) -> bool {
        self.object_export_rx.is_some() || self.actor_object_export_running
    }

    pub(super) fn extend_object_property_keys<'a, I>(&mut self, keys: I)
    where
        I: IntoIterator<Item = &'a str>,
    {
        for key in keys {
            match self
                .object_property_keys
                .binary_search_by(|existing| existing.as_str().cmp(key))
            {
                Ok(_) => {}
                Err(idx) => self.object_property_keys.insert(idx, key.to_string()),
            }
        }
    }

    pub(in crate::objects) fn invalidate_table_cache(&mut self) {
        self.table_cache_dirty = true;
    }

    pub fn request_zoom_to_object(&mut self, idx: usize) {
        self.analysis_hist_focus_object_index = Some(idx);
        self.pending_zoom_object_index = Some(idx);
    }

    pub fn take_pending_zoom_object_index(&mut self) -> Option<usize> {
        self.pending_zoom_object_index.take()
    }
}
