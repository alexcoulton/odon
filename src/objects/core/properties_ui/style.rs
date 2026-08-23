use super::*;

impl ObjectsLayer {
    pub fn has_data(&self) -> bool {
        self.objects.as_ref().is_some_and(|v| !v.is_empty())
    }

    pub fn set_display_transform(&mut self, display_transform: SpatialDataTransform2) {
        self.display_transform = display_transform;
    }

    pub fn is_busy(&self) -> bool {
        self.is_loading()
            || self.is_property_loading()
            || self.is_analyzing()
            || self.is_bulk_measuring()
    }

    pub fn is_loading(&self) -> bool {
        self.load_rx.is_some()
    }

    pub fn is_property_loading(&self) -> bool {
        self.property_load_rx.is_some()
    }

    pub fn status(&self) -> &str {
        &self.status
    }

    pub fn selected_object_index(&self) -> Option<usize> {
        self.selected_object_index
    }

    pub fn available_property_columns(&self) -> &[String] {
        if !self.property_store.available_columns().is_empty() {
            self.property_store.available_columns()
        } else {
            self.color_property_keys.as_slice()
        }
    }

    pub(crate) fn apply_actor_style_projection_json(
        &mut self,
        params: &serde_json::Value,
    ) -> Result<bool, String> {
        let mut changed = false;
        macro_rules! set_bool {
            ($field:ident, $name:literal) => {
                if let Some(value) = params.get($name).and_then(serde_json::Value::as_bool) {
                    changed |= self.$field != value;
                    self.$field = value;
                }
            };
        }
        macro_rules! set_unit_f32 {
            ($field:ident, $name:literal) => {
                if let Some(value) = params.get($name).and_then(serde_json::Value::as_f64) {
                    if !(0.0..=1.0).contains(&value) {
                        return Err(format!("{} must be between 0 and 1", $name));
                    }
                    let value = value as f32;
                    changed |= self.$field != value;
                    self.$field = value;
                }
            };
        }
        set_bool!(visible, "visible");
        set_bool!(fill_cells, "fill_cells");
        set_bool!(show_selection_overlay, "show_selection_overlay");
        set_unit_f32!(opacity, "opacity");
        set_unit_f32!(fill_opacity, "fill_opacity");
        set_unit_f32!(selected_fill_opacity, "selected_fill_opacity");
        if let Some(value) = params
            .get("width_screen_px")
            .and_then(serde_json::Value::as_f64)
        {
            if !value.is_finite() || value <= 0.0 || value > 100.0 {
                return Err("width_screen_px must be greater than 0 and at most 100".to_string());
            }
            let value = value as f32;
            changed |= self.width_screen_px != value;
            self.width_screen_px = value;
        }
        if let Some(values) = params
            .get("color_rgb")
            .and_then(serde_json::Value::as_array)
        {
            if values.len() != 3
                || values
                    .iter()
                    .any(|value| value.as_u64().is_none_or(|v| v > 255))
            {
                return Err("color_rgb must contain three integers from 0 to 255".to_string());
            }
            let color = [
                values[0].as_u64().unwrap() as u8,
                values[1].as_u64().unwrap() as u8,
                values[2].as_u64().unwrap() as u8,
            ];
            changed |= self.color_rgb != color;
            self.color_rgb = color;
        }
        if params.get("color_property").is_some() {
            let property = params
                .get("color_property")
                .and_then(serde_json::Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty());
            if let Some(property) = property
                && !self.property_store.has_loaded(property)
                && !self.objects.as_ref().is_some_and(|objects| {
                    objects
                        .iter()
                        .any(|object| object.inline_properties.contains_key(property))
                })
            {
                return Err(format!("object property '{property}' is not loaded"));
            }
            let before = self.color_property_key.clone();
            self.set_color_by_property(property.map(str::to_string));
            changed |= before != self.color_property_key;
        }
        if changed {
            self.generation = self.generation.wrapping_add(1).max(1);
        }
        Ok(changed)
    }
}
