//! Typed object-property column storage, lookup, and filtering.

use super::*;

#[derive(Debug, Clone, Default)]
pub(in crate::objects) struct ObjectPropertyStore {
    pub(in crate::objects) available_columns: Vec<String>,
    pub(in crate::objects) loaded_columns: HashMap<String, ObjectPropertyColumn>,
}

impl ObjectPropertyStore {
    pub(in crate::objects) fn from_available_columns(columns: Vec<String>) -> Self {
        Self {
            available_columns: columns,
            loaded_columns: HashMap::new(),
        }
    }

    pub(in crate::objects) fn available_columns(&self) -> &[String] {
        self.available_columns.as_slice()
    }

    pub(in crate::objects) fn has_loaded(&self, key: &str) -> bool {
        self.loaded_columns.contains_key(key)
    }

    pub(in crate::objects) fn loaded_keys(&self) -> Vec<String> {
        let mut keys = self.loaded_columns.keys().cloned().collect::<Vec<_>>();
        keys.sort();
        keys
    }

    pub(in crate::objects) fn numeric_keys(&self) -> Vec<String> {
        let mut keys = self
            .loaded_columns
            .iter()
            .filter(|(_, column)| column.is_numeric())
            .map(|(key, _)| key.clone())
            .collect::<Vec<_>>();
        keys.sort();
        keys
    }

    pub(in crate::objects) fn numeric_pairs(&self, key: &str) -> Option<Vec<(usize, f32)>> {
        self.loaded_columns
            .get(key)
            .and_then(ObjectPropertyColumn::numeric_pairs)
    }

    pub(in crate::objects) fn numeric_at(&self, key: &str, object_index: usize) -> Option<f64> {
        self.loaded_columns
            .get(key)
            .and_then(|column| column.numeric_at(object_index))
    }

    pub(in crate::objects) fn label_at(&self, key: &str, object_index: usize) -> Option<String> {
        self.loaded_columns
            .get(key)
            .and_then(|column| column.label_at(object_index))
    }

    pub(in crate::objects) fn insert_column(&mut self, key: String, column: ObjectPropertyColumn) {
        self.loaded_columns.insert(key, column);
    }

    pub(in crate::objects) fn loaded_column_is_categorical(
        &self,
        key: &str,
        max_distinct: usize,
    ) -> bool {
        self.loaded_columns
            .get(key)
            .is_some_and(|column| column.is_categorical(max_distinct))
    }

    pub(in crate::objects) fn filter_value_options(
        &self,
        key: &str,
        max_options: usize,
    ) -> Option<Vec<String>> {
        self.loaded_columns
            .get(key)
            .and_then(|column| column.filter_value_options(max_options))
    }
}

#[derive(Debug, Clone)]
pub(in crate::objects) enum ObjectPropertyColumn {
    Bool(Arc<Vec<Option<bool>>>),
    I64(Arc<Vec<Option<i64>>>),
    F64(Arc<Vec<Option<f64>>>),
    Dictionary {
        dictionary: Arc<Vec<String>>,
        values: Arc<Vec<Option<u32>>>,
    },
    Json(Arc<Vec<Option<serde_json::Value>>>),
}

#[derive(Debug, Clone)]
pub(in crate::objects) enum ObjectPropertyContainsMatcher {
    Bool {
        true_matches: bool,
        false_matches: bool,
    },
    I64 {
        needle: String,
    },
    F64 {
        needle: String,
    },
    Dictionary {
        matching_codes: Vec<bool>,
    },
    Json {
        needle: String,
    },
}

impl ObjectPropertyColumn {
    pub(in crate::objects) fn value_json_at(&self, object_index: usize) -> serde_json::Value {
        match self {
            Self::Bool(values) => values
                .get(object_index)
                .and_then(|value| *value)
                .map(serde_json::Value::Bool)
                .unwrap_or(serde_json::Value::Null),
            Self::I64(values) => values
                .get(object_index)
                .and_then(|value| *value)
                .map(serde_json::Value::from)
                .unwrap_or(serde_json::Value::Null),
            Self::F64(values) => values
                .get(object_index)
                .and_then(|value| *value)
                .and_then(serde_json::Number::from_f64)
                .map(serde_json::Value::Number)
                .unwrap_or(serde_json::Value::Null),
            Self::Dictionary { dictionary, values } => values
                .get(object_index)
                .and_then(|code| *code)
                .and_then(|code| dictionary.get(code as usize))
                .cloned()
                .map(serde_json::Value::String)
                .unwrap_or(serde_json::Value::Null),
            Self::Json(values) => values
                .get(object_index)
                .and_then(|value| value.clone())
                .unwrap_or(serde_json::Value::Null),
        }
    }

    pub(in crate::objects) fn from_values_by_row(
        objects: &[GeoJsonObjectFeature],
        values_by_row: &HashMap<usize, serde_json::Value>,
    ) -> Self {
        let values = objects
            .iter()
            .map(|obj| {
                obj.source_row_index
                    .and_then(|row_index| values_by_row.get(&row_index).cloned())
            })
            .collect::<Vec<_>>();
        Self::from_json_values(values)
    }

    pub(in crate::objects) fn from_json_values(values: Vec<Option<serde_json::Value>>) -> Self {
        let non_null = values.iter().flatten().collect::<Vec<_>>();
        if non_null.is_empty() {
            return Self::Json(Arc::new(values));
        }

        if non_null
            .iter()
            .all(|value| matches!(value, serde_json::Value::Bool(_)))
        {
            return Self::Bool(Arc::new(
                values
                    .into_iter()
                    .map(|value| match value {
                        Some(serde_json::Value::Bool(value)) => Some(value),
                        _ => None,
                    })
                    .collect(),
            ));
        }

        if non_null
            .iter()
            .all(|value| matches!(value, serde_json::Value::Number(_)))
        {
            if non_null
                .iter()
                .all(|value| json_number_to_i64(value).is_some())
            {
                return Self::I64(Arc::new(
                    values
                        .into_iter()
                        .map(|value| value.and_then(|value| json_number_to_i64(&value)))
                        .collect(),
                ));
            }
            return Self::F64(Arc::new(
                values
                    .into_iter()
                    .map(|value| value.and_then(|value| value.as_f64()))
                    .collect(),
            ));
        }

        if non_null
            .iter()
            .all(|value| matches!(value, serde_json::Value::String(_)))
        {
            let mut dictionary = Vec::<String>::new();
            let mut lookup = HashMap::<String, u32>::new();
            let mut encoded = Vec::with_capacity(values.len());
            for value in values {
                let Some(serde_json::Value::String(value)) = value else {
                    encoded.push(None);
                    continue;
                };
                let code = if let Some(code) = lookup.get(&value) {
                    *code
                } else {
                    let code = dictionary.len() as u32;
                    dictionary.push(value.clone());
                    lookup.insert(value, code);
                    code
                };
                encoded.push(Some(code));
            }
            return Self::Dictionary {
                dictionary: Arc::new(dictionary),
                values: Arc::new(encoded),
            };
        }

        Self::Json(Arc::new(values))
    }

    pub(in crate::objects) fn label_at(&self, object_index: usize) -> Option<String> {
        match self {
            Self::Bool(values) => values
                .get(object_index)
                .and_then(|value| value.map(|value| value.to_string())),
            Self::I64(values) => values
                .get(object_index)
                .and_then(|value| value.map(|value| value.to_string())),
            Self::F64(values) => values
                .get(object_index)
                .and_then(|value| value.map(|value| value.to_string())),
            Self::Dictionary { dictionary, values } => values
                .get(object_index)
                .and_then(|code| code.and_then(|code| dictionary.get(code as usize).cloned())),
            Self::Json(values) => values
                .get(object_index)
                .and_then(|value| value.as_ref())
                .and_then(property_scalar_value),
        }
    }

    pub(in crate::objects) fn display_at(&self, object_index: usize) -> Option<String> {
        match self {
            Self::Json(values) => values
                .get(object_index)
                .and_then(|value| value.as_ref())
                .map(column_value_to_display_text)
                .filter(|value| !value.trim().is_empty()),
            _ => self
                .label_at(object_index)
                .filter(|value| !value.trim().is_empty()),
        }
    }

    pub(in crate::objects) fn contains_matcher(
        &self,
        query: &str,
    ) -> ObjectPropertyContainsMatcher {
        let needle = query.trim().to_ascii_lowercase();
        match self {
            Self::Bool(_) => ObjectPropertyContainsMatcher::Bool {
                true_matches: "true".contains(&needle),
                false_matches: "false".contains(&needle),
            },
            Self::I64(_) => ObjectPropertyContainsMatcher::I64 { needle },
            Self::F64(_) => ObjectPropertyContainsMatcher::F64 { needle },
            Self::Dictionary { dictionary, .. } => {
                let matching_codes = dictionary
                    .iter()
                    .map(|label| label.to_ascii_lowercase().contains(&needle))
                    .collect::<Vec<_>>();
                ObjectPropertyContainsMatcher::Dictionary { matching_codes }
            }
            Self::Json(_) => ObjectPropertyContainsMatcher::Json { needle },
        }
    }

    pub(in crate::objects) fn matches_contains(
        &self,
        object_index: usize,
        matcher: &ObjectPropertyContainsMatcher,
    ) -> bool {
        match (self, matcher) {
            (
                Self::Bool(values),
                ObjectPropertyContainsMatcher::Bool {
                    true_matches,
                    false_matches,
                },
            ) => values
                .get(object_index)
                .and_then(|value| *value)
                .is_some_and(|value| if value { *true_matches } else { *false_matches }),
            (Self::I64(values), ObjectPropertyContainsMatcher::I64 { needle }) => values
                .get(object_index)
                .and_then(|value| *value)
                .is_some_and(|value| value.to_string().contains(needle)),
            (Self::F64(values), ObjectPropertyContainsMatcher::F64 { needle }) => values
                .get(object_index)
                .and_then(|value| *value)
                .is_some_and(|value| value.to_string().contains(needle)),
            (
                Self::Dictionary { values, .. },
                ObjectPropertyContainsMatcher::Dictionary { matching_codes },
            ) => values
                .get(object_index)
                .and_then(|code| *code)
                .is_some_and(|code| matching_codes.get(code as usize).copied().unwrap_or(false)),
            (Self::Json(values), ObjectPropertyContainsMatcher::Json { needle }) => values
                .get(object_index)
                .and_then(|value| value.as_ref())
                .is_some_and(|value| {
                    column_value_to_display_text(value)
                        .to_ascii_lowercase()
                        .contains(needle)
                }),
            _ => false,
        }
    }

    pub(in crate::objects) fn filter_value_options(
        &self,
        max_options: usize,
    ) -> Option<Vec<String>> {
        match self {
            Self::Bool(_) => Some(vec!["true".to_string(), "false".to_string()]),
            Self::Dictionary { dictionary, .. } => {
                if dictionary.is_empty() || dictionary.len() > max_options {
                    return None;
                }
                let mut values = dictionary.as_ref().clone();
                values.sort_by_key(|value| value.to_ascii_lowercase());
                Some(values)
            }
            Self::I64(_) | Self::F64(_) | Self::Json(_) => None,
        }
    }

    pub(in crate::objects) fn is_categorical(&self, max_distinct: usize) -> bool {
        let mut distinct = HashSet::new();
        for idx in 0..self.len() {
            let Some(label) = self.label_at(idx) else {
                continue;
            };
            distinct.insert(label);
            if distinct.len() > max_distinct {
                return false;
            }
        }
        !distinct.is_empty()
    }

    pub(in crate::objects) fn is_numeric(&self) -> bool {
        matches!(self, Self::I64(_) | Self::F64(_))
    }

    pub(in crate::objects) fn numeric_pairs(&self) -> Option<Vec<(usize, f32)>> {
        match self {
            Self::I64(values) => Some(
                values
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, value)| value.map(|value| (idx, value as f32)))
                    .filter(|(_, value)| value.is_finite())
                    .collect(),
            ),
            Self::F64(values) => Some(
                values
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, value)| value.map(|value| (idx, value as f32)))
                    .filter(|(_, value)| value.is_finite())
                    .collect(),
            ),
            _ => None,
        }
    }

    pub(in crate::objects) fn numeric_at(&self, object_index: usize) -> Option<f64> {
        match self {
            Self::I64(values) => values
                .get(object_index)
                .and_then(|value| value.map(|value| value as f64)),
            Self::F64(values) => values.get(object_index).and_then(|value| *value),
            _ => None,
        }
        .filter(|value| value.is_finite())
    }

    pub(in crate::objects) fn len(&self) -> usize {
        match self {
            Self::Bool(values) => values.len(),
            Self::I64(values) => values.len(),
            Self::F64(values) => values.len(),
            Self::Dictionary { values, .. } => values.len(),
            Self::Json(values) => values.len(),
        }
    }
}

pub(in crate::objects) fn json_number_to_i64(value: &serde_json::Value) -> Option<i64> {
    let number = value.as_number()?;
    number
        .as_i64()
        .or_else(|| number.as_u64().and_then(|value| i64::try_from(value).ok()))
}

pub(in crate::objects) fn column_value_to_display_text(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Null => String::new(),
        serde_json::Value::Bool(v) => v.to_string(),
        serde_json::Value::Number(v) => v.to_string(),
        serde_json::Value::String(v) => v.clone(),
        other => other.to_string(),
    }
}
