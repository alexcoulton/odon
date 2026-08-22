use std::collections::{HashMap, HashSet};

use serde_json::{Value, json};

use crate::control::{ControlError, ControlErrorKind};

#[derive(Debug, Clone)]
pub(crate) struct AnalysisModel {
    state: Value,
    generation: u64,
    operation_generations: HashMap<String, u64>,
    warmup_started: bool,
    warmup_running: bool,
    warmup_completed: usize,
    warmup_total: usize,
}

impl Default for AnalysisModel {
    fn default() -> Self {
        Self {
            state: default_analysis_state(),
            generation: 1,
            operation_generations: HashMap::new(),
            warmup_started: false,
            warmup_running: false,
            warmup_completed: 0,
            warmup_total: 0,
        }
    }
}

impl AnalysisModel {
    pub(crate) fn reset(&mut self) {
        *self = Self::default();
    }

    pub(crate) fn state(&self) -> &Value {
        &self.state
    }

    pub(crate) fn generation(&self) -> u64 {
        self.generation
    }

    pub(crate) fn replace(&mut self, params: &Value) -> Result<(), ControlError> {
        let state = params.get("state").unwrap_or(params);
        validate_analysis_state(state)?;
        self.state = normalized_analysis_state(state);
        self.generation = self.generation.wrapping_add(1).max(1);
        Ok(())
    }

    pub(crate) fn install_imported_state(&mut self, state: Value) -> Result<(), ControlError> {
        validate_analysis_state(&state)?;
        self.state = normalized_analysis_state(&state);
        self.generation = self.generation.wrapping_add(1).max(1);
        Ok(())
    }

    pub(crate) fn begin(&mut self, scope: &str) -> u64 {
        let generation = self
            .operation_generations
            .get(scope)
            .copied()
            .unwrap_or(0)
            .wrapping_add(1)
            .max(1);
        self.operation_generations
            .insert(scope.to_string(), generation);
        generation
    }

    pub(crate) fn is_current(&self, scope: &str, generation: u64) -> bool {
        self.operation_generations.get(scope).copied() == Some(generation)
    }

    pub(crate) fn begin_warmup(&mut self, total: usize) {
        self.warmup_started = true;
        self.warmup_running = true;
        self.warmup_completed = 0;
        self.warmup_total = total;
    }

    pub(crate) fn finish_warmup(&mut self, completed: usize) {
        self.warmup_running = false;
        self.warmup_completed = completed;
        self.warmup_total = self.warmup_total.max(completed);
    }

    pub(crate) fn fail_warmup(&mut self) {
        self.warmup_running = false;
    }

    pub(crate) fn warmup_snapshot(&self) -> Value {
        json!({
            "started":self.warmup_started,
            "running":self.warmup_running,
            "completed":self.warmup_completed,
            "total":self.warmup_total,
            "ready":self.warmup_started && !self.warmup_running,
        })
    }
}

pub(crate) fn default_analysis_state() -> Value {
    json!({
        "threshold_set_name":"",
        "threshold_elements":[],
        "threshold_selected_element":null,
        "follow_active_channel":true,
        "live_threshold_channel_name":null,
        "channel_mapping_overrides":{},
        "selection_elements":[],
        "selection_element_selected":null,
        "show_selection_overlay":true,
    })
}

fn normalized_analysis_state(state: &Value) -> Value {
    let defaults = default_analysis_state();
    let mut normalized = defaults.as_object().cloned().unwrap_or_default();
    if let Some(values) = state.as_object() {
        normalized.extend(values.clone());
    }
    Value::Object(normalized)
}

fn validate_analysis_state(state: &Value) -> Result<(), ControlError> {
    let object = state
        .as_object()
        .ok_or_else(|| invalid("analysis state must be an object"))?;
    let calls = object
        .get("threshold_elements")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut names = HashSet::new();
    for call in calls {
        let name = call
            .get("name")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|name| !name.is_empty())
            .ok_or_else(|| invalid("analysis call names must not be empty"))?;
        if !names.insert(name.to_string()) {
            return Err(invalid(format!("duplicate analysis call name '{name}'")));
        }
        for rule in call
            .get("rules")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let property = rule
                .get("column_key")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty());
            let finite = rule
                .get("value")
                .and_then(Value::as_f64)
                .is_some_and(f64::is_finite);
            if property.is_none() || !finite {
                return Err(invalid(
                    "analysis rules require a property and finite value",
                ));
            }
        }
    }
    names.clear();
    for selection in object
        .get("selection_elements")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let name = selection
            .get("name")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|name| !name.is_empty())
            .ok_or_else(|| invalid("named selections require unique non-empty names"))?;
        if !names.insert(name.to_string()) {
            return Err(invalid("named selections require unique non-empty names"));
        }
    }
    Ok(())
}

fn invalid(message: impl Into<String>) -> ControlError {
    ControlError::new(ControlErrorKind::InvalidParams, message)
}
