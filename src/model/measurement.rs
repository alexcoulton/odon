use serde_json::{Value, json};

use crate::control::{ControlError, ControlErrorKind};
use crate::data::document::DocumentDescriptor;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MeasurementMetric {
    Mean,
    Median,
}

impl MeasurementMetric {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Mean => "mean",
            Self::Median => "median",
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct MeasurementModel {
    pub(crate) metric: MeasurementMetric,
    pub(crate) level: usize,
    pub(crate) concurrency: usize,
    pub(crate) filtered_only: bool,
    pub(crate) prefix: String,
    pub(crate) running: bool,
    pub(crate) status: String,
    pub(crate) completed: usize,
    pub(crate) total: usize,
    pub(crate) generation: u64,
}

impl Default for MeasurementModel {
    fn default() -> Self {
        Self {
            metric: MeasurementMetric::Mean,
            level: 0,
            concurrency: 4,
            filtered_only: false,
            prefix: "mean_intensity_".to_string(),
            running: false,
            status: String::new(),
            completed: 0,
            total: 0,
            generation: 1,
        }
    }
}

impl MeasurementModel {
    pub(crate) fn reset(&mut self) {
        *self = Self::default();
    }

    pub(crate) fn configure(&mut self, params: &Value, levels: usize) -> Result<(), ControlError> {
        if self.running {
            return Err(invalid(
                "cannot reconfigure measurements while a run is active",
            ));
        }
        if let Some(metric) = params.get("metric").and_then(Value::as_str) {
            self.metric = match metric {
                "mean" => MeasurementMetric::Mean,
                "median" | "exact_median" => MeasurementMetric::Median,
                _ => return Err(invalid("metric must be 'mean' or 'median'")),
            };
            if params.get("prefix").is_none() {
                self.prefix = match self.metric {
                    MeasurementMetric::Mean => "mean_intensity_",
                    MeasurementMetric::Median => "median_intensity_",
                }
                .to_string();
            }
        }
        if let Some(level) = params.get("level") {
            self.level = level
                .as_u64()
                .and_then(|v| usize::try_from(v).ok())
                .filter(|v| *v < levels)
                .ok_or_else(|| invalid("measurement level is out of range"))?;
        }
        if let Some(value) = params.get("concurrency") {
            self.concurrency = value
                .as_u64()
                .and_then(|v| usize::try_from(v).ok())
                .filter(|v| (1..=64).contains(v))
                .ok_or_else(|| invalid("concurrency must be an integer from 1 to 64"))?;
        }
        if let Some(value) = params.get("filtered_only") {
            self.filtered_only = value
                .as_bool()
                .ok_or_else(|| invalid("filtered_only must be a boolean"))?;
        }
        if let Some(value) = params.get("prefix") {
            self.prefix = value
                .as_str()
                .map(str::trim)
                .filter(|v| !v.is_empty())
                .map(str::to_string)
                .ok_or_else(|| invalid("prefix must be a non-empty string"))?;
        }
        self.generation = self.generation.wrapping_add(1).max(1);
        Ok(())
    }

    pub(crate) fn begin(&mut self, total: usize) -> u64 {
        self.generation = self.generation.wrapping_add(1).max(1);
        self.running = true;
        self.completed = 0;
        self.total = total;
        self.status = format!("Measuring {} object(s)", total);
        self.generation
    }
    pub(crate) fn cancel(&mut self) -> bool {
        let running = self.running;
        self.generation = self.generation.wrapping_add(1).max(1);
        self.running = false;
        self.status = if running {
            "Measurement cancelled".to_string()
        } else {
            self.status.clone()
        };
        running
    }
    pub(crate) fn finish(&mut self, generation: u64, measured: usize) -> bool {
        if !self.running || self.generation != generation {
            return false;
        }
        self.running = false;
        self.completed = self.total;
        self.status = format!("Measured {measured} object(s)");
        self.generation = self.generation.wrapping_add(1).max(1);
        true
    }
    pub(crate) fn fail(&mut self, generation: u64, message: String) -> bool {
        if !self.running || self.generation != generation {
            return false;
        }
        self.running = false;
        self.status = message;
        self.generation = self.generation.wrapping_add(1).max(1);
        true
    }

    pub(crate) fn generation(&self) -> u64 {
        self.generation
    }
    pub(crate) fn snapshot(
        &self,
        descriptor: &DocumentDescriptor,
        target_count: usize,
        properties: Vec<String>,
    ) -> Value {
        json!({"running":self.running,"status":self.status,"metric":self.metric.as_str(),"level":self.level,
            "concurrency":self.concurrency,"filtered_only":self.filtered_only,"prefix":self.prefix,"target_count":target_count,
            "progress":{"completed":self.completed,"total":self.total},
            "levels":descriptor.levels.iter().map(|level| { let width=level.shape.get(descriptor.dims.x).copied(); let height=level.shape.get(descriptor.dims.y).copied();
                json!({"index":level.index,"downsample":level.downsample,"width":width,"height":height,
                    "label_raster_bytes":width.zip(height).map(|(w,h)| w.saturating_mul(h).saturating_mul(4))}) }).collect::<Vec<_>>(),
            "generated_properties":properties})
    }
}

fn invalid(message: impl Into<String>) -> ControlError {
    ControlError::new(ControlErrorKind::InvalidParams, message)
}
