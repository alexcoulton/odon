//! Automatic channel contrast pooled across every ROI in a mosaic.

use super::*;
use crate::settings::AutoContrastSettings;

/// Bounds the pixels read per channel, so opening a large mosaic stays quick however many ROIs it
/// has. The per-ROI share is capped because small mosaics gain little from finer levels.
const MAX_SCAN_PIXELS_PER_CHANNEL: u64 = 16_000_000;
const MAX_SCAN_PIXELS_PER_ITEM: u64 = 2_000_000;

#[derive(Debug, Clone)]
pub(crate) struct MosaicAutoContrastChannelSpec {
    pub index: usize,
    pub name: String,
    pub baseline_window: Option<(f32, f32)>,
}

#[derive(Clone)]
pub(crate) struct MosaicAutoContrastItemSpec {
    pub item_id: usize,
    pub document: ControlOpenedDocument,
    /// Mapping from global mosaic channel index to this document's channel index.
    pub channel_map: Vec<Option<u64>>,
    pub level: usize,
}

#[derive(Clone)]
pub(crate) struct MosaicAutoContrastSpec {
    pub resource_generation: u64,
    pub operation_generation: u64,
    pub settings: AutoContrastSettings,
    pub channels: Vec<MosaicAutoContrastChannelSpec>,
    pub items: Vec<MosaicAutoContrastItemSpec>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MosaicAutoContrastChannelResult {
    pub index: usize,
    pub min: u16,
    pub max: u16,
    pub sample_count: u64,
}

#[derive(Debug, Default)]
pub(crate) struct MosaicAutoContrastResult {
    pub channels: Vec<MosaicAutoContrastChannelResult>,
    pub failed_items: Vec<(usize, String)>,
}

#[derive(Debug, Clone)]
pub(super) struct MosaicAutoContrastState {
    operation_generation: u64,
    pending: bool,
    on_open_queued: bool,
    snapshot: Value,
}

impl Default for MosaicAutoContrastState {
    fn default() -> Self {
        Self {
            operation_generation: 0,
            pending: false,
            on_open_queued: false,
            snapshot: json!({"pending":false}),
        }
    }
}

impl MosaicModel {
    /// A newly opened mosaic has no running contrast job. Keeps counting operations so a worker
    /// finishing for the previous resource can never match.
    pub(crate) fn reset_auto_contrast(&mut self, on_open: bool) {
        self.auto_contrast.pending = false;
        self.auto_contrast.on_open_queued = on_open;
        self.auto_contrast.snapshot = json!({"pending":false});
    }

    pub(crate) fn auto_contrast_snapshot(&self) -> Value {
        self.auto_contrast.snapshot.clone()
    }

    pub(crate) fn auto_contrast_on_open_queued(&self) -> bool {
        self.auto_contrast.on_open_queued && !self.auto_contrast.pending
    }

    /// Whether opening has anything to do, given that saved and hand-set limits are kept.
    pub(crate) fn auto_contrast_on_open_has_channels(&self) -> bool {
        self.channels.iter().any(|channel| !channel.contrast_manual)
    }

    /// Skips channels already set by hand or saved in the project, unless `overwrite_manual`.
    pub(crate) fn prepare_auto_contrast(
        &mut self,
        settings: AutoContrastSettings,
        params: &Value,
    ) -> Result<MosaicAutoContrastSpec, ControlError> {
        self.require_resource()?;
        if params.get("viewport_id").is_some() || params.get("id").is_some() {
            return Err(invalid(
                "mosaic contrast applies to the whole mosaic, so viewport_id is not supported",
            ));
        }
        let overwrite_manual = match params.get("overwrite_manual") {
            Some(value) => value
                .as_bool()
                .ok_or_else(|| invalid("overwrite_manual must be a boolean"))?,
            None => true,
        };
        let requested = match params.get("channels") {
            Some(value) => value
                .as_array()
                .ok_or_else(|| invalid("channels must be an array"))?
                .iter()
                .map(|selector| self.channel_index(selector))
                .collect::<Result<Vec<_>, _>>()?,
            None => (0..self.channels.len()).collect(),
        };
        let mut channels = Vec::with_capacity(requested.len());
        for index in requested {
            let channel = &self.channels[index];
            if channel.contrast_manual && !overwrite_manual {
                continue;
            }
            channels.push(MosaicAutoContrastChannelSpec {
                index,
                name: channel.name.clone(),
                baseline_window: channel.window,
            });
        }
        if channels.is_empty() {
            return Err(invalid("no channels are eligible for automatic contrast"));
        }
        if overwrite_manual {
            for channel in &channels {
                self.channels[channel.index].contrast_manual = false;
            }
        }
        let resource = self
            .resource
            .as_ref()
            .ok_or_else(|| invalid("no mosaic is open"))?;
        let per_item_pixels = (MAX_SCAN_PIXELS_PER_CHANNEL / (resource.items.len().max(1) as u64))
            .min(MAX_SCAN_PIXELS_PER_ITEM);
        let items = resource
            .items
            .iter()
            .filter_map(|item| {
                let level = choose_scan_level(&item.document, per_item_pixels)?;
                Some(MosaicAutoContrastItemSpec {
                    item_id: item.id,
                    document: item.document.clone(),
                    channel_map: self
                        .channels
                        .iter()
                        .map(|global| {
                            item.document
                                .descriptor
                                .channels
                                .iter()
                                .find(|local| local.name == global.name)
                                .map(|local| local.index as u64)
                        })
                        .collect(),
                    level,
                })
            })
            .collect::<Vec<_>>();
        if items.is_empty() {
            return Err(invalid(
                "no mosaic images are readable for automatic contrast",
            ));
        }
        Ok(MosaicAutoContrastSpec {
            resource_generation: resource.generation,
            operation_generation: self
                .auto_contrast
                .operation_generation
                .wrapping_add(1)
                .max(1),
            settings,
            channels,
            items,
        })
    }

    pub(crate) fn mark_auto_contrast_started(&mut self, spec: &MosaicAutoContrastSpec) -> bool {
        if !self.auto_contrast_spec_targets_resource(spec)
            || spec.operation_generation <= self.auto_contrast.operation_generation
        {
            return false;
        }
        self.auto_contrast.operation_generation = spec.operation_generation;
        self.auto_contrast.pending = true;
        self.auto_contrast.on_open_queued = false;
        self.auto_contrast.snapshot = json!({
            "pending":true,
            "operation_generation":spec.operation_generation,
            "channels":spec.channels.len(),
            "items":spec.items.len(),
            "method":spec.settings.method,
        });
        true
    }

    /// Stops an on-open job from retrying when it could not be prepared.
    pub(crate) fn cancel_auto_contrast_on_open(&mut self, message: impl Into<String>) {
        self.auto_contrast.on_open_queued = false;
        self.auto_contrast.snapshot = json!({"pending":false,"error":message.into()});
    }

    pub(crate) fn install_auto_contrast(
        &mut self,
        spec: &MosaicAutoContrastSpec,
        result: &MosaicAutoContrastResult,
    ) -> Option<Value> {
        if !self.auto_contrast_is_current(spec) {
            return None;
        }
        let abs_max = self.abs_max().max(1.0);
        let mut applied = Vec::new();
        let mut skipped = Vec::new();
        for channel_spec in &spec.channels {
            let index = channel_spec.index;
            let name = channel_spec.name.as_str();
            let skip = |reason: &str| json!({"index":index,"name":name,"reason":reason});
            let Some(channel) = self.channels.get_mut(index) else {
                skipped.push(skip("channel_removed"));
                continue;
            };
            if channel.window != channel_spec.baseline_window || channel.contrast_manual {
                skipped.push(skip("contrast_changed_after_request"));
                continue;
            }
            let Some(computed) = result.channels.iter().find(|item| item.index == index) else {
                skipped.push(skip("no_samples"));
                continue;
            };
            let min = (computed.min as f32).clamp(0.0, abs_max);
            let max = (computed.max as f32).clamp(0.0, abs_max);
            // A blank channel has no meaningful range, and 0..1 would saturate any faint signal.
            if max <= min {
                skipped.push(skip("no_signal"));
                continue;
            }
            channel.window = Some((min, max));
            applied.push(json!({
                "index":index,
                "name":name,
                "min":min,
                "max":max,
                "n":computed.sample_count,
            }));
        }
        let response = json!({
            "mode":"mosaic",
            "applied":applied,
            "skipped":skipped,
            "failed_items":result
                .failed_items
                .iter()
                .map(|(item_id, error)| json!({"item_id":item_id,"error":error}))
                .collect::<Vec<_>>(),
            "method":spec.settings.method,
            "completed":true,
        });
        self.auto_contrast.pending = false;
        self.auto_contrast.snapshot = response.clone();
        self.auto_contrast.snapshot["pending"] = json!(false);
        self.auto_contrast.snapshot["operation_generation"] = json!(spec.operation_generation);
        Some(response)
    }

    pub(crate) fn fail_auto_contrast(
        &mut self,
        spec: &MosaicAutoContrastSpec,
        message: impl Into<String>,
    ) -> bool {
        if !self.auto_contrast_is_current(spec) {
            return false;
        }
        self.auto_contrast.pending = false;
        self.auto_contrast.on_open_queued = false;
        self.auto_contrast.snapshot = json!({
            "pending":false,
            "operation_generation":spec.operation_generation,
            "error":message.into(),
        });
        true
    }

    fn auto_contrast_spec_targets_resource(&self, spec: &MosaicAutoContrastSpec) -> bool {
        self.resource
            .as_ref()
            .is_some_and(|resource| resource.generation == spec.resource_generation)
    }

    fn auto_contrast_is_current(&self, spec: &MosaicAutoContrastSpec) -> bool {
        self.auto_contrast_spec_targets_resource(spec)
            && self.auto_contrast.pending
            && spec.operation_generation == self.auto_contrast.operation_generation
    }
}

/// The finest level within budget, or the coarsest available when every level exceeds it.
fn choose_scan_level(document: &ControlOpenedDocument, max_pixels: u64) -> Option<usize> {
    let descriptor = &document.descriptor;
    let pixels = |level: &crate::data::ome::LevelInfo| {
        level
            .shape
            .get(descriptor.dims.y)
            .copied()
            .unwrap_or(0)
            .saturating_mul(level.shape.get(descriptor.dims.x).copied().unwrap_or(0))
    };
    descriptor
        .levels
        .iter()
        .enumerate()
        .find(|(_, level)| pixels(level) > 0 && pixels(level) <= max_pixels)
        .map(|(index, _)| index)
        .or_else(|| descriptor.levels.len().checked_sub(1))
}
