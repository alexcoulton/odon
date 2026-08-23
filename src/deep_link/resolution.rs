use std::path::PathBuf;

use serde::Serialize;

use crate::data::project_config::ProjectRoi;

use super::{DeepLinkChannelOrder, DeepLinkRequest};

#[derive(Debug, Clone, Serialize)]
#[allow(dead_code)]
pub struct DeepLinkResolution {
    pub project_source: String,
    pub project_path: Option<PathBuf>,
    pub roi: ProjectRoi,
}

pub fn apply_example_defaults(request: &mut DeepLinkRequest, example: &str) {
    let normalized = normalize_example_name(example);
    if !matches!(normalized.as_str(), "synthetic5ch" | "synthetic" | "demo") {
        return;
    }
    if request.roi.is_none() {
        request.roi = Some("synthetic_5ch.ome.zarr".to_string());
    }
    if request.channel.is_none() {
        request.channel = Some("DAPI".to_string());
    }
    if request.visible_channels.is_empty() {
        request.visible_channels = vec!["DAPI".to_string(), "CD3".to_string(), "PanCK".to_string()];
    }
    if request.visible_channel_group.is_none() {
        request.visible_channel_group = Some("Synthetic example".to_string());
    }
    if request.channel_order.is_none() {
        request.channel_order = Some(DeepLinkChannelOrder::Listed);
    }
}

pub fn resolve_example_project_path(example: &str) -> Option<PathBuf> {
    let normalized = normalize_example_name(example);
    let project_name = match normalized.as_str() {
        "synthetic5ch" | "synthetic" | "demo" => "synthetic_5ch.project.json",
        _ => return None,
    };
    example_dirs()
        .into_iter()
        .map(|directory| directory.join(project_name))
        .find(|path| path.is_file())
}

/// Resolve a public deep-link ROI target against normalized project records.
///
/// Both the native project UI and the control actor use this function so matching, ambiguity, and
/// error semantics cannot drift.
pub fn resolve_roi_target(
    rois: &[ProjectRoi],
    roi_query: Option<&str>,
    sample_query: Option<&str>,
) -> Result<ProjectRoi, String> {
    let Some(roi_query) = roi_query.map(str::trim).filter(|value| !value.is_empty()) else {
        return Err("Deep link is missing a roi=... parameter.".to_string());
    };

    let roi_norm = normalize_link_match_text(roi_query);
    let sample_norm = sample_query
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(normalize_link_match_text);
    let mut matches = rois
        .iter()
        .filter(|roi| {
            sample_norm.as_ref().is_none_or(|sample| {
                roi_link_match_texts(roi)
                    .iter()
                    .any(|candidate| candidate.contains(sample))
            })
        })
        .filter(|roi| {
            roi_link_match_texts(roi)
                .iter()
                .any(|candidate| candidate == &roi_norm || candidate.contains(&roi_norm))
        })
        .cloned()
        .collect::<Vec<_>>();

    if matches.is_empty() {
        return Err(format!("No project ROI matches '{roi_query}'."));
    }
    let exact = matches
        .iter()
        .filter(|roi| {
            roi_link_match_texts(roi)
                .iter()
                .any(|candidate| candidate == &roi_norm)
        })
        .cloned()
        .collect::<Vec<_>>();
    if !exact.is_empty() {
        matches = exact;
    }
    if matches.len() == 1 {
        return Ok(matches.remove(0));
    }
    let examples = matches
        .iter()
        .take(4)
        .map(ProjectRoi::source_display)
        .collect::<Vec<_>>()
        .join("; ");
    Err(format!(
        "Deep link ROI '{roi_query}' matches {} project ROIs. Add sample=... or use a more specific roi path. Examples: {examples}",
        matches.len()
    ))
}

fn roi_link_match_texts(roi: &ProjectRoi) -> Vec<String> {
    let mut texts = Vec::new();
    let mut push = |value: String| {
        let normalized = normalize_link_match_text(&value);
        if !normalized.is_empty() && !texts.iter().any(|existing| existing == &normalized) {
            texts.push(normalized);
        }
    };
    push(roi.id.clone());
    if let Some(display_name) = roi.display_name.as_ref() {
        push(display_name.clone());
    }
    if let Some(dataset) = roi.dataset.as_ref() {
        push(dataset.clone());
    }
    if let Some(source_key) = roi.source_key() {
        push(source_key);
    }
    push(roi.source_display());
    if let Some(path) = roi.local_path() {
        push(path.to_string_lossy().to_string());
        if let Some(file_name) = path.file_name().and_then(|value| value.to_str()) {
            push(file_name.to_string());
        }
        let components = path
            .components()
            .filter_map(|component| component.as_os_str().to_str())
            .collect::<Vec<_>>();
        for pair in components.windows(2) {
            push(pair.join("/"));
        }
        for triple in components.windows(3) {
            push(triple.join("/"));
        }
    }
    for (key, value) in &roi.meta {
        push(key.clone());
        push(value.clone());
        push(format!("{key}:{value}"));
    }
    texts
}

fn normalize_link_match_text(value: &str) -> String {
    value
        .trim()
        .trim_matches('"')
        .replace('\\', "/")
        .to_ascii_lowercase()
}

fn normalize_example_name(value: &str) -> String {
    value
        .chars()
        .filter(|character| character.is_ascii_alphanumeric())
        .map(|character| character.to_ascii_lowercase())
        .collect()
}

fn example_dirs() -> Vec<PathBuf> {
    let mut directories = Vec::new();
    if let Ok(executable) = std::env::current_exe()
        && let Some(binary_directory) = executable.parent()
    {
        directories.push(binary_directory.join("examples"));
        directories.push(binary_directory.join("../Resources/examples"));
        directories.push(binary_directory.join("../../Resources/examples"));
    }
    directories.push(PathBuf::from("/usr/share/odon/examples"));
    if let Ok(current_directory) = std::env::current_dir() {
        directories.push(current_directory.join("fixtures"));
    }
    directories
}
