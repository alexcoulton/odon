use std::fs;
use std::path::Path;

use anyhow::{Context, anyhow};
use zarrs::storage::{ReadableStorageTraits, StoreKey};

pub fn read_node_attributes(
    dir: &Path,
) -> anyhow::Result<Option<serde_json::Map<String, serde_json::Value>>> {
    let v2 = dir.join(".zattrs");
    if v2.is_file() {
        let text = fs::read_to_string(&v2).with_context(|| format!("failed to read {v2:?}"))?;
        let value: serde_json::Value =
            serde_json::from_str(&text).context("failed to parse .zattrs JSON")?;
        let Some(obj) = value.as_object() else {
            return Err(anyhow!(".zattrs must be a JSON object"));
        };
        return Ok(Some(obj.clone()));
    }

    let v3 = dir.join("zarr.json");
    if v3.is_file() {
        let text = fs::read_to_string(&v3).with_context(|| format!("failed to read {v3:?}"))?;
        let value: serde_json::Value =
            serde_json::from_str(&text).context("failed to parse zarr.json")?;
        if let Some(obj) = value.get("attributes").and_then(|v| v.as_object()) {
            return Ok(Some(obj.clone()));
        }
        // Be permissive: some producers may put OME-NGFF metadata at the top-level of zarr.json.
        if let Some(obj) = value.as_object() {
            if obj.contains_key("multiscales") || obj.contains_key("omero") {
                return Ok(Some(obj.clone()));
            }
        }
        // zarr.json exists but no attributes; treat as empty attributes.
        return Ok(Some(serde_json::Map::new()));
    }

    Ok(None)
}

pub fn read_node_attributes_store(
    store: &dyn ReadableStorageTraits,
    prefix: &str,
) -> anyhow::Result<Option<serde_json::Map<String, serde_json::Value>>> {
    let prefix = prefix.trim_matches('/');

    let key_for = |name: &str| -> anyhow::Result<StoreKey> {
        let key = if prefix.is_empty() {
            name.to_string()
        } else {
            format!("{prefix}/{name}")
        };
        Ok(StoreKey::new(key).map_err(|e| anyhow!(e.to_string()))?)
    };

    let v2 = key_for(".zattrs")?;
    if let Some(bytes) = store.get(&v2).map_err(|e| anyhow!(e.to_string()))? {
        let text = String::from_utf8_lossy(bytes.as_ref()).to_string();
        let value: serde_json::Value =
            serde_json::from_str(&text).context("failed to parse .zattrs JSON")?;
        let Some(obj) = value.as_object() else {
            return Err(anyhow!(".zattrs must be a JSON object"));
        };
        return Ok(Some(obj.clone()));
    }

    let v3 = key_for("zarr.json")?;
    if let Some(bytes) = store.get(&v3).map_err(|e| anyhow!(e.to_string()))? {
        let text = String::from_utf8_lossy(bytes.as_ref()).to_string();
        let value: serde_json::Value =
            serde_json::from_str(&text).context("failed to parse zarr.json")?;
        if let Some(obj) = value.get("attributes").and_then(|v| v.as_object()) {
            return Ok(Some(obj.clone()));
        }
        // Be permissive: some producers may put OME-NGFF metadata at the top-level of zarr.json.
        if let Some(obj) = value.as_object() {
            if obj.contains_key("multiscales") || obj.contains_key("omero") {
                return Ok(Some(obj.clone()));
            }
        }
        return Ok(Some(serde_json::Map::new()));
    }

    Ok(None)
}

pub fn normalize_ngff_attributes(
    attrs: serde_json::Map<String, serde_json::Value>,
) -> serde_json::Map<String, serde_json::Value> {
    if attrs.contains_key("multiscales") {
        return attrs;
    }
    let Some(ome) = attrs.get("ome").and_then(|v| v.as_object()).cloned() else {
        return attrs;
    };
    let mut out = ome;
    if !out.contains_key("omero") {
        if let Some(omero) = attrs.get("omero").cloned() {
            out.insert("omero".to_string(), omero);
        }
    }
    out
}
