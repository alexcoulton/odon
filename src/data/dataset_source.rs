use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DatasetSource {
    Local(PathBuf),
    Http {
        base_url: String,
    },
    S3 {
        endpoint: String,
        region: String,
        bucket: String,
        prefix: String,
    },
}

impl DatasetSource {
    pub fn source_key(&self) -> String {
        match self {
            DatasetSource::Local(path) => format!("local:{}", path.to_string_lossy()),
            DatasetSource::Http { base_url } => format!("http:{base_url}"),
            DatasetSource::S3 {
                endpoint,
                region,
                bucket,
                prefix,
            } => format!("s3:{endpoint}|{region}|{bucket}|{prefix}"),
        }
    }

    pub fn is_local(&self) -> bool {
        matches!(self, DatasetSource::Local(_))
    }

    pub fn local_path(&self) -> Option<&Path> {
        match self {
            DatasetSource::Local(p) => Some(p.as_path()),
            _ => None,
        }
    }

    pub fn display_name(&self) -> String {
        match self {
            DatasetSource::Local(p) => p
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("<dataset>")
                .to_string(),
            DatasetSource::Http { base_url } => {
                let trimmed = base_url.trim_end_matches('/');
                trimmed
                    .rsplit('/')
                    .next()
                    .filter(|s| !s.is_empty())
                    .unwrap_or(trimmed)
                    .to_string()
            }
            DatasetSource::S3 { bucket, prefix, .. } => {
                let tail = prefix
                    .trim_end_matches('/')
                    .rsplit('/')
                    .next()
                    .unwrap_or("");
                if tail.is_empty() {
                    format!("s3://{bucket}")
                } else {
                    tail.to_string()
                }
            }
        }
    }
}
