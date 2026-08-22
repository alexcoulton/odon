use std::sync::Arc;

use anyhow::{Context, anyhow};
use object_store::ObjectStore;
use object_store::path::Path as ObjectPath;
use zarrs::storage::ReadableStorageTraits;

use super::dataset_source::DatasetSource;
use super::document::{DocumentDescriptor, OmeZarrDocumentResource, OpenedDocument};
use super::ome::OmeZarrDataset;

/// Session-only S3 credentials passed directly from the control actor to a bounded worker.
///
/// Deliberately does not implement `Debug` or `Serialize`: secrets must never enter snapshots,
/// diagnostics, events, or accidental debug output.
#[derive(Clone)]
pub struct S3SessionCredentials {
    pub endpoint: String,
    pub region: String,
    pub bucket: String,
    pub access_key: String,
    pub secret_key: String,
}

impl S3SessionCredentials {
    pub fn normalized(
        endpoint: &str,
        region: &str,
        bucket: &str,
        access_key: &str,
        secret_key: &str,
    ) -> Self {
        let endpoint = endpoint.trim().trim_end_matches('/');
        let endpoint = if endpoint.starts_with("http://") || endpoint.starts_with("https://") {
            endpoint.to_string()
        } else {
            format!("https://{endpoint}")
        };
        Self {
            endpoint,
            region: if region.trim().is_empty() {
                "auto".to_string()
            } else {
                region.trim().to_string()
            },
            bucket: bucket.trim().to_string(),
            access_key: access_key.trim().to_string(),
            secret_key: secret_key.trim().to_string(),
        }
    }

    pub fn redact_message(&self, message: &str) -> String {
        let mut redacted = message.to_string();
        for secret in [&self.access_key, &self.secret_key] {
            if !secret.is_empty() {
                redacted = redacted.replace(secret, "[redacted]");
            }
        }
        redacted
    }
}

pub trait RemoteDatasetBackend: Send + Sync {
    fn list_s3(
        &self,
        credentials: &S3SessionCredentials,
        prefix: &str,
    ) -> anyhow::Result<S3BrowseListing>;

    fn open_http(&self, url: &str) -> anyhow::Result<OpenedDocument<OmeZarrDocumentResource>>;

    fn open_s3(
        &self,
        credentials: &S3SessionCredentials,
        prefix: &str,
    ) -> anyhow::Result<OpenedDocument<OmeZarrDocumentResource>>;
}

pub struct CoreRemoteDatasetBackend;

impl RemoteDatasetBackend for CoreRemoteDatasetBackend {
    fn list_s3(
        &self,
        credentials: &S3SessionCredentials,
        prefix: &str,
    ) -> anyhow::Result<S3BrowseListing> {
        let browser = build_s3_browser(
            &credentials.endpoint,
            &credentials.region,
            &credentials.bucket,
            &credentials.access_key,
            &credentials.secret_key,
        )?;
        list_s3_prefix(&browser, prefix)
    }

    fn open_http(&self, url: &str) -> anyhow::Result<OpenedDocument<OmeZarrDocumentResource>> {
        let url = url.trim().trim_end_matches('/').to_string();
        let store = build_http_store(&url)?;
        let source = DatasetSource::Http {
            base_url: url.clone(),
        };
        let dataset = OmeZarrDataset::open_with_store(source, Arc::clone(&store))?;
        Ok(OpenedDocument {
            descriptor: DocumentDescriptor::from_ome_zarr(&dataset),
            resource: OmeZarrDocumentResource {
                dataset,
                store,
                runtime_guard: None,
            },
        })
    }

    fn open_s3(
        &self,
        credentials: &S3SessionCredentials,
        prefix: &str,
    ) -> anyhow::Result<OpenedDocument<OmeZarrDocumentResource>> {
        let prefix = normalize_prefix(prefix);
        let S3Store { store, runtime } = build_s3_store(
            &credentials.endpoint,
            &credentials.region,
            &credentials.bucket,
            &prefix,
            &credentials.access_key,
            &credentials.secret_key,
        )?;
        let source = DatasetSource::S3 {
            endpoint: credentials.endpoint.clone(),
            region: credentials.region.clone(),
            bucket: credentials.bucket.clone(),
            prefix,
        };
        let dataset = OmeZarrDataset::open_with_store(source, Arc::clone(&store))?;
        Ok(OpenedDocument {
            descriptor: DocumentDescriptor::from_ome_zarr(&dataset),
            resource: OmeZarrDocumentResource {
                dataset,
                store,
                runtime_guard: Some(runtime),
            },
        })
    }
}

pub fn build_http_store(base_url: &str) -> anyhow::Result<Arc<dyn ReadableStorageTraits>> {
    let base_url = base_url.trim().trim_end_matches('/');
    if base_url.is_empty() {
        anyhow::bail!("base URL is empty");
    }
    let base_url = if base_url.starts_with("http://") || base_url.starts_with("https://") {
        base_url.to_string()
    } else {
        format!("https://{base_url}")
    };
    let store = zarrs_http::HTTPStore::new(&base_url)
        .map_err(|e| anyhow!("failed to create HTTP store: {e}"))?;
    Ok(Arc::new(store))
}

pub struct S3Store {
    pub store: Arc<dyn ReadableStorageTraits>,
    pub runtime: Arc<tokio::runtime::Runtime>,
}

pub struct S3Browser {
    pub object_store: Arc<dyn ObjectStore>,
    pub runtime: Arc<tokio::runtime::Runtime>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct S3BrowseEntry {
    pub prefix: String,
    pub name: String,
    pub is_dataset: bool,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct S3BrowseListing {
    pub prefix: String,
    pub parent_prefix: Option<String>,
    pub entries: Vec<S3BrowseEntry>,
    pub current_is_dataset: bool,
}

#[derive(Clone)]
struct TokioBlockOn(tokio::runtime::Handle);

impl zarrs::storage::storage_adapter::async_to_sync::AsyncToSyncBlockOn for TokioBlockOn {
    fn block_on<F: core::future::Future>(&self, future: F) -> F::Output {
        self.0.block_on(future)
    }
}

pub fn build_s3_store(
    endpoint: &str,
    region: &str,
    bucket: &str,
    prefix: &str,
    access_key: &str,
    secret_key: &str,
) -> anyhow::Result<S3Store> {
    let browser = build_s3_browser(endpoint, region, bucket, access_key, secret_key)?;
    let prefix = prefix.trim().trim_matches('/');
    let object_store = object_store::prefix::PrefixStore::new(browser.object_store.clone(), prefix);

    let async_store = Arc::new(zarrs_object_store::AsyncObjectStore::new(object_store));
    let sync_store = zarrs::storage::storage_adapter::async_to_sync::AsyncToSyncStorageAdapter::new(
        async_store,
        TokioBlockOn(browser.runtime.handle().clone()),
    );

    Ok(S3Store {
        store: Arc::new(sync_store),
        runtime: browser.runtime,
    })
}

pub fn build_s3_browser(
    endpoint: &str,
    region: &str,
    bucket: &str,
    access_key: &str,
    secret_key: &str,
) -> anyhow::Result<S3Browser> {
    let endpoint = endpoint.trim();
    let region = region.trim();
    let bucket = bucket.trim();
    let access_key = access_key.trim();
    let secret_key = secret_key.trim();

    if endpoint.is_empty() {
        anyhow::bail!("S3 endpoint is empty");
    }
    if bucket.is_empty() {
        anyhow::bail!("S3 bucket is empty");
    }
    if access_key.is_empty() || secret_key.is_empty() {
        anyhow::bail!("S3 access/secret key are required");
    }

    let runtime =
        Arc::new(tokio::runtime::Runtime::new().context("failed to create tokio runtime for S3")?);

    let endpoint = if endpoint.starts_with("http://") || endpoint.starts_with("https://") {
        endpoint.to_string()
    } else {
        format!("https://{endpoint}")
    };
    let allow_http = endpoint.starts_with("http://");

    let s3 = object_store::aws::AmazonS3Builder::new()
        .with_endpoint(endpoint)
        .with_allow_http(allow_http)
        .with_region(if region.is_empty() { "auto" } else { region })
        .with_bucket_name(bucket)
        .with_access_key_id(access_key)
        .with_secret_access_key(secret_key)
        .build()
        .map_err(|e| anyhow!("failed to build S3 store: {e}"))?;

    Ok(S3Browser {
        object_store: Arc::new(s3),
        runtime,
    })
}

pub fn list_s3_prefix(browser: &S3Browser, prefix: &str) -> anyhow::Result<S3BrowseListing> {
    let prefix = normalize_prefix(prefix);
    let path = (!prefix.is_empty()).then(|| ObjectPath::from(prefix.clone()));
    let result = browser
        .runtime
        .handle()
        .block_on(browser.object_store.list_with_delimiter(path.as_ref()))
        .map_err(|e| anyhow!("failed to list S3 prefix: {e}"))?;

    let current_is_dataset = result.objects.iter().any(|meta| {
        let location = meta.location.to_string();
        let name = location.rsplit('/').next().unwrap_or("");
        matches!(name, "zarr.json" | ".zgroup" | ".zattrs")
    });

    let mut entries = result
        .common_prefixes
        .into_iter()
        .map(|path| {
            // `list_with_delimiter` common prefixes are often returned with a trailing `/`.
            // Normalize so dataset detection and naming work as expected.
            let prefix = path.to_string();
            let prefix = prefix.trim_end_matches('/').to_string();
            let name = prefix
                .split('/')
                .filter(|part| !part.is_empty())
                .next_back()
                .unwrap_or(&prefix)
                .to_string();
            // Many OME-Zarr datasets are named `*.zarr` (not necessarily `*.ome.zarr`).
            let is_dataset = prefix.ends_with(".ome.zarr") || prefix.ends_with(".zarr");
            S3BrowseEntry {
                prefix,
                name,
                is_dataset,
            }
        })
        .collect::<Vec<_>>();

    entries.sort_by(|a, b| {
        b.is_dataset
            .cmp(&a.is_dataset)
            .then_with(|| a.name.cmp(&b.name))
    });

    Ok(S3BrowseListing {
        prefix: prefix.clone(),
        parent_prefix: parent_prefix(&prefix),
        entries,
        current_is_dataset,
    })
}

fn normalize_prefix(prefix: &str) -> String {
    prefix.trim().trim_matches('/').to_string()
}

fn parent_prefix(prefix: &str) -> Option<String> {
    let mut parts = prefix
        .split('/')
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>();
    if parts.is_empty() {
        return None;
    }
    parts.pop();
    if parts.is_empty() {
        None
    } else {
        Some(parts.join("/"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remote_store_inputs_and_prefix_navigation_are_deterministic() {
        assert!(build_http_store("  ").is_err());
        assert!(build_http_store("example.test/data.ome.zarr/").is_ok());

        let cases = [
            ("", "", None),
            ("/study/", "study", None),
            (
                "/study/patient/roi/",
                "study/patient/roi",
                Some("study/patient"),
            ),
        ];
        for (raw, normalized, parent) in cases {
            assert_eq!(normalize_prefix(raw), normalized);
            assert_eq!(parent_prefix(normalized).as_deref(), parent);
        }
    }

    #[test]
    fn s3_builder_rejects_missing_credentials_before_network_access() {
        assert!(
            build_s3_browser("", "auto", "bucket", "key", "secret")
                .err()
                .expect("empty endpoint")
                .to_string()
                .contains("endpoint")
        );
        assert!(
            build_s3_browser("s3.example.test", "auto", "", "key", "secret")
                .err()
                .expect("empty bucket")
                .to_string()
                .contains("bucket")
        );
        assert!(
            build_s3_browser("s3.example.test", "auto", "bucket", "", "")
                .err()
                .expect("missing credentials")
                .to_string()
                .contains("access/secret")
        );
    }
}
