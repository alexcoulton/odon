use std::fs;
use std::net::{SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use super::PROTOCOL_VERSION;

const RUNTIME_DIR_ENV: &str = "ODON_RUNTIME_DIR";
const MANIFEST_PREFIX: &str = "instance-";
const MANIFEST_SUFFIX: &str = ".json";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct InstanceManifest {
    pub instance_id: String,
    pub pid: u32,
    pub endpoint: String,
    pub token: String,
    pub app_version: String,
    pub protocol_versions: Vec<u32>,
    pub started_at_unix_ms: u128,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project_path: Option<PathBuf>,
}

impl InstanceManifest {
    pub fn new(address: SocketAddr) -> anyhow::Result<Self> {
        Ok(Self {
            instance_id: random_uuid_like()?,
            pid: std::process::id(),
            endpoint: format!("tcp://{address}"),
            token: random_hex(32)?,
            app_version: env!("CARGO_PKG_VERSION").to_string(),
            protocol_versions: vec![PROTOCOL_VERSION],
            started_at_unix_ms: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis(),
            project_path: None,
        })
    }

    pub fn socket_addr(&self) -> anyhow::Result<SocketAddr> {
        self.endpoint
            .strip_prefix("tcp://")
            .ok_or_else(|| anyhow::anyhow!("unsupported Odon endpoint: {}", self.endpoint))?
            .parse()
            .map_err(anyhow::Error::from)
    }
}

#[derive(Debug)]
pub struct InstanceManifestGuard {
    manifest: InstanceManifest,
    path: PathBuf,
}

impl InstanceManifestGuard {
    pub fn publish(manifest: InstanceManifest) -> anyhow::Result<Self> {
        Self::publish_in(manifest, &runtime_dir()?)
    }

    fn publish_in(manifest: InstanceManifest, directory: &Path) -> anyhow::Result<Self> {
        create_private_directory(directory)?;
        let path = directory.join(format!(
            "{MANIFEST_PREFIX}{}{MANIFEST_SUFFIX}",
            manifest.instance_id
        ));
        let temporary = directory.join(format!(
            ".{MANIFEST_PREFIX}{}-{}.tmp",
            manifest.instance_id,
            std::process::id()
        ));
        let bytes = serde_json::to_vec_pretty(&manifest)?;
        write_private_file(&temporary, &bytes)?;
        fs::rename(&temporary, &path)?;
        Ok(Self { manifest, path })
    }

    pub fn manifest(&self) -> &InstanceManifest {
        &self.manifest
    }
}

impl Drop for InstanceManifestGuard {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

pub fn runtime_dir() -> anyhow::Result<PathBuf> {
    if let Some(path) = std::env::var_os(RUNTIME_DIR_ENV).filter(|value| !value.is_empty()) {
        return Ok(PathBuf::from(path));
    }
    if let Some(path) = dirs::runtime_dir() {
        return Ok(path.join("odon"));
    }
    if let Some(path) = dirs::cache_dir() {
        return Ok(path.join("odon").join("runtime"));
    }
    Ok(std::env::temp_dir().join("odon-runtime"))
}

pub fn discover_instances(clean_stale: bool) -> anyhow::Result<Vec<InstanceManifest>> {
    discover_instances_in(&runtime_dir()?, clean_stale)
}

fn discover_instances_in(
    directory: &Path,
    clean_stale: bool,
) -> anyhow::Result<Vec<InstanceManifest>> {
    let entries = match fs::read_dir(directory) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(error.into()),
    };
    let mut manifests = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if !name.starts_with(MANIFEST_PREFIX) || !name.ends_with(MANIFEST_SUFFIX) {
            continue;
        }
        let manifest = fs::read(&path)
            .ok()
            .and_then(|bytes| serde_json::from_slice::<InstanceManifest>(&bytes).ok());
        let Some(manifest) = manifest else {
            if clean_stale {
                let _ = fs::remove_file(path);
            }
            continue;
        };
        if manifest_is_live(&manifest) {
            manifests.push(manifest);
        } else if clean_stale {
            let _ = fs::remove_file(path);
        }
    }
    manifests.sort_by(|left, right| {
        right
            .started_at_unix_ms
            .cmp(&left.started_at_unix_ms)
            .then_with(|| left.instance_id.cmp(&right.instance_id))
    });
    Ok(manifests)
}

pub fn select_instance(instance_id: Option<&str>) -> anyhow::Result<InstanceManifest> {
    let instances = discover_instances(true)?;
    if let Some(instance_id) = instance_id {
        return instances
            .into_iter()
            .find(|instance| instance.instance_id == instance_id)
            .ok_or_else(|| anyhow::anyhow!("no running Odon instance has ID '{instance_id}'"));
    }
    match instances.as_slice() {
        [] => anyhow::bail!("no running Odon instances were discovered"),
        [instance] => Ok(instance.clone()),
        _ => anyhow::bail!(
            "multiple Odon instances are running; set ODON_INSTANCE_ID or select an instance explicitly"
        ),
    }
}

fn manifest_is_live(manifest: &InstanceManifest) -> bool {
    let Ok(address) = manifest.socket_addr() else {
        return false;
    };
    TcpStream::connect_timeout(&address, Duration::from_millis(150)).is_ok()
}

fn random_hex(byte_count: usize) -> anyhow::Result<String> {
    let mut bytes = vec![0u8; byte_count];
    getrandom::fill(&mut bytes)
        .map_err(|error| anyhow::anyhow!("OS randomness failed: {error}"))?;
    Ok(bytes.iter().map(|byte| format!("{byte:02x}")).collect())
}

pub(crate) fn random_uuid_like() -> anyhow::Result<String> {
    let value = random_hex(16)?;
    Ok(format!(
        "{}-{}-{}-{}-{}",
        &value[0..8],
        &value[8..12],
        &value[12..16],
        &value[16..20],
        &value[20..32]
    ))
}

fn create_private_directory(path: &Path) -> anyhow::Result<()> {
    fs::create_dir_all(path)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(0o700))?;
    }
    Ok(())
}

fn write_private_file(path: &Path, bytes: &[u8]) -> anyhow::Result<()> {
    use std::io::Write;
    let mut options = fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    struct TestDir(PathBuf);

    impl TestDir {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "odon-discovery-test-{}-{}",
                std::process::id(),
                random_hex(6).expect("random test directory")
            ));
            fs::create_dir_all(&path).expect("create test directory");
            Self(path)
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn manifest_roundtrip_and_guard_cleanup() {
        let directory = TestDir::new();
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind live endpoint");
        let manifest = InstanceManifest::new(listener.local_addr().unwrap()).unwrap();
        let instance_id = manifest.instance_id.clone();
        let guard = InstanceManifestGuard::publish_in(manifest.clone(), &directory.0).unwrap();

        let discovered = discover_instances_in(&directory.0, true).unwrap();
        assert_eq!(discovered, vec![manifest]);
        assert_eq!(guard.manifest().instance_id, instance_id);
        drop(guard);
        assert!(
            discover_instances_in(&directory.0, true)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn stale_and_malformed_manifests_are_removed() {
        let directory = TestDir::new();
        let stale_listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = stale_listener.local_addr().unwrap();
        drop(stale_listener);
        let stale = InstanceManifest::new(address).unwrap();
        let stale_path = directory.0.join(format!(
            "{MANIFEST_PREFIX}{}{MANIFEST_SUFFIX}",
            stale.instance_id
        ));
        fs::write(&stale_path, serde_json::to_vec(&stale).unwrap()).unwrap();
        let malformed_path = directory
            .0
            .join(format!("{MANIFEST_PREFIX}malformed{MANIFEST_SUFFIX}"));
        fs::write(&malformed_path, b"not json").unwrap();

        assert!(
            discover_instances_in(&directory.0, true)
                .unwrap()
                .is_empty()
        );
        assert!(!stale_path.exists());
        assert!(!malformed_path.exists());
    }
}
