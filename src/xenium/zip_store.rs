use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use parking_lot::Mutex;
use zarrs::storage::{
    Bytes, MaybeBytesIterator, StorageError, StoreKey,
    byte_range::{ByteRange, ByteRangeIterator},
};
use zip::ZipArchive;

/// A minimal zip-backed Zarr store for Xenium Explorer `*.zarr.zip` bundles.
///
/// This is intentionally read-only and keyed by the exact zip entry name.
#[derive(Debug)]
pub struct ZipStore {
    zip_path: PathBuf,
    inner: Mutex<ZipArchive<File>>,
    index: HashMap<String, usize>,
}

impl ZipStore {
    pub fn open(zip_path: &Path) -> anyhow::Result<Arc<Self>> {
        let zip_path = zip_path
            .canonicalize()
            .unwrap_or_else(|_| zip_path.to_path_buf());
        let f = File::open(&zip_path)?;
        let mut z = ZipArchive::new(f)?;
        let mut index = HashMap::new();
        for i in 0..z.len() {
            if let Ok(file) = z.by_index(i) {
                index.insert(file.name().to_string(), i);
            }
        }
        Ok(Arc::new(Self {
            zip_path,
            inner: Mutex::new(z),
            index,
        }))
    }

    fn read_full(&self, name: &str) -> Result<Option<Vec<u8>>, StorageError> {
        let Some(&idx) = self.index.get(name) else {
            return Ok(None);
        };

        let mut guard = self.inner.lock();
        let mut f = guard
            .by_index(idx)
            .map_err(|e| StorageError::Other(e.to_string()))?;
        if f.is_dir() {
            return Ok(None);
        }
        let mut out = Vec::with_capacity(f.size() as usize);
        f.read_to_end(&mut out)
            .map_err(|e| StorageError::Other(e.to_string()))?;
        Ok(Some(out))
    }
}

impl zarrs::storage::ReadableStorageTraits for ZipStore {
    fn get_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        byte_ranges: ByteRangeIterator<'a>,
    ) -> Result<MaybeBytesIterator<'a>, StorageError> {
        let name = key.as_str();
        let Some(all) = self.read_full(name)? else {
            return Ok(None);
        };

        let all = Arc::new(all);
        let size = all.len() as u64;
        let mut outputs: Vec<Result<Bytes, StorageError>> = Vec::new();
        for br in byte_ranges {
            let start = br.start(size);
            let end = br.end(size);
            if start > end || end > size {
                outputs.push(Err(StorageError::Other(format!(
                    "invalid byte range {br} for value of size {size}"
                ))));
                continue;
            }
            let s = start as usize;
            let e = end as usize;
            let bytes = if s >= e {
                Bytes::new()
            } else {
                Bytes::from(all[s..e].to_vec())
            };
            outputs.push(Ok(bytes));
        }

        Ok(Some(Box::new(outputs.into_iter())))
    }

    fn size_key(&self, key: &StoreKey) -> Result<Option<u64>, StorageError> {
        let name = key.as_str();
        let Some(&idx) = self.index.get(name) else {
            return Ok(None);
        };
        let mut guard = self.inner.lock();
        let f = guard
            .by_index(idx)
            .map_err(|e| StorageError::Other(e.to_string()))?;
        Ok(Some(f.size()))
    }

    fn supports_get_partial(&self) -> bool {
        // We can serve partial reads, but currently do so by reading the full file then slicing.
        true
    }
}
