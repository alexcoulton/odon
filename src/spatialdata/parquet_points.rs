use std::fs;
use std::path::{Path, PathBuf};

use anyhow::Context;
use arrow_array::Array;
use arrow_array::types::{ArrowDictionaryKeyType, Int16Type, Int32Type, Int64Type};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

#[derive(Debug, Clone)]
pub struct PointsPayload {
    pub xy: Vec<[f32; 2]>,
    pub meta: PointsMeta,
}

#[derive(Debug, Clone, Default)]
pub struct PointsMeta {
    pub z: Option<Vec<f32>>,
    pub feature: Option<FeatureData>,
    pub cell_id: Option<Vec<i32>>,
    pub overlaps_nucleus: Option<Vec<u8>>,
    pub transcript_id: Option<Vec<u64>>,
    pub qv: Option<Vec<f32>>,
}

#[derive(Debug, Clone, Default)]
pub struct FeatureData {
    pub dict: Vec<String>,
    pub ids: Vec<u32>,
}

#[derive(Debug, Clone)]
pub struct PointsLoadOptions {
    pub max_points: usize,
    /// Optional column name to treat as a "feature" (e.g. Xenium gene name).
    pub feature_column: Option<String>,
    /// Optional filter for a single feature (case-insensitive exact match).
    pub gene_filter: Option<String>,
}

impl Default for PointsLoadOptions {
    fn default() -> Self {
        Self {
            max_points: 200_000,
            feature_column: Some("feature_name".to_string()),
            gene_filter: None,
        }
    }
}

pub fn load_points_sample(
    points_parquet_dir: &Path,
    options: &PointsLoadOptions,
) -> anyhow::Result<PointsPayload> {
    let mut out_xy: Vec<[f32; 2]> = Vec::new();
    let mut meta = PointsMeta::default();
    let mut feature_builder = FeatureDictBuilder::default();

    let mut files: Vec<PathBuf> = Vec::new();
    for entry in fs::read_dir(points_parquet_dir)
        .with_context(|| format!("failed to read dir: {points_parquet_dir:?}"))?
    {
        let entry = entry?;
        let p = entry.path();
        if p.is_file() && p.extension().is_some_and(|e| e == "parquet") {
            files.push(p);
        }
    }
    files.sort();

    if files.is_empty() {
        anyhow::bail!(
            "no parquet files under {}",
            points_parquet_dir.to_string_lossy()
        );
    }

    // Important: don't just read `part.0` until `max_points` is filled. For Xenium, each `part.*`
    // often corresponds to a spatial region. If we stop early, the sample may be biased to a
    // single corner (e.g. only the upper-left of the image).
    //
    // Instead, we take a per-file quota so the preview is spatially representative.
    let max_total = if options.max_points == 0 {
        usize::MAX
    } else {
        options.max_points.max(1)
    };
    let base = max_total / files.len().max(1);
    let rem = max_total % files.len().max(1);
    for (i, file) in files.iter().enumerate() {
        let quota = base + usize::from(i < rem);
        if quota == 0 {
            continue;
        }
        let cap = out_xy.len().saturating_add(quota).min(max_total);
        load_points_from_file(
            file,
            options,
            cap,
            &mut out_xy,
            &mut meta,
            &mut feature_builder,
        )
        .with_context(|| format!("read points parquet: {file:?}"))?;
        if out_xy.len() >= max_total {
            break;
        }
    }

    if !feature_builder.dict.is_empty() && !feature_builder.ids.is_empty() {
        meta.feature = Some(FeatureData {
            dict: feature_builder.dict,
            ids: feature_builder.ids,
        });
    }

    Ok(PointsPayload { xy: out_xy, meta })
}

fn load_points_from_file(
    path: &Path,
    options: &PointsLoadOptions,
    max_total: usize,
    out_xy: &mut Vec<[f32; 2]>,
    meta: &mut PointsMeta,
    feature_builder: &mut FeatureDictBuilder,
) -> anyhow::Result<()> {
    let file = std::fs::File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let feature = options.feature_column.clone().unwrap_or_default();
    let want_gene_upper = options
        .gene_filter
        .as_ref()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_ascii_uppercase());
    if want_gene_upper.is_some() && feature.is_empty() {
        anyhow::bail!("gene_filter requires feature_column");
    }

    // Read all columns so we can support tooltips (Xenium uses additional metadata columns).
    let mut reader = builder.with_batch_size(65_536).build()?;

    while let Some(batch) = reader.next() {
        let batch = batch?;
        let n = batch.num_rows();
        if n == 0 {
            continue;
        }

        let schema = batch.schema();
        let x_i = schema
            .index_of("x")
            .context("missing required column 'x'")?;
        let y_i = schema
            .index_of("y")
            .context("missing required column 'y'")?;
        let z_i = schema.index_of("z").ok();
        let cell_i = schema.index_of("cell_id").ok();
        let overlaps_i = schema.index_of("overlaps_nucleus").ok();
        let transcript_i = schema.index_of("transcript_id").ok();
        let qv_i = schema.index_of("qv").ok();
        let feat_i = (!feature.is_empty())
            .then(|| schema.index_of(feature.as_str()).ok())
            .flatten();

        let x = batch.column(x_i).as_ref();
        let y = batch.column(y_i).as_ref();

        // Lazily allocate meta arrays if present.
        if z_i.is_some() && meta.z.is_none() {
            meta.z = Some(Vec::new());
        }
        if cell_i.is_some() && meta.cell_id.is_none() {
            meta.cell_id = Some(Vec::new());
        }
        if overlaps_i.is_some() && meta.overlaps_nucleus.is_none() {
            meta.overlaps_nucleus = Some(Vec::new());
        }
        if transcript_i.is_some() && meta.transcript_id.is_none() {
            meta.transcript_id = Some(Vec::new());
        }
        if qv_i.is_some() && meta.qv.is_none() {
            meta.qv = Some(Vec::new());
        }

        // Prepare typed column readers.
        let x_col = NumF32Col::try_new(x).context("column x")?;
        let y_col = NumF32Col::try_new(y).context("column y")?;
        let z_col = z_i
            .map(|i| NumF32Col::try_new(batch.column(i).as_ref()))
            .transpose()
            .context("column z")?;
        let cell_col = cell_i
            .map(|i| NumI32Col::try_new(batch.column(i).as_ref()))
            .transpose()
            .context("column cell_id")?;
        let overlaps_col = overlaps_i
            .map(|i| NumU8Col::try_new(batch.column(i).as_ref()))
            .transpose()
            .context("column overlaps_nucleus")?;
        let transcript_col = transcript_i
            .map(|i| NumU64Col::try_new(batch.column(i).as_ref()))
            .transpose()
            .context("column transcript_id")?;
        let qv_col = qv_i
            .map(|i| NumF32Col::try_new(batch.column(i).as_ref()))
            .transpose()
            .context("column qv")?;

        let feat_col = feat_i.map(|i| batch.column(i).as_ref());
        let (feat_get, feat_local_map, feat_match_local) =
            prepare_feature_batch(feat_col, feature_builder, want_gene_upper.as_deref())
                .context("prepare feature column")?;

        for row in 0..n {
            if out_xy.len() >= max_total {
                return Ok(());
            }
            let Some(xv) = x_col.get(row) else { continue };
            let Some(yv) = y_col.get(row) else { continue };

            if let Some(want_upper) = want_gene_upper.as_ref() {
                // If we have a dictionary column, use the precomputed match table.
                if let Some(match_local) = feat_match_local.as_ref() {
                    let Some(local_i) = feat_get.as_ref().and_then(|g| g.local_key(row)) else {
                        continue;
                    };
                    if !match_local.get(local_i).copied().unwrap_or(false) {
                        continue;
                    }
                } else if let Some(g) = feat_get.as_ref().and_then(|g| g.value(row)) {
                    if g.to_ascii_uppercase() != *want_upper {
                        continue;
                    }
                } else {
                    continue;
                }
            }

            out_xy.push([xv, yv]);

            // Feature ids (dictionary-mapped to a global dict).
            if let (Some(get), Some(local_map)) = (feat_get.as_ref(), feat_local_map.as_ref()) {
                if let Some(local_i) = get.local_key(row) {
                    if let Some(&gid) = local_map.get(local_i) {
                        feature_builder.ids.push(gid);
                    } else {
                        feature_builder.ids.push(0);
                    }
                } else {
                    feature_builder.ids.push(0);
                }
            }

            if let (Some(col), Some(out)) = (z_col.as_ref(), meta.z.as_mut()) {
                out.push(col.get(row).unwrap_or(f32::NAN));
            }
            if let (Some(col), Some(out)) = (cell_col.as_ref(), meta.cell_id.as_mut()) {
                out.push(col.get(row).unwrap_or(-1));
            }
            if let (Some(col), Some(out)) = (overlaps_col.as_ref(), meta.overlaps_nucleus.as_mut())
            {
                out.push(col.get(row).unwrap_or(0));
            }
            if let (Some(col), Some(out)) = (transcript_col.as_ref(), meta.transcript_id.as_mut()) {
                out.push(col.get(row).unwrap_or(0));
            }
            if let (Some(col), Some(out)) = (qv_col.as_ref(), meta.qv.as_mut()) {
                out.push(col.get(row).unwrap_or(f32::NAN));
            }
        }
    }

    Ok(())
}

fn get_utf8<'a>(array: &'a dyn arrow_array::Array, row: usize) -> anyhow::Result<Option<&'a str>> {
    if array.is_null(row) {
        return Ok(None);
    }
    if let Some(col) = array.as_any().downcast_ref::<arrow_array::StringArray>() {
        return Ok(Some(col.value(row)));
    }
    if let Some(col) = array
        .as_any()
        .downcast_ref::<arrow_array::LargeStringArray>()
    {
        return Ok(Some(col.value(row)));
    }
    macro_rules! dict_utf8 {
        ($key:ty) => {
            if let Some(col) = array
                .as_any()
                .downcast_ref::<arrow_array::DictionaryArray<$key>>()
            {
                if col.is_null(row) {
                    return Ok(None);
                }
                let keys = col.keys();
                let key_i64 = keys.value(row) as i64;
                if key_i64 < 0 {
                    return Err(anyhow::anyhow!("invalid dictionary key"));
                }
                return get_utf8(col.values().as_ref(), key_i64 as usize);
            }
        };
    }

    dict_utf8!(arrow_array::types::Int8Type);
    dict_utf8!(arrow_array::types::Int16Type);
    dict_utf8!(arrow_array::types::Int32Type);
    dict_utf8!(arrow_array::types::Int64Type);
    dict_utf8!(arrow_array::types::UInt8Type);
    dict_utf8!(arrow_array::types::UInt16Type);
    dict_utf8!(arrow_array::types::UInt32Type);
    dict_utf8!(arrow_array::types::UInt64Type);

    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xenium_points_parquet_smoke_if_present() {
        let dir = Path::new("data.zarr/points/transcripts/points.parquet");
        if !dir.is_dir() {
            eprintln!("skipping: {dir:?} not present");
            return;
        }

        let opts = PointsLoadOptions {
            max_points: 10_000,
            ..Default::default()
        };
        let payload = load_points_sample(dir, &opts).expect("load_points_sample");
        let xy = payload.xy;
        let meta = payload.meta;
        assert!(!xy.is_empty());
        let feat = meta.feature.expect("expected feature column");
        assert_eq!(xy.len(), feat.ids.len());
        let gene = feat
            .dict
            .get(feat.ids[0] as usize)
            .cloned()
            .unwrap_or_default();
        if !gene.trim().is_empty() {
            let opts_gene = PointsLoadOptions {
                max_points: 1_000,
                gene_filter: Some(gene.clone()),
                ..Default::default()
            };
            let payload2 = load_points_sample(dir, &opts_gene).expect("load_points_sample gene");
            let feat2 = payload2.meta.feature.expect("feature column");
            assert!(!feat2.ids.is_empty());
            assert!(
                feat2
                    .ids
                    .iter()
                    .all(|&i| feat2.dict[i as usize].eq_ignore_ascii_case(&gene))
            );
        }
    }
}

#[derive(Debug, Clone)]
struct FeatureDictBuilder {
    dict: Vec<String>,
    map: std::collections::HashMap<String, u32>,
    ids: Vec<u32>,
}

impl Default for FeatureDictBuilder {
    fn default() -> Self {
        Self {
            dict: Vec::new(),
            map: std::collections::HashMap::new(),
            ids: Vec::new(),
        }
    }
}

impl FeatureDictBuilder {
    fn intern(&mut self, s: &str) -> u32 {
        if let Some(&id) = self.map.get(s) {
            return id;
        }
        let id = self.dict.len() as u32;
        self.dict.push(s.to_string());
        self.map.insert(s.to_string(), id);
        id
    }
}

#[derive(Debug)]
enum FeatureGet<'a> {
    DictI16(&'a arrow_array::DictionaryArray<Int16Type>),
    DictI32(&'a arrow_array::DictionaryArray<Int32Type>),
    DictI64(&'a arrow_array::DictionaryArray<Int64Type>),
    Utf8(&'a arrow_array::StringArray),
    LargeUtf8(&'a arrow_array::LargeStringArray),
}

impl<'a> FeatureGet<'a> {
    fn local_key(&self, row: usize) -> Option<usize> {
        match self {
            FeatureGet::DictI16(col) => {
                if col.is_null(row) {
                    return None;
                }
                let k = col.keys().value(row) as i64;
                (k >= 0).then(|| k as usize)
            }
            FeatureGet::DictI32(col) => {
                if col.is_null(row) {
                    return None;
                }
                let k = col.keys().value(row) as i64;
                (k >= 0).then(|| k as usize)
            }
            FeatureGet::DictI64(col) => {
                if col.is_null(row) {
                    return None;
                }
                let k = col.keys().value(row);
                (k >= 0).then(|| k as usize)
            }
            _ => None,
        }
    }

    fn value(&self, row: usize) -> Option<&'a str> {
        match self {
            FeatureGet::DictI16(col) => {
                if col.is_null(row) {
                    return None;
                }
                let k = col.keys().value(row) as i64;
                if k < 0 {
                    return None;
                }
                get_utf8(col.values().as_ref(), k as usize).ok().flatten()
            }
            FeatureGet::DictI32(col) => {
                if col.is_null(row) {
                    return None;
                }
                let k = col.keys().value(row) as i64;
                if k < 0 {
                    return None;
                }
                get_utf8(col.values().as_ref(), k as usize).ok().flatten()
            }
            FeatureGet::DictI64(col) => {
                if col.is_null(row) {
                    return None;
                }
                let k = col.keys().value(row);
                if k < 0 {
                    return None;
                }
                get_utf8(col.values().as_ref(), k as usize).ok().flatten()
            }
            FeatureGet::Utf8(col) => (!col.is_null(row)).then(|| col.value(row)),
            FeatureGet::LargeUtf8(col) => (!col.is_null(row)).then(|| col.value(row)),
        }
    }
}

fn prepare_feature_batch<'a>(
    feat: Option<&'a dyn arrow_array::Array>,
    builder: &mut FeatureDictBuilder,
    want_gene_upper: Option<&str>,
) -> anyhow::Result<(Option<FeatureGet<'a>>, Option<Vec<u32>>, Option<Vec<bool>>)> {
    let Some(feat) = feat else {
        return Ok((None, None, None));
    };

    // Dictionary encoded (common for Xenium).
    if let Some(col) = feat
        .as_any()
        .downcast_ref::<arrow_array::DictionaryArray<Int16Type>>()
    {
        return prepare_feature_dict(col, builder, want_gene_upper).map(|(m, b)| {
            (
                Some(FeatureGet::DictI16(col)),
                Some(m),
                want_gene_upper.map(|_| b),
            )
        });
    }
    if let Some(col) = feat
        .as_any()
        .downcast_ref::<arrow_array::DictionaryArray<Int32Type>>()
    {
        return prepare_feature_dict(col, builder, want_gene_upper).map(|(m, b)| {
            (
                Some(FeatureGet::DictI32(col)),
                Some(m),
                want_gene_upper.map(|_| b),
            )
        });
    }
    if let Some(col) = feat
        .as_any()
        .downcast_ref::<arrow_array::DictionaryArray<Int64Type>>()
    {
        return prepare_feature_dict(col, builder, want_gene_upper).map(|(m, b)| {
            (
                Some(FeatureGet::DictI64(col)),
                Some(m),
                want_gene_upper.map(|_| b),
            )
        });
    }

    if let Some(col) = feat.as_any().downcast_ref::<arrow_array::StringArray>() {
        // Fallback: intern per-row (can be slow for huge point sets).
        // We don't build a local_map in this case.
        for v in col.iter().flatten() {
            let _ = builder.intern(v);
        }
        return Ok((Some(FeatureGet::Utf8(col)), None, None));
    }
    if let Some(col) = feat
        .as_any()
        .downcast_ref::<arrow_array::LargeStringArray>()
    {
        for v in col.iter().flatten() {
            let _ = builder.intern(v);
        }
        return Ok((Some(FeatureGet::LargeUtf8(col)), None, None));
    }

    Ok((None, None, None))
}

fn prepare_feature_dict<'a, K: ArrowDictionaryKeyType>(
    col: &'a arrow_array::DictionaryArray<K>,
    builder: &mut FeatureDictBuilder,
    want_gene_upper: Option<&str>,
) -> anyhow::Result<(Vec<u32>, Vec<bool>)> {
    let values = col.values().as_ref();
    let mut local_to_global: Vec<u32> = Vec::new();
    let mut match_local: Vec<bool> = Vec::new();

    if let Some(sa) = values.as_any().downcast_ref::<arrow_array::StringArray>() {
        local_to_global.reserve(sa.len());
        match_local.reserve(sa.len());
        for v in sa.iter() {
            let s = v.unwrap_or("");
            local_to_global.push(builder.intern(s));
            match_local.push(
                want_gene_upper.is_some_and(|w| s.to_ascii_uppercase() == w.to_ascii_uppercase()),
            );
        }
        return Ok((local_to_global, match_local));
    }
    if let Some(sa) = values
        .as_any()
        .downcast_ref::<arrow_array::LargeStringArray>()
    {
        local_to_global.reserve(sa.len());
        match_local.reserve(sa.len());
        for v in sa.iter() {
            let s = v.unwrap_or("");
            local_to_global.push(builder.intern(s));
            match_local.push(
                want_gene_upper.is_some_and(|w| s.to_ascii_uppercase() == w.to_ascii_uppercase()),
            );
        }
        return Ok((local_to_global, match_local));
    }

    anyhow::bail!("unsupported dictionary value type for feature column")
}

#[derive(Clone)]
enum NumF32Col<'a> {
    F32(&'a arrow_array::Float32Array),
    F64(&'a arrow_array::Float64Array),
}

impl<'a> NumF32Col<'a> {
    fn try_new(array: &'a dyn arrow_array::Array) -> anyhow::Result<Self> {
        if let Some(col) = array.as_any().downcast_ref::<arrow_array::Float32Array>() {
            return Ok(Self::F32(col));
        }
        if let Some(col) = array.as_any().downcast_ref::<arrow_array::Float64Array>() {
            return Ok(Self::F64(col));
        }
        anyhow::bail!("unsupported numeric type for f32 conversion")
    }

    fn get(&self, row: usize) -> Option<f32> {
        match self {
            Self::F32(col) => (!col.is_null(row)).then(|| col.value(row)),
            Self::F64(col) => (!col.is_null(row)).then(|| col.value(row) as f32),
        }
    }
}

#[derive(Clone)]
enum NumI32Col<'a> {
    I32(&'a arrow_array::Int32Array),
    I64(&'a arrow_array::Int64Array),
}

impl<'a> NumI32Col<'a> {
    fn try_new(array: &'a dyn arrow_array::Array) -> anyhow::Result<Self> {
        if let Some(col) = array.as_any().downcast_ref::<arrow_array::Int32Array>() {
            return Ok(Self::I32(col));
        }
        if let Some(col) = array.as_any().downcast_ref::<arrow_array::Int64Array>() {
            return Ok(Self::I64(col));
        }
        anyhow::bail!("unsupported numeric type for i32 conversion")
    }

    fn get(&self, row: usize) -> Option<i32> {
        match self {
            Self::I32(col) => (!col.is_null(row)).then(|| col.value(row)),
            Self::I64(col) => (!col.is_null(row)).then(|| col.value(row) as i32),
        }
    }
}

#[derive(Clone)]
enum NumU8Col<'a> {
    U8(&'a arrow_array::UInt8Array),
}

impl<'a> NumU8Col<'a> {
    fn try_new(array: &'a dyn arrow_array::Array) -> anyhow::Result<Self> {
        if let Some(col) = array.as_any().downcast_ref::<arrow_array::UInt8Array>() {
            return Ok(Self::U8(col));
        }
        anyhow::bail!("unsupported numeric type for u8 conversion")
    }

    fn get(&self, row: usize) -> Option<u8> {
        match self {
            Self::U8(col) => (!col.is_null(row)).then(|| col.value(row)),
        }
    }
}

#[derive(Clone)]
enum NumU64Col<'a> {
    U64(&'a arrow_array::UInt64Array),
}

impl<'a> NumU64Col<'a> {
    fn try_new(array: &'a dyn arrow_array::Array) -> anyhow::Result<Self> {
        if let Some(col) = array.as_any().downcast_ref::<arrow_array::UInt64Array>() {
            return Ok(Self::U64(col));
        }
        anyhow::bail!("unsupported numeric type for u64 conversion")
    }

    fn get(&self, row: usize) -> Option<u64> {
        match self {
            Self::U64(col) => (!col.is_null(row)).then(|| col.value(row)),
        }
    }
}
