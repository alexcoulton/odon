//! Renderer-side cache types for actor-computed image histograms.

#[derive(Debug, Clone)]
pub struct HistogramStats {
    pub min: f32,
    pub q1: f32,
    pub median: f32,
    pub q3: f32,
    pub max: f32,
    pub n: usize,
}

#[derive(Debug, Clone)]
pub struct HistogramResponse {
    pub request_id: u64,
    pub bins: Vec<u32>,
    pub stats: Option<HistogramStats>,
}
