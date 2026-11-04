//! Rolling frame-pacing statistics for renderer diagnostics.
//!
//! The interval between frame starts is the only timing that covers everything a user can feel:
//! CPU work, GL submission, GPU back-pressure, vsync, and paging. CPU-only timers miss most of it
//! because GL commands are queued and execute after they appear to return.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

const WINDOW_FRAMES: usize = 240;
/// egui only repaints on demand, so a longer gap means the viewer was idle rather than slow.
const IDLE_GAP: Duration = Duration::from_millis(500);
const MISSED_60_HZ_MS: f32 = 1_000.0 / 60.0 * 1.5;
const MISSED_30_HZ_MS: f32 = 1_000.0 / 30.0 * 1.5;

#[derive(Debug, Default)]
pub(crate) struct FramePacing {
    intervals_ms: VecDeque<f32>,
    update_ms: VecDeque<f32>,
    frame_started_at: Option<Instant>,
    frames: u64,
    idle_gaps: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, serde::Serialize)]
pub(crate) struct FramePacingSnapshot {
    pub frames: u64,
    pub idle_gaps: u64,
    pub window_samples: usize,
    pub interval_mean_ms: f32,
    pub interval_p50_ms: f32,
    pub interval_p95_ms: f32,
    pub interval_p99_ms: f32,
    pub interval_max_ms: f32,
    pub update_p50_ms: f32,
    pub update_p95_ms: f32,
    /// Frames slower than 1.5 display intervals at 60 Hz, which show up as visible judder.
    pub over_25_ms: usize,
    pub over_50_ms: usize,
}

impl FramePacing {
    pub(crate) fn begin_frame(&mut self, now: Instant) {
        if let Some(previous) = self.frame_started_at {
            let interval = now.saturating_duration_since(previous);
            if interval >= IDLE_GAP {
                self.idle_gaps = self.idle_gaps.saturating_add(1);
            } else {
                push_bounded(&mut self.intervals_ms, interval.as_secs_f32() * 1_000.0);
            }
        }
        self.frame_started_at = Some(now);
        self.frames = self.frames.saturating_add(1);
    }

    pub(crate) fn finish_update(&mut self, now: Instant) {
        if let Some(started_at) = self.frame_started_at {
            let update = now.saturating_duration_since(started_at);
            push_bounded(&mut self.update_ms, update.as_secs_f32() * 1_000.0);
        }
    }

    pub(crate) fn snapshot(&self) -> FramePacingSnapshot {
        let intervals = sorted(&self.intervals_ms);
        let updates = sorted(&self.update_ms);
        let mean = if intervals.is_empty() {
            0.0
        } else {
            intervals.iter().sum::<f32>() / intervals.len() as f32
        };
        FramePacingSnapshot {
            frames: self.frames,
            idle_gaps: self.idle_gaps,
            window_samples: intervals.len(),
            interval_mean_ms: mean,
            interval_p50_ms: percentile(&intervals, 0.50),
            interval_p95_ms: percentile(&intervals, 0.95),
            interval_p99_ms: percentile(&intervals, 0.99),
            interval_max_ms: intervals.last().copied().unwrap_or(0.0),
            update_p50_ms: percentile(&updates, 0.50),
            update_p95_ms: percentile(&updates, 0.95),
            over_25_ms: intervals
                .iter()
                .filter(|interval| **interval > MISSED_60_HZ_MS)
                .count(),
            over_50_ms: intervals
                .iter()
                .filter(|interval| **interval > MISSED_30_HZ_MS)
                .count(),
        }
    }
}

fn push_bounded(samples: &mut VecDeque<f32>, value: f32) {
    if samples.len() == WINDOW_FRAMES {
        samples.pop_front();
    }
    samples.push_back(value);
}

fn sorted(samples: &VecDeque<f32>) -> Vec<f32> {
    let mut sorted = samples.iter().copied().collect::<Vec<_>>();
    sorted.sort_unstable_by(f32::total_cmp);
    sorted
}

fn percentile(sorted: &[f32], fraction: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = (sorted.len() as f32 * fraction).ceil() as usize;
    sorted[rank.clamp(1, sorted.len()) - 1]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frames(pacing: &mut FramePacing, start: Instant, intervals_ms: &[u64]) -> Instant {
        let mut now = start;
        pacing.begin_frame(now);
        for interval in intervals_ms {
            now += Duration::from_millis(*interval);
            pacing.begin_frame(now);
        }
        now
    }

    #[test]
    fn summarises_intervals_and_counts_missed_frames() {
        let mut pacing = FramePacing::default();
        frames(&mut pacing, Instant::now(), &[16, 16, 16, 30, 60]);

        let snapshot = pacing.snapshot();

        assert_eq!(snapshot.frames, 6);
        assert_eq!(snapshot.window_samples, 5);
        assert_eq!(snapshot.over_25_ms, 2);
        assert_eq!(snapshot.over_50_ms, 1);
        assert!((snapshot.interval_p50_ms - 16.0).abs() < 0.01);
        assert!((snapshot.interval_max_ms - 60.0).abs() < 0.01);
    }

    #[test]
    fn idle_gaps_are_counted_without_polluting_the_timing_window() {
        let mut pacing = FramePacing::default();
        frames(&mut pacing, Instant::now(), &[16, 5_000, 16]);

        let snapshot = pacing.snapshot();

        assert_eq!(snapshot.idle_gaps, 1);
        assert_eq!(snapshot.window_samples, 2);
        assert!(snapshot.interval_max_ms < 17.0);
    }

    #[test]
    fn window_keeps_only_the_most_recent_frames() {
        let mut pacing = FramePacing::default();
        let slow = vec![100; WINDOW_FRAMES];
        let now = frames(&mut pacing, Instant::now(), &slow);
        frames(&mut pacing, now, &vec![10; WINDOW_FRAMES]);

        let snapshot = pacing.snapshot();

        assert_eq!(snapshot.window_samples, WINDOW_FRAMES);
        assert!(snapshot.interval_max_ms < 11.0);
    }

    #[test]
    fn update_time_is_measured_from_the_current_frame_start() {
        let mut pacing = FramePacing::default();
        let start = Instant::now();
        pacing.begin_frame(start);
        pacing.finish_update(start + Duration::from_millis(7));

        assert!((pacing.snapshot().update_p50_ms - 7.0).abs() < 0.01);
    }

    #[test]
    fn empty_window_reports_zeroes() {
        assert_eq!(
            FramePacing::default().snapshot(),
            FramePacingSnapshot::default()
        );
    }
}
