//! Line-sync pulse detection and period inference.
//!
//! Every supported SSTV mode begins each radio line with a 1200 Hz sync pulse
//! (Scottie places it mid-line, which is handled by
//! [`crate::modes::Mode::sync_offset_seconds`]). The *period* of those
//! pulses is highly mode specific, so a run of detected pulses identifies a
//! mode even when the VIS header is missing or damaged.
//!
//! Detection works on the measured [`Trajectory`] rather than raw audio, so
//! it costs one pass over a small model instead of a second pass over the
//! recording.

use crate::dsp::{Trajectory, mean, median};
use crate::modes::SYNC_HZ;

/// Half-width of the frequency window a pulse must fall inside, Hz.
pub const SYNC_TOLERANCE_HZ: f64 = 90.0;

/// One detected sync pulse.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pulse {
    /// Sample position of the pulse's leading edge.
    pub start: f64,
    /// Sample position of the pulse's trailing edge.
    pub end: f64,
    /// Frequency measured inside the pulse, Hz.
    pub hz: f64,
}

impl Pulse {
    /// Pulse duration in samples.
    #[must_use]
    pub fn len(&self) -> f64 {
        self.end - self.start
    }

    /// Whether the pulse has zero length.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() <= 0.0
    }

    /// Centre of the pulse in samples.
    #[must_use]
    pub fn centre(&self) -> f64 {
        (self.start + self.end) * 0.5
    }
}

/// Estimate the receiver frequency offset from the signal itself.
///
/// Sync pulses are the only sustained tones below the image band, so the
/// median of everything measured between 900 Hz and 1400 Hz is dominated by
/// them. The median is used rather than the mean because a single loud
/// transient must not move the estimate.
#[must_use]
pub fn estimate_shift(trajectory: &Trajectory) -> f64 {
    if trajectory.is_empty() {
        return 0.0;
    }
    let low = SYNC_HZ - 300.0;
    let high = SYNC_HZ + 200.0;
    let mut candidates: Vec<f64> = Vec::new();
    for hz in &trajectory.hz {
        if (low..=high).contains(hz) {
            candidates.push(*hz);
        }
    }
    if candidates.len() < 8 {
        // Not enough evidence; nominal is the only defensible answer.
        return 0.0;
    }
    let median = median(&mut candidates);
    // A shift large enough to push sync out of the window is not something
    // this estimator can see, so refuse to report an implausible value.
    (median - SYNC_HZ).clamp(-300.0, 200.0)
}

/// Detect sync pulses, given a frequency offset estimate.
///
/// `min_samples`/`max_samples` bound the acceptable pulse width, which keeps
/// slow sweeps through the sync frequency from being mistaken for pulses.
#[must_use]
pub fn detect(trajectory: &Trajectory, shift_hz: f64) -> Vec<Pulse> {
    let mut out = Vec::new();
    if trajectory.is_empty() || trajectory.xs.len() < 2 {
        return out;
    }
    let step = trajectory.xs[1] - trajectory.xs[0];
    if step <= 0.0 {
        return out;
    }
    let target = SYNC_HZ + shift_hz;
    let rate = f64::from(trajectory.rate());
    // Bounds on a plausible sync pulse. The upper bound is generous relative
    // to the longest real sync (PD, 20 ms) but deliberately below the width of
    // a VIS stop pulse merged with the following line's sync (about 39 ms), so
    // that merged run is split rather than accepted as one pulse.
    let min_samples = (0.0015 * rate).max(step);
    let max_samples = 0.035 * rate;

    // The narrowest plausible sync across the supported modes is Martin 1/2 at
    // 4.862 ms; a merged run is split using a width near that.
    let typical = 0.005 * rate;

    let mut run_start: Option<usize> = None;
    for (index, hz) in trajectory.hz.iter().enumerate() {
        let inside = (hz - target).abs() <= SYNC_TOLERANCE_HZ;
        match (inside, run_start) {
            (true, None) => run_start = Some(index),
            (false, Some(start)) => {
                out.extend(make_pulses(
                    trajectory,
                    start,
                    index,
                    min_samples,
                    max_samples,
                    typical,
                ));
                run_start = None;
            }
            _ => {}
        }
    }
    if let Some(start) = run_start {
        out.extend(make_pulses(
            trajectory,
            start,
            trajectory.hz.len(),
            min_samples,
            max_samples,
            typical,
        ));
    }
    out.sort_by(|a, b| a.start.total_cmp(&b.start));
    out
}

/// Build pulses from a run of sync-frequency samples.
///
/// # Over-long runs
///
/// A run can be longer than any real sync pulse. The common cause is a VIS
/// stop pulse sitting immediately before the first image line's sync: both are
/// 1200 Hz, so the detector sees one unbroken run even though two distinct
/// tones were transmitted.
///
/// Such a run is split by taking its **trailing** window of a plausible pulse
/// width as an additional pulse. The trailing position is the right one
/// because whatever precedes a line's sync is the previous element — the VIS
/// stop pulse, or the tail of the previous line — and the sync itself is last.
/// Discarding the run instead would lose the first line of every
/// VIS-prefixed transmission, and with it the sync chain used to identify a
/// mode when the header is missing entirely.
fn make_pulses(
    trajectory: &Trajectory,
    start: usize,
    end: usize,
    min_samples: f64,
    max_samples: f64,
    typical: f64,
) -> Vec<Pulse> {
    let start_sample = trajectory.xs[start];
    let full_end = trajectory.xs[end - 1];
    let len = full_end - start_sample;
    if len < min_samples {
        return Vec::new();
    }

    let mut out = Vec::new();
    if len <= max_samples {
        if let Some(pulse) = pulse_from(trajectory, start, end, start_sample, full_end) {
            out.push(pulse);
        }
        return out;
    }

    // The run is too long to be one pulse. Emit the *leading* window as one
    // candidate and the *trailing* window as another: a merged run has a real
    // pulse at one end, and which end depends on what preceded it. The
    // ranking later keeps whichever fits a consistent line period, so offering
    // both costs nothing and cannot invent a mode.
    let width = typical.clamp(min_samples, max_samples);
    let lead_end = start_sample + width;
    if let Some(pulse) = pulse_from(trajectory, start, end, start_sample, lead_end) {
        out.push(pulse);
    }
    let tail_start = (full_end - width).max(start_sample);
    if tail_start > start_sample {
        let trailing_hz: Vec<f64> = trajectory
            .xs
            .iter()
            .enumerate()
            .filter(|(_, x)| **x >= tail_start && **x <= full_end)
            .map(|(i, _)| trajectory.hz[i])
            .collect();
        if trailing_hz.len() >= 2 {
            let mut hz = trailing_hz;
            out.push(Pulse {
                start: tail_start,
                end: full_end,
                hz: median(&mut hz),
            });
        }
    }
    out
}

/// Build one pulse spanning the samples of a run that fall inside
/// `[from, to]`.
fn pulse_from(
    trajectory: &Trajectory,
    run_start: usize,
    run_end: usize,
    from: f64,
    to: f64,
) -> Option<Pulse> {
    let mut hz: Vec<f64> = trajectory
        .xs
        .iter()
        .enumerate()
        .take(run_end)
        .skip(run_start)
        .filter(|(_, x)| **x >= from && **x <= to)
        .map(|(i, _)| trajectory.hz[i])
        .collect();
    if hz.len() < 2 {
        return None;
    }
    Some(Pulse {
        start: from,
        end: to,
        hz: median(&mut hz),
    })
}

/// A run of pulses whose spacing matches one line period.
#[derive(Debug, Clone, PartialEq)]
pub struct Chain {
    /// Indices of the matched pulses, ascending.
    pub indices: Vec<usize>,
    /// Sample span covered by the chain.
    pub span: f64,
    /// Root-mean-square timing error as a fraction of the line period.
    pub timing_error: f64,
    /// Best-fit scale applied to the nominal line period.
    pub clock_rate: f64,
    /// Mean frequency offset of the matched pulses, Hz.
    pub shift_hz: f64,
    /// Fraction of the detected pulses between this chain's first and last
    /// member that the chain actually matched, including pulses it skipped.
    ///
    /// Measured against every pulse in the range rather than only against the
    /// ones the chain considered, because a chain can otherwise be perfectly
    /// dense while describing the wrong mode: a 300 ms period matches every
    /// *second* pulse of a 150 ms transmission, producing a long, regular
    /// chain for a mode that is not present. Counting the skipped pulses is
    /// what exposes that.
    pub density: f64,
}

impl Chain {
    /// Number of matched pulses.
    #[must_use]
    pub fn matched(&self) -> usize {
        self.indices.len()
    }
}

/// Fit a chain of pulses starting at `start` with the given nominal period.
///
/// Pulses are matched greedily: from the last accepted pulse, the next pulse
/// must land within `tolerance` of the predicted position. Matching stops at
/// the first gap, which is what makes a truncated recording simply produce a
/// shorter chain instead of a wrong one.
#[must_use]
pub fn chain(pulses: &[Pulse], start: usize, nominal_period: f64, tolerance: f64) -> Chain {
    let mut indices = vec![start];
    let mut expected = pulses[start].start + nominal_period;

    for (index, pulse) in pulses.iter().enumerate().skip(start + 1) {
        let error = pulse.start - expected;
        if error > tolerance {
            // The nearest pulse is already past the window; the chain broke.
            break;
        }
        if error.abs() <= tolerance {
            indices.push(index);
            expected = pulse.start + nominal_period;
        }
    }

    refine(pulses, &indices, nominal_period)
}

/// Least-squares refinement of a matched chain.
///
/// Fits `position = offset + index * period` so that both the start position
/// and the clock error come from all matched pulses rather than just the
/// first and last. This is what recovers a transmitter running slightly fast
/// or slow.
fn refine(pulses: &[Pulse], indices: &[usize], nominal_period: f64) -> Chain {
    // Work with position relative to the first matched pulse so the fit stays
    // well conditioned regardless of where in the file it sits. Matching is
    // greedy and contiguous, so the ordinal within the chain is the
    // independent variable.
    let origin = pulses[indices[0]].start;
    let n = indices.len() as f64;
    let mean_index = (n - 1.0) * 0.5;
    let mean_position = indices
        .iter()
        .map(|i| pulses[*i].start - origin)
        .sum::<f64>()
        / n;

    let mut covariance = 0.0_f64;
    let mut variance = 0.0_f64;
    for (ordinal, index) in indices.iter().enumerate() {
        #[allow(clippy::cast_precision_loss)]
        let x = ordinal as f64;
        let y = pulses[*index].start - origin;
        covariance += (x - mean_index) * (y - mean_position);
        variance += (x - mean_index) * (x - mean_index);
    }

    let period = if variance > f64::EPSILON && n >= 3.0 {
        covariance / variance
    } else {
        nominal_period
    };
    // Guard against a degenerate fit collapsing the clock estimate.
    let clock_rate = if nominal_period > 0.0 && period.is_finite() {
        (period / nominal_period).clamp(0.90, 1.10)
    } else {
        1.0
    };

    let span = pulses[*indices.last().unwrap_or(&indices[0])].start - origin;
    let timing_error = if n > 1.0 {
        let unique: std::collections::BTreeSet<usize> = indices.iter().copied().collect();
        #[allow(clippy::cast_precision_loss)]
        let nominal_span = (unique.len().saturating_sub(1)) as f64 * nominal_period;
        if nominal_span > 0.0 {
            ((span - nominal_span) / nominal_span).abs()
        } else {
            0.0
        }
    } else {
        1.0
    };

    let mut shifts: Vec<f64> = indices.iter().map(|i| pulses[*i].hz - SYNC_HZ).collect();
    let shift_hz = median(&mut shifts);

    // Density counts every detected pulse between the chain's first and last
    // member, including those the chain skipped.
    let first = indices[0];
    let last = *indices.last().unwrap_or(&indices[0]);
    let spanned = last.saturating_sub(first) + 1;
    #[allow(clippy::cast_precision_loss)]
    let density = if spanned > 0 {
        indices.len() as f64 / spanned as f64
    } else {
        0.0
    };

    Chain {
        indices: indices.to_vec(),
        span,
        timing_error,
        clock_rate,
        shift_hz,
        density,
    }
}

/// Score a chain in `0.0..=1.0`.
///
/// Both *how many* pulses matched and *how regular* their spacing was
/// contribute. A long chain with a small timing error is strong evidence;
/// four pulses spaced roughly right are not.
#[must_use]
pub fn chain_score(chain: &Chain) -> f64 {
    if chain.matched() < 3 {
        return 0.0;
    }
    // Sixteen matched pulses is about two seconds of a fast mode and
    // corresponds to a whole image in a slow one; beyond that, extra
    // confidence comes from timing regularity instead.
    let length = ((chain.matched() as f64 - 2.0) / 14.0).clamp(0.0, 1.0);
    let regularity = (1.0 - chain.timing_error * 25.0).clamp(0.0, 1.0);
    // Density separates a chain that accounts for every pulse it spans from
    // one that matches a half or a quarter of them, which is what a period
    // dividing the real one looks like.
    let density = chain.density.clamp(0.0, 1.0);
    (0.45 * length + 0.25 * regularity + 0.30 * density).clamp(0.0, 1.0)
}

/// Mean absolute deviation of pulse intervals in a span, in samples.
#[must_use]
pub fn interval_spread(pulses: &[Pulse], from: usize, to: usize) -> f64 {
    if to <= from + 1 {
        return f64::INFINITY;
    }
    let mut intervals: Vec<f64> = Vec::with_capacity(to - from);
    for window in pulses[from..=to].windows(2) {
        intervals.push(window[1].start - window[0].start);
    }
    let m = mean(&intervals);
    let mut deviations: Vec<f64> = intervals.iter().map(|i| (i - m).abs()).collect();
    if deviations.is_empty() {
        return f64::INFINITY;
    }
    median(&mut deviations)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pulse(start: f64, len: f64, hz: f64) -> Pulse {
        Pulse {
            start,
            end: start + len,
            hz,
        }
    }

    #[test]
    fn chain_matches_a_perfect_series() {
        let pulses: Vec<Pulse> = (0..20)
            .map(|i| pulse(i as f64 * 100.0, 20.0, 1200.0))
            .collect();
        let chain = chain(&pulses, 0, 100.0, 4.0);
        assert_eq!(chain.matched(), 20);
        assert!(chain.timing_error < 1e-9);
        assert!((chain.clock_rate - 1.0).abs() < 1e-9);
        assert!((chain.shift_hz).abs() < 1e-9);
        assert!(chain_score(&chain) > 0.9);
    }

    #[test]
    fn chain_recovers_a_clock_error() {
        // Pulses 2% further apart than nominal.
        let period = 102.0;
        let pulses: Vec<Pulse> = (0..40)
            .map(|i| pulse(i as f64 * period, 20.0, 1200.0))
            .collect();
        // A tight tolerance is needed to follow the drift.
        let chain = chain(&pulses, 0, 100.0, 6.0);
        assert!(chain.matched() > 10, "matched {}", chain.matched());
        assert!(
            (chain.clock_rate - 1.02).abs() < 0.005,
            "clock {}",
            chain.clock_rate
        );
    }

    #[test]
    fn chain_stops_at_a_gap() {
        let mut pulses: Vec<Pulse> = (0..10)
            .map(|i| pulse(i as f64 * 100.0, 20.0, 1200.0))
            .collect();
        // Insert a missing pulse by shifting everything after index 5.
        for p in pulses.iter_mut().skip(6) {
            p.start += 400.0;
            p.end += 400.0;
        }
        let chain = chain(&pulses, 0, 100.0, 4.0);
        assert_eq!(chain.matched(), 6, "should stop at the gap");
    }

    #[test]
    fn chain_needs_at_least_three_pulses_to_score() {
        let pulses: Vec<Pulse> = (0..2)
            .map(|i| pulse(i as f64 * 100.0, 20.0, 1200.0))
            .collect();
        let chain = chain(&pulses, 0, 100.0, 4.0);
        assert_eq!(chain_score(&chain), 0.0);
    }

    #[test]
    fn chain_of_one_is_handled() {
        let pulses = vec![pulse(0.0, 20.0, 1200.0)];
        let chain = chain(&pulses, 0, 100.0, 4.0);
        assert_eq!(chain.matched(), 1);
        assert_eq!(chain.clock_rate, 1.0);
        assert_eq!(chain_score(&chain), 0.0);
    }

    #[test]
    fn detect_finds_pulses_in_a_dirty_signal() {
        // Build a trajectory by hand: pulses every 100 samples with a +40 Hz
        // offset, plus noise excursions elsewhere.
        let rate = 22_050;
        let step = (0.001 * f64::from(rate)) as usize;
        let mut xs = Vec::new();
        let mut hz = Vec::new();
        for i in 0..4000 {
            let x = (i * step) as f64;
            xs.push(x);
            // A pulse every 150 ms occupying the first 9 ms.
            let phase = x % (0.150 * f64::from(rate));
            hz.push(if phase < 0.009 * f64::from(rate) {
                1240.0
            } else {
                1800.0
            });
        }
        let trajectory = Trajectory::from_model(xs, hz, rate);
        let shift = estimate_shift(&trajectory);
        assert!((shift - 40.0).abs() < 20.0, "estimated shift {shift}");
        let pulses = detect(&trajectory, shift);
        assert!(
            pulses.len() >= 20 && pulses.len() <= 30,
            "found {} pulses",
            pulses.len()
        );
        let chain = chain(&pulses, 0, 0.150 * f64::from(rate), 0.004 * f64::from(rate));
        assert!(chain.matched() >= 20, "matched {}", chain.matched());
        assert!(chain_score(&chain) > 0.8);
    }

    #[test]
    fn detect_ignores_a_slow_sweep_through_the_sync_frequency() {
        // A sweep that merely passes through 1200 Hz is not a pulse.
        let rate = 22_050;
        let step = (0.001 * f64::from(rate)) as usize;
        let count = 4000;
        let mut xs = Vec::new();
        let mut hz = Vec::new();
        for i in 0..count {
            let t = i as f64 / count as f64;
            xs.push((i * step) as f64);
            hz.push(1500.0 + 800.0 * t);
        }
        let trajectory = Trajectory::from_model(xs, hz, rate);
        let pulses = detect(&trajectory, 0.0);
        assert!(pulses.is_empty(), "sweep produced {} pulses", pulses.len());
    }

    #[test]
    fn estimate_shift_defaults_to_zero_without_evidence() {
        let rate = 22_050;
        let trajectory =
            Trajectory::from_model(vec![0.0, 1.0, 2.0], vec![2000.0, 2100.0, 2200.0], rate);
        assert_eq!(estimate_shift(&trajectory), 0.0);
    }

    #[test]
    fn pulse_accessors_are_sane() {
        let p = pulse(10.0, 5.0, 1200.0);
        assert!((p.len() - 5.0).abs() < 1e-9);
        assert!(!p.is_empty());
        assert!((p.centre() - 12.5).abs() < 1e-9);
        let empty = pulse(10.0, 0.0, 1200.0);
        assert!(empty.is_empty());
    }

    #[test]
    fn a_submultiple_period_is_penalised_by_density() {
        // The false positive this guards against: a mode whose line period is
        // an exact multiple of the real one matches every second pulse,
        // producing a long, perfectly regular chain for a transmission that is
        // not present.
        let pulses: Vec<Pulse> = (0..60)
            .map(|i| pulse(i as f64 * 100.0, 20.0, 1200.0))
            .collect();

        // The true period: every pulse matches.
        let truth = chain(&pulses, 0, 100.0, 4.0);
        assert_eq!(truth.matched(), 60);
        assert!((truth.density - 1.0).abs() < 1e-9, "{}", truth.density);

        // Double the period: the chain is just as regular but skips half.
        let half = chain(&pulses, 0, 200.0, 4.0);
        assert_eq!(half.matched(), 30);
        assert!(
            (half.density - 0.5).abs() < 0.02,
            "density should expose the skipped pulses, got {}",
            half.density
        );
        assert!(
            chain_score(&truth) > chain_score(&half) + 0.1,
            "the true period must outrank its multiple: {} vs {}",
            chain_score(&truth),
            chain_score(&half)
        );

        // Triple: a third of the pulses are matched.
        let third = chain(&pulses, 0, 300.0, 4.0);
        assert!(
            (third.density - 1.0 / 3.0).abs() < 0.02,
            "density {}",
            third.density
        );
        assert!(chain_score(&truth) > chain_score(&third));
    }

    #[test]
    fn interval_spread_measures_irregularity() {
        let regular: Vec<Pulse> = (0..10)
            .map(|i| pulse(i as f64 * 100.0, 20.0, 1200.0))
            .collect();
        assert!(interval_spread(&regular, 0, 9) < 1e-9);

        // `interval_spread` is a median absolute deviation, so it tolerates a
        // single disturbed interval by design: that robustness is what stops
        // one missed pulse from invalidating a whole chain. A genuinely
        // irregular series must still register.
        let mut irregular = regular.clone();
        for (index, p) in irregular.iter_mut().enumerate() {
            #[allow(clippy::cast_precision_loss)]
            let wobble = (index as f64 * 7.0).sin() * 30.0;
            p.start += wobble;
            p.end += wobble;
        }
        assert!(
            interval_spread(&irregular, 0, 9) > 5.0,
            "irregular spread was {}",
            interval_spread(&irregular, 0, 9)
        );

        // A single shifted pulse must NOT dominate the estimate.
        let mut one_off = regular.clone();
        one_off[5].start += 30.0;
        assert!(interval_spread(&one_off, 0, 9) < 30.0);

        assert!(interval_spread(&regular, 0, 0).is_infinite());
    }
}
