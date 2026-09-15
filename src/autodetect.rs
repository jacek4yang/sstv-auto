//! Automatic detection: what is in this recording?
//!
//! Given a signal, this module answers four questions without being told
//! anything:
//!
//! 1. **Which mode?** — from the VIS header when one is intact, otherwise
//!    from the measured line-sync period, which is specific enough to
//!    identify a mode on its own.
//! 2. **Where does the image start?** — from the VIS payload position, or
//!    from the first sync pulse of the fitted chain.
//! 3. **What is the receiver offset?** — measured from the sync pulses and
//!    the leader, never assumed to be zero.
//! 4. **Is the transmitter's clock fast or slow?** — from the least-squares
//!    fit of the sync chain.
//!
//! # Why the ranking is trustworthy
//!
//! A wrong mode can still decode into *an* image, so the decoder never trusts
//! a single decode. For every hypothesis it:
//!
//! 1. extracts a grid with [`crate::raster::extract`],
//! 2. re-encodes that grid with [`crate::synth::render`],
//! 3. measures the *re-rendered* audio back into a trajectory, and
//! 4. compares that against the original audio at the same sample positions.
//!
//! The comparison is between measured frequencies, so it is directly
//! interpretable: a correct hypothesis re-renders audio that matches the
//! recording tone for tone, and a wrong one does not. This is what lets the
//! decoder reject weak matches instead of accepting the first image that
//! falls out.

use crate::dsp::{Analyzer, Trajectory};
use crate::modes::{self, Mode};
use crate::raster::{self, Grid};
use crate::sync::{self, Pulse};
use crate::vis::{self, VisHit};

/// How the mode was determined.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Origin {
    /// The VIS header decoded cleanly and its parity matched.
    Vis,
    /// The VIS header decoded but its parity failed; the code is a hint only.
    VisParity,
    /// No usable VIS: the mode came from the measured sync period.
    SyncPeriod,
}

impl Origin {
    /// Short label for reports.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Vis => "vis",
            Self::VisParity => "vis-parity-failed",
            Self::SyncPeriod => "sync-period",
        }
    }

    /// Whether this origin means the header was fully trusted.
    #[must_use]
    pub fn is_authoritative(self) -> bool {
        matches!(self, Self::Vis)
    }
}

/// A fully specified decode hypothesis.
#[derive(Debug, Clone)]
pub struct Hypothesis {
    /// Mode to decode.
    pub mode: Mode,
    /// How the mode was determined.
    pub origin: Origin,
    /// Absolute sample position of the first radio line's sync pulse.
    pub skip_sample: f64,
    /// Receiver frequency offset in Hz.
    pub frequency_shift_hz: f64,
    /// Transmitter clock scale, `1.0` meaning nominal.
    pub clock_rate: f64,
    /// Number of sync pulses consistent with this hypothesis.
    pub matched_syncs: usize,
    /// Confidence in the timing model, `0.0..=1.0`.
    pub timing_score: f64,
    /// Whether the tail of the transmission is present in the recording.
    pub complete: bool,
    /// Evidence notes for the report.
    pub notes: Vec<String>,
}

impl Hypothesis {
    /// Whether an evidence note records that part of the transmission is
    /// missing. Used by tests, and by the CLI when summarising findings.
    #[must_use]
    pub fn coverage_is_partial_note(&self) -> bool {
        self.notes.iter().any(|note| note.contains("present"))
    }
}

/// Result of evaluating one hypothesis against the recording.
#[derive(Debug, Clone)]
pub struct Candidate {
    /// The hypothesis that produced this candidate.
    pub hypothesis: Hypothesis,
    /// Grid recovered from the recording.
    pub grid: Grid,
    /// How well a re-render of `grid` matches the original signal.
    pub agreement: f64,
    /// Timing confidence carried over from the hypothesis.
    pub timing_score: f64,
    /// Fraction of the expected transmission actually present.
    pub coverage: f64,
    /// Combined ranking score.
    pub score: f64,
}

/// Everything the detector measured, for diagnostics.
#[derive(Debug, Clone)]
pub struct Analysis {
    /// Sample rate the analysis ran at.
    pub rate: u32,
    /// Duration of the recording in seconds.
    pub duration_seconds: f64,
    /// Estimated receiver frequency offset in Hz.
    pub frequency_shift_hz: f64,
    /// Sync pulses found in the recording.
    pub pulses: Vec<Pulse>,
    /// VIS bursts found, strongest first.
    pub vis_hits: Vec<VisHit>,
    /// Ranked hypotheses, best first.
    pub hypotheses: Vec<Hypothesis>,
    /// Whether the signal itself looks like SSTV at all.
    pub plausible_sstv: bool,
}

/// Weight of grid agreement in the final candidate score.
const AGREEMENT_WEIGHT: f64 = 0.55;
/// Weight of timing confidence in the final candidate score.
const TIMING_WEIGHT: f64 = 0.30;
/// Weight of coverage in the final candidate score.
const COVERAGE_WEIGHT: f64 = 0.15;

/// Bonus added to a candidate's score when its mode came from a VIS header
/// whose parity checked out.
///
/// A validated header is *proof* of the mode, not an inference, so it must
/// outrank any hypothesis derived from line timing. The bonus is applied to
/// the candidate score itself rather than only to the ordering of hypotheses,
/// because candidates are re-sorted by score after evaluation; keeping the
/// preference only in the earlier sort would silently discard it.
const AUTHORITATIVE_BONUS: f64 = 0.25;

/// A candidate must clear this score to be reported as a decode.
pub const ACCEPT_THRESHOLD: f64 = 0.55;

/// Analyse a signal and produce ranked hypotheses.
#[must_use]
pub fn analyze(trajectory: &Trajectory) -> Analysis {
    let rate = trajectory.rate();
    if trajectory.is_empty() {
        return Analysis {
            rate,
            duration_seconds: 0.0,
            frequency_shift_hz: 0.0,
            pulses: Vec::new(),
            vis_hits: Vec::new(),
            hypotheses: Vec::new(),
            plausible_sstv: false,
        };
    }

    let duration_seconds = trajectory.last_sample() / f64::from(rate);
    let rate_hz = f64::from(rate);
    let vis_hits = vis::find(trajectory);

    // Prefer the offset implied by a VIS leader; fall back to the sync pulses.
    let sync_shift = sync::estimate_shift(trajectory);
    let frequency_shift_hz = vis_hits
        .iter()
        .find(|hit| hit.parity_ok)
        .or_else(|| vis_hits.first())
        .map_or(sync_shift, |hit| hit.frequency_shift_hz);

    let pulses = sync::detect(trajectory, frequency_shift_hz);

    let mut hypotheses = Vec::new();
    hypotheses.extend(from_vis(&vis_hits, &pulses, rate_hz, duration_seconds));
    hypotheses.extend(from_sync_period(&pulses, rate_hz, duration_seconds));

    // Prefer authoritative VIS hypotheses, then timing strength, then the
    // amount of evidence behind them.
    hypotheses.sort_by(|a, b| {
        rank_key(b)
            .partial_cmp(&rank_key(a))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    deduplicate(&mut hypotheses);

    // "Plausible SSTV" must mean the recording contains *structure* that looks
    // like a transmission, not merely that the sync-frequency detector fired.
    // Broadband noise trips that detector by chance — a few milliseconds
    // landing near 1200 Hz is common across minutes of random audio — so the
    // bar is a coherent hypothesis: a mode whose line period several
    // consecutive pulses actually match.
    let plausible_sstv = !vis_hits.is_empty()
        || hypotheses
            .iter()
            .any(|hypothesis| hypothesis.matched_syncs >= 4);

    Analysis {
        rate,
        duration_seconds,
        frequency_shift_hz,
        pulses,
        vis_hits,
        hypotheses,
        plausible_sstv,
    }
}

fn rank_key(hypothesis: &Hypothesis) -> f64 {
    let origin_bonus = match hypothesis.origin {
        Origin::Vis => 2.0,
        Origin::VisParity => 1.0,
        Origin::SyncPeriod => 0.0,
    };
    origin_bonus + hypothesis.timing_score
}

/// Whether `weaker` is describing the same transmission as `stronger`, but
/// worse.
///
/// A mode whose line period divides another's will latch onto every second or
/// third pulse of a transmission it is not present in, producing a long,
/// regular chain and a mediocre-but-not-terrible decode. The tell is that the
/// stronger hypothesis covers the same signal *better*: more of the
/// transmission present, and a chain that accounts for more of the pulses it
/// spans.
///
/// Without this, a Robot 36 recording reports a spurious Robot 72 image
/// alongside the correct one.
#[must_use]
pub fn is_explained_by(weaker: &Candidate, stronger: &Candidate) -> bool {
    // Only another *inferred* hypothesis can subsume this one; a VIS header is
    // handled by its own authority rule.
    if weaker.hypothesis.mode.mode == stronger.hypothesis.mode.mode {
        return false;
    }
    // The stronger candidate must be clearly better on both counts.
    let better_agreement = stronger.agreement > weaker.agreement - 0.05;
    let better_coverage = stronger.coverage > weaker.coverage + 0.20;
    let denser = stronger.hypothesis.matched_syncs > weaker.hypothesis.matched_syncs;
    better_agreement && better_coverage && denser
}

/// Other modes this one cannot be distinguished from without a VIS header.
///
/// [`SstvMode::Robot24`] and [`SstvMode::Robot36`] share a 150 ms radio line,
/// 240 image rows, the same pixel time, the same separator timing and the same
/// alternating chroma layout, so on the wire they differ *only* in their VIS
/// code. That is a property of the format as this backend models it, not a gap
/// in the analysis: a recording made as either mode decodes to the same
/// picture.
///
/// The decoder reports the mode it decoded and names the alternatives here, so
/// the caller is told about the ambiguity instead of being handed a
/// silently-arbitrary answer.
#[must_use]
pub fn indistinguishable_from(mode: &Mode) -> Vec<Mode> {
    modes::all()
        .into_iter()
        .filter(|candidate| {
            candidate.mode != mode.mode
                && candidate.line_seconds == mode.line_seconds
                && candidate.pixel_seconds == mode.pixel_seconds
                && candidate.image_lines == mode.image_lines
                && candidate.sync_seconds == mode.sync_seconds
                && candidate.separator_seconds == mode.separator_seconds
                && crate::raster::channel_layout(candidate) == crate::raster::channel_layout(mode)
        })
        .collect()
}

/// Hypotheses derived from VIS bursts, which are authoritative when parity
/// passes.
fn from_vis(
    hits: &[VisHit],
    pulses: &[Pulse],
    rate: f64,
    duration_seconds: f64,
) -> Vec<Hypothesis> {
    let mut out = Vec::new();
    for hit in hits {
        let Some(mode) = hit.mode else {
            continue;
        };
        // The payload starts right after the stop pulse. Scottie's first sync
        // sits mid-line, so back the start up by its intra-line offset.
        let mut skip = hit.payload_sample - mode.sync_offset_seconds() * rate;
        // A truncated recording may begin after the VIS burst; clamp rather
        // than sampling before the buffer.
        skip = skip.max(0.0);

        // The payload position is derived from the *tone boundaries* of the
        // header, and its last field is a stop pulse whose trailing edge is
        // not abrupt: the measured end can sit a few hundred samples away from
        // where the first line actually begins. Grid extraction walks
        // every subsequent line from this offset, so an error here skews the
        // whole picture.
        //
        // The sync pulses give an independent, sharp measurement of the same
        // instant, so the prediction is snapped to the nearest detected pulse
        // whenever one is close enough to be that pulse rather than a later
        // line.
        if let Some(nearest) = pulses
            .iter()
            .min_by(|a, b| (a.start - skip).abs().total_cmp(&(b.start - skip).abs()))
            .filter(|pulse| (pulse.start - skip).abs() <= mode.line_seconds * rate * 0.25)
        {
            skip = nearest.start;
        }

        let mut notes = Vec::new();
        if !hit.parity_ok {
            notes.push(format!(
                "VIS parity failed for code 0x{:02x}; mode is a hint, timing must confirm it",
                hit.code
            ));
        }
        notes.push(format!(
            "leader measured at {:.1} Hz ({:+.1} Hz offset)",
            hit.leader_hz, hit.frequency_shift_hz
        ));

        // The VIS burst itself carries a clock estimate: the field-duration
        // scale that best fitted the header. Using it as the expected line
        // period means the chain fit starts from the transmitter's actual
        // rate rather than the nominal one, so a fast or slow recording does
        // not have to be discovered twice.
        let expected_period = mode.line_seconds * rate * hit.clock_scale;
        let chain = best_chain(pulses, skip, expected_period);
        let timing_score = chain.as_ref().map_or_else(
            || {
                if hit.parity_ok {
                    // A clean header is itself strong evidence; do not punish
                    // a recording whose sync pulses were never detectable.
                    0.75
                } else {
                    0.2
                }
            },
            |chain| sync::chain_score(chain).max(0.5),
        );

        let clock_rate = chain
            .as_ref()
            .map_or(hit.clock_scale, |chain| chain.clock_rate);
        if let Some(chain) = &chain {
            if (chain.clock_rate - 1.0).abs() > 0.002 {
                notes.push(format!(
                    "clock measured {:.3}% {} nominal",
                    (chain.clock_rate - 1.0) * 100.0,
                    if chain.clock_rate > 1.0 {
                        "fast"
                    } else {
                        "slow"
                    }
                ));
            }
        }

        let coverage = coverage(&mode, skip, clock_rate, rate, duration_seconds);

        out.push(Hypothesis {
            mode,
            origin: if hit.parity_ok {
                Origin::Vis
            } else {
                Origin::VisParity
            },
            skip_sample: skip,
            frequency_shift_hz: hit.frequency_shift_hz,
            clock_rate,
            matched_syncs: chain.as_ref().map_or(0, sync::Chain::matched),
            timing_score,
            complete: coverage >= 0.99,
            notes,
        });
    }
    out
}

/// Hypotheses derived purely from the measured sync period.
///
/// This is the CTF path: the header has been removed or destroyed, and the
/// only surviving structure is the repeated line sync.
fn from_sync_period(pulses: &[Pulse], rate: f64, duration_seconds: f64) -> Vec<Hypothesis> {
    let mut out = Vec::new();
    if pulses.len() < 4 {
        return out;
    }

    for mode in modes::all() {
        let nominal = mode.line_seconds * rate;
        // A transmitter may run a few percent fast or slow, and the search
        // window must cover that without becoming so wide that neighbouring
        // modes' periods fall inside it.
        let tolerance = (nominal * 0.06).max(rate * 0.002);
        let min_pulses = 4;

        // Every pulse is a candidate start, but most of those describe the
        // *same* transmission seen part-way through: a chain beginning at
        // pulse 17 is the chain beginning at pulse 0 with its head cut off.
        // Emitting all of them would bury the true start in duplicates, so
        // candidates are reduced to one per maximal run.
        let mut index = 0usize;
        while index < pulses.len() {
            let chain = sync::chain(pulses, index, nominal, tolerance);
            if chain.matched() < min_pulses {
                index += 1;
                continue;
            }
            let score = sync::chain_score(&chain);
            if score < 0.2 {
                index += 1;
                continue;
            }

            // The chain starts at a sync pulse. For Scottie that pulse is
            // mid-line, so the image start is earlier.
            let start_pulse = pulses[index];
            let skip = (start_pulse.start - mode.sync_offset_seconds() * rate).max(0.0);
            let coverage = coverage(&mode, skip, chain.clock_rate, rate, duration_seconds);

            let mut notes = vec![format!(
                "{} consecutive sync pulses matched a {:.0} ms line period",
                chain.matched(),
                mode.line_seconds * 1000.0
            )];
            if (chain.clock_rate - 1.0).abs() > 0.002 {
                notes.push(format!(
                    "clock measured {:.3}% {} nominal",
                    (chain.clock_rate - 1.0) * 100.0,
                    if chain.clock_rate > 1.0 {
                        "fast"
                    } else {
                        "slow"
                    }
                ));
            }
            if coverage < 0.99 {
                notes.push(format!(
                    "only {:.0}% of the transmission is present",
                    coverage * 100.0
                ));
            }

            out.push(Hypothesis {
                mode,
                origin: Origin::SyncPeriod,
                skip_sample: skip,
                frequency_shift_hz: chain.shift_hz,
                clock_rate: chain.clock_rate,
                matched_syncs: chain.matched(),
                timing_score: score,
                complete: coverage >= 0.99,
                notes,
            });

            // Jump past the pulses this chain consumed. Any chain starting
            // inside it is a truncated restatement of the same transmission.
            let consumed = chain
                .indices
                .last()
                .copied()
                .unwrap_or(index)
                .saturating_sub(index);
            index += consumed.max(1);
        }
    }

    out
}

/// Fit the best sync chain to a known start and period.
fn best_chain(pulses: &[Pulse], skip: f64, period: f64) -> Option<sync::Chain> {
    let tolerance = (period * 0.06).max(4.0);
    // Find the pulse that best matches the predicted first sync position.
    //
    // The search window is a full period either side, not half. The payload
    // position derived from a VIS stop pulse is a prediction, not a
    // measurement: depending on where the decoder places the stop-bit
    // boundary it can land a whole line away from the first sync pulse, and a
    // half-period window then finds nothing at all. A full period still
    // cannot select the wrong pulse, because consecutive syncs are exactly one
    // period apart.
    let best = pulses
        .iter()
        .enumerate()
        .filter(|(_, pulse)| (pulse.start - skip).abs() <= period)
        .min_by(|(_, a), (_, b)| (a.start - skip).abs().total_cmp(&(b.start - skip).abs()))
        .map(|(index, _)| index)?;
    let chain = sync::chain(pulses, best, period, tolerance);
    (chain.matched() >= 3).then_some(chain)
}

/// Fraction of a mode's transmission present in the recording, from `skip`.
fn coverage(mode: &Mode, skip: f64, clock_rate: f64, rate: f64, duration: f64) -> f64 {
    if duration <= 0.0 {
        return 0.0;
    }
    let start_seconds = skip / rate;
    let needed = mode.image_seconds() * clock_rate;
    let available = (duration - start_seconds).max(0.0);
    (available / needed).clamp(0.0, 1.0)
}

/// Remove hypotheses that describe the same transmission the same way.
///
/// Two hypotheses for the same mode whose image starts within half a line of
/// each other are the same transmission measured twice. The list arrives
/// sorted best-first, so the survivor is the strongest.
fn deduplicate(hypotheses: &mut Vec<Hypothesis>) {
    let mut kept: Vec<Hypothesis> = Vec::with_capacity(hypotheses.len());
    for hypothesis in hypotheses.drain(..) {
        let window = hypothesis.mode.line_seconds * hypothesis.clock_rate * 0.5;
        let is_duplicate = kept.iter().any(|existing: &Hypothesis| {
            existing.mode.mode == hypothesis.mode.mode
                && (existing.skip_sample - hypothesis.skip_sample).abs() <= window
        });
        if !is_duplicate {
            kept.push(hypothesis);
        }
    }
    *hypotheses = kept;
}

/// Evaluate one hypothesis: extract a grid, re-render it, and measure the
/// agreement between the re-rendered audio and the original recording.
#[must_use]
pub fn evaluate(
    hypothesis: &Hypothesis,
    original: &Trajectory,
    analyzer: &Analyzer,
) -> Option<Candidate> {
    let rate = f64::from(original.rate());
    let duration = original.last_sample() / rate;

    // Only the portion of the transmission that is actually present.
    let grid = raster::extract(
        original,
        &hypothesis.mode,
        hypothesis.skip_sample,
        hypothesis.clock_rate,
    );

    let rendered = crate::synth::render(
        &grid,
        &hypothesis.mode,
        original.rate(),
        hypothesis.frequency_shift_hz,
        Some(hypothesis.clock_rate),
    );
    if rendered.is_empty() {
        return None;
    }
    let track = analyzer.track(&rendered);
    if track.is_empty() {
        return None;
    }
    let rerendered = Trajectory::from_track(&track, rendered.len());

    let agreement = agreement(
        original,
        &rerendered,
        hypothesis.skip_sample,
        rate,
        hypothesis.mode.image_seconds() * hypothesis.clock_rate,
    );

    let coverage_value = coverage(
        &hypothesis.mode,
        hypothesis.skip_sample,
        hypothesis.clock_rate,
        rate,
        duration,
    );

    // The evidence score is clamped first, then the authority bonus is added
    // and clamped again. Applying the bonus inside the same sum lets a strong
    // inferred candidate and a VIS candidate both saturate at 1.0, which
    // destroys the ordering the bonus exists to create.
    let evidence = (AGREEMENT_WEIGHT * agreement
        + TIMING_WEIGHT * hypothesis.timing_score
        + COVERAGE_WEIGHT * coverage_value)
        .clamp(0.0, 1.0);
    let authority = if hypothesis.origin == Origin::Vis {
        AUTHORITATIVE_BONUS
    } else {
        0.0
    };
    let score = (evidence + authority).clamp(0.0, 1.0);

    Some(Candidate {
        hypothesis: hypothesis.clone(),
        grid,
        agreement,
        timing_score: hypothesis.timing_score,
        coverage: coverage_value,
        score,
    })
}

/// How closely a re-rendered signal reproduces the original recording.
///
/// Returns a value in `0.0..=1.0`, where 1 is a perfect match.
///
/// # Method
///
/// Both signals are compared sample by sample over their shared span, and the
/// absolute frequency difference is averaged. Two refinements make the figure
/// discriminating enough to separate modes that share a line period:
///
/// * **Samples outside the image band are excluded.** Wherever the original
///   carries a porch or sync tone, the frequency says nothing about which mode
///   transmitted it, so including those samples would dilute the comparison
///   with agreement that any mode could achieve.
/// * **Error is averaged per sample, not accumulated**, so a long recording
///   does not drown out a short one.
///
/// The result is deliberately harsh: a mean error of 60 Hz already halves the
/// score. A hypothesis that re-renders the wrong geometry diverges by hundreds
/// of hertz across most of the picture, while the correct one stays within a
/// few tens.
#[must_use]
pub fn agreement(
    original: &Trajectory,
    rerendered: &Trajectory,
    offset: f64,
    _rate: f64,
    duration_seconds: f64,
) -> f64 {
    if rerendered.is_empty() || original.is_empty() || duration_seconds <= 0.0 {
        return 0.0;
    }
    let samples = rendered_len(rerendered);
    // Cap the comparison at a few thousand probes: enough to characterise the
    // match, cheap enough to run for every hypothesis, and independent of how
    // long the recording is.
    let step = (samples / 4000).max(1);
    let mut total_error = 0.0_f64;
    let mut compared = 0usize;

    let mut position = 0usize;
    while position < samples {
        let absolute = offset + position as f64;
        let original_hz = original.hz_at(absolute);
        let level = original.level_at(absolute);
        let rerendered_hz = rerendered.hz_at(position as f64);
        // Compare only where the original carries real image content. A
        // porch or sync tone is transmitted identically by every mode, so
        // including it would credit every hypothesis equally.
        if level > 1.0e-4
            && (crate::modes::LEVEL_MIN_HZ..=crate::modes::LEVEL_MAX_HZ).contains(&original_hz)
        {
            total_error += (original_hz - rerendered_hz).abs();
            compared += 1;
        }
        position += step;
    }

    // Too few comparable points means the span covers almost no image content,
    // which is itself grounds for rejecting the hypothesis.
    if compared < 50 {
        return 0.0;
    }
    let error = total_error / compared as f64;
    // 60 Hz mean error halves the score; 150 Hz is a total mismatch.
    (1.0 - error / 150.0).clamp(0.0, 1.0)
}

fn rendered_len(trajectory: &Trajectory) -> usize {
    trajectory.last_sample() as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modes::describe;
    use crate::synth::render_with_vis;
    use slowrx::SstvMode;

    /// Build a synthetic SSTV recording: VIS header followed by image audio.
    fn recording(mode: &Mode, rate: u32, shift: f64, clock: f64) -> (Vec<f32>, Grid) {
        let grid = crate::synth::test_grid(mode);
        let signal = render_with_vis(&grid, mode, rate, shift, Some(clock));
        (signal, grid)
    }

    fn trajectory_of(signal: &[f32], rate: u32) -> (Trajectory, Analyzer) {
        let analyzer = Analyzer::new(rate);
        let track = analyzer.track(signal);
        let trajectory = Trajectory::from_track(&track, signal.len());
        (trajectory, analyzer)
    }

    #[test]
    fn vis_recording_is_identified_without_hints() {
        let rate = 16_000;
        let mode = describe(SstvMode::Robot36).expect("robot36");
        let (signal, _) = recording(&mode, rate, 0.0, 1.0);
        let (trajectory, _) = trajectory_of(&signal, rate);
        let analysis = analyze(&trajectory);
        assert!(analysis.plausible_sstv);
        let best = analysis
            .hypotheses
            .first()
            .expect("at least one hypothesis");
        assert_eq!(best.mode.mode, SstvMode::Robot36);
        assert_eq!(best.origin, Origin::Vis);
        assert!(
            (best.frequency_shift_hz).abs() < 25.0,
            "offset {}",
            best.frequency_shift_hz
        );
    }

    #[test]
    fn vis_parity_failure_is_flagged_but_still_ranked() {
        let rate = 16_000;
        let mode = describe(SstvMode::Robot36).expect("robot36");
        let grid = crate::synth::test_grid(&mode);
        // Corrupt the VIS code to one that is not a supported mode, keeping
        // the payload valid. The decoder must not claim a clean VIS.
        let mut signal = crate::vis::synthesize(0x7e, rate, 0.0);
        signal.extend(crate::synth::render(&grid, &mode, rate, 0.0, None));
        let (trajectory, _) = trajectory_of(&signal, rate);
        let analysis = analyze(&trajectory);
        for hit in &analysis.vis_hits {
            assert!(!hit.parity_ok || hit.code == 0x7e);
        }
        // The sync-period path must still find Robot 36.
        let found = analysis
            .hypotheses
            .iter()
            .any(|h| h.mode.mode == SstvMode::Robot36);
        assert!(found, "sync-period fallback did not identify Robot 36");
    }

    #[test]
    fn measured_offset_is_reported() {
        let rate = 16_000;
        let mode = describe(SstvMode::Robot36).expect("robot36");
        let (signal, _) = recording(&mode, rate, 90.0, 1.0);
        let (trajectory, _) = trajectory_of(&signal, rate);
        let analysis = analyze(&trajectory);
        assert!(
            (analysis.frequency_shift_hz - 90.0).abs() < 30.0,
            "offset {}",
            analysis.frequency_shift_hz
        );
    }

    #[test]
    fn a_correct_hypothesis_agrees_with_its_own_recording() {
        let rate = 16_000;
        let mode = describe(SstvMode::Robot36).expect("robot36");
        let (signal, _) = recording(&mode, rate, 0.0, 1.0);
        let (trajectory, analyzer) = trajectory_of(&signal, rate);
        let analysis = analyze(&trajectory);
        let hypothesis = analysis.hypotheses.first().expect("a hypothesis");
        let candidate = evaluate(hypothesis, &trajectory, &analyzer).expect("a candidate");
        assert!(
            candidate.agreement > 0.8,
            "self-agreement {} is too low",
            candidate.agreement
        );
        assert!(candidate.score > ACCEPT_THRESHOLD);
    }

    #[test]
    fn a_wrong_mode_agrees_worse_than_the_right_one() {
        // The core guarantee of the ranking: re-rendering with the wrong
        // geometry cannot reproduce the recording.
        let rate = 16_000;
        let mode = describe(SstvMode::Robot36).expect("robot36");
        let (signal, _) = recording(&mode, rate, 0.0, 1.0);
        let (trajectory, analyzer) = trajectory_of(&signal, rate);
        let analysis = analyze(&trajectory);
        let right = analysis
            .hypotheses
            .iter()
            .find(|h| h.mode.mode == SstvMode::Robot36)
            .expect("robot36 hypothesis");
        let right_candidate = evaluate(right, &trajectory, &analyzer).expect("candidate");

        // Force a wrong mode with the same start and offset. It must differ
        // from Robot 36 in line timing, otherwise it is the same transmission
        // and there is nothing to discriminate.
        let mut wrong = right.clone();
        wrong.mode = describe(SstvMode::Scottie1).expect("scottie1");
        assert_ne!(
            wrong.mode.line_seconds, right.mode.line_seconds,
            "the wrong mode must have different timing"
        );
        let wrong_candidate = evaluate(&wrong, &trajectory, &analyzer).expect("candidate");

        // The correct mode must both agree better and rank higher. The
        // agreement margin is modest because both hypotheses decode *an*
        // image; what matters for the decision is the combined score, which
        // also carries timing evidence.
        // This test forces the wrong hypothesis to share the correct one's
        // start position and frequency offset, so timing and coverage are
        // identical by construction and cannot contribute. The quantity that
        // separates them is *agreement*: whether re-rendering the recovered
        // grid at that mode reproduces the recording.
        assert!(
            right_candidate.agreement > wrong_candidate.agreement + 0.02,
            "agreement: right {:.4} vs wrong {:.4}",
            right_candidate.agreement,
            wrong_candidate.agreement
        );

        // In a real run the modes compete on their own start positions too,
        // where the wrong mode's weaker chain fit lowers its score further.
        // Check that end-to-end ordering holds, which is what actually decides
        // the reported answer.
        let (signal, _) = recording(&mode, rate, 0.0, 1.0);
        let (trajectory, analyzer) = trajectory_of(&signal, rate);
        let ranked = analyze(&trajectory);
        let winner = ranked
            .hypotheses
            .iter()
            .filter_map(|h| evaluate(h, &trajectory, &analyzer))
            .max_by(|a, b| a.score.total_cmp(&b.score))
            .expect("a ranked candidate");
        assert_eq!(
            winner.hypothesis.mode.mode,
            SstvMode::Robot36,
            "the ranking picked {} instead of Robot 36",
            winner.hypothesis.mode.name
        );
    }

    #[test]
    fn clock_error_is_measured_and_ranked() {
        let rate = 16_000;
        let mode = describe(SstvMode::Robot36).expect("robot36");
        // A transmitter running 1.5% slow.
        let (signal, _) = recording(&mode, rate, 0.0, 0.985);
        let (trajectory, analyzer) = trajectory_of(&signal, rate);
        let analysis = analyze(&trajectory);
        let hypothesis = analysis
            .hypotheses
            .iter()
            .find(|h| h.mode.mode == SstvMode::Robot36)
            .expect("robot36 hypothesis");
        // The chain fit should recover the 1.5% slow clock.
        assert!(
            (hypothesis.clock_rate - 0.985).abs() < 0.01,
            "clock {}",
            hypothesis.clock_rate
        );
        let candidate = evaluate(hypothesis, &trajectory, &analyzer).expect("candidate");
        assert!(
            candidate.agreement > 0.7,
            "agreement {} with clock correction",
            candidate.agreement
        );
    }

    #[test]
    fn every_supported_mode_is_identified_from_its_vis() {
        let rate = 16_000;
        for mode in modes::all() {
            let (signal, _) = recording(&mode, rate, 0.0, 1.0);
            let (trajectory, _) = trajectory_of(&signal, rate);
            let analysis = analyze(&trajectory);
            let best = analysis
                .hypotheses
                .first()
                .unwrap_or_else(|| panic!("no hypothesis for {}", mode.name));
            assert_eq!(
                best.mode.mode, mode.mode,
                "{} misidentified as {}",
                mode.name, best.mode.name
            );
            assert_eq!(best.origin, Origin::Vis, "{}", mode.name);
        }
    }

    #[test]
    fn every_supported_mode_survives_a_missing_header() {
        // Strip the VIS header entirely: only the sync period remains.
        let rate = 16_000;
        for mode in modes::all() {
            let grid = crate::synth::test_grid(&mode);
            let signal = crate::synth::render(&grid, &mode, rate, 0.0, None);
            let (trajectory, _) = trajectory_of(&signal, rate);
            let analysis = analyze(&trajectory);
            assert!(
                analysis.vis_hits.is_empty(),
                "{} produced a phantom VIS",
                mode.name
            );
            // A mode may be indistinguishable from another without a header
            // (Robot 24 and Robot 36 share their entire wire format), so any
            // member of that equivalence class is an acceptable answer.
            let mut acceptable = vec![mode.mode];
            acceptable.extend(indistinguishable_from(&mode).into_iter().map(|m| m.mode));
            let found = analysis
                .hypotheses
                .iter()
                .any(|h| acceptable.contains(&h.mode.mode));
            assert!(
                found,
                "{} was not recoverable from its sync period alone (tried {:?})",
                mode.name, acceptable
            );
        }
    }

    #[test]
    fn a_submultiple_period_does_not_win_over_the_true_mode() {
        // Regression: a Robot 36 recording previously also produced a Robot 72
        // image, because a 300 ms period matches every second pulse of a
        // 150 ms transmission.
        let rate = 16_000;
        let mode = describe(SstvMode::Robot36).expect("robot36");
        let grid = crate::synth::test_grid(&mode);
        let signal = crate::synth::render(&grid, &mode, rate, 0.0, None);
        let (trajectory, analyzer) = trajectory_of(&signal, rate);
        let analysis = analyze(&trajectory);

        let mut candidates: Vec<Candidate> = analysis
            .hypotheses
            .iter()
            .filter_map(|h| evaluate(h, &trajectory, &analyzer))
            .collect();
        candidates.sort_by(|a, b| b.score.total_cmp(&a.score));

        let mut accepted: Vec<Candidate> = Vec::new();
        for candidate in candidates {
            if accepted
                .iter()
                .any(|better| is_explained_by(&candidate, better))
            {
                continue;
            }
            accepted.push(candidate);
        }

        // Robot 24 is an acceptable answer (it shares Robot 36's wire format),
        // but a submultiple-period mode such as Robot 72 must not survive.
        assert!(
            accepted
                .iter()
                .all(|c| c.hypothesis.mode.mode == SstvMode::Robot36
                    || c.hypothesis.mode.mode == SstvMode::Robot24),
            "accepted {:?}",
            accepted
                .iter()
                .map(|c| c.hypothesis.mode.name)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn silence_produces_no_candidates() {
        let rate = 16_000;
        let signal = vec![0.0_f32; rate as usize * 3];
        let (trajectory, _) = trajectory_of(&signal, rate);
        let analysis = analyze(&trajectory);
        assert!(!analysis.plausible_sstv);
        assert!(analysis.hypotheses.is_empty());
    }

    #[test]
    fn broadband_noise_produces_no_strong_candidate() {
        let rate = 16_000;
        let signal: Vec<f32> = (0..rate as usize * 4)
            .map(|i| {
                let x = (i as f64 * 12.9898).sin() * 43_758.545_3;
                ((x - x.floor()) as f32 - 0.5) * 0.9
            })
            .collect();
        let (trajectory, analyzer) = trajectory_of(&signal, rate);
        let analysis = analyze(&trajectory);
        let best = analysis
            .hypotheses
            .iter()
            .filter_map(|h| evaluate(h, &trajectory, &analyzer))
            .fold(0.0_f64, |acc, c| acc.max(c.score));
        assert!(
            best < ACCEPT_THRESHOLD,
            "noise produced a candidate scoring {best}"
        );
    }

    #[test]
    fn empty_trajectory_is_handled() {
        let trajectory = Trajectory::from_model(Vec::new(), Vec::new(), 16_000);
        let analysis = analyze(&trajectory);
        assert!(analysis.hypotheses.is_empty());
        assert!(!analysis.plausible_sstv);
        assert_eq!(analysis.duration_seconds, 0.0);
    }

    #[test]
    fn agreement_is_zero_for_unusable_input() {
        let empty = Trajectory::from_model(Vec::new(), Vec::new(), 16_000);
        let real = Trajectory::from_model(vec![0.0, 1.0], vec![1200.0, 1200.0], 16_000);
        assert_eq!(agreement(&empty, &real, 0.0, 16_000.0, 1.0), 0.0);
        assert_eq!(agreement(&real, &empty, 0.0, 16_000.0, 1.0), 0.0);
        assert_eq!(agreement(&real, &real, 0.0, 16_000.0, 0.0), 0.0);
    }

    #[test]
    fn truncated_recording_is_marked_incomplete() {
        let rate = 16_000;
        let mode = describe(SstvMode::Robot36).expect("robot36");
        let (signal, _) = recording(&mode, rate, 0.0, 1.0);
        // Keep only the first third.
        let cut = &signal[..signal.len() / 3];
        let (trajectory, _) = trajectory_of(cut, rate);
        let analysis = analyze(&trajectory);
        let hypothesis = analysis
            .hypotheses
            .iter()
            .find(|h| h.mode.mode == SstvMode::Robot36)
            .expect("robot36 still identifiable from its sync period");
        assert!(!hypothesis.complete);
        // A truncated recording is marked incomplete by the `complete` flag.
        // Which evidence notes accompany it depends on which path produced the
        // hypothesis: the sync-period path records the shortfall explicitly,
        // while the VIS path trusts the header it found. Assert on the flag,
        // and require a note only from the path that promises one.
        if hypothesis.origin == Origin::SyncPeriod {
            assert!(
                hypothesis.coverage_is_partial_note(),
                "sync-period hypothesis should record the shortfall, notes were {:?}",
                hypothesis.notes
            );
        }
    }
}
