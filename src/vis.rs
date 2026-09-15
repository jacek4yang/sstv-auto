//! VIS header detection.
//!
//! A VIS burst is `1900 Hz leader (300 ms)`, a `1200 Hz` start pulse
//! (30 ms), seven data bits of 30 ms each (`1100 Hz` = 1, `1300 Hz` = 0,
//! least significant bit first), a parity bit, and a `1200 Hz` stop pulse.
//!
//! This module searches for that structure directly instead of relying on a
//! VIS code being present: it finds leader spans first, then samples each bit
//! window from the measured [`Trajectory`]. Keeping the decoder here (rather
//! than only inside the raster backend) means a damaged burst still yields a
//! *candidate* that the blind path can weigh, and it gives the CLI a
//! meaningful diagnostic when parity fails.

use crate::dsp::{Trajectory, mean};
use crate::modes::{
    Mode, VIS_BIT_ONE_HZ, VIS_BIT_SECONDS, VIS_BIT_ZERO_HZ, VIS_BREAK_SECONDS, VIS_LEADER_HZ,
    VIS_LEADER_SECONDS, VIS_SEPARATOR_HZ, VIS_TOTAL_SECONDS,
};

/// A decoded VIS burst together with the evidence that produced it.
#[derive(Debug, Clone)]
pub struct VisHit {
    /// Decoded 7-bit VIS code.
    pub code: u8,
    /// Mode the code names, when it is one this build supports.
    pub mode: Option<Mode>,
    /// Sample position where the image payload is expected to begin.
    pub payload_sample: f64,
    /// Sample position where the leading 1900 Hz tone starts.
    pub leader_sample: f64,
    /// Measured leader frequency, Hz.
    pub leader_hz: f64,
    /// Receiver frequency offset implied by the leader, Hz.
    pub frequency_shift_hz: f64,
    /// Whether the parity bit matched.
    pub parity_ok: bool,
    /// Per-bit tone deviation from nominal, Hz. Diagnostic only.
    pub bit_errors_hz: [f64; 7],
    /// Median tone purity across the burst, `0.0..=1.0`.
    pub purity: f64,
    /// Field-duration scale that best fitted the burst, where 1.0 is nominal.
    pub clock_scale: f64,
}

/// Window in which a real leader pulse must sit, around 1900 Hz.
///
/// These tolerances are deliberately generous because a mistuned receiver
/// shifts *every* tone by the same amount. Widening them cannot create false
/// positives on its own: [`decode_at`] requires the complete
/// leader/break/leader/break/bits/parity/stop structure, and a shifted image
/// tone falls outside all of those windows at once.
const LEADER_TOLERANCE_HZ: f64 = 220.0;
/// Window in which a real start/stop pulse must sit, around 1200 Hz.
const SEPARATOR_TOLERANCE_HZ: f64 = 220.0;
/// Minimum fraction of a tone window that must actually be that tone.
const MIN_TONE_FRACTION: f64 = 0.6;
/// Analysis points taken inside each tone window.
const TONE_PROBES: usize = 5;

/// Measure how much of `[start, end)` sits within `tolerance` Hz of `target`.
///
/// Returns `(fraction, mean error in Hz)`. The fraction is what makes this
/// robust against dropouts: a burst with a lost bit still scores well on the
/// remaining six, and the caller decides what to do about the gap.
fn tone_fraction(
    trajectory: &Trajectory,
    start: f64,
    end: f64,
    target: f64,
    tolerance: f64,
) -> (f64, f64) {
    if end <= start {
        return (0.0, f64::INFINITY);
    }
    let mut inside = 0usize;
    let mut errors = Vec::with_capacity(TONE_PROBES);
    // Sample the middle 80% of the window so transitions never land in a
    // probe.
    let margin = 0.1 * (end - start);
    let lo = start + margin;
    let hi = end - margin;
    for i in 0..TONE_PROBES {
        let position = lo + (hi - lo) * (i as f64 + 0.5) / TONE_PROBES as f64;
        let hz = trajectory.hz_at(position);
        let error = hz - target;
        errors.push(error.abs());
        if error.abs() <= tolerance {
            inside += 1;
        }
    }
    let fraction = inside as f64 / TONE_PROBES as f64;
    (fraction, mean(&errors))
}

/// Median dominant frequency across the middle 80% of a window.
///
/// Used for bit decisions, where a single frame straddling a tone transition
/// must not decide the value.
fn window_median(trajectory: &Trajectory, start: f64, len: f64) -> f64 {
    let margin = 0.1 * len;
    let lo = start + margin;
    let hi = start + len - margin;
    let mut samples = Vec::with_capacity(TONE_PROBES);
    for i in 0..TONE_PROBES {
        let position = lo + (hi - lo) * (i as f64 + 0.5) / TONE_PROBES as f64;
        samples.push(trajectory.hz_at(position));
    }
    crate::dsp::median(&mut samples)
}

/// Find every VIS burst in a trajectory.
///
/// Candidates are ranked by the quality of the leader pulse first, because a
/// leader is the easiest part of the burst to detect reliably and every other
/// field is measured relative to it.
#[must_use]
pub fn find(trajectory: &Trajectory) -> Vec<VisHit> {
    let rate = f64::from(trajectory.rate());
    let leader_len = VIS_LEADER_SECONDS * rate;
    let burst_len = (VIS_TOTAL_SECONDS * rate) as usize;

    let mut hits = Vec::new();
    if trajectory.is_empty() {
        return hits;
    }

    // Search every position where a burst could start. The bound is relaxed
    // by the trajectory's own sample step rather than requiring a full
    // burst's worth of span to remain: the model ends where the last analysis
    // frame ends, which can legitimately be a step or two inside the final
    // burst. `decode_at` reads through `Trajectory::hz_at`, which clamps at
    // the end, so a burst that runs off the end of the recording is still
    // evaluated on the part that is present.
    let margin = if trajectory.xs.len() > 1 {
        trajectory.xs[1] - trajectory.xs[0]
    } else {
        1.0
    };
    let first = trajectory.xs[0];
    let last = trajectory.last_sample();
    // A burst cannot start unless a leader pair and at least the start pulse
    // still fit inside the model.
    if last - first + margin < 2.0 * VIS_LEADER_SECONDS * rate + VIS_BREAK_SECONDS * rate {
        return hits;
    }

    // Scan for a long stay near 1900 Hz. The candidate is then validated
    // against the *entire* burst structure by `decode_at`, so a loose scan
    // step costs a little work but never a false positive.
    let step = ((0.005 * rate) as usize).max(1);
    let mut cursor = 0usize;
    while first + cursor as f64 <= last {
        let start = first + cursor as f64;
        let (fraction, _) = tone_fraction(
            trajectory,
            start,
            start + leader_len,
            VIS_LEADER_HZ,
            LEADER_TOLERANCE_HZ,
        );
        if fraction >= MIN_TONE_FRACTION {
            if let Some(hit) = decode_searching_clock(trajectory, start) {
                // Resume past the burst just accepted; a repeated leader can
                // never be a second transmission.
                let next = (hit.payload_sample - first) as usize;
                cursor = next.max(cursor + step);
                // Only keep the strongest hit per burst.
                let duplicate = hits.last().is_some_and(|last: &VisHit| {
                    (last.leader_sample - hit.leader_sample).abs() < burst_len as f64
                });
                if !duplicate {
                    hits.push(hit);
                }
                continue;
            }
        }
        cursor += step;
    }

    hits.sort_by(|a, b| {
        b.purity
            .total_cmp(&a.purity)
            .then_with(|| a.leader_sample.total_cmp(&b.leader_sample))
    });
    hits
}

/// Tone-match quality of one VIS burst at a given clock scale.
///
/// Returns the fraction of the burst's ten fixed fields whose *centre* lies on
/// the tone the format requires: the two leaders and the break between them,
/// then the start pulse, the seven data bits, the parity bit and the stop
/// pulse.
///
/// # Why this is the right score to compare scales with
///
/// Ranking scales by a mean of per-field confidences is not scale-neutral: a
/// squeezed burst has proportionally shorter probe windows and scores
/// differently for reasons that have nothing to do with correctness. Counting
/// how many fields land on their expected tone is directly meaningful and
/// comparable: the correct scale puts every field on its tone, and any other
/// scale puts the later fields progressively further off.
fn field_match_score(trajectory: &Trajectory, leader_start: f64, scale: f64) -> f64 {
    let rate = f64::from(trajectory.rate());
    let leader_len = VIS_LEADER_SECONDS * rate * scale;
    let break_len = VIS_BREAK_SECONDS * rate * scale;
    let bit_len = VIS_BIT_SECONDS * rate * scale;

    // (offset, length, expected tone) for every field in transmission order.
    let mut fields: Vec<(f64, f64, f64)> = vec![
        (0.0, leader_len, VIS_LEADER_HZ),
        (leader_len, break_len, VIS_SEPARATOR_HZ),
        (leader_len + break_len, leader_len, VIS_LEADER_HZ),
    ];
    let after_second_leader = leader_len + break_len + leader_len;
    // Start pulse, then bits, parity and stop all share one length.
    fields.push((after_second_leader, bit_len, VIS_SEPARATOR_HZ));
    for index in 1..=9 {
        // The last field is the stop pulse; the one before it is the parity
        // bit. Both are fixed tones, so they are checked here too rather than
        // being left to the data decode.
        let tone = match index {
            9 => VIS_SEPARATOR_HZ,
            _ => 0.0, // data bits are checked by `decode_at_scale`
        };
        if tone == 0.0 {
            continue;
        }
        fields.push((after_second_leader + bit_len * index as f64, bit_len, tone));
    }

    let mut matched = 0usize;
    for (offset, len, tone) in &fields {
        let centre = leader_start + offset + len * 0.5;
        if (trajectory.hz_at(centre) - tone).abs() <= LEADER_TOLERANCE_HZ {
            matched += 1;
        }
    }
    matched as f64 / fields.len() as f64
}

/// Sample the full VIS structure, searching a small range of clock scales.
///
/// A transmitter running a percent or two fast or slow compresses or stretches
/// every VIS field by that factor. At a 30 ms bit, one percent is 300 us — far
/// less than the field itself, but enough that fixed field boundaries drift out
/// of alignment by the end of the burst and the stop pulse lands on the wrong
/// tone.
///
/// The nominal timing is tried first, then scales either side of it. The
/// winner is the scale whose fixed fields all land on their expected tones,
/// with parity valid as a tie-breaker, so a clean burst is decoded at exactly
/// 1.0.
#[must_use]
pub fn decode_searching_clock(trajectory: &Trajectory, leader_start: f64) -> Option<VisHit> {
    // Scales are tried outward from nominal so that a clean burst is decoded
    // once, at exactly 1.0, and a neighbouring scale can only win by being
    // genuinely better rather than by being tried first.
    const SCALES: [f64; 9] = [1.0, 1.01, 0.99, 1.02, 0.98, 1.03, 0.97, 1.04, 0.96];
    let mut best: Option<(f64, VisHit)> = None;
    for scale in SCALES {
        let Some(hit) = decode_at_scale(trajectory, leader_start, scale) else {
            continue;
        };
        let mut score = field_match_score(trajectory, leader_start, scale);
        // A burst whose parity checks out is strictly more trustworthy than
        // one whose does not, at any structural score.
        if hit.parity_ok {
            score += 1.0;
        }
        let better = best
            .as_ref()
            .is_none_or(|(best_score, _)| score > *best_score);
        if better {
            best = Some((score, hit));
        }
    }
    best.map(|(_, hit)| hit)
}

/// Sample the full VIS structure assuming the first leader tone starts at
/// `leader_start`.
///
/// The burst is `leader, break, leader, break, 7 bits, parity, stop`, and
/// *every* one of those fields is validated here. Checking the whole
/// structure — rather than just finding a leader and reading seven bit
/// windows — is what stops image audio from being mistaken for a header.
///
/// Returns `None` when any required field is missing or out of tolerance.
#[must_use]
pub fn decode_at(trajectory: &Trajectory, leader_start: f64) -> Option<VisHit> {
    decode_at_scale(trajectory, leader_start, 1.0)
}

/// Sample the VIS structure with every field duration scaled by `scale`.
#[must_use]
pub fn decode_at_scale(trajectory: &Trajectory, leader_start: f64, scale: f64) -> Option<VisHit> {
    let rate = f64::from(trajectory.rate());
    let bit_len = VIS_BIT_SECONDS * rate * scale;
    let leader_len = VIS_LEADER_SECONDS * rate * scale;
    // The break is 10 ms, not a 30 ms bit. Getting this wrong shifts every
    // following field by 20 ms and makes a real header undecodable.
    let break_len = VIS_BREAK_SECONDS * rate * scale;

    // The second leader tone, after the first break, is the most distinctive
    // feature of the format: two long 1900 Hz tones separated by a short
    // gap. Require both before reading anything else.
    let break_start = leader_start + leader_len;
    let (break_fraction, _) = tone_fraction(
        trajectory,
        break_start,
        break_start + break_len,
        VIS_SEPARATOR_HZ,
        SEPARATOR_TOLERANCE_HZ,
    );
    if break_fraction < MIN_TONE_FRACTION {
        return None;
    }

    let second_leader = break_start + break_len;
    let (leader_fraction, _) = tone_fraction(
        trajectory,
        second_leader,
        second_leader + leader_len,
        VIS_LEADER_HZ,
        LEADER_TOLERANCE_HZ,
    );
    if leader_fraction < MIN_TONE_FRACTION {
        return None;
    }

    // Second break, which closes the leader pair and opens the data field.
    let start_pulse = second_leader + leader_len;
    let (start_fraction, _) = tone_fraction(
        trajectory,
        start_pulse,
        start_pulse + bit_len,
        VIS_SEPARATOR_HZ,
        SEPARATOR_TOLERANCE_HZ,
    );
    if start_fraction < MIN_TONE_FRACTION {
        return None;
    }

    // Leader frequency, measured well inside the first leader to avoid its
    // edges. Both leaders are averaged so a single noisy pulse cannot swing
    // the offset estimate.
    let leader_hz = {
        let mut samples = Vec::with_capacity(2 * TONE_PROBES);
        for (from, len) in [(leader_start, leader_len), (second_leader, leader_len)] {
            let lo = from + 0.1 * len;
            let hi = from + 0.9 * len;
            for i in 0..TONE_PROBES {
                let position = lo + (hi - lo) * (i as f64 + 0.5) / TONE_PROBES as f64;
                samples.push(trajectory.hz_at(position));
            }
        }
        crate::dsp::median(&mut samples)
    };
    if (leader_hz - VIS_LEADER_HZ).abs() > LEADER_TOLERANCE_HZ {
        return None;
    }
    let shift = leader_hz - VIS_LEADER_HZ;

    // Bits are separated by only 200 Hz, which is comparable to the
    // tolerance a shifted receiver needs. Discriminate them *relative to the
    // measured shift* instead of with fixed windows, so a mistuned burst
    // reads as cleanly as a centred one.
    //
    // Polarity note: a VIS `1` is the *lower* tone (1100 Hz) and a `0` is the
    // higher one (1300 Hz). Writing this the intuitive way round silently
    // turns every code into its complement, which still passes parity because
    // `0x08 ^ 0x77` has even weight — the parity check alone cannot catch it.
    let one_hz = VIS_BIT_ONE_HZ + shift;
    let zero_hz = VIS_BIT_ZERO_HZ + shift;
    let midpoint = 0.5 * (one_hz + zero_hz);

    let mut code = 0u8;
    let mut parity = 0u8;
    let mut bit_errors_hz = [0.0_f64; 7];
    let mut purity_samples = Vec::with_capacity(7);
    let mut good_bits = 0usize;

    for (bit, slot) in bit_errors_hz.iter_mut().enumerate() {
        let start = start_pulse + bit_len * (bit as f64 + 1.0);
        // Median of several probes across the bit window, for the same reason
        // the pixel sampler uses a median: a single frame straddling a
        // transition must not decide the bit.
        let measured = window_median(trajectory, start, bit_len);
        let is_one = measured < midpoint;

        *slot = measured - if is_one { one_hz } else { zero_hz };

        let distance = (measured - midpoint).abs();
        // A tone sitting almost exactly between the two bit frequencies is
        // noise, not data.
        if distance < 25.0 {
            purity_samples.push(0.0);
            continue;
        }
        good_bits += 1;
        if is_one {
            code |= 1 << bit;
            parity ^= 1;
        }
        // Confidence in this bit: how far it sits inside its own window.
        purity_samples.push((distance / (0.5 * (zero_hz - one_hz).abs().max(1.0))).min(1.0));
    }

    // Parity bit.
    let parity_start = start_pulse + bit_len * 8.0;
    let parity_measured = window_median(trajectory, parity_start, bit_len);
    let parity_is_one = parity_measured < midpoint;
    let parity_distance = (parity_measured - midpoint).abs();
    let parity_ok = good_bits >= 7 && parity_distance >= 25.0 && parity_is_one == (parity == 1);
    purity_samples.push((parity_distance / 100.0).min(1.0));

    // Stop pulse closes the burst; its presence is what distinguishes a real
    // VIS from a chance run of similar tones.
    //
    // Ten 30 ms fields follow the second break: the start pulse, seven data
    // bits, the parity bit, and the stop pulse. The image therefore begins one
    // field after the stop pulse starts, which is where the burst ends.
    let stop_start = start_pulse + bit_len * 9.0;
    let payload_sample = start_pulse + bit_len * 10.0;
    let (stop_fraction, _) = tone_fraction(
        trajectory,
        stop_start,
        payload_sample,
        1200.0,
        SEPARATOR_TOLERANCE_HZ,
    );
    if stop_fraction < MIN_TONE_FRACTION {
        return None;
    }
    purity_samples.push(stop_fraction);

    let mode = crate::modes::from_vis(code);
    Some(VisHit {
        code,
        mode,
        payload_sample,
        leader_sample: leader_start,
        leader_hz,
        frequency_shift_hz: shift,
        parity_ok,
        bit_errors_hz,
        purity: mean(&purity_samples),
        clock_scale: scale,
    })
}

/// Synthesize a VIS burst as audio at `rate`.
///
/// Used by the blind path: once a mode has been inferred from timing, this
/// renders the one field the recording is missing, and the production raster
/// decoder then does the actual pixel recovery at full resolution.
#[must_use]
pub fn synthesize(code: u8, rate: u32, frequency_shift_hz: f64) -> Vec<f32> {
    let mut out = Vec::with_capacity((crate::modes::VIS_TOTAL_SECONDS * f64::from(rate)) as usize);
    let mut phase = 0.0_f64;
    let mut parity = 0u8;

    let emit = |out: &mut Vec<f32>, phase: &mut f64, hz: f64, seconds: f64| {
        let count = (seconds * f64::from(rate)).round() as usize;
        let step = std::f64::consts::TAU * (hz + frequency_shift_hz) / f64::from(rate);
        for _ in 0..count {
            out.push((phase.sin() * 0.70) as f32);
            *phase = (*phase + step).rem_euclid(std::f64::consts::TAU);
        }
    };

    // Two 300 ms leader tones bracket a short break: that doubled leader is
    // what the detector keys on, so emitting only one makes the synthesized
    // header unrecognisable. The break is 10 ms — deliberately not the 30 ms
    // used for the start, data, parity and stop fields.
    emit(&mut out, &mut phase, VIS_LEADER_HZ, VIS_LEADER_SECONDS);
    emit(&mut out, &mut phase, VIS_SEPARATOR_HZ, VIS_BREAK_SECONDS);
    emit(&mut out, &mut phase, VIS_LEADER_HZ, VIS_LEADER_SECONDS);
    emit(&mut out, &mut phase, VIS_SEPARATOR_HZ, VIS_BIT_SECONDS);
    for bit in 0..7 {
        let one = (code >> bit) & 1 == 1;
        parity ^= u8::from(one);
        emit(
            &mut out,
            &mut phase,
            if one { VIS_BIT_ONE_HZ } else { 1300.0 },
            VIS_BIT_SECONDS,
        );
    }
    emit(
        &mut out,
        &mut phase,
        if parity == 1 { VIS_BIT_ONE_HZ } else { 1300.0 },
        VIS_BIT_SECONDS,
    );
    emit(&mut out, &mut phase, 1200.0, VIS_BIT_SECONDS);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::Analyzer;
    use slowrx::SstvMode;

    /// Render a VIS burst and analyse it into a trajectory.
    fn trajectory_of(code: u8, shift: f64, leading_silence: f64) -> Trajectory {
        let rate = 22_050;
        let mut signal = vec![0.0_f32; (leading_silence * f64::from(rate)) as usize];
        signal.extend(synthesize(code, rate, shift));
        signal.extend(vec![0.0_f32; rate as usize / 4]);
        let analyzer = Analyzer::new(rate);
        let track = analyzer.track(&signal);
        Trajectory::from_track(&track, signal.len())
    }

    #[test]
    fn decodes_robot36_vis() {
        let trajectory = trajectory_of(0x08, 0.0, 0.05);
        let hits = find(&trajectory);
        assert!(!hits.is_empty(), "no VIS found");
        let hit = &hits[0];
        assert_eq!(hit.code, 0x08);
        assert!(hit.parity_ok);
        assert_eq!(hit.mode.map(|m| m.mode), Some(SstvMode::Robot36));
        assert!(hit.frequency_shift_hz.abs() < 15.0);
    }

    #[test]
    fn decodes_every_supported_vis_code() {
        for mode in crate::modes::all() {
            let trajectory = trajectory_of(mode.vis_code, 0.0, 0.05);
            let hits = find(&trajectory);
            let hit = hits
                .iter()
                .find(|h| h.code == mode.vis_code)
                .unwrap_or_else(|| panic!("no hit for {} (0x{:02x})", mode.name, mode.vis_code));
            assert!(hit.parity_ok, "{} parity", mode.name);
            assert_eq!(hit.mode.map(|m| m.mode), Some(mode.mode));
        }
    }

    #[test]
    fn reports_frequency_offset() {
        for shift in [-120.0, -40.0, 40.0, 120.0] {
            let trajectory = trajectory_of(0x08, shift, 0.05);
            let hits = find(&trajectory);
            assert!(!hits.is_empty(), "no VIS at shift {shift}");
            assert!(
                (hits[0].frequency_shift_hz - shift).abs() < 20.0,
                "shift {shift} measured {}",
                hits[0].frequency_shift_hz
            );
        }
    }

    #[test]
    fn finds_vis_after_leading_junk() {
        let trajectory = trajectory_of(0x2c, 0.0, 3.0);
        let hits = find(&trajectory);
        let hit = hits
            .iter()
            .find(|h| h.code == 0x2c)
            .expect("martin1 vis after 3s of silence");
        assert!(hit.parity_ok);
        // The payload must land about 3.9 s in, not at the start.
        assert!(
            hit.payload_sample > 3.0 * 22_050.0 * 0.8,
            "payload at {}",
            hit.payload_sample
        );
    }

    #[test]
    fn silence_yields_no_vis() {
        let rate = 22_050;
        let signal = vec![0.0_f32; rate as usize * 2];
        let analyzer = Analyzer::new(rate);
        let track = analyzer.track(&signal);
        let trajectory = Trajectory::from_track(&track, signal.len());
        assert!(find(&trajectory).is_empty());
    }

    #[test]
    fn image_audio_alone_yields_no_vis() {
        // A steady image tone is not a VIS burst: no leader, no bit pattern.
        let rate = 22_050;
        let signal: Vec<f32> = (0..rate as usize * 2)
            .map(|i| {
                (2.0 * std::f64::consts::PI * 1900.0 * i as f64 / f64::from(rate)).sin() as f32
            })
            .collect();
        let analyzer = Analyzer::new(rate);
        let track = analyzer.track(&signal);
        let trajectory = Trajectory::from_track(&track, signal.len());
        assert!(find(&trajectory).is_empty(), "false VIS from a plain tone");
    }

    #[test]
    fn truncated_leader_yields_no_vis() {
        let rate = 22_050;
        let full = synthesize(0x08, rate, 0.0);
        // Keep only the first 150 ms: half a leader and nothing else.
        let signal = full[..(0.150 * f64::from(rate)) as usize].to_vec();
        let analyzer = Analyzer::new(rate);
        let track = analyzer.track(&signal);
        let trajectory = Trajectory::from_track(&track, signal.len());
        assert!(find(&trajectory).is_empty());
    }

    #[test]
    fn synthesize_has_the_documented_length() {
        let rate = 48_000;
        let signal = synthesize(0x08, rate, 0.0);
        let seconds = signal.len() as f64 / f64::from(rate);
        assert!(
            (seconds - crate::modes::VIS_TOTAL_SECONDS).abs() < 1.0 / f64::from(rate),
            "VIS duration {seconds}, expected {}",
            crate::modes::VIS_TOTAL_SECONDS
        );
        // The burst is two leaders, one short break, and ten 30 ms fields:
        // the start pulse, seven data bits, the parity bit and the stop pulse.
        let expected = 2.0 * crate::modes::VIS_LEADER_SECONDS
            + crate::modes::VIS_BREAK_SECONDS
            + 10.0 * crate::modes::VIS_BIT_SECONDS;
        assert!(
            (seconds - expected).abs() < 0.001,
            "{seconds} vs {expected}"
        );
    }

    #[test]
    fn synthesized_burst_round_trips_through_the_decoder() {
        // Encoder and decoder must agree for every supported code, otherwise
        // the blind path would present a header the backend reads differently.
        let rate = 44_100;
        for mode in crate::modes::all() {
            let signal = synthesize(mode.vis_code, rate, 0.0);
            let analyzer = Analyzer::new(rate);
            let track = analyzer.track(&signal);
            let trajectory = Trajectory::from_track(&track, signal.len());
            let hit = find(&trajectory)
                .into_iter()
                .find(|h| h.code == mode.vis_code);
            assert!(
                hit.is_some(),
                "{} (0x{:02x}) did not round-trip",
                mode.name,
                mode.vis_code
            );
        }
    }

    #[test]
    fn vis_bit_polarity_is_not_inverted() {
        // Regression: reading a `1` as the *upper* of the two bit tones
        // returns the bitwise complement of the true code. That mistake is
        // invisible to the parity check whenever the complement has even
        // weight relative to the original, which is true for a lot of codes
        // (e.g. 0x08 -> 0x77), so it is asserted directly here.
        let rate = 22_050;
        for code in [0x08_u8, 0x2c, 0x3c, 0x5f, 0x00, 0x7f, 0x01] {
            let signal = synthesize(code, rate, 0.0);
            let analyzer = Analyzer::new(rate);
            let track = analyzer.track(&signal);
            let trajectory = Trajectory::from_track(&track, signal.len());
            let hit = find(&trajectory).into_iter().next().expect("a burst");
            assert_eq!(
                hit.code,
                code,
                "decoded 0x{:02x} as 0x{:02x} (complement is 0x{:02x})",
                code,
                hit.code,
                !code & 0x7f
            );
        }
    }

    #[test]
    fn payload_starts_one_stop_pulse_after_the_leader() {
        let rate = 22_050;
        let trajectory = trajectory_of(0x08, 0.0, 0.0);
        let hit = find(&trajectory).into_iter().next().expect("a hit");
        let expected = crate::modes::VIS_TOTAL_SECONDS * f64::from(rate);
        // The measured payload position is derived from tone boundaries, whose
        // trailing edge is not abrupt, so it lands within a fraction of a bit
        // of the nominal figure rather than exactly on it.
        assert!(
            (hit.payload_sample - expected).abs()
                < 0.6 * crate::modes::VIS_BIT_SECONDS * f64::from(rate),
            "payload at {} vs expected {expected}",
            hit.payload_sample
        );
    }
}
