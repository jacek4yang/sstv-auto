//! End-to-end orchestration: audio in, PNGs and a report out.
//!
//! The stages are deliberately ordered so that the cheap, authoritative
//! evidence is used before the expensive, speculative kind:
//!
//! 1. **Ingest** — decode to mono, pick the best channel.
//! 2. **Measure** — one FFT pass over the whole recording.
//! 3. **Detect** — VIS first; if that yields nothing usable, infer from the
//!    line-sync period.
//! 4. **Rank** — evaluate every hypothesis by re-rendering and comparing.
//! 5. **Decode** — run the full-resolution raster backend on the survivors
//!    only, so the expensive step happens a handful of times, not hundreds.
//! 6. **Report** — write PNGs and `report.json`.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};

use crate::audio::{self, ChannelChoice};
use crate::autodetect::{self, Candidate, Origin};
use crate::backend;
use crate::dsp::{Analyzer, Trajectory};
use crate::modes::{self, Mode};
use crate::report::{AudioReport, Detection, Report};
use crate::score;

/// Sample rate analysis runs at.
///
/// Half of 44.1 kHz, and above twice the highest tone any supported mode
/// transmits (2300 Hz). Analysis is a fixed cost per second of audio, so
/// choosing the lowest rate that keeps every tone of interest halves the work
/// of the FFT pass and shrinks every stored trajectory.
pub const ANALYSIS_RATE_HZ: u32 = 22_050;

/// A candidate must score at least this well to be reported as a decode.
const ACCEPT_SCORE: f64 = 0.55;

/// Fraction of a transmission that must be present before an *inferred* mode
/// is reported.
///
/// The tail-padding that lets a truncated recording decode also lets a wrong
/// hypothesis produce a plausible-looking image from whatever fragment of the
/// picture happens to align with it. Requiring most of the transmission to be
/// present is what separates "a partial transmission, correctly identified"
/// from "the wrong mode, fitted to a third of a signal".
///
/// The threshold sits below half deliberately, so a recording that genuinely
/// stops mid-transmission still decodes; it is the *combination* of low
/// coverage and an inferred mode that is rejected.
const MIN_INFERRED_COVERAGE: f64 = 0.50;

/// How many candidates to carry through to full-resolution decoding.
///
/// Small on purpose: the raster backend is by far the most expensive stage, and
/// by the time ranking has run, the correct hypothesis is normally the top one
/// with a clear margin. A handful of runners-up is kept so that a near-tie —
/// or a candidate whose grid looks right but whose raster decode fails — still
/// has a fallback.
pub const DEFAULT_MAX_CANDIDATES: usize = 4;

/// Tunable behaviour for a decode run.
#[derive(Debug, Clone)]
pub struct Options {
    /// Channel selection strategy.
    pub channel: ChannelChoice,
    /// Whether to fall back to sync-period inference when VIS is absent.
    pub blind: bool,
    /// How many ranked candidates to decode at full resolution.
    pub max_candidates: usize,
    /// Whether to write every rejected candidate as a PNG too.
    pub keep_candidates: bool,
    /// Decode only this mode, skipping detection.
    ///
    /// Useful when a recording is too damaged for automatic detection but the
    /// mode is known, for example from the challenge description. The start
    /// position, frequency offset and clock are still measured from the
    /// signal.
    pub forced_mode: Option<String>,
    /// Print per-stage diagnostics to stderr.
    pub verbose: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            channel: ChannelChoice::Auto,
            blind: true,
            max_candidates: DEFAULT_MAX_CANDIDATES,
            keep_candidates: false,
            forced_mode: None,
            verbose: false,
        }
    }
}

/// What a finished run produced.
#[derive(Debug, Clone)]
pub struct Outcome {
    /// Detections that were accepted, best first.
    pub detections: Vec<Detection>,
    /// Path of the written `report.json`.
    pub report_path: PathBuf,
    /// Whether the recording looked like SSTV at all.
    pub plausible_sstv: bool,
    /// Human-readable warnings.
    pub warnings: Vec<String>,
}

impl Outcome {
    /// Whether anything was decoded.
    #[must_use]
    pub fn decoded_anything(&self) -> bool {
        !self.detections.is_empty()
    }
}

/// Decode an audio file, writing PNGs and a report into `output_dir`.
///
/// # Errors
///
/// Fails when the input cannot be read or decoded, when the output directory
/// cannot be created, or when a PNG cannot be written. A recording that simply
/// contains no SSTV is not an error: it returns an [`Outcome`] with no
/// detections and a warning explaining what was and was not found.
pub fn run(input: &Path, output_dir: &Path, options: &Options) -> Result<Outcome> {
    let mut warnings = Vec::new();

    // --- 1. Ingest ---------------------------------------------------------
    let audio = audio::load(input, options.channel)?;
    let duration = audio.duration_seconds();
    if options.verbose {
        eprintln!(
            "audio: {duration:.2}s, {} Hz, {} channel(s) -> {}",
            audio.sample_rate, audio.original_channels, audio.channel_label
        );
    }
    if duration < 1.0 {
        warnings.push(format!(
            "the recording is only {duration:.2}s long; complete SSTV images take at least \
             8s (Robot 24) and commonly 36s or more, so a full decode is unlikely"
        ));
    }
    if audio::is_silent(&audio.samples) {
        warnings.push("the recording is digital silence".to_owned());
        std::fs::create_dir_all(output_dir)
            .with_context(|| format!("create {}", output_dir.display()))?;
        let report = Report {
            tool_version: env!("CARGO_PKG_VERSION").to_owned(),
            audio: describe_audio(input, &audio),
            hypotheses_evaluated: 0,
            detections: Vec::new(),
            warnings: warnings.clone(),
        };
        let report_path = output_dir.join("report.json");
        report.write(&report_path)?;
        return Ok(Outcome {
            detections: Vec::new(),
            report_path,
            plausible_sstv: false,
            warnings,
        });
    }

    // --- 2. Measure --------------------------------------------------------
    // Analysis runs at a fixed 22.05 kHz. Resampling here, once, keeps the
    // cost of every later stage independent of the file's own rate and lets
    // the line-timing arithmetic work in one unit everywhere else.
    let (analysis_samples, analysis_rate) = to_analysis_rate(&audio.samples, audio.sample_rate);
    let analyzer = Analyzer::new(analysis_rate);
    let track = analyzer.track(&analysis_samples);
    if track.is_empty() {
        warnings.push("the recording is too short to analyse".to_owned());
    }
    let trajectory = Trajectory::from_track(&track, analysis_samples.len());

    // Resolve an explicit mode request before detection, so a bad name fails
    // fast with a useful message instead of after a full analysis pass.
    let forced: Option<Mode> = match &options.forced_mode {
        Some(query) => match modes::from_name(query) {
            Some(mode) => Some(mode),
            None => {
                return Err(anyhow!(
                    "unknown mode `{query}`; supported modes are: {}",
                    modes::slug_list()
                ));
            }
        },
        None => None,
    };

    // --- 3. Detect ---------------------------------------------------------
    let mut analysis = autodetect::analyze(&trajectory);
    if let Some(mode) = forced {
        // Keep the measurement but override which mode is attempted. The
        // start position, offset and clock still come from the signal, so a
        // forced mode is a restriction, not a set of guesses.
        for hypothesis in &mut analysis.hypotheses {
            hypothesis.mode = mode;
            hypothesis.origin = Origin::SyncPeriod;
            hypothesis
                .notes
                .insert(0, format!("mode forced to {} by the caller", mode.name));
        }
        analysis
            .hypotheses
            .dedup_by(|a, b| a.skip_sample == b.skip_sample);
        if options.verbose {
            eprintln!(
                "mode: forced to {} ({} hypothesis/hypotheses)",
                mode.name,
                analysis.hypotheses.len()
            );
        }
    }
    if options.verbose {
        eprintln!(
            "detect: {} sync pulse(s), {} VIS burst(s), {} hypothesis/hypotheses, \
             frequency offset {:+.1} Hz",
            analysis.pulses.len(),
            analysis.vis_hits.len(),
            analysis.hypotheses.len(),
            analysis.frequency_shift_hz
        );
    }
    if !analysis.plausible_sstv {
        warnings.push(
            "no SSTV structure was found: there are no line-sync pulses and no VIS header. \
             The file may not contain an SSTV transmission, or the signal may be buried in \
             noise."
                .to_owned(),
        );
    }

    // --- 4. Rank -----------------------------------------------------------
    let mut candidates: Vec<Candidate> = analysis
        .hypotheses
        .iter()
        .filter(|hypothesis| options.blind || hypothesis.origin.is_authoritative())
        .filter_map(|hypothesis| autodetect::evaluate(hypothesis, &trajectory, &analyzer))
        .collect();
    candidates.sort_by(|a, b| b.score.total_cmp(&a.score));

    if options.verbose {
        eprintln!("rank: {} candidate(s)", candidates.len());
        for candidate in candidates.iter().take(options.max_candidates) {
            eprintln!(
                "  {:<10} by {:<17} score {:.3} (agreement {:.3}, timing {:.3}, coverage {:.2})",
                candidate.hypothesis.mode.name,
                candidate.hypothesis.origin.label(),
                candidate.score,
                candidate.agreement,
                candidate.timing_score,
                candidate.coverage
            );
        }
    }

    if candidates.is_empty() && analysis.plausible_sstv {
        warnings.push(
            "sync pulses were found but no mode's line period matched them consistently \
             enough to attempt a decode"
                .to_owned(),
        );
    }

    // --- 5a. Extract to grids for scoring ---------------------------------
    let mut scored: Vec<(Candidate, f64)> = Vec::new();
    for candidate in candidates {
        let quality = grid_quality(&candidate);
        scored.push((candidate, quality.clamp(0.0, 1.0)));
    }
    scored.sort_by(|a, b| combined(b).total_cmp(&combined(a)));

    // --- 5b. Reject weak and redundant matches ----------------------------
    //
    // Three independent reasons to drop a candidate, applied in order:
    //
    // 1. Its grid does not look like a picture at all.
    // 2. It is an inferred mode and either scores too low or covers too little
    //    of the transmission.
    // 3. A *better* candidate already explains the same signal: a mode whose
    //    line period divides another's latches onto every second pulse of a
    //    transmission it is not present in, and would otherwise be reported
    //    beside the correct answer.
    let mut accepted_grids: Vec<Candidate> = Vec::new();
    for (candidate, grid_quality) in &scored {
        if *grid_quality <= 0.0 {
            continue;
        }
        let hypothesis = &candidate.hypothesis;
        if !hypothesis.origin.is_authoritative()
            && (candidate.score < ACCEPT_SCORE || candidate.coverage < MIN_INFERRED_COVERAGE)
        {
            continue;
        }
        let redundant = accepted_grids
            .iter()
            .any(|better| autodetect::is_explained_by(candidate, better));
        if redundant {
            if options.verbose {
                eprintln!(
                    "  {:<10} dropped: a better candidate already explains this signal",
                    hypothesis.mode.name
                );
            }
            continue;
        }
        accepted_grids.push(candidate.clone());
    }

    if options.verbose {
        for (candidate, _) in &scored {
            let hypothesis = &candidate.hypothesis;
            if hypothesis.origin.is_authoritative() || candidate.score < ACCEPT_SCORE {
                continue;
            }
            if candidate.coverage < MIN_INFERRED_COVERAGE {
                eprintln!(
                    "  {:<10} rejected: only {:.0}% of the transmission is present",
                    hypothesis.mode.name,
                    candidate.coverage * 100.0
                );
            } else {
                eprintln!(
                    "  {:<10} accepted: {:.0}% present, {} sync pulses matched a {:.0} ms period",
                    hypothesis.mode.name,
                    candidate.coverage * 100.0,
                    hypothesis.matched_syncs,
                    hypothesis.mode.line_seconds * 1000.0
                );
            }
        }
    }

    if accepted_grids.is_empty() && analysis.plausible_sstv {
        let best = scored
            .first()
            .map(|(candidate, _)| candidate.score)
            .unwrap_or(0.0);
        warnings.push(format!(
            "SSTV structure was found but no mode explained it well enough to decode \
             (best score {best:.2}, threshold {ACCEPT_SCORE:.2})"
        ));
    }

    std::fs::create_dir_all(output_dir)
        .with_context(|| format!("create {}", output_dir.display()))?;

    // --- 5c + 6. Full-resolution decode and report ------------------------
    let mut detections = Vec::new();
    let candidate_dir = output_dir.join("candidates");
    if options.keep_candidates {
        std::fs::create_dir_all(&candidate_dir)
            .with_context(|| format!("create {}", candidate_dir.display()))?;
    }

    for (index, candidate) in accepted_grids
        .iter()
        .take(options.max_candidates)
        .enumerate()
    {
        // Two distinct situations, and they need opposite handling.
        //
        // When the mode came from a VIS header, that header is *in the
        // recording*. The backend must be handed the audio from the very
        // beginning so it can read the header itself and then decode the image
        // that follows. Trimming to the payload position would delete the
        // header the backend is looking for, which is why a VIS-detected
        // candidate fails if it is offset like an inferred one.
        //
        // When the mode was inferred from sync timing there is no header to
        // read, so one is synthesized from the mode and the measured frequency
        // offset and prepended, and the audio is trimmed to the first sync
        // pulse.
        let (inject_vis, start_sample) = if candidate.hypothesis.origin == Origin::Vis {
            (false, 0.0)
        } else {
            let scaled = candidate.hypothesis.skip_sample * f64::from(audio.sample_rate)
                / f64::from(analysis_rate);
            (true, scaled)
        };

        let decoded = backend::decode(
            &audio.samples,
            audio.sample_rate,
            &candidate.hypothesis,
            start_sample,
            inject_vis,
        )
        .with_context(|| format!("decode {} candidate", candidate.hypothesis.mode.name))?;

        let Some(image) = decoded else {
            if options.verbose {
                eprintln!(
                    "  {:<10} candidate {index} produced no full-resolution image",
                    candidate.hypothesis.mode.name
                );
            }
            continue;
        };

        let degenerate = score::is_degenerate(&image.quality);
        let noisy = score::looks_like_noise(&image.quality);
        if (degenerate || noisy) && !candidate.hypothesis.origin.is_authoritative() {
            if options.verbose {
                eprintln!(
                    "  {:<10} candidate {index} rejected: {}",
                    candidate.hypothesis.mode.name,
                    if degenerate {
                        "flat image"
                    } else {
                        "noise-like image"
                    }
                );
            }
            if options.keep_candidates {
                let path = candidate_dir.join(format!(
                    "candidate-{:03}-{}.png",
                    index + 1,
                    candidate.hypothesis.mode.short_name
                ));
                backend::save_png(&image, &path)?;
            }
            continue;
        }

        let path = output_dir.join(format!(
            "{:03}-{}.png",
            detections.len() + 1,
            candidate.hypothesis.mode.short_name
        ));
        backend::save_png(&image, &path)?;

        if options.verbose {
            eprintln!(
                "  {:<10} decoded {}x{} (quality {:.2}) -> {}",
                candidate.hypothesis.mode.name,
                image.width,
                image.height,
                image.quality.score,
                path.display()
            );
        }

        detections.push(build_detection(candidate, &image, analysis_rate, &path));
    }

    if options.keep_candidates {
        for (index, (candidate, _)) in scored.iter().enumerate() {
            if accepted_grids
                .iter()
                .take(options.max_candidates)
                .any(|kept| {
                    kept.hypothesis.mode.mode == candidate.hypothesis.mode.mode
                        && (kept.hypothesis.skip_sample - candidate.hypothesis.skip_sample).abs()
                            < 1.0
                })
            {
                continue;
            }
            let grid = &candidate.grid;
            let pixels = grid.to_rgb();
            let path = candidate_dir.join(format!(
                "candidate-{:03}-{}.png",
                index + 1,
                candidate.hypothesis.mode.short_name
            ));
            write_rgb(&pixels, grid.width, grid.height, &path)?;
        }
    }

    if detections.is_empty() && analysis.plausible_sstv {
        warnings.push(
            "SSTV structure was present, but the full-resolution decoder did not produce a \
             usable image. The recording may be too damaged or truncated."
                .to_owned(),
        );
    }

    let report = Report {
        tool_version: env!("CARGO_PKG_VERSION").to_owned(),
        audio: describe_audio(input, &audio),
        hypotheses_evaluated: analysis.hypotheses.len(),
        detections: detections.clone(),
        warnings: warnings.clone(),
    };
    let report_path = output_dir.join("report.json");
    report.write(&report_path)?;

    Ok(Outcome {
        detections,
        report_path,
        plausible_sstv: analysis.plausible_sstv,
        warnings,
    })
}

/// Score used to order candidates after grid extraction.
fn combined(pair: &(Candidate, f64)) -> f64 {
    pair.0.score * 0.7 + pair.1 * 0.3
}

/// Plausibility of a candidate's recovered grid.
///
/// A grid that scores zero is one the caller should not bother decoding at
/// full resolution: the ranking already decided the hypothesis fits the
/// signal, but the picture it implies still has to look like a picture.
fn grid_quality(candidate: &Candidate) -> f64 {
    let grid = &candidate.grid;
    let pixels = grid.to_rgb();
    score::measure(&pixels, grid.width, grid.height).score
}

/// Build the reported detection for a successfully decoded image.
fn build_detection(
    candidate: &Candidate,
    image: &backend::Image,
    analysis_rate: u32,
    path: &Path,
) -> Detection {
    let hypothesis = &candidate.hypothesis;
    let ambiguous = autodetect::indistinguishable_from(&hypothesis.mode)
        .into_iter()
        .map(|mode| mode.name.to_owned())
        .collect();

    // Confidence blends how well the hypothesis explained the signal with how
    // plausible the resulting picture is. An authoritative VIS header is
    // strong evidence in its own right, but it is not proof: the header can be
    // intact while the image data is destroyed, so image quality still counts.
    let header_bonus = if hypothesis.origin == Origin::Vis {
        0.10
    } else {
        0.0
    };
    let confidence = (0.55 * candidate.agreement
        + 0.20 * candidate.timing_score
        + 0.15 * candidate.coverage
        + 0.10 * image.quality.score
        + header_bonus)
        .min(1.0);

    Detection {
        mode: hypothesis.mode.name.to_owned(),
        mode_slug: hypothesis.mode.short_name.to_owned(),
        vis_code: hypothesis.mode.vis_code,
        detected_by: hypothesis.origin.label().to_owned(),
        confidence,
        signal_agreement: candidate.agreement,
        coverage: candidate.coverage,
        complete: hypothesis.complete,
        image_start_seconds: hypothesis.skip_sample / f64::from(analysis_rate),
        frequency_offset_hz: hypothesis.frequency_shift_hz,
        clock_rate: hypothesis.clock_rate,
        clock_error_percent: (hypothesis.clock_rate - 1.0) * 100.0,
        matched_syncs: hypothesis.matched_syncs,
        width: image.width,
        height: image.height,
        image_quality: image.quality.score,
        evidence: hypothesis.notes.clone(),
        ambiguous_with: ambiguous,
        output_png: path.to_path_buf(),
    }
}

fn describe_audio(input: &Path, audio: &audio::Audio) -> AudioReport {
    AudioReport {
        input: input.to_path_buf(),
        duration_seconds: audio.duration_seconds(),
        analysis_rate_hz: ANALYSIS_RATE_HZ,
        source_channels: audio.original_channels,
        selected_channel: audio.channel_label.clone(),
        normalized_peak: (crate::dsp::peak(&audio.samples) as f32).min(1.0),
    }
}

/// Move audio to the analysis sample rate.
///
/// Beyond resampling this also band-limits by decimation when the source rate
/// is higher, so aliasing cannot fold high-frequency noise down into the SSTV
/// band. The filter is a short moving average, which is cheap and has a
/// well-defined stop band; the analysis band ends at 2300 Hz and the analysis
/// rate is 22.05 kHz, so there is ample margin.
fn to_analysis_rate(samples: &[f32], rate: u32) -> (Vec<f32>, u32) {
    if rate == 0 || samples.is_empty() {
        return (Vec::new(), ANALYSIS_RATE_HZ);
    }
    if rate == ANALYSIS_RATE_HZ {
        return (samples.to_vec(), ANALYSIS_RATE_HZ);
    }
    if rate > ANALYSIS_RATE_HZ {
        let ratio = f64::from(rate) / f64::from(ANALYSIS_RATE_HZ);
        // Decimate in two steps: average `ratio` input samples, then linearly
        // interpolate the remainder. Averaging is the anti-alias filter.
        let width = ratio.floor().max(1.0) as usize;
        let filtered = box_filter(samples, width);
        let effective = rate as f64 / width as f64;
        (
            crate::dsp::resample_linear(&filtered, effective.round() as u32, ANALYSIS_RATE_HZ),
            ANALYSIS_RATE_HZ,
        )
    } else {
        (
            crate::dsp::resample_linear(samples, rate, ANALYSIS_RATE_HZ),
            ANALYSIS_RATE_HZ,
        )
    }
}

/// Moving average with the given window width.
fn box_filter(samples: &[f32], width: usize) -> Vec<f32> {
    if width <= 1 || samples.len() < width {
        return samples.to_vec();
    }
    let mut out = Vec::with_capacity(samples.len() / width);
    let mut index = 0usize;
    while index + width <= samples.len() {
        let sum: f32 = samples[index..index + width].iter().sum();
        #[allow(clippy::cast_precision_loss)]
        out.push(sum / width as f32);
        index += width;
    }
    out
}

/// Write an RGB pixel buffer to a PNG.
fn write_rgb(pixels: &[[u8; 3]], width: u32, height: u32, path: &Path) -> Result<()> {
    let expected = (width as usize) * (height as usize);
    if pixels.len() < expected {
        return Err(anyhow!(
            "refusing to write {}: {} pixels for a {width}x{height} image",
            path.display(),
            pixels.len()
        ));
    }
    let mut buffer = image::RgbImage::new(width, height);
    for (destination, source) in buffer.pixels_mut().zip(pixels.iter()) {
        *destination = image::Rgb(*source);
    }
    buffer
        .save(path)
        .with_context(|| format!("write {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn box_filter_averages_in_blocks() {
        let filtered = box_filter(&[1.0, 3.0, 5.0, 7.0], 2);
        assert_eq!(filtered, vec![2.0, 6.0]);
        // Degenerate widths pass the signal through unchanged.
        assert_eq!(box_filter(&[1.0, 2.0], 1), vec![1.0, 2.0]);
        assert_eq!(box_filter(&[1.0, 2.0], 9), vec![1.0, 2.0]);
    }

    #[test]
    fn analysis_rate_is_exact_for_a_matching_input() {
        let samples = vec![0.5_f32; 100];
        let (out, rate) = to_analysis_rate(&samples, ANALYSIS_RATE_HZ);
        assert_eq!(rate, ANALYSIS_RATE_HZ);
        assert_eq!(out, samples);
    }

    #[test]
    fn downsampling_preserves_a_tone() {
        let rate = 44_100u32;
        let count = rate as usize;
        let tone: Vec<f32> = (0..count)
            .map(|i| {
                (2.0 * std::f64::consts::PI * 1900.0 * i as f64 / f64::from(rate)).sin() as f32
            })
            .collect();
        let (out, analysis_rate) = to_analysis_rate(&tone, rate);
        assert_eq!(analysis_rate, ANALYSIS_RATE_HZ);
        let analyzer = Analyzer::new(analysis_rate);
        let track = analyzer.track(&out);
        let measured = track.hz_at(track.len() / 2);
        assert!((measured - 1900.0).abs() < 30.0, "measured {measured}");
    }

    #[test]
    fn upsampling_preserves_a_tone() {
        let rate = 8_000u32;
        let count = rate as usize;
        let tone: Vec<f32> = (0..count)
            .map(|i| {
                (2.0 * std::f64::consts::PI * 1200.0 * i as f64 / f64::from(rate)).sin() as f32
            })
            .collect();
        let (out, analysis_rate) = to_analysis_rate(&tone, rate);
        assert_eq!(analysis_rate, ANALYSIS_RATE_HZ);
        let analyzer = Analyzer::new(analysis_rate);
        let track = analyzer.track(&out);
        let measured = track.hz_at(track.len() / 2);
        assert!((measured - 1200.0).abs() < 30.0, "measured {measured}");
    }

    #[test]
    fn degenerate_audio_rate_is_handled() {
        let (out, rate) = to_analysis_rate(&[], 44_100);
        assert!(out.is_empty());
        assert_eq!(rate, ANALYSIS_RATE_HZ);
        let (out, _) = to_analysis_rate(&[1.0, 2.0], 0);
        assert!(out.is_empty());
    }

    #[test]
    fn write_rgb_rejects_a_short_buffer() {
        let path = std::env::temp_dir().join("sstv_auto_short.png");
        let error = write_rgb(&[[0, 0, 0]; 2], 8, 8, &path).unwrap_err();
        assert!(format!("{error}").contains("refusing"));
        assert!(!path.exists());
    }

    #[test]
    fn write_rgb_round_trips() {
        let path = std::env::temp_dir().join("sstv_auto_rgb_roundtrip.png");
        let pixels: Vec<[u8; 3]> = (0..16).map(|i| [i * 15, 0, 255 - i * 15]).collect();
        write_rgb(&pixels, 4, 4, &path).expect("write");
        let loaded = image::open(&path).expect("reopen").to_rgb8();
        assert_eq!(loaded.dimensions(), (4, 4));
        assert_eq!(loaded.get_pixel(1, 0).0, pixels[1]);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn running_against_a_missing_file_reports_the_path() {
        let options = Options::default();
        let error = run(
            Path::new("no-such-recording.wav"),
            &std::env::temp_dir().join("sstv_auto_missing"),
            &options,
        )
        .unwrap_err();
        let message = format!("{error:#}");
        assert!(
            message.contains("no-such-recording.wav"),
            "message {message}"
        );
    }
}
