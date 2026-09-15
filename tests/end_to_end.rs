#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]
//! End-to-end tests for the decode pipeline.
//!
//! Two kinds of test live here:
//!
//! * **Synthetic round trips.** A known image is encoded to SSTV audio with
//!   this crate's own synthesizer, then fed through the whole pipeline. These
//!   run everywhere and pin down the behaviour that must not regress.
//! * **The `audio.wav` acceptance test.** The real fixture, run against the
//!   real pipeline, asserted to decode into a valid image. It is skipped with a
//!   printed notice when the file is absent (for example in a fresh clone that
//!   does not carry the fixture), but it is never weakened when the file *is*
//!   present: if it is there, it must decode.

use std::path::{Path, PathBuf};

use sstv_auto::audio::ChannelChoice;
use sstv_auto::modes;
use sstv_auto::pipeline::{self, Options};

/// Directory under the target dir for test scratch space, so tests never write
/// into the source tree.
fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

/// Render a synthetic transmission to a temporary WAV file.
fn write_synthetic(
    directory: &Path,
    name: &str,
    mode: modes::Mode,
    rate: u32,
    shift: f64,
    clock: f64,
    with_vis: bool,
) -> PathBuf {
    let grid = sstv_auto::synth::test_grid(&mode);
    let signal = if with_vis {
        sstv_auto::synth::render_with_vis(&grid, &mode, rate, shift, Some(clock))
    } else {
        sstv_auto::synth::render(&grid, &mode, rate, shift, Some(clock))
    };
    let path = directory.join(name);

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(&path, spec).expect("create wav");
    for sample in &signal {
        let scaled = (f64::from(*sample).clamp(-1.0, 1.0) * 32_000.0).round() as i16;
        writer.write_sample(scaled).expect("write sample");
    }
    writer.finalize().expect("finalize wav");
    path
}

#[test]
fn synthetic_robot36_with_vis_decodes_without_hints() {
    let dir = scratch("synthetic_robot36");
    let path = write_synthetic(
        &dir,
        "robot36.wav",
        modes::from_name("robot36").expect("robot36"),
        44_100,
        0.0,
        1.0,
        true,
    );

    let outcome = pipeline::run(&path, &dir.join("out"), &Options::default()).expect("pipeline");
    assert!(
        outcome.decoded_anything(),
        "no image decoded; warnings: {:?}",
        outcome.warnings
    );

    let best = &outcome.detections[0];
    assert_eq!(best.mode_slug, "robot36");
    assert!(
        best.confidence > 0.5,
        "confidence too low: {}",
        best.confidence
    );
    assert!(image_is_real_content(&best.output_png));
}

#[test]
fn synthetic_pd120_with_vis_decodes_without_hints() {
    let dir = scratch("synthetic_pd120");
    let path = write_synthetic(
        &dir,
        "pd120.wav",
        modes::from_name("pd120").expect("pd120"),
        22_050,
        0.0,
        1.0,
        true,
    );

    let outcome = pipeline::run(&path, &dir.join("out"), &Options::default()).expect("pipeline");
    let detection = outcome
        .detections
        .iter()
        .find(|d| d.mode_slug == "pd120")
        .unwrap_or_else(|| {
            panic!(
                "pd120 not decoded; got {:?}; warnings {:?}",
                outcome
                    .detections
                    .iter()
                    .map(|d| d.mode_slug.clone())
                    .collect::<Vec<_>>(),
                outcome.warnings
            )
        });
    assert_eq!((detection.width, detection.height), (640, 496));
    assert!(image_is_real_content(&detection.output_png));
}

#[test]
fn a_removed_vis_header_is_recovered_from_sync_timing() {
    // The CTF case: the header has been stripped, leaving only the image.
    let dir = scratch("no_vis");
    let path = write_synthetic(
        &dir,
        "headerless.wav",
        modes::from_name("robot36").expect("robot36"),
        44_100,
        0.0,
        1.0,
        false,
    );

    let outcome = pipeline::run(&path, &dir.join("out"), &Options::default()).expect("pipeline");
    assert!(
        outcome.decoded_anything(),
        "headerless recording was not recovered; warnings: {:?}",
        outcome.warnings
    );

    // Robot 36 and Robot 24 share their entire wire format, so either is a
    // correct answer here; what matters is that the image is real.
    let best = &outcome.detections[0];
    assert!(
        best.mode_slug == "robot36" || best.mode_slug == "robot24",
        "unexpected mode {}",
        best.mode_slug
    );
    assert_ne!(
        best.detected_by, "vis",
        "a headerless recording must not claim a VIS detection"
    );
    assert!(
        !best.ambiguous_with.is_empty(),
        "the Robot 24/36 ambiguity must be reported"
    );
    assert!(image_is_real_content(&best.output_png));
}

#[test]
fn a_frequency_offset_is_measured_not_assumed() {
    let dir = scratch("shifted");
    let path = write_synthetic(
        &dir,
        "shifted.wav",
        modes::from_name("robot36").expect("robot36"),
        22_050,
        80.0,
        1.0,
        true,
    );

    let outcome = pipeline::run(&path, &dir.join("out"), &Options::default()).expect("pipeline");
    let best = outcome.detections.first().expect("a detection");
    assert!(
        (best.frequency_offset_hz - 80.0).abs() < 40.0,
        "offset measured as {} Hz, expected about 80",
        best.frequency_offset_hz
    );
}

#[test]
fn a_clock_error_is_measured_and_compensated() {
    let dir = scratch("clock");
    let path = write_synthetic(
        &dir,
        "slow.wav",
        modes::from_name("robot36").expect("robot36"),
        22_050,
        0.0,
        0.98,
        true,
    );

    let outcome = pipeline::run(&path, &dir.join("out"), &Options::default()).expect("pipeline");
    let best = outcome.detections.first().expect("a detection");
    assert!(
        (best.clock_rate - 0.98).abs() < 0.01,
        "clock measured as {}, expected about 0.98",
        best.clock_rate
    );
    assert!(image_is_real_content(&best.output_png));
}

#[test]
fn a_truncated_transmission_is_flagged_incomplete() {
    let dir = scratch("truncated");
    let mode = modes::from_name("robot36").expect("robot36");
    let grid = sstv_auto::synth::test_grid(&mode);
    let signal = sstv_auto::synth::render_with_vis(&grid, &mode, 22_050, 0.0, None);
    // Keep the header and just over half the image.
    let keep = (signal.len() as f64 * 0.55) as usize;
    let path = dir.join("truncated.wav");
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 22_050,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(&path, spec).expect("create wav");
    for sample in &signal[..keep] {
        let scaled = (f64::from(*sample).clamp(-1.0, 1.0) * 32_000.0).round() as i16;
        writer.write_sample(scaled).expect("write");
    }
    writer.finalize().expect("finalize");

    let outcome = pipeline::run(&path, &dir.join("out"), &Options::default()).expect("pipeline");
    let hypothesis = outcome.detections.first().expect("a detection");
    assert!(
        !hypothesis.complete,
        "a half-length recording must not be reported as complete"
    );
    assert!(
        hypothesis.coverage < 0.9,
        "coverage {} should reflect the missing tail",
        hypothesis.coverage
    );
}

#[test]
fn non_sstv_audio_produces_no_false_positive() {
    let dir = scratch("noise");
    let path = dir.join("noise.wav");
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 44_100,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(&path, spec).expect("create wav");
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    for _ in 0..44_100 * 5 {
        // xorshift: deterministic broadband noise.
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        writer.write_sample((state >> 48) as i16).expect("write");
    }
    writer.finalize().expect("finalize");

    let outcome = pipeline::run(&path, &dir.join("out"), &Options::default()).expect("pipeline");
    assert!(
        outcome.detections.is_empty(),
        "broadband noise produced {:?}",
        outcome
            .detections
            .iter()
            .map(|d| d.mode_slug.clone())
            .collect::<Vec<_>>()
    );
    assert!(
        !outcome.plausible_sstv,
        "noise must not be considered plausible SSTV"
    );
}

#[test]
fn digital_silence_is_handled_and_explained() {
    let dir = scratch("silence");
    let path = dir.join("silence.wav");
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 44_100,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(&path, spec).expect("create wav");
    for _ in 0..44_100 * 2 {
        writer.write_sample(0_i16).expect("write");
    }
    writer.finalize().expect("finalize");

    let outcome = pipeline::run(&path, &dir.join("out"), &Options::default()).expect("pipeline");
    assert!(outcome.detections.is_empty());
    assert!(
        outcome.warnings.iter().any(|w| w.contains("silence")),
        "silence should be named in the warnings: {:?}",
        outcome.warnings
    );
}

#[test]
fn a_missing_input_file_is_an_error_not_a_panic() {
    let dir = scratch("missing");
    let error = pipeline::run(
        &dir.join("does-not-exist.wav"),
        &dir.join("out"),
        &Options::default(),
    )
    .expect_err("missing input must fail");
    let message = format!("{error:#}");
    assert!(message.contains("does-not-exist.wav"), "message: {message}");
}

#[test]
fn unknown_input_content_is_rejected_clearly() {
    let dir = scratch("junk");
    let path = dir.join("junk.wav");
    std::fs::write(&path, b"this is not audio in any format").expect("write junk");
    let error =
        pipeline::run(&path, &dir.join("out"), &Options::default()).expect_err("junk must fail");
    let message = format!("{error}");
    assert!(
        message.contains("recognisable") || message.contains("audio"),
        "message: {message}"
    );
}

#[test]
fn an_unknown_forced_mode_is_rejected_by_name() {
    let dir = scratch("bad_mode");
    let path = write_synthetic(
        &dir,
        "robot36.wav",
        modes::from_name("robot36").expect("robot36"),
        22_050,
        0.0,
        1.0,
        true,
    );
    let options = Options {
        forced_mode: Some("martian9".to_owned()),
        ..Options::default()
    };
    let error =
        pipeline::run(&path, &dir.join("out"), &options).expect_err("unknown mode must fail");
    let message = format!("{error}");
    assert!(message.contains("martian9"), "message: {message}");
    assert!(
        message.contains("robot36"),
        "message should list modes: {message}"
    );
}

#[test]
fn forcing_a_mode_still_decodes_it() {
    let dir = scratch("forced_mode");
    let path = write_synthetic(
        &dir,
        "robot36.wav",
        modes::from_name("robot36").expect("robot36"),
        22_050,
        0.0,
        1.0,
        true,
    );
    let options = Options {
        forced_mode: Some("robot36".to_owned()),
        ..Options::default()
    };
    let outcome = pipeline::run(&path, &dir.join("out"), &options).expect("pipeline");
    let best = outcome.detections.first().expect("a detection");
    assert_eq!(best.mode_slug, "robot36");
    assert!(image_is_real_content(&best.output_png));
}

#[test]
fn report_json_is_written_and_parses() {
    let dir = scratch("report");
    let path = write_synthetic(
        &dir,
        "robot36.wav",
        modes::from_name("robot36").expect("robot36"),
        22_050,
        0.0,
        1.0,
        true,
    );
    let outcome = pipeline::run(&path, &dir.join("out"), &Options::default()).expect("pipeline");
    let text = std::fs::read_to_string(&outcome.report_path).expect("read report");
    let value: serde_json::Value = serde_json::from_str(&text).expect("report is valid JSON");
    assert!(value["detections"].is_array());
    assert!(value["audio"]["duration_seconds"].as_f64().unwrap_or(0.0) > 0.0);
}

#[test]
fn explicit_channels_are_honoured() {
    let dir = scratch("channels");
    // Put the signal in the right channel and silence in the left.
    let mode = modes::from_name("robot36").expect("robot36");
    let grid = sstv_auto::synth::test_grid(&mode);
    let signal = sstv_auto::synth::render_with_vis(&grid, &mode, 22_050, 0.0, None);
    let path = dir.join("stereo.wav");
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: 22_050,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(&path, spec).expect("create wav");
    for sample in &signal {
        let scaled = (f64::from(*sample).clamp(-1.0, 1.0) * 32_000.0).round() as i16;
        writer.write_sample(0_i16).expect("left");
        writer.write_sample(scaled).expect("right");
    }
    writer.finalize().expect("finalize");

    // Auto must find the channel carrying the signal.
    let auto = pipeline::run(&path, &dir.join("auto"), &Options::default()).expect("auto");
    assert!(
        auto.decoded_anything(),
        "auto channel selection failed; warnings {:?}",
        auto.warnings
    );

    // Explicitly asking for the silent channel must not invent an image.
    let left = pipeline::run(
        &path,
        &dir.join("left"),
        &Options {
            channel: ChannelChoice::Index(0),
            ..Options::default()
        },
    )
    .expect("left");
    assert!(
        left.detections.is_empty(),
        "the silent channel produced {} detection(s)",
        left.detections.len()
    );
}

/// Load a PNG and confirm it looks like decoded picture content.
///
/// This is the check that separates "a file was written" from "an image was
/// recovered": a plausible PNG must open, have non-zero dimensions, and carry
/// real variation rather than being flat, uniform noise or fully clipped.
fn image_is_real_content(path: &Path) -> bool {
    let Ok(image) = image::open(path) else {
        panic!("{} could not be opened as an image", path.display());
    };
    let rgb = image.to_rgb8();
    let (width, height) = rgb.dimensions();
    assert!(width > 0 && height > 0, "zero-sized image");

    let pixels: Vec<[u8; 3]> = rgb.pixels().map(|p| p.0).collect();
    let quality = sstv_auto::score::measure(&pixels, width, height);

    assert!(
        !sstv_auto::score::is_degenerate(&quality),
        "{} is flat or fully clipped: {quality:?}",
        path.display()
    );
    assert!(
        !sstv_auto::score::looks_like_noise(&quality),
        "{} looks like noise: {quality:?}",
        path.display()
    );
    true
}

/// The mandatory acceptance test: the real `audio.wav` fixture.
///
/// Runs the actual pipeline over the actual file and requires a valid,
/// genuinely-decoded image. The assertions here are deliberately about *what
/// was recovered* — a readable picture of real dimensions with real content —
/// and never about the fixture's size, checksum or expected pixels, so the
/// test stays a black-box check of the decoder.
#[test]
fn audio_wav_fixture_decodes_end_to_end() {
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("audio.wav");
    if !fixture.exists() {
        eprintln!(
            "skipping: {} is not present. The real acceptance test requires this fixture; \
             run it wherever the file is available.",
            fixture.display()
        );
        return;
    }

    let dir = scratch("audio_wav");
    let outcome =
        pipeline::run(&fixture, &dir, &Options::default()).expect("pipeline on audio.wav");

    assert!(
        outcome.decoded_anything(),
        "audio.wav did not decode; warnings: {:?}",
        outcome.warnings
    );

    let best = &outcome.detections[0];
    assert!(
        best.confidence > 0.5,
        "confidence {} is too low to call this a decode",
        best.confidence
    );
    assert!(best.width >= 320 && best.height >= 240, "{best:?}");
    assert!(
        image_is_real_content(&best.output_png),
        "the decoded image is not usable content"
    );

    // Robot 36 and Robot 24 are wire-identical, so either is acceptable; the
    // decoded image is the same either way.
    assert!(
        best.mode_slug == "robot36" || best.mode_slug == "robot24",
        "unexpected mode {}",
        best.mode_slug
    );

    // The fixture has an intact VIS header, so it must be used.
    let vis_detection = outcome.detections.iter().find(|d| d.detected_by == "vis");
    assert!(
        vis_detection.is_some(),
        "audio.wav has a valid VIS header but none was reported; got {:?}",
        outcome
            .detections
            .iter()
            .map(|d| (d.mode_slug.clone(), d.detected_by.clone()))
            .collect::<Vec<_>>()
    );

    eprintln!(
        "audio.wav decoded: {} at {}x{}, confidence {:.0}%, offset {:+.1} Hz, clock {:.4}, \
         matched {} sync pulses -> {}",
        best.mode,
        best.width,
        best.height,
        best.confidence * 100.0,
        best.frequency_offset_hz,
        best.clock_rate,
        best.matched_syncs,
        best.output_png.display()
    );
}
