//! Audio ingestion: container/codec decoding, channel selection, and level
//! safety.
//!
//! Decoding is **pure Rust** via Symphonia, which covers WAV/PCM, FLAC, MP3,
//! AAC/MP4, Ogg/Vorbis, MKV, AIFF and CAF. Nothing here shells out to an
//! external binary: the tool is expected to work on a machine that has only a
//! Rust toolchain installed.
//!
//! Channel selection matters more than it first appears. An SSTV recording is
//! mono in principle, but CTF audio is often shipped as stereo with the
//! signal on one side only, or with the two sides at different levels. Rather
//! than guessing, each channel is scored on how much energy it carries in the
//! SSTV band and the strongest is chosen.

use std::fs::File;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use symphonia::core::audio::{SampleBuffer, SignalSpec};
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use symphonia::default::{get_codecs, get_probe};

use crate::dsp::{Analyzer, rms};

/// Which audio channel to decode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelChoice {
    /// Score every channel and pick the one most likely to hold the signal.
    Auto,
    /// Average all channels.
    Mix,
    /// Use one specific zero-based channel.
    Index(usize),
}

impl ChannelChoice {
    /// Parse a `--channel` value.
    ///
    /// # Errors
    ///
    /// Returns a message naming the accepted forms when `value` is not one of
    /// them.
    pub fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "mix" | "mono" | "average" => Ok(Self::Mix),
            "left" | "l" => Ok(Self::Index(0)),
            "right" | "r" => Ok(Self::Index(1)),
            other => other.parse::<usize>().map(Self::Index).map_err(|_| {
                anyhow!("expected `auto`, `mix`, `left`, `right`, or a zero-based channel index")
            }),
        }
    }
}

/// Decoded audio, already reduced to a single channel.
#[derive(Debug, Clone)]
pub struct Audio {
    /// Mono samples, DC-removed and normalised to a safe peak.
    pub samples: Vec<f32>,
    /// Sample rate of `samples` in Hz.
    pub sample_rate: u32,
    /// Channel count of the source file.
    pub original_channels: usize,
    /// Human-readable description of the channel that was selected.
    pub channel_label: String,
    /// Path the audio came from.
    pub source: PathBuf,
}

impl Audio {
    /// Duration in seconds.
    #[must_use]
    pub fn duration_seconds(&self) -> f64 {
        if self.sample_rate == 0 {
            return 0.0;
        }
        self.samples.len() as f64 / f64::from(self.sample_rate)
    }
}

/// Peak level the decoder normalises to. Leaving headroom avoids clipping
/// artefacts in the analysis FFT.
const TARGET_PEAK: f32 = 0.95;

/// Load and prepare an audio file for analysis.
///
/// # Errors
///
/// Fails with a message naming the file when it cannot be read, is not
/// recognisably audio, or decodes to no samples.
pub fn load(path: &Path, channel: ChannelChoice) -> Result<Audio> {
    let probe = probe_file(path)?;
    let mut audio = decode(path, &probe, channel)?;
    prepare(&mut audio.samples);
    Ok(audio)
}

/// Basic facts about an audio file, gathered before decoding.
#[derive(Debug, Clone)]
pub struct Probe {
    /// Sample rate advertised by the container.
    pub sample_rate: u32,
    /// Channel count advertised by the container.
    pub channels: usize,
    /// Codec name, for diagnostics.
    pub codec: String,
}

impl Probe {
    /// Sample rate is only meaningful for real audio.
    #[must_use]
    pub fn is_plausible(&self) -> bool {
        (3_000..=768_000).contains(&self.sample_rate) && self.channels > 0 && self.channels <= 64
    }
}

/// Inspect a file's container and first track without decoding it.
///
/// # Errors
///
/// Fails when the file cannot be opened or does not look like audio.
pub fn probe_file(path: &Path) -> Result<Probe> {
    let meta = std::fs::metadata(path).with_context(|| {
        format!(
            "cannot read {}: file is missing or unreadable",
            path.display()
        )
    })?;
    if meta.is_dir() {
        bail!("{} is a directory, not an audio file", path.display());
    }
    if meta.len() == 0 {
        bail!("{} is empty", path.display());
    }

    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let stream = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = get_probe()
        .format(
            &hint,
            stream,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|err| {
            anyhow!(
                "{} is not a recognisable audio file ({err}). Supported containers include \
                 WAV, FLAC, MP3, AAC/MP4, Ogg/Vorbis, AIFF, CAF and MKV.",
                path.display()
            )
        })?;

    let track = probed
        .format
        .default_track()
        .ok_or_else(|| anyhow!("{} contains no audio track", path.display()))?;
    let params = &track.codec_params;
    Ok(Probe {
        sample_rate: params.sample_rate.unwrap_or(0),
        channels: params
            .channels
            .map_or(0, symphonia::core::audio::Channels::count),
        codec: format!("{:?}", params.codec),
    })
}

fn decode(path: &Path, probe: &Probe, channel: ChannelChoice) -> Result<Audio> {
    if !probe.is_plausible() {
        bail!(
            "{} reports an unusable audio format ({} Hz, {} channel(s))",
            path.display(),
            probe.sample_rate,
            probe.channels
        );
    }

    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let stream = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = get_probe()
        .format(
            &hint,
            stream,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .with_context(|| format!("probe {}", path.display()))?;

    let mut format = probed.format;
    let track = format
        .default_track()
        .ok_or_else(|| anyhow!("{} contains no audio track", path.display()))?;
    let track_id = track.id;
    let mut decoder = get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .with_context(|| format!("no decoder available for {}'s codec", path.display()))?;

    let mut channels: Vec<Vec<f32>> = Vec::new();
    let mut sample_rate = probe.sample_rate;

    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            // A file that ends mid-packet is normal for damaged CTF audio;
            // whatever decoded so far is still usable.
            Err(SymphoniaError::IoError(err))
                if err.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(SymphoniaError::ResetRequired) => bail!(
                "{} changes audio parameters part-way through, which this build does not support",
                path.display()
            ),
            Err(err) => return Err(err).with_context(|| format!("read {}", path.display())),
        };

        if packet.track_id() != track_id {
            continue;
        }

        // A single corrupt frame must not abandon a whole recording.
        let decoded = match decoder.decode(&packet) {
            Ok(decoded) => decoded,
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(err) => return Err(err).with_context(|| format!("decode {}", path.display())),
        };

        let spec: SignalSpec = *decoded.spec();
        let count = spec.channels.count();
        if count == 0 {
            continue;
        }
        if channels.is_empty() {
            channels.resize_with(count, Vec::new);
            sample_rate = spec.rate;
        }
        if count != channels.len() {
            bail!(
                "{} changes its channel count part-way through",
                path.display()
            );
        }

        let mut buffer = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
        buffer.copy_interleaved_ref(decoded);
        for frame in buffer.samples().chunks_exact(count) {
            for (index, sample) in frame.iter().enumerate() {
                channels[index].push(*sample);
            }
        }
    }

    if channels.is_empty() || channels.iter().all(Vec::is_empty) {
        bail!("{} decoded to no audio samples", path.display());
    }
    let frames = channels.iter().map(Vec::len).min().unwrap_or(0);
    if frames == 0 {
        bail!("{} decoded to no audio samples", path.display());
    }

    let (samples, channel_label) = select_channel(&channels, sample_rate, channel)?;
    Ok(Audio {
        samples,
        sample_rate: if sample_rate == 0 {
            probe.sample_rate
        } else {
            sample_rate
        },
        original_channels: channels.len(),
        channel_label,
        source: path.to_path_buf(),
    })
}

/// Choose the audio channel most likely to contain the SSTV signal.
fn select_channel(
    channels: &[Vec<f32>],
    sample_rate: u32,
    choice: ChannelChoice,
) -> Result<(Vec<f32>, String)> {
    match choice {
        ChannelChoice::Index(index) => {
            let channel = channels.get(index).ok_or_else(|| {
                anyhow!(
                    "channel {index} was requested but the file has {} channel(s)",
                    channels.len()
                )
            })?;
            Ok((channel.clone(), format!("channel {index}")))
        }
        ChannelChoice::Mix => Ok((mix(channels), "mix of all channels".to_owned())),
        ChannelChoice::Auto => {
            if channels.len() == 1 {
                return Ok((channels[0].clone(), "channel 0 (mono)".to_owned()));
            }
            let mut scored: Vec<(f64, usize, Vec<f32>)> = channels
                .iter()
                .enumerate()
                .map(|(index, channel)| (band_score(channel, sample_rate), index, channel.clone()))
                .collect();
            if channels.len() > 1 {
                let mixed = mix(channels);
                scored.push((band_score(&mixed, sample_rate), usize::MAX, mixed));
            }
            scored.sort_by(|a, b| b.0.total_cmp(&a.0));
            let (score, index, samples) = scored
                .into_iter()
                .next()
                .ok_or_else(|| anyhow!("no audio channels to choose from"))?;
            let label = if index == usize::MAX {
                "mix (best band activity)".to_owned()
            } else {
                format!("channel {index} (band activity {score:.1})")
            };
            Ok((samples, label))
        }
    }
}

/// Average several channels.
fn mix(channels: &[Vec<f32>]) -> Vec<f32> {
    let frames = channels.iter().map(Vec::len).min().unwrap_or(0);
    if frames == 0 || channels.is_empty() {
        return Vec::new();
    }
    #[allow(clippy::cast_precision_loss)]
    let scale = 1.0 / channels.len() as f32;
    let mut out = vec![0.0_f32; frames];
    for channel in channels {
        for (destination, source) in out.iter_mut().zip(channel.iter().take(frames)) {
            *destination += source * scale;
        }
    }
    out
}

/// Score how much of a channel's energy sits in the SSTV band rather than
/// outside it.
///
/// A channel carrying the signal has a dominant tone between roughly 1100 Hz
/// and 2300 Hz for most of its length. Speech, music and hiss do not, so the
/// ratio between in-band and out-of-band tone purity separates them.
fn band_score(samples: &[f32], sample_rate: u32) -> f64 {
    if samples.len() < 1024 || sample_rate == 0 {
        return 0.0;
    }
    // Cap the analysis to the first 30 s: enough to find the transmission,
    // cheap enough to run on every channel of a long recording.
    let limit = (f64::from(sample_rate) * 30.0) as usize;
    let slice = &samples[..samples.len().min(limit)];
    let analyzer = Analyzer::new(sample_rate);
    let track = analyzer.track(slice);
    if track.is_empty() {
        return 0.0;
    }
    let mut in_band = 0usize;
    let mut total = 0usize;
    let mut step = (track.len() / 4000).max(1);
    if slice.len() < analyzer.window_len() * 4 {
        step = 1;
    }
    let mut index = 0usize;
    while index < track.len() {
        let hz = track.hz_at(index);
        let purity = track.tone_likeness_at_sample((index * track.hop()) as f64);
        if purity > 0.05 {
            total += 1;
            if (1050.0..=2350.0).contains(&hz) {
                in_band += 1;
            }
        }
        index += step;
    }
    if total == 0 {
        return 0.0;
    }
    let ratio = in_band as f64 / total as f64;
    // Weight by the fraction of the file that even looks tonal, so a channel
    // that is mostly silence beats a channel of loud broadband noise.
    ratio * (total as f64 / (track.len() / step).max(1) as f64)
}

/// DC-remove and normalise a signal, guarding against silence.
///
/// The signal is scaled to [`TARGET_PEAK`] both up and down. Scaling *down* a
/// loud recording matters as much as scaling up a quiet one: the analysis is
/// about relative tone levels, so any consistent gain is harmless, while a
/// signal pinned against the full scale carries clipping distortion that is
/// not.
fn prepare(samples: &mut [f32]) {
    if samples.is_empty() {
        return;
    }
    crate::dsp::remove_dc(samples);
    if crate::dsp::peak(samples) <= 1.0e-6 {
        // Digital silence: nothing to normalise, and amplifying it would only
        // turn dither into signal.
        return;
    }
    crate::dsp::normalize_to_peak(samples, TARGET_PEAK);
}

/// Whether a signal is essentially digital silence.
#[must_use]
pub fn is_silent(samples: &[f32]) -> bool {
    crate::dsp::peak(samples) <= 1.0e-6
}

/// Root-mean-square level of a signal, for diagnostics.
#[must_use]
pub fn level(samples: &[f32]) -> f64 {
    rms(samples)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn channel_choice_parses_every_documented_form() {
        assert_eq!(ChannelChoice::parse("auto").ok(), Some(ChannelChoice::Auto));
        assert_eq!(ChannelChoice::parse("AUTO").ok(), Some(ChannelChoice::Auto));
        assert_eq!(ChannelChoice::parse("mix").ok(), Some(ChannelChoice::Mix));
        assert_eq!(ChannelChoice::parse("mono").ok(), Some(ChannelChoice::Mix));
        assert_eq!(
            ChannelChoice::parse("left").ok(),
            Some(ChannelChoice::Index(0))
        );
        assert_eq!(
            ChannelChoice::parse("right").ok(),
            Some(ChannelChoice::Index(1))
        );
        assert_eq!(
            ChannelChoice::parse("3").ok(),
            Some(ChannelChoice::Index(3))
        );
        assert_eq!(
            ChannelChoice::parse(" 2 ").ok(),
            Some(ChannelChoice::Index(2))
        );
    }

    #[test]
    fn channel_choice_rejects_nonsense_with_a_useful_message() {
        let error = ChannelChoice::parse("backwards").unwrap_err();
        let message = format!("{error}");
        assert!(message.contains("mix"), "message was {message}");
        assert!(message.contains("index"), "message was {message}");
    }

    #[test]
    fn probe_plausibility_bounds() {
        let good = Probe {
            sample_rate: 44_100,
            channels: 2,
            codec: "pcm".to_owned(),
        };
        assert!(good.is_plausible());
        for (rate, channels) in [(0, 1), (10, 1), (1_000_000, 1), (44_100, 0), (44_100, 100)] {
            assert!(
                !Probe {
                    sample_rate: rate,
                    channels,
                    codec: String::new()
                }
                .is_plausible(),
                "{rate} Hz / {channels} ch should be rejected"
            );
        }
    }

    #[test]
    fn mix_averages_and_survives_empty_input() {
        let a = vec![1.0_f32, 0.0, -1.0];
        let b = vec![0.0_f32, 1.0, 1.0];
        let mixed = mix(&[a, b]);
        assert_eq!(mixed, vec![0.5, 0.5, 0.0]);
        assert!(mix(&[]).is_empty());
        assert!(mix(&[vec![]]).is_empty());
    }

    #[test]
    fn mix_handles_channel_length_mismatch() {
        let mixed = mix(&[vec![1.0, 1.0, 1.0], vec![1.0]]);
        assert_eq!(mixed.len(), 1);
    }

    #[test]
    fn prepare_silences_dc_and_normalises_quiet_audio() {
        let mut quiet = vec![0.001_f32, -0.001, 0.001];
        prepare(&mut quiet);
        let peak = crate::dsp::peak(&quiet);
        assert!(peak > 0.5, "quiet audio was not boosted (peak {peak})");
        // Mean must be removed.
        let mean = quiet.iter().sum::<f32>() / quiet.len() as f32;
        assert!(mean.abs() < 1e-3);
    }

    #[test]
    fn prepare_leaves_silence_alone() {
        let mut silent = vec![0.0_f32; 64];
        prepare(&mut silent);
        assert!(is_silent(&silent));
        assert_eq!(level(&silent), 0.0);
    }

    #[test]
    fn prepare_never_amplifies_an_already_loud_signal() {
        // A signal louder than the target must be brought down to it, not
        // boosted further into clipping.
        let mut loud = vec![2.0_f32, -2.5, 1.0];
        prepare(&mut loud);
        let peak = crate::dsp::peak(&loud);
        assert!(peak <= f64::from(TARGET_PEAK) + 1e-6, "peak {peak}");
        assert!(loud.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn prepare_preserves_a_signal_already_at_target() {
        // Peak-normalising a signal that is already at the target must leave
        // it alone. The scaling check is on peak level, not per-sample value,
        // because DC removal can shift individual samples.
        let mut at_target = vec![TARGET_PEAK, -TARGET_PEAK, TARGET_PEAK / 2.0, 0.0];
        prepare(&mut at_target);
        let peak = crate::dsp::peak(&at_target);
        assert!((peak - f64::from(TARGET_PEAK)).abs() < 1e-5, "peak {peak}");
    }

    #[test]
    fn band_score_prefers_sstv_content_over_noise() {
        let rate = 22_050;
        let count = rate as usize;
        let sstv: Vec<f32> = (0..count)
            .map(|i| {
                let hz = 1200.0 + 600.0 * ((i / 100) % 3) as f64;
                (2.0 * std::f64::consts::PI * hz * i as f64 / f64::from(rate)).sin() as f32
            })
            .collect();
        let noise: Vec<f32> = (0..count)
            .map(|i| {
                let x = (i as f64 * 12.9898).sin() * 43_758.545_3;
                (x - x.floor()) as f32 * 2.0 - 1.0
            })
            .collect();
        let signal_score = band_score(&sstv, rate);
        let noise_score = band_score(&noise, rate);
        assert!(
            signal_score > noise_score,
            "SSTV {signal_score} should beat noise {noise_score}"
        );
    }

    #[test]
    fn band_score_returns_zero_for_degenerate_input() {
        assert_eq!(band_score(&[], 44_100), 0.0);
        assert_eq!(band_score(&[0.0; 10], 44_100), 0.0);
        assert_eq!(band_score(&[0.0; 10], 0), 0.0);
    }

    #[test]
    fn select_channel_honours_an_explicit_index() {
        let channels = vec![vec![0.0_f32; 100], vec![1.0_f32; 100]];
        let (samples, label) =
            select_channel(&channels, 44_100, ChannelChoice::Index(1)).expect("channel 1 exists");
        assert_eq!(samples[0], 1.0);
        assert!(label.contains('1'));
    }

    #[test]
    fn select_channel_reports_a_missing_index() {
        let channels = vec![vec![0.0_f32; 100]];
        let error = select_channel(&channels, 44_100, ChannelChoice::Index(5)).unwrap_err();
        let message = format!("{error}");
        assert!(message.contains('5'), "message was {message}");
        assert!(message.contains('1'), "message was {message}");
    }

    #[test]
    fn select_channel_picks_the_channel_carrying_the_signal() {
        // Channel 0 is silence, channel 1 carries a clear SSTV-like tone.
        let rate = 22_050;
        let tones: Vec<f32> = (0..rate as usize)
            .map(|i| {
                (2.0 * std::f64::consts::PI * 1900.0 * i as f64 / f64::from(rate)).sin() as f32
            })
            .collect();
        let channels = vec![vec![0.0_f32; tones.len()], tones];
        let (samples, label) =
            select_channel(&channels, rate, ChannelChoice::Auto).expect("auto selection");
        assert!(crate::dsp::peak(&samples) > 0.5, "picked a silent channel");
        assert!(label.contains('1'), "chose {label}");
    }

    #[test]
    fn load_reports_a_missing_file_clearly() {
        let error = load(Path::new("definitely-not-here.wav"), ChannelChoice::Auto).unwrap_err();
        let message = format!("{error:#}");
        assert!(
            message.contains("definitely-not-here.wav"),
            "message was {message}"
        );
    }

    #[test]
    fn probe_rejects_an_empty_file() {
        let dir = std::env::temp_dir().join("sstv_auto_empty_probe");
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let path = dir.join("empty.wav");
        std::fs::write(&path, b"").expect("write empty file");
        let error = probe_file(&path).unwrap_err();
        assert!(format!("{error}").contains("empty"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn probe_rejects_non_audio_content() {
        let dir = std::env::temp_dir().join("sstv_auto_junk_probe");
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let path = dir.join("junk.wav");
        std::fs::write(&path, b"this is definitely not audio at all, sorry").expect("write");
        let error = probe_file(&path).unwrap_err();
        let message = format!("{error}");
        assert!(
            message.contains("recognisable") || message.contains("audio"),
            "message was {message}"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn probe_rejects_a_directory() {
        let dir = std::env::temp_dir().join("sstv_auto_dir_probe");
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let error = probe_file(&dir).unwrap_err();
        assert!(format!("{error}").contains("directory"));
    }
}
