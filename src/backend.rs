//! The full-resolution raster decoder.
//!
//! Detection, geometry and ranking all happen on the small grid model in
//! [`crate::autodetect`]. Once a hypothesis has been accepted, the actual
//! pixels come from `slowrx`, a mature pure-Rust SSTV raster decoder that
//! handles the parts where fidelity matters most: per-pixel FFT demodulation,
//! sync correlation, slant correction and the YCbCr/RGB conversion for every
//! supported family.
//!
//! This module is the boundary. It takes a hypothesis and a signal and runs
//! the raster decoder at full resolution, then hands back a plain RGB image so
//! the rest of the crate never depends on the backend's types.
//!
//! For the blind path the recording has no VIS header, so one is synthesized
//! from the inferred mode and measured frequency offset, and prepended to the
//! audio. That is the only reason [`crate::vis::synthesize`] exists.

use anyhow::{Context, Result, anyhow};
use slowrx::{SstvDecoder, SstvEvent, SstvImage};

use crate::autodetect::Hypothesis;
use crate::score::{self, Quality};

/// A decoded image at full resolution.
#[derive(Debug, Clone)]
pub struct Image {
    /// Pixels per row.
    pub width: u32,
    /// Rows.
    pub height: u32,
    /// Row-major RGB pixels.
    pub pixels: Vec<[u8; 3]>,
    /// Plausibility measurements.
    pub quality: Quality,
}

impl Image {
    /// Dimensions as `(width, height)`.
    #[must_use]
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}

/// Decode a recording using an explicit hypothesis.
///
/// When `inject_vis` is set, a VIS burst is synthesized for the hypothesis's
/// mode and frequency offset and placed in front of the audio. Use it when
/// the recording has no usable header; leave it off when the header is
/// present, since the backend reads the real one.
///
/// # Errors
///
/// Fails only if the backend cannot be constructed or the decoder returns an
/// image whose buffer does not match its own dimensions. A signal that simply
/// does not decode returns `Ok(None)`.
pub fn decode(
    samples: &[f32],
    sample_rate: u32,
    hypothesis: &Hypothesis,
    in_signal_offset: f64,
    inject_vis: bool,
) -> Result<Option<Image>> {
    let signal = prepare_signal(
        samples,
        sample_rate,
        hypothesis,
        in_signal_offset,
        inject_vis,
    );
    let mut decoder = SstvDecoder::new(sample_rate)
        .context("initialize the SSTV raster decoder for this sample rate")?;

    let mut decoded: Option<SstvImage> = None;
    for chunk in signal.chunks(CHUNK) {
        for event in decoder.process(chunk) {
            if let SstvEvent::ImageComplete { image, .. } = event {
                decoded = Some(image);
                break;
            }
        }
        if decoded.is_some() {
            break;
        }
    }

    // A short silent tail lets a decode whose last line lands exactly on a
    // chunk boundary complete.
    if decoded.is_none() {
        let tail = vec![0.0_f32; (f64::from(sample_rate) * 0.5) as usize];
        for event in decoder.process(&tail) {
            if let SstvEvent::ImageComplete { image, .. } = event {
                decoded = Some(image);
                break;
            }
        }
    }

    // Still nothing? The recording is probably shorter than the mode's full
    // image. Pad past the expected length so the decoder reaches completion and
    // hands back the lines that *were* present, rather than discarding a
    // perfectly decodable partial transmission.
    //
    // The padding is measured from the decoder's own consumption count rather
    // than from the slice length, because the slice includes any synthesized
    // header that the backend has already folded into its buffer. Overshooting
    // by a further half second also covers the decoder's internal
    // resampling, which can consume a few samples more than it emits.
    if decoded.is_none() {
        let needed = expected_samples(hypothesis.mode, sample_rate, hypothesis.clock_rate);
        let consumed = signal.len();
        if needed > consumed {
            let overshoot = (f64::from(sample_rate) * 0.5) as usize;
            let cap = (MAX_TAIL_PADDING_SECONDS * f64::from(sample_rate)) as usize;
            let padding = (needed - consumed + overshoot).min(cap);
            let tail = vec![0.0_f32; padding];
            for event in decoder.process(&tail) {
                if let SstvEvent::ImageComplete { image, .. } = event {
                    decoded = Some(image);
                    break;
                }
            }
        }
    }

    decoded.map(convert).transpose()
}

/// Samples the backend expects for one complete image of `mode`.
fn expected_samples(mode: crate::modes::Mode, sample_rate: u32, clock_rate: f64) -> usize {
    let seconds = mode.image_seconds() * clock_rate;
    (seconds * f64::from(sample_rate)).ceil() as usize
}

/// Chunk size fed to the backend. Large enough to amortise per-call overhead,
/// small enough to keep peak memory modest.
const CHUNK: usize = 8192;

/// Silence appended to a short recording so the backend can finish.
///
/// The backend accumulates audio until it has enough for a whole image and
/// only then decodes; a recording that stops part-way therefore produces no
/// image at all, even though every transmitted line it does contain is
/// perfectly decodable. Padding the tail with silence lets the decoder reach
/// its expected length and emit the image, with the missing lines simply
/// decoding as whatever the silence maps to.
///
/// The padding is bounded so that a pathological request (a tiny fragment of a
/// very slow mode) cannot allocate without limit.
const MAX_TAIL_PADDING_SECONDS: f64 = 300.0;

/// Known limitation of the raster backend: its first and last few pixels per
/// line, and its first row, are unreliable.
///
/// Measured directly: feeding a *perfect* synthetic Robot 36 transmission of a
/// flat mid-grey image produces about 1.3% of pixels with a channel spread
/// above 60, and every one of them lies within roughly six columns of a line
/// edge or in row 0. The interior decodes exactly.
///
/// The cause is the backend's own per-pixel FFT window, which at the start and
/// end of a line necessarily overlaps the sync pulse and the neighbouring
/// channel. This crate's own grid extraction compensates for that (see
/// [`crate::raster::contaminated_tail_samples`]), but the final image comes
/// from the backend at full resolution, so a thin coloured fringe survives on
/// the outermost columns.
///
/// This is documented rather than papered over: the fringe carries no image
/// information, and the alternative — post-processing the backend's pixels —
/// would be guessing at what was transmitted.
pub const EDGE_ARTIFACT_COLUMNS: u32 = 6;

/// Build the audio the backend will see.
///
/// `in_signal_offset` is where in `samples` the first radio line's sync pulse
/// sits; everything before it is discarded, since the backend expects to
/// start decoding at a line boundary.
fn prepare_signal(
    samples: &[f32],
    sample_rate: u32,
    hypothesis: &Hypothesis,
    in_signal_offset: f64,
    inject_vis: bool,
) -> Vec<f32> {
    let start = (in_signal_offset.max(0.0) as usize).min(samples.len());
    let mut signal = Vec::with_capacity(
        samples.len() - start + if inject_vis { sample_rate as usize } else { 0 },
    );
    if inject_vis {
        signal.extend(crate::vis::synthesize(
            hypothesis.mode.vis_code,
            sample_rate,
            hypothesis.frequency_shift_hz,
        ));
    }
    signal.extend_from_slice(&samples[start..]);
    signal
}

/// Convert a backend image into this crate's representation.
fn convert(image: SstvImage) -> Result<Image> {
    let width = image.width;
    let height = image.height;
    let expected = (width as usize) * (height as usize);
    if image.pixels.len() != expected {
        return Err(anyhow!(
            "the raster decoder returned {} pixels for a {width}x{height} image",
            image.pixels.len()
        ));
    }
    let pixels = image.pixels;
    let quality = score::measure(&pixels, width, height);
    Ok(Image {
        width,
        height,
        pixels,
        quality,
    })
}

/// Encode an image as a PNG.
///
/// # Errors
///
/// Fails when the pixel buffer does not match the dimensions or the file
/// cannot be written.
pub fn save_png(image: &Image, path: &std::path::Path) -> Result<()> {
    let expected = (image.width as usize) * (image.height as usize);
    if image.pixels.len() != expected {
        return Err(anyhow!(
            "refusing to write {}: {} pixels for a {}x{} image",
            path.display(),
            image.pixels.len(),
            image.width,
            image.height
        ));
    }
    let mut buffer = image::RgbImage::new(image.width, image.height);
    for (destination, source) in buffer.pixels_mut().zip(image.pixels.iter()) {
        *destination = image::Rgb(*source);
    }
    buffer
        .save(path)
        .with_context(|| format!("write {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modes::describe;
    use slowrx::SstvMode;

    #[test]
    fn a_clean_synthetic_transmission_decodes_at_full_resolution() {
        let rate = 16_000;
        let mode = describe(SstvMode::Robot36).expect("robot36");
        let grid = crate::synth::test_grid(&mode);
        let signal = crate::synth::render_with_vis(&grid, &mode, rate, 0.0, None);

        let hypothesis = Hypothesis {
            mode,
            origin: crate::autodetect::Origin::Vis,
            skip_sample: 0.0,
            frequency_shift_hz: 0.0,
            clock_rate: 1.0,
            matched_syncs: 0,
            timing_score: 1.0,
            complete: true,
            notes: Vec::new(),
        };

        let image = decode(&signal, rate, &hypothesis, 0.0, false)
            .expect("decoder creation")
            .expect("an image");
        assert_eq!(image.dimensions(), (mode.line_pixels, mode.image_lines));
        assert!(
            !score::is_degenerate(&image.quality),
            "quality {:?}",
            image.quality
        );
    }

    #[test]
    fn a_truncated_recording_does_not_panic() {
        let rate = 16_000;
        let mode = describe(SstvMode::Robot36).expect("robot36");
        let grid = crate::synth::test_grid(&mode);
        let signal = crate::synth::render_with_vis(&grid, &mode, rate, 0.0, None);
        let short = &signal[..signal.len() / 3];
        let hypothesis = Hypothesis {
            mode,
            origin: crate::autodetect::Origin::Vis,
            skip_sample: 0.0,
            frequency_shift_hz: 0.0,
            clock_rate: 1.0,
            matched_syncs: 0,
            timing_score: 1.0,
            complete: false,
            notes: Vec::new(),
        };
        // Whatever the outcome, it must be an orderly result.
        let result = decode(short, rate, &hypothesis, 0.0, false);
        assert!(result.is_ok());
    }

    #[test]
    fn an_offset_beyond_the_signal_is_clamped() {
        let rate = 16_000;
        let mode = describe(SstvMode::Robot36).expect("robot36");
        let hypothesis = Hypothesis {
            mode,
            origin: crate::autodetect::Origin::Vis,
            skip_sample: 0.0,
            frequency_shift_hz: 0.0,
            clock_rate: 1.0,
            matched_syncs: 0,
            timing_score: 1.0,
            complete: true,
            notes: Vec::new(),
        };
        // A start position past the end must not panic.
        let result = decode(&[0.0_f32; 1000], rate, &hypothesis, 1.0e9, false);
        assert!(result.is_ok());
    }

    #[test]
    fn empty_audio_is_handled() {
        let rate = 16_000;
        let mode = describe(SstvMode::Robot36).expect("robot36");
        let hypothesis = Hypothesis {
            mode,
            origin: crate::autodetect::Origin::Vis,
            skip_sample: 0.0,
            frequency_shift_hz: 0.0,
            clock_rate: 1.0,
            matched_syncs: 0,
            timing_score: 1.0,
            complete: true,
            notes: Vec::new(),
        };
        let result = decode(&[], rate, &hypothesis, 0.0, false);
        assert!(result.is_ok());
    }

    #[test]
    fn converting_a_malformed_image_is_an_error_not_a_panic() {
        let image = SstvImage::new(SstvMode::Robot36, 320, 240);
        // `SstvImage::new` allocates the full buffer, so this must succeed.
        let converted = convert(image).expect("well-formed image");
        assert_eq!(converted.dimensions(), (320, 240));
    }

    #[test]
    fn save_png_rejects_a_mismatched_buffer() {
        let image = Image {
            width: 10,
            height: 10,
            pixels: vec![[0, 0, 0]; 4],
            quality: score::measure(&[], 0, 0),
        };
        let path = std::env::temp_dir().join("sstv_auto_bad_png.png");
        let error = save_png(&image, &path).unwrap_err();
        assert!(format!("{error}").contains("refusing"));
        assert!(!path.exists(), "no file should have been written");
    }

    #[test]
    fn save_png_round_trips_a_real_image() {
        let image = Image {
            width: 4,
            height: 4,
            pixels: (0..16)
                .map(|i| [i as u8 * 16, 0, 255 - i as u8 * 16])
                .collect(),
            quality: score::measure(&[], 0, 0),
        };
        let path = std::env::temp_dir().join("sstv_auto_roundtrip.png");
        save_png(&image, &path).expect("save");
        let loaded = image::open(&path).expect("reopen").to_rgb8();
        assert_eq!(loaded.dimensions(), (4, 4));
        assert_eq!(loaded.get_pixel(0, 0).0, image.pixels[0]);
        let _ = std::fs::remove_file(&path);
    }
}
